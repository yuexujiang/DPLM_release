"""
train_ddg_v2.py — Train the mutation-site-aware ddG model (EncoderSiteAware) on S8754;
                  evaluate on S669. MAE-prioritized, with inverse-mutation augmentation.

What differs from train_ddg.py
------------------------------
  * Uses model_ddg_v2.EncoderSiteAware + data_ddg_v2.prepare_dataloaders_v2 (batches are
    5-tuples: from_seqs, to_seqs, mut_pos, ddg, protein_ids; model called as
    net(from_seqs, to_seqs, mut_pos)).
  * Configurable regression loss (default MAE-friendly Huber): huber | mae | mse.
  * Optional differential LR: head params at lr * head_lr_multiplier (default 1.0 = off).
  * Selects the best checkpoint by validation MAE (best_model_mae.pth), and also keeps
    best-Spearman and best-MSE checkpoints. Reports MAE / RMSE / Spearman / Pearson.

Usage
-----
PYTHONPATH=/path/to/DPLM_ai accelerate launch evaluate/ddg_S669/train_ddg_v2.py \\
    --config_path evaluate/ddg_S669/ddg_config_siteaware.yaml \\
    --result_path ./results/ddg_siteaware \\
    --resume_path ./results/vivit5/checkpoints/checkpoint_best_val_whole_loss.pth
"""

import os
import sys
import argparse

import numpy as np
import yaml
import torch
import torch.nn.functional as F
from time import time
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
from accelerate import Accelerator
from box import Box
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

# ── project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model_ddg_v2 import prepare_models_v2                          # local
from data_ddg_v2 import prepare_dataloaders_v2                      # local
from utils.utils import (
    get_logging, prepare_saving_dir, test_gpu_cuda,
    save_checkpoint, load_esm2_checkpoint,
)


# ────────────────────────────────────────────────────────────────────────────
# 1.  Config loading
# ────────────────────────────────────────────────────────────────────────────

def _load_configs(yaml_dict, args):
    """Box-wrap the ddG YAML and apply CLI overrides."""
    cfg = Box(yaml_dict)
    cfg.optimizer.lr           = float(cfg.optimizer.lr)
    cfg.optimizer.weight_decay = float(cfg.optimizer.weight_decay)
    cfg.optimizer.eps          = float(cfg.optimizer.eps)
    cfg.optimizer.beta_1       = float(cfg.optimizer.beta_1)
    cfg.optimizer.beta_2       = float(cfg.optimizer.beta_2)
    if args.result_path:
        cfg.result_path = args.result_path
    if args.resume_path:
        cfg.resume.resume_path = args.resume_path
    if args.train_csv_path:
        cfg.train_settings.train_csv_path = args.train_csv_path
    if args.test_csv_path:
        cfg.test_settings.test_csv_path = args.test_csv_path
    if args.num_end_adapter_layers:
        cfg.encoder.adapter_h.num_end_adapter_layers = [
            int(x) for x in args.num_end_adapter_layers.split(',')]
    return cfg


# ────────────────────────────────────────────────────────────────────────────
# 2.  Loss
# ────────────────────────────────────────────────────────────────────────────

def _regression_loss(outputs, targets, configs, reduction='mean'):
    """Regression loss selected by configs.train_settings.loss.

    huber (default) → smooth_l1_loss(beta=huber_beta): robust, MAE-friendly.
    mae             → l1_loss.
    mse             → mse_loss.
    """
    loss_type = str(getattr(configs.train_settings, 'loss', 'huber')).lower()
    if loss_type == 'mse':
        return F.mse_loss(outputs, targets, reduction=reduction)
    if loss_type in ('mae', 'l1'):
        return F.l1_loss(outputs, targets, reduction=reduction)
    # default: huber / smooth_l1
    beta = float(getattr(configs.train_settings, 'huber_beta', 0.5))
    return F.smooth_l1_loss(outputs, targets, reduction=reduction, beta=beta)


# ────────────────────────────────────────────────────────────────────────────
# 3.  DPLM checkpoint initialisation
# ────────────────────────────────────────────────────────────────────────────

def _init_from_dplm(net, resume_path, logging, accelerator):
    """Load adapter_0 weights from a DPLM checkpoint into net.esm2 (adapter_0 stays
    frozen; only adapter_1 + head train). See load_esm2_checkpoint for key remapping."""
    raw_net = accelerator.unwrap_model(net)
    load_esm2_checkpoint(raw_net.esm2, resume_path)
    logging.info(f'DPLM checkpoint loaded → adapter_0 initialised: {resume_path}')


# ────────────────────────────────────────────────────────────────────────────
# 4.  Optimizer + scheduler
# ────────────────────────────────────────────────────────────────────────────

def _prepare_optimizer(net, configs, total_steps, logging):
    """AdamW + cosine-warmup scheduler.

    Optional differential LR: head params (name contains 'head') get base lr, and are
    rescaled to lr * head_lr_multiplier after each scheduler.step() in the loop (the
    scheduler library sets every group to the same value, so post-scaling is how the
    multiplier is applied). Multiplier 1.0 (default) leaves behavior identical to a
    single group.

    Returns (optimizer, scheduler, head_group_idx, head_lr_multiplier).
    """
    lr           = float(configs.optimizer.lr)
    weight_decay = float(configs.optimizer.weight_decay)
    eps          = float(configs.optimizer.eps)
    beta_1       = float(configs.optimizer.beta_1)
    beta_2       = float(configs.optimizer.beta_2)
    min_lr       = float(configs.optimizer.decay.min_lr)
    warmup       = int(configs.optimizer.decay.warmup)
    gamma        = float(configs.optimizer.decay.gamma)
    first_cycle  = int(configs.optimizer.decay.first_cycle_steps or total_steps)
    head_mult    = float(getattr(configs.optimizer, 'head_lr_multiplier', 1.0))

    head_params, backbone_params = [], []
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if 'head' in n else backbone_params).append(p)

    logging.info(f'Trainable — backbone tensors: {len(backbone_params)}, '
                 f'head tensors: {len(head_params)}; '
                 f'head_lr_multiplier={head_mult}')

    param_groups = [{'params': backbone_params, 'lr': lr}]
    head_group_idx = None
    if head_params:
        head_group_idx = len(param_groups)
        param_groups.append({'params': head_params, 'lr': lr})

    optimizer = torch.optim.AdamW(
        param_groups, lr=lr, betas=(beta_1, beta_2),
        weight_decay=weight_decay, eps=eps,
    )

    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=first_cycle,
        cycle_mult=1.0,
        max_lr=lr,
        min_lr=min_lr,
        warmup_steps=warmup,
        gamma=gamma,
    )
    return optimizer, scheduler, head_group_idx, head_mult


def _apply_head_lr(optimizer, head_group_idx, head_mult):
    """Rescale the head param-group LR after scheduler.step (no-op if mult == 1.0)."""
    if head_group_idx is not None and head_mult != 1.0:
        optimizer.param_groups[head_group_idx]['lr'] *= head_mult


# ────────────────────────────────────────────────────────────────────────────
# 5.  Checkpoint save
# ────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(epoch, path, net, optimizer, scheduler, accelerator):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ckpt = {
            'epoch': epoch,
            'model_state_dict':     accelerator.unwrap_model(net).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        save_checkpoint(ckpt, is_best=False, filename=path)


# ────────────────────────────────────────────────────────────────────────────
# 6.  Train / valid loops
# ────────────────────────────────────────────────────────────────────────────

def train(epoch, accelerator, dataloader, net, optimizer, scheduler,
          accum_iter, grad_clip, configs, head_group_idx, head_mult):
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0.0
    step_loss  = 0.0
    global_step = 0
    pred_list, label_list = [], []

    progress_bar = tqdm(
        range(int(np.ceil(len(dataloader) / accum_iter))),
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc='Train',
    )

    for from_seqs, to_seqs, mut_pos, ddg, _ in dataloader:
        with accelerator.accumulate(net):
            outputs = net(from_seqs, to_seqs, mut_pos)
            loss = _regression_loss(outputs, ddg, configs)

            avg_loss = accelerator.gather(loss.repeat(configs.train_settings.batch_size)).mean()
            step_loss += avg_loss.item() / accum_iter

            pred_list.append(outputs.detach())
            label_list.append(ddg.detach())

            accelerator.backward(loss)

            if accelerator.sync_gradients and configs.optimizer.grad_clip_norm > 0:
                accelerator.clip_grad_norm_(net.parameters(), configs.optimizer.grad_clip_norm)

            optimizer.step()
            scheduler.step()
            _apply_head_lr(optimizer, head_group_idx, head_mult)
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            epoch_loss += step_loss
            step_loss = 0.0
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f'{loss.item():.4f}',
                                     lr=f'{float(optimizer.param_groups[0]["lr"]):.2e}')

    preds  = torch.cat(pred_list,  dim=0).cpu().numpy()
    labels = torch.cat(label_list, dim=0).cpu().numpy()
    spear, _ = spearmanr(preds, labels)
    return epoch_loss / max(global_step, 1), spear


def valid(epoch, accelerator, dataloader, net, configs):
    net.eval()

    valid_loss = 0.0
    pred_list, label_list = [], []

    progress_bar = tqdm(
        range(len(dataloader)),
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc='Valid',
    )

    for from_seqs, to_seqs, mut_pos, ddg, _ in dataloader:
        with torch.inference_mode():
            outputs = net(from_seqs, to_seqs, mut_pos)
            losses  = _regression_loss(
                accelerator.gather(outputs),
                accelerator.gather(ddg),
                configs,
                reduction='none',
            )
            loss = losses.mean()

        pred_list.append(outputs.detach())
        label_list.append(ddg.detach())
        valid_loss += loss.item()
        progress_bar.update(1)

    valid_loss /= len(dataloader)
    preds  = torch.cat(pred_list,  dim=0).cpu().numpy()
    labels = torch.cat(label_list, dim=0).cpu().numpy()

    spear, _ = spearmanr(preds, labels)
    pear, _  = pearsonr(preds, labels)
    rmse      = float(np.sqrt(mean_squared_error(labels, preds)))
    mae       = float(np.mean(np.abs(labels - preds)))
    return valid_loss, spear, pear, rmse, mae


def _mut_label(from_seq, to_seq, mut_pos):
    """Build a mutation label + first-site (pos, wt_aa, mt_aa) from a directed mutation.

    `mut_pos` is a list of 0-based indices where the two sequences differ. The label
    is 1-based (e.g. 'A42G'); multi-site mutations are joined with ';'. Returns
    ('', None, '', '') when no differing positions are recorded.
    """
    if not mut_pos:
        return '', None, '', ''
    parts = [f'{from_seq[p]}{p + 1}{to_seq[p]}' for p in mut_pos]
    p0 = mut_pos[0]
    return ';'.join(parts), p0 + 1, from_seq[p0], to_seq[p0]


@torch.inference_mode()
def evaluate_direct_reverse(accelerator, dataloader, net, save_predictions=False):
    """Evaluate direct and reverse ΔΔG predictions + the antisymmetry metrics from
    Pancotti et al. 2022 (Briefings in Bioinformatics, bbab555).

    For each variant the model predicts:
        pred_dir = net(WT → MT)   compared to observed ΔΔG        (direct)
        pred_rev = net(MT → WT)   compared to observed −ΔΔG       (reverse)
    The reverse input is just the two sequence arguments swapped; the mutated
    positions are identical (the differing indices are symmetric).

    Antisymmetry (a perfectly antisymmetric predictor has pred_rev = −pred_dir):
        r_{d-r}  = Pearson(pred_dir, pred_rev)                    ideal = −1   (Eq. 1)
        <delta>  = mean( (pred_dir + pred_rev) / 2 )              ideal =  0   (Eq. 2)
    """
    net.eval()
    pdir_list, prev_list, y_list = [], [], []
    pred_rows = []

    progress_bar = tqdm(range(len(dataloader)),
                        disable=not accelerator.is_local_main_process,
                        leave=False, desc='Test(dir/rev)')
    for from_seqs, to_seqs, mut_pos, ddg, protein_ids in dataloader:
        out_dir = net(from_seqs, to_seqs, mut_pos)   # WT → MT
        out_rev = net(to_seqs, from_seqs, mut_pos)   # MT → WT (swap inputs)
        pdir_list.append(accelerator.gather(out_dir).detach())
        prev_list.append(accelerator.gather(out_rev).detach())
        y_list.append(accelerator.gather(ddg).detach())
        # Per-mutation predictions are collected from the local (ungathered) batch
        # on the main process only — correct for the single-process test runs used
        # for S669 evaluation.
        if save_predictions and accelerator.is_main_process:
            pd_dir = out_dir.detach().float().cpu().numpy().reshape(-1)
            pd_rev = out_rev.detach().float().cpu().numpy().reshape(-1)
            yd = ddg.detach().float().cpu().numpy().reshape(-1)
            for j in range(len(pd_dir)):
                mtype, pos, wt_aa, mt_aa = _mut_label(
                    from_seqs[j], to_seqs[j], mut_pos[j])
                pred_rows.append(dict(
                    protein_id=protein_ids[j], mut_type=mtype, position=pos,
                    wt_aa=wt_aa, mt_aa=mt_aa,
                    ddG=float(yd[j]), prediction=float(pd_dir[j]),
                    ddG_reverse=float(-yd[j]), prediction_reverse=float(pd_rev[j])))
        progress_bar.update(1)

    pred_dir = torch.cat(pdir_list, dim=0).cpu().numpy()
    pred_rev = torch.cat(prev_list, dim=0).cpu().numpy()
    y_dir    = torch.cat(y_list,    dim=0).cpu().numpy()
    y_rev    = -y_dir                                  # observed reverse ΔΔG

    def _metrics(pred, obs):
        pear, _  = pearsonr(pred, obs)
        spear, _ = spearmanr(pred, obs)
        rmse = float(np.sqrt(mean_squared_error(obs, pred)))
        mae  = float(np.mean(np.abs(obs - pred)))
        return pear, spear, rmse, mae

    dir_m = _metrics(pred_dir, y_dir)
    rev_m = _metrics(pred_rev, y_rev)
    r_dr, _ = pearsonr(pred_dir, pred_rev)             # Eq. 1  (ideal −1)
    bias    = float(np.mean((pred_dir + pred_rev) / 2.0))  # Eq. 2  (ideal 0)
    return {'direct': dir_m, 'reverse': rev_m, 'r_dr': float(r_dr), 'bias': bias,
            'pred_rows': pred_rows}


def _format_dir_rev_table(tag, res):
    """Render the direct/reverse/antisymmetry results as a Table-1-style block."""
    d, r = res['direct'], res['reverse']
    lines = [
        f'\n[{tag}] S669 test — direct / reverse / antisymmetry (Pancotti et al. 2022)',
        f'  {"":<9}{"Pearson":>9}{"Spearman":>10}{"RMSE":>9}{"MAE":>9}',
        f'  {"direct":<9}{d[0]:>9.4f}{d[1]:>10.4f}{d[2]:>9.4f}{d[3]:>9.4f}',
        f'  {"reverse":<9}{r[0]:>9.4f}{r[1]:>10.4f}{r[2]:>9.4f}{r[3]:>9.4f}',
        f'  antisymmetry:  r_d-r = {res["r_dr"]:.4f} (ideal -1.0)   '
        f'<delta> (bias) = {res["bias"]:.4f} (ideal 0.0)',
    ]
    return '\n'.join(lines)


def _save_predictions_csv(pred_rows, out_path):
    """Write per-mutation direct/reverse predictions to a CSV (mirrors ddg_mega)."""
    import pandas as pd
    cols = ['protein_id', 'mut_type', 'position', 'wt_aa', 'mt_aa',
            'ddG', 'prediction', 'ddG_reverse', 'prediction_reverse']
    pd.DataFrame(pred_rows, columns=cols).to_csv(out_path, index=False)
    print(f'[Predictions] {len(pred_rows)} test rows → {out_path}')


# ────────────────────────────────────────────────────────────────────────────
# 7.  CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train site-aware ESM2 for ddG prediction (S669).')
    p.add_argument('--config_path', '-c', required=True, help='Path to the YAML config file.')
    p.add_argument('--result_path', default=None, help='Override result_path from config.')
    p.add_argument('--resume_path', default=None,
                   help='DPLM checkpoint to initialise adapter_0 weights.')
    p.add_argument('--train_csv_path', default=None,
                   help='Override train_settings.train_csv_path (S8754 CSV).')
    p.add_argument('--test_csv_path', default=None,
                   help='Override test_settings.test_csv_path (S669 CSV).')
    p.add_argument('--num_end_adapter_layers', default=None,
                   help='Override encoder.adapter_h.num_end_adapter_layers, comma-separated '
                        '(e.g. "20,4"). First value must match the DPLM checkpoint depth.')
    p.add_argument('--save_predictions', action='store_true',
                   help='Also write a per-mutation predictions CSV for the S669 test set.')
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# 8.  main
# ────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.config_path) as f:
        yaml_dict = yaml.full_load(f)
    configs = _load_configs(yaml_dict, args)

    if isinstance(getattr(configs, 'fix_seed', None), int):
        torch.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()
    test_gpu_cuda()

    result_path, checkpoint_path = prepare_saving_dir(configs, args.config_path)
    logging = get_logging(result_path)

    accelerator = Accelerator(
        mixed_precision=configs.train_settings.mixed_precision,
        gradient_accumulation_steps=configs.train_settings.grad_accumulation,
        dispatch_batches=False,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers('ddg_tracker_v2', config=None)

    # ── Dataloaders ───────────────────────────────────────────────────────
    dataloaders = prepare_dataloaders_v2(configs)
    logging.info(f'dataloaders ready — train items: {len(dataloaders["train"].dataset)} '
                 f'(augment_inverse={getattr(configs.train_settings, "augment_inverse", False)}), '
                 f'val items: {len(dataloaders["valid"].dataset)}, '
                 f'test items: {len(dataloaders["test"].dataset)}')

    # ── Model ─────────────────────────────────────────────────────────────
    net = prepare_models_v2(configs, logging)
    logging.info('model ready')

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    total_steps = configs.train_settings.num_epochs * len(dataloaders['train'])
    optimizer, scheduler, head_group_idx, head_mult = _prepare_optimizer(
        net, configs, total_steps, logging)
    logging.info('optimizer ready')

    # ── Accelerate prepare (scheduler stepped manually — see train_ddg.py) ─
    (dataloaders['train'], dataloaders['valid'], dataloaders['test'],
     net, optimizer) = accelerator.prepare(
        dataloaders['train'], dataloaders['valid'], dataloaders['test'],
        net, optimizer,
    )

    # ── DPLM checkpoint init ──────────────────────────────────────────────
    if configs.resume.resume_path and os.path.exists(configs.resume.resume_path):
        _init_from_dplm(net, configs.resume.resume_path, logging, accelerator)
    else:
        logging.info('No DPLM checkpoint — training adapter_1 + head from scratch.')

    logging.info(f'train batches/epoch: {len(dataloaders["train"])}')
    logging.info(f'valid batches/epoch: {len(dataloaders["valid"])}')
    logging.info(f'test  batches:       {len(dataloaders["test"])}')

    best_valid_mae      = np.inf
    best_valid_mse      = np.inf
    best_valid_spearman = -np.inf

    for epoch in range(configs.train_settings.num_epochs):
        t0 = time()
        train_loss, train_cor = train(
            epoch, accelerator, dataloaders['train'],
            net, optimizer, scheduler,
            configs.train_settings.grad_accumulation,
            configs.optimizer.grad_clip_norm,
            configs, head_group_idx, head_mult,
        )
        if accelerator.is_main_process:
            logging.info(
                f'epoch {epoch} — {time()-t0:.1f}s  '
                f'train_loss={train_loss:.4f}  train_r={train_cor:.4f}'
            )

        if epoch % configs.valid_settings.do_every == 0:
            t0 = time()
            valid_loss, valid_cor, valid_pear, valid_rmse, valid_mae = valid(
                epoch, accelerator, dataloaders['valid'], net, configs,
            )
            if accelerator.is_main_process:
                logging.info(
                    f'  valid — {time()-t0:.1f}s  loss={valid_loss:.4f}  '
                    f'spearman={valid_cor:.4f}  pearson={valid_pear:.4f}  '
                    f'rmse={valid_rmse:.4f}  mae={valid_mae:.4f}'
                )

            # Primary selection: lowest validation MAE.
            if valid_mae < best_valid_mae:
                best_valid_mae = valid_mae
                _save_checkpoint(epoch,
                                 os.path.join(checkpoint_path, 'best_model_mae.pth'),
                                 net, optimizer, scheduler, accelerator)
                if accelerator.is_main_process:
                    logging.info(f'  → best MAE checkpoint saved (mae={valid_mae:.4f})')

            if valid_loss < best_valid_mse:
                best_valid_mse = valid_loss
                _save_checkpoint(epoch,
                                 os.path.join(checkpoint_path, 'best_model_mse.pth'),
                                 net, optimizer, scheduler, accelerator)

            if valid_cor > best_valid_spearman:
                best_valid_spearman = valid_cor
                _save_checkpoint(epoch,
                                 os.path.join(checkpoint_path, 'best_model_cor.pth'),
                                 net, optimizer, scheduler, accelerator)
                if accelerator.is_main_process:
                    logging.info(f'  → best Spearman checkpoint saved (r={valid_cor:.4f})')

    if accelerator.is_main_process:
        logging.info(f'Training done.  best_val_mae={best_valid_mae:.4f}  '
                     f'best_val_mse={best_valid_mse:.4f}  '
                     f'best_val_spearman={best_valid_spearman:.4f}')

    # ── Test evaluation on the best checkpoints ───────────────────────────
    # Report direct AND reverse prediction performance (Pearson/Spearman/RMSE/MAE)
    # plus the antisymmetry metrics r_{d-r} and bias <delta> (Pancotti et al. 2022).
    for tag, fname in [('best_model_mae', 'best_model_mae.pth'),
                       ('best_model_cor', 'best_model_cor.pth')]:
        ckpt_path = os.path.join(checkpoint_path, fname)
        if not os.path.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location='cpu')
        accelerator.unwrap_model(net).load_state_dict(ckpt['model_state_dict'])
        t0 = time()
        res = evaluate_direct_reverse(accelerator, dataloaders['test'], net,
                                      save_predictions=args.save_predictions)
        if accelerator.is_main_process:
            logging.info(_format_dir_rev_table(tag, res) +
                         f'\n  (eval {time()-t0:.1f}s)')
            if args.save_predictions:
                _save_predictions_csv(
                    res['pred_rows'],
                    os.path.join(result_path, f'predictions_{tag}.csv'))

    accelerator.end_training()
    accelerator.free_memory()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
