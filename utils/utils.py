import torch
import torch.nn.functional as F
import os
import yaml
import shutil
import numpy as np
import logging as log
from box import Box
from pathlib import Path
import datetime
from timm import optim
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import ast
from collections import OrderedDict
import esm
import esm_adapterH


def get_logging(result_path):
    logger = log.getLogger(result_path)
    logger.setLevel(log.INFO)

    fh = log.FileHandler(os.path.join(result_path, "logs.txt"))
    formatter = log.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = log.StreamHandler()
    logger.addHandler(sh)

    return logger


def prepare_saving_dir(configs, config_file_path):
    result_path = os.path.abspath(configs.result_path)

    checkpoint_path = os.path.join(result_path, 'checkpoints')
    figures_path = os.path.join(result_path, 'figures')
    Path(result_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(figures_path).mkdir(parents=True, exist_ok=True)

    shutil.copy(config_file_path, result_path)

    return result_path, checkpoint_path


def test_gpu_cuda():
    print('Testing gpu and cuda:')
    print('\tcuda is available:', torch.cuda.is_available())
    print('\tdevice count:', torch.cuda.device_count())
    print('\tcurrent device:', torch.cuda.current_device())
    print(f'\tdevice:', torch.cuda.device(0))
    print('\tdevice name:', torch.cuda.get_device_name(), end='\n\n')


def load_configs(config, args=None):
    tree_config = Box(config)
    tree_config.optimizer.lr_seq = float(tree_config.optimizer.lr_seq)
    tree_config.optimizer.lr_x = float(tree_config.optimizer.lr_x)
    tree_config.optimizer.decay.min_lr_x = float(tree_config.optimizer.decay.min_lr_x)
    tree_config.optimizer.decay.min_lr_seq = float(tree_config.optimizer.decay.min_lr_seq)
    tree_config.optimizer.weight_decay = float(tree_config.optimizer.weight_decay)
    tree_config.optimizer.eps = float(tree_config.optimizer.eps)
    tree_config.train_settings.temperature = float(tree_config.train_settings.temperature)
    tree_config.optimizer.beta_1 = float(tree_config.optimizer.beta_1)
    tree_config.optimizer.beta_2 = float(tree_config.optimizer.beta_2)
    tree_config.model.esm_encoder.lora.dropout = float(tree_config.model.esm_encoder.lora.dropout)
    if args is not None:
        if args.result_path:
            tree_config.result_path = args.result_path
        if args.resume_path:
            tree_config.resume.resume_path = args.resume_path
        if args.restart_optimizer is not None:
            tree_config.resume.restart_optimizer = args.restart_optimizer
    return tree_config


def load_checkpoints_md(simclr, configs,
                        optimizer_seq, optimizer_x, scheduler_seq, scheduler_x,
                        logging, resume_path, restart_optimizer=False):
    start_step = 0
    loss = None
    assert os.path.exists(resume_path), 'resume_path not exits'
    checkpoint = torch.load(resume_path)
    print(f"load checkpoints from {resume_path}")
    logging.info(f"load checkpoints from {resume_path}")

    def _load_filtered(module, sd, tag):
        """Shape-filter `sd`, load into `module` (strict=False), and REPORT the result.

        strict=False ignores missing/unexpected keys but NOT size mismatches for keys present
        in both. When resuming a DIFFERENT model (e.g. SPLM) to init only the backbone/adapters,
        its projector heads (projectors_protein/residue) have a different out_dim than this
        config's — we drop those mismatched tensors so our freshly-initialised projectors are
        kept instead of crashing. (ddg_S669's load_esm2_checkpoint avoids the crash only because
        it targets a raw ESM2 with no projectors, so those keys are merely 'unexpected'.)
        The missing/unexpected report matters here to confirm the ADAPTER tensors actually
        loaded rather than silently staying at random init.
        """
        msd = module.state_dict()
        kept, dropped = OrderedDict(), []
        for k, v in sd.items():
            if k in msd and hasattr(v, 'shape') and v.shape != msd[k].shape:
                dropped.append((k, tuple(v.shape), tuple(msd[k].shape)))
            else:
                kept[k] = v
        res = module.load_state_dict(kept, strict=False)
        missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
        matched = len(set(kept) & set(msd)) - len(dropped)

        def _emit(msg):                       # to BOTH stdout and the log file
            print(msg); logging.info(msg)     # (LCC .out captures logging, swallows print)

        _emit(f"[resume/{tag}] matched&loaded={matched}  shape-dropped={len(dropped)}  "
              f"missing={len(missing)}  unexpected={len(unexpected)}")
        for k, cs, ms in dropped:
            _emit(f"    [shape-dropped] {k}: ckpt{cs} vs model{ms}")
        mis_ad = [k for k in missing if 'adapter' in k]
        unx_ad = [k for k in unexpected if 'adapter' in k]
        _emit(f"  adapter tensors — missing(random-init): {len(mis_ad)}   "
              f"unexpected(ckpt-unused): {len(unx_ad)}")
        for k in mis_ad:
            _emit(f"    [missing-adapter] {k}")
        for k in unx_ad:
            _emit(f"    [unexpected-adapter] {k}")
        # a sample of the non-adapter missing/unexpected (trim to avoid flooding)
        mis_other = [x for x in missing if 'adapter' not in x]
        unx_other = [x for x in unexpected if 'adapter' not in x]
        _emit(f"  non-adapter missing: {len(mis_other)} (showing up to 40)")
        for k in mis_other[:40]:
            _emit(f"    [missing] {k}")
        if unx_other:
            _emit(f"  non-adapter unexpected: {len(unx_other)} (showing up to 40)")
            for k in unx_other[:40]:
                _emit(f"    [unexpected] {k}")
        return res

    if "state_dict1" in checkpoint:
        if any('adapter' in name for name, _ in simclr.model_seq.named_modules()):
            if np.sum(["adapter_layer_dict" in key for key in checkpoint['state_dict1'].keys()]) == 0:
                new_ordered_dict = OrderedDict()
                for key, value in checkpoint['state_dict1'].items():
                    if "adapter_layer_dict" not in key:
                        new_key = key.replace('adapter_layer', 'adapter_layer_dict.adapter_0')
                        new_ordered_dict[new_key] = value
                    else:
                        new_ordered_dict[key] = value
                _load_filtered(simclr.model_seq, new_ordered_dict, 'model_seq')
            else:
                _load_filtered(simclr.model_seq, checkpoint['state_dict1'], 'model_seq')
        else:
            _load_filtered(simclr.model_seq, checkpoint['state_dict1'], 'model_seq')

    if "state_x" in checkpoint:
        _load_filtered(simclr.model_x, checkpoint['state_x'], 'model_x')

    if 'logit_scale' in checkpoint and hasattr(simclr, 'logit_scale'):
        simclr.logit_scale.data = checkpoint['logit_scale']
        logging.info('logit_scale (learnable temperature) restored from checkpoint')

    if 'step' in checkpoint:
        if not restart_optimizer:
            if 'optimizer_x' in checkpoint and "scheduler_x" in checkpoint:
                optimizer_x.load_state_dict(checkpoint['optimizer_x'])
                logging.info('optimizer_x is loaded to resume training!')
                scheduler_x.load_state_dict(checkpoint['scheduler_x'])
                logging.info('scheduler_x is loaded to resume training!')
            if 'optimizer_seq' in checkpoint and 'scheduler_seq' in checkpoint:
                optimizer_seq.load_state_dict(checkpoint['optimizer_seq'])
                logging.info('optimizer_seq is loaded to resume training!')
                scheduler_seq.load_state_dict(checkpoint['scheduler_seq'])
                logging.info('scheduler_seq is loaded to resume training!')
            start_step = checkpoint['step'] + 1
            if 'loss' in checkpoint:
                loss = checkpoint['loss']

    return simclr, start_step, loss


def save_checkpoints(optimizer_x, optimizer_seq, result_path, simclr, n_steps, logging, epoch, loss):
    # checkpoint_name = f'checkpoint_{n_steps:07d}.pth'
    checkpoint_name = f'checkpoint_every_n.pth'
    save_path = os.path.join(result_path, 'checkpoints', checkpoint_name)

    ckpt = {
        'epoch': epoch,
        'step': n_steps,
        'state_dict1': simclr.model_seq.state_dict(),
        'state_x': simclr.model_x.state_dict(),
        'optimizer_x': optimizer_x.state_dict(),
        'optimizer_seq': optimizer_seq.state_dict(),
        'loss': loss,
    }
    if hasattr(simclr, 'logit_scale'):
        ckpt['logit_scale'] = simclr.logit_scale.data
    save_checkpoint(ckpt, is_best=False, filename=save_path)
    logging.info(f"Model checkpoint and metadata have been saved at {save_path}")


def save_best_checkpoints(optimizer_x, optimizer_seq, result_path, simclr, n_steps,
                          logging, epoch, best_loss, current_loss, best_cptfile, direction='low'):
    checkpoint_name = f'checkpoint_{best_cptfile}.pth'
    save_path = os.path.join(result_path, 'checkpoints', checkpoint_name)
    def _build_ckpt(loss_val):
        ckpt = {
            'epoch': epoch,
            'step': n_steps,
            'state_dict1': simclr.model_seq.state_dict(),
            'state_x': simclr.model_x.state_dict(),
            'optimizer_x': optimizer_x.state_dict(),
            'optimizer_seq': optimizer_seq.state_dict(),
            'loss': loss_val,
        }
        if hasattr(simclr, 'logit_scale'):
            ckpt['logit_scale'] = simclr.logit_scale.data
        return ckpt

    if direction == 'high':
        if current_loss > best_loss:
            save_checkpoint(_build_ckpt(current_loss), is_best=False, filename=save_path)
            logging.info(f"Model checkpoint and metadata have been saved at {save_path}")
            return current_loss
        else:
            return best_loss
    else:
        if current_loss < best_loss:
            save_checkpoint(_build_ckpt(current_loss), is_best=False, filename=save_path)
            logging.info(f"Model checkpoint and metadata have been saved at {save_path}")
            return current_loss
        else:
            return best_loss


def load_optimizers(model_seq, model_x, logging, configs):
    optimizer_seq = None
    optimizer_x = None
    if configs.optimizer.name.lower() == 'adabelief':
        if model_seq is not None:
            optimizer_seq = optim.AdaBelief(model_seq.parameters(), lr=configs.optimizer.lr_seq, eps=configs.optimizer.eps,
                                            decoupled_decay=True,
                                            weight_decay=configs.optimizer.weight_decay, rectify=False)
        if model_x is not None and configs.model.X_module == "MD":
            optimizer_x = optim.AdaBelief(model_x.parameters(), lr=configs.optimizer.lr_MD,
                                           eps=configs.optimizer.eps,
                                           decoupled_decay=True,
                                           weight_decay=configs.optimizer.weight_decay, rectify=False)
    elif configs.optimizer.name.lower() == 'adam':
        if model_seq is not None:
            optimizer_seq = torch.optim.AdamW(
                model_seq.parameters(), lr=float(configs.optimizer.lr_seq),
                betas=(configs.optimizer.beta_1, configs.optimizer.beta_2),
                weight_decay=float(configs.optimizer.weight_decay),
                eps=float(configs.optimizer.eps)
            )
        if model_x is not None:
            optimizer_x = torch.optim.AdamW(
                model_x.parameters(), lr=float(configs.optimizer.lr_x),
                betas=(configs.optimizer.beta_1, configs.optimizer.beta_2),
                weight_decay=float(configs.optimizer.weight_decay),
                eps=float(configs.optimizer.eps)
            )
    elif configs.optimizer.name.lower() == 'sgd':
        logging.info('use sgd optimizer')
        if model_x is not None:
            optimizer_x = torch.optim.SGD(model_x.parameters(), lr=float(configs.optimizer.lr_x),
                                           momentum=0.9, dampening=0,
                                           weight_decay=float(configs.optimizer.weight_decay))
        if model_seq is not None:
            optimizer_seq = torch.optim.SGD(model_seq.parameters(), lr=float(configs.optimizer.lr_seq), momentum=0.9,
                                            dampening=0, weight_decay=float(configs.optimizer.weight_decay))
    else:
        raise ValueError('wrong optimizer')
    return optimizer_seq, optimizer_x


def prepare_optimizer(model_seq, model_x, logging, configs):
    logging.info("prepare the optimizers")
    optimizer_seq, optimizer_x = load_optimizers(model_seq, model_x, logging, configs)
    scheduler_seq, scheduler_x = None, None

    logging.info("prepare the schedulers")
    first_cycle_steps = (configs.optimizer.decay.first_cycle_steps
                         or configs.train_settings.num_steps)
    if model_x is not None:
        scheduler_x = CosineAnnealingWarmupRestarts(
            optimizer_x,
            first_cycle_steps=first_cycle_steps,
            cycle_mult=1.0,
            max_lr=configs.optimizer.lr_x,
            min_lr=configs.optimizer.decay.min_lr_x,
            warmup_steps=configs.optimizer.decay.warmup,
            gamma=configs.optimizer.decay.gamma)
    if model_seq is not None:
        scheduler_seq = CosineAnnealingWarmupRestarts(
            optimizer_seq,
            first_cycle_steps=first_cycle_steps,
            cycle_mult=1.0,
            max_lr=configs.optimizer.lr_seq,
            min_lr=configs.optimizer.decay.min_lr_seq,
            warmup_steps=configs.optimizer.decay.warmup,
            gamma=configs.optimizer.decay.gamma)

    return scheduler_seq, scheduler_x, optimizer_seq, optimizer_x


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_esm2_checkpoint(model, checkpoint_path):
    """Load simclr.model_seq weights from a training checkpoint into a raw esm2 model.

    The training checkpoint stores the full ESM2-wrapper state dict (keys prefixed with
    "esm2.") under 'state_dict1'.  This function strips that prefix and handles the
    adapter_layer → adapter_layer_dict.adapter_0 rename needed for older checkpoints.
    """
    assert checkpoint_path is not None and os.path.exists(checkpoint_path), \
        f"Checkpoint not found: {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    assert 'state_dict1' in checkpoint, "Checkpoint does not contain 'state_dict1' key."

    if any('adapter' in name for name, _ in model.named_modules()):
        if np.sum(["adapter_layer_dict" in key for key in checkpoint['state_dict1'].keys()]) == 0:
            # Old checkpoint: rename adapter_layer → adapter_layer_dict.adapter_0
            new_dict = OrderedDict()
            for k, v in checkpoint['state_dict1'].items():
                nk = k.replace('adapter_layer', 'adapter_layer_dict.adapter_0') \
                     if 'adapter_layer_dict' not in k else k
                new_dict[nk] = v
            load_result = model.load_state_dict(new_dict, strict=False)
            load_path = "old-format (adapter_layer → adapter_layer_dict.adapter_0)"
        else:
            # New checkpoint: strip leading "esm2." so keys align with the raw model
            new_dict = OrderedDict(
                (k.replace("esm2.", ""), v)
                for k, v in checkpoint['state_dict1'].items()
            )
            load_result = model.load_state_dict(new_dict, strict=False)
            load_path = "new-format (strip 'esm2.' prefix)"
    else:
        # Plain ESM2 or LoRA — strip "esm2." prefix
        new_dict = OrderedDict(
            (k.replace("esm2.", ""), v)
            for k, v in checkpoint['state_dict1'].items()
        )
        load_result = model.load_state_dict(new_dict, strict=False)
        load_path = "plain ESM2/LoRA (strip 'esm2.' prefix)"

    # ── Report key mismatches (strict=False silently ignores these) ──────────
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    model_keys = set(dict(model.named_parameters()).keys()) | set(dict(model.named_buffers()).keys())
    ckpt_keys = set(new_dict.keys())
    matched = len(model_keys & ckpt_keys)
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"  load path        : {load_path}")
    print(f"  model tensors    : {len(model_keys)}")
    print(f"  checkpoint tensors: {len(ckpt_keys)}")
    print(f"  matched & loaded : {matched}")
    print(f"  MISSING keys (in model, NOT loaded from ckpt): {len(missing)}")
    for k in missing:
        print(f"      [missing]    {k}")
    print(f"  UNEXPECTED keys (in ckpt, not used by model) : {len(unexpected)}")
    for k in unexpected:
        print(f"      [unexpected] {k}")
    # Specifically flag adapter keys left at random init
    missing_adapter = [k for k in missing if 'adapter' in k]
    if missing_adapter:
        print(f"  ⚠ {len(missing_adapter)} ADAPTER tensor(s) MISSING — these stayed at "
              f"random init instead of loading DPLM weights:")
        for k in missing_adapter:
            print(f"      [missing-adapter] {k}")
