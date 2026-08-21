import argparse
import os
import yaml
import time
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch
import wandb
from model import prepare_models, AverageMeter, clip_infonce, clip_infonce_multipos, info_nce_loss
from model import log_negative_mean_logtis
from model import MaskedLMDataCollator

from utils.utils import prepare_optimizer, load_checkpoints_md, save_checkpoints, \
    load_configs, test_gpu_cuda, prepare_saving_dir, save_best_checkpoints
from utils.utils import get_logging
from utils.utils import accuracy
from utils.evaluation import test_DMS, test_rmsf_cor


def _get_temperature(simclr, configs):
    """Return the effective temperature scalar for clip_infonce.

    If learnable_temperature is enabled, derives τ = 1 / exp(logit_scale) (clamped so
    logit_scale can't grow so large that gradients vanish).  Otherwise returns the fixed
    scalar from the config — identical behaviour to the original code.
    """
    if getattr(configs.train_settings, 'learnable_temperature', False):
        return 1.0 / torch.exp(simclr.logit_scale).clamp(max=100)
    return configs.train_settings.temperature


def prepare_loss_MD(simclr, traj, batch_tokens, criterion, loss,
                    accelerator, configs,
                    masked_lm_data_collator):
    MLM_loss = 0
    features_MD, features_seq, residue_seq, graph_feature_embedding, residue_feature_embedding = simclr(
                    graph=traj, batch_tokens=batch_tokens, return_embedding=True)

    temperature = _get_temperature(simclr, configs)
    loss_type = getattr(configs.train_settings, 'loss_type', 'clip_infonce')
    if loss_type == 'info_nce':
        logits, labels = info_nce_loss(features_MD, features_seq, 2, temperature, accelerator)
        simclr_loss = torch.mean(criterion(logits, labels))
    else:
        logits, labels = clip_infonce(features_MD, features_seq, temperature, accelerator)
        simclr_loss = 0.5 * (torch.mean(criterion(logits, labels)) + torch.mean(criterion(logits.T, labels)))
    loss += simclr_loss

    if hasattr(configs.model.esm_encoder, "MLM"):
        if configs.model.esm_encoder.MLM.enable:
            mlm_inputs, mlm_labels = masked_lm_data_collator.mask_tokens(batch_tokens)
            if hasattr(configs.model.esm_encoder.MLM, "mode") and configs.model.esm_encoder.MLM.mode == "contrast":
                features_seq_mask, _ = simclr.model_seq(mlm_inputs)
                logits_mask, labels_mask = clip_infonce(features_seq_mask, features_seq,
                                                        temperature, accelerator)
                MLM_loss = torch.mean(criterion(logits_mask, labels_mask))
            else:
                prediction_scores = simclr.model_seq(mlm_inputs, return_logits=True)
                vocab_size = simclr.model_seq.alphabet.all_toks.__len__()
                MLM_loss = torch.mean(criterion(prediction_scores.view(-1, vocab_size), mlm_labels.view(-1)))

            loss += MLM_loss

    return loss, simclr_loss, MLM_loss, logits, labels


def prepare_loss_MD_multipos(simclr, trajs_list, batch_tokens, criterion, loss,
                             accelerator, configs, masked_lm_data_collator):
    """Multi-positive variant: forwards all R replicate embeddings and computes
    clip_infonce_multipos.  trajs_list is a list of R numpy arrays [B, 768]."""
    MLM_loss = 0

    # ── Sequence encoder (run once) ──────────────────────────────────────────
    features_seq, residue_seq, graph_feature_embedding, residue_feature_embedding = simclr(
        batch_tokens=batch_tokens, mode='sequence', return_embedding=True)

    # ── MD encoder: one forward per replicate ────────────────────────────────
    _md_noise = getattr(configs.train_settings, 'md_noise', None)
    features_MD_list = []
    for traj_r in trajs_list:
        traj_tensor = torch.tensor(np.array(traj_r)).float().to(accelerator.device)
        if _md_noise is not None and getattr(_md_noise, 'enable', False) and simclr.training:
            traj_tensor = traj_tensor + torch.randn_like(traj_tensor) * _md_noise.sigma
        features_MD_list.append(simclr(graph=traj_tensor, mode='MD'))   # [B, protein_out_dim]

    temperature = _get_temperature(simclr, configs)
    simclr_loss, logits, labels = clip_infonce_multipos(
        features_seq, features_MD_list, temperature, accelerator)
    loss = loss + simclr_loss

    # ── Optional MLM auxiliary loss (unchanged from single-rep version) ──────
    if hasattr(configs.model.esm_encoder, "MLM") and configs.model.esm_encoder.MLM.enable:
        mlm_inputs, mlm_labels = masked_lm_data_collator.mask_tokens(batch_tokens)
        if hasattr(configs.model.esm_encoder.MLM, "mode") and \
                configs.model.esm_encoder.MLM.mode == "contrast":
            features_seq_mask, _ = simclr.model_seq(mlm_inputs)
            logits_mask, labels_mask = clip_infonce(
                features_seq_mask, features_seq, temperature, accelerator)
            MLM_loss = torch.mean(criterion(logits_mask, labels_mask))
        else:
            prediction_scores = simclr.model_seq(mlm_inputs, return_logits=True)
            vocab_size = simclr.model_seq.alphabet.all_toks.__len__()
            MLM_loss = torch.mean(criterion(
                prediction_scores.view(-1, vocab_size), mlm_labels.view(-1)))
        loss = loss + MLM_loss

    return loss, simclr_loss, MLM_loss, logits, labels


def training_loop_MD(simclr, start_step, start_loss, start_rmsf,
                     train_loader, val_loader, test_loader, batch_converter, criterion,
                     optimizer_x, optimizer_seq, scheduler_x, scheduler_seq,
                     result_path, logging, configs, replicate,
                     masked_lm_data_collator=None,
                     compute_rmsf_cor=True, total_replicates=None, total_steps=None,
                     **kwargs):
    torch.cuda.empty_cache()
    accelerator = kwargs['accelerator']

    train_loss = 0
    n_steps = start_step
    n_sub_steps = start_step - 1
    epoch_num = 0

    # Step-budget formula generalized for callers with a different replicate count/total
    # (e.g. mdCATH pretraining, with N replicates instead of Atlas's 3). Defaults preserve
    # the original Atlas-only formula exactly.
    _n_rep = total_replicates if total_replicates is not None else 3
    _n_steps_total = total_steps if total_steps is not None else configs.train_settings.num_steps

    best_val_loss = start_loss
    best_val_graph_loss = start_loss
    # best_val_dms_corr = start_dms
    best_val_rmsf_cor = start_rmsf
    while True:
        epoch_num += 1
        if accelerator.is_main_process:
            logging.info(f"Epoch {epoch_num}")

        losses = AverageMeter()

        progress_bar = tqdm(range(0, int(np.ceil(len(train_loader) / configs.train_settings.gradient_accumulation))),
                            disable=True, leave=True, desc=f"Epoch {epoch_num} steps")
        bsz = configs.train_settings.batch_size
        start = time.time()
        for idx, batch in enumerate(train_loader):
            with accelerator.accumulate(simclr):
                simclr.train()
                batch_seq = [(batch['pid'][i], str(batch['seq'][i])) for i in range(len(batch['seq']))]
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
                trajs = batch["traj"]
                batch_tokens = batch_tokens.to(accelerator.device)
                loss = torch.tensor(0).float().to(accelerator.device)

                _multi_pos = getattr(configs.train_settings, 'multi_positive', False)
                if _multi_pos:
                    # trajs is list of R numpy arrays [B, 768] — noise applied inside
                    loss, simclr_loss, MLM_loss, logits, labels = prepare_loss_MD_multipos(
                        simclr, trajs, batch_tokens, criterion, loss, accelerator, configs,
                        masked_lm_data_collator)
                else:
                    trajs = torch.tensor(np.array(trajs)).to(accelerator.device)
                    # MD noise augmentation (training only)
                    _md_noise = getattr(configs.train_settings, 'md_noise', None)
                    if _md_noise is not None and getattr(_md_noise, 'enable', False) and simclr.training:
                        trajs = trajs + torch.randn_like(trajs) * _md_noise.sigma
                    loss, simclr_loss, MLM_loss, logits, labels = prepare_loss_MD(
                        simclr, trajs, batch_tokens, criterion, loss, accelerator, configs,
                        masked_lm_data_collator)

                avg_loss = accelerator.gather(loss.repeat(configs.train_settings.batch_size)).mean()
                train_loss += avg_loss.item() / configs.train_settings.gradient_accumulation

                # NaN guard: skip backward + optimizer step for any batch with NaN/Inf loss
                # to prevent soft NaN from propagating through Adam moment estimates.
                if torch.isnan(loss) or torch.isinf(loss):
                    if accelerator.is_main_process:
                        logging.warning(f"NaN/Inf loss detected at step {n_steps}, skipping batch")
                    optimizer_seq.zero_grad()
                    if not (configs.model.MD_encoder.fine_tuning.enable is False and
                            configs.model.MD_encoder.fine_tuning_projct.enable is False):
                        optimizer_x.zero_grad()
                    n_sub_steps += 1
                    continue

                accelerator.backward(loss)

                # Gradient clipping — controlled by optimizer.grad_clip_norm in config.
                # Set grad_clip_norm: 0 to disable.
                # NOTE: logit_scale is intentionally excluded from clipping.  Its gradient
                # (~B/τ ≈ 10 at init) would otherwise dominate the total norm and scale
                # adapter gradients down by ~10× every step, starving the adapters.
                # Adam with lr=1e-4 moves logit_scale safely without clipping.
                if configs.optimizer.grad_clip_norm > 0:
                    _unwrapped = accelerator.unwrap_model(simclr)
                    _clip_params = (
                        list(_unwrapped.model_seq.parameters()) +
                        list(_unwrapped.model_x.parameters())
                    )
                    accelerator.clip_grad_norm_(_clip_params, configs.optimizer.grad_clip_norm)

                if not (configs.model.MD_encoder.fine_tuning.enable is False and configs.model.MD_encoder.fine_tuning_projct.enable is False):
                    optimizer_x.step()

                optimizer_seq.step()

                if not (configs.model.MD_encoder.fine_tuning.enable is False and configs.model.MD_encoder.fine_tuning_projct.enable is False):
                    scheduler_x.step()

                scheduler_seq.step()

                losses.update(loss.item(), bsz)

                if not (configs.model.MD_encoder.fine_tuning.enable is False and configs.model.MD_encoder.fine_tuning_projct.enable is False):
                    optimizer_x.zero_grad()
                optimizer_seq.zero_grad()

            if accelerator.sync_gradients:
                if accelerator.is_main_process:
                    _temp_val = _get_temperature(simclr, configs)
                    _temp_scalar = _temp_val.item() if hasattr(_temp_val, 'item') else float(_temp_val)
                    wandb.log({
                        "train/step_loss": train_loss,
                        "train/lr_structure": optimizer_x.param_groups[0]['lr'],
                        "train/lr_sequence": optimizer_seq.param_groups[0]['lr'],
                        "train/temperature": _temp_scalar,
                    }, step=n_steps)

                if n_steps % configs.checkpoints_every == 0 and n_steps != 0:
                    print("save_checkpoints")
                    accelerator.wait_for_everyone()
                    loss_val, val_graph_loss = evaluation_loop_MD(simclr, val_loader, labels, batch_converter, criterion, configs,
                                    simclr_loss=simclr_loss, MLM_loss=MLM_loss, losses=losses,
                                    n_steps=n_steps, scheduler_seq=scheduler_seq, result_path=result_path,
                                    scheduler_x=scheduler_x, logits=logits,
                                    loss=loss, bsz=bsz,
                                    masked_lm_data_collator=masked_lm_data_collator, logging=logging,
                                    accelerator=accelerator)
                    save_checkpoints(accelerator.unwrap_model(optimizer_x),
                                     accelerator.unwrap_model(optimizer_seq),
                                     result_path, accelerator.unwrap_model(simclr), n_steps, logging, epoch_num, loss_val)

                if n_steps % configs.valid_settings.do_every == 0 and n_steps != 0:
                    print("in evaluation loop")
                    accelerator.wait_for_everyone()
                    loss_val, val_graph_loss = evaluation_loop_MD(simclr, val_loader, labels, batch_converter, criterion, configs,
                                    simclr_loss=simclr_loss, MLM_loss=MLM_loss, losses=losses,
                                    n_steps=n_steps, scheduler_seq=scheduler_seq, result_path=result_path,
                                    scheduler_x=scheduler_x, logits=logits,
                                    loss=loss, bsz=bsz,
                                    masked_lm_data_collator=masked_lm_data_collator, logging=logging,
                                    accelerator=accelerator)

                    # val_DMS_corr = test_DMS(configs, simclr.model_seq.esm2, simclr.model_seq.alphabet, n_steps=n_steps, logging=logging)
                    # best_val_dms_corr = save_best_checkpoints(accelerator.unwrap_model(optimizer_x),
                    #                  accelerator.unwrap_model(optimizer_seq),
                    #                  result_path, accelerator.unwrap_model(simclr), n_steps, logging, epoch_num,
                    #                  best_val_dms_corr, val_DMS_corr, 'best_val_dms_corr', direction='high')

                    # mdCATH has no RMSF ground truth — pretraining-on-mdCATH calls pass
                    # compute_rmsf_cor=False to skip this (test_rmsf_cor requires
                    # {pid}_analysis/{pid}.pdb + {pid}_RMSF.tsv, which only exist for Atlas).
                    if compute_rmsf_cor:
                        val_rmsf_cor = test_rmsf_cor(val_loader, simclr.model_seq.alphabet, configs, simclr.model_seq.esm2,
                                                     n_steps, logging, replicate)
                        val_rmsf_cor = abs(val_rmsf_cor)
                        if accelerator.is_main_process:
                            wandb.log({"val/val_rmsf_cor": val_rmsf_cor,}, step=n_steps)
                        best_val_rmsf_cor = save_best_checkpoints(accelerator.unwrap_model(optimizer_x),
                                         accelerator.unwrap_model(optimizer_seq),
                                         result_path, accelerator.unwrap_model(simclr), n_steps, logging, epoch_num,
                                         best_val_rmsf_cor, val_rmsf_cor, 'best_val_rmsf_cor', direction='high')

                    best_val_loss = save_best_checkpoints(accelerator.unwrap_model(optimizer_x),
                                     accelerator.unwrap_model(optimizer_seq),
                                     result_path, accelerator.unwrap_model(simclr), n_steps, logging, epoch_num,
                                     best_val_loss, loss_val, 'best_val_whole_loss')
                    # best_val_graph_loss = save_best_checkpoints(accelerator.unwrap_model(optimizer_x),
                    #                  accelerator.unwrap_model(optimizer_seq),
                    #                  result_path, accelerator.unwrap_model(simclr), n_steps, logging, epoch_num,
                    #                  best_val_graph_loss, val_graph_loss, 'best_val_graph_loss')

                progress_bar.update(1)
                n_steps += 1
                train_loss = 0

            n_sub_steps += 1
            repli_num = _n_steps_total if replicate == _n_rep - 1 else (_n_steps_total // _n_rep) * (replicate + 1)
            if n_steps > repli_num:
                break

        end = time.time()

        repli_num = _n_steps_total if replicate == _n_rep - 1 else (_n_steps_total // _n_rep) * (replicate + 1)
        if n_steps > repli_num:
            break

        if accelerator.is_main_process:
            logging.info(f"one epoch cost {(end - start):.2f}, number of trained steps {n_steps}")

    return n_steps, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, best_val_loss, best_val_rmsf_cor


def evaluation_loop_MD(simclr, val_loader, labels, batch_converter, criterion, configs, logging, **kwargs):
    accelerator = kwargs['accelerator']

    n_steps = kwargs['n_steps']
    result_path = kwargs['result_path']

    losses = kwargs['losses']
    loss = kwargs['loss']
    MLM_loss = kwargs['MLM_loss']
    simclr_loss = kwargs['simclr_loss']
    masked_lm_data_collator = kwargs['masked_lm_data_collator']
    logits = kwargs['logits']

    bsz = kwargs['bsz']
    scheduler_x = kwargs['scheduler_x']
    scheduler_seq = kwargs['scheduler_seq']

    _multi_pos = getattr(configs.train_settings, 'multi_positive', False)

    # In multi-positive mode logits are [B*R, B]; skip neg-sim metrics (different geometry).
    if not _multi_pos and logits is not None:
        l_prob = logits[:, 0].mean()
        negsim_struct_struct = log_negative_mean_logtis(logits, "struct_struct", bsz)
        negsim_struct_seq    = log_negative_mean_logtis(logits, "struct_seq",    bsz)
        negsim_seq_seq       = log_negative_mean_logtis(logits, "seq_seq",       bsz)
        top1, top5 = accuracy(logits, labels, topk=(1, 1))
    else:
        l_prob = logits[:, 0].mean() if logits is not None else torch.tensor(0.)
        negsim_struct_struct = negsim_struct_seq = negsim_seq_seq = 0.0
        top1 = [accuracy(logits, labels, topk=(1, 1))[0][0]
                if logits is not None else torch.tensor(0.)]

    if accelerator.is_main_process:
        logging.info(f'evaluation - step {n_steps}')
        logging.info(
            f"step:{n_steps} Loss:{loss:.4f},graph_loss:{simclr_loss:.4f},Top1 accuracy:{top1[0]},P:{l_prob.item():.2f},"
            f"N_st_st:{negsim_struct_struct:.2f},N_st_s:{negsim_struct_seq:.2f},N_s_s:{negsim_seq_seq:.2f}")

        train_log = {
            "train/loss": float(loss),
            "train/graph_loss": float(simclr_loss),
            "train/top1_accuracy": float(top1[0]),
            "train/p": l_prob.item(),
            "train/n_st_st": negsim_struct_struct,
            "train/n_st_s": negsim_struct_seq,
            "train/n_s_s": negsim_seq_seq,
        }
        if hasattr(configs.model.esm_encoder, "MLM") and configs.model.esm_encoder.MLM.enable:
            logging.info(f"step:{n_steps} MLM_loss: {MLM_loss:.4f}")
            train_log["train/mlm_loss"] = float(MLM_loss)
        wandb.log(train_log, step=n_steps)

    loss_val_sum = 0
    graph_loss_sum = 0
    MLM_loss_sum = 0
    l_prob, l_prob_residue = 0, 0
    negsim_struct_seq, negsim_struct_seq_residue = 0, 0
    negsim_seq_seq, negsim_seq_seq_residue = 0, 0
    negsim_struct_struct, negsim_struct_struct_residue = 0, 0

    k = 0
    progress_bar = tqdm(range(0, len(val_loader)), disable=True, leave=True, desc=f"Evaluation steps")
    for batch in val_loader:
        batch_seq = [(batch['pid'][i], str(batch['seq'][i])) for i in range(len(batch['seq']))]
        bsz = len(batch['seq'])     # was: len(batch) — that returned the dict's key count (3)
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq)
        trajs = batch["traj"]
        batch_tokens = batch_tokens.to(accelerator.device)

        simclr.eval()
        with torch.inference_mode():
            if _multi_pos:
                # trajs is already a list of R numpy arrays [B, 768] from collate_multipos
                loss_val_sum, graph_loss, MLM_loss, logits, labels = prepare_loss_MD_multipos(
                    simclr, trajs, batch_tokens, criterion, loss_val_sum, accelerator, configs,
                    masked_lm_data_collator)
                graph_loss_sum += graph_loss
                MLM_loss_sum   += MLM_loss
                # l_prob / negsim_* stay 0 — logits geometry ([B*R, B]) differs from single-rep
            else:
                trajs = torch.tensor(np.array(trajs)).to(accelerator.device)
                loss_val_sum, graph_loss, MLM_loss, logits, labels = prepare_loss_MD(
                    simclr, trajs, batch_tokens, criterion, loss_val_sum, accelerator, configs,
                    masked_lm_data_collator)
                graph_loss_sum += graph_loss
                MLM_loss_sum   += MLM_loss
                l_prob += logits[:, 0].mean().item()
                negsim_struct_struct += log_negative_mean_logtis(logits, "struct_struct", bsz)
                negsim_struct_seq    += log_negative_mean_logtis(logits, "struct_seq",    bsz)
                negsim_seq_seq       += log_negative_mean_logtis(logits, "seq_seq",       bsz)

            k = k + 1

        progress_bar.update(1)

    loss_val_sum = loss_val_sum / float(k)
    graph_loss_sum = graph_loss_sum / float(k)
    l_prob = l_prob / float(k)
    negsim_struct_struct = negsim_struct_struct / float(k)
    negsim_struct_seq = negsim_struct_seq / float(k)
    negsim_seq_seq = negsim_seq_seq / float(k)

    (loss_val, val_graph_loss, val_l_prob, val_negsim_struct_struct, val_negsim_struct_seq,
     val_negsim_seq_seq) = (float(loss_val_sum), float(graph_loss_sum), float(l_prob),
                                    float(negsim_struct_struct), float(negsim_struct_seq),
                                    float(negsim_seq_seq))

    MLM_loss_sum = float(MLM_loss_sum) / float(k)
    if accelerator.is_main_process:
        logging.info(
            f"step:{n_steps} Val_loss:{loss_val:.4f},val_graph_loss:{val_graph_loss:.4f},val_P:{val_l_prob:.2f},"
            f"val_N_st_st:{val_negsim_struct_struct:.2f},val_N_st_s:{val_negsim_struct_seq:.2f},"
            f"val_N_s_s:{val_negsim_seq_seq:.2f}")

        val_log = {
            "val/loss": loss_val,
            "val/graph_loss": val_graph_loss,
            "val/p": val_l_prob,
            "val/n_st_st": val_negsim_struct_struct,
            "val/n_st_s": val_negsim_struct_seq,
            "val/n_s_s": val_negsim_seq_seq,
        }
        if hasattr(configs.model.esm_encoder, "MLM") and configs.model.esm_encoder.MLM.enable:
            logging.info(f"step:{n_steps} val_MLM_loss: {MLM_loss_sum:.4f}")
            val_log["val/mlm_loss"] = MLM_loss_sum
        wandb.log(val_log, step=n_steps)

    return loss_val, val_graph_loss


def main(args, dict_configs, config_file_path):
    configs = load_configs(dict_configs, args)
    if args.seed:
        configs.fix_seed = args.seed

    if isinstance(configs.fix_seed, int):
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()

    result_path, checkpoint_path = prepare_saving_dir(configs, config_file_path)

    logging = get_logging(result_path)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=configs.train_settings.mixed_precision,
        gradient_accumulation_steps=configs.train_settings.gradient_accumulation,
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process:
        test_gpu_cuda()
        wandb.init(
            project="DPLM",
            name=os.path.basename(result_path),
            config=dict_configs,
            dir=result_path,
        )

    simclr = prepare_models(logging, configs, accelerator)
    if accelerator.is_main_process:
        logging.info('preparing model is done')

    scheduler_seq, scheduler_x, optimizer_seq, optimizer_x = prepare_optimizer(
        simclr.model_seq, simclr.model_x, logging, configs
    )

    # If learnable temperature is enabled, include logit_scale in the seq optimizer so
    # it gets trained and checkpointed alongside the sequence encoder weights.
    if getattr(configs.train_settings, 'learnable_temperature', False):
        optimizer_seq.add_param_group({
            'params': [simclr.logit_scale],
            'lr': float(configs.optimizer.lr_seq),
        })
        logging.info('learnable_temperature enabled — logit_scale added to optimizer_seq')

    if accelerator.is_main_process:
        logging.info('preparing optimizer is done')

    start_step = 0
    if configs.resume.resume:
        if configs.model.X_module == 'MD':
            simclr, start_step, loss = load_checkpoints_md(simclr, configs,
                            optimizer_seq, optimizer_x, scheduler_seq, scheduler_x,
                            logging, resume_path=configs.resume.resume_path, restart_optimizer=configs.resume.restart_optimizer)
            if loss is None:
                loss = np.inf
    else:
        loss = np.inf
    rmsf_cor=0.0

    alphabet = simclr.model_seq.alphabet
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=configs.model.esm_encoder.max_length)

    if hasattr(configs.model.esm_encoder, "MLM") and configs.model.esm_encoder.MLM.enable:
        masked_lm_data_collator = MaskedLMDataCollator(batch_converter,
                                                       mlm_probability=configs.model.esm_encoder.MLM.mask_ratio)
    else:
        masked_lm_data_collator = None

    # ── mdCATH pretraining (optional) ─────────────────────────────────────────────
    # When train_settings.mdCATH_data_repli_path is present, pretrain on mdCATH first
    # (loss-only — mdCATH has no RMSF ground truth), then proceed into Atlas training
    # exactly as below. Skipped when resuming, since pretraining presumably already
    # happened (or the user is deliberately resuming mid-Atlas-training).
    mdcath_train_paths = getattr(configs.train_settings, 'mdCATH_data_repli_path', None)
    if mdcath_train_paths and not configs.resume.resume:
        mdcath_test_paths = configs.train_settings.mdCATH_test_repli_path
        n_mdcath = len(mdcath_train_paths)
        mdcath_num_steps = getattr(configs.train_settings, 'mdCATH_num_steps',
                                   configs.train_settings.num_steps)

        from data.data_MD import prepare_dataloaders_mdcath
        mdcath_loaders = prepare_dataloaders_mdcath(configs)
        mdcath_loaders = [accelerator.prepare(*loaders) for loaders in mdcath_loaders]

        if accelerator.is_main_process:
            logging.info(f'preparing mdCATH dataloaders are done ({n_mdcath} replicates)')

        simclr, scheduler_seq, scheduler_x, optimizer_seq, optimizer_x = accelerator.prepare(
            simclr, scheduler_seq, scheduler_x, optimizer_seq, optimizer_x)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        if accelerator.is_main_process:
            logging.info(f"Start mdCATH pretraining for {mdcath_num_steps} steps across "
                         f"{n_mdcath} replicates (loss-only — mdCATH has no RMSF data).")

        mdcath_step, mdcath_loss, mdcath_rmsf_unused = 0, np.inf, 0.0
        for rep_idx, (train_dl, val_dl, test_dl) in enumerate(mdcath_loaders):
            (mdcath_step, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq,
             mdcath_loss, mdcath_rmsf_unused) = training_loop_MD(
                simclr, mdcath_step, mdcath_loss, mdcath_rmsf_unused,
                train_dl, val_dl, test_dl, batch_converter, criterion,
                optimizer_x, optimizer_seq, scheduler_x, scheduler_seq,
                result_path, logging, configs, replicate=rep_idx,
                masked_lm_data_collator=masked_lm_data_collator, accelerator=accelerator,
                compute_rmsf_cor=False, total_replicates=n_mdcath, total_steps=mdcath_num_steps)

        if accelerator.is_main_process:
            logging.info('mdCATH pretraining complete — starting Atlas training.')

        # mdCATH's loss isn't on the same scale as Atlas's, and step numbering for the
        # checkpoint-every-N mechanism should restart — both must reset before Atlas begins.
        start_step = 0
        loss = np.inf

        # Fresh optimizer + LR schedule for the Atlas phase: prepare_optimizer's
        # CosineAnnealingWarmupRestarts is built once for configs.train_settings.num_steps
        # (Atlas's), and mdCATH pretraining already consumed mdcath_num_steps worth of
        # stepping on that same schedule — re-running prepare_optimizer gives Atlas its own
        # fresh warmup→peak→decay cycle (and zeroed Adam state) instead of inheriting mdCATH's.
        scheduler_seq, scheduler_x, optimizer_seq, optimizer_x = prepare_optimizer(
            accelerator.unwrap_model(simclr).model_seq, accelerator.unwrap_model(simclr).model_x,
            logging, configs)

        if getattr(configs.train_settings, 'learnable_temperature', False):
            optimizer_seq.add_param_group({
                'params': [accelerator.unwrap_model(simclr).logit_scale],
                'lr': float(configs.optimizer.lr_seq),
            })
            logging.info('learnable_temperature enabled — logit_scale re-added to fresh optimizer_seq')

        if accelerator.is_main_process:
            logging.info('optimizer/scheduler reset for Atlas phase')

    _multi_pos = getattr(configs.train_settings, 'multi_positive', False)

    if _multi_pos:
        # ── Multi-positive mode: single dataloader, all 3 replicates per protein ──
        from data.data_MD import prepare_dataloaders_multipos
        train_dl, val_dl, test_dl = prepare_dataloaders_multipos(configs)
        train_dl, val_dl, test_dl = accelerator.prepare(train_dl, val_dl, test_dl)

        if accelerator.is_main_process:
            logging.info('preparing dataloaders are done (multi-positive mode)')

        simclr, scheduler_seq, scheduler_x, optimizer_seq, optimizer_x = accelerator.prepare(
            simclr, scheduler_seq, scheduler_x, optimizer_seq, optimizer_x)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        if accelerator.is_main_process:
            logging.info(f"Start contrastive training for {configs.train_settings.num_steps} steps.")
            train_steps = np.ceil(len(train_dl) / configs.train_settings.gradient_accumulation)
            logging.info(f'Number of train steps per epoch (multi-pos): {int(train_steps)}')
            logging.info(f"Training with: {accelerator.device} and fix_seed = {configs.fix_seed}")

        start_step, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, loss, rmsf_cor = training_loop_MD(
            simclr, start_step, loss, rmsf_cor, train_dl, val_dl, test_dl, batch_converter, criterion,
            optimizer_x, optimizer_seq, scheduler_x, scheduler_seq,
            result_path, logging, configs, replicate=2,
            masked_lm_data_collator=masked_lm_data_collator, accelerator=accelerator)

    else:
        # ── Sequential-replicate mode: 3 separate dataloaders ────────────────────
        from data.data_MD import prepare_dataloaders
        ((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
         (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
         (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2)) = prepare_dataloaders(configs)

        ((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
        (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
        (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2)) = accelerator.prepare(
            ((train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0),
             (train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1),
             (train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2)))

        if accelerator.is_main_process:
            logging.info('preparing dataloaders are done')

        simclr, scheduler_seq, scheduler_x, optimizer_seq, optimizer_x = accelerator.prepare(
            simclr, scheduler_seq, scheduler_x, optimizer_seq, optimizer_x)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        if accelerator.is_main_process:
            logging.info(f"Start contrastive training for {configs.train_settings.num_steps} steps.")
            train_steps = np.ceil(len(train_dataloader_repli_0) / configs.train_settings.gradient_accumulation)
            train_steps = train_steps * 3
            logging.info(f'Number of train steps per epoch: {int(train_steps)}')
            logging.info(f"Training with: {accelerator.device} and fix_seed = {configs.fix_seed}")

        start_step, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, loss, rmsf_cor = training_loop_MD(
            simclr, start_step, loss, rmsf_cor,
            train_dataloader_repli_0, val_dataloader_repli_0, test_dataloader_repli_0,
            batch_converter, criterion, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq,
            result_path, logging, configs, replicate=0,
            masked_lm_data_collator=masked_lm_data_collator, accelerator=accelerator)
        start_step, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, loss, rmsf_cor = training_loop_MD(
            simclr, start_step, loss, rmsf_cor,
            train_dataloader_repli_1, val_dataloader_repli_1, test_dataloader_repli_1,
            batch_converter, criterion, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq,
            result_path, logging, configs, replicate=1,
            masked_lm_data_collator=masked_lm_data_collator, accelerator=accelerator)
        start_step, accelerator, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq, loss, rmsf_cor = training_loop_MD(
            simclr, start_step, loss, rmsf_cor,
            train_dataloader_repli_2, val_dataloader_repli_2, test_dataloader_repli_2,
            batch_converter, criterion, optimizer_x, optimizer_seq, scheduler_x, scheduler_seq,
            result_path, logging, configs, replicate=2,
            masked_lm_data_collator=masked_lm_data_collator, accelerator=accelerator)

    if accelerator.is_main_process:
        wandb.finish()

    accelerator.free_memory()
    torch.cuda.empty_cache()
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument("--config_path", "-c", help="The location of config file", default='./config.yaml')
    parser.add_argument("--result_path", default=None,
                        help="result_path, if set by command line, overwrite the one in config.yaml, "
                             "by default is None")
    parser.add_argument("--resume_path", default=None,
                        help="if set, overwrite the one in config.yaml, by default is None")
    # parser.add_argument("--num_end_adapter_layers", default=None, help="num_end_adapter_layers")
    # parser.add_argument("--module_type", default=None, help="module_type for adapterh")
    parser.add_argument("--seed", default=None, type=int, help="random seed")
    parser.add_argument("--restart_optimizer", default=None, type=int, help="restart_optimizer")

    args_main = parser.parse_args()
    config_path = args_main.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main(args_main, config_file, config_path)
