"""
train_dccm_attn_v2.py — supervised DCCM benchmark with a CHANNEL-MATCHED, multi-seed protocol.

Why this exists
---------------
evaluate/DCCM/train_dccm_attn.py is correct on its own, but its fairness guarantee ("every
method contributes the SAME number of attention channels", train_dccm_attn.py:15-16) only
holds if every backbone is run with the same --num_attn_layers. Running DPLM and the
baselines as SEPARATE sbatch jobs with different values silently breaks it, because the
attention pathway is Linear(C -> 1) (model_dccm_attn.py:59), so C is model capacity:

    --num_attn_layers 0  ==  "all layers"  ->  C = 33 (ESM2), 24 (ProstT5), 12 (SeqDance)
    --num_attn_layers 10                   ->  C = 10

This script removes the failure mode instead of documenting it:

  1. every method runs in ONE process with ONE set of hyper-parameters,
  2. the realised channel count C is recorded per method and, unless --allow_unmatched_C is
     given, a mismatch is a hard error rather than a footnote,
  3. --num_attn_layers is capped to the shallowest backbone in the comparison so an equal C
     is actually achievable (SeqDance is a 12-layer esm2_t12_35M model),
  4. training is repeated over several seeds (--seeds); the per-protein score used for the
     headline comparison is the mean over seeds, which keeps small deltas from being seed
     noise. Backbone feature extraction happens ONCE per method and is reused across seeds,
     so k seeds cost k cheap predictor fits, not k backbone passes,
  5. predicted DCCM matrices for the test set are written to .npz so figures and case
     studies can be regenerated offline on a login node, with no GPU and no model loading.

Outputs (in --output_dir)
    per_protein_corr_attn_{M}.csv         seed-averaged per-protein Pearson/Spearman
    per_seed_corr_{M}.csv                 one row per (seed, protein)
    preds_{M}.npz                         predicted DCCM per test protein (float16)
    ground_truth.npz                      true DCCM per test protein (float16)
    protein_lengths.csv                   pid,length — enables the stratification figure
    run_manifest.json                     every setting, plus realised C per method
    dccm_attn_predictor_{M}_seed{S}.pth   checkpoints

Example (Delta)
    python evaluate/DCCM_v2/train_dccm_attn_v2.py \
      --train_data_path <processed_data> --test_data_path <processed_test> \
      --analysis_path <analysis> --dccm_dir <DCCM_dir3> --dccm_replicate R3 \
      --methods dplm+esm2+prostt5+seqdance \
      --dplm_config <cfg.yaml> --dplm_checkpoint <ckpt.pth> --seqdance_path <SeqDance/model> \
      --num_attn_layers 10 --head cosine --loss mix --fusion weighted \
      --seeds 0,1,2 --output_dir <out>
"""

import argparse
import csv
import json
import os
import re
import sys
from time import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'DCCM'))
sys.path.insert(0, os.path.join(_HERE, '..', 'methodology'))

from data_dccm_attn import load_attn_model, build_samples_attn, METHOD_KEY
from model_dccm_attn import DCCMAttnPredictor, protein_dccm_corr
from model_dccm import _upper_tri_mask, dccm_mse_loss
# (dccm_loss / _to_tensors / evaluate are inlined below)


# ── inlined from train_dccm_attn.py (the pre-v2 module, removed
#    in the public release; only these helpers were ever used from it) ─────
def _to_tensors(sample, device):
    emb  = torch.from_numpy(sample['emb']).to(device)                 # [L, D]
    attn = torch.from_numpy(sample['attn']).to(device).float()        # [C, L, L]
    dccm = torch.from_numpy(sample['dccm']).to(device)                # [L, L]
    return emb, attn, dccm

def dccm_corr_loss(pred, gt):
    """1 − Pearson over the strict upper triangle. Scale-invariant → does NOT shrink the
    predicted magnitudes the way MSE does (which is why MSE-trained maps look washed out)."""
    m = _upper_tri_mask(pred.shape[0], pred.device)
    p, g = pred[m], gt[m]
    p = p - p.mean()
    g = g - g.mean()
    corr = (p * g).sum() / (p.norm() * g.norm() + 1e-8)
    return 1.0 - corr

def dccm_loss(pred, gt, mode='mse', corr_lambda=1.0):
    if mode == 'mse':
        return dccm_mse_loss(pred, gt)
    if mode == 'corr':
        return dccm_corr_loss(pred, gt)
    if mode == 'mix':
        return dccm_mse_loss(pred, gt) + corr_lambda * dccm_corr_loss(pred, gt)
    raise ValueError(f"loss must be mse|corr|mix, got {mode}")

def evaluate(model, samples, device):
    """Return dict pid → (pearson, spearman)."""
    model.eval()
    results = {}
    with torch.no_grad():
        for s in samples:
            emb, attn, _ = _to_tensors(s, device)
            pred = model(emb, attn).cpu().numpy()
            results[s['pid']] = protein_dccm_corr(pred, s['dccm'])
    return results

# Transformer depth per backbone — used to cap --num_attn_layers so that an equal channel
# count is actually reachable for every method in the comparison.
BACKBONE_DEPTH = {'dplm': 33,       # esm2_t33_650M_UR50D + end adapters
                  'esm2': 33,       # esm2_t33_650M_UR50D
                  'esmc': 36,       # esmc_600m (esmc_300m has 30 — see _esmc_depth below)
                  'prostt5': 24,    # Rostlab/ProstT5 T5 encoder
                  'seqdance': 12}   # ESMDance, built on esm2_t12_35M_UR50D


def backbone_depth(method, esmc_model='esmc_600m'):
    """Depth of `method`'s transformer stack, resolving the ESM-C size actually requested.

    ESM-C is the one backbone whose depth depends on a CLI flag (300M → 30 blocks,
    600M → 36), so the static BACKBONE_DEPTH entry alone would over-count for the 300M
    size and let --num_attn_layers exceed what the model can supply.
    """
    if method == 'esmc':
        from esmc_local import esmc_num_layers
        return esmc_num_layers(esmc_model)
    return BACKBONE_DEPTH[method]


def parse_args():
    p = argparse.ArgumentParser(
        description='Channel-matched, multi-seed supervised DCCM benchmark.')
    p.add_argument('--train_data_path', required=True)
    p.add_argument('--test_data_path', required=True)
    p.add_argument('--analysis_path', required=True)
    p.add_argument('--methods', default='dplm+esm2+prostt5+seqdance',
                   help='Backbones: dplm, esm2, esmc, prostt5, seqdance. '
                        'Join with "+" (sbatch --export splits on commas).')
    p.add_argument('--esmc_model', default='esmc_600m',
                   choices=['esmc_300m', 'esmc_600m'],
                   help='ESM-C size when esmc is in --methods (default: esmc_600m).')
    p.add_argument('--dplm_config', default=None)
    p.add_argument('--dplm_checkpoint', default=None)
    p.add_argument('--seqdance_path', default=None)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--dccm_dir', default=None)
    p.add_argument('--dccm_replicate', default=None, choices=['R1', 'R2', 'R3'])
    # attention features
    p.add_argument('--num_attn_layers', type=int, default=10,
                   help='Last-N attention layers per backbone. Capped to the shallowest '
                        'backbone present so every method yields the same C. 0 = all '
                        'layers, which does NOT give equal C across backbones and is '
                        'therefore only allowed with --allow_unmatched_C.')
    p.add_argument('--head_reduce', default='mean', choices=['mean', 'none'])
    p.add_argument('--repr_layer', type=int, default=None)
    p.add_argument('--allow_unmatched_C', action='store_true',
                   help='Permit different attention-channel counts across methods. Off by '
                        'default: an unmatched C is the v1 bug this script exists to stop.')
    # predictor / optimisation
    p.add_argument('--head', default='cosine', choices=['cosine', 'bilinear'])
    p.add_argument('--loss', default='mix', choices=['mse', 'corr', 'mix'])
    p.add_argument('--corr_lambda', type=float, default=1.0)
    p.add_argument('--fusion', default='weighted', choices=['weighted', 'add'])
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--proj_dim', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--seeds', default='0,1,2',
                   help='Comma-separated training seeds. The headline per-protein score is '
                        'the mean over seeds.')
    p.add_argument('--max_train', type=int, default=None)
    p.add_argument('--max_test', type=int, default=None)
    p.add_argument('--save_preds', dest='save_preds', action='store_true', default=True)
    p.add_argument('--no_save_preds', dest='save_preds', action='store_false')
    return p.parse_args()


def train_one(train_split, val_split, test_samples, args, device, seed, log_prefix=''):
    """Fit the predictor for one seed; return (model, {pid: (pearson, spearman)})."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DCCMAttnPredictor(
        input_dim=train_split[0]['emb'].shape[1],
        attn_channels=train_split[0]['attn'].shape[0],
        hidden_dim=args.hidden_dim, proj_dim=args.proj_dim, dropout=args.dropout,
        head=args.head, fusion=args.fusion).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    best_val, best_state = -np.inf, None
    rng = np.random.default_rng(seed)
    for epoch in range(args.epochs):
        model.train()
        t0, total = time(), 0.0
        for k in rng.permutation(len(train_split)):
            emb, attn, dccm = _to_tensors(train_split[k], device)
            loss = dccm_loss(model(emb, attn), dccm, mode=args.loss,
                             corr_lambda=args.corr_lambda)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        val = evaluate(model, val_split, device)
        vp = [p for p, _ in val.values() if not np.isnan(p)]
        vm = float(np.mean(vp)) if vp else -np.inf
        print(f'{log_prefix}epoch {epoch:>3}  {time()-t0:5.1f}s  '
              f'loss={total/max(len(train_split),1):.4f}  val_pearson={vm:.4f}')
        if vm > best_val:
            best_val, best_state = vm, {k: v.detach().clone()
                                        for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f'{log_prefix}best val_pearson={best_val:.4f}')
    return model, evaluate(model, test_samples, device), best_val


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    methods = [m.strip().lower() for m in re.split(r'[,+;\s]+', args.methods) if m.strip()]
    seeds = [int(s) for s in re.split(r'[,+;\s]+', args.seeds) if s.strip()]
    print(f'Methods: {methods}   Seeds: {seeds}')

    # ── channel matching ──────────────────────────────────────────────────────
    if args.num_attn_layers and args.num_attn_layers > 0:
        depths = [backbone_depth(m, args.esmc_model)
                  for m in methods if m in BACKBONE_DEPTH]
        cap = min(depths) if depths else args.num_attn_layers
        num_attn_layers = min(args.num_attn_layers, cap)
        if num_attn_layers != args.num_attn_layers:
            print(f'[match] --num_attn_layers {args.num_attn_layers} exceeds the shallowest '
                  f'backbone in this comparison ({cap} layers); using {num_attn_layers} so '
                  f'every method contributes the same C.')
    else:
        num_attn_layers = None
        if not args.allow_unmatched_C:
            raise SystemExit(
                '--num_attn_layers 0 means "all layers", which gives a DIFFERENT channel '
                'count per backbone (ESM2 33, ProstT5 24, SeqDance 12) and therefore a '
                'different attention-pathway capacity per method. That is exactly the v1 '
                'confound. Pass a positive --num_attn_layers, or --allow_unmatched_C if '
                'you deliberately want the unmatched ablation.')

    from protein_level_emb_md import load_proteins
    train_proteins = load_proteins(args.train_data_path, args.analysis_path,
                                   max_proteins=args.max_train)
    test_proteins = load_proteins(args.test_data_path, args.analysis_path,
                                  max_proteins=args.max_test)
    print(f'Train proteins: {len(train_proteins)}   Test proteins: {len(test_proteins)}')

    manifest = {'methods': methods, 'seeds': seeds,
                'num_attn_layers_requested': args.num_attn_layers,
                'num_attn_layers_used': num_attn_layers,
                'head_reduce': args.head_reduce, 'head': args.head, 'loss': args.loss,
                'corr_lambda': args.corr_lambda, 'fusion': args.fusion,
                'hidden_dim': args.hidden_dim, 'proj_dim': args.proj_dim,
                'dropout': args.dropout, 'lr': args.lr,
                'weight_decay': args.weight_decay, 'epochs': args.epochs,
                'val_frac': args.val_frac, 'dccm_replicate': args.dccm_replicate,
                'train_data_path': args.train_data_path,
                'test_data_path': args.test_data_path,
                'attn_channels_realised': {}, 'val_pearson': {}, 'test_mean_pearson': {},
                # Headline per-protein scores are averaged over seeds, but only ONE seed's
                # predicted matrices are dumped (dumping every seed's would multiply the
                # .npz size by len(seeds) for no analytical gain). case_study.py reads this
                # so the panels can say which seed the pictures come from.
                'preds_seed': seeds[0] if args.save_preds else None}

    gt_saved = False
    for method in methods:
        disp = METHOD_KEY.get(method, method)
        print(f'\n================  {disp}  ================')
        model_bb, extra = load_attn_model(
            method, device, dplm_config=args.dplm_config,
            dplm_checkpoint=args.dplm_checkpoint, seqdance_path=args.seqdance_path,
            esmc_model=args.esmc_model)

        print('Building TRAIN samples …')
        train_samples = build_samples_attn(
            method, train_proteins, model_bb, extra, args.analysis_path, device,
            repr_layer=args.repr_layer, num_attn_layers=num_attn_layers,
            head_reduce=args.head_reduce, dccm_dir=args.dccm_dir,
            replicate=args.dccm_replicate)
        print('Building TEST samples …')
        test_samples = build_samples_attn(
            method, test_proteins, model_bb, extra, args.analysis_path, device,
            repr_layer=args.repr_layer, num_attn_layers=num_attn_layers,
            head_reduce=args.head_reduce, dccm_dir=args.dccm_dir,
            replicate=args.dccm_replicate)
        del model_bb
        if device == 'cuda':
            torch.cuda.empty_cache()
        if not train_samples or not test_samples:
            print(f'[{disp}] no usable samples — skipping.')
            continue

        C = int(train_samples[0]['attn'].shape[0])
        manifest['attn_channels_realised'][disp] = C
        print(f'[{disp}] attention channels C={C}')

        # ground truth + lengths, written once (identical for every method)
        if not gt_saved:
            np.savez_compressed(
                os.path.join(args.output_dir, 'ground_truth.npz'),
                **{s['pid']: np.asarray(s['dccm'], dtype=np.float16) for s in test_samples})
            with open(os.path.join(args.output_dir, 'protein_lengths.csv'), 'w',
                      newline='') as f:
                w = csv.writer(f)
                w.writerow(['pid', 'length'])
                for s in test_samples:
                    w.writerow([s['pid'], s['dccm'].shape[0]])
            gt_saved = True

        # the train/val split is seed-independent so every method and seed sees the same one
        split_rng = np.random.default_rng(0)
        idx = split_rng.permutation(len(train_samples))
        n_val = max(1, int(round(args.val_frac * len(train_samples))))
        val_split = [train_samples[i] for i in sorted(idx[:n_val].tolist())]
        tr_split = [train_samples[i] for i in idx[n_val:]]
        print(f'[{disp}] train={len(tr_split)}  val={len(val_split)}  '
              f'test={len(test_samples)}')

        per_seed_rows, per_seed_scores, last_preds = [], {}, None
        for seed in seeds:
            model, test_res, best_val = train_one(
                tr_split, val_split, test_samples, args, device, seed,
                log_prefix=f'[{disp} seed={seed}] ')
            manifest['val_pearson'].setdefault(disp, {})[str(seed)] = best_val
            per_seed_scores[seed] = test_res
            for pid, (pe, sp) in test_res.items():
                per_seed_rows.append([seed, pid, f'{pe:.6f}', f'{sp:.6f}'])

            torch.save({'model_state_dict': model.state_dict(),
                        'input_dim': test_samples[0]['emb'].shape[1],
                        'attn_channels': C, 'head': args.head, 'fusion': args.fusion,
                        'loss': args.loss, 'hidden_dim': args.hidden_dim,
                        'proj_dim': args.proj_dim, 'dropout': args.dropout,
                        'num_attn_layers': num_attn_layers,
                        'head_reduce': args.head_reduce, 'repr_layer': args.repr_layer,
                        'seed': seed, 'method': disp},
                       os.path.join(args.output_dir,
                                    f'dccm_attn_predictor_{disp}_seed{seed}.pth'))

            if args.save_preds and seed == seeds[0]:
                # matrices from the first seed, for the case-study figures
                model.eval()
                last_preds = {}
                with torch.no_grad():
                    for s in test_samples:
                        emb, attn, _ = _to_tensors(s, device)
                        last_preds[s['pid']] = model(emb, attn).cpu().numpy() \
                                                    .astype(np.float16)

        with open(os.path.join(args.output_dir, f'per_seed_corr_{disp}.csv'), 'w',
                  newline='') as f:
            w = csv.writer(f)
            w.writerow(['seed', 'pid', 'pearson', 'spearman'])
            w.writerows(per_seed_rows)

        # headline score = mean over seeds, per protein
        pids = sorted(test_samples[0] and {s['pid'] for s in test_samples})
        csv_path = os.path.join(args.output_dir, f'per_protein_corr_attn_{disp}.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['pid', 'pearson', 'spearman', 'n_seeds'])
            for pid in pids:
                pe = [per_seed_scores[s][pid][0] for s in seeds if pid in per_seed_scores[s]]
                sp = [per_seed_scores[s][pid][1] for s in seeds if pid in per_seed_scores[s]]
                if not pe:
                    continue
                w.writerow([pid, f'{np.nanmean(pe):.6f}', f'{np.nanmean(sp):.6f}', len(pe)])
        print(f'[{disp}] seed-averaged per-protein correlations → {csv_path}')

        means = [np.nanmean([per_seed_scores[s][pid][0] for s in seeds if pid in
                             per_seed_scores[s]]) for pid in pids]
        means = [m for m in means if not np.isnan(m)]
        manifest['test_mean_pearson'][disp] = float(np.mean(means))
        per_seed_means = {str(s): float(np.nanmean([v[0] for v in
                                                    per_seed_scores[s].values()]))
                          for s in seeds}
        print(f'[{disp}] TEST mean Pearson (seed-averaged) = {np.mean(means):.4f}   '
              f'per-seed = {per_seed_means}')

        if last_preds is not None:
            np.savez_compressed(os.path.join(args.output_dir, f'preds_{disp}.npz'),
                                **last_preds)
            print(f'[{disp}] predicted matrices → preds_{disp}.npz')

        del train_samples, test_samples
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Write the manifest BEFORE the fairness check, so a failed check still leaves the
    # full record of what was run on disk to diagnose from.
    with open(os.path.join(args.output_dir, 'run_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    # ── fairness check ────────────────────────────────────────────────────────
    realised = manifest['attn_channels_realised']
    if len(set(realised.values())) > 1 and not args.allow_unmatched_C:
        raise SystemExit(
            f'Attention-channel counts differ across methods: {realised}. The attention '
            f'pathway is Linear(C -> 1), so this is unequal capacity and the comparison is '
            f'not fair. Lower --num_attn_layers to {min(realised.values())} and re-run.')

    print('\n=== Channel-matched supervised DCCM summary ===')
    print(f'  C per method: {realised}')
    for m, v in sorted(manifest['test_mean_pearson'].items(), key=lambda kv: -kv[1]):
        print(f'  {m:<10} test mean Pearson = {v:.4f}')
    print('\nDone.')


if __name__ == '__main__':
    main()
