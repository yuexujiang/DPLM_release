"""
unsup_dccm_attn_v2.py — unsupervised (training-free) DCCM benchmark, with matrix dumping.

Same read-out as evaluate/DCCM/unsup_dccm_attn.py (attention map, optionally gated by the
embedding cosine), with three additions:

  1. the read-out matrices for every protein are written to .npz, so case-study figures and
     re-analyses can be produced offline on a login node with no GPU,
  2. the realised attention-channel count per method is recorded in run_manifest.json,
  3. --match_layers caps --num_attn_layers to the shallowest backbone in the comparison, so
     a "same last-N layers for everyone" protocol can be run as a robustness check against
     the default "all layers for everyone".

Note on fairness: unlike the supervised predictor, the unsupervised read-out has no learned
per-channel weights — it averages the selected layers into one map — so "all layers for
every backbone" is already a consistent protocol and is the default here. --match_layers is
offered because a reviewer may reasonably ask whether the ranking depends on backbone depth.

Outputs (in --output_dir)
    unsup_attn_corr_{M}.csv    per-protein Pearson/Spearman
    readout_{M}.npz            read-out matrix per protein (float16), if --save_mats
    ground_truth.npz           true DCCM per protein (float16)
    protein_lengths.csv        pid,length
    run_manifest.json
"""

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'DCCM'))
sys.path.insert(0, os.path.join(_HERE, '..', 'methodology'))

from data_dccm import load_method_model, build_samples, METHOD_KEY
from data_dccm_attn import load_attn_model, build_samples_attn, ATTN_METHODS
from model_dccm import protein_dccm_corr
from unsup_dccm import feature_similarity as _feature_similarity
from train_dccm_attn_v2 import BACKBONE_DEPTH, backbone_depth


def feature_similarity(emb, sim='cosine', center=False):
    """Pairwise residue similarity, optionally with the anisotropy direction removed.

    center=True subtracts the mean residue embedding of THIS protein before the similarity,
    which is the standard correction for transformer-embedding anisotropy. See the
    --center_emb help text for why it matters for DCCM specifically.
    """
    e = np.asarray(emb, dtype=np.float64)
    if center:
        e = e - e.mean(axis=0, keepdims=True)          # mean over residues, not features
    return _feature_similarity(e, sim=sim)


def parse_args():
    p = argparse.ArgumentParser(
        description='Unsupervised DCCM benchmark with read-out matrix dumping.')
    p.add_argument('--data_path', required=True)
    p.add_argument('--analysis_path', required=True)
    p.add_argument('--methods', default='dplm+esm2+prostt5+seqdance',
                   help='Backbones: dplm, esm2, esmc, prostt5, seqdance, splm.')
    p.add_argument('--esmc_model', default='esmc_600m',
                   choices=['esmc_300m', 'esmc_600m'],
                   help='ESM-C size when esmc is in --methods (default: esmc_600m).')
    p.add_argument('--sim', default='cosine', choices=['cosine', 'pearson', 'dot'])
    p.add_argument('--center_emb', action='store_true',
                   help="Subtract the per-protein MEAN RESIDUE EMBEDDING before computing "
                        "the similarity. Transformer embeddings are anisotropic — they sit "
                        "in a narrow cone around a dominant common direction, so almost "
                        "every pairwise cosine comes out positive. Since the read-out is "
                        "attn * cos and attention is a non-negative softmax, an uncentred "
                        "ESM2-family backbone can barely express ANTI-correlated motion at "
                        "all, while the ground-truth DCCM is ~53%% negative. Removing the "
                        "common direction restores the negative half of the range. Note "
                        "this is NOT --sim pearson, which centres each residue vector "
                        "across the feature dim instead of across residues.")
    p.add_argument('--attn_readout', default='weighted', choices=['weighted', 'attn'])
    p.add_argument('--num_attn_layers', type=int, default=0,
                   help='Average attention over the last N layers (0 = all layers).')
    p.add_argument('--match_layers', action='store_true',
                   help='Cap --num_attn_layers to the shallowest backbone present, so every '
                        'method averages the same number of layers.')
    p.add_argument('--repr_layer', type=int, default=None)
    p.add_argument('--dplm_config', default=None)
    p.add_argument('--dplm_checkpoint', default=None)
    p.add_argument('--seqdance_path', default=None)
    p.add_argument('--splm_path', default=None)
    p.add_argument('--splm_config', default=None)
    p.add_argument('--splm_checkpoint', default=None)
    p.add_argument('--splm_python', default=None)
    p.add_argument('--splm_max_length', type=int, default=1022)
    p.add_argument('--splm_cache_pkl', default=None)
    p.add_argument('--dccm_dir', default=None)
    p.add_argument('--dccm_replicate', default=None, choices=['R1', 'R2', 'R3'])
    p.add_argument('--output_dir', required=True)
    p.add_argument('--max_proteins', type=int, default=None)
    p.add_argument('--save_mats', dest='save_mats', action='store_true', default=True)
    p.add_argument('--no_save_mats', dest='save_mats', action='store_false')
    p.add_argument('--max_saved_mats', type=int, default=400,
                   help='Cap on how many read-out matrices are written per method (the '
                        'training set has ~1800 proteins). Proteins are taken in sorted '
                        'pid order so the saved subset is the same for every method.')
    return p.parse_args()


def _device():
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def _empty_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def eval_attn_method(method, proteins, args, device, num_attn_layers):
    """Attention read-out. Returns ([{pid, mat, dccm}], realised_channel_count)."""
    model_bb, extra = load_attn_model(
        method, device, dplm_config=args.dplm_config,
        dplm_checkpoint=args.dplm_checkpoint, seqdance_path=args.seqdance_path,
        esmc_model=args.esmc_model)
    samples = build_samples_attn(
        method, proteins, model_bb, extra, args.analysis_path, device,
        repr_layer=args.repr_layer, num_attn_layers=num_attn_layers,
        head_reduce='mean', dccm_dir=args.dccm_dir, replicate=args.dccm_replicate)
    del model_bb
    _empty_cache()

    C = int(samples[0]['attn'].shape[0]) if samples else 0
    recs = []
    for s in samples:
        A = np.asarray(s['attn'], dtype=np.float64).mean(axis=0)      # [L, L]
        mat = A * feature_similarity(s["emb"], sim=args.sim, center=args.center_emb) \
            if args.attn_readout == 'weighted' else A
        recs.append({'pid': s['pid'], 'mat': mat, 'dccm': s['dccm']})
    return recs, C


def eval_emb_method(method, proteins, args, device):
    """Embedding-similarity read-out (splm). Returns ([{pid, mat, dccm}], 0)."""
    models_dict = load_method_model(
        method, device, seqdance_path=args.seqdance_path,
        dplm_config=args.dplm_config, dplm_checkpoint=args.dplm_checkpoint,
        proteins=proteins, splm_path=args.splm_path, splm_config=args.splm_config,
        splm_checkpoint=args.splm_checkpoint, splm_python=args.splm_python,
        splm_max_length=args.splm_max_length, splm_cache_pkl=args.splm_cache_pkl,
        esmc_model=args.esmc_model)
    samples = build_samples(proteins, models_dict, args.analysis_path, device,
                            dccm_dir=args.dccm_dir, replicate=args.dccm_replicate)
    del models_dict
    _empty_cache()
    return [{'pid': s['pid'], 'mat': feature_similarity(s['emb'], sim=args.sim, center=args.center_emb),
             'dccm': s['dccm']} for s in samples], 0


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = _device()
    print(f'Device: {device}')

    methods = [m.strip().lower() for m in re.split(r'[,+;\s]+', args.methods) if m.strip()]
    print(f'Methods: {methods}')

    num_attn_layers = args.num_attn_layers if args.num_attn_layers > 0 else None
    if args.match_layers and num_attn_layers is not None:
        depths = [backbone_depth(m, args.esmc_model)
                  for m in methods if m in BACKBONE_DEPTH]
        if depths:
            capped = min(num_attn_layers, min(depths))
            if capped != num_attn_layers:
                print(f'[match] capping --num_attn_layers {num_attn_layers} → {capped} '
                      f'(shallowest backbone present)')
            num_attn_layers = capped
    elif args.match_layers:
        raise SystemExit('--match_layers needs a positive --num_attn_layers.')

    from protein_level_emb_md import load_proteins
    proteins = load_proteins(args.data_path, args.analysis_path,
                             max_proteins=args.max_proteins)
    print(f'Eval proteins: {len(proteins)}')

    manifest = {'methods': methods, 'sim': args.sim, 'center_emb': bool(args.center_emb),
                'attn_readout': args.attn_readout,
                'num_attn_layers_requested': args.num_attn_layers,
                'num_attn_layers_used': num_attn_layers,
                'match_layers': args.match_layers,
                'dccm_replicate': args.dccm_replicate, 'data_path': args.data_path,
                'attn_channels_realised': {}, 'mean_pearson': {}}

    gt_saved = False
    for method in methods:
        disp = METHOD_KEY.get(method, method)
        kind = (('attn-weighted' if args.attn_readout == 'weighted' else 'attn')
                + (f'*{args.sim}' if args.attn_readout == 'weighted' else '')) \
            if method in ATTN_METHODS else f'emb-{args.sim}'
        print(f'\n================  {disp}  (read-out: {kind})  ================')

        if method in ATTN_METHODS:
            recs, C = eval_attn_method(method, proteins, args, device, num_attn_layers)
        else:
            recs, C = eval_emb_method(method, proteins, args, device)
        if not recs:
            print(f'[{disp}] no usable samples — skipping.')
            continue
        manifest['attn_channels_realised'][disp] = C

        results = {r['pid']: protein_dccm_corr(r['mat'], r['dccm']) for r in recs}
        csv_path = os.path.join(args.output_dir, f'unsup_attn_corr_{disp}.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['pid', 'pearson', 'spearman', 'readout'])
            for pid, (pe, sp) in results.items():
                w.writerow([pid, f'{pe:.6f}', f'{sp:.6f}', kind])
        print(f'[{disp}] per-protein correlations → {csv_path}')

        pears = [p for p, _ in results.values() if not np.isnan(p)]
        manifest['mean_pearson'][disp] = float(np.mean(pears))
        print(f'[{disp}] mean Pearson={np.mean(pears):.4f}  '
              f'median={np.median(pears):.4f}  n={len(pears)}  C={C}')

        by_pid = {r['pid']: r for r in recs}
        keep = sorted(by_pid)[:args.max_saved_mats]      # same subset for every method
        if not gt_saved:
            np.savez_compressed(
                os.path.join(args.output_dir, 'ground_truth.npz'),
                **{p: np.asarray(by_pid[p]['dccm'], dtype=np.float16) for p in keep})
            with open(os.path.join(args.output_dir, 'protein_lengths.csv'), 'w',
                      newline='') as f:
                w = csv.writer(f)
                w.writerow(['pid', 'length'])
                for r in recs:
                    w.writerow([r['pid'], r['dccm'].shape[0]])
            gt_saved = True

        if args.save_mats:
            np.savez_compressed(
                os.path.join(args.output_dir, f'readout_{disp}.npz'),
                **{p: np.asarray(by_pid[p]['mat'], dtype=np.float16) for p in keep})
            print(f'[{disp}] read-out matrices ({len(keep)}) → readout_{disp}.npz')

        del recs, by_pid
        _empty_cache()

    with open(os.path.join(args.output_dir, 'run_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print('\n=== Unsupervised DCCM summary ===')
    for m, v in sorted(manifest['mean_pearson'].items(), key=lambda kv: -kv[1]):
        print(f'  {m:<10} mean Pearson = {v:.4f}  (C={manifest["attn_channels_realised"][m]})')
    print('\nDone.')


if __name__ == '__main__':
    main()
