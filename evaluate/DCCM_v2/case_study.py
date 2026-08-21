"""
case_study.py — multi-method DCCM case panels, with an explicit, recorded selection rule.

A case-study figure is an illustration, not evidence: whichever proteins you show, the
aggregate table is what supports the claim. What makes an illustration honest is that the
rule used to pick it is stated and reproducible, and that the reader can see where the rule
sits in the full distribution. So this script:

  * picks proteins by a named --criterion instead of by eye,
  * writes cases_manifest.json with the rule, the candidate pool, every selected protein's
    per-method r, and that protein's PERCENTILE in the Δ distribution,
  * prints a caption line, ready to paste, that states the rule and the aggregate result,
  * can also emit the losses (--criterion ref_worst), which is what you show a reviewer who
    asks "where does your method fail?".

Criteria
    representative  Δ (ref − best baseline) closest to the MEDIAN Δ. The honest default:
                    these panels look like the typical protein, not the best one.
    ref_best        largest Δ — the clearest wins. Legitimate as an illustration ONLY when
                    the caption says so; the manifest records the percentile to keep you
                    honest.
    ref_worst       most negative Δ — the failure cases.
    named           an explicit --pids list.

Runs offline from the .npz dumps written by train_dccm_attn_v2.py / unsup_dccm_attn_v2.py,
so no GPU and no model loading.

Example
    python evaluate/DCCM_v2/case_study.py \
        --run_dir results/dccm_v2_sup_matched \
        --output_dir results/dccm_v2_analysis/sup_matched/cases \
        --criterion representative --n_cases 3
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dccm_stats import paired_frame
from dccm_post import POSTPROCESS
import plot_dccm_v2 as P


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    da, db = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    return float((a * b).sum() / (da * db)) if da > 0 and db > 0 else float('nan')

_MAT_PATTERNS = [re.compile(r'preds_(?P<m>.+)\.npz$'),
                 re.compile(r'readout_(?P<m>.+)\.npz$')]
_CSV_PATTERNS = [re.compile(r'per_protein_corr_attn_(?P<m>.+)\.csv$'),
                 re.compile(r'unsup_attn_corr_(?P<m>.+)\.csv$')]


def _discover(run_dir, patterns, ext):
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir, f'*.{ext}'))):
        base = os.path.basename(path)
        for pat in patterns:
            mo = pat.match(base)
            if mo:
                out.setdefault(mo.group('m'), path)
                break
    return out


def parse_args():
    p = argparse.ArgumentParser(description='Multi-method DCCM case panels.')
    p.add_argument('--run_dir', required=True,
                   help='Directory with preds_*.npz / readout_*.npz, ground_truth.npz and '
                        'the per-protein correlation CSVs.')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--ref', default='DPLM')
    p.add_argument('--metric', default='pearson', choices=['pearson', 'spearman'])
    p.add_argument('--criterion', default='representative',
                   choices=['representative', 'ref_best', 'ref_worst', 'named'])
    p.add_argument('--n_cases', type=int, default=3)
    p.add_argument('--pids', nargs='*', default=None,
                   help='Explicit protein IDs (with --criterion named).')
    p.add_argument('--min_length', type=int, default=0,
                   help='Skip proteins shorter than this (tiny maps read poorly).')
    p.add_argument('--max_length', type=int, default=10 ** 6)
    p.add_argument('--display', default='zscore', choices=['zscore', 'shared'],
                   help="'zscore' (default) standardises each predicted panel over its own "
                        "off-diagonal entries — an affine transform, so the reported "
                        "Pearson r is unchanged, but diagonal-dominated unsupervised "
                        "read-outs become readable. 'shared' keeps raw magnitudes.")
    p.add_argument('--clip', type=float, default=3.0,
                   help='Colour-scale limit in SD units when --display zscore.')
    p.add_argument('--postprocess', default='none',
                   choices=['none', 'dcentre', 'dcentre_norm'],
                   help="Post-hoc correction applied to each predicted matrix BEFORE "
                        "plotting and before the panel's r is recomputed. 'dcentre_norm' "
                        "double-centres and rescales to correlation form, which lets an "
                        "all-positive attention read-out express anti-correlation — see "
                        "dccm_post.py. Unlike --display, this CHANGES the matrix, so the "
                        "panel r is recomputed from the corrected map and the figure says "
                        "so. Aggregate numbers must come from dccm_post.py, not from here.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    mats = _discover(args.run_dir, _MAT_PATTERNS, 'npz')
    csvs = _discover(args.run_dir, _CSV_PATTERNS, 'csv')
    gt_path = os.path.join(args.run_dir, 'ground_truth.npz')
    if not mats:
        raise SystemExit(f'No preds_*.npz / readout_*.npz in {args.run_dir}. Re-run with '
                         f'train_dccm_attn_v2.py or unsup_dccm_attn_v2.py, which save them.')
    if not os.path.exists(gt_path):
        raise SystemExit(f'Missing {gt_path}')
    if args.ref not in mats:
        raise SystemExit(f'Reference {args.ref!r} not among {sorted(mats)}')

    print(f'Methods with matrices: {sorted(mats)}')
    gt = np.load(gt_path)
    loaded = {m: np.load(p) for m, p in mats.items()}

    # Supervised runs average the reported r over seeds but dump only one seed's matrices.
    # Say so on the panel rather than letting the picture and the number silently disagree.
    seed_note = ''
    man_path = os.path.join(args.run_dir, 'run_manifest.json')
    if os.path.exists(man_path):
        with open(man_path) as f:
            man = json.load(f)
        seeds = man.get('seeds') or []
        preds_seed = man.get('preds_seed')
        if preds_seed is not None and len(seeds) > 1:
            seed_note = (f'; maps shown are from seed {preds_seed}, '
                         f'r is averaged over seeds {seeds}')

    # per-protein scores, paired across methods
    common_csv = {m: p for m, p in csvs.items() if m in mats}
    pids_all, data, _ = paired_frame(common_csv, metric=args.metric)
    others = [m for m in data if m != args.ref]

    # candidate pool: proteins that have a saved matrix for EVERY method and a length in range
    pool, delta = [], []
    for i, pid in enumerate(pids_all):
        if pid not in gt.files or any(pid not in loaded[m].files for m in loaded):
            continue
        L = gt[pid].shape[0]
        if not (args.min_length <= L <= args.max_length):
            continue
        d = data[args.ref][i] - max(data[m][i] for m in others)   # vs BEST baseline
        pool.append(pid)
        delta.append(d)
    if not pool:
        raise SystemExit('No protein has a saved matrix for every method in the length '
                         'range requested.')
    delta = np.array(delta)
    print(f'Candidate pool: {len(pool)} proteins   '
          f'Δ(ref − best baseline): mean={delta.mean():+.4f} median={np.median(delta):+.4f}')

    if args.criterion == 'named':
        if not args.pids:
            raise SystemExit('--criterion named requires --pids')
        chosen = [p for p in args.pids if p in pool]
        missing = [p for p in args.pids if p not in pool]
        if missing:
            print(f'[warn] not in the candidate pool, skipped: {missing}')
    elif args.criterion == 'ref_best':
        chosen = [pool[i] for i in np.argsort(-delta)[:args.n_cases]]
    elif args.criterion == 'ref_worst':
        chosen = [pool[i] for i in np.argsort(delta)[:args.n_cases]]
    else:                                       # representative
        med = np.median(delta)
        chosen = [pool[i] for i in np.argsort(np.abs(delta - med))[:args.n_cases]]

    idx_of = {pid: i for i, pid in enumerate(pids_all)}
    pool_idx = {pid: i for i, pid in enumerate(pool)}
    notes = {'representative': 'selected as the protein(s) whose margin over the best '
                               'baseline is closest to the dataset median margin',
             'ref_best': f'selected as the largest margin for {args.ref} — an illustration, '
                         f'not a typical case',
             'ref_worst': f'selected as the largest margin AGAINST {args.ref} — failure case',
             'named': 'selected by explicit protein ID'}

    manifest = {'run_dir': args.run_dir, 'criterion': args.criterion,
                'metric': args.metric, 'reference': args.ref,
                'candidate_pool_size': len(pool),
                'delta_vs_best_baseline': {'mean': float(delta.mean()),
                                           'median': float(np.median(delta))},
                'cases': []}

    for pid in chosen:
        i = idx_of[pid]
        per_r = {m: float(data[m][i]) for m in data}
        d = float(delta[pool_idx[pid]])
        pct = float((delta < d).mean() * 100)
        post_note = ('' if args.postprocess == 'none' else
                     f'; maps double-centred ({args.postprocess}) — r recomputed on the '
                     f'corrected map')
        note = (f'{notes[args.criterion]}; Δ vs. best baseline = {d:+.3f} '
                f'({pct:.0f}th percentile of {len(pool)} proteins){seed_note}{post_note}')
        preds = {m: np.asarray(loaded[m][pid], dtype=float) for m in loaded}
        if args.postprocess != 'none':
            # The correction changes the matrix, so the r printed on the panel must be
            # recomputed from the corrected map — otherwise the picture and the number
            # would describe different objects.
            preds = {m: POSTPROCESS[args.postprocess](preds[m]) for m in preds}
            g = np.asarray(gt[pid], dtype=float)
            iu_p = np.triu_indices(g.shape[0], k=1)
            gv = g[iu_p]
            per_r = {m: _pearson(preds[m][iu_p], gv) for m in preds}
        P.plot_case_panel(pid, np.asarray(gt[pid], dtype=float), preds, args.output_dir,
                          ref=args.ref, per_method_r=per_r,
                          fname=f'case_{args.criterion}_{pid}.png', selection_note=note,
                          display=args.display, clip=args.clip)
        manifest['cases'].append({'pid': pid, 'length': int(gt[pid].shape[0]),
                                  'per_method_r': per_r, 'delta_vs_best_baseline': d,
                                  'percentile_in_pool': pct})
        print(f'  {pid}: ' + '  '.join(f'{m}={per_r[m]:.3f}' for m in sorted(per_r))
              + f'   Δ={d:+.3f} ({pct:.0f}th pct)')

    with open(os.path.join(args.output_dir, 'cases_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    ref_mean = data[args.ref].mean()
    best_other = max(others, key=lambda m: data[m].mean())
    caption = (f'DCCM case studies ({args.criterion} selection). Panels show the MD '
               f'ground-truth DCCM and each model\'s prediction on a shared colour scale. '
               f'Proteins were chosen by the stated rule over {len(pool)} candidates, not '
               f'by inspection. Across the full test set (n={data[args.ref].size}), '
               f'{args.ref} reaches mean {args.metric} r = {ref_mean:.3f} vs. '
               f'{data[best_other].mean():.3f} for the strongest baseline ({best_other}).')
    with open(os.path.join(args.output_dir, 'caption.txt'), 'w') as f:
        f.write(caption + '\n')
    print('\nCaption:\n' + caption)
    print('\nDone.')


if __name__ == '__main__':
    main()
