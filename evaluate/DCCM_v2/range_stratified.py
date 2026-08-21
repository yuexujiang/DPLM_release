"""
range_stratified.py — is the DPLM advantage actually LONG-RANGE?

`length_trend.py` shows the margin grows with chain length, which is *consistent* with a
long-range explanation but does not prove one: longer proteins differ in many ways besides
having more distant residue pairs.

This script tests the claim directly. Every DCCM is a matrix over residue pairs (i, j), and
each pair has a sequence separation |i − j|. So instead of correlating the whole upper
triangle at once, recompute the per-protein correlation **within separation bands**:

    short   |i-j| in [1, 6)      local backbone / helix turn
    medium  |i-j| in [6, 12)     secondary-structure scale
    long    |i-j| in [12, 24)
    xlong   |i-j| >= 24          tertiary contacts, domain-domain coupling

If DPLM's advantage comes from modelling long-range coupling, its margin over the baselines
should be small or absent in the short band and largest in the xlong band. If instead the
margin is flat across bands, the "long-range" story is wrong and the length trend has some
other cause — which is a result worth knowing before it appears in a paper.

This is a strictly stronger test than the length trend because it is computed against the MD
ground truth on the very pairs in question, rather than inferred from a protein-level summary.

Runs offline from the saved .npz matrices — no GPU.

Example
    python evaluate/DCCM_v2/range_stratified.py \
        --run_dir results/dccm_v2_unsup_testset \
        --output_dir results/dccm_v2_analysis/unsup_v2_testset
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dccm_stats import wilcoxon_signed, bootstrap_ci, holm

_MAT_PATTERNS = [re.compile(r'preds_(?P<m>.+)\.npz$'),
                 re.compile(r'readout_(?P<m>.+)\.npz$')]

BANDS = [('short  |i-j| 1-5',    1,   6),
         ('medium |i-j| 6-11',   6,  12),
         ('long   |i-j| 12-23', 12,  24),
         ('xlong  |i-j| >=24',  24, 10 ** 9)]


def _discover(run_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir, '*.npz'))):
        base = os.path.basename(path)
        for pat in _MAT_PATTERNS:
            mo = pat.match(base)
            if mo:
                out.setdefault(mo.group('m'), path)
                break
    return out


def _pearson(a, b):
    if a.size < 3:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    da, db = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    if da == 0 or db == 0:
        return np.nan
    return float((a * b).sum() / (da * db))


def parse_args():
    p = argparse.ArgumentParser(description='Sequence-separation-stratified DCCM accuracy.')
    p.add_argument('--run_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--ref', default='DPLM')
    p.add_argument('--min_pairs', type=int, default=50,
                   help='Skip a (protein, band) cell with fewer pairs than this.')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    mats = _discover(args.run_dir)
    gt_path = os.path.join(args.run_dir, 'ground_truth.npz')
    if not mats or not os.path.exists(gt_path):
        raise SystemExit(f'Need ground_truth.npz and preds_*/readout_*.npz in {args.run_dir}')
    if args.ref not in mats:
        raise SystemExit(f'Reference {args.ref!r} not among {sorted(mats)}')

    gt = np.load(gt_path)
    loaded = {m: np.load(p) for m, p in mats.items()}
    methods = [args.ref] + sorted(m for m in loaded if m != args.ref)
    pids = [p for p in sorted(gt.files) if all(p in loaded[m].files for m in loaded)]
    print(f'Methods: {methods}\nProteins with a matrix for every method: {len(pids)}')

    # per-band, per-protein correlation for every method
    scores = {b[0]: {m: [] for m in methods} for b in BANDS}
    kept = {b[0]: [] for b in BANDS}
    for pid in pids:
        g = np.asarray(gt[pid], dtype=np.float64)
        L = g.shape[0]
        ii, jj = np.triu_indices(L, k=1)
        sep = jj - ii
        gv = g[ii, jj]
        preds = {m: np.asarray(loaded[m][pid], dtype=np.float64)[ii, jj] for m in methods}
        for name, lo, hi in BANDS:
            sel = (sep >= lo) & (sep < hi)
            if sel.sum() < args.min_pairs:
                continue
            vals = {m: _pearson(preds[m][sel], gv[sel]) for m in methods}
            if any(np.isnan(v) for v in vals.values()):
                continue
            for m in methods:
                scores[name][m].append(vals[m])
            kept[name].append(pid)

    results = {'run_dir': args.run_dir, 'reference': args.ref, 'bands': {}}
    lines = [f'# Separation-stratified DCCM accuracy — {os.path.basename(args.run_dir)}',
             '',
             'Per-protein Pearson r computed **within** each sequence-separation band, then',
             'compared to the reference by paired Wilcoxon signed-rank across proteins.',
             'A long-range advantage should show up as a margin that grows from the short',
             'band to the xlong band.',
             '']

    for name, _, _ in BANDS:
        n = len(kept[name])
        if n == 0:
            print(f'\n[{name}] no protein had enough pairs — skipped')
            continue
        arr = {m: np.array(scores[name][m], dtype=float) for m in methods}
        print(f'\n=== {name}   (n={n} proteins) ===')
        for m in methods:
            print(f'  {m:<10} mean r = {arr[m].mean():.4f}')

        others = [m for m in methods if m != args.ref]
        deltas = {m: arr[args.ref] - arr[m] for m in others}
        pvals = [wilcoxon_signed(arr[args.ref], arr[m]) for m in others]
        adj = holm(pvals)

        lines += [f'### {name}  (n={n})', '',
                  '| Method | mean r | Δ (DPLM − method) | 95% CI | DPLM wins | p (Holm) |',
                  '|---|---|---|---|---|---|',
                  f'| **{args.ref}** | **{arr[args.ref].mean():.4f}** | — | — | — | — |']
        results['bands'][name] = {'n_proteins': n,
                                  'mean_r': {m: float(arr[m].mean()) for m in methods},
                                  'vs_ref': {}}
        for m, p, pa in zip(others, pvals, adj):
            d = deltas[m]
            lo, hi = bootstrap_ci(d)
            win = float((d > 0).mean())
            star = '***' if pa < 1e-3 else '**' if pa < 1e-2 else '*' if pa < 0.05 else 'ns'
            print(f'    {args.ref} − {m:<9} Δ={d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  '
                  f'win={100*win:.1f}%  p_holm={pa:.2e} {star}')
            lines.append(f'| {m} | {arr[m].mean():.4f} | {d.mean():+.4f} | '
                         f'[{lo:+.4f}, {hi:+.4f}] | {100*win:.1f}% | {pa:.1e} {star} |')
            results['bands'][name]['vs_ref'][m] = {
                'delta': float(d.mean()), 'ci': [lo, hi], 'win_rate': win,
                'p_holm': float(pa)}
        lines.append('')

    # the headline: does the margin grow from short to xlong?
    lines += ['## Margin by band (the actual test)', '',
              '| Baseline | ' + ' | '.join(b[0].split()[0] for b in BANDS) + ' |',
              '|---' * (len(BANDS) + 1) + '|']
    print('\n=== Margin (DPLM − baseline) across bands ===')
    for m in sorted(x for x in methods if x != args.ref):
        row, cells = [], []
        for name, _, _ in BANDS:
            if name in results['bands'] and m in results['bands'][name]['vs_ref']:
                v = results['bands'][name]['vs_ref'][m]['delta']
                row.append(f'{v:+.4f}')
                cells.append(v)
            else:
                row.append('—')
        print(f'  {m:<10} ' + '  '.join(f'{c:+.4f}' for c in cells))
        lines.append(f'| {m} | ' + ' | '.join(row) + ' |')

    md = os.path.join(args.output_dir, 'range_stratified.md')
    with open(md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(args.output_dir, 'range_stratified.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nWrote {md}')


if __name__ == '__main__':
    main()
