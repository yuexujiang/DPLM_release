"""
length_trend.py — does the reference model's margin depend on chain length?

The aggregate DCCM tables answer "is DPLM better on average". They do not answer the
question a reviewer actually asks, which is "better *where*, and is there a reason?".

DCCM measures correlated motion between residue pairs. Long chains have more long-range
pairs, so if a backbone's advantage comes from modelling long-range coupling rather than
local neighbourhood structure, the per-protein margin should grow with length. That is a
falsifiable prediction, and this script tests it:

  * Spearman correlation between chain length and the paired margin Δ = r_ref − r_baseline
    (Spearman, not Pearson, because the length distribution is heavily right-skewed),
  * the margin in the shortest vs longest length quintile, so the effect has units,
  * a permutation p-value as a distribution-free cross-check on the rank correlation.

A positive, significant ρ means the margin genuinely widens on longer chains. A ρ near zero
means the advantage is uniform — still a fine result, just a different claim. Report whichever
one comes out; the point is that the claim is tested rather than eyeballed off a bar chart.

Example
    python evaluate/DCCM_v2/length_trend.py \
        --run_dir results/dccm_v2_unsup_full \
        --output_dir results/dccm_v2_analysis/unsup_v2_full
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dccm_stats import paired_frame

_CSV_PATTERNS = [re.compile(r'per_protein_corr_attn_(?P<m>.+)\.csv$'),
                 re.compile(r'unsup_attn_corr_(?P<m>.+)\.csv$')]


def _discover(run_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir, '*.csv'))):
        base = os.path.basename(path)
        for pat in _CSV_PATTERNS:
            mo = pat.match(base)
            if mo:
                out.setdefault(mo.group('m'), path)
                break
    return out


def _spearman(x, y):
    """Spearman ρ via Pearson on ranks (average ranks for ties)."""
    def rank(v):
        order = np.argsort(v, kind='mergesort')
        r = np.empty(len(v), dtype=float)
        sv = v[order]
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    return float(np.corrcoef(rx, ry)[0, 1])


def _perm_p(x, y, rho, n_perm=10000, seed=0):
    """Two-sided permutation p-value for the rank correlation."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    count = 0
    for _ in range(n_perm):
        if abs(_spearman(x, rng.permutation(y))) >= abs(rho):
            count += 1
    return (count + 1) / (n_perm + 1)


def parse_args():
    p = argparse.ArgumentParser(description='Length dependence of the paired DCCM margin.')
    p.add_argument('--run_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--lengths_csv', default=None,
                   help='Defaults to <run_dir>/protein_lengths.csv')
    p.add_argument('--ref', default='DPLM')
    p.add_argument('--n_perm', type=int, default=2000,
                   help='Permutations for the distribution-free check (0 to skip).')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    csvs = _discover(args.run_dir)
    if args.ref not in csvs:
        raise SystemExit(f'Reference {args.ref!r} not among {sorted(csvs)}')
    lengths_csv = args.lengths_csv or os.path.join(args.run_dir, 'protein_lengths.csv')
    if not os.path.exists(lengths_csv):
        raise SystemExit(f'Missing {lengths_csv} — needed to stratify by length.')
    L = {r['pid']: int(r['length']) for r in csv.DictReader(open(lengths_csv))}

    results = {'run_dir': args.run_dir, 'reference': args.ref, 'metrics': {}}
    lines = [f'# Length dependence of the {args.ref} margin — {os.path.basename(args.run_dir)}',
             '',
             'Δ = r_ref − r_baseline, per protein. ρ is the Spearman correlation between',
             'chain length and Δ. A positive ρ means the margin widens on longer chains,',
             'which is what you expect if the advantage comes from long-range coupling.',
             '']

    for metric in ['pearson', 'spearman']:
        pids, data, _ = paired_frame(csvs, metric=metric)
        lens = np.array([L[p] for p in pids], dtype=float)
        q_lo, q_hi = np.quantile(lens, [0.2, 0.8])
        others = [m for m in data if m != args.ref]

        lines += [f'### {metric.capitalize()} r  (n={len(pids)})', '',
                  '| Baseline | ρ(length, Δ) | p (perm) | Δ shortest quintile | '
                  'Δ longest quintile |', '|---|---|---|---|---|']
        results['metrics'][metric] = {'n': len(pids), 'baselines': {}}

        print(f'\n=== {metric}  (n={len(pids)}) ===')
        for m in sorted(others):
            d = data[args.ref] - data[m]
            rho = _spearman(lens, d)
            p = _perm_p(lens, d, rho, n_perm=args.n_perm) if args.n_perm else float('nan')
            lo = float(d[lens <= q_lo].mean())
            hi = float(d[lens >= q_hi].mean())
            print(f'  {args.ref}-{m:<9} rho={rho:+.3f}  p_perm={p:.2e}  '
                  f'short Δ={lo:+.4f} -> long Δ={hi:+.4f}')
            lines.append(f'| {m} | {rho:+.3f} | {p:.1e} | {lo:+.4f} | {hi:+.4f} |')
            results['metrics'][metric]['baselines'][m] = {
                'spearman_rho_length_vs_delta': rho, 'p_permutation': p,
                'delta_shortest_quintile': lo, 'delta_longest_quintile': hi,
                'length_quintile_cuts': [float(q_lo), float(q_hi)]}
        lines.append('')

    md = os.path.join(args.output_dir, 'length_trend.md')
    with open(md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(args.output_dir, 'length_trend.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nWrote {md}')


if __name__ == '__main__':
    main()
