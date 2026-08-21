"""
analyze_dccm.py — paired re-analysis of DCCM benchmark runs.

Reads the per-protein correlation CSVs written by the DCCM scripts (v1 or v2), aligns every
method onto the common protein set, and produces:

  * summary_{setting}.csv    — mean/median/sd per method plus the paired comparison to the
                               reference method (delta, bootstrap CI, win rate, Wilcoxon p,
                               Holm-adjusted p)
  * summary_{setting}.md     — the same as a markdown table
  * fig1_paired_box          — boxplot with PAIRED significance brackets
  * fig2_paired_scatter      — per-protein scatter vs. each baseline
  * fig3_delta_hist          — distribution of per-protein differences
  * fig4_stratified          — paired delta binned by protein length (needs --lengths_csv)

It reads only CSVs, so it runs on a login node — no GPU, no model loading.

Examples
--------
  # unsupervised run (all methods in one directory)
  python evaluate/DCCM_v2/analyze_dccm.py \
      --run_dirs results/dccm_output_unsup_attwt_trainset_all4_attnum0 \
      --setting unsup_v1 --output_dir results/dccm_v2_analysis/unsup_v1

  # supervised run that was split over two directories
  python evaluate/DCCM_v2/analyze_dccm.py \
      --run_dirs results/dccm_output_attwt_loss_dplm results/dccm_output_attwt_loss_other3 \
      --setting sup_v1 --output_dir results/dccm_v2_analysis/sup_v1
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

from dccm_stats import compare_to_ref, format_table, paired_frame, write_table_csv
import plot_dccm_v2 as P


# CSV name → method, for every layout the DCCM scripts produce.
_PATTERNS = [
    re.compile(r'per_protein_corr_attn_(?P<m>.+)\.csv$'),   # supervised, attention predictor
    re.compile(r'per_protein_corr_(?P<m>.+)\.csv$'),        # supervised, embedding predictor
    re.compile(r'unsup_attn_corr_(?P<m>.+)\.csv$'),         # unsupervised, attention read-out
    re.compile(r'unsup_corr_(?P<m>.+)\.csv$'),              # unsupervised, embedding read-out
]


def discover(run_dirs):
    """{method: csv_path} discovered across one or more run directories."""
    found = {}
    for d in run_dirs:
        for path in sorted(glob.glob(os.path.join(d, '*.csv'))):
            base = os.path.basename(path)
            for pat in _PATTERNS:
                mo = pat.match(base)
                if mo:
                    method = mo.group('m')
                    if method in found:
                        print(f'[warn] {method} found twice; keeping {found[method]}, '
                              f'ignoring {path}')
                    else:
                        found[method] = path
                    break
    return found


def read_lengths(path):
    """CSV with columns pid,length → {pid: int}."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                out[row['pid']] = float(row['length'])
            except (KeyError, ValueError):
                continue
    return out


def parse_args():
    p = argparse.ArgumentParser(description='Paired re-analysis of DCCM benchmark runs.')
    p.add_argument('--run_dirs', nargs='+', required=True,
                   help='Directories holding per-protein correlation CSVs.')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--setting', default='run',
                   help='Label for this comparison, used in filenames and titles.')
    p.add_argument('--ref', default='DPLM', help='Reference method (default: DPLM).')
    p.add_argument('--metrics', nargs='+', default=['pearson', 'spearman'])
    p.add_argument('--methods', nargs='*', default=None,
                   help='Restrict to these methods (default: all discovered).')
    p.add_argument('--lengths_csv', default=None,
                   help='Optional CSV (pid,length) to enable the stratification panel.')
    p.add_argument('--n_boot', type=int, default=10000)
    p.add_argument('--title_prefix', default='')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    found = discover(args.run_dirs)
    if args.methods:
        keep = {m.lower() for m in args.methods}
        found = {m: p for m, p in found.items() if m.lower() in keep}
    if not found:
        raise SystemExit(f'No correlation CSVs found in {args.run_dirs}')
    if args.ref not in found:
        raise SystemExit(f'Reference {args.ref!r} not among discovered methods '
                         f'{sorted(found)}')

    print(f'Discovered {len(found)} methods:')
    for m, path in sorted(found.items()):
        print(f'  {m:<10} {path}')

    lengths_map = read_lengths(args.lengths_csv) if args.lengths_csv else None
    manifest = {'setting': args.setting, 'reference': args.ref,
                'run_dirs': args.run_dirs, 'sources': found, 'metrics': {}}

    md_parts = [f'# DCCM benchmark — {args.setting}', '',
                'Paired re-analysis. All methods are scored on the same proteins, so the '
                'comparison uses the Wilcoxon **signed-rank** test on per-protein '
                'differences (the v1 code used the unpaired Mann-Whitney U test), and every '
                'method is restricted to the protein set that **all** methods scored.', '']

    for metric in args.metrics:
        pids, data, dropped = paired_frame(found, metric=metric)
        n_common = len(pids)
        print(f'\n=== {metric}: paired n={n_common} ===')
        for m, k in sorted(dropped.items()):
            if k:
                print(f'  [note] {m}: {k} protein(s) dropped to reach the common set')

        summary, rows = compare_to_ref(data, ref=args.ref, n_boot=args.n_boot)
        for m in sorted(summary):
            s = summary[m]
            print(f'  {m:<10} mean={s["mean"]:.4f}  median={s["median"]:.4f}  n={s["n"]}')
        for r in sorted(rows, key=lambda x: -x['delta']):
            print(f'    {args.ref} − {r["method"]:<9} Δ={r["delta"]:+.4f} '
                  f'[{r["ci_lo"]:+.4f},{r["ci_hi"]:+.4f}]  '
                  f'win={100*r["win_rate"]:.1f}%  p_holm={r["p_holm"]:.2e} {r["stars"]}')

        metric_name = {'pearson': 'Pearson r', 'spearman': 'Spearman r'}.get(metric, metric)
        write_table_csv(os.path.join(args.output_dir, f'summary_{args.setting}_{metric}.csv'),
                        summary, rows, ref=args.ref, metric=metric, setting=args.setting)
        md_parts.append(format_table(summary, rows, ref=args.ref, metric=metric_name,
                                     title=f'{metric_name} — {args.setting}'))
        md_parts.append('')

        manifest['metrics'][metric] = {
            'n_paired': n_common, 'dropped': dropped,
            'summary': summary,
            'comparisons': rows,
        }

        tp = (args.title_prefix + ' ') if args.title_prefix else ''
        suffix = f'_{args.setting}_{metric}.png'
        P.plot_paired_box(data, args.output_dir, ref=args.ref, metric_name=metric_name,
                          fname=f'fig1_paired_box{suffix}',
                          title=f'{tp}DCCM — paired on n={n_common} proteins '
                                f'(signed-rank vs. {args.ref})')
        P.plot_paired_scatter(data, args.output_dir, ref=args.ref, metric_name=metric_name,
                              fname=f'fig2_paired_scatter{suffix}',
                              title=f'{tp}Per-protein paired comparison (n={n_common})')
        P.plot_delta_hist(data, args.output_dir, ref=args.ref, metric_name=metric_name,
                          fname=f'fig3_delta_hist{suffix}',
                          title=f'{tp}Paired per-protein differences (n={n_common})')
        if lengths_map:
            lengths = np.array([lengths_map.get(p, np.nan) for p in pids], dtype=float)
            P.plot_stratified(data, lengths, args.output_dir, ref=args.ref,
                              metric_name=metric_name,
                              fname=f'fig4_stratified{suffix}',
                              title=f'{tp}Where {args.ref} wins — Δ by protein length')

    md_path = os.path.join(args.output_dir, f'summary_{args.setting}.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_parts) + '\n')
    with open(os.path.join(args.output_dir, f'manifest_{args.setting}.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nWrote {md_path}')
    print('Done.')


if __name__ == '__main__':
    main()
