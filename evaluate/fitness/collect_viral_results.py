"""Consolidate the per-method ProteinGym viral wt-mt-RLA outputs into shareable tables.

`predict_fitness_viral.py` writes one CSV per (method, assay):

    <output_dir>/<Method>/<DMS_id>_predict.csv     columns: mutant, DMS_score,
                                                   DMS_score_bin, prediction

i.e. ground truth (`DMS_score`) and predicted score (`prediction`) for every mutation,
already. This script joins those into:

  1. `all_predictions.csv`  — ONE row per mutation, ground truth once, one prediction
                              column per method:
                              DMS_id, mutant, DMS_score, DMS_score_bin,
                              pred_DPLM, pred_ESM2, pred_ProstT5, pred_SeqDance
  2. `per_assay_spearman.csv` — per-assay Spearman for every method (the headline table),
                              plus n and the assay's WT length.
  3. `method_summary.csv`   — mean / median Spearman per method across the 23 assays.
  4. `paired_tests.csv`     — Wilcoxon signed-rank on per-assay Spearman for every method
                              pair (n=23), Holm-corrected. This is the test that says
                              whether one method really beats another across assays.

    python evaluate/fitness/collect_viral_results.py \
        --results_dir ./results/proteingym_viral \
        --manifest /path/to/DPLM_data/proteingym/viral23_manifest.csv
"""
import argparse
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results_dir', required=True,
                   help='Dir containing one subdir per method (DPLM/ESM2/ProstT5/SeqDance).')
    p.add_argument('--manifest', required=True, help='viral23_manifest.csv')
    p.add_argument('--out_dir', default=None, help='Default: --results_dir')
    args = p.parse_args()
    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    lens = dict(zip(manifest.DMS_id, manifest.target_seq.str.len()))
    methods = sorted(d for d in os.listdir(args.results_dir)
                     if os.path.isdir(os.path.join(args.results_dir, d))
                     and any(f.endswith('_predict.csv')
                             for f in os.listdir(os.path.join(args.results_dir, d))))
    if not methods:
        raise SystemExit(f'No method subdirs with *_predict.csv under {args.results_dir}')
    print(f'[collect] methods found: {methods}')

    merged, rho_rows = [], []
    for dms_id in manifest.DMS_id:
        base = None
        for method in methods:
            f = os.path.join(args.results_dir, method, f'{dms_id}_predict.csv')
            if not os.path.exists(f):
                continue
            df = pd.read_csv(f)
            keep = df[['mutant', 'DMS_score', 'prediction']].rename(
                columns={'prediction': f'pred_{method}'})
            if 'DMS_score_bin' in df.columns and base is None:
                keep.insert(2, 'DMS_score_bin', df['DMS_score_bin'])
            base = keep if base is None else base.merge(
                keep[['mutant', f'pred_{method}']], on='mutant', how='outer')

            v = df.dropna(subset=['DMS_score', 'prediction'])
            rho_rows.append(dict(DMS_id=dms_id, method=method,
                                 spearman=spearmanr(v.DMS_score, v.prediction).statistic
                                 if len(v) > 2 else np.nan,
                                 n=len(v), seq_len=lens.get(dms_id)))
        if base is not None:
            base.insert(0, 'DMS_id', dms_id)
            merged.append(base)

    if not merged:
        raise SystemExit('No per-assay prediction files found.')

    all_pred = pd.concat(merged, ignore_index=True)
    all_pred.to_csv(os.path.join(out_dir, 'all_predictions.csv'), index=False)
    print(f'[collect] all_predictions.csv — {len(all_pred)} mutations, '
          f'columns: {list(all_pred.columns)}')

    rho = pd.DataFrame(rho_rows)
    wide = rho.pivot(index='DMS_id', columns='method', values='spearman')
    wide = wide.join(rho.pivot(index='DMS_id', columns='method', values='n')
                     .add_prefix('n_')).reset_index()
    wide.to_csv(os.path.join(out_dir, 'per_assay_spearman.csv'), index=False)

    summary = (rho.groupby('method').spearman
               .agg(mean_spearman='mean', median_spearman='median',
                    n_assays=lambda s: s.notna().sum())
               .sort_values('mean_spearman', ascending=False).reset_index())
    summary.to_csv(os.path.join(out_dir, 'method_summary.csv'), index=False)
    print('\n[collect] method summary (mean Spearman across assays):')
    print(summary.to_string(index=False))

    # Paired Wilcoxon on per-assay Spearman, Holm-corrected over all pairs.
    piv = rho.pivot(index='DMS_id', columns='method', values='spearman')
    pairs = []
    for a, b in combinations(sorted(piv.columns), 2):
        d = (piv[a] - piv[b]).dropna()
        if len(d) < 3:
            continue
        pairs.append(dict(method_a=a, method_b=b, n=len(d), delta=d.mean(),
                          wins_a=float((d > 0).mean()), p_raw=wilcoxon(d).pvalue))
    if pairs:
        pt = pd.DataFrame(pairs).sort_values('p_raw').reset_index(drop=True)
        m = len(pt)
        pt['p_holm'] = np.minimum.accumulate(
            (pt.p_raw * (m - np.arange(m))).clip(upper=1.0)[::-1])[::-1]
        pt.to_csv(os.path.join(out_dir, 'paired_tests.csv'), index=False)
        print('\n[collect] paired Wilcoxon on per-assay Spearman (Holm-corrected):')
        print(pt.to_string(index=False))

    print(f'\n[collect] wrote 4 tables to {out_dir}')


if __name__ == '__main__':
    main()
