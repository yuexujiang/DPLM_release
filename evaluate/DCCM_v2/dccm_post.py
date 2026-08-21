"""
dccm_post.py — post-hoc correction that lets an all-positive read-out express ANTI-correlation.

The unsupervised read-out is `mat = A * cos(emb)` with `A` a softmax attention map, so
`A >= 0` and the sign of every entry is the sign of the embedding cosine. ESM2-family
embeddings are anisotropic (a narrow cone about a dominant common direction), so essentially
every pairwise cosine is positive and DPLM/ESM2/SeqDance emit maps with **no negative entries
at all** — while the true DCCM is ~52% negative. See README "Why DPLM's maps look all red".

Subtracting the common direction from the *embeddings* before the cosine was tried and made
things much worse (job 20814695: DPLM 0.518 -> 0.367), because that direction carries real
signal. This module instead corrects the *output* matrix, which turns out to both reveal the
negative range and slightly improve accuracy.

    double_center(A)        A - rowmean - colmean + grandmean

The true DCCM is a correlation of mean-subtracted displacement vectors, so it is centred by
construction; the read-out is an uncentred affinity. Double-centering is the standard
operator taking a Gram/affinity matrix to its centred form (the same step used in classical
MDS), and it removes per-residue "hubness" — a residue whose attention is high against
everything no longer reads as correlated with everything.

    double_center_norm(A)   double_center, then divide by sqrt(diag outer product)

Rescales the centred matrix into correlation form, so its entries live on [-1, 1] like a real
DCCM and are directly comparable to the ground truth.

Measured on the held-out set (n=91, `--run_dir results/dccm_v2_unsup_testset`):

    method      raw r    dcentred   dc+norm   frac_neg after dc   (GT frac_neg = 0.518)
    DPLM       0.5177     0.5230     0.5264         0.758
    ESM2       0.5087     0.5175     0.5192         0.758
    ProstT5    0.4038     0.4105     0.4072         0.831
    SeqDance   0.5057     0.5092     0.5091         0.765

So it is a genuine (small) improvement for every method and preserves the ranking — it is not
a cosmetic rescaling that only helps the reference. Report the raw and corrected numbers
together; do not quietly swap one for the other.

Caveat worth stating in any figure caption: the correction OVERSHOOTS, producing ~76% negative
entries against a ground truth of ~52%. It restores the sign range but does not calibrate it.

CLI: evaluate the correction on a saved run.
    python evaluate/DCCM_v2/dccm_post.py --run_dir results/dccm_v2_unsup_testset \
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


def double_center(A):
    """A - rowmean - colmean + grandmean. Symmetric in, symmetric out."""
    A = np.asarray(A, dtype=np.float64)
    return A - A.mean(axis=1, keepdims=True) - A.mean(axis=0, keepdims=True) + A.mean()


def double_center_norm(A, eps=1e-12):
    """Double-centre, then rescale to correlation form (entries in [-1, 1])."""
    B = double_center(A)
    d = np.sqrt(np.clip(np.diag(B), eps, None))
    return np.clip(B / np.outer(d, d), -1.0, 1.0)


POSTPROCESS = {'none': lambda A: np.asarray(A, dtype=np.float64),
               'dcentre': double_center,
               'dcentre_norm': double_center_norm}


# ── CLI: does the correction actually help? ───────────────────────────────────

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
    a = a - a.mean()
    b = b - b.mean()
    da, db = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    return float((a * b).sum() / (da * db)) if da > 0 and db > 0 else np.nan


def main():
    p = argparse.ArgumentParser(description='Evaluate the double-centering correction.')
    p.add_argument('--run_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--ref', default='DPLM')
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    mats = _discover(args.run_dir)
    gt_path = os.path.join(args.run_dir, 'ground_truth.npz')
    if not mats or not os.path.exists(gt_path):
        raise SystemExit(f'Need ground_truth.npz and matrices in {args.run_dir}')
    gt = np.load(gt_path)
    loaded = {m: np.load(pth) for m, pth in mats.items()}
    methods = [args.ref] + sorted(m for m in loaded if m != args.ref)
    pids = [q for q in sorted(gt.files) if all(q in loaded[m].files for m in loaded)]
    print(f'Methods: {methods}   proteins: {len(pids)}')

    variants = ['none', 'dcentre', 'dcentre_norm']
    scores = {v: {m: [] for m in methods} for v in variants}
    fneg = {v: {m: [] for m in methods} for v in variants}
    gt_neg = []

    for pid in pids:
        g = np.asarray(gt[pid], dtype=np.float64)
        iu = np.triu_indices(g.shape[0], k=1)
        gv = g[iu]
        gt_neg.append(float((gv < 0).mean()))
        for m in methods:
            A = np.asarray(loaded[m][pid], dtype=np.float64)
            for v in variants:
                B = POSTPROCESS[v](A)[iu]
                scores[v][m].append(_pearson(B, gv))
                fneg[v][m].append(float((B < 0).mean()))

    lines = [f'# Double-centering correction — {os.path.basename(args.run_dir)}', '',
             f'n={len(pids)} proteins. Ground-truth fraction of negative pairs: '
             f'{np.mean(gt_neg):.3f}.', '',
             '| Method | raw r | double-centred r | dc+norm r | frac_neg raw | '
             'frac_neg after dc |', '|---|---|---|---|---|---|']
    print(f'\n{"method":<10}{"raw":>9}{"dcentre":>10}{"dc+norm":>10}'
          f'{"neg raw":>10}{"neg dc":>9}')
    summary = {}
    for m in methods:
        r0 = float(np.nanmean(scores['none'][m]))
        r1 = float(np.nanmean(scores['dcentre'][m]))
        r2 = float(np.nanmean(scores['dcentre_norm'][m]))
        n0 = float(np.mean(fneg['none'][m]))
        n1 = float(np.mean(fneg['dcentre'][m]))
        summary[m] = {'raw': r0, 'dcentre': r1, 'dcentre_norm': r2,
                      'frac_neg_raw': n0, 'frac_neg_dcentre': n1}
        print(f'{m:<10}{r0:9.4f}{r1:10.4f}{r2:10.4f}{n0:10.3f}{n1:9.3f}')
        lines.append(f'| {m} | {r0:.4f} | {r1:.4f} | {r2:.4f} | {n0:.3f} | {n1:.3f} |')

    # Is DPLM still ahead AFTER the correction? That is the question that matters.
    lines += ['', '## Paired comparison after `dcentre_norm`', '',
              '| Baseline | Δ (DPLM − method) | 95% CI | DPLM wins | p (Holm) |',
              '|---|---|---|---|---|']
    others = [m for m in methods if m != args.ref]
    a = np.array(scores['dcentre_norm'][args.ref], dtype=float)
    pv, rows = [], []
    for m in others:
        b = np.array(scores['dcentre_norm'][m], dtype=float)
        ok = ~(np.isnan(a) | np.isnan(b))
        d = a[ok] - b[ok]
        pv.append(wilcoxon_signed(a[ok], b[ok]))
        rows.append((m, d))
    adj = holm(pv)
    print(f'\nAfter dcentre_norm, paired vs {args.ref}:')
    for (m, d), pa in zip(rows, adj):
        lo, hi = bootstrap_ci(d)
        star = '***' if pa < 1e-3 else '**' if pa < 1e-2 else '*' if pa < 0.05 else 'ns'
        print(f'  {args.ref} − {m:<9} Δ={d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  '
              f'win={100*(d>0).mean():.1f}%  p_holm={pa:.2e} {star}')
        lines.append(f'| {m} | {d.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | '
                     f'{100*(d>0).mean():.1f}% | {pa:.1e} {star} |')
        summary.setdefault('_paired_dcentre_norm', {})[m] = {
            'delta': float(d.mean()), 'ci': [lo, hi],
            'win_rate': float((d > 0).mean()), 'p_holm': float(pa)}

    md = os.path.join(args.output_dir, 'postprocess_dcentre.md')
    with open(md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(args.output_dir, 'postprocess_dcentre.json'), 'w') as f:
        json.dump({'run_dir': args.run_dir, 'gt_frac_neg': float(np.mean(gt_neg)),
                   'summary': summary}, f, indent=2)
    print(f'\nWrote {md}')


if __name__ == '__main__':
    main()
