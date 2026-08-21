"""
dccm_stats.py — paired statistics for per-protein DCCM correlations.

Every method in the DCCM benchmark is evaluated on the SAME proteins, so the per-protein
correlations are *paired*. The v1 plotting code (evaluate/DCCM/plot_dccm.py:70) compared
methods with a Mann-Whitney U test, which treats the two samples as independent — that
discards the pairing, loses power, and is the wrong null hypothesis for this design.

This module provides the paired equivalents:

  paired_frame      : align per-protein CSVs onto the set of proteins ALL methods scored
                      (v1 boxplots mixed n=92 and n=91, so the boxes weren't comparable)
  wilcoxon_signed   : two-sided Wilcoxon signed-rank on the per-protein differences
  bootstrap_ci      : percentile bootstrap CI for the mean paired difference
  holm              : Holm-Bonferroni correction across the baselines compared to DPLM
  compare_to_ref    : the full table (delta, CI, win rate, p, corrected p) for one metric

Pure numpy: scipy is used for the Wilcoxon p-value when available, with an exact-enough
normal approximation (tie- and zero-corrected) as a fallback so the module also runs in a
bare-numpy environment such as a login node.
"""

import csv
import math
import os

import numpy as np

try:                                     # scipy is present in the Delta training env and in
    from scipy.stats import wilcoxon as _scipy_wilcoxon   # python/miniforge3_datascience
    _HAVE_SCIPY = True
except Exception:                        # bare-numpy fallback
    _HAVE_SCIPY = False


# ── loading ───────────────────────────────────────────────────────────────────

def read_corr_csv(path):
    """Read a per-protein correlation CSV → {pid: {'pearson': float, 'spearman': float}}.

    Handles both v1 layouts: per_protein_corr_attn_{M}.csv (supervised, columns
    pid/pearson/spearman) and unsup_attn_corr_{M}.csv (unsupervised, which adds a
    'readout' column).
    """
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row['pid']] = {'pearson': float(row['pearson']),
                               'spearman': float(row['spearman'])}
    return out


def paired_frame(method_to_path, metric='pearson'):
    """{method: csv_path} → (pids, {method: np.array}) over the COMMON, non-NaN proteins.

    Restricting to the intersection is what makes the comparison paired. Any protein that
    a single method failed on is dropped for every method, and the number dropped is
    returned so it can be reported rather than silently absorbed.
    """
    tables = {m: read_corr_csv(p) for m, p in method_to_path.items()}
    common = set.intersection(*[set(t) for t in tables.values()])

    # drop proteins where any method produced NaN (e.g. a constant read-out row)
    usable = [pid for pid in sorted(common)
              if all(not math.isnan(tables[m][pid][metric]) for m in tables)]

    n_dropped = {m: len(tables[m]) - len(usable) for m in tables}
    data = {m: np.array([tables[m][pid][metric] for pid in usable], dtype=float)
            for m in tables}
    return usable, data, n_dropped


# ── tests ─────────────────────────────────────────────────────────────────────

def _wilcoxon_normal_approx(d):
    """Two-sided Wilcoxon signed-rank p via normal approximation.

    Zeros are dropped (Wilcoxon's original handling) and tied |d| get average ranks with
    the standard tie correction to the variance.
    """
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    n = d.size
    if n == 0:
        return 1.0

    a = np.abs(d)
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(n, dtype=float)
    sa = a[order]
    i = 0
    while i < n:                                  # average ranks within ties
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1

    w_plus = ranks[d > 0].sum()
    mu = n * (n + 1) / 4.0
    _, counts = np.unique(a, return_counts=True)
    tie_term = float((counts ** 3 - counts).sum())
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return 1.0
    z = (w_plus - mu) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2.0))     # two-sided


def wilcoxon_signed(a, b):
    """Two-sided Wilcoxon signed-rank p-value for paired samples a, b."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.all(d == 0):
        return 1.0
    if _HAVE_SCIPY:
        try:
            return float(_scipy_wilcoxon(a, b, alternative='two-sided').pvalue)
        except Exception:
            pass
    return _wilcoxon_normal_approx(d)


def bootstrap_ci(d, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean of the paired differences d."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def cohen_dz(d):
    """Paired effect size: mean(d) / sd(d)."""
    d = np.asarray(d, dtype=float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float('nan')


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)            # enforce monotonicity
        adj[i] = min(1.0, running)
    return adj


def stars(p):
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 5e-2:
        return '*'
    return 'ns'


# ── the comparison table ──────────────────────────────────────────────────────

def compare_to_ref(data, ref='DPLM', n_boot=10000, seed=0):
    """Paired comparison of every method in `data` against `ref`.

    data: {method: np.array of per-protein correlations, all the same length and aligned}

    Returns (summary, comparisons):
      summary     : {method: {mean, median, sd, n}}
      comparisons : list of dicts, one per non-ref method, with the paired delta, its
                    bootstrap CI, the fraction of proteins on which ref wins, the raw
                    Wilcoxon p and the Holm-adjusted p across the whole family.
    """
    if ref not in data:
        raise KeyError(f'reference method {ref!r} not in {sorted(data)}')

    summary = {m: {'mean': float(v.mean()), 'median': float(np.median(v)),
                   'sd': float(v.std(ddof=1)) if v.size > 1 else float('nan'),
                   'n': int(v.size)}
               for m, v in data.items()}

    others = [m for m in data if m != ref]
    rows, pvals = [], []
    for m in others:
        d = data[ref] - data[m]
        lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
        p = wilcoxon_signed(data[ref], data[m])
        pvals.append(p)
        rows.append({'method': m,
                     'delta': float(d.mean()),
                     'ci_lo': lo, 'ci_hi': hi,
                     'win_rate': float((d > 0).mean()),
                     'dz': cohen_dz(d),
                     'p_raw': p})

    if rows:
        for row, adj in zip(rows, holm(pvals)):
            row['p_holm'] = float(adj)
            row['stars'] = stars(adj)
    return summary, rows


def format_table(summary, rows, ref='DPLM', metric='Pearson r', title=''):
    """Render the comparison as a markdown table (for the README / paper notes)."""
    lines = []
    if title:
        lines += [f'### {title}', '']
    n = summary[ref]['n']
    lines += [f'Metric: **{metric}**, paired on n={n} proteins, reference = **{ref}**.', '']
    lines += ['| Method | mean | median | Δ (ref − method) | 95% CI | ref wins | dz | p (Wilcoxon) | p (Holm) | |',
              '|---|---|---|---|---|---|---|---|---|---|']
    lines.append(f"| **{ref}** | **{summary[ref]['mean']:.4f}** | "
                 f"{summary[ref]['median']:.4f} | — | — | — | — | — | — | |")
    for r in sorted(rows, key=lambda x: -x['delta']):
        m = r['method']
        lines.append(
            f"| {m} | {summary[m]['mean']:.4f} | {summary[m]['median']:.4f} | "
            f"{r['delta']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
            f"{100 * r['win_rate']:.1f}% | {r['dz']:+.3f} | {r['p_raw']:.2e} | "
            f"{r['p_holm']:.2e} | {r['stars']} |")
    return '\n'.join(lines)


def write_table_csv(path, summary, rows, ref='DPLM', metric='pearson', setting=''):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['setting', 'metric', 'reference', 'method', 'n', 'mean', 'median', 'sd',
                    'delta_vs_ref', 'ci_lo', 'ci_hi', 'ref_win_rate', 'cohen_dz',
                    'p_wilcoxon', 'p_holm'])
        s = summary[ref]
        w.writerow([setting, metric, ref, ref, s['n'], f"{s['mean']:.6f}",
                    f"{s['median']:.6f}", f"{s['sd']:.6f}", '', '', '', '', '', '', ''])
        for r in rows:
            s = summary[r['method']]
            w.writerow([setting, metric, ref, r['method'], s['n'], f"{s['mean']:.6f}",
                        f"{s['median']:.6f}", f"{s['sd']:.6f}", f"{r['delta']:.6f}",
                        f"{r['ci_lo']:.6f}", f"{r['ci_hi']:.6f}",
                        f"{r['win_rate']:.4f}", f"{r['dz']:.4f}",
                        f"{r['p_raw']:.6e}", f"{r['p_holm']:.6e}"])
    return path
