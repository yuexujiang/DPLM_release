"""
plot_dccm_v2.py — figures for the DCCM benchmark that respect the paired design.

What is different from evaluate/DCCM/plot_dccm.py:

  * significance brackets use the paired Wilcoxon signed-rank test, not Mann-Whitney
    (the methods are scored on the same proteins — see dccm_stats.py),
  * every panel is drawn on the common protein set, and the dropped count is printed,
  * a paired scatter panel: one point per protein, DPLM on y, baseline on x. Points above
    the diagonal are proteins where DPLM wins. This is the most honest "is DPLM better"
    visual — it shows the whole distribution, including the losses, rather than two
    summary boxes that overlap,
  * a paired-difference histogram with the mean delta and its bootstrap CI,
  * a stratification panel (delta vs. protein length) for reporting WHERE a method wins,
  * a multi-method case panel: ground truth and every method's predicted DCCM side by side
    on one shared colour scale.

Figures are deliberately plain (no chart-junk) and readable in grayscale via marker shape.
"""

import os
import textwrap

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dccm_stats import bootstrap_ci, stars, wilcoxon_signed

METHOD_COLORS = {
    'DPLM':     '#2166ac',
    'ESM2':     '#d6604d',
    'SeqDance': '#9970ab',
    'ProstT5':  '#4dac26',
    'SPLM':     '#f4a582',
}
REF_DEFAULT = 'DPLM'


def _color(m):
    return METHOD_COLORS.get(m, '#888888')


def _order(data, ref):
    """Reference first, then the rest by descending mean."""
    others = sorted([m for m in data if m != ref], key=lambda m: -data[m].mean())
    return [ref] + others


# ── 1. paired boxplot ─────────────────────────────────────────────────────────

def plot_paired_box(data, output_dir, ref=REF_DEFAULT, metric_name='Pearson r',
                    fname='fig1_paired_box.png', title=None):
    """Boxplot with paired-Wilcoxon brackets against the reference method."""
    methods = _order(data, ref)
    n = data[ref].size

    fig, ax = plt.subplots(figsize=(1.7 * len(methods) + 3.2, 5.6))
    pos = np.arange(1, len(methods) + 1)
    vals = [data[m] for m in methods]

    bp = ax.boxplot(vals, positions=pos, patch_artist=True, showmeans=True, widths=0.55,
                    showfliers=False,
                    medianprops=dict(color='black', linewidth=2),
                    meanprops=dict(marker='D', markerfacecolor='white',
                                   markeredgecolor='black', markersize=6))
    for patch, m in zip(bp['boxes'], methods):
        patch.set_facecolor(_color(m))
        patch.set_alpha(0.85 if m == ref else 0.45)
        patch.set_edgecolor('black' if m == ref else 'dimgray')
        patch.set_linewidth(1.8 if m == ref else 1.0)

    # jittered points, subsampled when there are thousands of proteins
    rng = np.random.default_rng(0)
    show = min(n, 400)
    sel = rng.choice(n, size=show, replace=False) if show < n else np.arange(n)
    for i, m in enumerate(methods):
        jit = rng.uniform(-0.14, 0.14, size=show)
        ax.scatter(pos[i] + jit, data[m][sel], s=9, alpha=0.30,
                   color=_color(m), zorder=3, linewidths=0)

    # paired significance vs reference
    ymax = max(np.percentile(v, 99) for v in vals)
    ymin = min(np.percentile(v, 1) for v in vals)
    span = ymax - ymin
    y = ymax + 0.06 * span
    for i, m in enumerate(methods):
        if m == ref:
            continue
        p = wilcoxon_signed(data[ref], data[m])
        d = data[ref] - data[m]
        h = 0.018 * span
        ax.plot([pos[0], pos[0], pos[i], pos[i]], [y, y + h, y + h, y],
                lw=1.2, color='dimgray')
        ax.text((pos[0] + pos[i]) / 2, y + h + 0.004 * span,
                f'{stars(p)}  Δ={d.mean():+.4f}  ({100*(d>0).mean():.0f}% wins)',
                ha='center', va='bottom', fontsize=8.5, color='black')
        y += 0.13 * span

    ax.set_xticks(pos)
    ax.set_xticklabels([f'{m}\nmean={data[m].mean():.3f}' for m in methods], fontsize=10)
    ax.set_ylabel(f'Per-protein {metric_name}\n(predicted vs. true DCCM)', fontsize=11)
    ax.set_title(title or f'DCCM benchmark — paired on n={n} proteins '
                          f'(Wilcoxon signed-rank vs. {ref})', fontsize=12)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim(ymin - 0.05 * span, y + 0.05 * span)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, fname)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'[plot] {out}')
    return out


# ── 2. paired scatter ─────────────────────────────────────────────────────────

def plot_paired_scatter(data, output_dir, ref=REF_DEFAULT, metric_name='Pearson r',
                        fname='fig2_paired_scatter.png', title=None):
    """One panel per baseline: per-protein ref (y) vs. baseline (x), with the y=x line.

    Every point above the diagonal is a protein where the reference method is better. This
    is the figure that shows a win *distribution* rather than a difference of two means.
    """
    others = [m for m in _order(data, ref) if m != ref]
    n = data[ref].size

    fig, axes = plt.subplots(1, len(others), figsize=(4.1 * len(others), 5.0),
                             squeeze=False)
    axes = axes[0]

    allv = np.concatenate([data[m] for m in data])
    lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    for ax, m in zip(axes, others):
        x, y = data[m], data[ref]
        wins = y > x
        ax.fill_between([lo, hi], [lo, hi], [hi, hi], color=_color(ref),
                        alpha=0.07, lw=0, zorder=0)
        ax.scatter(x[~wins], y[~wins], s=11, alpha=0.45, color='#999999',
                   linewidths=0, zorder=2, label=f'{m} better ({100*(~wins).mean():.0f}%)')
        ax.scatter(x[wins], y[wins], s=11, alpha=0.55, color=_color(ref),
                   linewidths=0, zorder=3, label=f'{ref} better ({100*wins.mean():.0f}%)')
        ax.plot([lo, hi], [lo, hi], color='black', lw=1.2, ls='--', zorder=4)

        d = y - x
        cl, ch = bootstrap_ci(d)
        p = wilcoxon_signed(y, x)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel(f'{m}   {metric_name}', fontsize=10)
        ax.set_ylabel(f'{ref}   {metric_name}', fontsize=10)
        ax.set_title(f'{ref} vs. {m}\nΔ={d.mean():+.4f} '
                     f'[{cl:+.4f}, {ch:+.4f}]   {stars(p)}', fontsize=10.5)
        ax.grid(alpha=0.22)
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9, markerscale=1.8)

    fig.suptitle(title or f'Per-protein paired comparison (n={n}) — '
                          f'points above the diagonal favour {ref}', fontsize=12)
    # equal-aspect panels leave the xlabel outside a plain tight_layout box
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, fname)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'[plot] {out}')
    return out


# ── 3. paired-difference histogram ────────────────────────────────────────────

def plot_delta_hist(data, output_dir, ref=REF_DEFAULT, metric_name='Pearson r',
                    fname='fig3_delta_hist.png', title=None):
    """Distribution of per-protein Δ = ref − baseline, with mean and bootstrap CI."""
    others = [m for m in _order(data, ref) if m != ref]
    fig, axes = plt.subplots(1, len(others), figsize=(3.9 * len(others), 3.6),
                             squeeze=False)
    axes = axes[0]

    for ax, m in zip(axes, others):
        d = data[ref] - data[m]
        cl, ch = bootstrap_ci(d)
        rng = max(abs(np.percentile(d, 0.5)), abs(np.percentile(d, 99.5)))
        bins = np.linspace(-rng, rng, 51)
        ax.hist(d[d <= 0], bins=bins, color='#999999', alpha=0.75, label=f'{m} better')
        ax.hist(d[d > 0], bins=bins, color=_color(ref), alpha=0.75, label=f'{ref} better')
        ax.axvline(0, color='black', lw=1.1, ls='--')
        ax.axvline(d.mean(), color='crimson', lw=1.8,
                   label=f'mean Δ={d.mean():+.4f}')
        ax.axvspan(cl, ch, color='crimson', alpha=0.15, lw=0)
        ax.set_xlabel(f'Δ {metric_name}  ({ref} − {m})', fontsize=10)
        ax.set_ylabel('proteins', fontsize=10)
        ax.set_title(f'{ref} − {m}', fontsize=10.5)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.22)

    fig.suptitle(title or f'Paired per-protein differences (n={data[ref].size})',
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, fname)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'[plot] {out}')
    return out


# ── 4. stratification ─────────────────────────────────────────────────────────

def plot_stratified(data, lengths, output_dir, ref=REF_DEFAULT, metric_name='Pearson r',
                    n_bins=5, fname='fig4_stratified.png', title=None):
    """Mean Δ (ref − baseline) within protein-length quantile bins, with bootstrap CIs.

    Reports WHERE a method wins instead of collapsing to a single average. `lengths` is an
    array aligned with the data arrays; proteins with unknown length are dropped.
    """
    others = [m for m in _order(data, ref) if m != ref]
    lengths = np.asarray(lengths, dtype=float)
    ok = ~np.isnan(lengths)
    if ok.sum() < n_bins * 3:
        print('[plot] not enough length annotations for stratification — skipping')
        return None

    edges = np.quantile(lengths[ok], np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6
    bin_idx = np.digitize(lengths, edges) - 1
    bin_idx[~ok] = -1

    fig, ax = plt.subplots(figsize=(1.6 * n_bins + 4, 4.4))
    width = 0.8 / len(others)
    centers = np.arange(n_bins)

    for k, m in enumerate(others):
        means, los, his = [], [], []
        for b in range(n_bins):
            sel = bin_idx == b
            d = data[ref][sel] - data[m][sel]
            if d.size < 3:
                means.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            cl, ch = bootstrap_ci(d, n_boot=4000)
            means.append(d.mean()); los.append(cl); his.append(ch)
        means = np.array(means); los = np.array(los); his = np.array(his)
        xpos = centers - 0.4 + width * (k + 0.5)
        ax.bar(xpos, means, width=width * 0.92, color=_color(m), alpha=0.55,
               edgecolor='black', linewidth=0.7, label=f'{ref} − {m}')
        ax.errorbar(xpos, means, yerr=[means - los, his - means], fmt='none',
                    ecolor='black', elinewidth=1.0, capsize=2.5)

    ax.axhline(0, color='black', lw=1.0)
    ax.set_xticks(centers)
    ax.set_xticklabels([f'{int(edges[b])}–{int(edges[b+1])}\n'
                        f'(n={(bin_idx == b).sum()})' for b in range(n_bins)], fontsize=9)
    ax.set_xlabel('Protein length (residues), quantile bins', fontsize=10.5)
    ax.set_ylabel(f'Δ {metric_name}  ({ref} − baseline)', fontsize=10.5)
    ax.set_title(title or f'Where {ref} wins — paired Δ by protein length '
                          f'(bars above 0 favour {ref})', fontsize=11.5)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, fname)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'[plot] {out}')
    return out


# ── 5. multi-method case panel ────────────────────────────────────────────────

def _zscore_offdiag(mat, iu, clip=3.0):
    """Standardise a predicted map by its own off-diagonal mean/SD, clipped to ±clip SD.

    This is an AFFINE per-panel transform, so the Pearson r reported in the panel title is
    exactly unchanged by it (Pearson is invariant to x -> ax + b for a > 0); clipping only
    affects the extreme near-diagonal tail, which saturates the colour map either way.

    It exists because the unsupervised read-outs are strongly diagonal-dominated: pooling
    them on one raw colour scale puts the 99th percentile around 0.01, which renders every
    off-diagonal structure as white and makes the panels unreadable even when r ≈ 0.6.
    """
    mat = np.asarray(mat, dtype=float)
    off = mat[iu]
    mu, sd = float(off.mean()), float(off.std())
    if sd <= 0:
        return np.zeros_like(mat)
    return np.clip((mat - mu) / sd, -clip, clip)


def plot_case_panel(pid, gt, preds, output_dir, ref=REF_DEFAULT, per_method_r=None,
                    fname=None, selection_note='', display='zscore', clip=3.0):
    """Ground truth + every method's predicted DCCM, on a colour scale shared by the
    predicted panels so method-to-method differences are pattern differences, not
    artefacts of per-panel rescaling.

    preds: {method: [L, L] array}

    display:
      'zscore' (default) — each predicted panel standardised over its own off-diagonal
                           entries and clipped to ±clip SD. Affine, so every reported
                           Pearson r is unchanged. Use for unsupervised read-outs, whose
                           raw magnitudes are tiny and diagonal-dominated.
      'shared'           — raw values on one scale, limit = 99th percentile of
                           |off-diagonal| pooled across methods. Faithful to raw
                           magnitude; only readable when the methods share a magnitude.
    """
    methods = [ref] + [m for m in preds if m != ref]
    gt = np.asarray(gt, dtype=float)
    L = gt.shape[0]
    iu = np.triu_indices(L, k=1)

    if display == 'zscore':
        shown = {m: _zscore_offdiag(preds[m], iu, clip) for m in methods}
        vlim = clip
        cbar_label = (f'predicted, per-panel z-score over off-diagonal, clipped ±{clip:g} '
                      f'(affine — Pearson r unchanged)')
    else:
        shown = {m: np.asarray(preds[m], dtype=float) for m in methods}
        pooled = np.concatenate([np.abs(shown[m][iu]) for m in methods])
        vlim = float(np.percentile(pooled, 99)) or 1.0
        cbar_label = f'predicted, shared raw scale ±{vlim:.2g}'

    ncol = len(methods) + 1
    fig_w = 2.75 * ncol + 1.6
    fig, axes = plt.subplots(1, ncol, figsize=(fig_w, 3.9),
                             constrained_layout=True)

    im0 = axes[0].imshow(gt, cmap='RdBu_r', vmin=-1, vmax=1, origin='lower',
                         interpolation='nearest')
    axes[0].set_title('Ground truth (MD)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Residue j', fontsize=9)
    axes[0].set_ylabel('Residue i', fontsize=9)

    im = None
    for ax, m in zip(axes[1:], methods):
        mat = shown[m].copy()
        np.fill_diagonal(mat, vlim)              # a real DCCM has diag ≡ 1; never learned
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-vlim, vmax=vlim, origin='lower',
                       interpolation='nearest')
        r = None if per_method_r is None else per_method_r.get(m)
        ax.set_title(m if r is None else f'{m}   r = {r:.3f}', fontsize=11,
                     fontweight='bold' if m == ref else 'normal',
                     color=_color(m) if m == ref else 'black')
        ax.set_xlabel('Residue j', fontsize=9)
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(2.2 if m == ref else 0.8)
            s.set_edgecolor(_color(ref) if m == ref else 'black')

    # one colourbar for the ground truth (±1) and one shared by every predicted panel
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02, location='bottom')
    fig.colorbar(im, ax=list(axes[1:]), fraction=0.046, pad=0.02, location='bottom',
                 label=cbar_label)

    sub = f'{pid}   (L={L})'
    if selection_note:
        # The note carries the selection rule and any correction applied, so it must stay
        # legible: wrap it to the figure width instead of letting it run off both edges.
        sub += '\n' + textwrap.fill(selection_note, width=int(fig_w * 10.5))
    fig.suptitle(sub, fontsize=10.5)

    os.makedirs(output_dir, exist_ok=True)
    fname = fname or f'case_{pid}.png'
    out = os.path.join(output_dir, fname)
    plt.savefig(out, dpi=200)
    plt.close()
    print(f'[plot] {out}')
    return out
