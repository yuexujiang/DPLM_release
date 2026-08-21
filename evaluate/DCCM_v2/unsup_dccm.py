"""
unsup_dccm.py — UNSUPERVISED (training-free) DCCM evaluation.

Instead of training a head to predict the DCCM (train_dccm.py / train_dccm_attn.py), this probe
asks a simpler question directly: *do residues that the backbone represents similarly also move
in a correlated way?* For every protein it builds a pairwise feature-similarity matrix S [L, L]
from the per-residue embeddings and correlates its upper triangle against the ground-truth DCCM
upper triangle. Nothing is trained — the score reflects the geometry of the frozen embedding
space alone.

This mirrors the *feature-similarity vs DCCM-similarity* idea behind the attention-augmented
predictor (model_dccm_attn.py: DCCM ~ a symmetric pairwise score over residue features), but as a
zero-parameter read-out so all five backbones are directly comparable — including SPLM, which
runs out-of-process and only returns per-residue embeddings (no attention maps). The shared,
attention-free signal is therefore per-residue feature similarity.

Similarity choices (--sim):
  cosine  : S[i,j] = cos(e_i, e_j)                      (default; matches DCCM's [-1,1] range)
  pearson : S[i,j] = cos(center(e_i), center(e_j))      (Pearson corr between residue vectors)
  dot     : S[i,j] = e_i · e_j                          (unnormalised)

Methods compared: dplm, esm2, prostt5, seqdance, splm  (via data_dccm.load_method_model).

Example (Delta; SLURM --export splits on commas → join methods with '+'):
  python evaluate/DCCM/unsup_dccm.py \
    --data_path <processed_test> --analysis_path <analysis_dir> --output_dir <out> \
    --methods dplm+esm2+prostt5+seqdance+splm \
    --dplm_config <dplm.yaml> --dplm_checkpoint <dplm.pth> \
    --seqdance_path <SeqDance-main/model> \
    --splm_path <SPLM repo> --splm_config <splm.yaml> --splm_checkpoint <splm.pth> \
    --splm_python <splm_v2 env python>
"""

import os
import sys
import re
import csv
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'methodology'))

from data_dccm import load_method_model, build_samples, METHOD_KEY
from model_dccm import protein_dccm_corr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
METHOD_COLORS = {'ESM2': '#d6604d', 'SeqDance': '#9970ab',
                 'DPLM': '#2166ac', 'ProstT5': '#4dac26'}
# (plot_corr_comparison / draw_dccm_pair inlined below)


# ── inlined from plot_dccm.py (the pre-v2 module, removed
#    in the public release; only these helpers were ever used from it) ─────
def _pval_str(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'

def draw_dccm_pair(pid, gt, pred, output_dir, method='', pearson=None, fname=None,
                   match_scale=False):
    """Side-by-side heatmaps of ground-truth and predicted DCCM for one protein.

    The predicted diagonal is set to 1 for display (a real DCCM has diag≡1; the model never
    learns it — the loss/metric exclude the diagonal). `match_scale` shows the predicted panel
    on ITS OWN colour range (±99th-percentile of |off-diagonal|), which reveals the structure
    an MSE-trained regressor captures but compresses in magnitude — the colorbar then reports
    the true (smaller) predicted range, so nothing is misrepresented.
    """
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float).copy()
    np.fill_diagonal(pred, 1.0)                       # diagonal is trivially 1

    if match_scale:
        iu = np.triu_indices(pred.shape[0], k=1)
        vpred = float(np.percentile(np.abs(pred[iu]), 99)) or 1.0
        pred_name = f'Predicted (display ±{vpred:.2f})'
    else:
        vpred, pred_name = 1.0, 'Predicted'

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mat, name, vlim in ((axes[0], gt, 'Ground truth', 1.0),
                                (axes[1], pred, pred_name, vpred)):
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-vlim, vmax=vlim, origin='lower',
                       interpolation='nearest')
        ax.set_title(name, fontsize=12)
        ax.set_xlabel('Residue j')
        ax.set_ylabel('Residue i')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    subtitle = f'{pid}' + (f'  [{method}]' if method else '')
    if pearson is not None:
        subtitle += f'   Pearson r = {pearson:.3f}'
    fig.suptitle(f'DCCM — {subtitle}', fontsize=13)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fname = fname or f'dccm_heatmap_{method}_{pid}.png'
    out = os.path.join(output_dir, fname)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[plot] DCCM heatmap saved → {out}')
    return out

def plot_corr_comparison(per_method_corrs, output_dir, metric_name='Pearson r',
                         fname='dccm_pred_corr_comparison.png', title=None):
    """per_method_corrs: {method: [per-protein correlation, ...]}."""
    methods = [m for m in per_method_corrs if len(per_method_corrs[m]) > 0]
    if not methods:
        print('[plot] no data to plot.')
        return

    fig, ax = plt.subplots(figsize=(1.8 * len(methods) + 3, 5))
    positions = list(range(1, len(methods) + 1))
    data = [per_method_corrs[m] for m in methods]
    colors = [METHOD_COLORS.get(m, '#888888') for m in methods]

    bp = ax.boxplot(data, positions=positions, patch_artist=True,
                    showmeans=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2),
                    meanprops=dict(marker='D', markerfacecolor='white',
                                   markeredgecolor='black', markersize=6))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    rng = np.random.default_rng(0)
    for i, (vals, color) in enumerate(zip(data, colors)):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(positions[i] + jitter, vals, s=18, alpha=0.6, color=color, zorder=3)

    # Mann-Whitney vs the first method.
    ref = data[0]
    y_max = max(max(v) if v else 0 for v in data)
    bracket_y = y_max + 0.05
    for i in range(1, len(methods)):
        if not ref or not data[i]:
            continue
        _, pval = mannwhitneyu(ref, data[i], alternative='two-sided')
        h = 0.02
        ax.plot([positions[0], positions[0], positions[i], positions[i]],
                [bracket_y, bracket_y + h, bracket_y + h, bracket_y],
                lw=1.2, color='dimgray')
        ax.text((positions[0] + positions[i]) / 2, bracket_y + h + 0.01,
                f'{_pval_str(pval)}\n(p={pval:.1e})',
                ha='center', va='bottom', fontsize=8.5, color='dimgray')
        bracket_y += 0.12

    xticklabels = [f'{m}\n(n={len(per_method_corrs[m])}, '
                   f'med={np.median(per_method_corrs[m]):.3f})' for m in methods]
    ax.set_xticks(positions)
    ax.set_xticklabels(xticklabels, fontsize=10)
    ax.set_ylabel(f'Per-protein {metric_name}\n(predicted vs true DCCM)', fontsize=11)
    ax.set_title(title or 'DCCM prediction — per-protein correlation on test set',
                 fontsize=12)
    ax.axhline(0, color='gray', lw=0.8, linestyle='--')
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, fname)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[plot] correlation comparison saved → {out}')


# ── Feature-similarity matrix ─────────────────────────────────────────────────

def feature_similarity(emb, sim='cosine'):
    """Per-residue embedding [L, D] → pairwise similarity matrix [L, L].

    cosine  : row-normalise, then Gram matrix (values in [-1, 1]).
    pearson : center each residue vector across D, then cosine (Pearson r between residues).
    dot     : raw Gram matrix (unnormalised).
    """
    e = np.asarray(emb, dtype=np.float64)
    if sim == 'pearson':
        e = e - e.mean(axis=1, keepdims=True)
    if sim in ('cosine', 'pearson'):
        norm = np.linalg.norm(e, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        e = e / norm
    elif sim != 'dot':
        raise ValueError(f"sim must be cosine|pearson|dot, got {sim}")
    return e @ e.T                                   # [L, L] symmetric


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Unsupervised DCCM evaluation: corr(feature similarity, DCCM).')
    p.add_argument('--data_path', required=True,
                   help='processed_* dir with the protein IDs to evaluate on.')
    p.add_argument('--analysis_path', required=True,
                   help='Base dir with {pid}_analysis/{pid}.pdb + {pid}_R{1,2,3}.xtc.')
    p.add_argument('--methods', default='dplm+esm2+prostt5+seqdance+splm',
                   help='Backbones to compare (dplm, esm2, esmc, prostt5, seqdance, splm). '
                        'Join with "+" (sbatch --export splits on commas).')
    p.add_argument('--esmc_model', default='esmc_600m',
                   choices=['esmc_300m', 'esmc_600m'],
                   help='ESM-C size when esmc is in --methods (default: esmc_600m).')
    p.add_argument('--sim', default='cosine', choices=['cosine', 'pearson', 'dot'],
                   help='Pairwise feature-similarity used as the DCCM read-out.')
    # backbone paths (only those for the requested methods are needed)
    p.add_argument('--dplm_config', default=None)
    p.add_argument('--dplm_checkpoint', default=None)
    p.add_argument('--seqdance_path', default=None)
    p.add_argument('--splm_path', default=None)
    p.add_argument('--splm_config', default=None)
    p.add_argument('--splm_checkpoint', default=None)
    p.add_argument('--splm_python', default=None)
    p.add_argument('--splm_max_length', type=int, default=1022)
    p.add_argument('--splm_cache_pkl', default=None)
    # ground-truth DCCM
    p.add_argument('--dccm_dir', default=None,
                   help='Optional cache dir for {pid}_dccm_R{1,2,3}.npy.')
    p.add_argument('--dccm_replicate', default=None, choices=['R1', 'R2', 'R3'],
                   help='Single MD replicate for DCCM; omit to average R1/R2/R3.')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--max_proteins', type=int, default=None)
    p.add_argument('--num_example_heatmaps', type=int, default=3,
                   help='Number of proteins to draw feature-sim-vs-DCCM heatmaps for.')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda'
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        device = 'cpu'
    print(f'Device: {device}   similarity: {args.sim}')

    from protein_level_emb_md import load_proteins
    proteins = load_proteins(args.data_path, args.analysis_path,
                             max_proteins=args.max_proteins)
    print(f'Eval proteins: {len(proteins)}')

    methods = [m.strip().lower() for m in re.split(r'[,+;\s]+', args.methods) if m.strip()]
    print(f'Methods to run: {methods}')

    per_method_pearson, per_method_spearman = {}, {}

    for method in methods:
        disp = METHOD_KEY.get(method, method)
        print(f'\n================  {disp}  ================')
        # SPLM precomputes a {sequence: [L,D]} lookup out-of-process, so it needs the full
        # eval protein list up front; the other backbones ignore `proteins`.
        models_dict = load_method_model(
            method, device, seqdance_path=args.seqdance_path,
            dplm_config=args.dplm_config, dplm_checkpoint=args.dplm_checkpoint,
            proteins=proteins, splm_path=args.splm_path, splm_config=args.splm_config,
            splm_checkpoint=args.splm_checkpoint, splm_python=args.splm_python,
            splm_max_length=args.splm_max_length, splm_cache_pkl=args.splm_cache_pkl,
            esmc_model=args.esmc_model)

        print('Building samples (embeddings + ground-truth DCCM) …')
        samples = build_samples(proteins, models_dict, args.analysis_path, device,
                                dccm_dir=args.dccm_dir, replicate=args.dccm_replicate)
        del models_dict
        if device == 'cuda':
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        if not samples:
            print(f'[{disp}] no usable samples — skipping.')
            continue

        # ── Zero-parameter read-out: corr(feature similarity, DCCM) per protein ──
        results = {}
        for s in samples:
            S = feature_similarity(s['emb'], sim=args.sim)      # [L, L]
            results[s['pid']] = protein_dccm_corr(S, s['dccm'])

        csv_path = os.path.join(args.output_dir, f'unsup_corr_{disp}.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['pid', 'pearson', 'spearman'])
            for pid, (pear, spear) in results.items():
                w.writerow([pid, f'{pear:.6f}', f'{spear:.6f}'])
        print(f'[{disp}] per-protein correlations → {csv_path}')

        pears  = [p for p, _ in results.values() if not np.isnan(p)]
        spears = [s for _, s in results.values() if not np.isnan(s)]
        per_method_pearson[disp]  = pears
        per_method_spearman[disp] = spears
        print(f'[{disp}] mean Pearson={np.mean(pears):.4f}  median={np.median(pears):.4f}  '
              f'(n={len(pears)})')

        # Example feature-sim-vs-DCCM heatmaps for a few proteins.
        for s in samples[:args.num_example_heatmaps]:
            S = feature_similarity(s['emb'], sim=args.sim)
            pear, _ = protein_dccm_corr(S, s['dccm'])
            draw_dccm_pair(s['pid'], s['dccm'], S, args.output_dir,
                           method=f'{disp}-unsup', pearson=pear)

    if per_method_pearson:
        plot_corr_comparison(per_method_pearson, args.output_dir,
                             metric_name='Pearson r',
                             fname='unsup_dccm_pearson_comparison.png')
        plot_corr_comparison(per_method_spearman, args.output_dir,
                             metric_name='Spearman r',
                             fname='unsup_dccm_spearman_comparison.png')

        print('\n=== Unsupervised DCCM summary (mean / median per-protein Pearson) ===')
        for disp in per_method_pearson:
            pears = per_method_pearson[disp]
            print(f'  {disp:<10} mean={np.mean(pears):.4f}  median={np.median(pears):.4f}  '
                  f'n={len(pears)}')

    print('\nDone.')


if __name__ == '__main__':
    main()
