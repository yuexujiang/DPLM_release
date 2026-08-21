"""
protein_level_emb_md.py — per-protein residue-pair embedding-vs-DCCM correlation,
for DPLM, ESM2, ProstT5, SeqDance.

For each protein, computes its DCCM (Dynamic Cross-Correlation Matrix) directly from
the raw MD trajectory: for every residue pair (i, j), how correlated their positional
fluctuations are over the simulation (-1 = anti-correlated motion, +1 = correlated
motion, ~0 = independent). This is then correlated (Spearman) against the
residue-pair embedding cosine-similarity matrix, giving one score per protein per
embedding method — a direct test of whether residues with similar embeddings
actually move together in the simulation, not just whether they share a similar
*scalar* dynamics value (which is all the RSA analyses in rmsf.py test).

NOTE: This file previously computed a protein-protein RSA (pooled-embedding
similarity vs a 6-d dynamics-descriptor similarity, across all protein pairs). That
analysis has been disabled (commented out, kept for reference) in favor of the
DCCM-based residue-pair analysis below. The descriptor loader and pooled-embedding
cache it depended on have been removed entirely, since the active analysis only
needs each protein's sequence and per-residue embeddings.

Model loading and residue-embedding extraction are reused from rmsf.py rather than
duplicated.

Usage
-----
PYTHONPATH=. python evaluate/methodology/protein_level_emb_md.py \\
    --data_path     /path/to/DPLM_data/Atlas_geom2vec_test/ \\
    --analysis_path /path/to/DPLM_data/analysis/ \\
    --dplm_config   ./configs/config_vivit5_delta.yaml \\
    --dplm_checkpoint ./results/vivit5/checkpoints/checkpoint_best_val_whole_loss.pth \\
    --seqdance_path <SeqDance-main/model> \\
    --output_dir    ./evaluate/methodology/protein_level_output/ \\
    --dccm_dir      ./evaluate/methodology/dccm_cache/ \\
    --max_proteins  30
"""

# ============================================================
# 0.  Imports
# ============================================================
import argparse
import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import spearmanr, mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.evaluation import pdb2seq

sys.path.insert(0, os.path.dirname(__file__))
from rmsf import (  # noqa: E402  (local import after sys.path setup)
    load_dplm, load_esm2, load_prostt5, load_seqdance,
    build_embedding_cache,
    METHOD_ORDER, METHOD_COLORS,
)

warnings.filterwarnings('ignore')


# ============================================================
# 1.  Protein enumeration
# ============================================================

def load_proteins(data_path, analysis_path, max_proteins=None):
    """Return a list of dicts, one per valid protein: {'pid': str, 'sequence': str}.

    Only requires a readable {pid}_analysis/{pid}.pdb — trajectory (.xtc)
    availability is checked later, per-replicate, by the DCCM loader.

    Protein IDs are derived from filenames in data_path as the first
    '#'-separated token, mirroring rmsf.py's load_protein_data.
    """
    seen = set()
    proteins = []

    for fname in sorted(os.listdir(data_path)):
        fname = fname.strip()
        if not fname:
            continue
        parts = fname.split('#')
        if len(parts) < 2:
            continue
        pid = parts[0]
        if pid in seen:
            continue
        seen.add(pid)

        pdb_file = os.path.join(analysis_path, f'{pid}_analysis', f'{pid}.pdb')
        if not os.path.exists(pdb_file):
            continue

        try:
            sequence = str(pdb2seq(pdb_file))
        except Exception as e:
            print(f'  [skip] {pid}: {e}')
            continue

        proteins.append({'pid': pid, 'sequence': sequence})

        if max_proteins and len(proteins) >= max_proteins:
            break

    print(f'Loaded {len(proteins)} proteins.')
    return proteins


# ============================================================
# 3.  (disabled) RSA: protein-protein embedding similarity vs descriptor similarity
# ============================================================
# Superseded by the DCCM-based residue-pair analysis below (section 3b). Kept here
# for reference — uncomment to re-run the protein-protein RSA instead/in addition.

# def _cosine_sim_matrix(X):
#     norms = np.linalg.norm(X, axis=1, keepdims=True)
#     norms[norms == 0] = 1e-8
#     unit = X / norms
#     return unit @ unit.T
#
#
# def _descriptor_sim_matrix(D):
#     """Negative Euclidean distance on z-scored descriptor columns → similarity."""
#     mu = D.mean(axis=0, keepdims=True)
#     sigma = D.std(axis=0, keepdims=True)
#     sigma[sigma == 0] = 1e-8
#     Dz = (D - mu) / sigma
#     diff = Dz[:, None, :] - Dz[None, :, :]
#     dist = np.sqrt(np.sum(diff ** 2, axis=-1))
#     return -dist
#
#
# def compute_rsa_scores(valid_proteins, cache, n_permutations=1000, seed=0):
#     """Per-method RSA Spearman score between pooled-embedding similarity and
#     descriptor similarity across all protein pairs, plus a Mantel permutation p-value.
#
#     Returns
#     -------
#     dict: {method: {'rsa': float, 'pval': float, 'n_proteins': int}}
#     """
#     methods = [m for m in METHOD_ORDER if m in next(iter(cache.values()))]
#     pids = [p['pid'] for p in valid_proteins]
#     n = len(pids)
#
#     descriptors = np.stack([p['descriptor'] for p in valid_proteins], axis=0)  # [N, 6]
#     desc_sim = _descriptor_sim_matrix(descriptors)
#     iu = np.triu_indices(n, k=1)
#     desc_sim_flat = desc_sim[iu]
#
#     rng = np.random.default_rng(seed)
#     results = {}
#
#     for method in methods:
#         emb_matrix = np.stack([cache[pid][method] for pid in pids], axis=0)  # [N, D]
#         emb_sim = _cosine_sim_matrix(emb_matrix)
#         emb_sim_flat = emb_sim[iu]
#
#         rsa, _ = spearmanr(emb_sim_flat, desc_sim_flat)
#
#         # Mantel permutation test: shuffle protein order on the embedding side only.
#         perm_rsa = np.empty(n_permutations, dtype=np.float32)
#         for i in range(n_permutations):
#             perm = rng.permutation(n)
#             emb_sim_perm = emb_sim[np.ix_(perm, perm)][iu]
#             perm_rsa[i], _ = spearmanr(emb_sim_perm, desc_sim_flat)
#         pval = float(np.mean(np.abs(perm_rsa) >= np.abs(rsa)))
#
#         results[method] = {'rsa': float(rsa), 'pval': pval, 'n_proteins': n}
#         print(f'  {method}: RSA = {rsa:.4f}  (Mantel p = {pval:.4f}, n = {n} proteins)')
#
#     return results
#
#
# def plot_rsa_bar(rsa_results, output_dir):
#     methods = [m for m in METHOD_ORDER if m in rsa_results]
#     vals = [rsa_results[m]['rsa'] for m in methods]
#     colors = [METHOD_COLORS[m] for m in methods]
#
#     fig, ax = plt.subplots(figsize=(6, 5))
#     bars = ax.bar(methods, vals, color=colors, alpha=0.85)
#
#     for bar, m in zip(bars, methods):
#         pval = rsa_results[m]['pval']
#         p_str = _pval_str(pval)
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width() / 2,
#                 height + (0.01 if height >= 0 else -0.03),
#                 f'{height:.3f}\n{p_str}',
#                 ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
#
#     ax.axhline(0, color='gray', lw=0.8, linestyle='--')
#     ax.set_ylabel('RSA Spearman r\n(pooled-emb similarity vs descriptor similarity)',
#                   fontsize=11)
#     n_proteins = next(iter(rsa_results.values()))['n_proteins']
#     ax.set_title(f'Protein-level RSA  (n = {n_proteins} proteins)', fontsize=12)
#     ax.grid(axis='y', alpha=0.25)
#     plt.tight_layout()
#
#     os.makedirs(output_dir, exist_ok=True)
#     out = os.path.join(output_dir, 'protein_level_rsa.png')
#     plt.savefig(out, dpi=150)
#     plt.close()
#     print(f'[Protein-level RSA] saved → {out}')


def _pval_str(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


# ============================================================
# 3b.  DCCM: residue-pair embedding cosine similarity vs Dynamic Cross-Correlation
# ============================================================

def compute_dccm_matrix(pdb_file, xtc_file):
    """CA-based Dynamic Cross-Correlation Matrix from one MD replicate.

    Superposes all frames onto frame 0 (CA atoms) to remove rigid-body motion,
    then correlates per-residue displacement vectors across the trajectory.
    Returns [L, L] float32 matrix, values in [-1, 1], diagonal == 1.
    """
    import mdtraj as md

    traj = md.load(xtc_file, top=pdb_file)
    ca_idx = traj.topology.select('name CA')
    traj.superpose(traj, frame=0, atom_indices=ca_idx)
    pos = traj.xyz[:, ca_idx, :]                          # [T, L, 3]
    disp = pos - pos.mean(axis=0, keepdims=True)
    num  = np.einsum('tik,tjk->ij', disp, disp)
    norm = np.sqrt(np.einsum('tik,tik->i', disp, disp))
    norm[norm == 0] = 1e-8
    dccm = num / np.outer(norm, norm)
    return np.clip(dccm, -1.0, 1.0).astype(np.float32)


def get_dccm_for_replicate(pid, analysis_path, replicate, dccm_dir=None):
    """Load a cached DCCM for one replicate (R1/R2/R3), or compute + cache it."""
    if dccm_dir:
        cache_path = os.path.join(dccm_dir, f'{pid}_dccm_{replicate}.npy')
        if os.path.exists(cache_path):
            return np.load(cache_path)

    adir = os.path.join(analysis_path, f'{pid}_analysis')
    pdb_file = os.path.join(adir, f'{pid}.pdb')
    xtc_file = os.path.join(adir, f'{pid}_{replicate}.xtc')
    if not (os.path.exists(pdb_file) and os.path.exists(xtc_file)):
        return None

    try:
        dccm = compute_dccm_matrix(pdb_file, xtc_file)
    except Exception as e:
        print(f'  [DCCM] {pid}/{replicate}: {e}')
        return None

    if dccm_dir:
        os.makedirs(dccm_dir, exist_ok=True)
        np.save(os.path.join(dccm_dir, f'{pid}_dccm_{replicate}.npy'), dccm)

    return dccm


def load_protein_dccm(pid, analysis_path, dccm_dir=None, replicate=None):
    """Return the DCCM matrix for one protein: a single replicate if `replicate`
    is given (e.g. 'R1'), else the elementwise average of R1/R2/R3."""
    reps = [replicate] if replicate else ['R1', 'R2', 'R3']
    mats = [m for m in (get_dccm_for_replicate(pid, analysis_path, r, dccm_dir)
                        for r in reps) if m is not None]
    if not mats:
        return None
    return np.mean(np.stack(mats, axis=0), axis=0)


def _emb_dccm_corr(emb, dccm):
    """Spearman correlation between residue-pair embedding cosine similarity and
    the corresponding DCCM value, for one protein.

    Both sides are similarities in [-1, 1] (unlike the RMSF/Bfactor/Neq *difference*
    matrices used in rmsf.py's RSA analysis), so no distance flip is applied here —
    we correlate similarity directly against similarity.
    """
    L = emb.shape[0]
    if L < 4 or dccm is None or dccm.shape != (L, L):
        return np.nan

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    unit = emb / norms
    cos_sim = unit @ unit.T

    iu = np.triu_indices(L, k=1)
    corr, _ = spearmanr(cos_sim[iu], dccm[iu])
    return corr


def plot_dccm_corr_boxplot(valid_proteins, residue_cache, analysis_path, output_dir,
                           dccm_dir=None, replicate=None):
    """Boxplot of per-protein Spearman(emb cosine sim, DCCM); Mann-Whitney p-values
    vs DPLM. Mirrors rmsf.py's plot_rsa_boxplot layout."""
    methods = [m for m in METHOD_ORDER if m in next(iter(residue_cache.values()))]

    per_protein_corrs = {m: [] for m in methods}
    for prot in valid_proteins:
        pid = prot['pid']
        if pid not in residue_cache:
            continue
        dccm = load_protein_dccm(pid, analysis_path, dccm_dir=dccm_dir, replicate=replicate)
        if dccm is None:
            print(f'  [skip] {pid}: no DCCM available')
            continue
        for method in methods:
            if method not in residue_cache[pid]:
                continue
            corr = _emb_dccm_corr(residue_cache[pid][method], dccm)
            if not np.isnan(corr):
                per_protein_corrs[method].append(corr)

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = list(range(1, len(methods) + 1))
    data = [per_protein_corrs[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

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
        ax.scatter(positions[i] + jitter, vals,
                   s=18, alpha=0.6, color=color, zorder=3)

    dplm_corrs = per_protein_corrs.get('DPLM', [])
    y_max = max(max(v) if v else 0 for v in data)
    bracket_y = y_max + 0.05

    for i, method in enumerate(methods):
        if method == 'DPLM' or not dplm_corrs or not per_protein_corrs[method]:
            continue
        stat, pval = mannwhitneyu(dplm_corrs, per_protein_corrs[method],
                                  alternative='two-sided')
        p_str = _pval_str(pval)
        x1, x2 = 1, positions[i]
        h = 0.02
        ax.plot([x1, x1, x2, x2],
                [bracket_y, bracket_y + h, bracket_y + h, bracket_y],
                lw=1.2, color='dimgray')
        ax.text((x1 + x2) / 2, bracket_y + h + 0.01, p_str,
                ha='center', va='bottom', fontsize=9, color='dimgray')
        bracket_y += 0.10

    ax.set_xticks(positions)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Per-protein Spearman r\n(emb cosine sim vs DCCM)", fontsize=11)
    rep_str = replicate if replicate else 'avg(R1,R2,R3)'
    ax.set_title(f"Residue-pair embedding similarity vs DCCM  ({rep_str})", fontsize=12)
    ax.axhline(0, color='gray', lw=0.8, linestyle='--')
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f'dccm_corr_boxplot_{replicate or "avg"}.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[DCCM correlation] saved → {out}')


# ============================================================
# 4.  Argument parsing + main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Per-protein residue-pair embedding-vs-DCCM correlation.')
    p.add_argument('--data_path',       required=True,
                   help='Dir of trajectory/processed files (used to enumerate protein IDs).')
    p.add_argument('--analysis_path',   required=True,
                   help='Base dir containing {pid}_analysis/ subfolders.')
    p.add_argument('--dplm_config',     required=True,
                   help='Path to training config YAML (for DPLM adapter settings).')
    p.add_argument('--dplm_checkpoint', required=True,
                   help='Path to training checkpoint .pth.')
    p.add_argument('--seqdance_path',   default=None,
                   help='Path to SeqDance-main/model/ directory. '
                        'Omit to skip SeqDance analysis.')
    p.add_argument('--output_dir',      default='./protein_level_output',
                   help='Directory to save all figures.')
    p.add_argument('--max_proteins',    type=int, default=None,
                   help='Cap number of proteins (for quick testing).')
    p.add_argument('--dccm_dir',        default=None,
                   help='Directory to cache/load precomputed DCCM .npy files '
                        '({pid}_dccm_R1/R2/R3.npy). Omit to compute in memory '
                        'every run without caching.')
    p.add_argument('--dccm_replicate',  default=None, choices=['R1', 'R2', 'R3'],
                   help='Single MD replicate to use for DCCM. Omit to average '
                        'the DCCM matrices from R1/R2/R3.')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Enumerate proteins ─────────────────────────────────────────────
    print('\n=== Loading proteins ===')
    proteins = load_proteins(
        args.data_path, args.analysis_path, max_proteins=args.max_proteins)
    if not proteins:
        print('No valid proteins found. Exiting.')
        return

    # ── Step 2: Load models ───────────────────────────────────────────────────
    print('\n=== Loading models ===')
    models_dict = {}

    print('Loading DPLM …')
    models_dict['DPLM'] = load_dplm(args.dplm_config, args.dplm_checkpoint, device)

    print('Loading ESM2 …')
    models_dict['ESM2'] = load_esm2(device)

    print('Loading ProstT5 …')
    models_dict['ProstT5'] = load_prostt5(device)

    if args.seqdance_path:
        print('Loading SeqDance …')
        models_dict['SeqDance'] = load_seqdance(args.seqdance_path, device)
    else:
        print('SeqDance: --seqdance_path not provided, skipping.')

    # ── Step 3: (disabled) protein-protein RSA ────────────────────────────────
    # print('\n=== Protein-level RSA: pooled embedding similarity vs descriptor similarity ===')
    # cache, valid_proteins = build_pooled_embedding_cache(proteins, models_dict, device)
    # rsa_results = compute_rsa_scores(valid_proteins, cache, n_permutations=1000)
    # plot_rsa_bar(rsa_results, args.output_dir)

    # ── Step 3b: DCCM correlation ──────────────────────────────────────────────
    print('\n=== Computing per-residue embeddings (for DCCM correlation) ===')
    residue_cache, valid_proteins = build_embedding_cache(proteins, models_dict, device)
    if not valid_proteins:
        print('No valid proteins after per-residue embedding. Exiting.')
        return

    print('\n=== Residue-pair embedding similarity vs DCCM ===')
    plot_dccm_corr_boxplot(valid_proteins, residue_cache, args.analysis_path,
                          args.output_dir, dccm_dir=args.dccm_dir,
                          replicate=args.dccm_replicate)

    print(f'\nAll figures saved to: {args.output_dir}')


if __name__ == '__main__':
    main()

'''

# ── Example run ───────────────────────────────────────────────────────────────
PYTHONPATH=. python evaluate/methodology/protein_level_emb_md.py \
    --data_path     /path/to/DPLM_data/Atlas_geom2vec_test/ \
    --analysis_path /path/to/DPLM_data/analysis/ \
    --dplm_config   ./configs/config_vivit5_delta.yaml \
    --dplm_checkpoint ./results/vivit5/checkpoints/checkpoint_best_val_whole_loss.pth \
    --seqdance_path <SeqDance-main/model> \
    --output_dir    ./evaluate/methodology/protein_level_output/ \
    --max_proteins  30

PYTHONPATH=. python evaluate/methodology/protein_level_emb_md.py \
    --data_path     /path/to/DPLM_data/processed_test_rep0/ \
    --analysis_path /path/to/DPLM_data/analysis/ \
    --dplm_config   ./results/vivit_info2/config_vivit_comp_v2.yaml \
    --dplm_checkpoint ./results/vivit_info2/checkpoints/checkpoint_best_val_whole_loss.pth \
    --seqdance_path <SeqDance-main/model> \
    --output_dir    ./evaluate/methodology/protein_level_output/vivit_info2/ \
    --dccm_dir      /path/to/DPLM_data/DCCM_dir \
    --dccm_replicate R1 \



'''