"""
phase_sep_viz.py — Unsupervised clustering and embedding visualization for
                   phase-separation protein datasets (tableS1–S5).

Compares embeddings from DPLM, ESM2, ProstT5 and (optionally) SeqDance via:
  - t-SNE (PCA pre-reduction to 50 dims, TSNE perplexity=30)
  - K-means (k=2) clustering with Adjusted Rand Index vs. true labels
  - Individual figures per method: one plot coloured by true class labels and one
    coloured by k-means clusters (ARI in the title).

--model_type selects the model: d-plm (loads the ESM2 baseline too), ESM2, or prostt5
(each of the latter two runs that model alone). --seqdance_path adds SeqDance on top.

Usage
-----
# ProstT5 only (no checkpoint needed), tableS1
PYTHONPATH=. python evaluate/Phase_sep/phase_sep_viz.py \\
    --model_type prostt5 \\
    --data_dir ./evaluate/Phase_sep/ \\
    --output_dir ./evaluate/Phase_sep/results_prostt5/ \\
    --tables S1

# ESM2 only (no checkpoint needed), tableS1
PYTHONPATH=. python evaluate/Phase_sep/phase_sep_viz.py \\
    --model_type ESM2 \\
    --data_dir ./evaluate/Phase_sep/ \\
    --output_dir ./evaluate/Phase_sep/results/ \\
    --tables S1

# DPLM + ESM2 + SeqDance on all five tables
PYTHONPATH=. python evaluate/Phase_sep/phase_sep_viz.py \\
    --model_type d-plm \\
    --checkpoint_path ./results/vivit5/checkpoints/checkpoint_best_val_whole_loss.pth \\
    --config_path ./configs/config_vivit5_delta.yaml \\
    --seqdance_path /path/to/SeqDance-main/model/ \\
    --data_dir ./evaluate/Phase_sep/ \\
    --output_dir ./evaluate/Phase_sep/results/ \\
    --tables S1 S2 S3 S4 S5 \\
    --save_emb
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import yaml
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

# Project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'methodology'))

import esm
import esm_adapterH
from utils.utils import load_configs, load_dplm_checkpoint
# Reuse the canonical ProstT5 loader + per-residue extractor (same as rmsf.py, the ddg_mega /
# ddg_S669 baselines and phase_separation_xgboost_prostt5.py all use) instead of duplicating it.
# SPLM-V2-GVP: runs out-of-process (its packages collide with DPLM_ai's) — see splm_embed.
sys.path.insert(0, os.path.dirname(__file__))

warnings.filterwarnings('ignore')


# ────────────────────────────────────────────────────────────────────────────
# 1.  Model loading
# ────────────────────────────────────────────────────────────────────────────



def _load_dplm(checkpoint_path, config_path, device):
    """Load DPLM (ESM2 + Houlsby adapters from checkpoint)."""
    with open(config_path) as f:
        config_file = yaml.full_load(f)
    configs = load_configs(config_file, args=None)

    if configs.model.esm_encoder.adapter_h.enable:
        model, alphabet = esm_adapterH.pretrained.esm2_t33_650M_UR50D(
            configs.model.esm_encoder.adapter_h)
    else:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    load_dplm_checkpoint(model, checkpoint_path)
    model.eval().to(device)
    print(f'[DPLM] checkpoint loaded: {checkpoint_path}')
    return model, alphabet




# ────────────────────────────────────────────────────────────────────────────
# 2.  Sequence encoding
# ────────────────────────────────────────────────────────────────────────────

def _encode_esm(model, alphabet, sequences, device, batch_size=8):
    """Mean-pool ESM2/DPLM residue embeddings (excl. CLS and EOS). Returns [N, 1280]."""
    batch_converter = alphabet.get_batch_converter()
    all_embs = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        data = [(f'p{j}', s) for j, s in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
        reps = out['representations'][model.num_layers]   # [B, L+2, D]
        for b in range(reps.shape[0]):
            mask = ((tokens[b] != alphabet.padding_idx) &
                    (tokens[b] != alphabet.cls_idx) &
                    (tokens[b] != alphabet.eos_idx))
            all_embs.append(reps[b][mask].mean(0).cpu().numpy())
    return np.stack(all_embs, axis=0)   # [N, 1280]






# ────────────────────────────────────────────────────────────────────────────
# 3.  Data loading — two Excel formats
# ────────────────────────────────────────────────────────────────────────────

def load_s1_format(path):
    """Parse tableS1, S3, S4, S5 — col[0]=class, col[1]=FASTA-style entries.

    Returns (sequences, label_names, labels) as numpy arrays.
    """
    df = pd.read_excel(path, header=None)
    records, current_class, current_id = [], None, None
    for _, row in df.iterrows():
        if pd.notna(row[0]):
            current_class = str(row[0]).strip()
        if pd.notna(row[1]):
            val = str(row[1]).strip()
            if val.startswith('>'):
                current_id = val.split()[0][1:]
            elif current_class is not None:
                records.append({'class': current_class, 'id': current_id, 'sequence': val})
    data = pd.DataFrame(records)
    sequences   = data['sequence'].values
    label_names = data['class'].values
    labels      = (data['class'] == 'Positive').astype(int).values
    print(f'  {os.path.basename(path)}: {len(sequences)} sequences '
          f'(pos={labels.sum()}, neg={(labels==0).sum()})')
    return sequences, label_names, labels


def load_s2_format(path):
    """Parse tableS2 — paired rows, 4-way labels (Pos/Neg × Disordered/Folded).

    Returns (sequences, label_names, combined_labels, labels).
    """
    df = pd.read_excel(path, header=None)
    df = df.iloc[1:].reset_index(drop=True)
    records = []
    for i in range(0, len(df) - 1, 2):
        row_a, row_b = df.iloc[i], df.iloc[i + 1]
        records.append({
            'class':    str(row_a[0]).strip() if pd.notna(row_a[0]) else None,
            'subclass': str(row_a[1]).strip() if pd.notna(row_a[1]) else '',
            'sequence': str(row_b[2]).strip() if pd.notna(row_b[2]) else '',
        })
    data = pd.DataFrame(records).dropna(subset=['class'])
    data = data[data['sequence'] != '']
    sequences       = data['sequence'].values
    label_names     = data['class'].values
    combined_labels = np.array([f"{c} / {s}" for c, s in
                                zip(data['class'], data['subclass'])])
    labels          = (data['class'] == 'Positive').astype(int).values
    print(f'  {os.path.basename(path)}: {len(sequences)} sequences')
    for lbl, cnt in zip(*np.unique(combined_labels, return_counts=True)):
        print(f'    {lbl}: {cnt}')
    return sequences, label_names, combined_labels, labels


# ────────────────────────────────────────────────────────────────────────────
# 4.  Dimensionality reduction
# ────────────────────────────────────────────────────────────────────────────

def _reduce_tsne(emb_np, n_pca=50, perplexity=30, random_state=42):
    """StandardScaler → PCA(50) → TSNE(perplexity=30).
    Returns (emb2d [N,2], emb_scaled [N,D]).
    """
    scaled  = StandardScaler().fit_transform(emb_np)
    n_pca   = min(n_pca, scaled.shape[0] - 1, scaled.shape[1])
    pca_out = PCA(n_components=n_pca, random_state=random_state).fit_transform(scaled)
    emb2d   = TSNE(n_components=2, perplexity=perplexity,
                   random_state=random_state, max_iter=1000).fit_transform(pca_out)
    return emb2d, scaled


def _cluster_ari(scaled, labels, rounds=1, base_seed=42, n_init=10):
    """K-means (k=2) ARI vs true labels.

    The embeddings are deterministic and KMeans(random_state) is deterministic, so the ARI
    itself has NO run-to-run randomness. To estimate a variance we vary the KMeans init seed
    across `rounds` (the one legitimately stochastic knob), measuring how sensitive the ARI is
    to clustering initialisation.

    Returns (ref_cluster, ari_mean, ari_std, ari_list). ref_cluster (from base_seed) is used
    for the plot colouring so figures stay reproducible.
    """
    aris, ref_cluster = [], None
    for r in range(max(1, rounds)):
        km = KMeans(n_clusters=2, random_state=base_seed + r, n_init=n_init)
        cluster = km.fit_predict(scaled)
        aris.append(adjusted_rand_score(labels, cluster))
        if r == 0:
            ref_cluster = cluster
    aris = np.array(aris, dtype=float)
    return ref_cluster, float(aris.mean()), float(aris.std()), aris


# ────────────────────────────────────────────────────────────────────────────
# 5.  Plotting
# ────────────────────────────────────────────────────────────────────────────

def _scatter_panel(ax, emb2d, color_vals, palette, title, subtitle):
    """Colored scatter with legend — identical to notebook helper."""
    for val in list(dict.fromkeys(color_vals)):
        mask = np.array(color_vals) == val
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                   color=palette[val], alpha=0.75, s=35,
                   edgecolors='white', linewidths=0.4, label=str(val))
    ax.set_title(title, fontsize=12, fontweight='bold', color='#2d2d2d', pad=16)
    ax.text(0.5, 1.025, subtitle, transform=ax.transAxes,
            ha='center', fontsize=9.5, color='#555555')
    ax.set_xlabel('t-SNE 1', fontsize=9)
    ax.set_ylabel('t-SNE 2', fontsize=9)
    ax.legend(fontsize=8.5, framealpha=0.88, loc='best')
    ax.set_facecolor('#FFFFFF')
    for sp in ax.spines.values():
        sp.set_edgecolor('#CCCCCC')
    ax.tick_params(labelsize=8)


def _save_single_panel(table_name, method, kind, emb2d, color_vals,
                       palette, subtitle, output_dir):
    """Draw ONE scatter panel to its own figure and save it."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    fig.patch.set_facecolor('#F8F9FA')
    _scatter_panel(ax, emb2d, color_vals, palette, method, subtitle)
    os.makedirs(output_dir, exist_ok=True)
    safe_method = str(method).replace(' ', '_').replace('/', '_')
    out = os.path.join(output_dir,
                       f'embedding_{table_name}_{safe_method}_{kind}.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[Plot] saved → {out}')


def _plot_table(table_name, panel_data, output_dir):
    """Save each panel individually (no grouped grid).

    For every method, two standalone figures are written:
      • {..}_truelabels.png — coloured by ground-truth class
      • {..}_kmeans.png      — coloured by k-means cluster (title shows ARI)

    panel_data: list of dicts, each with keys:
        method, emb2d, cluster, ari, label_names, labels, combined_labels
    """
    is_s2 = (table_name == 'S2')

    BINARY_COLORS   = {'Negative': '#4C72B0', 'Positive': '#DD8452'}
    COMBINED_COLORS = {
        'Negative / Disordered': '#4C72B0',
        'Negative / Folded':     '#55A868',
        'Positive / Disordered': '#DD8452',
        'Positive / Folded':     '#C44E52',
    }
    CLUSTER_COLORS = {'Cluster 0': '#2ca02c', 'Cluster 1': '#d62728'}

    for pd_ in panel_data:
        method  = pd_['method']
        emb2d   = pd_['emb2d']
        cluster = pd_['cluster']
        ari     = pd_['ari']
        ari_std = pd_.get('ari_std', 0.0)

        # True-label panel
        if is_s2 and pd_['combined_labels'] is not None:
            row0_vals    = pd_['combined_labels']
            row0_palette = COMBINED_COLORS
        else:
            row0_vals    = pd_['label_names']
            row0_palette = BINARY_COLORS
        _save_single_panel(table_name, method, 'truelabels',
                           emb2d, row0_vals, row0_palette,
                           'True class labels', output_dir)

        # K-means cluster panel
        cluster_vals = [f'Cluster {c}' for c in cluster]
        ari_str = f'ARI = {ari:.3f} ± {ari_std:.3f}' if ari_std > 0 else f'ARI = {ari:.3f}'
        _save_single_panel(table_name, method, 'kmeans',
                           emb2d, cluster_vals, CLUSTER_COLORS,
                           f'K-means (k=2)  |  {ari_str}', output_dir)


# ────────────────────────────────────────────────────────────────────────────
# 6.  Per-table analysis
# ────────────────────────────────────────────────────────────────────────────

def analyze_table(table_name, data_dir, models_dict, dance_model, tokenizer,
                  device, output_dir, batch_size=8, save_emb=False,
                  prostt5_pair=None, splm_cfg=None, ari_rounds=1, esmc_pair=None):
    """Load one table, embed with the selected models, t-SNE, cluster, plot.

    models_dict : {'DPLM': (model, alphabet), 'ESM2': (model, alphabet)} — may be empty.
    dance_model : SeqDance model or None.
    prostt5_pair: (model, tokenizer) for ProstT5, or None.
    esmc_pair   : (model, tokenizer) for ESM-C, or None.
    splm_cfg    : argparse.Namespace with splm_path/splm_config/splm_checkpoint/… or None.
    """
    fpath = os.path.join(data_dir, f'table{table_name}.xlsx')
    if not os.path.exists(fpath):
        print(f'[{table_name}] not found: {fpath} — skipping.')
        return

    print(f'\n=== Table {table_name} ===')
    if table_name == 'S2':
        sequences, label_names, combined_labels, labels = load_s2_format(fpath)
    else:
        sequences, label_names, labels = load_s1_format(fpath)
        combined_labels = None

    panel_data = []

    # ── ESM2-style models (DPLM, ESM2) ───────────────────────────────────
    for method_name, (mdl, alph) in models_dict.items():
        print(f'  [{method_name}] encoding {len(sequences)} sequences …')
        emb = _encode_esm(mdl, alph, list(sequences), device, batch_size)
        if save_emb:
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, f'{table_name}_{method_name}_emb.npy'), emb)

        emb2d, scaled = _reduce_tsne(emb)
        cluster, ari, ari_std, _ = _cluster_ari(scaled, labels, rounds=ari_rounds)
        print(f'  [{method_name}] ARI = {ari:.4f} ± {ari_std:.4f}  (rounds={ari_rounds})')

        panel_data.append(dict(
            method=method_name, emb2d=emb2d, cluster=cluster, ari=ari, ari_std=ari_std,
            label_names=label_names, labels=labels, combined_labels=combined_labels,
        ))

    # ── ProstT5 ──────────────────────────────────────────────────────────

    # ── ESM-C ────────────────────────────────────────────────────────────
    # Kept separate from models_dict because ESM-C uses a HF-style tokenizer, not a
    # fair-esm alphabet/batch_converter, so _encode_esm does not apply.

    # ── SPLM ─────────────────────────────────────────────────────────────

    # ── SeqDance ─────────────────────────────────────────────────────────

    if panel_data:
        _plot_table(table_name, panel_data, output_dir)


# ────────────────────────────────────────────────────────────────────────────
# 7.  CLI + main
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Phase separation t-SNE visualization: DPLM vs ESM2 vs SeqDance.')
    p.add_argument('--model_type',      default='d-plm',
                   choices=['d-plm', 'ESM2', 'esmc', 'prostt5', 'splm'],
                   help='Model to evaluate (default: d-plm). d-plm additionally loads the '
                        'ESM2 baseline; ESM2 / prostt5 / splm run that model alone.')
    p.add_argument('--checkpoint_path', default=None,
                   help='DPLM checkpoint .pth (required when model_type=d-plm)')
    p.add_argument('--config_path',     default=None,
                   help='Training config YAML (required when model_type=d-plm)')
    p.add_argument('--data_dir',        default='./evaluate/Phase_sep/',
                   help='Directory containing tableS1.xlsx … tableS5.xlsx')
    p.add_argument('--output_dir',      default='./evaluate/Phase_sep/results/',
                   help='Output directory for PNG figures')
    p.add_argument('--tables',          nargs='+',
                   default=['S1', 'S2', 'S3', 'S4', 'S5'],
                   help='Which tables to process (default: all five)')
    p.add_argument('--batch_size',      type=int, default=8,
                   help='Encoding batch size for ESM2/DPLM (default: 8)')
    p.add_argument('--save_emb',        action='store_true',
                   help='Save embeddings as .npy files in output_dir for reuse')
    p.add_argument('--ari_rounds',      type=int, default=1,
                   help='K-means runs (each with a different init seed) to estimate ARI '
                        'mean ± std. The embeddings + KMeans are otherwise deterministic, so '
                        'this measures ARI sensitivity to clustering init. Default: 1 (no std).')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load the selected model(s) ─────────────────────────────────────────
    print('\n=== Loading models ===')
    models_dict = {}          # ESM-style backbones only
    prostt5_pair = None
    esmc_pair = None
    splm_cfg = None

    # DPLM only. The baseline encoders (ESM2, ESM-C, ProstT5, SPLM) that this script
    # originally compared against are not part of the public release.
    assert args.checkpoint_path and args.config_path, \
        '--checkpoint_path and --config_path are required.'
    dplm_model, dplm_alphabet = _load_dplm(args.checkpoint_path, args.config_path, device)
    models_dict['DPLM'] = (dplm_model, dplm_alphabet)

    dance_model, tokenizer = None, None      # SeqDance is a baseline; not shipped

    # ── Process each table ─────────────────────────────────────────────────
    for table_name in args.tables:
        analyze_table(
            table_name, args.data_dir,
            models_dict, dance_model, tokenizer,
            device, args.output_dir,
            batch_size=args.batch_size,
            save_emb=args.save_emb,
            prostt5_pair=prostt5_pair,
            esmc_pair=esmc_pair,
            splm_cfg=splm_cfg,
            ari_rounds=args.ari_rounds,
        )

    print('\nDone.')


if __name__ == '__main__':
    main()

"""
PYTHONPATH=. python evaluate/Phase_sep/phase_sep_viz.py \
--checkpoint_path /work/nvme/bcnr/jyx/DPLM_ai/results/vivit3/checkpoints/checkpoint_best_val_rmsf_cor.pth \
--config_path /work/nvme/bcnr/jyx/DPLM_ai/results/vivit3/config_vivit3.yaml \
--seqdance_path /work/nvme/bcnr/jyx/dplm/SeqDance-main/SeqDance-main/model/ \
--data_dir /scratch/bcnr/yjiang12/DPLM_data/Phase_sep/ \
--output_dir ./evaluate/Phase_sep/results/ \
--batch_size 1

PYTHONPATH=. python evaluate/Phase_sep/phase_sep_viz.py \\
    --model_type d-plm \\
    --checkpoint_path ./results/vivit5/checkpoints/checkpoint_best_val_whole_loss.pth \\
    --config_path ./configs/config_vivit5_delta.yaml \\
    --seqdance_path /path/to/SeqDance-main/model/ \\
    --data_dir ./evaluate/Phase_sep/ \\
    --output_dir ./evaluate/Phase_sep/results/ \\
    --tables S1 S2 S3 S4 S5 \\
    --save_emb
    --batch_size 1

"""