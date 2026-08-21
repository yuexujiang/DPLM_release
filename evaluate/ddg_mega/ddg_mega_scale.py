"""
ddg_mega_scale.py — Per-protein ddG prediction on the Mega-Scale stability dataset.

For each protein in the Mega-Scale dataset (Tsuboyama et al., 2023):
  1. Encode the wild-type sequence → mean-pooled ESM2/DPLM embedding [1280-dim].
  2. Batch-encode all single-mutant sequences → [N_muts, 1280].
  3. Feature = mean_pool(mutant) − mean_pool(wildtype)  [1280-dim].
     (Equivalent to the original residue-level mean(mt − wt) since mean is linear
      and sequence lengths are identical after filtering out indels.)
  4. 50/50 random train/test split per protein.
  5. Fit LinearRegression on the train half; evaluate Spearman r and MAE on test.
Report median per-protein Spearman and MAE across all proteins, and save a boxplot.

Usage
-----
# ESM2 baseline (no checkpoint needed)
PYTHONPATH=. python evaluate/ddg_mega/ddg_mega_scale.py \\
    --csv_path  /path/to/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \\
    --model_type ESM2 \\
    --output_dir ./evaluate/ddg_mega/results_esm2

# DPLM (fine-tuned ESM2 + Houlsby adapters)
PYTHONPATH=. python evaluate/ddg_mega/ddg_mega_scale.py \\
    --csv_path  /path/to/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \\
    --model_type d-plm \\
    --config_path    ./configs/config_vivit5_delta.yaml \\
    --checkpoint_path ./results/vivit5/checkpoints/checkpoint_best_val_whole_loss.pth \\
    --output_dir ./evaluate/ddg_mega/results_dplm
"""

import os
import re
import sys
import pickle
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from tqdm import tqdm

# Project root on path so we can import utils, esm, esm_adapterH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import esm
import esm_adapterH
from utils.utils import load_configs, load_esm2_checkpoint

warnings.filterwarnings('ignore')


# ────────────────────────────────────────────────────────────────────────────
# 1.  Data loading
# ────────────────────────────────────────────────────────────────────────────

def load_dataset(csv_path):
    """Load and pre-process the Mega-Scale stability CSV.

    Filters to entries with a numeric ddG_ML value and no indels.
    Averages ddG_ML across replicate experiments on the same sequence.
    Returns a DataFrame indexed by '{WT_name}${mut_type}'.
    """
    df = pd.read_csv(csv_path)

    # Keep rows with a real ddG value and no insertion/deletion mutations
    df = df[(df['ddG_ML'] != '-') & (~df['mut_type'].str.contains('ins|del', na=False))]
    df['ddG_ML'] = df['ddG_ML'].astype(float)

    # Average ddG across replicate experiments on the same sequence
    ddg_mean = df[['aa_seq', 'ddG_ML']].groupby('aa_seq').mean()
    df = pd.merge(
        df.sort_values('mut_type', ascending=False)
          .drop_duplicates('aa_seq')[['aa_seq', 'mut_type', 'WT_name', 'WT_cluster']],
        ddg_mean,
        on='aa_seq',
        how='left',
    )
    df.index = df['WT_name'] + '$' + df['mut_type']

    n_proteins = df['WT_name'].nunique()
    print(f'[Dataset] {len(df)} entries, {n_proteins} proteins after filtering.')
    return df


# ────────────────────────────────────────────────────────────────────────────
# 2.  Model loading
# ────────────────────────────────────────────────────────────────────────────

def _load_model(args, device):
    """Return (model, alphabet) for DPLM (fine-tuned) or plain ESM2.

    Follows the same adapter_h / lora / plain branching as MD_emb_eval.py
    and ablation.py, but does not load the MoBYMLP projector (not needed here).
    """
    if args.model_type == 'd-plm':
        assert args.config_path and args.checkpoint_path, \
            '--config_path and --checkpoint_path are required for d-plm.'

        with open(args.config_path) as f:
            config_file = yaml.full_load(f)
        configs = load_configs(config_file, args=None)

        if configs.model.esm_encoder.adapter_h.enable:
            model, alphabet = esm_adapterH.pretrained.esm2_t33_650M_UR50D(
                configs.model.esm_encoder.adapter_h)
        elif configs.model.esm_encoder.lora.enable:
            from peft import LoraConfig, get_peft_model
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            lora_targets = ['self_attn.q_proj', 'self_attn.k_proj',
                            'self_attn.v_proj', 'self_attn.out_proj']
            target_modules = []
            if configs.model.esm_encoder.lora.esm_num_end_lora > 0:
                start = max(model.num_layers
                            - configs.model.esm_encoder.lora.esm_num_end_lora, 0)
                for idx in range(start, model.num_layers):
                    for name in lora_targets:
                        target_modules.append(f'layers.{idx}.{name}')
            peft_config = LoraConfig(
                inference_mode=False,
                r=configs.model.esm_encoder.lora.r,
                lora_alpha=configs.model.esm_encoder.lora.alpha,
                target_modules=target_modules,
                lora_dropout=configs.model.esm_encoder.lora.dropout,
                bias='none',
            )
            model = get_peft_model(model, peft_config)
        else:
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        load_esm2_checkpoint(model, args.checkpoint_path)

    else:  # 'ESM2' — plain pretrained model, no fine-tuning
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        print('[ESM2 base] loaded (no fine-tuning)')

    model.eval().to(device)
    return model, alphabet


# ────────────────────────────────────────────────────────────────────────────
# 3.  Sequence encoding
# ────────────────────────────────────────────────────────────────────────────

def _encode_sequences(model, alphabet, sequences, device, batch_size=8):
    """Mean-pool ESM2/DPLM embeddings over residues, excluding CLS and EOS.

    Args:
        sequences : list of str
    Returns:
        np.ndarray [N, 1280]
    """
    batch_converter = alphabet.get_batch_converter()
    all_embs = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        data = [(f'p{j}', s) for j, s in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens,
                            repr_layers=[model.num_layers],
                            return_contacts=False)
        reps = results['representations'][model.num_layers]   # [B, L+2, 1280]

        for b in range(reps.shape[0]):
            mask = ((tokens[b] != alphabet.padding_idx) &
                    (tokens[b] != alphabet.cls_idx) &
                    (tokens[b] != alphabet.eos_idx))
            emb = reps[b][mask].mean(0).cpu().numpy()         # [1280]
            all_embs.append(emb)

    return np.stack(all_embs, axis=0)   # [N, 1280]


# ────────────────────────────────────────────────────────────────────────────
# 4.  Per-protein feature extraction
# ────────────────────────────────────────────────────────────────────────────

_MUT_RE = re.compile(r'^([A-Za-z])(\d+)([A-Za-z])$')


def parse_mut_type(mut_type):
    """'D1Q' → (wt_aa='D', position=1 (1-based), mt_aa='Q'). ('', None, '') if not a single sub."""
    m = _MUT_RE.match(str(mut_type).strip())
    if not m:
        return '', None, ''
    return m.group(1).upper(), int(m.group(2)), m.group(3).upper()


def _compute_features(df, pro, model, alphabet, device, batch_size):
    """Encode WT + all mutants for protein `pro` and build difference features.

    Returns:
        mut_labels : list[str]          mutation labels (excluding 'wt')
        X          : np.ndarray [N, 1280]  feature = mean_pool(mut) − mean_pool(wt)
        y          : np.ndarray [N]        ddG_ML values
    Returns ([], None, None) if the protein has no mutations or is missing WT.
    """
    wt_key = f'{pro}$wt'
    if wt_key not in df.index:
        return [], None, None

    wt_seq = df.loc[wt_key, 'aa_seq']
    wt_emb = _encode_sequences(model, alphabet, [wt_seq],
                                device=device, batch_size=1)[0]   # [1280]

    mut_labels = [m for m in df[df['WT_name'] == pro]['mut_type'].values
                  if m != 'wt']
    if not mut_labels:
        return [], None, None

    # Batch-encode all mutant sequences at once
    mut_seqs = [df.loc[f'{pro}${mt}', 'aa_seq'] for mt in mut_labels]
    mut_embs = _encode_sequences(model, alphabet, mut_seqs,
                                  device=device, batch_size=batch_size)  # [N, 1280]

    X = mut_embs - wt_emb[np.newaxis, :]   # broadcast-subtract WT; [N, 1280]
    y = np.array([df.loc[f'{pro}${mt}', 'ddG_ML'] for mt in mut_labels])
    return mut_labels, X, y


# ────────────────────────────────────────────────────────────────────────────
# 5.  Per-protein evaluation loop
# ────────────────────────────────────────────────────────────────────────────

def evaluate_proteins(df, model, alphabet, device,
                      max_proteins=None, seed=42, batch_size=8, save_predictions=False):
    """50/50 train/test split per protein; fit LinearRegression; collect metrics.

    Returns:
        per_corr  : list[float]  per-protein test Spearman r
        per_mae   : list[float]  per-protein test MAE (kcal/mol)
        models    : dict {WT_name: LinearRegression}  the learnt per-protein regressors
        pred_rows : list[dict]   per test-half mutation (empty unless save_predictions)
    """
    rng = np.random.default_rng(seed)

    proteins = sorted(df['WT_name'].unique())
    if max_proteins is not None:
        proteins = proteins[:max_proteins]

    per_corr, per_mae = [], []
    models, pred_rows = {}, []

    for pro in tqdm(proteins, desc='Proteins'):
        mut_labels, X, y = _compute_features(
            df, pro, model, alphabet, device, batch_size)

        if X is None or len(X) < 4:   # need ≥ 2 train + 2 test samples
            continue

        # Shuffle and split 50/50
        idx = rng.permutation(len(X))
        n_train = len(idx) // 2
        tr, te = idx[:n_train], idx[n_train:]
        X_train, y_train = X[tr], y[tr]
        X_test,  y_test  = X[te], y[te]

        reg = LinearRegression()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        corr, _ = spearmanr(y_test, y_pred)
        mae      = mean_absolute_error(y_test, y_pred)

        if not np.isnan(corr):
            per_corr.append(corr)
            per_mae.append(mae)
            models[pro] = reg
            if save_predictions:
                for j, ti in enumerate(te):
                    wt_aa, pos, mt_aa = parse_mut_type(mut_labels[ti])
                    pred_rows.append(dict(
                        protein_id=pro, mut_type=mut_labels[ti], position=pos,
                        wt_aa=wt_aa, mt_aa=mt_aa,
                        ddG=float(y[ti]), prediction=float(y_pred[j])))

    print(f'\n[Evaluation] {len(per_corr)} proteins evaluated successfully.')
    return per_corr, per_mae, models, pred_rows


# ────────────────────────────────────────────────────────────────────────────
# 5b.  Site-aware feature (mirrors ddg_S669 model_ddg_v2.EncoderSiteAware)
# ────────────────────────────────────────────────────────────────────────────

def _encode_residues_esm(model, alphabet, seq, device):
    """Per-residue ESM2/DPLM embedding [L, D] for ONE sequence (CLS/EOS/pad stripped),
    so index r == residue r (matching rmsf's ProstT5/SeqDance extractors)."""
    bc = alphabet.get_batch_converter()
    _, _, tokens = bc([('p', str(seq))])
    tokens = tokens.to(device)
    with torch.no_grad():
        reps = model(tokens, repr_layers=[model.num_layers],
                     return_contacts=False)['representations'][model.num_layers]
    mask = ((tokens[0] != alphabet.padding_idx) &
            (tokens[0] != alphabet.cls_idx) &
            (tokens[0] != alphabet.eos_idx))
    return reps[0][mask].cpu().numpy()          # [L, D]


def mut_site_feature(res_wt, res_mt, pos):
    """Site-aware feature for one mutation from per-residue arrays [L, D].

    Returns [site_to−site_from, global_to−global_from, site_to, site_from] (4·D), mirroring
    ddg_S669 EncoderSiteAware. Falls back to the global mean at the site when `pos` is missing
    or out of range (e.g. a multi-site or unparsable mut_type).
    """
    gw = res_wt.mean(0)
    gm = res_mt.mean(0)
    i = (pos - 1) if pos else None
    sw = res_wt[i] if (i is not None and 0 <= i < len(res_wt)) else gw
    sm = res_mt[i] if (i is not None and 0 <= i < len(res_mt)) else gm
    return np.concatenate([sm - sw, gm - gw, sm, sw]).astype(np.float32)


def evaluate_proteins_site(df, residue_encodes, device, max_proteins=None, seed=42,
                           save_predictions=False):
    """Site-aware variant of evaluate_proteins, generalized over a LIST of per-backbone
    residue encoders (each: seq -> [L, D]). Each backbone contributes a 4·D_i site feature;
    they are concatenated. Same 50/50 split + LinearRegression + Spearman/MAE protocol.
    """
    rng = np.random.default_rng(seed)
    proteins = sorted(df['WT_name'].unique())
    if max_proteins is not None:
        proteins = proteins[:max_proteins]

    per_corr, per_mae, models, pred_rows = [], [], {}, []
    for pro in tqdm(proteins, desc='Proteins'):
        wt_key = f'{pro}$wt'
        if wt_key not in df.index:
            continue
        mut_labels = [m for m in df[df['WT_name'] == pro]['mut_type'].values if m != 'wt']
        if not mut_labels:
            continue
        wt_seq = df.loc[wt_key, 'aa_seq']
        res_wt = [renc(wt_seq) for renc in residue_encodes]      # per-backbone WT [L, D]

        feats, y, kept = [], [], []
        for mt in mut_labels:
            seq = df.loc[f'{pro}${mt}', 'aa_seq']
            _, pos, _ = parse_mut_type(mt)
            res_mt = [renc(seq) for renc in residue_encodes]
            f = np.concatenate([mut_site_feature(res_wt[b], res_mt[b], pos)
                                for b in range(len(residue_encodes))])
            feats.append(f); y.append(df.loc[f'{pro}${mt}', 'ddG_ML']); kept.append(mt)
        if len(feats) < 4:
            continue
        X = np.stack(feats, axis=0); y = np.array(y)

        idx = rng.permutation(len(X))
        n_train = len(idx) // 2
        tr, te = idx[:n_train], idx[n_train:]
        reg = LinearRegression().fit(X[tr], y[tr])
        y_pred = reg.predict(X[te])
        corr, _ = spearmanr(y[te], y_pred)
        if np.isnan(corr):
            continue
        per_corr.append(corr)
        per_mae.append(mean_absolute_error(y[te], y_pred))
        models[pro] = reg
        if save_predictions:
            for j, ti in enumerate(te):
                wt_aa, pos, mt_aa = parse_mut_type(kept[ti])
                pred_rows.append(dict(
                    protein_id=pro, mut_type=kept[ti], position=pos,
                    wt_aa=wt_aa, mt_aa=mt_aa,
                    ddG=float(y[ti]), prediction=float(y_pred[j])))

    print(f'\n[Evaluation/site] {len(per_corr)} proteins evaluated successfully.')
    return per_corr, per_mae, models, pred_rows


# ────────────────────────────────────────────────────────────────────────────
# 6.  Plotting helpers + plot
# ────────────────────────────────────────────────────────────────────────────

def _compute_five_number(data):
    """Return (whislo, q1, med, q3, whishi) matching matplotlib's default 1.5*IQR whiskers.

    Equivalent to the "Min/Max w/o outliers" values shown in box-plot summaries.
    """
    arr = np.array(data)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    whislo = float(arr[arr >= q1 - 1.5 * iqr].min())
    whishi = float(arr[arr <= q3 + 1.5 * iqr].max())
    return whislo, q1, med, q3, whishi


def plot_results(per_corr, per_mae, output_dir, model_label='Model'):
    """Multi-method boxplots: reference methods (pre-computed stats) + evaluated model.

    Reference method statistics (Min w/o outliers, Q1, Median, Q3, Max w/o outliers)
    are hard-coded from prior evaluations on the same Mega-Scale dataset.
    Jittered individual points are shown only for the evaluated model (actual data available).
    """
    # ── Pre-computed reference stats ─────────────────────────────────────────
    # [whislo, q1, median, q3, whishi]  (i.e. min/max without outliers, Q1, Q2, Q3)
    REF_CORR = {
        'SeqDance': [0.17, 0.37, 0.44, 0.53, 0.74],
        'ESMDance': [0.33, 0.51, 0.58, 0.66, 0.84],
        'ESM2':     [0.08, 0.40, 0.56, 0.65, 0.88],
    }
    REF_MAE = {
        'SeqDance': [0.22, 0.51, 0.63, 0.78, 1.15],
        'ESMDance': [0.21, 0.44, 0.54, 0.67, 0.99],
        'ESM2':     [0.26, 0.44, 0.56, 0.66, 0.96],
    }

    # Box colors: reference methods (muted) + evaluated model (highlighted)
    METHOD_COLORS = {
        'SeqDance': '#b2b2b2',   # medium gray
        'ESMDance': '#a6cee3',   # light blue
        'ESM2':     '#fdbf6f',   # light orange
    }
    MODEL_COLOR = '#2166ac'      # dark blue — evaluated model

    ref_names  = list(REF_CORR.keys())          # ['SeqDance', 'ESMDance', 'ESM2']
    all_names  = ref_names + [model_label]       # reference + evaluated model
    n_methods  = len(all_names)
    model_pos  = n_methods                       # 1-based position of evaluated model

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    panels = [
        (per_corr, REF_CORR, 'Per-protein Spearman r',    'Spearman r'),
        (per_mae,  REF_MAE,  'Per-protein MAE (kcal/mol)', 'MAE (kcal/mol)'),
    ]

    for ax, (data, ref_stats, title, ylabel) in zip(axes, panels):
        # Build bxp stats list: reference methods first, then evaluated model
        stats_list = []
        for name in ref_names:
            w_lo, q1, med, q3, w_hi = ref_stats[name]
            stats_list.append(dict(med=med, q1=q1, q3=q3,
                                   whislo=w_lo, whishi=w_hi,
                                   fliers=[], label=name))

        w_lo, q1, med, q3, w_hi = _compute_five_number(data)
        stats_list.append(dict(med=med, q1=q1, q3=q3,
                               whislo=w_lo, whishi=w_hi,
                               fliers=[], label=model_label))

        bp = ax.bxp(
            stats_list,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=2),
            widths=0.5,
        )

        # Colour each box
        for patch, name in zip(bp['boxes'], all_names):
            color = MODEL_COLOR if name == model_label \
                    else METHOD_COLORS.get(name, '#cccccc')
            patch.set_facecolor(color)
            patch.set_alpha(0.75 if name == model_label else 0.65)

        # Colour whiskers and caps to match their box
        for i, name in enumerate(all_names):
            color = MODEL_COLOR if name == model_label \
                    else METHOD_COLORS.get(name, '#cccccc')
            for element in ('whiskers', 'caps'):
                # Each box contributes 2 elements (lower + upper)
                bp[element][2 * i].set_color(color)
                bp[element][2 * i + 1].set_color(color)

        # Jitter overlay for evaluated model only (we have the actual data)
        rng = np.random.default_rng(1)
        jitter = rng.uniform(-0.18, 0.18, size=len(data))
        ax.scatter(model_pos + jitter, data, s=12, alpha=0.4,
                   color=MODEL_COLOR, zorder=3, edgecolors='none')

        ax.set_title(f'{title}\n{model_label} median = {np.median(data):.4f}',
                     fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.25)
        ax.set_xlim(0.3, n_methods + 0.7)

    fig.suptitle(
        f'Mega-Scale ddG prediction — per-protein test performance\n'
        f'(50/50 split, LinearRegression on Δembedding features, n={len(per_corr)} proteins)',
        fontsize=11,
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, 'ddg_mega_scale_results.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[Plot] saved → {out}')


def plot_results_precomputed(output_dir, model_label='DPLM'):
    """Multi-method boxplots using fully pre-computed statistics — no raw data needed.

    All four methods' 5-number summaries (whislo, Q1, median, Q3, whishi) are
    hard-coded so the figure can be regenerated without re-running inference.

    Stats format: [whislo, q1, median, q3, whishi]
    (i.e. min/max without outliers using 1.5×IQR rule, Q1, Q2, Q3)
    """
    ALL_CORR = {
        'SeqDance':   [0.17,   0.37,   0.44,   0.53,   0.74  ],
        'ESMDance':   [0.33,   0.51,   0.58,   0.66,   0.84  ],
        'ESM2':       [0.08,   0.40,   0.56,   0.65,   0.88  ],
        # model_label:  [0.5589, 0.7188, 0.7882, 0.8292, 0.9139],
        model_label:  [0.555, 0.7298, 0.8049, 0.8467, 0.9237],
    }
    ALL_MAE = {
        'SeqDance':   [0.22,   0.51,   0.63,   0.78,   1.15  ],
        'ESMDance':   [0.21,   0.44,   0.54,   0.67,   0.99  ],
        'ESM2':       [0.26,   0.44,   0.56,   0.66,   0.96  ],
        # model_label:  [0.1762, 0.3296, 0.4287, 0.5779, 0.9323],
        model_label:  [0.1698, 0.3072, 0.4124, 0.5494, 0.9028],
    }

    METHOD_COLORS = {
        'SeqDance': '#b2b2b2',   # medium gray
        'ESMDance': '#a6cee3',   # light blue
        'ESM2':     '#fdbf6f',   # light orange
    }
    MODEL_COLOR = '#2166ac'      # dark blue — evaluated model

    all_names = list(ALL_CORR.keys())   # preserves insertion order
    n_methods = len(all_names)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    panels = [
        (ALL_CORR, 'Per-protein Spearman r',    'Spearman r'),
        (ALL_MAE,  'Per-protein MAE (kcal/mol)', 'MAE (kcal/mol)'),
    ]

    for ax, (stats_dict, title, ylabel) in zip(axes, panels):
        stats_list = []
        for name in all_names:
            w_lo, q1, med, q3, w_hi = stats_dict[name]
            stats_list.append(dict(med=med, q1=q1, q3=q3,
                                   whislo=w_lo, whishi=w_hi,
                                   fliers=[], label=name))

        bp = ax.bxp(
            stats_list,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=2),
            widths=0.5,
        )

        for patch, name in zip(bp['boxes'], all_names):
            color = MODEL_COLOR if name == model_label \
                    else METHOD_COLORS.get(name, '#cccccc')
            patch.set_facecolor(color)
            patch.set_alpha(0.75 if name == model_label else 0.65)

        for i, name in enumerate(all_names):
            color = MODEL_COLOR if name == model_label \
                    else METHOD_COLORS.get(name, '#cccccc')
            for element in ('whiskers', 'caps'):
                bp[element][2 * i].set_color(color)
                bp[element][2 * i + 1].set_color(color)

        med_val = stats_dict[model_label][2]
        ax.set_title(f'{title}\n{model_label} median = {med_val:.4f}', fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.25)
        ax.set_xlim(0.3, n_methods + 0.7)

    fig.suptitle(
        f'Mega-Scale ddG prediction — per-protein test performance\n'
        f'(50/50 split, LinearRegression on Δembedding features)',
        fontsize=11,
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, 'ddg_mega_scale_results_precomputed.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[Plot] saved → {out}')


# ────────────────────────────────────────────────────────────────────────────
# 7.  CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Per-protein ddG prediction on the Mega-Scale stability dataset.')
    p.add_argument('--csv_path',        required=True,
                   help='Path to the Tsuboyama2023 CSV file.')
    p.add_argument('--model_type',      default='d-plm',
                   choices=['d-plm', 'ESM2'],
                   help='Model to evaluate: d-plm (fine-tuned) or ESM2 baseline. '
                        '(default: d-plm)')
    p.add_argument('--config_path',     default=None,
                   help='Training config YAML (required for d-plm).')
    p.add_argument('--checkpoint_path', default=None,
                   help='Checkpoint .pth (required for d-plm).')
    p.add_argument('--output_dir',      default='./ddg_mega_results',
                   help='Directory to save the figure and summary text.')
    p.add_argument('--max_proteins',    type=int, default=None,
                   help='Cap number of proteins evaluated (useful for quick testing).')
    p.add_argument('--seed',            type=int, default=42,
                   help='Random seed for the 50/50 train/test split. (default: 42)')
    p.add_argument('--batch_size',      type=int, default=8,
                   help='Batch size for ESM2/DPLM sequence encoding. (default: 8)')
    p.add_argument('--save_predictions', action='store_true',
                   help='Also write a per-mutation predictions CSV (test halves).')
    p.add_argument('--feature', default='mean', choices=['mean', 'site'],
                   help="'mean' = mean_pool(mut)−mean_pool(wt) [D] (default); 'site' = "
                        "site-aware 4·D feature at the mutated residue (like ddg_S669).")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# 8.  main
# ────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Dataset ───────────────────────────────────────────────────────
    print('\n=== Step 1: Loading dataset ===')
    df = load_dataset(args.csv_path)

    # ── Step 2: Model ─────────────────────────────────────────────────────────
    print(f'\n=== Step 2: Loading model ({args.model_type}) ===')
    model, alphabet = _load_model(args, device)

    # ── Step 3: Evaluate ──────────────────────────────────────────────────────
    print(f'\n=== Step 3: Per-protein evaluation (50/50 split, LinearRegression, '
          f'feature={args.feature}) ===')
    if args.feature == 'site':
        residue_encode = lambda seq: _encode_residues_esm(model, alphabet, seq, device)
        per_corr, per_mae, models, pred_rows = evaluate_proteins_site(
            df, [residue_encode], device, max_proteins=args.max_proteins,
            seed=args.seed, save_predictions=args.save_predictions)
    else:
        per_corr, per_mae, models, pred_rows = evaluate_proteins(
            df, model, alphabet, device,
            max_proteins=args.max_proteins,
            seed=args.seed,
            batch_size=args.batch_size,
            save_predictions=args.save_predictions,
        )

    if not per_corr:
        print('No proteins were evaluated successfully. Exiting.')
        return

    # ── Save the learnt per-protein models (+ optional predictions CSV) ────────
    model_path = os.path.join(args.output_dir, 'ddg_mega_models.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)
    print(f'[Model] {len(models)} per-protein LinearRegression models → {model_path}')

    if args.save_predictions:
        pred_path = os.path.join(args.output_dir, 'ddg_mega_predictions.csv')
        pd.DataFrame(pred_rows, columns=['protein_id', 'mut_type', 'position',
                                         'wt_aa', 'mt_aa', 'ddG', 'prediction']
                     ).to_csv(pred_path, index=False)
        print(f'[Predictions] {len(pred_rows)} test-half rows → {pred_path}')

    # ── Step 4: Summary ───────────────────────────────────────────────────────
    print('\n=== Results ===')
    print(f'  Proteins evaluated   : {len(per_corr)}')
    print(f'  Spearman r  — min (w/o outliers) / Q1 / median / Q3 / max (w/o outliers)')
    print(f'                  {np.percentile(per_corr, 0):.4f}  '
          f'{np.percentile(per_corr, 25):.4f}  '
          f'{np.median(per_corr):.4f}  '
          f'{np.percentile(per_corr, 75):.4f}  '
          f'{np.percentile(per_corr, 100):.4f}  (mean: {np.mean(per_corr):.4f})')
    print(f'  MAE (kcal/mol) — min (w/o outliers) / Q1 / median / Q3 / max (w/o outliers)')
    print(f'                  {np.percentile(per_mae, 0):.4f}  '
          f'{np.percentile(per_mae, 25):.4f}  '
          f'{np.median(per_mae):.4f}  '
          f'{np.percentile(per_mae, 75):.4f}  '
          f'{np.percentile(per_mae, 100):.4f}  (mean: {np.mean(per_mae):.4f})')

    # ── Step 5: Plot ──────────────────────────────────────────────────────────
    print('\n=== Step 4: Plotting ===')
    plot_results(per_corr, per_mae, args.output_dir, model_label=args.model_type)

    # ── Step 6: Save summary text ─────────────────────────────────────────────
    c_lo, c_q1, c_med, c_q3, c_hi = _compute_five_number(per_corr)
    m_lo, m_q1, m_med, m_q3, m_hi = _compute_five_number(per_mae)

    summary_path = os.path.join(args.output_dir, 'ddg_mega_scale_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f'model_type             : {args.model_type}\n')
        f.write(f'checkpoint             : {args.checkpoint_path}\n')
        f.write(f'config                 : {args.config_path}\n')
        f.write(f'n_proteins             : {len(per_corr)}\n')
        f.write('\n--- Spearman r distribution ---\n')
        f.write(f'min (w/o outliers)     : {c_lo:.4f}\n')
        f.write(f'25th percentile (Q1)   : {c_q1:.4f}\n')
        f.write(f'median                 : {c_med:.4f}\n')
        f.write(f'mean                   : {np.mean(per_corr):.4f}\n')
        f.write(f'75th percentile (Q3)   : {c_q3:.4f}\n')
        f.write(f'max (w/o outliers)     : {c_hi:.4f}\n')
        f.write('\n--- MAE distribution (kcal/mol) ---\n')
        f.write(f'min (w/o outliers)     : {m_lo:.4f}\n')
        f.write(f'25th percentile (Q1)   : {m_q1:.4f}\n')
        f.write(f'median                 : {m_med:.4f}\n')
        f.write(f'mean                   : {np.mean(per_mae):.4f}\n')
        f.write(f'75th percentile (Q3)   : {m_q3:.4f}\n')
        f.write(f'max (w/o outliers)     : {m_hi:.4f}\n')
    print(f'[Summary] saved → {summary_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()


"""
PYTHONPATH=. python ./evaluate/ddg_mega/ddg_mega_scale.py \
--csv_path /path/to/DPLM_data/ddg/Tsuboyama2023_Dataset2_Dataset3_20230416.csv \
--config_path /work/nvme/bcnr/jyx/DPLM_ai/results/vivit3/config_vivit3.yaml \
--checkpoint_path /work/nvme/bcnr/jyx/DPLM_ai/results/vivit3/checkpoints/checkpoint_best_val_rmsf_cor.pth \
--output_dir ./evaluate/ddg_mega/results/ \
--batch_size 32
"""