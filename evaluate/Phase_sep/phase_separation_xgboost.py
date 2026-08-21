"""
phase_separation_xgboost.py — Phase Separation Protein Prediction Pipeline

Training data: Molphase_train_pos.xlsx (label=1) + Molphase_train_neg.xlsx (label=0)
Encoder     : DPLM or ESM2 → mean-pooled 1280-dim embeddings
Classifier  : XGBoost
Evaluation  : tableS1 – tableS5 (Positive=1, Negative=0)
              TableS2 also split into disorder-only / fold-only subsets

Usage
-----
# Full pipeline with DPLM checkpoint
PYTHONPATH=. python evaluate/Phase_sep/phase_separation_xgboost.py \\
    --model_type d-plm \\
    --checkpoint_path ./results/vivit5/checkpoints/checkpoint_best_val_whole_loss.pth \\
    --config_path ./configs/config_vivit5_delta.yaml \\
    --output_path ./evaluate/Phase_sep/xgboost/ \\
    --train_pos ./evaluate/Phase_sep/Molphase_train_pos.xlsx \\
    --train_neg ./evaluate/Phase_sep/Molphase_train_neg.xlsx \\
    --test_dir  ./evaluate/Phase_sep/

# Quick smoke-test with random embeddings (no GPU needed)
PYTHONPATH=. python evaluate/Phase_sep/phase_separation_xgboost.py \\
    --use_dummy_encoder \\
    --output_path ./evaluate/Phase_sep/xgboost_test/
"""

import os
import sys
import argparse
import pickle
import re
import warnings

import numpy as np
import pandas as pd
import yaml
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score,
    recall_score, average_precision_score,
)
import xgboost as xgb

# Project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import esm
import esm_adapterH
from utils.utils import load_configs, load_esm2_checkpoint

warnings.filterwarnings('ignore')


# ────────────────────────────────────────────────────────────────────────────
# 1.  CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Phase Separation XGBoost Pipeline')
    p.add_argument('--model_type',       default='d-plm', choices=['d-plm', 'ESM2'],
                   help='Encoder model (default: d-plm)')
    p.add_argument('--checkpoint_path',  default=None,
                   help='DPLM checkpoint .pth (required for d-plm)')
    p.add_argument('--config_path',      default=None,
                   help='Training config YAML (required for d-plm)')
    p.add_argument('--output_path',      default='./evaluate/Phase_sep/xgboost/',
                   help='Directory to save model, embeddings, results')
    p.add_argument('--train_pos',        default='./evaluate/Phase_sep/Molphase_train_pos.xlsx',
                   help='Positive training Excel file')
    p.add_argument('--train_neg',        default='./evaluate/Phase_sep/Molphase_train_neg.xlsx',
                   help='Negative training Excel file')
    p.add_argument('--test_dir',         default='./evaluate/Phase_sep/',
                   help='Directory containing tableS1.xlsx … tableS5.xlsx')
    p.add_argument('--save_model',       default='xgb_phase_sep.pkl',
                   help='Filename for the saved XGBoost model (default: xgb_phase_sep.pkl)')
    p.add_argument('--batch_size',       type=int, default=8,
                   help='Encoding batch size (default: 8)')
    p.add_argument('--use_dummy_encoder', action='store_true',
                   help='Use random 480-dim embeddings (for debugging without GPU)')
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# 2.  Model loading
# ────────────────────────────────────────────────────────────────────────────

def _load_model(args, device):
    """Return (model, alphabet) for DPLM or plain ESM2.
    Returns (None, None) when use_dummy_encoder is set.
    """
    if args.use_dummy_encoder:
        return None, None

    if args.model_type == 'd-plm':
        assert args.checkpoint_path and args.config_path, \
            '--checkpoint_path and --config_path are required for d-plm.'
        with open(args.config_path) as f:
            config_file = yaml.full_load(f)
        configs = load_configs(config_file, args=None)

        if configs.model.esm_encoder.adapter_h.enable:
            model, alphabet = esm_adapterH.pretrained.esm2_t33_650M_UR50D(
                configs.model.esm_encoder.adapter_h)
        else:
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        load_esm2_checkpoint(model, args.checkpoint_path)
        print(f'[DPLM] checkpoint loaded: {args.checkpoint_path}')
    else:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        print('[ESM2 base] loaded (no fine-tuning)')

    model.eval().to(device)
    return model, alphabet


# ────────────────────────────────────────────────────────────────────────────
# 3.  Encoding helpers
# ────────────────────────────────────────────────────────────────────────────

def _encode_batch(model, alphabet, sequences, device, batch_size=8):
    """Mean-pool ESM2/DPLM residue embeddings (excl. CLS / EOS). Returns [N, 1280]."""
    batch_converter = alphabet.get_batch_converter()
    all_embs = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        data = [(f'p{j}', s) for j, s in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
        reps = out['representations'][model.num_layers]   # [B, L+2, 1280]
        for b in range(reps.shape[0]):
            mask = ((tokens[b] != alphabet.padding_idx) &
                    (tokens[b] != alphabet.cls_idx) &
                    (tokens[b] != alphabet.eos_idx))
            all_embs.append(reps[b][mask].mean(0).cpu().numpy())
    return np.stack(all_embs, axis=0)


def _get_embedding(seq, model, alphabet, device, use_dummy):
    """Return mean-pooled embedding for one sequence."""
    if use_dummy:
        rng = np.random.default_rng(abs(hash(seq)) % (2 ** 31))
        return rng.random(480).astype(np.float32)
    return _encode_batch(model, alphabet, [seq], device, batch_size=1)[0]


def embed_sequences(sequences, model, alphabet, device, use_dummy, desc=''):
    """Embed a list of sequences with a progress counter. Returns [N, D]."""
    vecs = []
    n = len(sequences)
    for i, seq in enumerate(sequences):
        if (i + 1) % 50 == 0 or i == 0:
            print(f'  [{desc}] {i+1}/{n}')
        vecs.append(_get_embedding(seq, model, alphabet, device, use_dummy))
    return np.vstack(vecs)


def cached_embed(sequences, cache_path, model, alphabet, device, use_dummy, desc=''):
    """Load embeddings from cache .npy if available, otherwise compute and save."""
    if os.path.exists(cache_path):
        print(f'  Loading cached embeddings: {cache_path}')
        return np.load(cache_path)
    print(f'  Computing embeddings for {len(sequences)} sequences ({desc}) …')
    X = embed_sequences(sequences, model, alphabet, device, use_dummy, desc=desc)
    np.save(cache_path, X)
    print(f'  Saved to {cache_path}')
    return X


# ────────────────────────────────────────────────────────────────────────────
# 4.  Data loading
# ────────────────────────────────────────────────────────────────────────────

def load_flat_sequences(path, col='Sequence'):
    """Load sequences from a simple tabular Excel file (Molphase format)."""
    df = pd.read_excel(path)
    seqs = df[col].dropna().str.strip().tolist()
    seqs = [s for s in seqs if s]
    print(f'  Loaded {len(seqs)} sequences from {os.path.basename(path)}')
    return seqs


def load_labeled_sequences(path):
    """Parse the interleaved label/header/sequence format used in tableS1–S5.

    Layout (0-indexed, no header row):
      col-0: 'Positive' or 'Negative' (then NaN for body rows)
      col-1: alternating FASTA-header ('>...') and sequence lines
              (tableS2 may have sequence in col-2)

    Returns (sequences, labels) as parallel lists.
    """
    df = pd.read_excel(path, header=None)
    sequences, labels = [], []
    current_label  = None
    pending_header = False
    n_cols = df.shape[1]

    for _, row in df.iterrows():
        cell0 = str(row.iloc[0]).strip()
        if cell0.lower() in ('positive', 'negative'):
            current_label = 1 if cell0.lower() == 'positive' else 0

        if current_label is None:
            continue

        for col_idx in range(1, n_cols):
            val = str(row.iloc[col_idx]).strip() if pd.notna(row.iloc[col_idx]) else ''
            if not val or val.lower() == 'nan':
                continue
            # Skip short non-sequence tokens (e.g. 'Disordered', 'Folded' in tableS2)
            if len(val) < 10 and not val.startswith('>'):
                continue
            if val.startswith('>'):
                pending_header = True
            elif pending_header:
                clean = re.sub(r'[^A-Za-z]', '', val)
                if len(clean) > 5:
                    sequences.append(clean.upper())
                    labels.append(current_label)
                    pending_header = False

    print(f'  Loaded {len(sequences)} sequences '
          f'(pos={labels.count(1)}, neg={labels.count(0)}) '
          f'from {os.path.basename(path)}')
    return sequences, labels


# ────────────────────────────────────────────────────────────────────────────
# 5.  XGBoost training
# ────────────────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train):
    """Train XGBoost classifier and return the fitted model."""
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=50)
    return clf


# ────────────────────────────────────────────────────────────────────────────
# 6.  Evaluation helpers
# ────────────────────────────────────────────────────────────────────────────

def _compute_metrics(name, y_true, y_pred, y_prob):
    """Compute classification metrics; handle edge-case of single class."""
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan')
    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = float('nan')
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)

    print(f'  ROC-AUC  : {auc:.4f}')
    print(f'  PRC-AUC  : {auprc:.4f}')
    print(f'  Accuracy : {acc:.4f}')
    print(f'  F1       : {f1:.4f}')
    print(f'  Precision: {prec:.4f}')
    print(f'  Recall   : {rec:.4f}')

    return dict(
        Dataset=name, N=len(y_true),
        N_pos=int(y_true.sum()), N_neg=int((y_true == 0).sum()),
        ROC_AUC=round(auc, 4), PRC_AUC=round(auprc, 4),
        Accuracy=round(acc, 4), F1=round(f1, 4),
        Precision=round(prec, 4), Recall=round(rec, 4),
    )


def evaluate_one(name, seqs, y_true, xgb_model, model, alphabet, device,
                 use_dummy, output_path, batch_size):
    """Embed → predict → compute metrics for one dataset. Returns metrics dict."""
    if len(seqs) == 0:
        print(f'  {name}: no sequences — skipping.')
        return None
    cache_path = os.path.join(output_path, f'{name.lower()}_emb.npy')
    X = cached_embed(seqs, cache_path, model, alphabet, device, use_dummy, desc=name)
    y_true  = np.array(y_true)
    y_pred  = xgb_model.predict(X)
    y_prob  = xgb_model.predict_proba(X)[:, 1]
    return _compute_metrics(name, y_true, y_pred, y_prob)


def evaluate_s2_subsets(xgb_model, fpath, model, alphabet, device,
                        use_dummy, output_path, batch_size):
    """Evaluate TableS2 disorder-only and fold-only subsets.

    The fixed index slicing ([:100], [100:200], [200:300], [300:]) assumes the
    standard 4×100 layout: 100 pos-disordered, 100 pos-folded,
    100 neg-disordered, 100 neg-folded — as per the original dataset.
    """
    seqs, y_true = load_labeled_sequences(fpath)
    cache_path   = os.path.join(output_path, 'tables2_emb.npy')
    X_all = cached_embed(seqs, cache_path, model, alphabet, device, use_dummy,
                         desc='TableS2')
    y_true = np.array(y_true)

    # disorder = pos-disordered (0:100) + neg-disordered (200:300)
    disorder_X     = np.vstack([X_all[:100],      X_all[200:300]])
    disorder_y     = np.hstack([y_true[:100],      y_true[200:300]])
    # folded    = pos-folded    (100:200) + neg-folded    (300:)
    fold_X         = np.vstack([X_all[100:200],   X_all[300:]])
    fold_y         = np.hstack([y_true[100:200],  y_true[300:]])

    results = []
    for tag, X, y in [('TableS2_disorder_only', disorder_X, disorder_y),
                      ('TableS2_fold_only',     fold_X,     fold_y)]:
        print(f'\n--- {tag} ---')
        y_pred = xgb_model.predict(X)
        y_prob = xgb_model.predict_proba(X)[:, 1]
        r = _compute_metrics(tag, y, y_pred, y_prob)
        if r:
            results.append(r)
    return results


# ────────────────────────────────────────────────────────────────────────────
# 7.  Benchmark comparison plot
# ────────────────────────────────────────────────────────────────────────────

def plot_benchmark(output_path):
    """Bar chart: DPLM vs 9 competing methods on AUROC and AUPRC (TableS1 results)."""
    # Final benchmark values (TableS1 comparison)
    # AUROC=[0.990, 0.981, 0.981, 0.933, 0.916, 0.263, 0.479, 0.741, 0.911, 0.682]
    # AUPRC=[0.994, 0.987, 0.982, 0.941, 0.902, 0.493, 0.489, 0.793, 0.862, 0.719]
    # method_name=["DPLM", "MolPhase", "DeePhase","Fuzdrop","PSPHunter",
    #              "LLPhyScore","PSAP","PSPire","Phaseek","PICNIC"]
    
    # AUROC=[0.892, 0.876, 0.884, 0.837, 0.832, 0.406, 0.530, 0.769, 0.836, 0.643]
    # AUPRC=[0.859, 0.834, 0.850, 0.834, 0.794, 0.542, 0.510, 0.810, 0.770, 0.621]
    # method_name=["DPLM", "MolPhase", "DeePhase","Fuzdrop","PSPHunter",
    #              "LLPhyScore","PSAP","PSPire","Phaseek","PICNIC"]
                 
    # AUROC=[0.920, 0.888, 0.886, 0.818, 0.890, 0.467, 0.585, 0.768, 0.801, 0.744]
    # AUPRC=[0.916, 0.846, 0.893, 0.830, 0.871, 0.563, 0.574, 0.756, 0.743, 0.746]
    # method_name=["DPLM", "MolPhase", "DeePhase","Fuzdrop","PSPHunter",
    #              "LLPhyScore","PSAP","PSPire","Phaseek","PICNIC"]
    
    # AUROC=[0.920, 0.741, 0.891, 0.595, 0.799, 0.361, 0.590, 0.802, 0.486, 0.781]
    # AUPRC=[0.915, 0.519, 0.893, 0.507, 0.695, 0.375, 0.448, 0.763, 0.321, 0.654]
    # method_name=["DPLM", "MolPhase", "DeePhase","Fuzdrop","PSPHunter",
    #              "LLPhyScore","PSAP","PSPire","Phaseek","PICNIC"]
    
    # AUROC=[0.973, 0.908, 0.866, 0.897, 0.791, 0.256, 0.294, 0.590, 0.582, 0.674]
    # AUPRC=[0.981, 0.914, 0.888, 0.900, 0.823, 0.483, 0.483, 0.563, 0.606, 0.670]
    # method_name=["DPLM", "MolPhase", "DeePhase","Fuzdrop","PSPHunter",
    #              "LLPhyScore","PSAP","PSPire","Phaseek","PICNIC"]
    
    # #disordered proteins only
    # AUROC=[0.932, 0.884, 0.918, 0.914, 0.848, 0.677, 0.606, 0.839, 0.854, 0.642]
    # AUPRC=[0.888, 0.810, 0.873, 0.903, 0.810, 0.707, 0.577, 0.875, 0.780, 0.608]
    # method_name=["DPLM", "MolPhase", "DeePhase","Fuzdrop","PSPHunter",
    #              "LLPhyScore","PSAP","PSPire","Phaseek","PICNIC"]
    
    # #folded proteins only
    AUROC=[0.98, 0.962, 0.969, 0.898, 0.855, 0.105, 0.464, 0.701, 0.897, 0.641]
    AUPRC=[0.988, 0.974, 0.973, 0.921, 0.811, 0.332, 0.476, 0.757, 0.837, 0.640]
    method_name=["DPLM", "MolPhase", "DeePhase","Fuzdrop","PSPHunter",
                 "LLPhyScore","PSAP","PSPire","Phaseek","PICNIC"]

    # AUROC = [0.886, 0.876, 0.884, 0.837, 0.832, 0.406, 0.530, 0.769, 0.836, 0.643]
    # AUPRC = [0.858, 0.834, 0.850, 0.834, 0.794, 0.542, 0.510, 0.810, 0.770, 0.621]
    # methods = ['DPLM', 'MolPhase', 'DeePhase', 'Fuzdrop', 'PSPHunter',
    #            'LLPhyScore', 'PSAP', 'PSPire', 'Phaseek', 'PICNIC']

    order          = np.argsort(AUROC)[::-1]
    AUROC_sorted   = np.array(AUROC)[order]
    AUPRC_sorted   = np.array(AUPRC)[order]
    methods_sorted = np.array(method_name)[order]

    x     = np.arange(len(methods_sorted))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, AUROC_sorted, width, label='AUROC', color='steelblue')
    bars2 = ax.bar(x + width / 2, AUPRC_sorted, width, label='AUPRC', color='coral')

    ax.set_xticks(x)
    ax.set_xticklabels(methods_sorted, rotation=30, ha='right')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.set_title('Phase Separation Prediction: AUROC vs AUPRC by Method')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    out = os.path.join(output_path, 'benchmark_comparison.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[Plot] benchmark saved → {out}')


# ────────────────────────────────────────────────────────────────────────────
# 8.  main
# ────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    os.makedirs(args.output_path, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────
    print('\n=== Loading encoder ===')
    model, alphabet = _load_model(args, device)

    # ── Training data ──────────────────────────────────────────────────────
    print('\n=== Loading training data ===')
    pos_seqs = load_flat_sequences(args.train_pos)
    neg_seqs = load_flat_sequences(args.train_neg)

    X_pos = cached_embed(pos_seqs,
                         os.path.join(args.output_path, 'train_pos_emb.npy'),
                         model, alphabet, device, args.use_dummy_encoder,
                         desc='train-pos')
    X_neg = cached_embed(neg_seqs,
                         os.path.join(args.output_path, 'train_neg_emb.npy'),
                         model, alphabet, device, args.use_dummy_encoder,
                         desc='train-neg')

    X_train = np.vstack([X_pos, X_neg])
    y_train = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs))
    print(f'Training set: {X_train.shape}  '
          f'pos={y_train.sum()}  neg={(y_train==0).sum()}')

    # ── Train XGBoost ──────────────────────────────────────────────────────
    print('\n=== Training XGBoost ===')
    xgb_model = train_xgboost(X_train, y_train)

    model_path = os.path.join(args.output_path, args.save_model)
    with open(model_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f'Model saved → {model_path}')

    # ── Evaluate on tableS1 – tableS5 ─────────────────────────────────────
    print('\n=== Evaluating on test sets ===')
    test_files = {f'TableS{i}': os.path.join(args.test_dir, f'tableS{i}.xlsx')
                  for i in range(1, 6)}
    results = []

    for name, fpath in test_files.items():
        if not os.path.exists(fpath):
            print(f'  {name}: file not found — skipping.')
            continue
        print(f'\n--- {name} ---')
        seqs, y_true = load_labeled_sequences(fpath)
        r = evaluate_one(name, seqs, y_true, xgb_model,
                         model, alphabet, device, args.use_dummy_encoder,
                         args.output_path, args.batch_size)
        if r:
            results.append(r)

    # ── TableS2 disorder / fold subsets ───────────────────────────────────
    s2_path = os.path.join(args.test_dir, 'tableS2.xlsx')
    if os.path.exists(s2_path):
        print('\n=== TableS2 subsets (disorder-only / fold-only) ===')
        results.extend(
            evaluate_s2_subsets(xgb_model, s2_path,
                                model, alphabet, device, args.use_dummy_encoder,
                                args.output_path, args.batch_size)
        )

    # ── Save summary CSV ───────────────────────────────────────────────────
    summary_df   = pd.DataFrame(results)
    summary_path = os.path.join(args.output_path, 'evaluation_results.csv')
    summary_df.to_csv(summary_path, index=False)

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(summary_df.to_string(index=False))
    print(f'\nResults saved → {summary_path}')

    # ── Benchmark comparison plot ──────────────────────────────────────────
    print('\n=== Benchmark plot ===')
    # plot_benchmark(args.output_path)

    print('\nDone.')


if __name__ == '__main__':
    main()

"""
PYTHONPATH=. python evaluate/Phase_sep/phase_separation_xgboost.py \
--checkpoint_path /work/nvme/bcnr/jyx/DPLM_ai/results/vivit3/checkpoints/checkpoint_best_val_rmsf_cor.pth \
--config_path /work/nvme/bcnr/jyx/DPLM_ai/results/vivit3/config_vivit3.yaml \
--train_pos /path/to/DPLM_data/Phase_sep/Molphase_train_pos.xlsx \
--train_neg /path/to/DPLM_data/Phase_sep/Molphase_train_neg.xlsx \
--output_path ./evaluate/Phase_sep/xgboost/ \
--test_dir /path/to/DPLM_data/Phase_sep/ \
--batch_size 1


"""