"""
train_cryptobench_xgb.py — Train/evaluate an XGBoost cryptic binding-site classifier
on CryptoBench.

Same evaluation protocol as train_cryptobench.py (4-fold CV to pick an F1 threshold,
then a final model trained on all folds and evaluated on the held-out test set), but the
per-residue neural classifier is replaced by gradient-boosted trees. XGBoost is a
position-independent per-residue classifier, so — like the MLP — it operates on the flat
[N_residues, D] feature matrix (residue order does not matter).

Class imbalance (~5% binding residues) is handled with scale_pos_weight = n_neg / n_pos.

XGBoost hyperparameters are read from an optional `xgb:` block in the config (falls back
to sensible defaults if absent, so cryptobench_config.yaml works unchanged).

Usage:
  python evaluate/cryptobench/train_cryptobench_xgb.py \
    --config_path evaluate/cryptobench/cryptobench_config.yaml \
    --result_path ./results/cryptobench_xgb \
    --emb_dir     <embeddings_dir> \
    --dataset_dir <cryptobench-dataset>
"""

import os
import sys
import json
import argparse
import numpy as np
import yaml
from pathlib import Path
from time import time

import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, matthews_corrcoef,
    confusion_matrix,
)
from box import Box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.utils import get_logging, prepare_saving_dir

sys.path.insert(0, os.path.dirname(__file__))
from data_cryptobench import CryptoBenchDataset


# ── Metrics (identical to train_cryptobench.py) ────────────────────────────────

def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    auc   = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    acc   = accuracy_score(labels, preds)
    mcc   = matthews_corrcoef(labels, preds)
    f1    = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return dict(auc=auc, auprc=auprc, acc=acc, mcc=mcc, f1=f1,
                tpr=tpr, fpr=fpr, threshold=threshold)


def find_best_threshold(probs: np.ndarray, labels: np.ndarray, thresholds=None) -> float:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.99, 95)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f = f1_score(labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


# ── Feature assembly ──────────────────────────────────────────────────────────

def stack_residues(datasets):
    """Flatten all proteins' per-residue embeddings into (X [N, D], y [N]).

    Residue order is irrelevant for a tree model, so every protein's [L, D] block is
    simply concatenated. Reads the already-loaded arrays from each dataset's `.items`
    (list of (apo_id, emb[:L], labels)).
    """
    X_parts, y_parts = [], []
    for ds in datasets:
        for _apo_id, emb, labels in ds.items:
            X_parts.append(np.asarray(emb, dtype=np.float32))
            y_parts.append(np.asarray(labels))
    if not X_parts:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.vstack(X_parts), np.concatenate(y_parts)


# ── Model ─────────────────────────────────────────────────────────────────────

def make_xgb(configs, y_train):
    """Build an XGBClassifier from config (with defaults) + imbalance weighting.

    Parameters mirror the working repo pattern in
    evaluate/Phase_sep/phase_separation_xgboost.py (CPU `hist`, use_label_encoder=False,
    n_jobs=-1) to guarantee compatibility with the installed xgboost version; individual
    fields can be overridden via an optional `xgb:` block in the config.
    """
    xcfg = getattr(configs, 'xgb', Box({}))
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    return xgb.XGBClassifier(
        n_estimators     = int(getattr(xcfg, 'n_estimators', 500)),
        max_depth        = int(getattr(xcfg, 'max_depth', 6)),
        learning_rate    = float(getattr(xcfg, 'learning_rate', 0.05)),
        subsample        = float(getattr(xcfg, 'subsample', 0.8)),
        colsample_bytree = float(getattr(xcfg, 'colsample_bytree', 0.8)),
        min_child_weight = float(getattr(xcfg, 'min_child_weight', 1.0)),
        reg_lambda       = float(getattr(xcfg, 'reg_lambda', 1.0)),
        scale_pos_weight = scale_pos_weight,
        eval_metric      = str(getattr(xcfg, 'eval_metric', 'logloss')),
        use_label_encoder = False,
        random_state     = int(getattr(xcfg, 'random_state', 42)),
        n_jobs           = int(getattr(xcfg, 'n_jobs', -1)),
    )


# ── Config loading ────────────────────────────────────────────────────────────

def load_configs(yaml_dict, args):
    cfg = Box(yaml_dict)
    if args.result_path:
        cfg.result_path = args.result_path
    if args.emb_dir:
        cfg.emb_dir = args.emb_dir
    if args.dataset_dir:
        cfg.dataset_dir = args.dataset_dir
    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config_path',  '-c', required=True)
    p.add_argument('--result_path',  default=None)
    p.add_argument('--emb_dir',      default=None, help='Override emb_dir from config')
    p.add_argument('--dataset_dir',  default=None, help='Override dataset_dir from config')
    p.add_argument('--save_predictions', action='store_true',
                   help='Write a per-residue test predictions CSV.')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config_path) as f:
        yaml_dict = yaml.full_load(f)
    configs = load_configs(yaml_dict, args)

    result_path, checkpoint_path = prepare_saving_dir(configs, args.config_path)
    log = get_logging(result_path)

    dataset_dir = Path(configs.dataset_dir)
    emb_dir     = Path(configs.emb_dir)

    with open(dataset_dir / 'dataset.json') as f:
        dataset = json.load(f)
    with open(dataset_dir / 'folds.json') as f:
        folds = json.load(f)

    # Build per-fold datasets
    fold_datasets = []
    for i in range(4):
        fold_datasets.append(CryptoBenchDataset(
            folds[f'train-{i}'], dataset, emb_dir, max_len=configs.max_len))
        log.info(f"train-fold-{i}: {len(fold_datasets[-1])} proteins  "
                 f"{sum(len(it[2]) for it in fold_datasets[-1].items)} residues")

    test_dataset = CryptoBenchDataset(
        folds['test'], dataset, emb_dir, max_len=configs.max_len)
    log.info(f"test: {len(test_dataset)} proteins  "
             f"{sum(len(it[2]) for it in test_dataset.items)} residues")

    # ── Sanity check: catch a near-empty dataset before training on garbage ──────
    fold_checks = [(f'train-{i}', fold_datasets[i], len(folds[f'train-{i}']))
                  for i in range(4)] + [('test', test_dataset, len(folds['test']))]
    for name, ds, expected in fold_checks:
        if expected > 0 and len(ds) < 0.5 * expected:
            raise RuntimeError(
                f"{name}: only {len(ds)}/{expected} proteins loaded (<50%) — "
                f"embeddings are likely missing or incomplete. Check "
                f"{emb_dir}/failed.txt and re-run the embedding step first.")

    # ── 4-fold cross-validation to find best threshold ────────────────────────
    log.info("=== 4-fold cross-validation for threshold selection ===")
    cv_thresholds, cv_metrics = [], []

    for val_fold in range(4):
        log.info(f"CV fold {val_fold}: val=fold-{val_fold}, "
                 f"train=folds-{[i for i in range(4) if i!=val_fold]}")
        X_tr, y_tr = stack_residues([fold_datasets[i] for i in range(4) if i != val_fold])
        X_val, y_val = stack_residues([fold_datasets[val_fold]])

        t0 = time()
        model = make_xgb(configs, y_tr)
        model.fit(X_tr, y_tr)
        val_probs = model.predict_proba(X_val)[:, 1]
        thresh = find_best_threshold(val_probs, y_val)
        cv_thresholds.append(thresh)
        cv_metrics.append(compute_metrics(val_probs, y_val, thresh))

        m = cv_metrics[-1]
        log.info(f"  fold-{val_fold} ({time()-t0:.1f}s) val results: "
                 f"AUC={m['auc']:.4f}  AUPRC={m['auprc']:.4f}  "
                 f"ACC={m['acc']:.4f}  TPR={m['tpr']:.4f}  FPR={m['fpr']:.4f}  "
                 f"MCC={m['mcc']:.4f}  F1={m['f1']:.4f}  thr={m['threshold']:.2f}")

    final_threshold = float(np.mean(cv_thresholds))
    log.info(f"\nCV mean threshold: {final_threshold:.3f}  "
             f"(per-fold: {[f'{t:.2f}' for t in cv_thresholds]})")

    mean_cv = {k: float(np.mean([m[k] for m in cv_metrics]))
               for k in cv_metrics[0] if k != 'threshold'}
    log.info(f"CV mean metrics: " + "  ".join(
        f"{k}={v:.4f}" for k, v in mean_cv.items()))

    # ── Final model: train on all 4 folds, evaluate on test ──────────────────
    log.info("\n=== Training final model on all train folds ===")
    X_all, y_all = stack_residues(fold_datasets)
    final_model = make_xgb(configs, y_all)
    final_model.fit(X_all, y_all)

    model_path = os.path.join(checkpoint_path, 'xgb_model.json')
    final_model.save_model(model_path)
    with open(os.path.join(checkpoint_path, 'threshold.json'), 'w') as f:
        json.dump({'threshold': final_threshold}, f)
    log.info(f"Model saved → {model_path}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    log.info(f"\n=== Test evaluation (threshold={final_threshold:.3f}) ===")
    X_te, y_te = stack_residues([test_dataset])
    test_probs = final_model.predict_proba(X_te)[:, 1]
    tm = compute_metrics(test_probs, y_te, final_threshold)
    log.info(
        f"[pLM-XGB] Test results:\n"
        f"  AUC    = {tm['auc']:.4f}\n"
        f"  AUPRC  = {tm['auprc']:.4f}\n"
        f"  ACC    = {tm['acc']:.4f}\n"
        f"  TPR    = {tm['tpr']:.4f}\n"
        f"  FPR    = {tm['fpr']:.4f}\n"
        f"  MCC    = {tm['mcc']:.4f}\n"
        f"  F1     = {tm['f1']:.4f}\n"
    )

    # ── Per-residue predictions CSV (protein_id, residue_index, label, prob, pred) ──
    if args.save_predictions:
        import csv as _csv
        pred_path = os.path.join(result_path, 'predictions.csv')
        with open(pred_path, 'w', newline='') as f:
            w = _csv.writer(f)
            w.writerow(['protein_id', 'residue_index', 'label', 'probability', 'prediction'])
            for apo_id, emb, labels in test_dataset.items:
                probs = final_model.predict_proba(np.asarray(emb, dtype=np.float32))[:, 1]
                for i in range(len(probs)):
                    w.writerow([apo_id, i, int(labels[i]), f'{probs[i]:.6f}',
                                int(probs[i] >= final_threshold)])
        log.info(f"[predictions] per-residue test predictions → {pred_path}")


if __name__ == '__main__':
    main()
