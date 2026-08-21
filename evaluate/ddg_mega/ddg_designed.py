"""ddg_designed.py — ddG on the DE NOVO DESIGNED subset of the Mega-Scale dataset.

The 156 designed proteins are the artificial (no evolutionary history) subset of the
412-protein Tsuboyama-2023 set, so a model that relies on homology has nothing to draw on
here — which is what makes this subset a test of signal beyond evolutionary information.

Two evaluation modes, both writing per-mutation predictions:

  --mode supervised   per-protein 50/50 split, mean-pool(mut) - mean-pool(wt) feature,
                      LinearRegression, per-protein Spearman + MAE.
                      Identical protocol to ddg_mega_scale_baseline.evaluate_proteins, which
                      is imported and reused — only the protein subset and the set of
                      backbones differ. Methods: dplm | esm2 | prostt5 | esmdance
  --mode zeroshot     no training. Methods and their native scorers:
                      dplm      -> wt-mt-RLA  (-log mean_i ||rep_mt[i] - rep_wt[i]||)
                      esm2      -> masked-marginal LLR at the mutated position
                      seqdance  -> dynamic-property change, quantile-normalised, geometric
                                   mean, negated (SeqDance's own notebook procedure)
                      scorers imported from evaluate/fitness/predict_fitness_viral.py

⚠ SeqDance vs ESMDance: `rmsf.load_seqdance` defaults to the **ESMDance** HF repo, because
`from_pretrained` is a classmethod that rebuilds from the repo's config.json. `--method
esmdance` and `--method seqdance` therefore select genuinely different models here, and
every earlier result in this tree labelled "SeqDance" was in fact ESMDance.

    PYTHONPATH=. python evaluate/ddg_mega/ddg_designed.py --mode supervised --method dplm \
        --dplm_config <cfg> --dplm_checkpoint <ckpt> --output_dir <dir> --save_predictions
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', 'methodology')))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', 'fitness')))
sys.path.insert(0, _HERE)

from rmsf import (load_esm2, load_prostt5, load_seqdance,                      # noqa: E402
                  get_residue_emb_prostt5, get_residue_emb_seqdance)
from ddg_mega_scale import _load_model, _encode_sequences                       # noqa: E402
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from ddg_mega_scale import parse_mut_type                        # noqa: E402
# (evaluate_proteins / compute_features are inlined below)


# ── inlined from ddg_mega_scale_baseline.py (the pre-v2 module, removed
#    in the public release; only these helpers were ever used from it) ─────
def compute_features(df, pro, encode, batch_size):
    wt_key = f'{pro}$wt'
    if wt_key not in df.index:
        return [], None, None
    wt_seq = df.loc[wt_key, 'aa_seq']
    wt_emb = encode([wt_seq], 1)[0]

    mut_labels = [m for m in df[df['WT_name'] == pro]['mut_type'].values if m != 'wt']
    if not mut_labels:
        return [], None, None
    mut_seqs = [df.loc[f'{pro}${mt}', 'aa_seq'] for mt in mut_labels]
    mut_embs = encode(mut_seqs, batch_size)

    X = mut_embs - wt_emb[np.newaxis, :]
    y = np.array([df.loc[f'{pro}${mt}', 'ddG_ML'] for mt in mut_labels])
    return mut_labels, X, y

def evaluate_proteins(df, encode, max_proteins=None, seed=42, batch_size=8,
                      save_predictions=False):
    rng = np.random.default_rng(seed)
    proteins = sorted(df['WT_name'].unique())
    if max_proteins is not None:
        proteins = proteins[:max_proteins]

    per_corr, per_mae = [], []
    models, pred_rows = {}, []
    for pro in tqdm(proteins, desc='Proteins'):
        mut_labels, X, y = compute_features(df, pro, encode, batch_size)
        if X is None or len(X) < 4:
            continue
        idx = rng.permutation(len(X))
        n_train = len(idx) // 2
        tr, te = idx[:n_train], idx[n_train:]

        reg = LinearRegression().fit(X[tr], y[tr])
        y_pred = reg.predict(X[te])
        corr, _ = spearmanr(y[te], y_pred)
        if not np.isnan(corr):
            per_corr.append(corr)
            per_mae.append(mean_absolute_error(y[te], y_pred))
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

# The de novo designed subset of the Mega-scale set: 146 proteins / 123,245 mutations.
# NOT 156 — an earlier version classified a protein as designed whenever its WT_name did
# not begin with a 4-character PDB id, which wrongly swept in the 10 'v2*' entries (e.g.
# v2_2LC2.pdb, v2K43S_2KVV.pdb). Those embed real PDB ids and are NATURAL domains, so they
# break the premise of this benchmark, which is that the proteins have no evolutionary
# homologs. Excluding them reproduces the 331 natural / 148 designed split of the source
# paper (146 of the 148 survive load_dataset's ddG/indel quality filter).
DESIGNED_CSV = ('/path/to/DPLM_data/ddg_designed/'
                'tsuboyama_designed146_mutations.csv')
SUPERVISED_METHODS = ('dplm', 'esm2', 'prostt5', 'esmdance')
ZEROSHOT_METHODS = ('dplm', 'esm2', 'seqdance')


def load_designed(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    print(f'[designed] {len(df)} mutations across {df.WT_name.nunique()} designed proteins')
    return df


# ── supervised: mean-pooled protein embeddings -> LinearRegression ──────────────

def build_encode(method, args, device):
    """-> encode(seqs, batch_size) -> [N, D] mean-pooled protein embeddings."""
    if method == 'dplm':
        if not (args.dplm_config and args.dplm_checkpoint):
            raise ValueError('dplm requires --dplm_config and --dplm_checkpoint')
        shim = SimpleNamespace(model_type='d-plm', config_path=args.dplm_config,
                               checkpoint_path=args.dplm_checkpoint)
        model, alphabet = _load_model(shim, device)
        return lambda s, bs: _encode_sequences(model, alphabet, s, device=device, batch_size=bs)
    if method == 'esm2':
        model, alphabet = load_esm2(device, args.esm_model)
        return lambda s, bs: _encode_sequences(model, alphabet, s, device=device, batch_size=bs)
    if method == 'prostt5':
        model, tok = load_prostt5(device)
        return lambda s, bs: np.stack(
            [get_residue_emb_prostt5(model, tok, x, device).mean(axis=0) for x in s]
        ).astype(np.float32)
    if method == 'esmdance':
        model, tok = load_seqdance(args.seqdance_path, device, hf_repo='ChaoHou/ESMDance')
        return lambda s, bs: np.stack(
            [get_residue_emb_seqdance(model, tok, x, device).mean(axis=0) for x in s]
        ).astype(np.float32)
    raise ValueError(f'unknown supervised method {method!r} (expect {SUPERVISED_METHODS})')


# ── zero-shot: each model's native scorer ──────────────────────────────────────

def run_zeroshot(method, args, df, device, log=print):
    """-> DataFrame(WT_name, mut_type, ddG_ML, prediction) plus per-protein Spearman."""
    from predict_fitness_viral import (score_mask_marginals, score_dynamics,
                                       reps_esm_style)

    if method == 'dplm':
        shim = SimpleNamespace(model_type='d-plm', config_path=args.dplm_config,
                               checkpoint_path=args.dplm_checkpoint)
        model, alphabet = _load_model(shim, device)
    elif method == 'esm2':
        model, alphabet = load_esm2(device, args.esm_model)
    elif method == 'seqdance':
        # the TRUE from-scratch SeqDance, not ESMDance — see the module docstring
        model, tok = load_seqdance(args.seqdance_path, device, hf_repo=args.seqdance_repo)
    else:
        raise ValueError(f'unknown zero-shot method {method!r} (expect {ZEROSHOT_METHODS})')

    rows, rhos = [], []
    groups = list(df.groupby('WT_name'))
    if args.max_proteins:
        groups = groups[:args.max_proteins]
    for i, (pro, g) in enumerate(groups, 1):
        wt_rows = g[g.mut_type == 'wt']
        muts = g[g.mut_type != 'wt']
        if wt_rows.empty or len(muts) < 4:
            continue
        wt_seq = str(wt_rows.iloc[0].aa_seq)
        mut_seqs = muts.aa_seq.astype(str).tolist()

        if method == 'esm2':
            preds = score_mask_marginals(model, alphabet, wt_seq,
                                         muts.mut_type.astype(str).tolist(),
                                         device, args.token_budget)
        elif method == 'dplm':
            with torch.no_grad():
                wt = reps_esm_style(model, alphabet, [wt_seq], device)[0]
                preds = np.full(len(mut_seqs), np.nan)
                ok = [k for k, s in enumerate(mut_seqs) if len(s) == len(wt_seq)]
                bs = max(1, args.token_budget // max(len(wt_seq), 1))
                for st in range(0, len(ok), bs):
                    idx = ok[st:st + bs]
                    mt = reps_esm_style(model, alphabet, [mut_seqs[k] for k in idx], device)
                    d = torch.linalg.vector_norm(mt - wt.unsqueeze(0), dim=-1).mean(dim=-1)
                    preds[idx] = (-torch.log(d + 1e-8)).cpu().numpy()
        else:
            preds, _feats, _pair = score_dynamics(model, tok, wt_seq, mut_seqs, device,
                                                  args.max_len_pair, log=lambda s: None)

        y = muts.ddG_ML.astype(float).values
        ok = np.isfinite(preds) & np.isfinite(y)
        if ok.sum() > 3:
            rhos.append(spearmanr(y[ok], preds[ok]).statistic)
        rows.append(pd.DataFrame({'WT_name': pro, 'mut_type': muts.mut_type.values,
                                  'ddG_ML': y, 'prediction': preds}))
        if i % 25 == 0 or i == len(groups):
            log(f'  [{method}] {i} proteins  running mean rho={np.nanmean(rhos):+.4f}')
    return pd.concat(rows, ignore_index=True), np.asarray(rhos, dtype=float)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', required=True, choices=['supervised', 'zeroshot'])
    p.add_argument('--method', required=True)
    p.add_argument('--csv_path', default=DESIGNED_CSV)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--dplm_config', default=None)
    p.add_argument('--dplm_checkpoint', default=None)
    p.add_argument('--esm_model', default='esm2_t33_650M_UR50D')
    p.add_argument('--seqdance_path',
                   default=None,
                   help='Path to SeqDance-main/model/. SeqDance is a baseline encoder and is NOT shipped in this release.')
    p.add_argument('--seqdance_repo', default='ChaoHou/SeqDance',
                   help='zeroshot seqdance only: ChaoHou/SeqDance (true from-scratch model) '
                        'or ChaoHou/ESMDance')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--token_budget', type=int, default=16384)
    p.add_argument('--max_len_pair', type=int, default=1024)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_proteins', type=int, default=None)
    p.add_argument('--save_predictions', action='store_true')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = load_designed(args.csv_path)
    print(f'=== designed-protein ddG | mode={args.mode} method={args.method} device={device} ===')

    if args.mode == 'supervised':
        encode = build_encode(args.method, args, device)
        per_corr, per_mae, models, pred_rows = evaluate_proteins(
            df, encode, max_proteins=args.max_proteins, seed=args.seed,
            batch_size=args.batch_size, save_predictions=args.save_predictions)
        rhos = np.asarray(per_corr, dtype=float)
        if args.save_predictions and pred_rows:
            pd.DataFrame(pred_rows, columns=['protein_id', 'mut_type', 'position', 'wt_aa',
                                             'mt_aa', 'ddG', 'prediction']).to_csv(
                os.path.join(args.output_dir, 'predictions.csv'), index=False)
        extra = f'  mean MAE = {np.mean(per_mae):.4f}'
    else:
        preds, rhos = run_zeroshot(args.method, args, df, device)
        if args.save_predictions:
            preds.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
        extra = ''

    summary = (f'mode           : {args.mode}\nmethod         : {args.method}\n'
               f'n_proteins     : {len(rhos)}\n'
               f'mean Spearman  : {np.nanmean(rhos):.4f}\n'
               f'median Spearman: {np.nanmedian(rhos):.4f}\n')
    if args.mode == 'supervised':
        summary += f'mean MAE       : {np.mean(per_mae):.4f}\n'
    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write(summary)
    np.save(os.path.join(args.output_dir, 'per_protein_spearman.npy'), rhos)
    print(summary + extra)
    print(f'[out] {args.output_dir}')


if __name__ == '__main__':
    main()
