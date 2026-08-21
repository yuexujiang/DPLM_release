"""wt-mt-RLA zero-shot mutation-effect scoring on ProteinGym's viral DMS assays.

Multi-method counterpart to `predict_fitness.py`, which is DPLM-only. Scores
DPLM / ESM2 / ProstT5 / SeqDance with the SAME representation-distance metric:

    score(mutant) = -log( mean_i || rep_mt[i] - rep_wt[i] ||_2  + 1e-8 )

over per-residue representations of the model's final layer, BOS/EOS stripped — i.e.
`predict_fitness.py:score_wt_mt_rla` with `similarity='euclidean_distance'`.

Two deliberate differences from `predict_fitness.py`, both of which remove failure modes
rather than change the metric:

1. **Mutant sequences come from ProteinGym's own `mutated_sequence` column** instead of
   being reconstructed by applying `mutant` at `pos - offset_idx`. ProteinGym ships the
   full mutated sequence per row, so the offset bookkeeping (and its WT-mismatch /
   out-of-range failure modes, and its silent-misalignment risk for multi-mutants) simply
   does not arise. Verified per assay: every `mutated_sequence` is the same length as the
   manifest's `target_seq`.
2. **Mutants are scored in batches.** Every mutant of a substitution assay has exactly the
   same length as the WT, so a batch needs no padding and the forward pass is numerically
   identical to scoring one at a time — but ~10-30x faster, which is what makes 213k
   forward passes per method tractable.

Model loading is imported from `evaluate/methodology/rmsf.py` so there is exactly one
definition of "how DPLM/ESM2/ProstT5/SeqDance are loaded" in the tree.

    python evaluate/fitness/predict_fitness_viral.py \
        --methods esm2 \
        --manifest /path/to/DPLM_data/proteingym/viral23_manifest.csv \
        --dms_dir  /path/to/DPLM_data/proteingym/DMS_ProteinGym_substitutions \
        --output_dir ./results/proteingym_viral
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from evaluate.methodology.rmsf import (
    load_dplm, load_esm2, load_prostt5, load_seqdance,
)

METHODS = ('dplm', 'esm2', 'prostt5', 'seqdance')

# Method-appropriate zero-shot scoring. `rla` is the representation-distance score in
# predict_fitness.py; the other two are the scoring each model family is actually designed
# for, so a comparison across them is not confounded by an ill-suited read-out.
#
#   rla             DPLM  — -log mean_i ||rep_mt[i] - rep_wt[i]||, final layer
#   mask_marginals  ESM2  — masked-LM log-likelihood ratio, sum_j logP(mt_j) - logP(wt_j)
#                           at each mutated position j with that position masked
#   dynamics        SeqDance — mean |relative change| of each predicted dynamic property
#                           between WT and mutant, quantile-normalised across the assay's
#                           mutants, combined by geometric mean (SeqDance's own
#                           notebook/zero_shot_mutation.ipynb procedure)
SCORE_MODES = ('rla', 'mask_marginals', 'dynamics')
DEFAULT_MODE = {'dplm': 'rla', 'esm2': 'mask_marginals',
                'seqdance': 'dynamics', 'prostt5': 'rla'}

# SeqDance's 23 predicted dynamic properties (model/config.py res_feature_idx + pair_feature_idx)
SEQDANCE_RES_FEATURES = ('sasa_mean', 'sasa_std', 'rmsf_nor', 'ss', 'chi', 'phi', 'psi',
                         'nma_res1', 'nma_res2', 'nma_res3')
SEQDANCE_PAIR_FEATURES = ('vdw', 'hbbb', 'hbsb', 'hbss', 'hp', 'sb', 'pc', 'ps', 'ts',
                          'corr', 'nma_pair1', 'nma_pair2', 'nma_pair3')
SEQDANCE_EPS = 1e-2          # matches the notebook's epsilon

# Cap on tokens per forward batch. Batch size is derived as max(1, BUDGET // L) so that
# short assays batch aggressively and the 3423-residue ZIKV assay drops to batch 1-2.
DEFAULT_TOKEN_BUDGET = 16384


# ─────────────────────────────────────────────────────────────────────────────
# batched per-residue representations, one function per model API
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def reps_esm_style(model, alphabet, seqs, device):
    """DPLM and ESM2 (fair-esm API) -> [B, L, D] float32 on GPU."""
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([(f'p{i}', s) for i, s in enumerate(seqs)])
    tokens = tokens.to(device)
    out = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
    return out['representations'][model.num_layers][:, 1:-1, :].float()


@torch.no_grad()
def reps_prostt5(model, tokenizer, seqs, device):
    import re as _re
    spaced = ['<AA2fold> ' + ' '.join(list(_re.sub(r'[UZOB]', 'X', s))) for s in seqs]
    ids = tokenizer(spaced, add_special_tokens=True, padding=False, return_tensors='pt').to(device)
    out = model(ids.input_ids, attention_mask=ids.attention_mask)
    return out.last_hidden_state[:, 1:-1, :].float()


@torch.no_grad()
def reps_seqdance(model, tokenizer, seqs, device):
    raw = tokenizer(list(seqs), return_tensors='pt').to(device)
    out = model(raw, return_res_emb=True, return_attention_map=False,
                return_res_pred=False, return_pair_pred=False)
    return out['res_emb'][:, 1:-1, :].float()


@torch.no_grad()
def score_mask_marginals(model, alphabet, wt_seq, mutants, device, token_budget):
    """ESM-1v/ESM2 masked-marginal LLR.

    One forward per RESIDUE (with that residue masked), not per mutant — so an assay with
    42k mutants over a 735-residue protein costs 735 forwards, not 42328. Masked positions
    are batched; every masked copy has the same length so no padding is involved.
    """
    bc = alphabet.get_batch_converter()
    _, _, tokens = bc([('wt', wt_seq)])
    tokens = tokens.to(device)
    L = len(wt_seq)

    log_probs = torch.empty(L, len(alphabet.all_toks), dtype=torch.float32, device=device)
    bs = max(1, token_budget // max(L, 1))
    for start in range(0, L, bs):
        idx = list(range(start, min(start + bs, L)))
        batch = tokens.repeat(len(idx), 1)
        for r, i in enumerate(idx):
            batch[r, i + 1] = alphabet.mask_idx           # +1 for BOS
        out = model(batch, repr_layers=[], return_contacts=False)['logits']
        for r, i in enumerate(idx):
            log_probs[i] = torch.log_softmax(out[r, i + 1].float(), dim=-1)

    scores = np.full(len(mutants), np.nan, dtype=np.float64)
    lp = log_probs.cpu().numpy()
    for k, mut in enumerate(mutants):
        try:
            total = 0.0
            for one in str(mut).split(':'):
                wt_aa, pos, mt_aa = one[0], int(one[1:-1]), one[-1]
                i = pos - 1                                # ProteinGym mutants are 1-based
                if not (0 <= i < L) or wt_seq[i] != wt_aa:
                    raise ValueError(f'{one}: WT mismatch/out of range')
                total += lp[i, alphabet.get_idx(mt_aa)] - lp[i, alphabet.get_idx(wt_aa)]
            scores[k] = total
        except (ValueError, IndexError, KeyError):
            pass
    return scores


@torch.no_grad()
def _seqdance_props(model, tokenizer, seq, device, use_pair):
    raw = tokenizer([seq], return_tensors='pt').to(device)
    out = model(raw, return_res_emb=False, return_attention_map=False,
                return_res_pred=True, return_pair_pred=use_pair)
    return {k: v for k, v in out.items() if isinstance(v, torch.Tensor)}


def score_dynamics(model, tokenizer, wt_seq, mut_seqs, device, max_len_pair, log=print):
    """SeqDance dynamic-property zero-shot (its own notebook's procedure).

    per feature f:  d_f(mutant) = mean( |P_f^MT - P_f^WT| / (|P_f^WT| + 1e-2) )
    then quantile-normalise the [n_mutants, n_features] matrix across mutants and combine
    by geometric mean. Returned NEGATED so that, like the other scorers, a HIGHER value
    means a more fit / less disruptive mutation.
    """
    L = len(wt_seq)
    use_pair = L <= max_len_pair
    if not use_pair:
        log(f'    L={L} > max_len_pair={max_len_pair}: using the 10 residue-level '
            f'properties only (the pairwise ones are [L,L,13] and do not fit)')
    wt = _seqdance_props(model, tokenizer, wt_seq, device, use_pair)
    feats = sorted(wt)

    rows = np.full((len(mut_seqs), len(feats)), np.nan, dtype=np.float64)
    for k, ms in enumerate(mut_seqs):
        if not isinstance(ms, str) or len(ms) != L:
            continue
        mt = _seqdance_props(model, tokenizer, ms, device, use_pair)
        for j, f in enumerate(feats):
            rows[k, j] = ((mt[f] - wt[f]).abs()
                          / (wt[f].abs() + SEQDANCE_EPS)).mean().item()

    ok = ~np.isnan(rows).any(axis=1)
    scores = np.full(len(mut_seqs), np.nan, dtype=np.float64)
    if ok.sum() >= 2:
        X = rows[ok]
        # quantile normalisation across mutants (columns = features), as in the notebook
        ranks = np.argsort(np.argsort(X, axis=0), axis=0)
        rank_means = np.mean(np.sort(X, axis=0), axis=1)
        Xn = np.zeros_like(X)
        for j in range(X.shape[1]):
            Xn[:, j] = rank_means[ranks[:, j]]
        gm = np.exp(np.mean(np.log(Xn + 1e-8), axis=1))
        scores[ok] = -gm            # negate: larger disruption -> lower fitness
    return scores, feats, use_pair


def build_backend(method, args, device):
    """-> (rep_fn(seqs)->[B,L,D], label)."""
    if method == 'dplm':
        model, alphabet = load_dplm(args.dplm_config, args.dplm_checkpoint, device)
        return (lambda seqs: reps_esm_style(model, alphabet, seqs, device)), 'DPLM'
    if method == 'esm2':
        model, alphabet = load_esm2(device, args.esm_model)
        fn = lambda seqs: reps_esm_style(model, alphabet, seqs, device)   # noqa: E731
        fn.raw = (model, alphabet)
        return fn, 'ESM2'
    if method == 'prostt5':
        model, tok = load_prostt5(device)
        return (lambda seqs: reps_prostt5(model, tok, seqs, device)), 'ProstT5'
    if method == 'seqdance':
        model, tok = load_seqdance(args.seqdance_path, device)
        fn = lambda seqs: reps_seqdance(model, tok, seqs, device)         # noqa: E731
        fn.raw = (model, tok)
        return fn, 'SeqDance'
    raise ValueError(f'Unknown method {method!r} (expected one of {METHODS})')


# ─────────────────────────────────────────────────────────────────────────────

def score_assay(rep_fn, wt_seq, mut_seqs, token_budget, log=print):
    """-> np.array of wt-mt-RLA scores, NaN where the mutant could not be scored."""
    L = len(wt_seq)
    wt = rep_fn([wt_seq])[0]                       # [L, D]

    scores = np.full(len(mut_seqs), np.nan, dtype=np.float64)
    # Only same-length mutants are scorable — an indel would break the residue pairing that
    # the metric is defined over. ProteinGym substitution assays should have none.
    ok = [i for i, s in enumerate(mut_seqs) if isinstance(s, str) and len(s) == L]
    n_skip = len(mut_seqs) - len(ok)
    if n_skip:
        log(f'    {n_skip} mutant(s) skipped (length != WT — not a substitution)')

    bs = max(1, token_budget // max(L, 1))
    for start in range(0, len(ok), bs):
        idx = ok[start:start + bs]
        mt = rep_fn([mut_seqs[i] for i in idx])    # [b, L, D]
        # per-residue L2 distance, averaged over residues -> one number per mutant
        d = torch.linalg.vector_norm(mt - wt.unsqueeze(0), dim=-1).mean(dim=-1)
        scores[idx] = (-torch.log(d + 1e-8)).cpu().numpy()
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--methods', default='esm2',
                   help="'+'-separated subset of " + '|'.join(METHODS))
    p.add_argument('--manifest', required=True, help='viral23_manifest.csv')
    p.add_argument('--dms_dir', required=True, help='DMS_ProteinGym_substitutions/')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--token_budget', type=int, default=DEFAULT_TOKEN_BUDGET)
    p.add_argument('--score_mode', default=None, choices=SCORE_MODES,
                   help='Override the per-method default: ' +
                        ', '.join(f'{k}->{v}' for k, v in DEFAULT_MODE.items()))
    p.add_argument('--max_seq_len', type=int, default=None,
                   help='Drop assays whose WT is longer than this. The SeqDance paper uses '
                        '1024; pass it so every method is scored on an IDENTICAL assay set '
                        '(5 of the 23 viral assays exceed it). Excluded assays are listed.')
    p.add_argument('--max_len_pair', type=int, default=1024,
                   help='dynamics mode: above this length the 13 pairwise properties are '
                        'skipped ([L,L,13] plus [L,L,240] attentions stops fitting).')
    p.add_argument('--max_mutants', type=int, default=None,
                   help='Cap mutants per assay (smoke tests only — changes the result).')
    p.add_argument('--assay_subset', default=None, help="'+'-separated DMS_ids to restrict to.")
    p.add_argument('--dplm_config', default=None)
    p.add_argument('--dplm_checkpoint', default=None)
    p.add_argument('--esm_model', default='esm2_t33_650M_UR50D')
    p.add_argument('--seqdance_path',
                   default=None,
                   help='Path to SeqDance-main/model/. SeqDance is a baseline encoder and is NOT shipped in this release.')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    manifest = pd.read_csv(args.manifest)
    if args.assay_subset:
        keep = set(args.assay_subset.split('+'))
        manifest = manifest[manifest.DMS_id.isin(keep)]
    if args.max_seq_len:
        too_long = manifest[manifest.seq_len > args.max_seq_len]
        if len(too_long):
            print(f'[viral] max_seq_len={args.max_seq_len}: EXCLUDING {len(too_long)} assay(s) '
                  f'-> ' + ', '.join(f'{r.DMS_id}({r.seq_len})' for r in too_long.itertuples()),
                  flush=True)
        manifest = manifest[manifest.seq_len <= args.max_seq_len]
    print(f'[viral] device={device}  assays={len(manifest)}  '
          f'methods={args.methods}  token_budget={args.token_budget}', flush=True)

    for method in args.methods.split('+'):
        if method not in METHODS:
            raise ValueError(f'Unknown method {method!r}')
        if method == 'dplm' and not (args.dplm_config and args.dplm_checkpoint):
            raise ValueError('dplm requires --dplm_config and --dplm_checkpoint')

        rep_fn, label = build_backend(method, args, device)
        mode_used = args.score_mode or DEFAULT_MODE[method]
        print(f'[viral] {label}: score_mode={mode_used}', flush=True)
        label = f'{label}_{mode_used}' if mode_used != 'rla' else label
        out_dir = os.path.join(args.output_dir, label)
        os.makedirs(out_dir, exist_ok=True)
        rows, t_method = [], time.time()
        _first_assay = True

        for _, m in manifest.iterrows():
            dms_id, t0 = m.DMS_id, time.time()
            csv_path = os.path.join(args.dms_dir, m.DMS_filename)
            if not os.path.exists(csv_path):
                print(f'  [{label}] {dms_id}: CSV missing, skipped', flush=True)
                rows.append(dict(DMS_id=dms_id, spearman=np.nan, n=0, seconds=0.0))
                continue

            df = pd.read_csv(csv_path)
            if args.max_mutants:
                df = df.head(args.max_mutants)
            mode = args.score_mode or DEFAULT_MODE[method]
            try:
                if mode == 'mask_marginals':
                    preds = score_mask_marginals(*rep_fn.raw, m.target_seq,
                                                 df.mutant.tolist(), device,
                                                 args.token_budget)
                elif mode == 'dynamics':
                    preds, feats, used_pair = score_dynamics(
                        *rep_fn.raw, m.target_seq, df.mutated_sequence.tolist(), device,
                        args.max_len_pair,
                        log=lambda s: print(f'  [{label}] {dms_id}{s}', flush=True))
                    if _first_assay:
                        print(f'  [{label}] dynamic properties used: {len(feats)} '
                              f'({"incl." if used_pair else "NO"} pairwise) -> {feats}',
                              flush=True)
                        _first_assay = False
                else:
                    preds = score_assay(rep_fn, m.target_seq, df.mutated_sequence.tolist(),
                                        args.token_budget,
                                        log=lambda s: print(f'  [{label}] {dms_id}{s}', flush=True))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f'  [{label}] {dms_id}: OOM at L={len(m.target_seq)} — '
                      f'retrying with token_budget//4', flush=True)
                preds = score_assay(rep_fn, m.target_seq, df.mutated_sequence.tolist(),
                                    max(args.token_budget // 4, len(m.target_seq)))

            df['prediction'] = preds
            valid = df.dropna(subset=['DMS_score', 'prediction'])
            rho = spearmanr(valid.DMS_score, valid.prediction).statistic if len(valid) > 2 else np.nan
            # Drop `mutated_sequence` from the saved file: it is fully recoverable from the
            # ProteinGym CSV via `mutant`, and keeping it would cost ~800 MB across the four
            # methods on a /project fileset that has ~22 GB free.
            df.drop(columns=['mutated_sequence'], errors='ignore').to_csv(
                os.path.join(out_dir, f'{dms_id}_predict.csv'), index=False)
            dt = time.time() - t0
            rows.append(dict(DMS_id=dms_id, spearman=rho, n=len(valid), seconds=round(dt, 1)))
            print(f'  [{label}] {dms_id}: n={len(valid)} L={len(m.target_seq)} '
                  f'spearman={rho:+.4f}  ({dt:.0f}s)', flush=True)

        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(args.output_dir, f'summary_{label}.csv'), index=False)
        print(f'[{label}] mean spearman = {res.spearman.mean():+.4f} '
              f'(median {res.spearman.median():+.4f}, {res.spearman.notna().sum()} assays, '
              f'{(time.time() - t_method) / 60:.1f} min)', flush=True)

        del rep_fn
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
