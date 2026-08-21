"""
ablation.py — Adapter ablation test: trained DPLM vs random-adapter DPLM.

Three arms, all scored on the same proteins:
  • DPLM          — trained adapter weights as checkpointed
  • DPLM-random   — same ESM2 backbone, adapters randomised (--random_mode)
  • ESM2          — base ESM2, no adapters at all  ← THE CONTROL

⚠ The ESM2 arm exists because without it the figure cannot distinguish the two things a
reader most needs to tell apart: "randomising destroyed the learned signal" versus
"randomising switched the adapters off, so this is just ESM2 again". Earlier versions of this
script plotted only the first two arms, which is why the random panel looked like an ESM2
panel with no way to check. Use --no_esm2 to drop it only if you already know the answer.

Measured on lcc_adam_v9 (2026-08-12): under --random_mode gaussian the adapter branch stays
live at 97% of trained magnitude (0.4315 vs 0.4458), so the adapters are NOT off — yet the
model still sits ~0.98 cosine from base ESM2 while trained DPLM sits at 0.87. Random
perturbations are incoherent and partly cancel across 1280 dims and 20 layers; the trained
shift is coherent and accumulates. `adapter_branch_magnitude()` reports this per run and
warns if an arm's branch has actually collapsed.

Analyses
--------
1. Scatter  : residue embedding norm vs RMSF (all proteins pooled), one panel per arm.
2. Boxplot  : per-protein Spearman(norm, RMSF) distribution, with PAIRED Wilcoxon tests
              (the arms share proteins, so the old unpaired Mann-Whitney was both wrong and
              silently skipped whenever more than two arms were present).

Usage
-----
PYTHONPATH=. python evaluate/methodology/ablation.py \\
    --data_path      /path/to/DPLM_data/processed_test_rep0/ \\
    --analysis_path  /path/to/DPLM_data/analysis/ \\
    --dplm_config    ./configs/config_vivit5_delta.yaml \\
    --dplm_checkpoint ./results/vivit5/checkpoints/checkpoint_best_val_whole_loss.pth \\
    --output_dir     ./evaluate/methodology/ablation_output/ \\
    --max_proteins   30
"""

# ============================================================
# 0.  Imports  (top-level: only side-effect-free packages)
# ============================================================
import argparse
import os
import re
import sys
import warnings
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.utils import load_configs, load_esm2_checkpoint

# Reuse data loading and embedding extraction from rmsf.py (same directory)
sys.path.insert(0, os.path.dirname(__file__))
from rmsf import load_protein_data, get_residue_emb_esm_style

warnings.filterwarnings('ignore')

# ============================================================
# Method palette
# ============================================================
METHOD_ORDER  = ['DPLM', 'DPLM-random', 'ESM2']
METHOD_COLORS = {
    'DPLM':        '#2166ac',   # blue  — trained
    'DPLM-random': '#d6604d',   # red   — randomised adapters
    'ESM2':        '#999999',   # grey  — no adapters at all (the control)
}
METHOD_LABELS = {
    'DPLM':        'DPLM (trained)',
    'DPLM-random': 'DPLM (random adapters)',
    'ESM2':        'ESM2 (no adapters)',
}

# ============================================================
# 1.  Model loading
# ============================================================

def _build_dplm(config_path, checkpoint_path, device):
    """Shared helper: load ESM2 + Houlsby adapters from a checkpoint."""
    import esm_adapterH

    with open(config_path) as f:
        config_file = yaml.full_load(f)
    configs = load_configs(config_file, args=None)

    adapter_cfg = configs.model.esm_encoder.adapter_h
    model_name = getattr(configs.model.esm_encoder, 'model_name', 'esm2_t33_650M_UR50D')
    model, alphabet = getattr(esm_adapterH.pretrained, model_name)(adapter_cfg)
    load_esm2_checkpoint(model, checkpoint_path)
    return model, alphabet


def load_dplm_trained(config_path, checkpoint_path, device):
    """DPLM with trained (checkpointed) adapter weights."""
    model, alphabet = _build_dplm(config_path, checkpoint_path, device)
    model.eval().to(device)
    print(f'[DPLM-trained] loaded from {checkpoint_path}')
    return model, alphabet


def randomize_adapters(model, seed=0, mode='gaussian'):
    """Destroy the LEARNED STRUCTURE of every Houlsby adapter, in-place.

    The MLP1 adapter is a residual branch `module(x) + x` whose module ENDS in a LayerNorm
    (esm_adapterH/adapter.py). LayerNorm(x) = γ·normalize(x) + β, so the branch magnitude is
    set almost entirely by γ: forcing γ ≈ 0.02 would shrink the branch to ~2% → near-identity
    → the model silently collapses to base ESM2 and the "ablation" measures nothing.

    Three modes, in increasing faithfulness to "same distribution, no structure":

      gaussian  (default, unchanged behaviour) — per tensor, draw ~ N(mean_t, std_t) from
                that tensor's own trained moments. Matches mean and variance.
      shuffle   — randomly PERMUTE each tensor's trained values. Matches the entire
                empirical distribution exactly (every moment, not just the first two), so a
                heavy-tailed weight matrix stays heavy-tailed. The strictest "same weights,
                wrong arrangement" control.
      reinit    — reset_parameters() on every submodule: the adapter as it was BEFORE
                training. Note this is near-identity by design for Houlsby adapters, so it
                is expected to look like base ESM2; it answers "what did training add?"
                rather than "is the learned structure special?".

    VERIFIED (2026-08-12, lcc_adam_v9, CPU): under `gaussian` the branch stays live —
    ||module(x)||/||x|| is 0.4315 random vs 0.4458 trained, a ratio of 0.968. The adapters
    are NOT being switched off. That the randomised model still lands ~0.98 cosine on base
    ESM2 is a property of incoherent perturbations partly cancelling across 1280 dims and 20
    layers, not an artefact of this function.
    """
    torch.manual_seed(seed)   # repeatable "random" ablation
    n_reset = 0
    with torch.no_grad():
        for layer in model.layers:
            if hasattr(layer, 'adapter_layer_dict') and layer.adapter_layer_dict is not None:
                for adapter_list in layer.adapter_layer_dict.values():
                    for adapter in adapter_list:
                        if mode == 'reinit':
                            for m in adapter.modules():
                                if hasattr(m, 'reset_parameters'):
                                    m.reset_parameters()
                                    n_reset += 1
                            continue
                        for _name, p in adapter.named_parameters():
                            if mode == 'shuffle':
                                flat = p.detach().reshape(-1)
                                p.copy_(flat[torch.randperm(flat.numel())].reshape(p.shape))
                            else:
                                mean_t = p.detach().float().mean().item()
                                std_t  = p.detach().float().std().item()
                                if std_t == 0.0:
                                    # Degenerate (constant tensor): jitter so we don't
                                    # produce an all-equal tensor.
                                    p.normal_(mean=mean_t, std=1e-6)
                                else:
                                    p.normal_(mean=mean_t, std=std_t)
                            n_reset += 1
    print(f'[randomize_adapters] mode={mode} touched {n_reset} adapter tensors (seed={seed})')


def adapter_branch_magnitude(model, alphabet, device, seq=None):
    """Diagnostic: mean ||module(x)|| / ||x|| over every adapter call for one sequence.

    Reports whether the adapters are actually DOING anything. A value near 0 means the
    branch is off and any "random adapter" comparison is really just base ESM2.
    """
    seq = seq or ('MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKAL'
                  'PDAQFEVVHSLAKWKR')
    ratios, hooks = [], []

    def hook(_mod, inp, out):
        x = inp[0].detach().float()
        ratios.append((out.detach().float() - x).norm().item() / max(x.norm().item(), 1e-9))

    for layer in model.layers:
        d = getattr(layer, 'adapter_layer_dict', None)
        if d is None:
            continue
        for al in d.values():
            for ad in al:
                hooks.append(ad.register_forward_hook(hook))
    _, _, toks = alphabet.get_batch_converter()([('x', seq)])
    with torch.no_grad():
        model(toks.to(device), repr_layers=[33])
    for h in hooks:
        h.remove()
    return float(np.mean(ratios)) if ratios else 0.0


def load_dplm_random(config_path, checkpoint_path, device, seed=0, mode='gaussian'):
    """DPLM with the same ESM2 backbone but all adapter weights randomised."""
    model, alphabet = _build_dplm(config_path, checkpoint_path, device)
    randomize_adapters(model, seed=seed, mode=mode)
    model.eval().to(device)
    print(f'[DPLM-random] adapters randomised (mode={mode}, seed={seed}).')
    return model, alphabet


def load_esm2_plain(device, model_name='esm2_t33_650M_UR50D'):
    """Base ESM2 — the control that tells you whether 'random adapters' == 'no adapters'.

    Without this arm the ablation figure cannot distinguish "randomising destroyed the
    learned signal" from "randomising switched the adapters off", which is the single most
    important thing a reader wants to know from it.
    """
    import esm
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval().to(device)
    print(f'[ESM2] base {model_name} loaded (no adapters).')
    return model, alphabet


# ============================================================
# 2.  Build embedding cache
# ============================================================

def build_embedding_cache(proteins, models_dict, device):
    """Return {pid: {method: np.ndarray[L, 1280]}} and filtered protein list."""
    cache = {}
    valid_proteins = []

    for prot in proteins:
        pid, seq = prot['pid'], prot['sequence']
        L = len(seq)
        embs = {}
        ok = True

        for method, (model, alphabet) in models_dict.items():
            try:
                emb = get_residue_emb_esm_style(model, alphabet, seq, device)
            except Exception as e:
                print(f'  [skip] {pid} / {method}: {e}')
                ok = False
                break

            if emb.shape[0] != L:
                print(f'  [skip] {pid} / {method}: emb len {emb.shape[0]} ≠ seq len {L}')
                ok = False
                break

            embs[method] = emb

        if ok:
            cache[pid] = embs
            valid_proteins.append(prot)
            print(f'  cached {pid}  ({L} residues)')

    print(f'\nEmbedding cache: {len(valid_proteins)} valid proteins.')
    return cache, valid_proteins


# ============================================================
# 3.  Analysis 1 — Scatter: emb norm vs RMSF
# ============================================================

def plot_scatter_norm_vs_rmsf(proteins, cache, output_dir):
    """1×2 scatter; each point = one amino acid (all proteins pooled).

    Left panel  : DPLM (trained adapters)
    Right panel : DPLM (random adapters)
    """
    methods = [m for m in METHOD_ORDER if m in next(iter(cache.values()))]

    fig, axes = plt.subplots(1, len(methods),
                             figsize=(6 * len(methods), 5),
                             sharey=False)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        all_norms, all_rmsf = [], []

        for prot in proteins:
            pid = prot['pid']
            if pid not in cache or method not in cache[pid]:
                continue
            emb  = cache[pid][method]           # [L, D]
            rmsf = prot['metric']                # [L]
            norms = np.linalg.norm(emb, axis=1)
            all_norms.extend(norms.tolist())
            all_rmsf.extend(rmsf.tolist())

        all_norms = np.array(all_norms)
        all_rmsf  = np.array(all_rmsf)
        corr, pval = spearmanr(all_norms, all_rmsf)

        ax.scatter(all_norms, all_rmsf,
                   s=3, alpha=0.25,
                   color=METHOD_COLORS[method],
                   rasterized=True)

        # Trend line
        m_coef, b_coef = np.polyfit(all_norms, all_rmsf, 1)
        x_line = np.linspace(all_norms.min(), all_norms.max(), 300)
        ax.plot(x_line, m_coef * x_line + b_coef,
                color='black', lw=1.2, linestyle='--')

        ax.set_title(
            f'{METHOD_LABELS[method]}\nSpearman r = {corr:.3f}  (p = {pval:.2e})',
            fontsize=11)
        ax.set_xlabel('Embedding norm')
        ax.set_ylabel('RMSF (Å)')
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        'Residue embedding norm vs RMSF — trained vs random adapters\n'
        '(all proteins pooled)',
        fontsize=12)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, 'ablation_scatter_norm_vs_rmsf.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[Analysis 1] saved → {out}')


# ============================================================
# 4.  Analysis 2 — Per-protein Spearman boxplot + p-value
# ============================================================

def _pval_str(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def plot_proteinwise_spearman_boxplot(proteins, cache, output_dir):
    """Boxplot of per-protein Spearman r(norm, RMSF); Mann-Whitney vs DPLM."""
    methods = [m for m in METHOD_ORDER if m in next(iter(cache.values()))]

    per_protein_corrs = {m: [] for m in methods}
    for prot in proteins:
        pid = prot['pid']
        if pid not in cache:
            continue
        rmsf = prot['metric']
        for method in methods:
            if method not in cache[pid]:
                continue
            norms = np.linalg.norm(cache[pid][method], axis=1)
            if len(norms) < 3:
                continue
            corr, _ = spearmanr(norms, rmsf)
            if not np.isnan(corr):
                per_protein_corrs[method].append(corr)

    fig, ax = plt.subplots(figsize=(max(6, 2.1 * len(methods)), 5.6))
    positions = list(range(1, len(methods) + 1))
    data   = [per_protein_corrs[m] for m in methods]
    colors = [METHOD_COLORS[m]     for m in methods]
    labels = [METHOD_LABELS[m]     for m in methods]

    bp = ax.boxplot(data, positions=positions, patch_artist=True,
                    showmeans=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2),
                    meanprops=dict(marker='D', markerfacecolor='white',
                                   markeredgecolor='black', markersize=6))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Jitter overlay
    rng = np.random.default_rng(0)
    for i, (vals, color) in enumerate(zip(data, colors)):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(positions[i] + jitter, vals,
                   s=20, alpha=0.65, color=color, zorder=3,
                   edgecolors='white', linewidths=0.3)

    # Every arm is scored on the SAME proteins, so the correct test is the PAIRED Wilcoxon,
    # not the unpaired Mann-Whitney used previously (which also silently vanished whenever
    # more than two arms were present, via a `len(methods) == 2` guard).
    # The comparison that answers "are random adapters just ESM2?" is random-vs-ESM2.
    # Order matters: only the first few get drawn as brackets, so the comparisons that
    # answer "are random adapters just ESM2?" go FIRST — that is the question this figure
    # exists to settle. DPLM-vs-random follows; all pairs are printed regardless.
    pairs, y_max = [], max((max(v, default=0) for v in data), default=0)
    rand = [m for m in methods if m.startswith('DPLM-random')]
    for a, b in ([(m, 'ESM2') for m in rand] + [('DPLM', 'ESM2')] +
                 [('DPLM', m) for m in rand]):
        va, vb = per_protein_corrs.get(a, []), per_protein_corrs.get(b, [])
        if not va or not vb or len(va) != len(vb):
            continue
        d = np.asarray(va, float) - np.asarray(vb, float)
        ok = np.isfinite(d)
        if ok.sum() < 3:
            continue
        try:
            pval = wilcoxon(np.asarray(va)[ok], np.asarray(vb)[ok]).pvalue
        except ValueError:                       # all-zero differences
            pval = 1.0
        pairs.append((a, b, float(d[ok].mean()), float((d[ok] > 0).mean()), float(pval)))

    if pairs:
        print('\n  paired Wilcoxon (same proteins, per-protein Spearman):')
        for a, b, dm, w, pv in pairs:
            print(f'    {a:22s} - {b:22s}  delta={dm:+.4f}  wins={100*w:5.1f}%  p={pv:.3g}')
        idx = {m: i + 1 for i, m in enumerate(methods)}
        for k, (a, b, dm, _w, pv) in enumerate(pairs[:3]):
            if a not in idx or b not in idx:
                continue
            bh = y_max + 0.05 + 0.09 * k
            x1, x2 = idx[a], idx[b]
            ax.plot([x1, x1, x2, x2], [bh, bh + 0.02, bh + 0.02, bh], lw=1.2, color='dimgray')
            ax.text((x1 + x2) / 2, bh + 0.025,
                    f'{_pval_str(pv)}  (Δ={dm:+.3f}, paired p={pv:.1e})',
                    ha='center', va='bottom', fontsize=8, color='dimgray')

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9, rotation=18, ha='right')
    ax.set_ylabel('Per-protein Spearman r  (norm vs RMSF)', fontsize=10)
    ax.set_title(
        'Adapter ablation — Spearman correlation distribution\n'
        '(trained vs randomised adapters vs base ESM2)',
        fontsize=11)
    ax.axhline(0, color='gray', lw=0.8, linestyle='--')
    ax.grid(axis='y', alpha=0.25)

    # Sample-size annotation below each box
    for i, (method, vals) in enumerate(zip(methods, data)):
        ax.text(positions[i], ax.get_ylim()[0] + 0.01,
                f'n={len(vals)}', ha='center', va='bottom',
                fontsize=8, color='dimgray')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, 'ablation_proteinwise_spearman_boxplot.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[Analysis 2] saved → {out}')

    # Print summary statistics
    print('\n--- Spearman correlation summary ---')
    for method, vals in zip(methods, data):
        if vals:
            arr = np.array(vals)
            print(f'  {METHOD_LABELS[method]:35s}  '
                  f'mean={arr.mean():.3f}  median={np.median(arr):.3f}  '
                  f'std={arr.std():.3f}  n={len(arr)}')


# ============================================================
# 5.  Argument parsing + main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Adapter ablation: trained DPLM vs random-adapter DPLM.')
    p.add_argument('--data_path',       required=True,
                   help='Dir of trajectory/processed files (protein ID enumeration).')
    p.add_argument('--analysis_path',   required=True,
                   help='Base dir containing {pid}_analysis/{pid}.pdb + {pid}_RMSF.tsv.')
    p.add_argument('--dplm_config',     required=True,
                   help='Path to training config YAML.')
    p.add_argument('--dplm_checkpoint', required=True,
                   help='Path to training checkpoint .pth.')
    p.add_argument('--output_dir',      default='./ablation_output',
                   help='Directory to save all figures.')
    p.add_argument('--max_proteins',    type=int, default=None,
                   help='Cap number of proteins (for quick testing).')
    p.add_argument('--rmsf_col',        default='RMSF_R1',
                   help='Column name in RMSF TSV (default: RMSF_R1).')
    p.add_argument('--random_mode',     default='gaussian',
                   choices=['gaussian', 'shuffle', 'reinit'],
                   help='How to destroy the adapters: gaussian = per-tensor N(mean,std); '
                        'shuffle = permute the trained values (matches the FULL empirical '
                        'distribution); reinit = pre-training init.')
    p.add_argument('--random_seeds',    default='0',
                   help="'+'-separated seeds; >1 seed shows how much of the random arm is "
                        'draw-to-draw noise rather than a stable effect.')
    p.add_argument('--no_esm2', action='store_true',
                   help='Drop the base-ESM2 control arm. Not recommended — without it the '
                        'figure cannot show whether random adapters == no adapters.')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Load protein data ─────────────────────────────────────────────
    print('\n=== Loading protein data ===')
    proteins = load_protein_data(
        args.data_path, args.analysis_path,
        rmsf_col=args.rmsf_col,
        max_proteins=args.max_proteins,
    )
    if not proteins:
        print('No valid proteins found. Exiting.')
        return

    # ── Step 2: Load both model variants ─────────────────────────────────────
    print('\n=== Loading models ===')
    print('Loading DPLM (trained) …')
    dplm_trained = load_dplm_trained(
        args.dplm_config, args.dplm_checkpoint, device)

    models_dict = {'DPLM': dplm_trained}

    # Report whether the adapters are actually contributing. If this is ~0 for the random
    # arm, the "ablation" is silently just base ESM2 and every downstream number is void.
    mag_tr = adapter_branch_magnitude(dplm_trained[0], dplm_trained[1], device)
    print(f'[diag] trained adapter branch ||module(x)||/||x|| = {mag_tr:.4f}')

    seeds = [int(s) for s in re.split(r'[+,\s]+', args.random_seeds) if s.strip()]
    for sd in seeds:
        key = 'DPLM-random' if len(seeds) == 1 else f'DPLM-random-s{sd}'
        print(f'Loading DPLM (random adapters, mode={args.random_mode}, seed={sd}) …')
        m = load_dplm_random(args.dplm_config, args.dplm_checkpoint, device,
                             seed=sd, mode=args.random_mode)
        mag_rd = adapter_branch_magnitude(m[0], m[1], device)
        print(f'[diag] {key} adapter branch ||module(x)||/||x|| = {mag_rd:.4f}  '
              f'(ratio vs trained = {mag_rd / max(mag_tr, 1e-9):.3f})')
        if mag_rd < 0.05 * mag_tr:
            print(f'[diag] ⚠ {key} branch is effectively OFF — this arm is base ESM2, '
                  f'not a random-adapter model. Interpret with care.')
        models_dict[key] = m
        if key not in METHOD_ORDER:
            METHOD_ORDER.insert(-1, key)
            METHOD_COLORS[key] = '#d6604d'
            METHOD_LABELS[key] = f'DPLM (random adapters, seed {sd})'

    if not args.no_esm2:
        print('Loading ESM2 (no adapters) — the control …')
        models_dict['ESM2'] = load_esm2_plain(device)
    else:
        METHOD_ORDER.remove('ESM2')

    # ── Step 3: Build embedding cache ─────────────────────────────────────────
    print('\n=== Computing residue embeddings ===')
    cache, valid_proteins = build_embedding_cache(proteins, models_dict, device)
    if not valid_proteins:
        print('No valid proteins after embedding. Exiting.')
        return

    # ── Step 4: Analyses ──────────────────────────────────────────────────────
    print('\n=== Analysis 1: Scatter (norm vs RMSF, amino acid level) ===')
    plot_scatter_norm_vs_rmsf(valid_proteins, cache, args.output_dir)

    print('\n=== Analysis 2: Per-protein Spearman boxplot + p-value ===')
    plot_proteinwise_spearman_boxplot(valid_proteins, cache, args.output_dir)

    print(f'\nAll figures saved to: {args.output_dir}')


if __name__ == '__main__':
    main()

'''
# ── Example run ───────────────────────────────────────────────────────────────
PYTHONPATH=. python evaluate/methodology/ablation.py \
    --data_path      /path/to/DPLM_data/processed_test_rep0/ \
    --analysis_path  /path/to/DPLM_data/analysis/ \
    --dplm_config    ./configs/config_vivit3_delta.yaml \
    --dplm_checkpoint ./results/vivit3_ori/checkpoints/checkpoint_best_val_whole_loss.pth \
    --output_dir     ./evaluate/methodology/ablation_output/test \
    --max_proteins   30
'''

