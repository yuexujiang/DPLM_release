"""
rmsf.py — residue-embedding vs MD-dynamics-metric analysis for DPLM, ESM2, ProstT5,
          SeqDance and SPLM (SPLM-V2-GVP).

SPLM is optional and runs OUT-OF-PROCESS (--splm_path): its package names collide with this
project's (esm_adapterH, utils.utils) so it cannot be imported here. See load_splm().

All analyses share a single --data_path / --analysis_path pair, and all act on a
single per-residue target metric selected via --metric {rmsf,bfactor,neq}.
Model loading happens inside functions — importing this file is side-effect free.

Analyses
--------
1. Scatter  : emb norm vs metric per amino acid (all proteins pooled), 2×2 subplot.
2. Boxplot  : per-protein Spearman(norm, metric) distribution, p-values vs DPLM.
3. t-SNE    : (disabled) residue embeddings coloured by metric value.
4. KMeans   : (disabled) metric binned into 5 equal buckets → KMeans(k=5) → ARI.
5. RSA      : per-protein Spearman between residue-residue embedding distance and
              residue-residue |metric_i - metric_j| difference (Representational
              Similarity Analysis) — captures directional information that a single
              norm scalar discards. Same boxplot/significance-bracket layout as
              Analysis 2.

Usage
-----
PYTHONPATH=. python evaluate/methodology/rmsf.py \\
    --data_path     /path/to/DPLM_data/Atlas_geom2vec_test/ \\
    --analysis_path /path/to/DPLM_data/analysis/ \\
    --dplm_config   ./configs/config_vivit5_delta.yaml \\
    --dplm_checkpoint ./results/vivit5/checkpoints/checkpoint_best_val_whole_loss.pth \\
    --seqdance_path <SeqDance-main/model> \\
    --output_dir    ./evaluate/methodology/rmsf_output/ \\
    --metric        rmsf \\
    --max_proteins  30
"""

# ============================================================
# 0.  Imports  (top-level: only side-effect-free packages)
# ============================================================
import argparse
import os
import sys
import re
import warnings
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from collections import OrderedDict
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Project utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.utils import load_configs, load_dplm_checkpoint
from utils.evaluation import pdb2seq

warnings.filterwarnings('ignore')

# ============================================================
# 1.  Data loading
# ============================================================

# Per-residue metric file suffix + column(s) to average, keyed by --metric value.
METRIC_FILES = {
    'rmsf':    {'suffix': 'RMSF',    'cols': ['RMSF_R1', 'RMSF_R2', 'RMSF_R3']},
    'bfactor': {'suffix': 'Bfactor', 'cols': ['Bfactor']},
    'neq':     {'suffix': 'Neq',     'cols': ['Neq_R1', 'Neq_R2', 'Neq_R3']},
}


def load_protein_data(data_path, analysis_path, metric='rmsf', rmsf_col=None,
                      max_proteins=None):
    """Return a list of dicts, one per valid protein.

    Each dict: {'pid': str, 'sequence': str, 'metric': np.ndarray[L]}

    `metric` selects which per-residue target to load: 'rmsf', 'bfactor', or 'neq'.
    For 'rmsf'/'neq' (which have R1/R2/R3 replicate columns), the mean across
    replicates is used unless `rmsf_col` names a single specific column.

    Protein IDs are derived from filenames in data_path as the first two
    underscore-separated tokens, e.g. "1abc_A_prod_R1.h5" → "1abc_A".
    """
    if metric not in METRIC_FILES:
        raise ValueError(f'Unknown metric: {metric!r} (choices: {list(METRIC_FILES)})')
    suffix = METRIC_FILES[metric]['suffix']
    cols   = [rmsf_col] if rmsf_col else METRIC_FILES[metric]['cols']

    seen = set()
    proteins = []

    for fname in sorted(os.listdir(data_path)):
        fname = fname.strip()
        if not fname:
            continue
        parts = fname.split('#')
        if len(parts) < 2:
            continue
        # pid = '_'.join(parts[:2])
        pid = parts[0]
        if pid in seen:
            continue
        seen.add(pid)

        pdb_file    = os.path.join(analysis_path, f'{pid}_analysis', f'{pid}.pdb')
        metric_file = os.path.join(analysis_path, f'{pid}_analysis', f'{pid}_{suffix}.tsv')

        if not os.path.exists(pdb_file) or not os.path.exists(metric_file):
            continue

        try:
            sequence = str(pdb2seq(pdb_file))
            df = pd.read_csv(metric_file, sep='\t')
            present_cols = [c for c in cols if c in df.columns]
            if not present_cols:
                print(f'  [skip] {pid}: none of {cols} found in {metric_file}')
                continue
            values = df[present_cols].values.astype(np.float32).mean(axis=1)
        except Exception as e:
            print(f'  [skip] {pid}: {e}')
            continue

        # Require sequence and metric to have matching lengths
        if len(sequence) != len(values):
            print(f'  [skip] {pid}: seq len {len(sequence)} ≠ {metric} len {len(values)}')
            continue

        proteins.append({'pid': pid, 'sequence': sequence, 'metric': values})

        if max_proteins and len(proteins) >= max_proteins:
            break

    print(f'Loaded {len(proteins)} proteins  (metric={metric}).')
    return proteins


# ============================================================
# 2.  Model loading  (imports deferred to inside each function)
# ============================================================

def load_dplm(config_path, checkpoint_path, device):
    """Load DPLM (ESM2 + Houlsby adapters) from a training checkpoint."""
    import esm_adapterH

    with open(config_path) as f:
        config_file = yaml.full_load(f)
    configs = load_configs(config_file, args=None)

    adapter_cfg = configs.model.esm_encoder.adapter_h
    model_name = getattr(configs.model.esm_encoder, 'model_name', 'esm2_t33_650M_UR50D')
    model, alphabet = getattr(esm_adapterH.pretrained, model_name)(adapter_cfg)
    load_dplm_checkpoint(model, checkpoint_path)

    model.eval().to(device)
    print(f'[DPLM] loaded from {checkpoint_path}')
    return model, alphabet


def load_esm2(device, model_name='esm2_t33_650M_UR50D'):
    """Load vanilla ESM2 (no fine-tuning). model_name defaults to 650M."""
    import esm as esm_pkg
    model, alphabet = getattr(esm_pkg.pretrained, model_name)()
    model.eval().to(device)
    print(f'[ESM2] base model loaded ({model_name}).')
    return model, alphabet


def load_esmc(device, model_name='esmc_600m'):
    """Load vanilla ESM-C (no fine-tuning). model_name defaults to 600M (1152-d).

    ESM-C ships in EvolutionaryScale's `esm` PyPI package, whose top-level module name
    collides with `fair-esm` (used above for ESM2), so it cannot be installed alongside.
    `esmc_local/` vendors the architecture and loads the official EvolutionaryScale
    weights — see esmc_local/__init__.py.
    """
    from esmc_local import load_esmc as _load
    return _load(model_name, device)


def load_prostt5(device):
    """Load ProstT5 encoder and its tokenizer.

    Forces safetensors weights: newer `transformers` versions refuse to load
    torch.load-format (.bin) checkpoints unless torch >= 2.6 (CVE-2025-32434 guard),
    and this project pins torch 2.3.0 (install_env.sh). Rostlab/ProstT5 publishes
    safetensors weights, so requesting them explicitly avoids the torch.load path
    entirely without needing to upgrade torch project-wide.
    """
    from transformers import T5Tokenizer, T5EncoderModel
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
    model = T5EncoderModel.from_pretrained('Rostlab/ProstT5', use_safetensors=True)
    model.eval().to(device)
    print('[ProstT5] model loaded.')
    return model, tokenizer


def load_seqdance(seqdance_model_path, device, hf_repo='ChaoHou/ESMDance'):
    """Load an ESMwrap dynamics model from local code + HuggingFace weights.

    ⚠ NAMING: despite this function's name, the DEFAULT `hf_repo` is **ESMDance**, and that
    is what every prior result in this project labelled "SeqDance" actually used.
    `PyTorchModelHubMixin.from_pretrained` is a CLASSMETHOD, so the `model_select` passed to
    the constructor below is discarded and the model is rebuilt from the repo's own
    config.json:

        ChaoHou/ESMDance -> model_select='esmdance' (freeze_esm=True,  pretrained ESM2 kept)
        ChaoHou/SeqDance -> model_select='seqdance' (freeze_esm=False, ESM2 randomised)

    Verified empirically. Pass hf_repo='ChaoHou/SeqDance' for the true from-scratch model.
    """
    from transformers import AutoTokenizer
    sys.path.insert(0, seqdance_model_path)
    from model import ESMwrap  # noqa: E402  (local SeqDance import)
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
    model = ESMwrap('model_35M', 'seqdance').from_pretrained(hf_repo)

    # transformers >= 4.5x defaults ESM to SDPA, which never materialises the attention
    # weights: `output_attentions=True` is ignored (with a warning) and the output has no
    # 'attentions' entry, so SeqDance's own forward -- which does
    # `torch.cat(esm_output['attentions'], ...)` at model.py:114 -- dies with KeyError
    # 'attentions'. That silently skipped every protein in the DCCM attention benchmark
    # (job 5388329). Forcing eager attention restores the maps; it does not change the
    # residue embeddings, so embedding-only callers are unaffected.
    if hasattr(model, 'esm2'):
        try:
            model.esm2.set_attn_implementation('eager')
        except AttributeError:
            model.esm2.config._attn_implementation = 'eager'

    model.eval().to(device)
    variant = 'ESMDance' if 'ESMDance' in hf_repo else 'SeqDance'
    print(f'[{variant}] model loaded from {hf_repo} (freeze_esm={model.freeze_esm}).')
    return model, tokenizer


# ============================================================
# 3.  Residue embedding extraction  (one function per API)
# ============================================================

def get_residue_emb_esm_style(model, alphabet, sequence, device):
    """ESM2-compatible forward pass → per-residue embeddings [L, 1280]."""
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([('p', sequence)])
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
    # Strip CLS (pos 0) and EOS (last non-padding token)
    emb = out['representations'][model.num_layers][0, 1:-1, :].cpu().numpy()
    return emb  # [L, 1280]


def get_residue_emb_esmc(model, tokenizer, sequence, device):
    """ESM-C forward pass → per-residue embeddings [L, D].

    ESM-C wraps sequences in <cls> … <eos> exactly like ESM2, so the same 1:-1 strip applies.
    """
    from esmc_local import get_residue_emb_esmc as _emb
    return _emb(model, tokenizer, sequence, device)


def get_residue_emb_prostt5(model, tokenizer, sequence, device):
    """ProstT5 forward pass → per-residue embeddings [L, D]."""
    import re as _re
    seq_clean = _re.sub(r'[UZOB]', 'X', sequence)
    seq_spaced = '<AA2fold> ' + ' '.join(list(seq_clean))
    ids = tokenizer([seq_spaced], add_special_tokens=True,
                    padding=False, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(ids.input_ids, attention_mask=ids.attention_mask)
    # Strip BOS and EOS tokens
    emb = out.last_hidden_state[0, 1:-1, :].cpu().numpy()
    return emb  # [L, D]


def load_splm(proteins, splm_path, splm_config, splm_checkpoint, device,
              splm_python=None, max_length=1022, cache_pkl=None):
    """Generate SPLM-V2-GVP residue embeddings for all proteins via a SUBPROCESS.

    SPLM cannot be imported in-process: its model.py does `import esm_adapterH` and ships its
    own utils/utils.py, both of which collide with DPLM_ai's already-imported modules (it would
    silently bind to the wrong adapter package); it also imports gvp/torch_geometric at module
    level. So we invoke its documented CLI (README §2.4)
    `python -m utils.generate_seq_embedding --residue_level`, which writes {id: [L, D]}.

    SPLM's sequence branch returns `features_residue = residue_feature[mask]` with
    padding/cls/eos removed (model.py:375-398) — i.e. already [L, D] real residues,
    pre-projection (1280-d for ESM2-650M), matching how DPLM/ESM2 are used here.

    Returns (emb_by_seq, None): a {sequence: np.ndarray[L, D]} lookup. Keying by sequence (not
    pid) lets build_embedding_cache's existing dispatch stay unchanged.
    """
    import pickle
    import subprocess
    import tempfile

    if cache_pkl and os.path.exists(cache_pkl):
        print(f'[SPLM] loading cached embeddings: {cache_pkl}')
        with open(cache_pkl, 'rb') as f:
            emb_by_id = pickle.load(f)
    else:
        python_exe = splm_python or sys.executable
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = os.path.join(tmpdir, 'rmsf_proteins.fasta')
            with open(fasta_path, 'w') as f:
                for prot in proteins:
                    f.write(f">{prot['pid']}\n{prot['sequence']}\n")
            print(f'[SPLM] wrote {len(proteins)} sequences → {fasta_path}')

            out_name = 'splm_residue_emb.pkl'
            cmd = [
                python_exe, '-m', 'utils.generate_seq_embedding',
                '--input_seq', fasta_path,
                '--config_path', splm_config,
                '--checkpoint_path', splm_checkpoint,
                '--result_path', tmpdir,
                '--out_file', out_name,
                '--residue_level',
                '--truncate_inference', '0',
                '--max_length_inference', str(max_length),
            ]
            # PYTHONPATH is OVERRIDDEN (not appended): the LCC runners export
            # PYTHONPATH=<DPLM_ai root>, which would otherwise let this subprocess import
            # DPLM_ai's esm_adapterH/utils instead of SPLM's — the exact collision we are
            # isolating against.
            env = dict(os.environ)
            env['PYTHONPATH'] = splm_path
            print(f'[SPLM] running: {" ".join(cmd)}  (cwd={splm_path})')
            subprocess.run(cmd, cwd=splm_path, env=env, check=True)

            with open(os.path.join(tmpdir, out_name), 'rb') as f:
                emb_by_id = pickle.load(f)

            if cache_pkl:
                os.makedirs(os.path.dirname(os.path.abspath(cache_pkl)), exist_ok=True)
                with open(cache_pkl, 'wb') as f:
                    pickle.dump(emb_by_id, f)
                print(f'[SPLM] cached embeddings → {cache_pkl}')

    # Re-key by sequence so the per-sequence dispatch in build_embedding_cache works unchanged.
    emb_by_seq = {}
    missing = 0
    for prot in proteins:
        emb = emb_by_id.get(prot['pid'])
        if emb is None:
            missing += 1
            continue
        emb_by_seq[prot['sequence']] = np.asarray(emb)
    print(f'[SPLM] loaded embeddings for {len(emb_by_seq)} sequences'
          + (f'  ({missing} pid(s) missing from the pickle)' if missing else ''))
    return emb_by_seq, None


def get_residue_emb_splm(emb_by_seq, _extra, sequence, device):
    """Look up the precomputed SPLM residue embedding for one sequence → [L, D]."""
    emb = emb_by_seq.get(sequence)
    if emb is None:
        raise KeyError('no SPLM embedding for this sequence')
    return np.asarray(emb, dtype=np.float32)


def get_residue_emb_seqdance(model, tokenizer, sequence, device):
    """SeqDance forward pass → per-residue embeddings [L, D]."""
    raw_input = tokenizer([sequence], return_tensors='pt').to(device)
    with torch.no_grad():
        out = model(raw_input, return_res_emb=True,
                    return_attention_map=False,
                    return_res_pred=False,
                    return_pair_pred=False)
    emb = out['res_emb'][:, 1:-1, :].squeeze(0).cpu().numpy()
    return emb  # [L, D]


# ============================================================
# 4.  Build embedding cache
# ============================================================

METHOD_ORDER = ['DPLM', 'ESM2', 'ESMC', 'ProstT5', 'SeqDance', 'SPLM']
METHOD_COLORS = {
    'DPLM':     '#2166ac',
    'ESM2':     '#d6604d',
    'ESMC':     '#b2182b',
    'ProstT5':  '#4dac26',
    'SeqDance': '#9970ab',
    'SPLM':     '#e08214',
}


def build_embedding_cache(proteins, models_dict, device):
    """Compute per-residue embeddings for every protein × method.

    Returns
    -------
    cache : dict
        {pid: {method_name: np.ndarray[L, D]}}
        Proteins where any method returns a length mismatch are excluded.
    valid_proteins : list of dict
        Subset of `proteins` that passed all length checks.
    """
    esm_style_methods = {'DPLM', 'ESM2'}
    cache = {}
    valid_proteins = []

    for prot in proteins:
        pid, seq = prot['pid'], prot['sequence']
        L = len(seq)
        embs = {}
        ok = True

        for method, (model, extra) in models_dict.items():
            try:
                if method in esm_style_methods:
                    emb = get_residue_emb_esm_style(model, extra, seq, device)
                elif method == 'ESMC':
                    emb = get_residue_emb_esmc(model, extra, seq, device)
                elif method == 'ProstT5':
                    emb = get_residue_emb_prostt5(model, extra, seq, device)
                elif method == 'SeqDance':
                    emb = get_residue_emb_seqdance(model, extra, seq, device)
                elif method == 'SPLM':
                    emb = get_residue_emb_splm(model, extra, seq, device)
                else:
                    raise ValueError(f'Unknown method: {method}')
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
# 4b.  Save per-residue feature table (one file per method)
# ============================================================

def save_residue_feature_table(valid_proteins, cache, output_dir, metric='rmsf'):
    """Write, for each method, a per-residue CSV with the raw data behind the plots.

    Columns: protein_id, residue_index, amino_acid, {metric}, emb_norm  (one row/residue).
    `emb_norm` is the raw positive L2 norm of the residue embedding (the scatter/boxplot
    functions negate it for ProstT5/SeqDance for display only — here it is stored unsigned).
    """
    methods = [m for m in METHOD_ORDER if m in next(iter(cache.values()))]
    os.makedirs(output_dir, exist_ok=True)

    for method in methods:
        rows = []
        for prot in valid_proteins:
            pid = prot['pid']
            if pid not in cache or method not in cache[pid]:
                continue
            seq  = prot['sequence']
            vals = prot['metric']                              # [L]
            norms = np.linalg.norm(cache[pid][method], axis=1)  # [L], raw positive norm
            for i in range(len(vals)):
                rows.append((pid, i, seq[i], float(vals[i]), float(norms[i])))

        df = pd.DataFrame(rows, columns=['protein_id', 'residue_index',
                                         'amino_acid', metric, 'emb_norm'])
        out = os.path.join(output_dir, f'residue_features_{method}.csv')
        df.to_csv(out, index=False)
        print(f'[residue table] {method}: {len(df)} residues → {out}')


# ============================================================
# 5.  Analysis 1 — Scatter: emb norm vs RMSF (amino acid level)
# ============================================================

def plot_scatter_norm_vs_metric(proteins, cache, output_dir, metric='rmsf'):
    """2×2 scatter (one subplot per method); each point = one amino acid."""
    methods = [m for m in METHOD_ORDER if m in next(iter(cache.values()))]
    ncols = 2
    nrows = (len(methods) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]
        all_norms, all_metric = [], []

        for prot in proteins:
            pid = prot['pid']
            if pid not in cache or method not in cache[pid]:
                continue
            emb  = cache[pid][method]          # [L, D]
            vals = prot['metric']              # [L]
            # norms = np.linalg.norm(emb, axis=1)
            if method in ['ProstT5', 'SeqDance']:
                norms = -np.linalg.norm(emb, axis=1)
            else:
                norms = np.linalg.norm(emb, axis=1)
            all_norms.extend(norms.tolist())
            all_metric.extend(vals.tolist())

        all_norms  = np.array(all_norms)
        all_metric = np.array(all_metric)
        corr, pval = spearmanr(all_norms, all_metric)

        ax.scatter(all_norms, all_metric, s=3, alpha=0.25,
                   color=METHOD_COLORS[method], rasterized=True)

        # Trend line
        m, b = np.polyfit(all_norms, all_metric, 1)
        x_line = np.linspace(all_norms.min(), all_norms.max(), 200)
        ax.plot(x_line, m * x_line + b, color='black', lw=1.2, linestyle='--')

        ax.set_title(f'{method}\nSpearman r = {corr:.3f}  (p = {pval:.2e})',
                     fontsize=11)
        ax.set_xlabel('Embedding norm')
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.25)

    # Hide unused axes
    for ax in axes[len(methods):]:
        ax.set_visible(False)

    fig.suptitle(f'Residue embedding norm vs {metric.upper()} (all proteins pooled)', fontsize=13)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f'scatter_norm_vs_{metric}.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[Analysis 1] saved → {out}')


# ============================================================
# 6.  Analysis 2 — Protein-wise Spearman boxplot + p-values
# ============================================================

def plot_proteinwise_spearman_boxplot(proteins, cache, output_dir, metric='rmsf'):
    """Boxplot of per-protein Spearman correlations; Mann-Whitney p-values vs DPLM."""
    methods = [m for m in METHOD_ORDER if m in next(iter(cache.values()))]

    per_protein_corrs = {m: [] for m in methods}
    for prot in proteins:
        pid = prot['pid']
        if pid not in cache:
            continue
        vals = prot['metric']
        for method in methods:
            if method not in cache[pid]:
                continue
            emb   = cache[pid][method]
            if method in ['ProstT5', 'SeqDance']:
                norms = -np.linalg.norm(emb, axis=1)
            else:
                norms = np.linalg.norm(emb, axis=1)
            if len(norms) < 3:
                continue
            corr, _ = spearmanr(norms, vals)
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

    # Jitter overlay
    rng = np.random.default_rng(0)
    for i, (vals, color) in enumerate(zip(data, colors)):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(positions[i] + jitter, vals,
                   s=18, alpha=0.6, color=color, zorder=3)

    # Mann-Whitney p-values: compare each method against DPLM
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
    ax.set_ylabel(f"Per-protein Spearman r (norm vs {metric.upper()})", fontsize=11)
    ax.set_title(f"Protein-wise Spearman distribution ({metric.upper()} vs embedding norm)", fontsize=12)
    ax.axhline(0, color='gray', lw=0.8, linestyle='--')
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f'proteinwise_spearman_boxplot_{metric}.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[Analysis 2] saved → {out}')


def _pval_str(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


# ============================================================
# 7.  Analysis 3 — t-SNE coloured by RMSF  (DPLM vs ESM2)
# ============================================================

# def plot_tsne_colored_rmsf(proteins, cache, output_dir, perplexity=30):
#     """Pool all residues; t-SNE on DPLM and ESM2 embeddings; colour by RMSF."""
#     methods = ['DPLM', 'ESM2']
#     available = [m for m in methods if m in next(iter(cache.values()))]
#     if not available:
#         print('[Analysis 3] No DPLM or ESM2 embeddings found — skipping.')
#         return

#     # Pool all residues
#     embs  = {m: [] for m in available}
#     rmsfs = []
#     for prot in proteins:
#         pid = prot['pid']
#         if pid not in cache:
#             continue
#         # Use DPLM as reference length (they were checked equal during caching)
#         rmsfs.extend(prot['rmsf'].tolist())
#         for m in available:
#             if m in cache[pid]:
#                 embs[m].extend(cache[pid][m].tolist())

#     rmsfs = np.array(rmsfs, dtype=np.float32)

#     fig, axes = plt.subplots(1, len(available), figsize=(7 * len(available), 6))
#     if len(available) == 1:
#         axes = [axes]

#     # Shared colorbar limits
#     vmin, vmax = np.percentile(rmsfs, [2, 98])

#     for ax, method in zip(axes, available):
#         X = np.array(embs[method], dtype=np.float32)
#         perp = min(perplexity, X.shape[0] - 1)
#         print(f'  [t-SNE] {method}: {X.shape[0]} residues × {X.shape[1]} dims …')
#         tsne = TSNE(n_components=2, perplexity=perp,
#                     random_state=42, max_iter=1000, verbose=0)
#         coords = tsne.fit_transform(X)   # [N, 2]

#         sc = ax.scatter(coords[:, 0], coords[:, 1],
#                         c=rmsfs, cmap='coolwarm',
#                         vmin=vmin, vmax=vmax,
#                         s=5, alpha=0.4, rasterized=True,
#                         linewidths=0)
#         ax.set_title(f'{method} residue embeddings\ncoloured by RMSF', fontsize=12)
#         ax.set_xlabel('t-SNE 1')
#         ax.set_ylabel('t-SNE 2')
#         ax.grid(True, alpha=0.2)

#     # Shared colorbar
#     cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02)
#     cbar.set_label('RMSF (Å)', fontsize=11)

#     fig.suptitle('t-SNE of residue embeddings coloured by RMSF', fontsize=13)
#     plt.tight_layout()
#     os.makedirs(output_dir, exist_ok=True)
#     out = os.path.join(output_dir, 'tsne_rmsf_dplm_vs_esm2.png')
#     plt.savefig(out, dpi=150)
#     plt.close()
#     print(f'[Analysis 3] saved → {out}')


# ============================================================
# 8.  Analysis 4 — RMSF binning + KMeans + ARI  (DPLM vs ESM2)
# ============================================================

# def plot_kmeans_ari(proteins, cache, output_dir, n_bins=5, perplexity=30):
#     """Bin RMSF into n_bins equal-count bins; KMeans(k=n_bins); compute ARI."""
#     methods = ['DPLM', 'ESM2']
#     available = [m for m in methods if m in next(iter(cache.values()))]
#     if len(available) < 1:
#         print('[Analysis 4] No DPLM or ESM2 embeddings found — skipping.')
#         return

#     # Pool all residues
#     embs  = {m: [] for m in available}
#     rmsfs = []
#     for prot in proteins:
#         pid = prot['pid']
#         if pid not in cache:
#             continue
#         rmsfs.extend(prot['rmsf'].tolist())
#         for m in available:
#             if m in cache[pid]:
#                 embs[m].extend(cache[pid][m].tolist())

#     rmsfs = np.array(rmsfs, dtype=np.float32)

#     # Equal-count binning
#     sorted_idx  = np.argsort(rmsfs)
#     rmsf_bins   = np.empty(len(rmsfs), dtype=int)
#     for b, chunk in enumerate(np.array_split(sorted_idx, n_bins)):
#         rmsf_bins[chunk] = b

#     # KMeans + ARI
#     ari_scores = {}
#     km_labels  = {}
#     tsne_coords = {}

#     for method in available:
#         X = np.array(embs[method], dtype=np.float32)
#         km = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
#         km_labels[method]  = km.fit_predict(X)
#         ari_scores[method] = adjusted_rand_score(rmsf_bins, km_labels[method])
#         print(f'  ARI {method}: {ari_scores[method]:.4f}')

#         perp = min(perplexity, X.shape[0] - 1)
#         print(f'  [t-SNE] {method}: {X.shape[0]} residues × {X.shape[1]} dims …')
#         tsne = TSNE(n_components=2, perplexity=perp,
#                     random_state=42, max_iter=1000, verbose=0)
#         tsne_coords[method] = tsne.fit_transform(X)

#     # Plot 2×2: (RMSF bins, KMeans clusters) × (DPLM, ESM2)
#     bin_cmap = plt.cm.RdYlBu_r
#     bin_colors = [bin_cmap(i / (n_bins - 1)) for i in range(n_bins)]

#     nrows, ncols = 2, len(available)
#     fig, axes = plt.subplots(nrows, ncols,
#                              figsize=(6 * ncols, 5 * nrows))
#     if ncols == 1:
#         axes = axes.reshape(nrows, 1)

#     for col, method in enumerate(available):
#         coords = tsne_coords[method]
#         ari    = ari_scores[method]

#         # Row 0: coloured by RMSF bin (ground truth)
#         ax0 = axes[0, col]
#         for b in range(n_bins):
#             mask = rmsf_bins == b
#             ax0.scatter(coords[mask, 0], coords[mask, 1],
#                         c=[bin_colors[b]], s=5, alpha=0.45,
#                         label=f'RMSF bin {b}', rasterized=True)
#         ax0.set_title(f'{method}  (ARI = {ari:.4f})\nColoured by RMSF bin', fontsize=11)
#         ax0.set_xlabel('t-SNE 1')
#         ax0.set_ylabel('t-SNE 2')
#         ax0.grid(True, alpha=0.2)

#         # Row 1: coloured by KMeans cluster
#         ax1 = axes[1, col]
#         for b in range(n_bins):
#             mask = km_labels[method] == b
#             ax1.scatter(coords[mask, 0], coords[mask, 1],
#                         c=[bin_colors[b]], s=5, alpha=0.45,
#                         label=f'Cluster {b}', rasterized=True)
#         ax1.set_title(f'{method}\nColoured by KMeans cluster', fontsize=11)
#         ax1.set_xlabel('t-SNE 1')
#         ax1.set_ylabel('t-SNE 2')
#         ax1.grid(True, alpha=0.2)

#     # Shared legend
#     legend_patches = [mpatches.Patch(color=bin_colors[b], label=f'Bin {b}')
#                       for b in range(n_bins)]
#     fig.legend(handles=legend_patches, loc='lower center',
#                ncol=n_bins, fontsize=9,
#                title='RMSF bin / KMeans cluster', title_fontsize=9,
#                bbox_to_anchor=(0.5, -0.02))

#     ari_str = '  '.join(f'{m} ARI={ari_scores[m]:.4f}' for m in available)
#     fig.suptitle(f'RMSF bins vs KMeans clusters\n{ari_str}', fontsize=12)
#     plt.tight_layout(rect=[0, 0.04, 1, 1])

#     os.makedirs(output_dir, exist_ok=True)
#     out = os.path.join(output_dir, 'kmeans_ari_dplm_vs_esm2.png')
#     plt.savefig(out, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f'[Analysis 4] saved → {out}')


# ============================================================
# 8b.  Analysis 5 — RSA: residue-pair embedding distance vs metric difference
# ============================================================

def _rsa_protein_corr(emb, vals):
    """Spearman correlation between residue-pair embedding distance and
    residue-pair |metric_i - metric_j| difference, for one protein.

    emb  : [L, D] residue embeddings
    vals : [L]    per-residue metric values

    Unlike the norm-vs-metric correlation (Analyses 1/2), this uses the full
    embedding vector — it tests whether residues that are *close in embedding
    space* also have *similar dynamics*, regardless of which direction or
    magnitude encodes that information.
    """
    L = emb.shape[0]
    if L < 4:
        return np.nan

    # Pairwise cosine distance (1 - cosine similarity), upper triangle only.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    unit = emb / norms
    cos_sim = unit @ unit.T
    emb_dist = 1.0 - cos_sim

    val_diff = np.abs(vals[:, None] - vals[None, :])

    iu = np.triu_indices(L, k=1)
    emb_dist_flat = emb_dist[iu]
    val_diff_flat = val_diff[iu]

    corr, _ = spearmanr(emb_dist_flat, val_diff_flat)
    return corr


def plot_rsa_boxplot(proteins, cache, output_dir, metric='rmsf'):
    """Boxplot of per-protein RSA Spearman correlations; Mann-Whitney p-values vs DPLM.

    RSA = Representational Similarity Analysis: correlates residue-residue
    embedding distance against residue-residue metric difference, per protein,
    then aggregates across proteins exactly like Analysis 2's norm-based boxplot.
    A positive correlation means residues far apart in embedding space tend to
    have very different dynamics (and close residues tend to be similar) —
    the embedding space is organized by the dynamics metric.
    """
    methods = [m for m in METHOD_ORDER if m in next(iter(cache.values()))]

    per_protein_rsa = {m: [] for m in methods}
    for prot in proteins:
        pid = prot['pid']
        if pid not in cache:
            continue
        vals = prot['metric']
        for method in methods:
            if method not in cache[pid]:
                continue
            emb = cache[pid][method]
            corr = _rsa_protein_corr(emb, vals)
            if not np.isnan(corr):
                per_protein_rsa[method].append(corr)

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = list(range(1, len(methods) + 1))
    data = [per_protein_rsa[m] for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]

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
    for i, (rsa_vals, color) in enumerate(zip(data, colors)):
        jitter = rng.uniform(-0.12, 0.12, size=len(rsa_vals))
        ax.scatter(positions[i] + jitter, rsa_vals,
                   s=18, alpha=0.6, color=color, zorder=3)

    # Mann-Whitney p-values: compare each method against DPLM
    dplm_rsa = per_protein_rsa.get('DPLM', [])
    y_max = max(max(v) if v else 0 for v in data)
    bracket_y = y_max + 0.05

    for i, method in enumerate(methods):
        if method == 'DPLM' or not dplm_rsa or not per_protein_rsa[method]:
            continue
        stat, pval = mannwhitneyu(dplm_rsa, per_protein_rsa[method],
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
    ax.set_ylabel(f"Per-protein RSA Spearman r\n(emb distance vs Δ{metric.upper()})", fontsize=11)
    ax.set_title(f"RSA: residue-pair embedding distance vs {metric.upper()} difference", fontsize=12)
    ax.axhline(0, color='gray', lw=0.8, linestyle='--')
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f'rsa_boxplot_{metric}.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[Analysis 5] saved → {out}')


# ============================================================
# 9.  Argument parsing + main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='RMSF vs embedding-norm analysis for DPLM, ESM2, ProstT5, SeqDance.')
    p.add_argument('--data_path',       required=True,
                   help='Dir of trajectory/processed files (used to enumerate protein IDs).')
    p.add_argument('--analysis_path',   required=True,
                   help='Base dir containing {pid}_analysis/{pid}.pdb + {pid}_RMSF.tsv.')
    p.add_argument('--methods',         default=None,
                   help='Restrict which backbones to run, e.g. "esmc" or "dplm+esm2+esmc". '
                        'Choices: dplm, esm2, esmc, prostt5, seqdance, splm. '
                        'Default (omitted) keeps the historic behaviour: DPLM + ESM2 + '
                        'ProstT5, plus SeqDance/SPLM when their paths are given.')
    p.add_argument('--esmc_model',      default='esmc_600m',
                   choices=['esmc_300m', 'esmc_600m'],
                   help='ESM-C size when esmc is selected (default: esmc_600m, 1152-d).')
    p.add_argument('--dplm_config',     default=None,
                   help='Path to training config YAML (for DPLM adapter settings).')
    p.add_argument('--dplm_checkpoint', default=None,
                   help='Path to training checkpoint .pth.')
    p.add_argument('--seqdance_path',   default=None,
                   help='Path to SeqDance-main/model/ directory. '
                        'Omit to skip SeqDance analysis.')
    # ── SPLM-V2-GVP (run out-of-process; omit --splm_path to skip) ──────────
    p.add_argument('--splm_path',       default=None,
                   help='Path to the SPLM-V2-GVP repo root. Omit to skip SPLM analysis.')
    p.add_argument('--splm_config',     default=None,
                   help='SPLM config YAML (e.g. configs/config_plddtallweight_noseq_'
                        'rotary_foldseek.yaml). Required with --splm_path.')
    p.add_argument('--splm_checkpoint', default=None,
                   help='SPLM checkpoint .pth (e.g. checkpoint_0280000_gvp.pth). '
                        'Required with --splm_path.')
    p.add_argument('--splm_python',     default=None,
                   help='Python interpreter for the SPLM subprocess (default: this one). '
                        'Use it if SPLM lives in a different conda env (torch_geometric/gvp).')
    p.add_argument('--splm_max_length', type=int, default=1022,
                   help='SPLM truncation length at inference (default: 1022). Proteins longer '
                        'than this get truncated by SPLM and are then dropped by the '
                        'emb-len == seq-len check.')
    p.add_argument('--splm_cache_pkl',  default=None,
                   help='Reuse/save the generated SPLM embedding pickle here, so repeat runs '
                        'skip regeneration.')
    p.add_argument('--output_dir',      default='./rmsf_output',
                   help='Directory to save all figures.')
    p.add_argument('--max_proteins',    type=int, default=None,
                   help='Cap number of proteins (for quick testing).')
    p.add_argument('--metric',          choices=list(METRIC_FILES), default='rmsf',
                   help='Per-residue target metric to analyze (default: rmsf).')
    p.add_argument('--rmsf_col',        default=None,
                   help='Override: use a single specific column instead of '
                        'averaging replicate columns (e.g. RMSF_R1).')
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
        metric=args.metric,
        rmsf_col=args.rmsf_col,
        max_proteins=args.max_proteins,
    )
    if not proteins:
        print('No valid proteins found. Exiting.')
        return

    # ── Step 2: Load models ───────────────────────────────────────────────────
    print('\n=== Loading models ===')
    models_dict = {}

    # --methods restricts the run to a subset; omitting it keeps the historic default
    # (DPLM + ESM2 + ProstT5, with SeqDance/SPLM gated on their paths being supplied).
    if args.methods:
        selected = {m.strip().lower() for m in re.split(r'[,+;\s]+', args.methods) if m.strip()}
        unknown = selected - {'dplm', 'esm2', 'esmc', 'prostt5', 'seqdance', 'splm'}
        if unknown:
            raise ValueError(f'Unknown --methods entries: {sorted(unknown)}')
    else:
        selected = {'dplm', 'esm2', 'prostt5'}
        if args.seqdance_path:
            selected.add('seqdance')
        if args.splm_path:
            selected.add('splm')
    print(f'Selected methods: {sorted(selected)}')

    if 'dplm' in selected:
        if not (args.dplm_config and args.dplm_checkpoint):
            raise ValueError('dplm requires --dplm_config and --dplm_checkpoint')
        print('Loading DPLM …')
        models_dict['DPLM'] = load_dplm(args.dplm_config, args.dplm_checkpoint, device)

    if 'esm2' in selected:
        print('Loading ESM2 …')
        models_dict['ESM2'] = load_esm2(device)

    if 'esmc' in selected:
        print('Loading ESM-C …')
        models_dict['ESMC'] = load_esmc(device, args.esmc_model)

    if 'prostt5' in selected:
        print('Loading ProstT5 …')
        models_dict['ProstT5'] = load_prostt5(device)

    if 'seqdance' in selected:
        if not args.seqdance_path:
            raise ValueError('seqdance requires --seqdance_path')
        print('Loading SeqDance …')
        models_dict['SeqDance'] = load_seqdance(args.seqdance_path, device)

    if 'splm' in selected:
        assert args.splm_path and args.splm_config and args.splm_checkpoint, \
            '--splm_path, --splm_config and --splm_checkpoint are required for splm.'
        print('Generating SPLM embeddings (subprocess) …')
        models_dict['SPLM'] = load_splm(
            proteins, args.splm_path, args.splm_config, args.splm_checkpoint, device,
            splm_python=args.splm_python, max_length=args.splm_max_length,
            cache_pkl=args.splm_cache_pkl)

    # ── Step 3: Build embedding cache ─────────────────────────────────────────
    print('\n=== Computing residue embeddings ===')
    cache, valid_proteins = build_embedding_cache(proteins, models_dict, device)
    if not valid_proteins:
        print('No valid proteins after embedding. Exiting.')
        return

    # ── Step 3b: Save per-residue feature tables (one CSV per method) ──────────
    print('\n=== Saving per-residue feature tables ===')
    save_residue_feature_table(valid_proteins, cache, args.output_dir, metric=args.metric)

    # ── Step 4: Run analyses ──────────────────────────────────────────────────
    print(f'\n=== Analysis 1: Scatter (norm vs {args.metric.upper()}, amino acid level) ===')
    plot_scatter_norm_vs_metric(valid_proteins, cache, args.output_dir, metric=args.metric)

    print('\n=== Analysis 2: Protein-wise Spearman boxplot ===')
    plot_proteinwise_spearman_boxplot(valid_proteins, cache, args.output_dir, metric=args.metric)

    # print('\n=== Analysis 3: t-SNE coloured by RMSF (DPLM vs ESM2) ===')
    # plot_tsne_colored_rmsf(valid_proteins, cache, args.output_dir)

    # print('\n=== Analysis 4: RMSF binning + KMeans + ARI ===')
    # plot_kmeans_ari(valid_proteins, cache, args.output_dir)

    # print('\n=== Analysis 5: RSA (residue-pair embedding distance vs metric difference) ===')
    # plot_rsa_boxplot(valid_proteins, cache, args.output_dir, metric=args.metric)

    print(f'\nAll figures saved to: {args.output_dir}')


if __name__ == '__main__':
    main()


# ── Example run ───────────────────────────────────────────────────────────────
# PYTHONPATH=. python evaluate/methodology/rmsf.py \
#     --data_path     /path/to/DPLM_data/Atlas_geom2vec_test/ \
#     --analysis_path /path/to/DPLM_data/analysis/ \
#     --dplm_config   ./configs/config_vivit5_delta.yaml \
#     --dplm_checkpoint ./results/vivit5/checkpoints/checkpoint_best_val_whole_loss.pth \
#     --seqdance_path <SeqDance-main/model> \
#     --output_dir    ./evaluate/methodology/rmsf_output/ \
#     --max_proteins  30

# PYTHONPATH=. python evaluate/methodology/rmsf.py \
#     --data_path     /path/to/DPLM_data/processed_test_rep0/ \
#     --analysis_path /path/to/DPLM_data/analysis/ \
#     --dplm_config   ./results/vivit4/config_vivit4_delta.yaml \
#     --dplm_checkpoint ./results/vivit4/checkpoints/checkpoint_best_val_whole_loss.pth \
#     --seqdance_path <SeqDance-main/model> \
#     --output_dir    ./evaluate/methodology/rmsf_output/ \
#     --max_proteins  30
