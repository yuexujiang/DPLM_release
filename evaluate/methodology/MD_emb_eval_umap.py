"""
MD embedding evaluation:
  1. Extract original MD pooled embeddings (768-dim) from h5 files, averaging
     across fragments with 30-residue overlap.
  2. Project them through the trained VIVIT/MD projector → 256-dim.
  3. Extract ESM2 (base) and DPLM (fine-tuned seq model) embeddings → 1280-dim.
  4. t-SNE all four embedding types together and visualize.

Usage:
  python MD_emb_eval.py \
    --data_dir /path/to/processed_data_rep0 \
    --checkpoint /path/to/checkpoint.pth \
    --config_path /path/to/config.yaml \
    --seq_model_location /path/to/checkpoint.pth \
    --output_dir ./md_emb_eval_output \
    [--max_proteins 50]
"""

import os
import sys
import re
import argparse
import glob
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
import umap
from sklearn.preprocessing import normalize
from scipy.stats import spearmanr

# Project root on path so we can import model, utils, esm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import esm
import esm_adapterH
from utils.utils import load_configs
from peft import LoraConfig, get_peft_model

# Same-directory import: reuse the DCCM computation/caching machinery (compute from
# trajectory + cache to disk) instead of duplicating it.
sys.path.insert(0, os.path.dirname(__file__))
from protein_level_emb_md import load_protein_dccm  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal MoBYMLP – mirrors model.py so we can load state_x without importing
# the full model module (which depends on torch_geometric, gvp, etc.)
# ---------------------------------------------------------------------------
class MoBYMLP(nn.Module):
    def __init__(self, in_dim=768, inner_dim=512, out_dim=256, num_layers=2):
        super().__init__()
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)
        self.linear_out = nn.Linear(in_dim if num_layers == 1 else inner_dim, out_dim)

    def forward(self, x):
        return self.linear_out(self.linear_hidden(x))


# ---------------------------------------------------------------------------
# 1.  Load and assemble protein-level MD embeddings from h5 fragment files
# ---------------------------------------------------------------------------

def load_md_embeddings(data_dir, max_proteins=None):
    """
    Returns:
        pids        : list of protein IDs (e.g. "16pk_A"),  len N
        md_embs     : np.ndarray [N, 768]  averaged over fragments
        pid_to_seq  : dict protein_id -> sequence (from frag0 or longest frag)
    """
    # group files by protein id (first '#'-split part)
    frag_groups = defaultdict(list)
    for fpath in glob.glob(os.path.join(data_dir, "*.h5")):
        fname = os.path.basename(fpath)
        parts = fname.split("#")
        protein_id = parts[0]            # e.g. "16pk_A"
        frag_idx = int(parts[1].replace("frag", ""))  # numeric fragment index
        frag_groups[protein_id].append((frag_idx, fpath))

    if max_proteins is not None:
        protein_ids = sorted(frag_groups.keys())[:max_proteins]
    else:
        protein_ids = sorted(frag_groups.keys())

    pids, md_embs, pid_to_seq = [], [], {}

    for protein_id in protein_ids:
        frags = sorted(frag_groups[protein_id], key=lambda x: x[0])  # sort by frag idx
        pooled_list = []
        seq_longest = ""

        for frag_idx, fpath in frags:
            fname = os.path.basename(fpath)
            h5key = "#".join(fname.split("#")[:2])  # "protein#fragN"
            with h5py.File(fpath, "r") as f:
                pooled = f[h5key]["pooled"][:]      # (768,)
                seq = f[h5key].attrs["seq"]

            pooled_list.append(pooled.astype(np.float32))
            if len(seq) > len(seq_longest):
                seq_longest = seq

        if len(pooled_list) == 1:
            protein_emb = pooled_list[0]
        else:
            # Multiple fragments overlap by 30 residues.
            # The pooled vector is already averaged over residues inside each
            # fragment, so we weight each fragment by its non-overlapping
            # residue count (total - 30 overlap per boundary) and average.
            n_frags = len(pooled_list)
            # Each interior fragment contributes one overlap on each side,
            # frag-0 and last-frag contribute only one overlap.
            # Simplest correct approach: straight average (all frags have the
            # same nominal length; the overlap is already pooled in).
            protein_emb = np.mean(np.stack(pooled_list, axis=0), axis=0)

        pids.append(protein_id)
        md_embs.append(protein_emb)
        pid_to_seq[protein_id] = seq_longest

    return pids, np.stack(md_embs, axis=0), pid_to_seq


def load_md_embeddings_len(data_dir, pro_len):
    """Like load_md_embeddings, but selects proteins whose sequence length
    is within [pro_len - 30, pro_len + 30] amino acids.

    Sequence length is taken as the longest fragment's sequence length stored
    in the h5 file (exact for single-fragment proteins; a lower bound for
    multi-fragment proteins, but sufficient for relative size filtering).

    Args:
        data_dir : str  — folder of *.h5 files
        pro_len  : int  — target sequence length (centre of the ±30 window)

    Returns:
        pids       : list[str]         protein IDs that pass the length filter
        md_embs    : np.ndarray [N, 768]
        pid_to_seq : dict  protein_id -> sequence string
    """
    lo, hi = pro_len - 5, pro_len + 5

    # Group h5 files by protein id (same logic as load_md_embeddings)
    frag_groups = defaultdict(list)
    for fpath in glob.glob(os.path.join(data_dir, "*.h5")):
        fname = os.path.basename(fpath)
        parts = fname.split("#")
        protein_id = parts[0]
        frag_idx = int(parts[1].replace("frag", ""))
        frag_groups[protein_id].append((frag_idx, fpath))

    pids, md_embs, pid_to_seq = [], [], {}

    for protein_id in sorted(frag_groups.keys()):
        frags = sorted(frag_groups[protein_id], key=lambda x: x[0])
        pooled_list = []
        seq_longest = ""

        for frag_idx, fpath in frags:
            fname = os.path.basename(fpath)
            h5key = "#".join(fname.split("#")[:2])
            with h5py.File(fpath, "r") as f:
                pooled = f[h5key]["pooled"][:]   # (768,)
                seq    = f[h5key].attrs["seq"]
            pooled_list.append(pooled.astype(np.float32))
            if len(seq) > len(seq_longest):
                seq_longest = seq

        # Length filter — use longest fragment length as proxy for protein size
        if not (lo <= len(seq_longest) <= hi):
            continue

        if len(pooled_list) == 1:
            protein_emb = pooled_list[0]
        else:
            protein_emb = np.mean(np.stack(pooled_list, axis=0), axis=0)

        pids.append(protein_id)
        md_embs.append(protein_emb)
        pid_to_seq[protein_id] = seq_longest

    print(f"  [load_md_embeddings_len] pro_len={pro_len} ±30  →  "
          f"{len(pids)} proteins selected (length range [{lo}, {hi}])")
    return pids, np.stack(md_embs, axis=0), pid_to_seq


# ---------------------------------------------------------------------------
# 2.  Project MD embeddings with the trained projector (VIVIT/MD branch)
# ---------------------------------------------------------------------------

def load_md_projector(config_path, checkpoint_path, in_dim=768, device="cpu"):
    with open(config_path) as f:
        config_file = yaml.full_load(f)
    configs = load_configs(config_file, args=None)

    inner_dim = configs.model.MD_encoder.protein_inner_dim
    out_dim = configs.model.protein_out_dim
    num_layers = configs.model.protein_num_projector
    """Load projectors_protein MLP from checkpoint state_x."""
    projector = MoBYMLP(in_dim=in_dim, inner_dim=inner_dim,
                        out_dim=out_dim, num_layers=num_layers)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "state_x" in ckpt, "Checkpoint must contain 'state_x' for the MD model."
    proj_sd = OrderedDict(
        {k.replace("projectors_protein.", ""): v
         for k, v in ckpt["state_x"].items()
         if k.startswith("projectors_protein.")}
    )
    projector.load_state_dict(proj_sd, strict=True)
    projector.eval()
    projector.to(device)
    print(f"[MD projector] loaded from {checkpoint_path}")
    return projector


def get_projected_md_embs(md_embs_np, projector, device="cpu", batch_size=64):
    """md_embs_np: [N, 768] → returns np.ndarray [N, 256]."""
    out = []
    projector.eval()
    with torch.no_grad():
        for i in range(0, len(md_embs_np), batch_size):
            batch = torch.tensor(md_embs_np[i:i + batch_size],
                                 dtype=torch.float32).to(device)
            out.append(projector(batch).cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# 3.  Load ESM2 / DPLM seq model and extract embeddings
# ---------------------------------------------------------------------------

def load_esm2_base(device="cpu", model_name='esm2_t33_650M_UR50D'):
    """Plain ESM2, no fine-tuning. model_name defaults to 650M."""
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval()
    model.to(device)
    print(f"[ESM2 base] loaded ({model_name})")
    return model, alphabet


def load_dplm_model(checkpoint_path, config_path, device="cpu"):
    """
    Load DPLM = ESM2 + adapter fine-tuned from checkpoint state_dict1.
    Also returns the seq projectors_protein MLP so we can get 256-dim outputs.
    """
    with open(config_path) as f:
        config_file = yaml.full_load(f)
    configs = load_configs(config_file, args=None)

    enc = configs.model.esm_encoder
    model_name = getattr(enc, 'model_name', 'esm2_t33_650M_UR50D')

    if enc.adapter_h.enable:
        model, alphabet = getattr(esm_adapterH.pretrained, model_name)(enc.adapter_h)
    elif enc.lora.enable:
        model, alphabet = getattr(esm.pretrained, model_name)()
        lora_targets = ["self_attn.q_proj", "self_attn.k_proj",
                        "self_attn.v_proj", "self_attn.out_proj"]
        target_modules = []
        if enc.lora.esm_num_end_lora > 0:
            start = max(model.num_layers - enc.lora.esm_num_end_lora, 0)
            for idx in range(start, model.num_layers):
                for name in lora_targets:
                    target_modules.append(f"layers.{idx}.{name}")
        peft_config = LoraConfig(
            inference_mode=False,
            r=enc.lora.r,
            lora_alpha=enc.lora.alpha,
            target_modules=target_modules,
            lora_dropout=enc.lora.dropout,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
    else:
        model, alphabet = getattr(esm.pretrained, model_name)()

    # Load backbone weights from state_dict1
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "state_dict1" in ckpt, "Checkpoint must contain 'state_dict1'."
    sd = ckpt["state_dict1"]

    if any("adapter" in n for n, _ in model.named_modules()):
        if not any("adapter_layer_dict" in k for k in sd):
            new_sd = OrderedDict()
            for k, v in sd.items():
                new_k = k.replace("adapter_layer", "adapter_layer_dict.adapter_0") \
                    if "adapter_layer_dict" not in k else k
                new_sd[new_k] = v
            sd = new_sd

    model.load_state_dict(sd, strict=False)
    model.eval()
    model.to(device)

    # Build and load the protein-level projector from state_dict1.
    # Keys are "projectors_protein.*" inside state_dict1.
    esm_embed_dim = model.embed_dim  # 1280 for esm2_t33_650M
    seq_projector = MoBYMLP(
        in_dim=esm_embed_dim,
        inner_dim=configs.model.esm_encoder.protein_inner_dim,
        out_dim=configs.model.protein_out_dim,
        num_layers=configs.model.protein_num_projector,
    )
    proj_sd = OrderedDict(
        {k.replace("projectors_protein.", ""): v
         for k, v in ckpt["state_dict1"].items()
         if k.startswith("projectors_protein.")}
    )
    seq_projector.load_state_dict(proj_sd, strict=True)
    seq_projector.eval()
    seq_projector.to(device)

    print(f"[DPLM seq model + projector] loaded from {checkpoint_path}")
    return model, alphabet, seq_projector


def encode_sequences(sequences, model, alphabet, device="cpu", batch_size=8):
    """
    sequences: list of str
    Returns: np.ndarray [N, 1280]  (mean-pooled over residues, excluding CLS/EOS)
    """
    batch_converter = alphabet.get_batch_converter()
    all_embs = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        data = [(f"p{j}", s) for j, s in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens,
                            repr_layers=[model.num_layers],
                            return_contacts=False)
        reps = results["representations"][model.num_layers]  # [B, L, 1280]

        # mean pool excluding CLS (pos 0) and EOS (last valid token)
        for b in range(reps.shape[0]):
            mask = ((tokens[b] != alphabet.padding_idx) &
                    (tokens[b] != alphabet.cls_idx) &
                    (tokens[b] != alphabet.eos_idx))
            emb = reps[b][mask].mean(0).cpu().numpy()
            all_embs.append(emb)

    return np.stack(all_embs, axis=0)


def get_projected_seq_embs(raw_embs_np, projector, device="cpu", batch_size=64):
    """raw_embs_np: [N, 1280] → returns np.ndarray [N, 256]."""
    out = []
    projector.eval()
    with torch.no_grad():
        for i in range(0, len(raw_embs_np), batch_size):
            batch = torch.tensor(raw_embs_np[i:i + batch_size],
                                 dtype=torch.float32).to(device)
            out.append(projector(batch).cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# 4.  t-SNE visualization
# ---------------------------------------------------------------------------

def tsne_plot(pids, md_raw, md_proj, esm2_emb, dplm_emb, output_path, esmc_emb=None):
    """
    Runs a single joint t-SNE on all embedding types (after L2 normalisation).

    Visual encoding:
      - Colour encodes embedding type (fixed):
          blue       : MD raw (original VIVIT/MD pooled, 768-d)
          lightblue  : MD projected (256-d)
          red        : ESM2 base (original, 1280-d)
          lightcoral : DPLM projected (256-d)
          darkred    : ESM-C base (960/1152-d) — only when `esmc_emb` is given
      - Each point is annotated with its protein index so the same number
        appears once per embedding type for the same protein.

    `esmc_emb` is optional so runs that do not request ESM-C produce exactly the same
    four-block figure as before.
    """
    N = len(pids)
    assert md_raw.shape[0] == N
    assert md_proj.shape[0] == N
    assert esm2_emb.shape[0] == N
    assert dplm_emb.shape[0] == N
    if esmc_emb is not None:
        assert esmc_emb.shape[0] == N

    # (label, array, colour, marker, markersize) — order fixes the block layout below.
    blocks = [
        ("MD raw (768-d)",         md_raw,   "steelblue",    "o", 9),
        ("MD projected (256-d)",   md_proj,  "lightskyblue", "o", 9),
        ("ESM2 base (1280-d)",     esm2_emb, "firebrick",    "s", 9),
        ("DPLM projected (256-d)", dplm_emb, "lightcoral",   "s", 9),
    ]
    if esmc_emb is not None:
        blocks.append((f"ESM-C base ({esmc_emb.shape[1]}-d)", esmc_emb, "darkred", "^", 9))

    # L2-normalise then zero-pad all blocks to the same width
    def l2(x):
        return normalize(x.astype(np.float32), norm="l2")

    max_D = max(arr.shape[1] for _, arr, _, _, _ in blocks)

    def pad(x):
        x = l2(x)
        if x.shape[1] < max_D:
            x = np.concatenate(
                [x, np.zeros((x.shape[0], max_D - x.shape[1]), dtype=np.float32)],
                axis=1)
        return x

    combined = np.concatenate([pad(arr) for _, arr, _, _, _ in blocks], axis=0)

    print(f"Running UMAP on {combined.shape[0]} points × {combined.shape[1]} dims …")
    reducer = umap.UMAP(n_components=2, n_neighbors=min(15, N - 1),
                        metric="cosine", random_state=42, verbose=True)
    coords = reducer.fit_transform(combined)   # [len(blocks)*N, 2]

    # Fixed colours per embedding type
    TYPE_STYLE = [
        (label, coords[i * N:(i + 1) * N], colour, marker, ms)
        for i, (label, _, colour, marker, ms) in enumerate(blocks)
    ]

    fig, ax = plt.subplots(figsize=(11, 9))

    # Draw all scatter points first (zorder=2), then labels on top (zorder=3)
    for label, coords_block, colour, marker, ms in TYPE_STYLE:
        ax.scatter(coords_block[:, 0], coords_block[:, 1],
                   c=colour, marker=marker, s=ms ** 2,
                   zorder=2, linewidths=0.4, edgecolors="white",
                   label=label)

    # Annotate each point with its protein index
    fontsize = max(5, min(8, 120 // N))   # shrink font if many proteins
    for i in range(N):
        label_str = str(i)
        for _, coords_block, colour, _, _ in TYPE_STYLE:
            ax.annotate(
                label_str,
                xy=(coords_block[i, 0], coords_block[i, 1]),
                fontsize=fontsize,
                ha="center", va="center",
                color="black",
                zorder=3,
            )

    ax.legend(loc="upper right", framealpha=0.85, fontsize=9,
              markerscale=1.2)
    ax.set_title(
        "UMAP (cosine): " + " · ".join(lbl for lbl, _, _, _, _ in TYPE_STYLE) + "\n"
        f"Numbers identify proteins — same number = same protein across all "
        f"{len(TYPE_STYLE)} views",
        fontsize=11,
    )
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    fig_path = os.path.join(output_path, "umap_emb_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[UMAP] figure saved → {fig_path}")

    # Also save a protein-index legend as a separate text file
    legend_path = os.path.join(output_path, "protein_index_legend.txt")
    with open(legend_path, "w") as f:
        for i, pid in enumerate(pids):
            f.write(f"{i}\t{pid}\n")
    print(f"[t-SNE] protein index legend saved → {legend_path}")


# ---------------------------------------------------------------------------
# 5.  Load avg_rmsf / avg_gyr from atlas_proteins.csv
# ---------------------------------------------------------------------------

def load_rmsf_gyr(csv_path, pids):
    """
    Read atlas_proteins.csv and return aligned avg_rmsf and avg_gyr arrays.

    The CSV `pdb` column uses format "16pkA" (no underscore).
    Protein IDs in code use "16pk_A".
    Mapping: insert '_' before the last uppercase letter.
      e.g.  "16pkA"  → "16pk_A"
            "1b2sE"  → "1b2s_E"

    Returns:
        rmsf_vals : np.ndarray [N]  (np.nan where protein not found in CSV)
        gyr_vals  : np.ndarray [N]  (np.nan where protein not found in CSV)
    """
    df = pd.read_csv(csv_path)
    pid_map = {}
    for _, row in df.iterrows():
        pdb_raw = str(row["pdb"]).strip()
        # Insert '_' before the trailing uppercase chain letter
        pid_converted = re.sub(r'([A-Z])$', r'_\1', pdb_raw)
        pid_map[pid_converted] = (float(row["avg_rmsf"]), float(row["avg_gyr"]))

    rmsf_vals, gyr_vals = [], []
    missing = []
    for pid in pids:
        if pid in pid_map:
            rmsf_vals.append(pid_map[pid][0])
            gyr_vals.append(pid_map[pid][1])
        else:
            rmsf_vals.append(np.nan)
            gyr_vals.append(np.nan)
            missing.append(pid)

    if missing:
        print(f"  [load_rmsf_gyr] {len(missing)} protein(s) not found in CSV: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

    return np.array(rmsf_vals, dtype=np.float32), np.array(gyr_vals, dtype=np.float32)


def load_rmsd_gyr_std(analysis_path, pids, rmsd_col=None, gyr_col=None):
    """Per-protein RMSD-std and gyration-std, computed directly from
    {analysis_path}/{pid}_analysis/{pid}_RMSD.tsv and {pid}_gyrate.tsv.

    rmsd_col/gyr_col: a single column name to use (e.g. 'RMSD_R1'), or None to
    average the per-replicate std across RMSD_R1/R2/R3 (resp. gyr_R1/R2/R3) —
    same convention as rmsf.py's --rmsf_col.

    Returns: rmsd_std_vals, gyr_std_vals — np.ndarray [N], np.nan where missing.
    """
    default_rmsd_cols = ['RMSD_R1', 'RMSD_R2', 'RMSD_R3']
    default_gyr_cols  = ['gyr_R1', 'gyr_R2', 'gyr_R3']

    rmsd_std_vals, gyr_std_vals = [], []
    missing = []
    for pid in pids:
        adir = os.path.join(analysis_path, f'{pid}_analysis')
        rmsd_file = os.path.join(adir, f'{pid}_RMSD.tsv')
        gyr_file  = os.path.join(adir, f'{pid}_gyrate.tsv')

        rmsd_std = gyr_std = np.nan
        try:
            if os.path.exists(rmsd_file):
                df = pd.read_csv(rmsd_file, sep='\t')
                cols = [rmsd_col] if rmsd_col else default_rmsd_cols
                cols = [c for c in cols if c in df.columns]
                if cols:
                    rmsd_std = float(np.nanmean(np.nanstd(df[cols].values, axis=0)))
            if os.path.exists(gyr_file):
                df = pd.read_csv(gyr_file, sep='\t')
                cols = [gyr_col] if gyr_col else default_gyr_cols
                cols = [c for c in cols if c in df.columns]
                if cols:
                    gyr_std = float(np.nanmean(np.nanstd(df[cols].values, axis=0)))
        except Exception as e:
            print(f'  [load_rmsd_gyr_std] {pid}: {e}')

        if np.isnan(rmsd_std) or np.isnan(gyr_std):
            missing.append(pid)
        rmsd_std_vals.append(rmsd_std)
        gyr_std_vals.append(gyr_std)

    if missing:
        print(f'  [load_rmsd_gyr_std] {len(missing)} protein(s) missing data: '
              f'{missing[:5]}{"..." if len(missing) > 5 else ""}')

    return np.array(rmsd_std_vals, dtype=np.float32), np.array(gyr_std_vals, dtype=np.float32)


# ---------------------------------------------------------------------------
# 5c.  Protein-level RSA: pooled embeddings vs a DCCM-derived descriptor
# ---------------------------------------------------------------------------
#
# md_raw/dplm_raw here are pooled to one vector per protein (no per-residue
# dimension), so the residue-pair DCCM correlation in protein_level_emb_md.py
# (_emb_dccm_corr, needs an [L, D] embedding matrix) cannot be reused directly.
# Instead, reduce each protein's [L, L] DCCM matrix to a small descriptor vector
# and run the same protein-protein RSA pattern protein_level_emb_md.py used for
# its (now-disabled) RMSF/Bfactor/Neq/RMSD/Rg descriptor.

def compute_dccm_descriptor(dccm):
    """Reduce one protein's [L, L] DCCM matrix to a 3-d descriptor:
        [mean(|off-diag DCCM|), frac(|off-diag DCCM| > 0.5), top-eigenvalue fraction]

    - mean(|off-diag|): overall magnitude of inter-residue correlated motion.
    - frac(|off-diag| > 0.5): how many residue pairs move strongly in concert.
    - top eigenvalue fraction: how much of the total correlated-motion "weight" is
      captured by the single most dominant collective mode — high for rigid-domain-like
      motion, low for diffuse/decoupled local fluctuations.
    """
    L = dccm.shape[0]
    iu = np.triu_indices(L, k=1)
    offdiag = dccm[iu]
    mean_abs = float(np.mean(np.abs(offdiag)))
    frac_strong = float(np.mean(np.abs(offdiag) > 0.5))
    eigvals = np.linalg.eigvalsh(dccm)
    top_eig_frac = float(eigvals[-1] / np.sum(np.abs(eigvals)))
    return np.array([mean_abs, frac_strong, top_eig_frac], dtype=np.float32)


def load_dccm_descriptors(analysis_path, pids, dccm_dir=None, replicate=None):
    """Per-protein DCCM descriptor matrix [N, 3]; NaN row where DCCM is unavailable
    (missing trajectory files etc.) — same missing-data convention as load_rmsd_gyr_std."""
    descriptors = []
    missing = []
    for pid in pids:
        dccm = load_protein_dccm(pid, analysis_path, dccm_dir=dccm_dir, replicate=replicate)
        if dccm is not None:
            descriptors.append(compute_dccm_descriptor(dccm))
        else:
            descriptors.append(np.full(3, np.nan, dtype=np.float32))
            missing.append(pid)

    if missing:
        print(f'  [load_dccm_descriptors] {len(missing)} protein(s) missing DCCM data: '
              f'{missing[:5]}{"..." if len(missing) > 5 else ""}')

    return np.stack(descriptors, axis=0)


def _cosine_sim_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    unit = X / norms
    return unit @ unit.T


def _descriptor_sim_matrix(D):
    """Negative Euclidean distance on z-scored descriptor columns → similarity."""
    mu = D.mean(axis=0, keepdims=True)
    sigma = D.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1e-8
    Dz = (D - mu) / sigma
    diff = Dz[:, None, :] - Dz[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return -dist


def compute_emb_dccm_rsa(emb_matrix, descriptor_matrix, n_permutations=1000, seed=0):
    """RSA Spearman + Mantel permutation p-value between protein-protein embedding
    similarity and protein-protein DCCM-descriptor similarity.

    Drops proteins with any-NaN descriptor row before computing similarity matrices.

    Returns
    -------
    dict: {'rsa': float, 'pval': float, 'n_proteins': int}  or None if too few proteins.
    """
    valid_mask = ~np.isnan(descriptor_matrix).any(axis=1)
    n = int(valid_mask.sum())
    if n < 4:
        print(f'  [compute_emb_dccm_rsa] only {n} proteins with valid DCCM descriptors '
              f'(<4) — skipping.')
        return None

    emb_valid = emb_matrix[valid_mask]
    desc_valid = descriptor_matrix[valid_mask]

    emb_sim = _cosine_sim_matrix(emb_valid)
    desc_sim = _descriptor_sim_matrix(desc_valid)
    iu = np.triu_indices(n, k=1)
    emb_sim_flat = emb_sim[iu]
    desc_sim_flat = desc_sim[iu]

    rsa, _ = spearmanr(emb_sim_flat, desc_sim_flat)

    rng = np.random.default_rng(seed)
    perm_rsa = np.empty(n_permutations, dtype=np.float32)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        emb_sim_perm = emb_sim[np.ix_(perm, perm)][iu]
        perm_rsa[i], _ = spearmanr(emb_sim_perm, desc_sim_flat)
    pval = float(np.mean(np.abs(perm_rsa) >= np.abs(rsa)))

    print(f'  RSA = {rsa:.4f}  (Mantel p = {pval:.4f}, n = {n} proteins)')
    return {'rsa': float(rsa), 'pval': pval, 'n_proteins': n}


def _pval_str(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def plot_dccm_rsa_bar(rsa_results, output_dir):
    """Bar chart of RSA scores, one bar per embedding type. `rsa_results` values that
    are None (too few valid proteins) are skipped."""
    labels = [k for k, v in rsa_results.items() if v is not None]
    if not labels:
        print('[DCCM RSA] no embedding type had enough valid proteins — skipping plot.')
        return

    vals = [rsa_results[k]['rsa'] for k in labels]
    colors = ['steelblue', 'lightcoral', 'firebrick', 'lightskyblue'][:len(labels)]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)

    for bar, k in zip(bars, labels):
        pval = rsa_results[k]['pval']
        p_str = _pval_str(pval)
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                height + (0.01 if height >= 0 else -0.03),
                f'{height:.3f}\n{p_str}',
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)

    ax.axhline(0, color='gray', lw=0.8, linestyle='--')
    ax.set_ylabel('RSA Spearman r\n(pooled-emb similarity vs DCCM-descriptor similarity)',
                  fontsize=11)
    n_proteins = rsa_results[labels[0]]['n_proteins']
    ax.set_title(f'Protein-level DCCM RSA  (n = {n_proteins} proteins)', fontsize=12)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, 'dccm_rsa_bar.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[DCCM RSA] saved → {out}')


# ---------------------------------------------------------------------------
# 6.  t-SNE coloured by a continuous biophysical property
# ---------------------------------------------------------------------------

def tsne_property_plot(embs_np, property_vals, emb_label, property_name,
                       pids, output_path):
    """
    Run t-SNE on `embs_np` and draw a scatter plot coloured by `property_vals`.

    Points with NaN property values are drawn in grey and not included in the
    colorbar scaling.

    Args:
        embs_np       : np.ndarray [N, D]
        property_vals : np.ndarray [N]  (may contain np.nan)
        emb_label     : str  e.g. "MD_raw" or "DPLM_raw_1280"
        property_name : str  e.g. "avg_rmsf" or "avg_gyr"
        pids          : list of str  protein IDs (for the index annotation)
        output_path   : directory to save the figure
    """
    N = len(pids)
    assert embs_np.shape[0] == N

    embs_l2 = normalize(embs_np.astype(np.float32), norm="l2")

    print(f"  Running UMAP for {emb_label} × {property_name} "
          f"({N} proteins) …")
    reducer = umap.UMAP(n_components=2, n_neighbors=min(15, N - 1),
                        metric="cosine", random_state=42, verbose=False)
    coords = reducer.fit_transform(embs_l2)   # [N, 2]

    valid_mask = ~np.isnan(property_vals)
    vals_valid = property_vals[valid_mask]

    fig, ax = plt.subplots(figsize=(9, 7))

    # Grey points for proteins with no property data
    if (~valid_mask).any():
        ax.scatter(coords[~valid_mask, 0], coords[~valid_mask, 1],
                   c="lightgrey", s=8 ** 2, zorder=2,
                   linewidths=0.3, edgecolors="white", label="no data")

    # Coloured points — use plasma (sequential, perceptually uniform) and clip
    # to 2nd–98th percentile so outliers don't compress the bulk of the data
    # into a narrow colour band.
    vmin = float(np.percentile(vals_valid, 2))
    vmax = float(np.percentile(vals_valid, 98))
    sc = ax.scatter(coords[valid_mask, 0], coords[valid_mask, 1],
                    c=vals_valid, cmap="plasma",
                    vmin=vmin, vmax=vmax,
                    s=8 ** 2, zorder=3,
                    linewidths=0.3, edgecolors="white")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, extend="both")
    cbar.set_label(property_name, fontsize=10)

    # Protein-index annotations
    fontsize = max(5, min(8, 120 // N))
    for i in range(N):
        ax.annotate(str(i), xy=(coords[i, 0], coords[i, 1]),
                    fontsize=fontsize, ha="center", va="center",
                    color="black", zorder=4)

    ax.set_title(f"UMAP (cosine) of {emb_label} coloured by {property_name}", fontsize=11)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.grid(True, alpha=0.25)
    if (~valid_mask).any():
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    os.makedirs(output_path, exist_ok=True)
    safe_label = emb_label.replace(" ", "_")
    fname = f"umap_{safe_label}_{property_name}.png"
    fig_path = os.path.join(output_path, fname)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  [UMAP property] saved → {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MD embedding evaluation and t-SNE")
    p.add_argument("--data_dir", required=True,
                   help="Folder of processed h5 files, e.g. …/processed_data_rep0")
    p.add_argument("--checkpoint", required=True,
                   help="Checkpoint .pth containing state_x and state_dict1")
    p.add_argument("--config_path", required=True,
                   help="YAML config used for training (for DPLM seq model)")
    # Optional separate checkpoint for the seq model (defaults to same checkpoint)
    p.add_argument("--seq_checkpoint", default=None,
                   help="Separate checkpoint for the seq model (default: same as --checkpoint)")
    p.add_argument("--output_dir", default="./md_emb_eval_output",
                   help="Directory to save the figure and optional embeddings")
    p.add_argument("--max_proteins", type=int, default=None,
                   help="Limit number of proteins (useful for quick testing)")
    p.add_argument("--pro_len", type=int, default=None,
                   help="Select proteins whose sequence length is within pro_len ± 30 aa "
                        "(takes precedence over --max_proteins when both are given)")
    p.add_argument("--esmc", action="store_true",
                   help="Also extract vanilla ESM-C mean-pooled embeddings and add them as "
                        "a fifth block in the joint UMAP (see esmc_local/).")
    p.add_argument("--esmc_model", default="esmc_600m",
                   choices=["esmc_300m", "esmc_600m"],
                   help="ESM-C size when --esmc is set (default: esmc_600m, 1152-d).")
    p.add_argument("--save_npy", action="store_true",
                   help="Also save embeddings as .npy files")
    p.add_argument("--csv_path", default=None,
                   help="Path to atlas_proteins.csv for avg_rmsf/avg_gyr coloured t-SNE plots")
    p.add_argument("--analysis_path", default=None,
                   help="Base dir containing {pid}_analysis/{pid}_RMSD.tsv + "
                        "{pid}_gyrate.tsv. Enables RMSD-std/gyr-std coloured UMAP plots.")
    p.add_argument("--rmsd_col", default=None,
                   help="Single column to use from {pid}_RMSD.tsv (e.g. RMSD_R1). "
                        "Omit to average over RMSD_R1/R2/R3.")
    p.add_argument("--gyr_col", default=None,
                   help="Single column to use from {pid}_gyrate.tsv (e.g. gyr_R1). "
                        "Omit to average over gyr_R1/R2/R3.")
    p.add_argument("--dccm_dir", default=None,
                   help="Directory to cache/load precomputed DCCM .npy files "
                        "({pid}_dccm_R1/R2/R3.npy). Omit to compute in memory "
                        "every run without caching. Enables Step 6 (protein-level "
                        "DCCM RSA) together with --analysis_path.")
    p.add_argument("--dccm_replicate", default=None, choices=["R1", "R2", "R3"],
                   help="Single MD replicate to use for DCCM. Omit to average "
                        "the DCCM matrices from R1/R2/R3.")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    seq_ckpt = args.seq_checkpoint or args.checkpoint

    # --- Step 1: original MD embeddings -----------------------------------------
    print("\n=== Step 1: Loading MD fragment embeddings ===")
    if args.pro_len is not None:
        pids, md_raw, pid_to_seq = load_md_embeddings_len(args.data_dir, args.pro_len)
    else:
        pids, md_raw, pid_to_seq = load_md_embeddings(args.data_dir, args.max_proteins)
    sequences = [pid_to_seq[p] for p in pids]
    print(f"  Proteins loaded : {len(pids)}")
    print(f"  MD raw shape    : {md_raw.shape}")   # [N, 768]

    # --- Step 2: projected MD embeddings ----------------------------------------
    print("\n=== Step 2: Projecting MD embeddings through trained MLP ===")
    projector = load_md_projector(
        args.config_path,
        args.checkpoint,
        in_dim=768,
        device=device,
    )
    md_proj = get_projected_md_embs(md_raw, projector, device=device)
    print(f"  MD projected shape : {md_proj.shape}")   # [N, 256]

    # --- Step 3: ESM2 base embeddings -------------------------------------------
    print("\n=== Step 3: Extracting ESM2 base embeddings ===")
    esm2_model, esm2_alphabet = load_esm2_base(device=device)
    esm2_emb = encode_sequences(sequences, esm2_model, esm2_alphabet,
                                 device=device)
    print(f"  ESM2 emb shape : {esm2_emb.shape}")   # [N, 1280]

    # --- DPLM seq embeddings (projected, 256-d) ----------------------------------
    print("\n=== Step 3b: Extracting DPLM projected seq embeddings ===")
    dplm_model, dplm_alphabet, seq_projector = load_dplm_model(
        seq_ckpt, args.config_path, device=device)
    dplm_raw = encode_sequences(sequences, dplm_model, dplm_alphabet,
                                device=device)                  # [N, 1280]
    dplm_emb = get_projected_seq_embs(dplm_raw, seq_projector, device=device)
    print(f"  DPLM raw shape       : {dplm_raw.shape}")   # [N, 1280]
    print(f"  DPLM projected shape : {dplm_emb.shape}")   # [N, 256]

    # --- Step 3c: ESM-C base embeddings (optional) -------------------------------
    esmc_emb = None
    if args.esmc:
        print("\n=== Step 3c: Extracting ESM-C base embeddings ===")
        from esmc_local import load_esmc, encode_sequences_esmc
        esmc_model, esmc_tokenizer = load_esmc(args.esmc_model, device)
        esmc_emb = encode_sequences_esmc(esmc_model, esmc_tokenizer, sequences,
                                         device=device, batch_size=8)
        print(f"  ESM-C emb shape : {esmc_emb.shape}")   # [N, 960 or 1152]

    # Optionally save embeddings
    if args.save_npy:
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(os.path.join(args.output_dir, "pids.npy"), np.array(pids))
        np.save(os.path.join(args.output_dir, "md_raw.npy"), md_raw)
        np.save(os.path.join(args.output_dir, "md_proj.npy"), md_proj)
        np.save(os.path.join(args.output_dir, "esm2_emb.npy"), esm2_emb)
        np.save(os.path.join(args.output_dir, "dplm_raw.npy"), dplm_raw)
        np.save(os.path.join(args.output_dir, "dplm_proj.npy"), dplm_emb)
        if esmc_emb is not None:
            np.save(os.path.join(args.output_dir, "esmc_emb.npy"), esmc_emb)
        print(f"  Embeddings saved to {args.output_dir}")

    # --- Step 4: t-SNE visualisation --------------------------------------------
    print("\n=== Step 4: t-SNE visualisation ===")
    tsne_plot(pids, md_raw, md_proj, esm2_emb, dplm_emb, args.output_dir,
              esmc_emb=esmc_emb)

    # --- Step 5: property-coloured t-SNE (MD raw + DPLM raw, by rmsf / gyr) ----
    # if args.csv_path:
    #     print("\n=== Step 5: Property-coloured t-SNE (avg_rmsf, avg_gyr) ===")
    #     rmsf_vals, gyr_vals = load_rmsf_gyr(args.csv_path, pids)
    #     for emb, label in [(md_raw, "MD_raw"), (dplm_raw, "DPLM_raw_1280")]:
    #         tsne_property_plot(emb, rmsf_vals, label, "avg_rmsf",
    #                            pids, args.output_dir)
    #         tsne_property_plot(emb, gyr_vals,  label, "avg_gyr",
    #                            pids, args.output_dir)

    # # --- Step 5b: property-coloured t-SNE (MD raw + DPLM raw, by RMSD/gyr std) -
    if args.analysis_path:
        print("\n=== Step 5b: Property-coloured UMAP (RMSD std, gyration std) ===")
        rmsd_std_vals, gyr_std_vals = load_rmsd_gyr_std(
            args.analysis_path, pids, rmsd_col=args.rmsd_col, gyr_col=args.gyr_col)
        for emb, label in [(md_raw, "MD_raw"), (dplm_raw, "DPLM_raw_1280")]:
            tsne_property_plot(emb, rmsd_std_vals, label, "rmsd_std",
                               pids, args.output_dir)
            tsne_property_plot(emb, gyr_std_vals,  label, "gyr_std",
                               pids, args.output_dir)

    # --- Step 6: protein-level RSA (pooled embeddings vs DCCM descriptor) ------
    # if args.analysis_path:
    #     print("\n=== Step 6: Protein-level RSA (pooled embeddings vs DCCM descriptor) ===")
    #     dccm_desc = load_dccm_descriptors(args.analysis_path, pids,
    #                                       dccm_dir=args.dccm_dir,
    #                                       replicate=args.dccm_replicate)
    #     rsa_results = {}
    #     for emb, label in [(md_raw, "MD_raw"), (dplm_raw, "DPLM_raw_1280")]:
    #         print(f"  {label}:")
    #         rsa_results[label] = compute_emb_dccm_rsa(emb, dccm_desc)
    #     plot_dccm_rsa_bar(rsa_results, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
### LCC
PYTHONPATH=. python ./evaluate/methodology/MD_emb_eval.py \
    --data_dir /path/to/DPLM_data/processed_data_rep0 \
    --checkpoint ./results/vivit_3/checkpoints/checkpoint_best_val_rmsf_cor.pth \
    --config_path ./results/vivit_3/config_vivit3.yaml \
    --seq_checkpoint ./results/vivit_3/checkpoints/checkpoint_best_val_rmsf_cor.pth \
    --output_dir ./evaluate \
    --md_inner_dim 2048 \
    --csv_path /path/to/DPLM_data/atlas_proteins.csv \
    --max_proteins 30 \

### Delta
PYTHONPATH=. python ./evaluate/methodology/MD_emb_eval.py \
    --data_dir /path/to/DPLM_data/processed_data_rep0 \
    --checkpoint ./results/vivit3_resume_long2/checkpoints/checkpoint_best_val_whole_loss.pth \
    --config_path ./results/vivit3_resume_long2/config_vivit3_delta.yaml \
    --output_dir ./evaluate/methodology \
    --csv_path /path/to/DPLM_data/atlas_proteins.csv \
    --max_proteins 30 \

PYTHONPATH=. python ./evaluate/methodology/MD_emb_eval.py \
    --data_dir /path/to/DPLM_data/processed_data_rep0 \
    --checkpoint ./results/vivit3/checkpoints/checkpoint_best_val_rmsf_cor.pth \
    --config_path ./results/vivit3/config_vivit3.yaml \
    --seq_checkpoint ./results/vivit3/checkpoints/checkpoint_best_val_rmsf_cor.pth \
    --output_dir ./evaluate/methodology/MD_output/vivit3/  \
    --csv_path /path/to/DPLM_data/atlas_proteins.csv \
    --pro_len 190 

PYTHONPATH=. python ./evaluate/methodology/MD_emb_eval_umap.py \
    --data_dir /path/to/DPLM_data/processed_data_rep0 \
    --checkpoint ./results/vivit_info/checkpoints/checkpoint_best_val_whole_loss.pth \
    --config_path ./results/vivit_info/config_vivit_comp.yaml \
    --output_dir ./evaluate/methodology/MD_output/vivit_info_umap_100_5_rep0/  \
    --csv_path /path/to/DPLM_data/atlas_proteins.csv \
    --pro_len 100 

PYTHONPATH=. python ./evaluate/methodology/MD_emb_eval_umap.py \
    --data_dir /path/to/DPLM_data/processed_data_rep0 \
    --checkpoint ./results/vivit3_ori/checkpoints/checkpoint_best_val_whole_loss.pth \
    --config_path ./results/vivit3_ori/config_vivit3_delta.yaml \
    --output_dir ./evaluate/methodology/MD_output/vivit3_ori_umap_100_5_rep0/  \
    --csv_path /path/to/DPLM_data/atlas_proteins.csv \
    --pro_len 100 

PYTHONPATH=. python ./evaluate/methodology/MD_emb_eval_umap.py \
    --data_dir /path/to/DPLM_data/processed_data_rep0 \
    --checkpoint ./results/vivit_info2/checkpoints/checkpoint_best_val_whole_loss.pth \
    --config_path ./results/vivit_info2/config_vivit_comp_v2.yaml \
    --output_dir ./evaluate/methodology/MD_output/vivit_info2_umap_100_5_rep0_new/  \
    --csv_path /path/to/DPLM_data/atlas_proteins.csv \
    --pro_len 100 \
    --analysis_path /path/to/DPLM_data/analysis/ \
    --rmsd_col RMSD_R1 \
    --gyr_col gyr_R1


PYTHONPATH=. python ./evaluate/methodology/MD_emb_eval_umap.py \
    --data_dir /path/to/DPLM_data/processed_data_rep0 \
    --checkpoint ./results/vivit_info2/checkpoints/checkpoint_best_val_whole_loss.pth \
    --config_path ./results/vivit_info2/config_vivit_comp_v2.yaml \
    --output_dir ./evaluate/methodology/MD_output/vivit_info2_umap_180_5_rep0_new/  \
    --csv_path /path/to/DPLM_data/atlas_proteins.csv \
    --pro_len 180 \
    --analysis_path /path/to/DPLM_data/analysis/ \
    --rmsd_col RMSD_R1 \
    --gyr_col gyr_R1

PYTHONPATH=. python ./evaluate/methodology/MD_emb_eval_umap.py \
    --data_dir /path/to/DPLM_data/processed_test_rep0 \
    --checkpoint ./results/vivit3_ori/checkpoints/checkpoint_best_val_whole_loss.pth \
    --config_path ./results/vivit3_ori/config_vivit3_delta.yaml \
    --output_dir ./evaluate/methodology/MD_output/vivit3_ori_umap_100_5_rep0/  \
    --csv_path /path/to/DPLM_data/atlas_proteins.csv \
    --dccm_dir      /path/to/DPLM_data/DCCM_dir \
    --dccm_replicate R1 \
    --analysis_path /path/to/DPLM_data/analysis/ \
    
PYTHONPATH=. python ./evaluate/methodology/MD_emb_eval_umap.py \
    --data_dir /path/to/DPLM_data/processed_test_rep0 \
    --checkpoint ./results/vivit3_ori/checkpoints/checkpoint_best_val_whole_loss.pth \
    --config_path ./results/vivit3_ori/config_vivit3_delta.yaml \
    --output_dir ./evaluate/methodology/MD_output/vivit3_ori_umap_100_5_rep0/  \
    --csv_path /path/to/DPLM_data/atlas_proteins.csv \
    --analysis_path /path/to/DPLM_data/analysis/ \
    --rmsd_col RMSD_R1 \
    --gyr_col gyr_R1

"""