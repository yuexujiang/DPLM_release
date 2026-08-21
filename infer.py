"""
DPLM Inference Script
=====================
Extracts pre-projection ESM2 embeddings (1280-dim) for user-supplied protein sequences
using the trained DPLM checkpoint.

Usage examples
--------------
# Single sequence on the command line:
python infer.py --checkpoint_path results/test/checkpoints/checkpoint_0003000.pth \
                --input MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ \
                --output embeddings.npy

# Multiple sequences as separate arguments:
python infer.py --checkpoint_path results/test/checkpoints/checkpoint_0003000.pth \
                --input SEQ1 SEQ2 SEQ3 \
                --output embeddings.npy

# FASTA file:
python infer.py --checkpoint_path results/test/checkpoints/checkpoint_0003000.pth \
                --input proteins.fasta \
                --output embeddings.npy

# Plain text file (one sequence per line):
python infer.py --checkpoint_path results/test/checkpoints/checkpoint_0003000.pth \
                --input sequences.txt \
                --output embeddings.npy

Output
------
--mode protein  (default) : saves embeddings.npy of shape [N, 1280]
--mode residue            : saves a list of per-residue arrays to embeddings.npz
                            (keys: "labels", "arr_0", "arr_1", ...)
"""

import argparse
import os
import yaml
import numpy as np
import torch

import esm
import esm_adapterH
from peft import LoraConfig, get_peft_model

from utils.utils import load_configs, load_dplm_checkpoint


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_dplm_model(configs, device):
    """Instantiate the DPLM sequence encoder: ESM2 plus the Houlsby adapters that the
    contrastive training fits (and LoRA, if the config enables it).

    This builds the ARCHITECTURE only — call load_dplm_checkpoint() afterwards to load the
    trained DPLM weights into it. The architecture is derived from the config so it matches
    exactly what was used during training.

    Returns
    -------
    model   : nn.Module  — DPLM encoder, ready for weight loading
    alphabet : esm.Alphabet
    """
    enc = configs.model.esm_encoder
    model_name = getattr(enc, 'model_name', 'esm2_t33_650M_UR50D')

    if enc.adapter_h.enable:
        model, alphabet = getattr(esm_adapterH.pretrained, model_name)(enc.adapter_h)
    elif enc.lora.enable:
        model, alphabet = getattr(esm.pretrained, model_name)()
        lora_targets = [
            "self_attn.q_proj", "self_attn.k_proj",
            "self_attn.v_proj", "self_attn.out_proj",
        ]
        target_modules = []
        if enc.lora.esm_num_end_lora > 0:
            start_idx = max(model.num_layers - enc.lora.esm_num_end_lora, 0)
            for idx in range(start_idx, model.num_layers):
                for layer_name in lora_targets:
                    target_modules.append(f"layers.{idx}.{layer_name}")
        peft_config = LoraConfig(
            inference_mode=True,
            r=enc.lora.r,
            lora_alpha=enc.lora.alpha,
            target_modules=target_modules,
            lora_dropout=enc.lora.dropout,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
    else:
        model, alphabet = getattr(esm.pretrained, model_name)()

    model.eval()
    model.to(device)
    return model, alphabet


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def read_input_sequences(input_list):
    """Parse sequences from a FASTA file, a plain-text file, or raw strings.

    Parameters
    ----------
    input_list : List[str]
        Either a single-element list with a file path (.fasta/.fa/.txt),
        or a list of raw amino-acid sequence strings.

    Returns
    -------
    List[Tuple[str, str]]  — (label, sequence) pairs
    """
    if len(input_list) == 1:
        path = input_list[0]
        if path.endswith(('.fasta', '.fa')):
            return _parse_fasta(path)
        if path.endswith('.txt'):
            return _parse_txt(path)

    # treat every element as a raw sequence string
    return [(f"seq_{i}", seq.strip()) for i, seq in enumerate(input_list)]


def _parse_fasta(path):
    sequences = []
    label, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('>'):
                if label is not None:
                    sequences.append((label, ''.join(buf)))
                label = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if label is not None:
        sequences.append((label, ''.join(buf)))
    return sequences


def _parse_txt(path):
    sequences = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            seq = line.strip()
            if seq:
                sequences.append((f"seq_{i}", seq))
    return sequences


# ---------------------------------------------------------------------------
# Embedding extraction  (follows the main() pattern in aa_emb_MDfeature.py)
# ---------------------------------------------------------------------------

def embed_sequences(model, alphabet, seq_list, batch_size, max_length, device, mode):
    """Extract pre-projection ESM2 representations for a list of sequences.

    Follows the same pattern as main() in evaluate/aa_emb_MDfeature.py:
      tokens → model(..., repr_layers=[num_layers]) → representations[num_layers]

    Parameters
    ----------
    seq_list   : List[Tuple[str, str]]  — (label, sequence)
    batch_size : int
    max_length : int  — sequences are truncated to this length
    mode       : "protein" | "residue"
                 protein  → mean-pool over residues → [N, 1280]
                 residue  → per-residue arrays,      list of [L_i, 1280]

    Returns
    -------
    labels     : List[str]
    embeddings : np.ndarray [N, 1280]  if mode=="protein"
                 List[np.ndarray]      if mode=="residue"
    """
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=max_length)
    num_layers = model.num_layers

    all_labels = []
    all_embeddings = []  # protein mode: list of [1280] arrays; residue: list of [L, 1280]

    for start in range(0, len(seq_list), batch_size):
        batch = seq_list[start: start + batch_size]
        batch_labels, _, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[num_layers], return_contacts=False)

        # shape: [B, seq_len+2, 1280]  (+2 = CLS and EOS tokens)
        representations = results["representations"][num_layers]

        for i, (label, _) in enumerate(batch):
            # strip CLS (pos 0) and EOS (last non-padding position)
            # use the actual token length to find EOS
            token_len = (batch_tokens[i] != alphabet.padding_idx).sum().item()
            # positions 1 .. token_len-2  are the real residue tokens
            residue_emb = representations[i, 1: token_len - 1, :]  # [L, 1280]

            if mode == "protein":
                protein_emb = residue_emb.mean(dim=0)  # [1280]
                all_embeddings.append(protein_emb.cpu().numpy())
            else:
                all_embeddings.append(residue_emb.cpu().numpy())

            all_labels.append(label)

        print(f"  embedded {min(start + batch_size, len(seq_list))}/{len(seq_list)} sequences")

    if mode == "protein":
        return all_labels, np.stack(all_embeddings, axis=0)  # [N, 1280]
    else:
        return all_labels, all_embeddings  # list of [L_i, 1280]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract DPLM pre-projection ESM2 embeddings (1280-dim) for protein sequences."
    )
    parser.add_argument(
        "--config_path", "-c",
        default="./configs/config_vivit3.yaml",
        help="Path to config yaml (default: ./configs/config_vivit3.yaml)",
    )
    parser.add_argument(
        "--checkpoint_path", required=True,
        help="Path to trained .pth checkpoint file",
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="Protein sequences: a FASTA file, a .txt file (one seq per line), "
             "or one or more raw sequence strings",
    )
    parser.add_argument(
        "--output", default="./embeddings.npy",
        help="Output file path (default: ./embeddings.npy). "
             "Use .npy for protein mode or .npz for residue mode.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Number of sequences per forward pass (default: 8)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: 'cuda' or 'cpu'. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--mode", choices=["protein", "residue"], default="protein",
        help="'protein': mean-pooled [N,1280]; 'residue': per-residue list (default: protein)",
    )
    args = parser.parse_args()

    # ---- device ----
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # ---- config ----
    with open(args.config_path) as f:
        config_dict = yaml.full_load(f)
    configs = load_configs(config_dict, args=None)

    # ---- model ----
    print("Building ESM2 model...")
    model, alphabet = build_dplm_model(configs, device)

    print("Loading checkpoint...")
    load_dplm_checkpoint(model, args.checkpoint_path)
    model.eval()

    # ---- sequences ----
    print("Reading input sequences...")
    seq_list = read_input_sequences(args.input)
    print(f"  {len(seq_list)} sequence(s) found")

    # ---- embed ----
    max_length = configs.model.esm_encoder.max_length
    print(f"Embedding sequences (batch_size={args.batch_size}, max_length={max_length}, mode={args.mode})...")
    labels, embeddings = embed_sequences(
        model, alphabet, seq_list,
        batch_size=args.batch_size,
        max_length=max_length,
        device=device,
        mode=args.mode,
    )

    # ---- save ----
    if args.mode == "protein":
        np.save(args.output, embeddings)
        print(f"\nSaved embeddings to {args.output}  shape={embeddings.shape}")
        for label, emb in zip(labels, embeddings):
            print(f"  {label}: {emb.shape}")
    else:
        # residue mode: variable length per protein — save as .npz
        out_path = args.output if args.output.endswith(".npz") else args.output.replace(".npy", ".npz")
        save_dict = {"labels": np.array(labels)}
        for i, emb in enumerate(embeddings):
            save_dict[f"arr_{i}"] = emb
        np.savez(out_path, **save_dict)
        print(f"\nSaved per-residue embeddings to {out_path}")
        for label, emb in zip(labels, embeddings):
            print(f"  {label}: {emb.shape}")


if __name__ == "__main__":
    main()
