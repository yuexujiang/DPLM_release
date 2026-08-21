"""
embed_dplm.py — Extract DPLM per-residue embeddings for all CryptoBench apo structures.

For each apo protein in the CryptoBench dataset (train + test folds):
  1. Extract the chain's observed-residue sequence from the local CIF file
     (falls back to RCSB REST API if CIF not found)
  2. Run DPLM ESM2-650M + Houlsby adapters to get per-residue embeddings [L, 1280]
  3. Save as {apo_id}{chain}.npy, plus {apo_id}{chain}_seqids.json listing the
     "{chain}_{auth_seq_id}" composite key for each embedding row, in the output directory

apo_pocket_selection entries (e.g. "B_16") use auth_seq_id, which does NOT
equal the 0-based embedding index whenever residues are missing from the
structure (common — only ~32% of CryptoBench chains start at auth_seq_id 1).
The *_seqids.json sidecar file lets downstream code (data_cryptobench.py)
look up the correct embedding index for each annotated residue.

`apo_chain` can be a single letter (e.g. "B") or a hyphenated multichain id
(e.g. "A-B") for apo structures CryptoBench treats as multimeric — see the
official tutorial (tutorial/tutorial.ipynb, cell 26: get_multichain_apo_binding_sites).
For multichain entries, the sequence is the concatenation of each sub-chain's
observed-residue sequence in order, and seq_ids carry a "{chain}_{auth_seq_id}"
composite key throughout (not just for multichain) so residue numbers from
different chains never collide once concatenated.

Usage:
  python evaluate/cryptobench/embed_dplm.py \
    --dataset_dir   evaluate/cryptobench/cryptobench/cryptobench-dataset \
    --cif_dir       evaluate/cryptobench/cryptobench/cryptobench-dataset/auxiliary-data/cif-files \
    --output_dir    /path/to/DPLM_data/cryptobench_embeddings \
    --checkpoint    results/vivit5/checkpoints/checkpoint_best_val_whole_loss.pth \
    --config_path   configs/config_vivit5_delta.yaml \
    --batch_size    4
"""

import os
import sys
import json
import argparse
import traceback
import numpy as np
from pathlib import Path

import torch
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.Data.IUPACData import protein_letters_3to1

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from infer import build_esm2_model, embed_sequences
from utils.utils import load_configs, load_esm2_checkpoint


# ── CIF sequence extraction ───────────────────────────────────────────────────
#
# Modeled directly on the official CryptoBench tutorial's get_sequence_with_indices
# (tutorial/tutorial.ipynb, cell 33), which uses Bio.PDB.MMCIFParser + chain.get_residues()
# rather than hand-parsing _atom_site lines as text — more robust against CIF format
# quirks (quoted/multi-line fields, alternate conformations) than a manual text parser.

def _extract_single_chain(cif_path: str, chain_id: str):
    """Parse one chain's observed standard residues from a CIF file via Biopython.

    Returns
    -------
    seq     : str  — one-letter amino acid sequence in residue order
              (unknown residues encoded as 'X')
    seq_ids : list[str] — "{chain_id}_{auth_seq_id}" composite key for each position
              in `seq`, same order/length as `seq`. The composite key (rather than a
              bare auth_seq_id) is what apo_pocket_selection entries already use
              (e.g. "B_16"), and avoids residue-number collisions when multiple
              chains are concatenated for multichain apo entries.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(Path(cif_path).stem, cif_path)

    try:
        chain = structure[0][chain_id]
    except KeyError:
        return '', []

    seq, seq_ids = [], []
    for residue in chain.get_residues():
        if residue.get_id()[0][0] != ' ':
            # Skip HETATM/water entries — same filter as the official tutorial
            # (residue.get_id()[0][0] == ' ' marks a standard amino acid residue).
            continue
        resname = residue.get_resname().title()
        aa = protein_letters_3to1.get(resname, 'X')
        seq.append(aa)
        seq_ids.append(f'{chain_id}_{residue.get_id()[1]}')

    return ''.join(seq), seq_ids


def extract_sequence_from_cif(cif_path: str, chain_id: str):
    """Parse a CIF file's observed-residue sequence for one or more chains.

    `chain_id` may be a single letter ("B") or a hyphenated multichain id
    ("A-B"); for multichain ids, each sub-chain's sequence/seq_ids are
    extracted independently and concatenated in order.

    Returns
    -------
    seq     : str
    seq_ids : list[str] — "{chain}_{auth_seq_id}" composite keys, same length as seq.
    """
    sub_chains = chain_id.split('-')
    seq_parts, seq_ids_parts = [], []
    for sub_chain in sub_chains:
        sub_seq, sub_seq_ids = _extract_single_chain(cif_path, sub_chain)
        if not sub_seq:
            return '', []   # any missing sub-chain invalidates the whole multichain entry
        seq_parts.append(sub_seq)
        seq_ids_parts.extend(sub_seq_ids)

    return ''.join(seq_parts), seq_ids_parts


# ── Fallback: RCSB REST API ───────────────────────────────────────────────────

def fetch_sequence_from_rcsb(pdb_id: str, chain_id: str) -> str:
    """Fallback sequence fetch via RCSB FASTA endpoint (requires internet)."""
    import requests, time
    fasta_url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}"
    for attempt in range(3):
        try:
            r = requests.get(fasta_url, timeout=15)
            if r.status_code == 200:
                seq = _parse_fasta_chain(r.text, pdb_id.upper(), chain_id)
                if seq:
                    return seq
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return ''


def _parse_fasta_chain(fasta_text: str, pdb_id: str, chain_id: str) -> str:
    current_chain = None
    current_seq   = []
    for line in fasta_text.splitlines():
        if line.startswith('>'):
            header = line[1:].upper()
            found = (f"{pdb_id}_{chain_id}" in header or
                     f"{pdb_id}:{chain_id}|" in header or
                     header.split('|')[0] == f"{pdb_id}:{chain_id}")
            current_chain = found
            if found:
                current_seq = []
        elif current_chain:
            current_seq.append(line.strip())
    return ''.join(current_seq)


# ── Sequence getter (CIF-first with RCSB fallback) ───────────────────────────

def get_sequence(pdb_id: str, chain_id: str, cif_dir: Path):
    """Returns (seq, seq_ids). seq_ids maps embedding index -> "{chain}_{auth_seq_id}".

    `chain_id` may be a single letter or a hyphenated multichain id (e.g. "A-B").

    The RCSB FASTA fallback has no auth_seq_id info and doesn't support multichain
    lookups, so it falls back to naive 1-based numbering ('{chain_id}_1', '{chain_id}_2',
    ...) — this is only approximately correct (wrong if residues are missing from the
    structure) and is used only when the CIF file is unavailable.
    """
    cif_path = cif_dir / f"{pdb_id.lower()}.cif"
    if cif_path.exists():
        seq, seq_ids = extract_sequence_from_cif(str(cif_path), chain_id)
        if seq:
            return seq, seq_ids
        print(f"  [warn] CIF found but chain {chain_id} not parsed — trying RCSB")
    else:
        print(f"  [warn] CIF not found for {pdb_id} — trying RCSB")

    seq = fetch_sequence_from_rcsb(pdb_id, chain_id)
    seq_ids = [f'{chain_id}_{i + 1}' for i in range(len(seq))]
    if seq:
        print(f"  [warn] Using naive 1-based numbering for {pdb_id}/{chain_id} "
              f"(RCSB fallback has no auth_seq_id) — pocket label mapping may be wrong")
    return seq, seq_ids


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_dir',  required=True,
                   help='Path to cryptobench-dataset/ folder')
    p.add_argument('--cif_dir',
                   default=None,
                   help='Path to unzipped CIF files (default: dataset_dir/auxiliary-data/cif-files)')
    p.add_argument('--output_dir',   required=True,
                   help='Directory to save .npy embedding files')
    p.add_argument('--checkpoint',   required=True,
                   help='DPLM checkpoint .pth path')
    p.add_argument('--config_path',  default='configs/config_vivit5_delta.yaml')
    p.add_argument('--batch_size',   type=int, default=4)
    p.add_argument('--max_length',   type=int, default=1022,
                   help='Max residues per sequence (ESM2 limit)')
    p.add_argument('--device',       default='cuda')
    p.add_argument('--skip_existing', action='store_true',
                   help='Skip apo structures whose .npy already exists')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir     = Path(args.output_dir)
    dataset_dir = Path(args.dataset_dir)
    cif_dir     = Path(args.cif_dir) if args.cif_dir else \
                  dataset_dir / 'auxiliary-data' / 'cif-files'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CIF directory: {cif_dir}  (exists: {cif_dir.exists()})")

    with open(dataset_dir / 'dataset.json') as f:
        dataset = json.load(f)
    with open(dataset_dir / 'folds.json') as f:
        folds = json.load(f)

    # Collect all unique (apo_id, chain) pairs
    all_apo_ids = set()
    for apo_ids in folds.values():
        all_apo_ids.update(apo_ids)

    pairs = []
    for apo_id in sorted(all_apo_ids):
        if apo_id not in dataset:
            print(f"[warn] {apo_id} not in dataset.json — skip")
            continue
        chain    = dataset[apo_id][0]['apo_chain']
        out_path = out_dir / f"{apo_id}{chain}.npy"
        if args.skip_existing and out_path.exists():
            continue
        pairs.append((apo_id, chain, out_path))

    print(f"Total (apo_id, chain) pairs to embed: {len(pairs)}")

    # Build DPLM model
    import yaml
    with open(args.config_path) as f:
        cfg_dict = yaml.full_load(f)
    configs = load_configs(cfg_dict)
    device  = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, alphabet = build_esm2_model(configs, device)

    print(f"Loading DPLM checkpoint: {args.checkpoint}")
    load_esm2_checkpoint(model, args.checkpoint)
    model.eval()

    # Embed in batches
    failed       = []
    seq_buffer   = []
    path_buffer  = []
    seqids_buffer = []   # parallel list: seq_ids for each entry in seq_buffer
    apo_chain_buffer = []   # parallel list: (apo_id, chain) for each entry in seq_buffer

    def flush_buffer():
        if not seq_buffer:
            return
        _, emb_list = embed_sequences(
            model, alphabet, seq_buffer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            mode='residue',
        )
        for (label, _), emb, path, seq_ids in zip(seq_buffer, emb_list, path_buffer, seqids_buffer):
            emb = emb.astype(np.float32)
            np.save(path, emb)
            # Save the auth_seq_id for each embedding row, truncated to match emb length
            # (embed_sequences may truncate to max_length)
            seqids_path = path.with_name(path.stem + '_seqids.json')
            with open(seqids_path, 'w') as f:
                json.dump(seq_ids[:emb.shape[0]], f)
            print(f"  saved {path.name}  shape={emb.shape}")
        seq_buffer.clear()
        path_buffer.clear()
        seqids_buffer.clear()
        apo_chain_buffer.clear()

    def flush_buffer_safe():
        """flush_buffer(), but a crash marks the whole pending buffer failed instead
        of propagating and killing the rest of the run."""
        pending = list(apo_chain_buffer)
        try:
            flush_buffer()
        except Exception as e:
            print(f"  [ERROR] flush_buffer() raised an exception: {e}")
            traceback.print_exc()
            failed.extend(pending)
            seq_buffer.clear()
            path_buffer.clear()
            seqids_buffer.clear()
            apo_chain_buffer.clear()

    for i, (apo_id, chain, out_path) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {apo_id} chain {chain} ...")
        try:
            seq, seq_ids = get_sequence(apo_id, chain, cif_dir)

            if not seq:
                print(f"  [ERROR] Could not obtain sequence for {apo_id}/{chain}")
                failed.append((apo_id, chain))
                continue

            print(f"  length={len(seq)}")
            seq_buffer.append((f"{apo_id}{chain}", seq))
            path_buffer.append(out_path)
            seqids_buffer.append(seq_ids)
            apo_chain_buffer.append((apo_id, chain))

            if len(seq_buffer) >= args.batch_size:
                flush_buffer_safe()
        except Exception as e:
            # A single malformed CIF/protein must not abort the entire run — log it,
            # mark this protein failed, and keep going. Without this, one bad structure
            # would silently zero out every remaining protein in the dataset.
            print(f"  [ERROR] {apo_id}/{chain} raised an exception: {e}")
            traceback.print_exc()
            failed.append((apo_id, chain))

    flush_buffer_safe()

    print(f"\nDone. Embedded {len(pairs) - len(failed)} / {len(pairs)} structures.")
    if failed:
        print(f"Failed ({len(failed)}):")
        for apo_id, chain in failed:
            print(f"  {apo_id} chain {chain}")
        fail_path = out_dir / 'failed.txt'
        with open(fail_path, 'w') as f:
            f.write('\n'.join(f"{a}{c}" for a, c in failed))
        print(f"Failed list saved to {fail_path}")


if __name__ == '__main__':
    main()
