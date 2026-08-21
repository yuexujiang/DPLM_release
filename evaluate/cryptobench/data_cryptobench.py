"""
data_cryptobench.py — Dataset and DataLoader for CryptoBench DPLM classifier.

Each item: per-residue DPLM embedding [L, 1280] + binary label [L] (1=binding).
Labels are built by unioning all apo_pocket_selection entries across all holo
structures for that apo protein (one apo can have multiple holo partners).
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


# ── Dataset ───────────────────────────────────────────────────────────────────

class CryptoBenchDataset(Dataset):
    """Per-residue binary classification on CryptoBench apo structures.

    Each __getitem__ returns all residues of one apo protein as a single item.
    The DataLoader should use batch_size=1 and the collate_fn below for
    per-residue batching across multiple proteins.
    """

    def __init__(self, apo_ids: List[str], dataset: dict, emb_dir: Path,
                 max_len: int = 1022):
        """
        Parameters
        ----------
        apo_ids  : list of apo PDB IDs (e.g. ['1a4u', ...])
        dataset  : parsed dataset.json dict
        emb_dir  : directory containing {apo_id}{chain}.npy files
        max_len  : truncate sequences longer than this
        """
        self.emb_dir = Path(emb_dir)
        self.max_len = max_len
        self.items = []  # (apo_id, chain, emb_path, label_array)

        for apo_id in apo_ids:
            if apo_id not in dataset:
                continue
            entries = dataset[apo_id]
            chain = entries[0]['apo_chain']
            emb_path = self.emb_dir / f"{apo_id}{chain}.npy"
            if not emb_path.exists():
                continue

            emb = np.load(emb_path)   # [L, 1280]
            L = min(emb.shape[0], max_len)

            # apo_pocket_selection entries look like "B_16" = chain B, auth_seq_id 16.
            # auth_seq_id != embedding index whenever residues are missing from the
            # structure, so we look up the actual position via the seqids sidecar
            # file written by embed_dplm.py. The sidecar stores the same
            # "{chain}_{auth_seq_id}" composite key as apo_pocket_selection entries
            # (required for multichain apo_chain values like "A-B", where bare
            # auth_seq_id alone would collide across chains once concatenated).
            seqids_path = emb_path.with_name(emb_path.stem + '_seqids.json')
            if seqids_path.exists():
                with open(seqids_path) as f:
                    seq_ids = json.load(f)
                seq_id_to_idx = {sid: idx for idx, sid in enumerate(seq_ids)}
            else:
                # No sidecar (older, pre-multichain embeddings) — fall back to naive
                # auth_seq_id-1, single-chain only.
                seq_id_to_idx = None

            # Union all pocket selections across all holo entries
            binding_indices = set()
            for entry in entries:
                for res in entry['apo_pocket_selection']:
                    # Format: "{chain}_{auth_seq_id}", e.g. "B_16"
                    if seq_id_to_idx is not None:
                        idx = seq_id_to_idx.get(res)
                    else:
                        try:
                            idx = int(res.split('_', 1)[1]) - 1
                        except ValueError:
                            idx = None
                    if idx is not None and 0 <= idx < L:
                        binding_indices.add(idx)

            labels = np.zeros(L, dtype=np.int64)
            for idx in binding_indices:
                labels[idx] = 1

            self.items.append((apo_id, emb[:L], labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        apo_id, emb, labels = self.items[i]
        return apo_id, torch.tensor(emb, dtype=torch.float32), torch.tensor(labels)


def collate_fn(batch):
    """Flatten residues from multiple proteins into one big tensor for batched loss."""
    apo_ids, embs, labels = zip(*batch)
    # Concatenate all residues: [sum(L_i), 1280] and [sum(L_i)]
    embs_cat   = torch.cat(embs,   dim=0)
    labels_cat = torch.cat(labels, dim=0)
    return list(apo_ids), embs_cat, labels_cat


def flat_collate_fn(batch):
    """Return per-protein tensors (used for evaluation, batch_size=1)."""
    return batch[0]


# ── DataLoader factory ────────────────────────────────────────────────────────

def prepare_dataloaders(configs, dataset: dict, emb_dir: Path):
    """Return train DataLoaders for each fold and a test DataLoader.

    Returns
    -------
    fold_datasets : list of 4 CryptoBenchDataset (one per train fold)
    test_dataset  : CryptoBenchDataset for the test split
    """
    folds_path = Path(configs.dataset_dir) / 'folds.json'
    with open(folds_path) as f:
        folds = json.load(f)

    test_dataset = CryptoBenchDataset(
        folds['test'], dataset, emb_dir,
        max_len=configs.max_len,
    )

    fold_datasets = []
    for fold_idx in range(4):
        apo_ids = folds[f'train-{fold_idx}']
        fold_datasets.append(CryptoBenchDataset(
            apo_ids, dataset, emb_dir, max_len=configs.max_len,
        ))

    return fold_datasets, test_dataset


def make_loader(dataset: CryptoBenchDataset, batch_size: int = 1,
                shuffle: bool = False, num_workers: int = 0,
                flat: bool = False) -> DataLoader:
    """Wrap a CryptoBenchDataset in a DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=flat_collate_fn if flat else collate_fn,
    )
