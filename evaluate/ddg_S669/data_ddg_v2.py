"""
data_ddg_v2.py — DataLoader for the mutation-site-aware ddG model (train_ddg_v2.py).

Differences from data_ddg.py:
  * Each sample is a *directed* mutation (from_seq → to_seq) with the mutated residue
    positions precomputed (0-based indices where the two sequences differ).
  * Optional inverse-mutation augmentation (train split ONLY): for every
    (WT → MT, +ddG) row we also emit (MT → WT, −ddG) with the same positions. This
    doubles the training data and teaches the physical antisymmetry
    ΔΔG(WT→MT) = −ΔΔG(MT→WT). Validation / test are never augmented.

The protein-wise train/val split and protein-id parsing are reused from data_ddg.py so
the split matches the existing pipeline exactly (test_size=0.2, random_state=42).

Both CSVs must contain columns: name, wt_seq, mut_seq, ddG.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Reuse the id parser + split convention from the original pipeline.
# (extract_protein_id is inlined below)


# ── inlined from data_ddg.py (the pre-v2 module, removed
#    in the public release; only these helpers were ever used from it) ─────
def extract_protein_id(name):
    """Extract protein ID from name: e.g. rcsb_1A7V_A_A104H_6_25 → '1A7V_A'."""
    parts = name.split("_")
    return f"{parts[1]}_{parts[2]}"


# ────────────────────────────────────────────────────────────────────────────
# Mutation-position helper
# ────────────────────────────────────────────────────────────────────────────

def diff_positions(wt_seq: str, mut_seq: str):
    """Return the 0-based residue indices where wt_seq and mut_seq differ.

    ddG mutations in S8754/S669 are substitutions only (equal-length sequences),
    so the differing indices are exactly the mutated sites. If the lengths differ
    (unexpected — e.g. an indel), return an empty list so the model falls back to
    global-only features rather than mis-indexing.
    """
    if len(wt_seq) != len(mut_seq):
        return []
    return [i for i in range(len(wt_seq)) if wt_seq[i] != mut_seq[i]]


# ────────────────────────────────────────────────────────────────────────────
# Dataset + collate
# ────────────────────────────────────────────────────────────────────────────

class StabilityDatasetV2(Dataset):
    """Directed-mutation dataset with precomputed mutation positions.

    Each item: {protein_id, from_seq, to_seq, mut_pos (list[int]), ddg (float tensor)}.

    augment_inverse: if True, every row contributes two items — the direct mutation
    (WT→MT, +ddG) and its inverse (MT→WT, −ddG). Use ONLY on the train split.
    """

    def __init__(self, df, augment_inverse: bool = False):
        df = df.reset_index(drop=True)
        self.items = []
        for _, row in df.iterrows():
            wt, mut = str(row["wt_seq"]), str(row["mut_seq"])
            pos = diff_positions(wt, mut)
            ddg = float(row["ddG"])
            pid = row["protein_id"]
            # Direct: WT → MT, +ddG
            self.items.append((pid, wt, mut, pos, ddg))
            if augment_inverse:
                # Inverse: MT → WT, −ddG (same mutated positions)
                self.items.append((pid, mut, wt, pos, -ddg))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pid, from_seq, to_seq, pos, ddg = self.items[idx]
        return {
            "protein_id": pid,
            "from_seq":   from_seq,
            "to_seq":     to_seq,
            "mut_pos":    pos,
            "ddg":        torch.tensor(ddg, dtype=torch.float32),
        }


def collate_fn_v2(batch):
    """Collate into (from_seqs, to_seqs, mut_pos_list, ddg_tensor, protein_ids)."""
    return (
        [b["from_seq"]   for b in batch],
        [b["to_seq"]     for b in batch],
        [b["mut_pos"]    for b in batch],
        torch.stack([b["ddg"] for b in batch]),
        [b["protein_id"] for b in batch],
    )


# ────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ────────────────────────────────────────────────────────────────────────────

def prepare_dataloaders_v2(configs):
    """Build train/val/test DataLoaders for the site-aware model.

    Paths:
        configs.train_settings.train_csv_path  (S8754, protein-wise 80/20 split)
        configs.test_settings.test_csv_path    (S669)
    Inverse augmentation applied to the TRAIN split only when
    configs.train_settings.augment_inverse is True.
    """
    augment_inverse = bool(getattr(configs.train_settings, 'augment_inverse', False))

    # ── Train / Val (protein-wise 80/20 split, identical to data_ddg.py) ──────
    df = pd.read_csv(configs.train_settings.train_csv_path)
    df["protein_id"] = df["name"].apply(extract_protein_id)

    proteins = df["protein_id"].unique()
    train_prots, val_prots = train_test_split(proteins, test_size=0.2, random_state=42)

    train_df = df[df["protein_id"].isin(train_prots)]
    val_df   = df[df["protein_id"].isin(val_prots)]

    # ── Test (S669) ───────────────────────────────────────────────────────
    df_test = pd.read_csv(configs.test_settings.test_csv_path)
    df_test["protein_id"] = df_test["name"].apply(extract_protein_id)

    train_ds = StabilityDatasetV2(train_df, augment_inverse=augment_inverse)
    val_ds   = StabilityDatasetV2(val_df,   augment_inverse=False)
    test_ds  = StabilityDatasetV2(df_test,  augment_inverse=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=configs.train_settings.batch_size,
        shuffle=True,
        num_workers=configs.train_settings.num_workers,
        collate_fn=collate_fn_v2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=configs.valid_settings.batch_size,
        shuffle=False,
        num_workers=configs.valid_settings.num_workers,
        collate_fn=collate_fn_v2,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=configs.test_settings.batch_size,
        shuffle=False,
        num_workers=configs.test_settings.num_workers,
        collate_fn=collate_fn_v2,
    )

    return {"train": train_loader, "valid": val_loader, "test": test_loader}
