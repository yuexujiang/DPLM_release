"""
model_dccm.py — DCCM-matrix predictor from per-residue embeddings.

Given a protein's per-residue embeddings [L, D], predict its Dynamic Cross-Correlation
Matrix (DCCM) [L, L] with values in [-1, 1]. The predictor projects each residue to a
low-dim vector and scores every residue pair; all heads are symmetric so the predicted
matrix is symmetric (as a real DCCM is).

Heads
-----
  cosine   : DCCM_pred[i,j] = cosine(proj_i, proj_j).  Symmetric, diag==1, in [-1,1].
             O(L^2) memory — safe for long proteins.
  bilinear : DCCM_pred[i,j] = tanh(proj_i^T W_sym proj_j).  Symmetric, in [-1,1].
             O(L^2) memory, more expressive than cosine. (default)
  mlp      : symmetric pair MLP on [proj_i * proj_j, |proj_i - proj_j|] → tanh.
             O(L^2 * P) memory — only use on shorter proteins.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr


class DCCMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, proj_dim=128, dropout=0.2,
                 head='bilinear'):
        super().__init__()
        assert head in ('cosine', 'bilinear', 'mlp')
        self.head = head
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )
        if head == 'bilinear':
            # Initialise near identity so the initial prediction ~ cosine-like.
            self.W = nn.Parameter(torch.eye(proj_dim) + 0.01 * torch.randn(proj_dim, proj_dim))
        elif head == 'mlp':
            self.pair_mlp = nn.Sequential(
                nn.Linear(2 * proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, 1),
            )

    def forward(self, emb):
        """emb: [L, D] → predicted DCCM [L, L] in [-1, 1]."""
        z = self.proj(emb)                       # [L, P]
        if self.head == 'cosine':
            zn = F.normalize(z, dim=1)
            return zn @ zn.T                     # symmetric, diag==1, [-1,1]
        if self.head == 'bilinear':
            w_sym = 0.5 * (self.W + self.W.T)
            s = (z @ w_sym) @ z.T                # [L, L] symmetric
            return torch.tanh(s)
        # mlp
        L, P = z.shape
        zi = z.unsqueeze(1).expand(L, L, P)
        zj = z.unsqueeze(0).expand(L, L, P)
        feat = torch.cat([zi * zj, (zi - zj).abs()], dim=-1)   # symmetric features
        return torch.tanh(self.pair_mlp(feat).squeeze(-1))     # [L, L]


def _upper_tri_mask(L, device):
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)


def dccm_mse_loss(pred, gt):
    """MSE over the strict upper triangle (off-diagonal residue pairs)."""
    m = _upper_tri_mask(pred.shape[0], pred.device)
    return F.mse_loss(pred[m], gt[m])


def protein_dccm_corr(pred_np, gt_np):
    """Pearson & Spearman between predicted and ground-truth DCCM upper triangles.

    pred_np, gt_np: np.ndarray [L, L]. Returns (pearson, spearman) or (nan, nan).
    """
    L = pred_np.shape[0]
    if L < 4 or gt_np.shape != (L, L):
        return np.nan, np.nan
    iu = np.triu_indices(L, k=1)
    p, g = pred_np[iu], gt_np[iu]
    if np.std(p) == 0 or np.std(g) == 0:
        return np.nan, np.nan
    pear = pearsonr(p, g)[0]
    spear = spearmanr(p, g)[0]
    return pear, spear
