"""
model_dccm_attn.py — attention-augmented DCCM predictor.

Extends DCCMPredictor (model_dccm.py) with a second, natively-pairwise pathway that reads
the backbone's self-attention maps. Per residue pair (i, j):

    s_emb[i,j]  = per-residue pathway  (bilinear / cosine on projected residue embeddings)
    a[i,j]      = learned weighted sum over the C symmetrized attention channels

and the two are fused into the prediction one of two ways (`fusion`):

    add      : DCCM_pred = tanh( s_emb + a )              (attention adds a coupling term)
    weighted : DCCM_pred = tanh( s_emb * (1 + a) )        (attention GATES the emb score;
               gate = 1 + a, default)                      the supervised analog of the
                                                           attention-weighted read-out in
                                                           unsup_dccm_attn.py)

`a` is a 1x1 "convolution" over channels: Linear(C → 1) at every pair — the classic
logistic-regression-on-attention contact predictor, so DPLM's adapter-reshaped attention has
a direct path to the output. It is zero-initialised, so BOTH fusions start out as the pure
per-residue predictor (add: s_emb + 0; weighted: s_emb * 1) and learn the attention
contribution from there. Both pathways are symmetric (attention channels are symmetrized in
data_dccm_attn.py; the bilinear form uses a symmetrized weight), so the prediction is
symmetric, as a real DCCM is.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the loss + correlation helpers so the attention variant stays metric-compatible.
from model_dccm import dccm_mse_loss, protein_dccm_corr, _upper_tri_mask  # noqa: F401


class DCCMAttnPredictor(nn.Module):
    def __init__(self, input_dim, attn_channels, hidden_dim=512, proj_dim=128,
                 dropout=0.2, head='bilinear', fusion='weighted'):
        super().__init__()
        assert head in ('cosine', 'bilinear')
        assert fusion in ('add', 'weighted')
        self.head = head
        self.fusion = fusion
        self.attn_channels = attn_channels

        # ── Per-residue pathway (same projector as DCCMPredictor) ──────────────
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )
        if head == 'bilinear':
            self.W = nn.Parameter(torch.eye(proj_dim) + 0.01 * torch.randn(proj_dim, proj_dim))

        # ── Attention pathway: learned weighted sum over C channels ────────────
        # Linear over the channel dim, applied per pair (a 1x1 conv). Initialised to
        # ~zero so training starts from the pure per-residue predictor and the attention
        # contribution is learned additively (stable, and a clean ablation baseline).
        self.attn_readout = nn.Linear(attn_channels, 1)
        nn.init.zeros_(self.attn_readout.weight)
        nn.init.zeros_(self.attn_readout.bias)

    def _emb_score(self, emb):
        """Per-residue pre-activation score [L, L] (before tanh)."""
        z = self.proj(emb)                               # [L, P]
        if self.head == 'cosine':
            zn = F.normalize(z, dim=1)
            return zn @ zn.T                             # symmetric, [-1, 1]
        w_sym = 0.5 * (self.W + self.W.T)
        return (z @ w_sym) @ z.T                         # [L, L] symmetric

    def _attn_score(self, attn):
        """Attention pre-activation score [L, L] from channels [C, L, L]."""
        # [C, L, L] → [L, L, C] → Linear(C→1) → [L, L]
        a = attn.permute(1, 2, 0).float()                # [L, L, C]
        return self.attn_readout(a).squeeze(-1)          # [L, L]

    def forward(self, emb, attn):
        """emb: [L, D],  attn: [C, L, L] → predicted DCCM [L, L] in [-1, 1]."""
        s_emb = self._emb_score(emb)
        a = self._attn_score(attn)
        if self.fusion == 'weighted':
            s = s_emb * (1.0 + a)                # attention gates the per-residue score
        else:
            s = s_emb + a                        # attention adds a coupling term
        return torch.tanh(s)
