"""Adaptive Layer Normalization conditionnee par l'action.

Suit DiT (Peebles et Xie, 2023).

Init AdaLN a zero pour stabiliser l'entrainement: au debut, AdaLN ne fait rien,
et le signal d'action est integre progressivement.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .transformer import FeedForward


class AdaLNBlock(nn.Module):
    """TransformerBlock avec AdaLN au lieu de LayerNorm.

    L'action est projetee en (gamma1, beta1, alpha1, gamma2, beta2, alpha2) qui modulent
    les deux LayerNorms et les deux residuals.
    """

    def __init__(self, d_model: int, n_heads: int, action_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout, causal=causal)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = FeedForward(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(action_dim, 6 * d_model, bias=True),
        )
        # Init derniere couche a zero -> AdaLN identite au debut.
        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)

    def forward(self, x: torch.Tensor, action: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # action: (B, T, action_dim) ou (B, action_dim) broadcast.
        if action.dim() == 2:
            action = action.unsqueeze(1).expand(-1, x.size(1), -1)
        params = self.adaln_modulation(action)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params.chunk(6, dim=-1)

        h = self.norm1(x) * (1 + gamma1) + beta1
        x = x + alpha1 * self.attn(h, attn_mask=attn_mask)

        h = self.norm2(x) * (1 + gamma2) + beta2
        x = x + alpha2 * self.ffn(h)
        return x
