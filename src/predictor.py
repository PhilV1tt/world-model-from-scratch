"""Latent dynamics predictor.

Empilement de AdaLNBlock conditionne par l'action,
masque causal temporel sur l'historique de N frames.

Configuration: 6 couches, 16 tetes, dropout 0.1.
Action injectee via AdaLN a chaque couche.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .adaln import AdaLNBlock


class Predictor(nn.Module):
    def __init__(
        self,
        z_dim: int = 192,
        action_dim: int = 2,
        d_model: int = 384,
        depth: int = 6,
        n_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_history: int = 8,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(z_dim, d_model)
        self.action_proj = nn.Linear(action_dim, action_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, max_history, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            AdaLNBlock(d_model, n_heads, action_dim=action_dim, mlp_ratio=mlp_ratio, dropout=dropout, causal=True)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, z_dim),
            nn.BatchNorm1d(z_dim),
        )

    def forward(self, z_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        """
        z_seq: (B, T, z_dim) latents historiques.
        action_seq: (B, T, action_dim) actions correspondantes.
        Retourne: (B, T, z_dim) predictions z_hat_{t+1} pour chaque t.
        """
        B, T, _ = z_seq.shape
        x = self.input_proj(z_seq)
        x = x + self.pos_embed[:, :T]

        a = self.action_proj(action_seq)
        for block in self.blocks:
            x = block(x, action=a)

        x = self.norm(x)
        x_flat = x.reshape(B * T, self.d_model)
        out = self.output_proj(x_flat)
        return out.reshape(B, T, self.z_dim)
