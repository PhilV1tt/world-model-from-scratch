"""Transformer blocks: pre-LN style GPT-2 / ViT.

FeedForward, connexions residuelles, empilement.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN block: x = x + attn(LN(x)); x = x + ffn(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, causal: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout, causal=causal)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerStack(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, causal: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, dropout, causal) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return x
