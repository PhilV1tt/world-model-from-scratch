"""Multi-head self-attention from scratch.

Suit la structure GPT-2 / ViT canonique.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None, dropout=0.0):
    """Attention scaled dot-product (Vaswani et al. 2017).

    q, k, v: (B, H, T, d_head)
    mask: (T, T) ou (B, 1, T, T), True = position autorisee, False = masquee.
    """
    d_head = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    if dropout > 0.0:
        attn = F.dropout(attn, p=dropout)
    return attn @ v, attn


class MultiHeadAttention(nn.Module):
    """MHA avec projections Q/K/V separees et masque causal optionnel."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, causal: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} doit etre divisible par n_heads={n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.causal:
            causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            mask = causal_mask if attn_mask is None else (causal_mask & attn_mask)
        else:
            mask = attn_mask

        y, _ = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out(y)
