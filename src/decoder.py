"""Decoder leger pour visualiser ce qui est encode dans le latent.

Pas dans la perte principale, juste un outil de diagnostic.
Cross-attention entre query tokens (1 par patch cible) et le latent z (1 token).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import FeedForward


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, mlp_ratio=mlp_ratio)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.cross_attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + attn_out
        q = q + self.ffn(self.norm2(q))
        return q


class LatentDecoder(nn.Module):
    def __init__(
        self,
        z_dim: int = 192,
        img_size: int = 64,
        patch_size: int = 8,
        out_chans: int = 3,
        d_model: int = 192,
        depth: int = 2,
        n_heads: int = 3,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.num_patches = (img_size // patch_size) ** 2

        self.z_proj = nn.Linear(z_dim, d_model)
        self.queries = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.blocks = nn.ModuleList([CrossAttentionBlock(d_model, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.to_pixels = nn.Linear(d_model, patch_size * patch_size * out_chans)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        kv = self.z_proj(z).unsqueeze(1)
        q = self.queries.expand(B, -1, -1)
        for block in self.blocks:
            q = block(q, kv)
        q = self.norm(q)
        patches = self.to_pixels(q)

        h = w = self.img_size // self.patch_size
        patches = patches.reshape(B, h, w, self.patch_size, self.patch_size, self.out_chans)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        img = patches.reshape(B, self.out_chans, self.img_size, self.img_size)
        return torch.sigmoid(img)
