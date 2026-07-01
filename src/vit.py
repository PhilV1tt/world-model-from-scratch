"""Vision Transformer encoder (ViT-Tiny).

Adapte a parking-v0 (image 64x64 RGB ou niveaux de gris), patch 8, latent 192.
Ici on prend patch=8 sur 64x64 -> 64 patches, plus rapide a entrainer sur M5.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import TransformerStack


class PatchEmbed(nn.Module):
    """Patchification + projection lineaire en une convolution."""

    def __init__(self, img_size: int = 64, patch_size: int = 8, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        assert img_size % patch_size == 0, f"img_size {img_size} doit etre multiple de patch_size {patch_size}"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, embed_dim, H/p, W/p) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        n_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        proj_dim: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim or embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.blocks = TransformerStack(depth, embed_dim, n_heads, mlp_ratio, dropout=dropout, causal=False)
        self.norm = nn.LayerNorm(embed_dim)

        # Projection: MLP a 1 couche avec BatchNorm pour casser la LN finale,
        # sinon SIGReg ne peut pas etre optimise efficacement.
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.proj_head(cls_out)
