"""Tests pour ViT-Tiny encoder."""
import torch

from src.vit import ViT, PatchEmbed


def test_patch_embed_shape():
    pe = PatchEmbed(img_size=64, patch_size=8, in_chans=3, embed_dim=192)
    x = torch.randn(2, 3, 64, 64)
    y = pe(x)
    assert y.shape == (2, 64, 192)


def test_vit_forward_shape():
    vit = ViT(img_size=64, patch_size=8, in_chans=3, embed_dim=192, depth=4, n_heads=3)
    vit.train()
    x = torch.randn(4, 3, 64, 64)
    z = vit(x)
    assert z.shape == (4, 192)


def test_vit_grayscale():
    vit = ViT(img_size=64, patch_size=8, in_chans=1, embed_dim=192, depth=2, n_heads=3)
    vit.train()
    x = torch.randn(4, 1, 64, 64)
    z = vit(x)
    assert z.shape == (4, 192)


def test_vit_param_count_roughly_5M_full_config():
    """Config paper: depth=12, d=192, heads=3, patch=8, img=64."""
    vit = ViT(img_size=64, patch_size=8, in_chans=3, embed_dim=192, depth=12, n_heads=3)
    n = sum(p.numel() for p in vit.parameters())
    assert 4_000_000 < n < 7_000_000, f"ViT-Tiny expected ~5M params, got {n:,}"
