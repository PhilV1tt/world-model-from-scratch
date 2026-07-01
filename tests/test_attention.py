"""Tests pour scaled dot-product attention et MultiHeadAttention."""
import math
import pytest
import torch

from src.attention import scaled_dot_product_attention, MultiHeadAttention


def test_attention_output_shape():
    B, H, T, d = 2, 3, 5, 16
    q = torch.randn(B, H, T, d)
    k = torch.randn(B, H, T, d)
    v = torch.randn(B, H, T, d)
    out, attn = scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, H, T, d)
    assert attn.shape == (B, H, T, T)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, H, T), atol=1e-5)


def test_causal_mask_zeros_future():
    """Le masque causal doit donner exactement 0 sur les positions futures (apres softmax)."""
    torch.manual_seed(0)
    B, H, T, d = 1, 1, 4, 4
    q = torch.randn(B, H, T, d)
    k = torch.randn(B, H, T, d)
    v = torch.randn(B, H, T, d)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    _, attn = scaled_dot_product_attention(q, k, v, mask=mask)
    upper = attn[0, 0].triu(diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-7), f"upper triangular not zero: {upper}"
    lower = attn[0, 0].tril()
    assert torch.allclose(lower.sum(dim=-1), torch.ones(T), atol=1e-5)


def test_mha_forward_shape():
    B, T, d = 4, 7, 64
    mha = MultiHeadAttention(d_model=d, n_heads=8, dropout=0.0, causal=False)
    x = torch.randn(B, T, d)
    y = mha(x)
    assert y.shape == (B, T, d)


def test_mha_causal_no_future_leak():
    """Modifier x[:, t+1:] ne doit pas changer y[:, :t+1] avec masque causal."""
    torch.manual_seed(0)
    B, T, d = 1, 6, 32
    mha = MultiHeadAttention(d_model=d, n_heads=4, dropout=0.0, causal=True)
    mha.eval()
    x = torch.randn(B, T, d)
    y1 = mha(x)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(B, T - 3, d)
    y2 = mha(x2)
    assert torch.allclose(y1[:, :3], y2[:, :3], atol=1e-5), "future tokens leaked into past"


def test_mha_param_count():
    """4 * d^2 params + biais (3*d + d) pour MHA standard."""
    d = 64
    mha = MultiHeadAttention(d_model=d, n_heads=4)
    expected_w = 4 * d * d
    expected_b = 3 * d + d
    actual = sum(p.numel() for p in mha.parameters())
    assert actual == expected_w + expected_b, f"got {actual}, expected {expected_w + expected_b}"
