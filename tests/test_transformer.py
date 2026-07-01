"""Tests pour TransformerBlock et TransformerStack."""
import torch

from src.transformer import FeedForward, TransformerBlock, TransformerStack


def test_feedforward_shape():
    d = 32
    ffn = FeedForward(d_model=d, mlp_ratio=4.0)
    x = torch.randn(2, 5, d)
    y = ffn(x)
    assert y.shape == x.shape


def test_transformer_block_residual():
    d = 32
    block = TransformerBlock(d_model=d, n_heads=4)
    x = torch.randn(2, 6, d)
    y = block(x)
    assert y.shape == x.shape


def test_transformer_stack_depth_changes_output():
    d = 32
    torch.manual_seed(0)
    s2 = TransformerStack(n_layers=2, d_model=d, n_heads=4)
    s2.eval()
    s4 = TransformerStack(n_layers=4, d_model=d, n_heads=4)
    s4.eval()
    x = torch.randn(2, 6, d)
    y2 = s2(x)
    y4 = s4(x)
    assert not torch.allclose(y2, y4), "deeper stack should produce different output"
