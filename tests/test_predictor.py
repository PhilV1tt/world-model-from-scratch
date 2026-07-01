"""Tests pour Predictor."""
import torch

from src.predictor import Predictor


def test_predictor_shape():
    pred = Predictor(z_dim=192, action_dim=2, d_model=128, depth=2, n_heads=4)
    pred.train()
    z = torch.randn(4, 3, 192)
    a = torch.randn(4, 3, 2)
    z_hat = pred(z, a)
    assert z_hat.shape == (4, 3, 192)


def test_predictor_action_changes_output():
    """Different actions -> different predictions (apres une init non-zero des AdaLN)."""
    torch.manual_seed(0)
    pred = Predictor(z_dim=64, action_dim=2, d_model=64, depth=2, n_heads=4)
    for block in pred.blocks:
        for p in block.adaln_modulation[-1].parameters():
            p.data.add_(torch.randn_like(p) * 0.1)
    pred.eval()
    z = torch.randn(2, 4, 64)
    a1 = torch.zeros(2, 4, 2)
    a2 = torch.ones(2, 4, 2)
    y1 = pred(z, a1)
    y2 = pred(z, a2)
    assert not torch.allclose(y1, y2, atol=1e-3)


def test_predictor_param_count_full_config():
    """Config paper: d_model=384, depth=6, heads=16 -> ~10M params."""
    pred = Predictor(z_dim=192, action_dim=2, d_model=384, depth=6, n_heads=16)
    n = sum(p.numel() for p in pred.parameters())
    assert 7_000_000 < n < 14_000_000, f"Predictor ~10M expected, got {n:,}"
