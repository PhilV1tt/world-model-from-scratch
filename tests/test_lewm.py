"""Tests pour LeWM modele complet."""
import torch
import torch.nn as nn

from src.lewm import LeWM, LeWMConfig


class AddActionPredictor(nn.Module):
    def forward(self, z_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        assert z_seq.shape[:2] == action_seq.shape[:2]
        return z_seq + action_seq.sum(dim=-1, keepdim=True)


def test_lewm_param_count_15M():
    """Config paper: ~15M params (5M encoder + 10M predictor)."""
    cfg = LeWMConfig()
    model = LeWM(cfg)
    n = model.num_parameters()
    assert 12_000_000 < n < 18_000_000, f"LeWM ~15M expected, got {n:,}"
    print(f"LeWM total params: {n:,}")


def test_lewm_forward_smoke():
    """Forward complet sur un mini batch produit une loss finie et differentiable."""
    cfg = LeWMConfig(enc_depth=2, pred_depth=2, sigreg_n_proj=64)
    model = LeWM(cfg)
    model.train()
    B, T = 2, 3
    obs = torch.rand(B, T + 1, 3, 64, 64)
    actions = torch.rand(B, T, 2) * 2 - 1
    out = model(obs, actions)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["loss_ar"])
    out["loss"].backward()
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert has_grad


def test_lewm_forward_with_autoregressive_loss_backprops():
    cfg = LeWMConfig(enc_depth=1, pred_depth=1, pred_d_model=64, pred_heads=4, sigreg_n_proj=16)
    model = LeWM(cfg)
    model.train()
    obs = torch.rand(2, 4, 3, 64, 64)
    actions = torch.rand(2, 3, 2) * 2 - 1
    out = model(obs, actions, autoregressive_weight=0.2)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["loss_ar"])
    assert float(out["loss_ar"]) > 0.0
    out["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)


def test_lewm_rollout_latents_shape_and_cost():
    cfg = LeWMConfig(enc_depth=1, pred_depth=1, sigreg_n_proj=16, max_history=3)
    model = LeWM(cfg)
    z0 = torch.randn(2, cfg.embed_dim)
    actions = torch.randn(2, 4, cfg.action_dim)
    rollout = model.rollout_latents(z0, actions)
    assert rollout.shape == (2, 4, cfg.embed_dim)
    cost = model.latent_goal_cost(rollout, torch.randn(2, cfg.embed_dim))
    assert cost.shape == (2,)
    assert torch.isfinite(cost).all()


def test_rollout_latents_matches_autoregressive_steps():
    cfg = LeWMConfig(
        embed_dim=12,
        enc_depth=1,
        pred_d_model=16,
        pred_depth=1,
        pred_heads=4,
        action_dim=1,
        sigreg_n_proj=16,
        max_history=2,
    )
    model = LeWM(cfg)
    model.predictor = AddActionPredictor()
    z0 = torch.zeros(1, cfg.embed_dim)
    actions = torch.tensor([[[1.0], [2.0], [3.0]]])

    rollout = model.rollout_latents(z0, actions)

    expected = torch.tensor([1.0, 3.0, 6.0]).view(1, 3, 1).expand(1, 3, cfg.embed_dim)
    assert torch.allclose(rollout, expected)


def test_rollout_latents_with_history_uses_aligned_actions():
    cfg = LeWMConfig(
        embed_dim=12,
        enc_depth=1,
        pred_d_model=16,
        pred_depth=1,
        pred_heads=4,
        action_dim=1,
        sigreg_n_proj=16,
        max_history=2,
    )
    model = LeWM(cfg)
    model.predictor = AddActionPredictor()
    z_hist = torch.stack([torch.zeros(cfg.embed_dim), torch.full((cfg.embed_dim,), 10.0)], dim=0).unsqueeze(0)
    actions = torch.tensor([[[4.0], [1.0], [2.0]]])

    rollout = model.rollout_latents(z_hist, actions)

    expected = torch.tensor([11.0, 13.0]).view(1, 2, 1).expand(1, 2, cfg.embed_dim)
    assert torch.allclose(rollout, expected)


def test_rollout_latents_rejects_history_without_actions():
    cfg = LeWMConfig(
        embed_dim=12,
        enc_depth=1,
        pred_d_model=16,
        pred_depth=1,
        pred_heads=4,
        action_dim=1,
        sigreg_n_proj=16,
        max_history=3,
    )
    model = LeWM(cfg)
    z_hist = torch.zeros(2, 3, cfg.embed_dim)
    actions = torch.zeros(2, 1, cfg.action_dim)
    try:
        model.rollout_latents(z_hist, actions)
    except ValueError as exc:
        assert "too short" in str(exc)
    else:
        raise AssertionError("expected ValueError")
