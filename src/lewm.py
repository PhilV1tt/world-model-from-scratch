"""LeWorldModel: assemblage encoder + predictor.

Passage avant complet: perte MSE en latent, SIGReg, terme de variance/covariance.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import ViT
from .predictor import Predictor
from .sigreg import sigreg


@dataclass
class LeWMConfig:
    img_size: int = 64
    patch_size: int = 8
    in_chans: int = 3
    embed_dim: int = 192
    enc_depth: int = 12
    enc_heads: int = 3
    enc_dropout: float = 0.0

    pred_d_model: int = 384
    pred_depth: int = 6
    pred_heads: int = 16
    pred_dropout: float = 0.1

    action_dim: int = 2
    sigreg_weight: float = 0.1
    sigreg_n_proj: int = 256
    var_weight: float = 1.0
    cov_weight: float = 0.04
    var_gamma: float = 1.0
    max_history: int = 8


class LeWM(nn.Module):
    def __init__(self, cfg: LeWMConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ViT(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
            depth=cfg.enc_depth,
            n_heads=cfg.enc_heads,
            dropout=cfg.enc_dropout,
            proj_dim=cfg.embed_dim,
        )
        self.predictor = Predictor(
            z_dim=cfg.embed_dim,
            action_dim=cfg.action_dim,
            d_model=cfg.pred_d_model,
            depth=cfg.pred_depth,
            n_heads=cfg.pred_heads,
            dropout=cfg.pred_dropout,
            max_history=cfg.max_history,
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, C, H, W) ou (B, T, C, H, W). Retourne (B, embed_dim) ou (B, T, embed_dim)."""
        if obs.dim() == 5:
            B, T, C, H, W = obs.shape
            z = self.encoder(obs.reshape(B * T, C, H, W))
            return z.reshape(B, T, -1)
        return self.encoder(obs)

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        autoregressive_weight: float = 0.0,
    ) -> dict:
        """
        obs_seq: (B, T+1, C, H, W) frames consecutives.
        action_seq: (B, T, action_dim) actions a_0,...,a_{T-1} entre frames.

        Encode toutes les frames, predit les latents futurs depuis les latents passes,
        compare aux latents cibles.
        """
        B, Tp1, C, H, W = obs_seq.shape
        T = Tp1 - 1

        z_all = self.encode(obs_seq)
        z_input = z_all[:, :T]
        z_target = z_all[:, 1:].detach()

        z_hat = self.predictor(z_input, action_seq)

        loss_pred = F.mse_loss(z_hat, z_target)

        z_for_sig = z_all.reshape(-1, z_all.shape[-1])
        loss_sig = sigreg(z_for_sig, n_proj=self.cfg.sigreg_n_proj)

        # Anti-collapse VICReg. SIGReg est invariant a l'echelle (il standardise z
        # avant le test de gaussianite) donc il n'empeche pas le collapse a lui seul.
        # Variance hinge: garder un ecart-type par dimension >= var_gamma a travers les echantillons.
        std = torch.sqrt(z_for_sig.var(dim=0) + 1e-4)
        loss_var = torch.mean(F.relu(self.cfg.var_gamma - std))
        # Covariance: decorreler les dimensions du latent.
        zc = z_for_sig - z_for_sig.mean(dim=0, keepdim=True)
        n = zc.shape[0]
        d = zc.shape[1]
        cov = (zc.T @ zc) / max(n - 1, 1)
        loss_cov = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / d

        loss = (
            loss_pred
            + self.cfg.sigreg_weight * loss_sig
            + self.cfg.var_weight * loss_var
            + self.cfg.cov_weight * loss_cov
        )
        loss_ar = z_all.new_zeros(())
        if autoregressive_weight > 0:
            z_rollout = self.rollout_latents(z_all[:, :1], action_seq)
            loss_ar = F.mse_loss(z_rollout, z_target)
            loss = loss + float(autoregressive_weight) * loss_ar

        return {
            "loss": loss,
            "loss_pred": loss_pred.detach(),
            "loss_sigreg": loss_sig.detach(),
            "loss_var": loss_var.detach(),
            "loss_cov": loss_cov.detach(),
            "loss_ar": loss_ar.detach(),
            "z_hat": z_hat.detach(),
            "z_target": z_target.detach(),
        }

    def rollout_latents(
        self,
        z_init: torch.Tensor,
        action_seq: torch.Tensor,
        history_size: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive latent rollout.

        z_init: (B, D) current latent or (B, H, D) history latents.
        action_seq: (B, T, A) future actions when z_init is (B, D).
        If z_init is (B, H, D), action_seq must include H-1 history actions
        followed by future actions. Returns only predicted future latents.
        """
        if action_seq.dim() != 3:
            raise ValueError("action_seq must have shape (B, T, action_dim)")
        if z_init.dim() == 2:
            z_seq = z_init.unsqueeze(1)
        elif z_init.dim() == 3:
            z_seq = z_init
        else:
            raise ValueError("z_init must have shape (B, D) or (B, H, D)")
        if z_seq.shape[0] != action_seq.shape[0]:
            raise ValueError("z_init and action_seq batch sizes differ")
        if action_seq.shape[-1] != self.cfg.action_dim:
            raise ValueError("action_seq last dim must match cfg.action_dim")

        history_size = int(history_size or self.cfg.max_history)
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        history_size = min(history_size, self.cfg.max_history)

        past_actions = z_seq.shape[1] - 1
        if action_seq.shape[1] < past_actions:
            raise ValueError("action_seq is too short for the provided latent history")
        future_steps = action_seq.shape[1] - past_actions
        if future_steps == 0:
            return z_seq.new_empty(z_seq.shape[0], 0, z_seq.shape[-1])

        preds = []
        for t in range(future_steps):
            action_pos = past_actions + t
            ctx_len = min(history_size, z_seq.shape[1], action_pos + 1)
            z_ctx = z_seq[:, -ctx_len:]
            a_ctx = action_seq[:, action_pos + 1 - ctx_len : action_pos + 1]
            z_next = self.predictor(z_ctx, a_ctx)[:, -1:]
            preds.append(z_next)
            z_seq = torch.cat([z_seq, z_next], dim=1)
            if z_seq.shape[1] > history_size:
                z_seq = z_seq[:, -history_size:]
        return torch.cat(preds, dim=1)

    def latent_goal_cost(self, z_rollout: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """Last-step latent MSE cost, matching the LeWM eval pattern."""
        if z_goal.dim() == 2:
            z_goal = z_goal.unsqueeze(1)
        goal = z_goal[:, -1:, :].expand(z_rollout.shape[0], 1, z_rollout.shape[-1])
        return F.mse_loss(z_rollout[:, -1:, :], goal.detach(), reduction="none").sum(dim=(1, 2))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
