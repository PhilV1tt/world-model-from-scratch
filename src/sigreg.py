"""SIGReg: Sketched Isotropic Gaussian Regularizer.

Pousse la distribution des embeddings vers N(0, I) en haute dimension
en utilisant Cramer-Wold + test d'Epps-Pulley sur des projections 1D.

Reference: Balestriero & LeCun 2025 (LeJEPA), arxiv 2511.08544.
Section 3.1 + Appendix A du paper LeWM.

Algorithme:
  1. Echantillonner M directions u^(m) uniformes sur S^{d-1}.
  2. Projeter Z (B, d) sur chaque u -> h^(m) (B,).
  3. Appliquer le test d'Epps-Pulley sur chaque projection 1D.
  4. Moyenner les statistiques.
"""
from __future__ import annotations

import math
import torch


def epps_pulley_statistic(h: torch.Tensor, n_knots: int = 8, t_min: float = 0.2, t_max: float = 4.0) -> torch.Tensor:
    """Statistique d'Epps-Pulley pour tester la normalite univariee.

    h: (M, B) projections 1D, **standardisees** (mean 0, std 1).

    Calcule int_{-inf}^{inf} w(t) |phi_N(t; h) - phi_0(t)|^2 dt
    avec w(t) = exp(-t^2/2) (poids gaussien, paper LeJEPA),
    par quadrature trapezoidale sur [t_min, t_max].

    Comme phi_0 est la fct caracteristique de N(0,1) -> phi_0(t) = exp(-t^2/2),
    et le test est symetrique, on integre seulement sur [t_min, t_max] (positif).
    """
    M, B = h.shape
    t = torch.linspace(t_min, t_max, n_knots, device=h.device)
    t = t.view(1, n_knots, 1)
    h_ = h.unsqueeze(1)
    arg = t * h_
    cos_real = torch.cos(arg).mean(dim=2)
    sin_imag = torch.sin(arg).mean(dim=2)

    target_real = torch.exp(-(t.squeeze(2) ** 2) / 2.0)
    diff_real = cos_real - target_real
    diff_imag = sin_imag

    integrand = (diff_real ** 2 + diff_imag ** 2) * torch.exp(-(t.squeeze(2) ** 2) / 2.0)

    dt = (t_max - t_min) / (n_knots - 1)
    w = torch.full_like(integrand, dt)
    w[:, 0] = dt / 2.0
    w[:, -1] = dt / 2.0
    stat = (integrand * w).sum(dim=1)
    return stat


def sigreg(z: torch.Tensor, n_proj: int = 1024, n_knots: int = 8, generator: torch.Generator | None = None) -> torch.Tensor:
    """SIGReg loss : pousse z (N, d) vers une isotropique gaussienne N(0, I).

    z: (N, d) embeddings (peut concatener encoder + target).
    n_proj: nombre de projections aleatoires (M=1024 dans la reference).
    """
    if z.dim() > 2:
        z = z.reshape(-1, z.shape[-1])
    N, d = z.shape

    z_mean = z.mean(dim=0, keepdim=True)
    z_std = z.std(dim=0, keepdim=True).clamp(min=1e-6)
    z_norm = (z - z_mean) / z_std

    if generator is None:
        u = torch.randn(n_proj, d, device=z.device)
    else:
        u = torch.randn(n_proj, d, device=z.device, generator=generator)
    u = u / u.norm(dim=1, keepdim=True).clamp(min=1e-6)

    h = z_norm @ u.T
    h = h.T

    h_mean = h.mean(dim=1, keepdim=True)
    h_std = h.std(dim=1, keepdim=True).clamp(min=1e-6)
    h = (h - h_mean) / h_std

    stat = epps_pulley_statistic(h, n_knots=n_knots)
    return stat.mean()
