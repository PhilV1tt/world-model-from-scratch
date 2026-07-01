"""Tests pour SIGReg: collapse, identite/gaussien, intermediaire."""
import torch

from src.sigreg import sigreg, epps_pulley_statistic


def test_sigreg_gaussian_low():
    """Echantillons i.i.d. de N(0, I) -> SIGReg doit etre faible."""
    torch.manual_seed(0)
    z = torch.randn(2048, 64)
    s = sigreg(z, n_proj=512)
    assert s.item() < 0.05, f"expected SIGReg ~0 on gaussian, got {s.item():.4f}"


def test_sigreg_collapse_high():
    """Tous les vecteurs identiques -> SIGReg doit exploser (collapse)."""
    torch.manual_seed(0)
    z = torch.zeros(2048, 64) + torch.randn(1, 64) * 0.001
    s_collapse = sigreg(z, n_proj=512)

    z_gauss = torch.randn(2048, 64)
    s_gauss = sigreg(z_gauss, n_proj=512)
    assert s_collapse > s_gauss * 5, f"collapse SIGReg ({s_collapse:.4f}) should >> gaussian ({s_gauss:.4f})"


def test_sigreg_uniform_intermediate():
    """Distribution uniforme [-1, 1]^d -> SIGReg intermediaire (pas Gaussian, pas collapse)."""
    torch.manual_seed(0)
    z_unif = torch.rand(2048, 64) * 2 - 1
    s_unif = sigreg(z_unif, n_proj=512)
    z_gauss = torch.randn(2048, 64)
    s_gauss = sigreg(z_gauss, n_proj=512)
    assert s_unif > s_gauss, f"uniform SIGReg ({s_unif:.4f}) should > gaussian ({s_gauss:.4f})"


def test_sigreg_differentiable():
    """SIGReg doit etre differentiable pour pouvoir backprop."""
    z = torch.randn(64, 32, requires_grad=True)
    s = sigreg(z, n_proj=128)
    s.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_epps_pulley_zero_on_standard_normal():
    """h ~ N(0,1) -> Epps-Pulley statistic proche de 0."""
    torch.manual_seed(0)
    h = torch.randn(8, 4096)
    stat = epps_pulley_statistic(h)
    assert stat.mean().item() < 0.02, f"got {stat.mean().item():.4f}"
