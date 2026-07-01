"""Tests pour AdaLNBlock."""
import torch

from src.adaln import AdaLNBlock


def test_adaln_block_shape():
    d = 64
    block = AdaLNBlock(d_model=d, n_heads=8, action_dim=2)
    x = torch.randn(2, 5, d)
    a = torch.randn(2, 2)
    y = block(x, a)
    assert y.shape == x.shape


def test_adaln_init_zero_means_identity():
    """A l'initialisation, AdaLN modulation = 0 -> action n'a aucun effet sur le residual.

    Avec init zero, gamma=beta=alpha=0, donc:
      h = LN(x) * 1 + 0 = LN(x)  (LayerNorm sans affine)
      x = x + 0 * attn(h) = x
      x = x + 0 * ffn(LN(x))     = x
    Donc forward = identite.
    """
    d = 64
    block = AdaLNBlock(d_model=d, n_heads=8, action_dim=2)
    block.eval()
    x = torch.randn(2, 5, d)
    a1 = torch.randn(2, 2)
    a2 = torch.randn(2, 2)
    y1 = block(x, a1)
    y2 = block(x, a2)
    assert torch.allclose(y1, y2, atol=1e-6), "init AdaLN should give identical output for any action"
    assert torch.allclose(y1, x, atol=1e-6), "init AdaLN should be identity"

def test_adaln_after_perturbation_uses_action():
    """Apres un step de gradient sur AdaLN, l'action doit influencer la sortie."""
    torch.manual_seed(0)
    d = 64
    block = AdaLNBlock(d_model=d, n_heads=8, action_dim=2)
    for p in block.adaln_modulation[-1].parameters():
        p.data.add_(torch.randn_like(p) * 0.1)
    block.eval()
    x = torch.randn(2, 5, d)
    a1 = torch.zeros(2, 2)
    a2 = torch.ones(2, 2) * 5.0
    y1 = block(x, a1)
    y2 = block(x, a2)
    assert not torch.allclose(y1, y2, atol=1e-3), "action should change output after AdaLN init perturbation"
