from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from scripts.train import check_lr_jump, compute_total_steps, cosine_lr, evaluate_validation, last_log_lr, scheduled_ar_weight
from src.lewm import LeWM, LeWMConfig


def test_resume_uses_checkpoint_schedule():
    total = compute_total_steps(
        step=120,
        start_epoch=3,
        epochs=2,
        steps_per_epoch=50,
        max_steps=0,
        total_epochs_hint=0,
        is_resume=True,
        resume_schedule="checkpoint",
        checkpoint_schedule={"total_steps_target": 1000},
    )
    assert total == 1000


def test_old_checkpoint_resume_extends_from_current_step():
    total = compute_total_steps(
        step=120,
        start_epoch=3,
        epochs=2,
        steps_per_epoch=50,
        max_steps=0,
        total_epochs_hint=0,
        is_resume=True,
        resume_schedule="checkpoint",
        checkpoint_schedule=None,
    )
    assert total == 220


def test_lr_jump_guard_detects_large_jump():
    msg = check_lr_jump(previous_lr=1e-5, next_lr=1e-3, factor=5)
    assert msg and "LR jump" in msg
    assert check_lr_jump(previous_lr=1e-4, next_lr=2e-4, factor=5) is None


def test_last_log_lr_reads_latest_valid_lr(tmp_path):
    log = tmp_path / "train_log.csv"
    log.write_text("step,lr\n0,0.0\n1,1e-4\n", encoding="utf-8")
    assert last_log_lr(log) == 1e-4
    assert last_log_lr(Path("/missing")) is None


def test_cosine_lr_is_continuous_under_same_total():
    total = 1000
    a = cosine_lr(200, total, 2e-4)
    b = cosine_lr(201, total, 2e-4)
    assert abs(a - b) < 1e-6


def test_autoregressive_weight_ramps_linearly():
    assert scheduled_ar_weight(0.2, step=0, ramp_steps=100) == 0.0
    assert scheduled_ar_weight(0.2, step=50, ramp_steps=100) == 0.1
    assert scheduled_ar_weight(0.2, step=200, ramp_steps=100) == 0.2
    assert scheduled_ar_weight(0.2, step=1, ramp_steps=0) == 0.2


def test_validation_batchnorm_batch_stats_restores_buffers():
    cfg = LeWMConfig(enc_depth=1, pred_depth=1, pred_d_model=64, pred_heads=4, sigreg_n_proj=16)
    model = LeWM(cfg)
    model.train()
    samples = [
        {
            "obs": torch.rand(3, 3, 64, 64),
            "actions": torch.rand(2, 2) * 2 - 1,
        }
        for _ in range(4)
    ]
    dl = DataLoader(samples, batch_size=2)

    bn_modules = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    before = [
        (
            m.running_mean.clone(),
            m.running_var.clone(),
            m.num_batches_tracked.clone(),
        )
        for m in bn_modules
    ]

    metrics = evaluate_validation(
        model,
        None,
        dl,
        device=torch.device("cpu"),
        bf16=False,
        max_batches=1,
        autoregressive_weight=0.1,
        bn_stats="batch",
    )

    assert model.training
    assert metrics["val_loss"] > 0
    for module, (mean, var, tracked) in zip(bn_modules, before):
        assert torch.equal(module.running_mean, mean)
        assert torch.equal(module.running_var, var)
        assert torch.equal(module.num_batches_tracked, tracked)


def test_validation_rejects_unknown_batchnorm_mode():
    cfg = LeWMConfig(enc_depth=1, pred_depth=1, pred_d_model=64, pred_heads=4, sigreg_n_proj=16)
    model = LeWM(cfg)
    dl = DataLoader(
        [
            {
                "obs": torch.rand(3, 3, 64, 64),
                "actions": torch.rand(2, 2),
            }
        ],
        batch_size=1,
    )
    with pytest.raises(ValueError):
        evaluate_validation(
            model,
            None,
            dl,
            device=torch.device("cpu"),
            bf16=False,
            max_batches=1,
            autoregressive_weight=0.0,
            bn_stats="bad",
        )
