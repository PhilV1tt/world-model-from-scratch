"""Training script LeWM sur dataset parking-v0."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.lewm import LeWM, LeWMConfig
from src.data import ParkingTrajectoryDataset
from src.decoder import LatentDecoder


LOG_FIELDS = [
    "step",
    "epoch",
    "loss",
    "loss_pred",
    "loss_sigreg",
    "loss_decoder",
    "loss_ar",
    "ar_weight",
    "val_loss",
    "val_loss_pred",
    "val_loss_sigreg",
    "val_loss_ar",
    "lr",
    "grad_norm",
    "loss_ema",
    "steps_per_sec",
    "latent_mean",
    "latent_std",
    "latent_coll",
    "elapsed",
]
SCHEDULE_NAME = "cosine"


def distributed_is_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_module(module):
    return module.module if hasattr(module, "module") else module


def setup_distributed(args) -> tuple[bool, int, int, int | None, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = args.distributed or world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training currently requires CUDA.")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.dist_backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return True, rank, world_size, local_rank, torch.device("cuda", local_rank)

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    return False, 0, 1, None, device


def distributed_average(values: dict[str, float], device: torch.device) -> dict[str, float]:
    if not distributed_is_available():
        return values
    keys = list(values.keys())
    tensor = torch.tensor([float(values[k]) for k in keys], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return {key: float(tensor[i].item()) for i, key in enumerate(keys)}


def fmt_hms(s: float) -> str:
    s = max(0, int(s))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h:d}h{m:02d}m{sec:02d}s" if h else f"{m:d}m{sec:02d}s"


def cosine_lr(step: int, total: int, base_lr: float, warmup: int = 1000) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def compute_total_steps(
    *,
    step: int,
    start_epoch: int,
    epochs: int,
    steps_per_epoch: int,
    max_steps: int,
    total_epochs_hint: int,
    is_resume: bool,
    resume_schedule: str,
    checkpoint_schedule: dict | None,
) -> int:
    if (
        is_resume
        and resume_schedule == "checkpoint"
        and checkpoint_schedule
        and checkpoint_schedule.get("total_steps_target")
    ):
        return int(checkpoint_schedule["total_steps_target"])

    if total_epochs_hint > 0:
        total = total_epochs_hint * steps_per_epoch
    elif is_resume and resume_schedule in {"checkpoint", "extend"}:
        total = step + epochs * steps_per_epoch
    else:
        total = (start_epoch + epochs) * steps_per_epoch

    if max_steps > 0:
        total = min(total, max_steps)
    return max(total, step + 1)


def last_log_lr(*paths: Path) -> float | None:
    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open("r", newline="") as f:
                rows = list(csv.DictReader(f))
        except OSError:
            continue
        for row in reversed(rows):
            try:
                return float(row["lr"])
            except (KeyError, TypeError, ValueError):
                continue
    return None


def check_lr_jump(previous_lr: float | None, next_lr: float, factor: float) -> str | None:
    if previous_lr is None or previous_lr <= 0 or next_lr <= previous_lr * factor:
        return None
    return f"LR jump detected: previous={previous_lr:.3e}, next={next_lr:.3e}, factor>{factor:g}"


def existing_log_fields(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None
    return header or None


def row_for_fields(row: dict, fields: list[str]) -> dict:
    return {field: row.get(field, "") for field in fields}


def autocast_context(device: torch.device, bf16: bool):
    if bf16 and device.type in {"mps", "cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def decoder_loss(model, decoder, obs: torch.Tensor) -> tuple[torch.Tensor | None, float]:
    if decoder is None:
        return None, 0.0
    B, T, C, H, W = obs.shape
    obs_flat = obs.reshape(B * T, C, H, W)
    with torch.no_grad():
        z_for_dec = unwrap_module(model).encode(obs).reshape(B * T, -1)
    rec = decoder(z_for_dec)
    loss_dec = F.mse_loss(rec, obs_flat)
    return loss_dec, float(loss_dec.detach())


def scheduled_ar_weight(base_weight: float, step: int, ramp_steps: int) -> float:
    if base_weight <= 0:
        return 0.0
    if ramp_steps <= 0:
        return base_weight
    return base_weight * min(1.0, max(0.0, step / ramp_steps))


@contextmanager
def batchnorm_batch_stats(*modules):
    """Use batch stats for BatchNorm during validation, then restore buffers.

    The projection heads use BatchNorm. During training the loss sees batch
    statistics, while eval-mode running stats can lag badly because the model is
    still changing. Validation should measure the same normalization path as
    training without mutating running stats.
    """

    saved = []
    for module in modules:
        if module is None:
            continue
        for child in module.modules():
            if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
                saved.append(
                    (
                        child,
                        child.training,
                        child.momentum,
                        child.running_mean.clone() if child.running_mean is not None else None,
                        child.running_var.clone() if child.running_var is not None else None,
                        child.num_batches_tracked.clone() if child.num_batches_tracked is not None else None,
                    )
                )
                child.train()
                child.momentum = 0.0
    try:
        yield
    finally:
        for child, training, momentum, running_mean, running_var, num_batches_tracked in saved:
            child.train(training)
            child.momentum = momentum
            if running_mean is not None:
                child.running_mean.copy_(running_mean)
            if running_var is not None:
                child.running_var.copy_(running_var)
            if num_batches_tracked is not None:
                child.num_batches_tracked.copy_(num_batches_tracked)


@torch.no_grad()
def evaluate_validation(
    model: LeWM,
    decoder: LatentDecoder | None,
    val_dl: DataLoader | None,
    *,
    device: torch.device,
    bf16: bool,
    max_batches: int,
    autoregressive_weight: float,
    bn_stats: str = "batch",
) -> dict:
    if val_dl is None:
        return {}
    if bn_stats not in {"batch", "running"}:
        raise ValueError("bn_stats must be 'batch' or 'running'")
    was_training = model.training
    decoder_was_training = decoder.training if decoder else False
    model.eval()
    if decoder:
        decoder.eval()
    totals = {"val_loss": 0.0, "val_loss_pred": 0.0, "val_loss_sigreg": 0.0, "val_loss_ar": 0.0}
    n_batches = 0
    bn_context = batchnorm_batch_stats(model, decoder) if bn_stats == "batch" else nullcontext()
    with bn_context:
        for i, batch in enumerate(val_dl):
            if i >= max_batches:
                break
            obs = batch["obs"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            with autocast_context(device, bf16):
                out = model(obs, actions, autoregressive_weight=autoregressive_weight)
                loss = out["loss"]
                loss_dec, _ = decoder_loss(model, decoder, obs)
                if loss_dec is not None:
                    loss = loss + 0.5 * loss_dec
            totals["val_loss"] += float(loss.detach())
            totals["val_loss_pred"] += float(out["loss_pred"])
            totals["val_loss_sigreg"] += float(out["loss_sigreg"])
            totals["val_loss_ar"] += float(out["loss_ar"])
            n_batches += 1
    if was_training:
        model.train()
    if decoder:
        decoder.train(decoder_was_training)
    if distributed_is_available():
        device_t = torch.tensor(
            [totals["val_loss"], totals["val_loss_pred"], totals["val_loss_sigreg"], totals["val_loss_ar"], float(n_batches)],
            device=device,
            dtype=torch.float32,
        )
        dist.all_reduce(device_t, op=dist.ReduceOp.SUM)
        totals["val_loss"] = float(device_t[0].item())
        totals["val_loss_pred"] = float(device_t[1].item())
        totals["val_loss_sigreg"] = float(device_t[2].item())
        totals["val_loss_ar"] = float(device_t[3].item())
        n_batches = int(device_t[4].item())
    if n_batches == 0:
        return {}
    return {k: float(v / n_batches) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/parking/train.h5")
    parser.add_argument("--out", type=str, default="runs/last")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distributed", action="store_true", help="enable DistributedDataParallel; torchrun sets this automatically via WORLD_SIZE")
    parser.add_argument("--dist-backend", type=str, default="nccl")
    parser.add_argument("--max-steps", type=int, default=0, help="0 = entire epoch")
    parser.add_argument("--decoder", action="store_true", help="train aux decoder for visualization")
    parser.add_argument("--enc-depth", type=int, default=6)
    parser.add_argument("--pred-depth", type=int, default=3)
    parser.add_argument("--sigreg-weight", type=float, default=0.1)
    parser.add_argument("--sigreg-n-proj", type=int, default=128)
    parser.add_argument("--var-weight", type=float, default=1.0, help="VICReg variance hinge weight (anti-collapse)")
    parser.add_argument("--cov-weight", type=float, default=0.04, help="VICReg covariance weight (latent decorrelation)")
    parser.add_argument("--var-gamma", type=float, default=1.0, help="target per-dim std floor for the variance hinge")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    parser.add_argument("--bf16", action="store_true", help="use bfloat16 autocast on MPS")
    parser.add_argument("--resume", type=str, default="", help="path to ckpt to resume from")
    parser.add_argument("--resume-schedule", choices=["checkpoint", "extend", "args"], default="checkpoint", help="checkpoint=reuse ckpt schedule if present, extend=decay over this extra run, args=use total-epochs-hint/start_epoch")
    parser.add_argument("--total-epochs-hint", type=int, default=0, help="explicit cosine schedule horizon in epochs (0 = infer safely)")
    parser.add_argument("--allow-lr-jump", action="store_true", help="allow resume even if next LR is > lr-jump-factor times previous logged LR")
    parser.add_argument("--lr-jump-factor", type=float, default=5.0)
    parser.add_argument("--stop-after-steps", type=int, default=0, help="stop this run after N optimizer steps, but keep the normal LR schedule")
    parser.add_argument("--save-every-steps", type=int, default=0, help="also refresh ckpt_last.pt every N optimizer steps")
    parser.add_argument("--save-step-archives", action="store_true", help="with --save-every-steps, also keep ckpt_stepXXXXXX.pt archives")
    parser.add_argument("--log-every-steps", type=int, default=1, help="write train metrics every N optimizer steps")
    parser.add_argument("--val-every-steps", type=int, default=0, help="run validation every N optimizer steps (0 = disabled)")
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--validate-at-start", action="store_true", help="also run validation before the first optimizer step")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="deterministic fallback val split for datasets without episode_split")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--val-bn-stats", choices=["batch", "running"], default="batch", help="BatchNorm stats used during validation")
    parser.add_argument("--autoregressive-weight", type=float, default=0.0, help="extra latent rollout loss weight")
    parser.add_argument("--autoregressive-ramp-steps", type=int, default=0, help="linearly ramp AR loss over N optimizer steps")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    distributed, rank, world_size, local_rank, device = setup_distributed(args)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    n_threads = int(os.environ.get("WM_TORCH_THREADS", "0")) or os.cpu_count() or 8
    torch.set_num_threads(n_threads)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if is_main_process(rank):
        dist_note = f" | DDP world_size={world_size}" if distributed else ""
        print(f"device: {device}  | torch threads: {n_threads}{dist_note}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_val = args.val_every_steps > 0
    ds = ParkingTrajectoryDataset(
        args.data,
        seq_len=args.seq_len,
        frame_skip=args.frame_skip,
        split=args.train_split if use_val else None,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )
    if is_main_process(rank):
        print(f"Dataset: {len(ds)} samples (seq_len={args.seq_len}, frame_skip={args.frame_skip})")
    dl_generator = torch.Generator()
    dl_generator.manual_seed(args.seed)
    train_sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=True) if distributed else None
    pin_memory = device.type == "cuda"
    dl = DataLoader(
        ds, batch_size=args.batch, shuffle=train_sampler is None, sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        generator=dl_generator,
    )
    val_dl = None
    if use_val:
        val_ds = ParkingTrajectoryDataset(
            args.data,
            seq_len=args.seq_len,
            frame_skip=args.frame_skip,
            split=args.val_split,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
        )
        if len(val_ds) > 0:
            val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if distributed else None
            val_dl = DataLoader(
                val_ds,
                batch_size=args.batch,
                shuffle=False,
                sampler=val_sampler,
                num_workers=0,
                pin_memory=pin_memory,
                drop_last=False,
            )
            if is_main_process(rank):
                print(f"Validation: {len(val_ds)} samples (split={args.val_split})")
        elif is_main_process(rank):
            print("Validation disabled: val split is empty")

    cfg = LeWMConfig(
        img_size=64,
        patch_size=8,
        in_chans=3,
        embed_dim=192,
        enc_depth=args.enc_depth,
        enc_heads=3,
        pred_d_model=384,
        pred_depth=args.pred_depth,
        pred_heads=16,
        action_dim=2,
        sigreg_weight=args.sigreg_weight,
        sigreg_n_proj=args.sigreg_n_proj,
        var_weight=args.var_weight,
        cov_weight=args.cov_weight,
        var_gamma=args.var_gamma,
        max_history=args.seq_len + 2,
    )
    model = LeWM(cfg).to(device)
    if is_main_process(rank):
        print(f"Model params: {model.num_parameters():,}")
    if distributed and args.compile:
        raise RuntimeError("--compile is not enabled for DDP in this script.")
    if args.compile:
        model = torch.compile(model)

    decoder = None
    if args.decoder:
        decoder = LatentDecoder(z_dim=cfg.embed_dim, img_size=64, patch_size=8, out_chans=3, d_model=192, depth=2, n_heads=3).to(device)

    step = 0
    start_epoch = 0
    checkpoint_schedule = None
    optimizer_state = None
    if args.resume:
        if is_main_process(rank):
            print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer_state = ckpt.get("optimizer")
        if decoder and "decoder" in ckpt:
            decoder.load_state_dict(ckpt["decoder"])
        step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        checkpoint_schedule = ckpt.get("schedule")
        if args.resume_schedule == "checkpoint" and checkpoint_schedule and checkpoint_schedule.get("base_lr"):
            args.lr = float(checkpoint_schedule["base_lr"])
        if is_main_process(rank):
            print(f"Resumed at step={step}, starting epoch={start_epoch}")

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if decoder:
            decoder = DDP(decoder, device_ids=[local_rank], output_device=local_rank)

    params = list(model.parameters())
    if decoder:
        params += list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    if optimizer_state is not None:
        opt.load_state_dict(optimizer_state)

    total_steps = compute_total_steps(
        step=step,
        start_epoch=start_epoch,
        epochs=args.epochs,
        steps_per_epoch=len(dl),
        max_steps=args.max_steps,
        total_epochs_hint=args.total_epochs_hint,
        is_resume=bool(args.resume),
        resume_schedule=args.resume_schedule,
        checkpoint_schedule=checkpoint_schedule,
    )
    if is_main_process(rank):
        print(f"Total steps target (for cosine): {total_steps}")

    next_lr = cosine_lr(step, total_steps, args.lr)
    previous_lr = last_log_lr(log_path := out_dir / "train_log.csv", Path(args.resume).parent / "train_log.csv" if args.resume else Path())
    lr_jump = check_lr_jump(previous_lr, next_lr, args.lr_jump_factor)
    if lr_jump and not args.allow_lr_jump:
        raise RuntimeError(f"{lr_jump}. Use --resume-schedule extend/checkpoint correctly or pass --allow-lr-jump.")
    if lr_jump and is_main_process(rank):
        print(f"WARNING: {lr_jump}")

    log_mode = "a" if args.resume and log_path.exists() else "w"
    log_fields = existing_log_fields(log_path) if log_mode == "a" else None
    log_fields = log_fields or LOG_FIELDS
    log_f = None
    log_writer = None
    if is_main_process(rank):
        log_f = open(log_path, log_mode, newline="")
        log_writer = csv.DictWriter(log_f, fieldnames=log_fields)
        if log_mode == "w":
            log_writer.writeheader()

        with open(out_dir / "config.json", "w") as f:
            json.dump(
                {
                    **vars(args),
                    "cfg": cfg.__dict__,
                    "schedule": {
                        "schedule_name": SCHEDULE_NAME,
                        "total_steps_target": total_steps,
                        "steps_per_epoch": len(dl),
                        "base_lr": args.lr,
                        "warmup_steps": 1000,
                        "world_size": world_size,
                        "per_gpu_batch": args.batch,
                    },
                },
                f,
                indent=2,
            )

    t0 = time()
    end_epoch = start_epoch + args.epochs
    steps_this_run = 0
    should_stop = False
    loss_ema = None
    last_val = {}

    def save_ckpt(epoch: int, step: int, archive_name: str | None = None):
        if not is_main_process(rank):
            return
        ckpt = {
            "model": unwrap_module(model).state_dict(),
            "optimizer": opt.state_dict(),
            "cfg": cfg.__dict__,
            "epoch": epoch,
            "step": step,
            "schedule": {
                "schedule_name": SCHEDULE_NAME,
                "total_steps_target": total_steps,
                "steps_per_epoch": len(dl),
                "base_lr": args.lr,
                "warmup_steps": 1000,
                "resume_schedule": args.resume_schedule,
                "world_size": world_size,
                "per_gpu_batch": args.batch,
            },
        }
        if decoder:
            ckpt["decoder"] = unwrap_module(decoder).state_dict()
        torch.save(ckpt, out_dir / "ckpt_last.pt")
        if archive_name:
            torch.save(ckpt, out_dir / archive_name)

    for epoch in range(start_epoch, end_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{end_epoch}", disable=not is_main_process(rank))
        for batch in pbar:
            obs = batch["obs"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            lr = cosine_lr(step, total_steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr

            with autocast_context(device, args.bf16):
                ar_weight = scheduled_ar_weight(args.autoregressive_weight, step, args.autoregressive_ramp_steps)
                out = model(obs, actions, autoregressive_weight=ar_weight)
                loss = out["loss"]
                loss_dec, loss_dec_val = decoder_loss(model, decoder, obs)
                if loss_dec is not None:
                    loss = loss + 0.5 * loss_dec

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            elapsed = time() - t0
            should_log = args.log_every_steps <= 1 or step % args.log_every_steps == 0
            raw_metrics = None
            loss_val = None
            if should_log:
                raw_metrics = distributed_average(
                    {
                        "loss": float(loss.detach()),
                        "loss_pred": float(out["loss_pred"]),
                        "loss_sigreg": float(out["loss_sigreg"]),
                        "loss_decoder": loss_dec_val,
                        "loss_ar": float(out["loss_ar"]),
                        "ar_weight": ar_weight,
                        "grad_norm": float(grad_norm),
                        "latent_mean": float(out["z_target"].mean()),
                        "latent_std": float(out["z_target"].std()),
                        "latent_coll": float(out["z_target"].reshape(-1, out["z_target"].shape[-1]).std(dim=0).mean()),
                    },
                    device,
                )
                loss_val = raw_metrics["loss"]
                loss_ema = loss_val if loss_ema is None else 0.05 * loss_val + 0.95 * loss_ema
            should_validate = (
                args.val_every_steps > 0
                and val_dl is not None
                and ((args.validate_at_start and steps_this_run == 0) or (step > 0 and step % args.val_every_steps == 0))
            )
            if should_validate:
                last_val = evaluate_validation(
                    model,
                    decoder,
                    val_dl,
                    device=device,
                    bf16=args.bf16,
                    max_batches=args.val_batches,
                    autoregressive_weight=ar_weight,
                    bn_stats=args.val_bn_stats,
                )
            if should_log and is_main_process(rank):
                row = {
                    "step": step,
                    "epoch": epoch,
                    "loss": loss_val,
                    "loss_pred": raw_metrics["loss_pred"],
                    "loss_sigreg": raw_metrics["loss_sigreg"],
                    "loss_decoder": raw_metrics["loss_decoder"],
                    "loss_ar": raw_metrics["loss_ar"],
                    "ar_weight": raw_metrics["ar_weight"],
                    "val_loss": last_val.get("val_loss", ""),
                    "val_loss_pred": last_val.get("val_loss_pred", ""),
                    "val_loss_sigreg": last_val.get("val_loss_sigreg", ""),
                    "val_loss_ar": last_val.get("val_loss_ar", ""),
                    "lr": lr,
                    "grad_norm": raw_metrics["grad_norm"],
                    "loss_ema": loss_ema,
                    "steps_per_sec": world_size * (steps_this_run + 1) / max(elapsed, 1e-9),
                    "latent_mean": raw_metrics["latent_mean"],
                    "latent_std": raw_metrics["latent_std"],
                    "latent_coll": raw_metrics["latent_coll"],
                    "elapsed": elapsed,
                }
                log_writer.writerow(row_for_fields(row, log_fields))
                log_f.flush()

            if should_log and is_main_process(rank):
                if step > 0:
                    s_per_step = elapsed / max(1, steps_this_run + 1)
                    eta_total = (total_steps - step) * s_per_step
                else:
                    eta_total = 0.0
                pbar.set_postfix(
                    loss=f"{loss_val:.3f}",
                    pred=f"{raw_metrics['loss_pred']:.3f}",
                    sig=f"{raw_metrics['loss_sigreg']:.3f}",
                    ar=f"{raw_metrics['loss_ar']:.3f}",
                    arw=f"{raw_metrics['ar_weight']:.2f}",
                    val=f"{last_val.get('val_loss', float('nan')):.3f}" if last_val else ".",
                    lr=f"{lr:.1e}",
                    g=f"{step}/{total_steps}",
                    eta=fmt_hms(eta_total),
                )

            step += 1
            steps_this_run += 1
            if args.save_every_steps > 0 and steps_this_run % args.save_every_steps == 0:
                archive = f"ckpt_step{step:06d}.pt" if args.save_step_archives else None
                save_ckpt(epoch, step, archive)
            if args.stop_after_steps > 0 and steps_this_run >= args.stop_after_steps:
                should_stop = True
                break
            if args.max_steps > 0 and step >= args.max_steps:
                should_stop = True
                break

        save_ckpt(epoch, step, f"ckpt_epoch{epoch:03d}.pt")
        if should_stop:
            break

    if log_f:
        log_f.close()
    if is_main_process(rank):
        print(f"Done. Logs: {log_path}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
