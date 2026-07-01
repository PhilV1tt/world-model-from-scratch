"""Evaluation stable pour decider si un run LeWM progresse vraiment.

Compare sur les memes seeds:
- random baseline
- PD heuristic baseline
- model CEM
- model CEM + PD warm-start

Sorties:
- episodes.csv: une ligne par episode/planner
- summary.json: stats agregees
- gifs/: best/median/worst par planner si demande
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.plan import (
    DEFAULT_ACTION_L2,
    DEFAULT_ALONG_WEIGHT,
    DEFAULT_COLLISION_WEIGHT,
    DEFAULT_HEADING_WEIGHT,
    DEFAULT_LATERAL_WEIGHT,
    DEFAULT_POSE_COST_WEIGHT,
    DEFAULT_ACTION_SMOOTHING,
    DEFAULT_SMOOTH_L2,
    DEFAULT_TRAJECTORY_WEIGHT,
    load_model,
    random_policy,
    run_episode,
    run_policy_episode,
)
from src.parking_env import HeuristicPDPolicy, make_parking
from src.parking_metrics import final_pose_metrics, parked_success


VALID_PLANNERS = ("random", "pd", "model", "model_pd")
CSV_FIELDS = (
    "planner",
    "episode",
    "seed",
    "init_dist",
    "init_ang",
    "final_dist",
    "final_ang",
    "success",
    "strict_success",
    "lateral_offset_m",
    "along_offset_m",
    "abs_lateral_offset_m",
    "abs_along_offset_m",
    "heading_error_deg",
    "abs_heading_error_deg",
    "speed_mps",
    "collided",
    "n_steps",
)


def parse_planners(raw: str) -> list[str]:
    planners = [p.strip() for p in raw.split(",") if p.strip()]
    if not planners:
        raise ValueError("at least one planner is required")
    unknown = [p for p in planners if p not in VALID_PLANNERS]
    if unknown:
        raise ValueError(f"unknown planner(s): {', '.join(unknown)}")
    return planners


def save_rollout_gif(result: dict, path: Path):
    frames = []
    sep = np.full((64, 4, 3), 255, dtype=np.uint8)
    for frame in result["frames"]:
        composed = np.concatenate([frame, sep, result["goal_img"]], axis=1)
        frames.append(composed.repeat(4, axis=0).repeat(4, axis=1))
    if frames:
        imageio.mimsave(path, frames, fps=20)


def legacy_success(result: dict) -> bool:
    return result["final_dist"] < 1.5 and result["final_ang"] < 15.0


def strict_metric_row(env) -> dict:
    metrics = final_pose_metrics(env)
    return {
        "strict_success": int(parked_success(metrics)),
        "lateral_offset_m": float(metrics["lateral_offset_m"]),
        "along_offset_m": float(metrics["along_offset_m"]),
        "abs_lateral_offset_m": float(metrics["abs_lateral_offset_m"]),
        "abs_along_offset_m": float(metrics["abs_along_offset_m"]),
        "heading_error_deg": float(metrics["heading_error_deg"]),
        "abs_heading_error_deg": float(metrics["abs_heading_error_deg"]),
        "speed_mps": float(metrics["speed_mps"]),
        "collided": int(bool(metrics["collided"])),
    }


def _mean_optional(rows: list[dict], key: str) -> float | None:
    if not rows or any(key not in r for r in rows):
        return None
    values = np.array([r[key] for r in rows], dtype=np.float32)
    return float(values.mean())


def summarize(rows: list[dict]) -> dict:
    final_dist = np.array([r["final_dist"] for r in rows], dtype=np.float32)
    final_ang = np.array([r["final_ang"] for r in rows], dtype=np.float32)
    init_dist = np.array([r["init_dist"] for r in rows], dtype=np.float32)
    success = np.array([r["success"] for r in rows], dtype=np.float32)
    strict_success = np.array([r.get("strict_success", r["success"]) for r in rows], dtype=np.float32)
    return {
        "episodes": len(rows),
        "mean_init_dist": float(init_dist.mean()),
        "mean_final_dist": float(final_dist.mean()),
        "median_final_dist": float(np.median(final_dist)),
        "best_final_dist": float(final_dist.min()),
        "worst_final_dist": float(final_dist.max()),
        "mean_final_ang": float(final_ang.mean()),
        "success_rate": float(success.mean()),
        "strict_success_rate": float(strict_success.mean()),
        "mean_abs_lateral_offset_m": _mean_optional(rows, "abs_lateral_offset_m"),
        "mean_abs_along_offset_m": _mean_optional(rows, "abs_along_offset_m"),
        "mean_abs_heading_error_deg": _mean_optional(rows, "abs_heading_error_deg"),
        "collision_rate": _mean_optional(rows, "collided"),
        "mean_dist_delta": float((init_dist - final_dist).mean()),
    }


def eval_policy_name(name: str, env, model, device, args, seed: int) -> dict:
    if name == "random":
        return run_policy_episode(env, random_policy, max_env_steps=args.max_env_steps)
    if name == "pd":
        return run_policy_episode(env, HeuristicPDPolicy(noise=0.0), max_env_steps=args.max_env_steps)
    if name in {"model", "model_pd"}:
        return run_episode(
            model,
            env,
            horizon=args.horizon,
            mpc_apply=args.mpc_apply,
            max_env_steps=args.max_env_steps,
            pop_size=args.pop,
            elites=args.elites,
            iters=args.iters,
            device=device,
            warm_start_pd=name == "model_pd",
            action_l2=args.action_l2,
            smooth_l2=args.smooth_l2,
            action_smoothing=args.action_smoothing,
            trajectory_weight=args.trajectory_weight,
            pose_cost_weight=args.pose_cost_weight,
            lateral_weight=args.lateral_weight,
            along_weight=args.along_weight,
            heading_weight=args.heading_weight,
            collision_weight=args.collision_weight,
            seed=seed,
        )
    raise ValueError(f"unknown planner: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--planners", type=str, default="random,pd,model,model_pd")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--mpc-apply", type=int, default=3)
    parser.add_argument("--max-env-steps", type=int, default=80)
    parser.add_argument("--pop", type=int, default=150)
    parser.add_argument("--elites", type=int, default=15)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--action-l2", type=float, default=DEFAULT_ACTION_L2)
    parser.add_argument("--smooth-l2", type=float, default=DEFAULT_SMOOTH_L2)
    parser.add_argument("--action-smoothing", type=float, default=DEFAULT_ACTION_SMOOTHING)
    parser.add_argument("--trajectory-weight", type=float, default=DEFAULT_TRAJECTORY_WEIGHT)
    parser.add_argument("--pose-cost-weight", type=float, default=DEFAULT_POSE_COST_WEIGHT)
    parser.add_argument("--lateral-weight", type=float, default=DEFAULT_LATERAL_WEIGHT)
    parser.add_argument("--along-weight", type=float, default=DEFAULT_ALONG_WEIGHT)
    parser.add_argument("--heading-weight", type=float, default=DEFAULT_HEADING_WEIGHT)
    parser.add_argument("--collision-weight", type=float, default=DEFAULT_COLLISION_WEIGHT)
    parser.add_argument("--gifs", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")

    ckpt = args.ckpt or ((args.run_dir / "ckpt_last.pt") if args.run_dir else Path("runs/last/ckpt_last.pt"))
    out_dir = args.out_dir or (ckpt.parent / "eval_protocol")
    planners = parse_planners(args.planners)

    needs_model = any(p.startswith("model") for p in planners)
    device = torch.device(args.device)
    model = None
    if needs_model:
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        model, _ = load_model(str(ckpt), device)

    out_dir.mkdir(parents=True, exist_ok=True)
    gif_dir = out_dir / "gifs"
    if args.gifs:
        gif_dir.mkdir(exist_ok=True)

    rows = []
    rollouts = {planner: [] for planner in planners}
    t0 = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    for planner in planners:
        for ep in range(args.episodes):
            seed = args.seed + ep
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = make_parking(seed=seed, img_size=64)
            out = eval_policy_name(planner, env, model, device, args, seed)
            metrics = strict_metric_row(env)
            env.close()
            success = legacy_success(out)
            row = {
                "planner": planner,
                "episode": ep,
                "seed": seed,
                "init_dist": float(out["init_dist"]),
                "init_ang": float(out["init_ang"]),
                "final_dist": float(out["final_dist"]),
                "final_ang": float(out["final_ang"]),
                "success": int(success),
                **metrics,
                "n_steps": int(out["n_steps"]),
            }
            rows.append(row)
            rollouts[planner].append((row, out))
            print(
                f"{planner:8s} ep={ep:02d} seed={seed} "
                f"final={row['final_dist']:.2f}m ang={row['final_ang']:.1f} "
                f"success={bool(success)} strict={bool(row['strict_success'])}"
            )

    csv_path = out_dir / "episodes.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "ckpt": str(ckpt),
        "out_dir": str(out_dir),
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "seeds": [args.seed + ep for ep in range(args.episodes)],
        "episodes": args.episodes,
        "planner_order": planners,
        "planners": {},
        "elapsed_sec": time.time() - t0,
        "planner_config": {
            "horizon": args.horizon,
            "mpc_apply": args.mpc_apply,
            "max_env_steps": args.max_env_steps,
            "pop": args.pop,
            "elites": args.elites,
            "iters": args.iters,
            "action_l2": args.action_l2,
            "smooth_l2": args.smooth_l2,
            "action_smoothing": args.action_smoothing,
            "trajectory_weight": args.trajectory_weight,
            "pose_cost_weight": args.pose_cost_weight,
            "lateral_weight": args.lateral_weight,
            "along_weight": args.along_weight,
            "heading_weight": args.heading_weight,
            "collision_weight": args.collision_weight,
        },
        "artifacts": {
            "gif_dir": str(gif_dir) if args.gifs else None,
            "gifs": {},
        },
    }
    for planner in planners:
        planner_rows = [r for r in rows if r["planner"] == planner]
        summary["planners"][planner] = summarize(planner_rows)
        if args.gifs and rollouts[planner]:
            ranked = sorted(rollouts[planner], key=lambda item: item[0]["final_dist"])
            picks = [("best", ranked[0]), ("median", ranked[len(ranked) // 2]), ("worst", ranked[-1])]
            summary["artifacts"]["gifs"][planner] = {}
            for label, (row, rollout) in picks:
                gif_path = gif_dir / f"{planner}_{label}_ep{row['episode']:02d}_d{row['final_dist']:.1f}m.gif"
                save_rollout_gif(rollout, gif_path)
                summary["artifacts"]["gifs"][planner][label] = str(gif_path)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for planner in planners:
        stats = summary["planners"][planner]
        print(
            f"{planner:8s} success={100 * stats['success_rate']:.1f}% "
            f"strict={100 * stats['strict_success_rate']:.1f}% "
            f"mean={stats['mean_final_dist']:.2f}m median={stats['median_final_dist']:.2f}m "
            f"best={stats['best_final_dist']:.2f}m delta={stats['mean_dist_delta']:.2f}m"
        )
    print(f"saved {csv_path}")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
