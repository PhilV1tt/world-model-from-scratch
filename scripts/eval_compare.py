"""Compare plusieurs runs LeWM avec les memes seeds d'eval."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio

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


def parse_planners(text: str) -> list[str]:
    planners = [p.strip() for p in text.split(",") if p.strip()]
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


def evaluate_planner(planner: str, env, model, device, seed: int, args) -> dict:
    if planner == "random":
        return run_policy_episode(env, random_policy, max_env_steps=args.max_env_steps)
    if planner == "pd":
        return run_policy_episode(env, HeuristicPDPolicy(noise=0.0), max_env_steps=args.max_env_steps)
    if planner in {"model", "model_pd"}:
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
            warm_start_pd=planner == "model_pd",
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
    raise ValueError(f"unknown planner: {planner}")


def evaluate_run(run_dir: Path, planners: list[str], args) -> list[dict]:
    ckpt = run_dir / "ckpt_last.pt"
    device = torch.device(args.device)
    needs_model = any(p.startswith("model") for p in planners)
    model = None
    if needs_model:
        model, _ = load_model(str(ckpt), device)
    rows = []
    rollouts = {planner: [] for planner in planners}
    for planner in planners:
        for ep in range(args.episodes):
            seed = args.seed + ep
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = make_parking(seed=seed, img_size=64)
            out = evaluate_planner(planner, env, model, device, seed, args)
            metrics = strict_metric_row(env)
            env.close()
            success = legacy_success(out)
            row = {
                "run": run_dir.name,
                "planner": planner,
                "episode": ep,
                "seed": seed,
                "init_dist": float(out["init_dist"]),
                "final_dist": float(out["final_dist"]),
                "dist_delta": float(out["init_dist"] - out["final_dist"]),
                "init_ang": float(out["init_ang"]),
                "final_ang": float(out["final_ang"]),
                "success": int(success),
                **metrics,
                "n_steps": int(out["n_steps"]),
            }
            rows.append(row)
            rollouts[planner].append((row, out))
    if args.gif_dir:
        for planner, planner_rollouts in rollouts.items():
            gif_dir = args.gif_dir / run_dir.name / planner
            gif_dir.mkdir(parents=True, exist_ok=True)
            ranked = sorted(planner_rollouts, key=lambda item: item[0]["final_dist"])
            picks = [("best", ranked[0]), ("median", ranked[len(ranked) // 2]), ("worst", ranked[-1])]
            for label, (row, rollout) in picks:
                save_rollout_gif(rollout, gif_dir / f"{label}_ep{row['episode']:02d}_d{row['final_dist']:.1f}m.gif")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/experiments/compare_eval.csv"))
    parser.add_argument("--gif-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--mpc-apply", type=int, default=3)
    parser.add_argument("--max-env-steps", type=int, default=80)
    parser.add_argument("--pop", type=int, default=150)
    parser.add_argument("--elites", type=int, default=15)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--warm-start-pd", action="store_true")
    parser.add_argument("--planners", type=str, default="random,pd,model,model_pd")
    parser.add_argument("--action-l2", type=float, default=DEFAULT_ACTION_L2)
    parser.add_argument("--smooth-l2", type=float, default=DEFAULT_SMOOTH_L2)
    parser.add_argument("--action-smoothing", type=float, default=DEFAULT_ACTION_SMOOTHING)
    parser.add_argument("--trajectory-weight", type=float, default=DEFAULT_TRAJECTORY_WEIGHT)
    parser.add_argument("--pose-cost-weight", type=float, default=DEFAULT_POSE_COST_WEIGHT)
    parser.add_argument("--lateral-weight", type=float, default=DEFAULT_LATERAL_WEIGHT)
    parser.add_argument("--along-weight", type=float, default=DEFAULT_ALONG_WEIGHT)
    parser.add_argument("--heading-weight", type=float, default=DEFAULT_HEADING_WEIGHT)
    parser.add_argument("--collision-weight", type=float, default=DEFAULT_COLLISION_WEIGHT)
    args = parser.parse_args()

    planners = parse_planners(args.planners)
    if args.warm_start_pd:
        planners = [p for p in planners if p != "model"]
        if "model_pd" not in planners:
            planners.append("model_pd")

    rows = []
    for run_dir in args.runs:
        rows.extend(evaluate_run(run_dir, planners, args))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for run_name in sorted({r["run"] for r in rows}):
        for planner in planners:
            run_rows = [r for r in rows if r["run"] == run_name and r["planner"] == planner]
            final = np.array([r["final_dist"] for r in run_rows], dtype=np.float32)
            success = np.array([r["success"] for r in run_rows], dtype=np.float32)
            strict_success = np.array([r["strict_success"] for r in run_rows], dtype=np.float32)
            delta = np.array([r["dist_delta"] for r in run_rows], dtype=np.float32)
            print(
                f"{run_name}/{planner}: final={final.mean():.2f}m "
                f"best={final.min():.2f}m delta={delta.mean():.2f}m "
                f"success={success.mean()*100:.1f}% strict={strict_success.mean()*100:.1f}% n={len(run_rows)}"
            )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
