"""Sweep reproductible des hyperparams CEM/MPC pour un checkpoint."""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

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
    run_episode,
)
from scripts.plan import random_policy, run_policy_episode
from src.parking_env import HeuristicPDPolicy, make_parking
from src.parking_metrics import final_pose_metrics, parked_success


VALID_PLANNERS = ("random", "pd", "model", "model_pd")


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_planners(text: str) -> list[str]:
    planners = [p.strip() for p in text.split(",") if p.strip()]
    if not planners:
        raise ValueError("at least one planner is required")
    unknown = [p for p in planners if p not in VALID_PLANNERS]
    if unknown:
        raise ValueError(f"unknown planner(s): {', '.join(unknown)}")
    return planners


def _mean_optional(values: list[float] | None) -> float | None:
    if values is None:
        return None
    arr = np.array(values, dtype=np.float32)
    if arr.size == 0:
        return None
    return float(arr.mean())


def summarize(
    init_dist: list[float],
    final_dist: list[float],
    final_ang: list[float],
    success: list[float],
    strict_success: list[float] | None = None,
    abs_lateral_offset_m: list[float] | None = None,
    abs_along_offset_m: list[float] | None = None,
    abs_heading_error_deg: list[float] | None = None,
    collided: list[float] | None = None,
) -> dict:
    init = np.array(init_dist, dtype=np.float32)
    final = np.array(final_dist, dtype=np.float32)
    ang = np.array(final_ang, dtype=np.float32)
    ok = np.array(success, dtype=np.float32)
    strict = np.array(strict_success if strict_success is not None else success, dtype=np.float32)
    return {
        "mean_init_dist": float(init.mean()),
        "mean_final_dist": float(final.mean()),
        "median_final_dist": float(np.median(final)),
        "best_final_dist": float(final.min()),
        "worst_final_dist": float(final.max()),
        "mean_final_ang": float(ang.mean()),
        "success_rate": float(ok.mean()),
        "strict_success_rate": float(strict.mean()),
        "mean_abs_lateral_offset_m": _mean_optional(abs_lateral_offset_m),
        "mean_abs_along_offset_m": _mean_optional(abs_along_offset_m),
        "mean_abs_heading_error_deg": _mean_optional(abs_heading_error_deg),
        "collision_rate": _mean_optional(collided),
        "mean_dist_delta": float((init - final).mean()),
    }


def eval_planner(planner: str, model, args, device, horizon: int, mpc_apply: int, pop: int, iters: int) -> dict:
    init_dist, final_dist, final_ang, success = [], [], [], []
    strict_success = []
    abs_lateral_offset_m, abs_along_offset_m, abs_heading_error_deg, collided = [], [], [], []
    for ep in range(args.episodes):
        seed = args.seed + ep
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = make_parking(seed=seed, img_size=64)
        if planner == "random":
            out = run_policy_episode(env, random_policy, max_env_steps=args.max_env_steps)
        elif planner == "pd":
            out = run_policy_episode(env, HeuristicPDPolicy(noise=0.0), max_env_steps=args.max_env_steps)
        else:
            out = run_episode(
                model,
                env,
                horizon=horizon,
                mpc_apply=mpc_apply,
                max_env_steps=args.max_env_steps,
                pop_size=pop,
                elites=min(args.elites, pop),
                iters=iters,
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
        metrics = final_pose_metrics(env)
        env.close()
        ok = out["final_dist"] < 1.5 and out["final_ang"] < 15.0
        init_dist.append(float(out["init_dist"]))
        final_dist.append(float(out["final_dist"]))
        final_ang.append(float(out["final_ang"]))
        success.append(float(ok))
        strict_success.append(float(parked_success(metrics)))
        abs_lateral_offset_m.append(float(metrics["abs_lateral_offset_m"]))
        abs_along_offset_m.append(float(metrics["abs_along_offset_m"]))
        abs_heading_error_deg.append(float(metrics["abs_heading_error_deg"]))
        collided.append(float(bool(metrics["collided"])))
    return summarize(
        init_dist,
        final_dist,
        final_ang,
        success,
        strict_success=strict_success,
        abs_lateral_offset_m=abs_lateral_offset_m,
        abs_along_offset_m=abs_along_offset_m,
        abs_heading_error_deg=abs_heading_error_deg,
        collided=collided,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("runs/last/ckpt_last.pt"))
    parser.add_argument("--out", type=Path, default=Path("runs/experiments/planner_sweep.csv"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--horizons", type=str, default="3,5")
    parser.add_argument("--mpc-apply", type=str, default="1,3,5")
    parser.add_argument("--pops", type=str, default="80,150")
    parser.add_argument("--iters", type=str, default="3,8")
    parser.add_argument("--elites", type=int, default=15)
    parser.add_argument("--max-env-steps", type=int, default=80)
    parser.add_argument("--warm-start-pd", action="store_true")
    parser.add_argument("--planners", type=str, default="random,pd,model,model_pd")
    parser.add_argument("--include-baselines", action=argparse.BooleanOptionalAction, default=True)
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

    device = torch.device(args.device)
    planners = parse_planners(args.planners)
    if not args.include_baselines:
        planners = [p for p in planners if p in {"model", "model_pd"}]
    if args.warm_start_pd:
        planners = [p for p in planners if p != "model"]
        if "model_pd" not in planners:
            planners.append("model_pd")
    if not planners:
        raise ValueError("no planner left to evaluate")

    needs_model = any(p.startswith("model") for p in planners)
    model = None
    if needs_model:
        model, _ = load_model(str(args.ckpt), device)

    rows = []
    for planner in planners:
        if planner in {"random", "pd"}:
            stats = eval_planner(planner, model, args, device, horizon=0, mpc_apply=0, pop=0, iters=0)
            row = {
                "planner": planner,
                "ckpt": "",
                "horizon": 0,
                "mpc_apply": 0,
                "pop": 0,
                "iters": 0,
                "warm_start_pd": 0,
                "elites": 0,
                "episodes": args.episodes,
                "seed": args.seed,
                "max_env_steps": args.max_env_steps,
                "action_l2": 0.0,
                "smooth_l2": 0.0,
                "action_smoothing": 0.0,
                "trajectory_weight": 0.0,
                "pose_cost_weight": 0.0,
                **stats,
            }
            rows.append(row)
            print(row)

    grid = itertools.product(
        parse_ints(args.horizons),
        parse_ints(args.mpc_apply),
        parse_ints(args.pops),
        parse_ints(args.iters),
    )
    model_planners = [p for p in planners if p.startswith("model")]
    for horizon, mpc_apply, pop, iters in grid:
        for planner in model_planners:
            stats = eval_planner(planner, model, args, device, horizon, mpc_apply, pop, iters)
            row = {
                "planner": planner,
                "ckpt": str(args.ckpt),
                "horizon": horizon,
                "mpc_apply": mpc_apply,
                "pop": pop,
                "iters": iters,
                "warm_start_pd": int(planner == "model_pd"),
                "elites": min(args.elites, pop),
                "episodes": args.episodes,
                "seed": args.seed,
                "max_env_steps": args.max_env_steps,
                "action_l2": args.action_l2,
                "smooth_l2": args.smooth_l2,
                "action_smoothing": args.action_smoothing,
                "trajectory_weight": args.trajectory_weight,
                "pose_cost_weight": args.pose_cost_weight,
                **stats,
            }
            rows.append(row)
            print(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        fieldnames = [
            "planner",
            "ckpt",
            "horizon",
            "mpc_apply",
            "pop",
            "iters",
            "warm_start_pd",
            "elites",
            "episodes",
            "seed",
            "max_env_steps",
            "action_l2",
            "smooth_l2",
            "action_smoothing",
            "trajectory_weight",
            "pose_cost_weight",
            "mean_init_dist",
            "mean_final_dist",
            "median_final_dist",
            "best_final_dist",
            "worst_final_dist",
            "mean_final_ang",
            "success_rate",
            "strict_success_rate",
            "mean_abs_lateral_offset_m",
            "mean_abs_along_offset_m",
            "mean_abs_heading_error_deg",
            "collision_rate",
            "mean_dist_delta",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
