"""Eval baselines (random, PD heuristique) sur parking-v0.

Compare avec les resultats du planner CEM (issus de plan.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.parking_env import make_parking, goal_distance, HeuristicPDPolicy


def random_policy(env):
    return np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)


def run_baseline(env, policy, max_steps: int = 60):
    obs, info = env.reset()
    init_dist, init_ang = goal_distance(env)
    for step in range(max_steps):
        a = policy(env)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    final_dist, final_ang = goal_distance(env)
    return {"init_dist": init_dist, "init_ang": init_ang, "final_dist": final_dist, "final_ang": final_ang}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--out", type=str, default="runs/last/baselines.npz")
    args = parser.parse_args()

    np.random.seed(args.seed)
    pd = HeuristicPDPolicy(noise=0.0)

    rand_res, pd_res = [], []
    for ep in range(args.episodes):
        env = make_parking(seed=args.seed + ep, img_size=64)
        rand_res.append(run_baseline(env, random_policy, max_steps=args.max_steps))
        env.close()

        env = make_parking(seed=args.seed + ep, img_size=64)
        pd_res.append(run_baseline(env, pd, max_steps=args.max_steps))
        env.close()

    def stats(res, label):
        init_d = np.array([r["init_dist"] for r in res])
        final_d = np.array([r["final_dist"] for r in res])
        success = (final_d < 1.5) & (np.array([r["final_ang"] for r in res]) < 15.0)
        print(f"{label}: init dist mean={init_d.mean():.2f}, final dist mean={final_d.mean():.2f}, success rate={success.mean()*100:.1f}%")
        return final_d, success

    rand_d, rand_s = stats(rand_res, "random")
    pd_d, pd_s = stats(pd_res, "PD heuristic")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, random_final_dist=rand_d, random_success=rand_s, pd_final_dist=pd_d, pd_success=pd_s)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
