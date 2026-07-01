"""Eval honnete du stationnement, par planner, avec la VRAIE metrique stricte.

Compare des planners sur des graines fixes et reporte position, vrai cap, taux
"garee entre les lignes" (dist<0.15 m ET |cap|<5 deg), et collisions.

Planners:
  model       : planner CEM latent du world model (le modele imagine et choisit)
  model_warm  : idem, warm-start par la politique PD (plan initial vers le but)
  sim_mpc     : MPC privilegie (planifie dans le vrai simulateur, optimise la vraie pose)
  pose        : controleur de pose privilegie (feedback)
  pd          : baseline heuristique

Exemple:
  python scripts/eval_parking.py --ckpt runs/lewm_fix_v1/ckpt_last.pt --planner model_warm --episodes 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.parking_env import make_parking, render_pixel, render_goal_pixel, goal_distance, HeuristicPDPolicy, PoseParkController
from src.parking_metrics import final_pose_metrics, user_strict_success
from src.lewm import LeWM, LeWMConfig
from scripts.live_viewer import to_t_batch, cem_plan_batch
from scripts.plan import cem_plan_sim
from src.parking_control import BiarcPursuit


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = LeWMConfig(**ck["cfg"])
    m = LeWM(cfg).to(device); m.load_state_dict(ck["model"]); m.eval()
    return m, int(ck.get("step", -1))


@torch.no_grad()
def episode(planner, env, model, device, horizon, mpc_apply, max_steps, warm, rng):
    pd = HeuristicPDPolicy(noise=0.0)
    pose = PoseParkController()
    biarc = BiarcPursuit(env) if planner == "biarc" else None
    step = 0
    while step < max_steps:
        d, _ = goal_distance(env)
        if planner == "model_finalize" and d < 1.6:
            # finition privilegiee pres du but: cap fort pour se redresser entre les lignes
            plan = cem_plan_sim(env, horizon=10, pop_size=28, elites=6, iters=3,
                                seed=int(rng.integers(1 << 30)), w_pos=0.4, w_lat=1.0, w_along=0.5,
                                w_head=0.2, w_speed=0.25, w_coll=30.0)
        elif planner in ("model", "model_warm", "model_finalize"):
            cur = render_pixel(env); goal = render_goal_pixel(env)
            zc = model.encode(to_t_batch(cur[None], device)); zg = model.encode(to_t_batch(goal[None], device))
            if zc.dim() == 3:
                zc = zc.reshape(1, -1); zg = zg.reshape(1, -1)
            w = np.repeat(pd(env)[None, :], horizon, axis=0)[None] if planner in ("model_warm", "model_finalize") else None
            plan = cem_plan_batch(model, zc, zg, horizon=horizon, pop=256, elites=32, iters=6, warm_start=w)[0]
        elif planner == "sim_mpc":
            plan = cem_plan_sim(env, horizon=horizon, pop_size=32, elites=6, iters=3,
                                seed=int(rng.integers(1 << 30)), w_pos=0.6, w_lat=0.5, w_along=0.2,
                                w_axis=0.4, w_speed=0.25, w_coll=30.0)
        elif planner == "pose":
            plan = np.repeat(pose(env)[None, :], mpc_apply, axis=0)
        elif planner == "pd":
            plan = np.repeat(pd(env)[None, :], mpc_apply, axis=0)
        elif planner == "biarc":
            plan = np.asarray(biarc.act(), dtype=np.float32)[None, :]
        else:
            raise ValueError(planner)
        for k in range(min(mpc_apply, len(plan), max_steps - step)):
            env.step(np.asarray(plan[k], dtype=np.float32))
            step += 1
            if bool(getattr(env.unwrapped.controlled_vehicles[0], "crashed", False)):
                return final_pose_metrics(env)
    return final_pose_metrics(env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs" / "lewm_fix_v1" / "ckpt_last.pt"))
    ap.add_argument("--planner", default="model_warm",
                    choices=["model", "model_warm", "model_finalize", "sim_mpc", "pose", "pd", "biarc"])
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--variant", default="standard")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--mpc-apply", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=150)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = None
    if args.planner.startswith("model"):
        model, step = load_model(args.ckpt, device)
        print(f"model ckpt step {step}")
    rng = np.random.default_rng(args.seed)

    dists, heads, lats, strict, coll = [], [], [], [], []
    for ep in range(args.episodes):
        seed = args.seed + ep
        env = make_parking(seed=seed, img_size=64, env_variant=args.variant)
        m = episode(args.planner, env, model, device, args.horizon, args.mpc_apply, args.max_steps,
                    args.planner == "model_warm", rng)
        s = user_strict_success(m)
        dists.append(float(m["dist"])); heads.append(float(m["abs_heading_error_deg"]))
        lats.append(float(m["abs_lateral_offset_m"])); strict.append(s); coll.append(bool(m["collided"]))
        print(f"  seed={seed} dist={m['dist']:.3f}m cap={m['abs_heading_error_deg']:5.1f}deg lat={m['abs_lateral_offset_m']:.2f} coll={bool(m['collided'])} PARKED={s}")
        env.close()

    d = np.array(dists); h = np.array(heads)
    print(f"\n[{args.planner}] variant={args.variant} n={args.episodes}")
    print(f"  position: mean={d.mean():.3f}m median={np.median(d):.3f}m  <0.15m: {int((d<0.15).sum())}/{len(d)}")
    print(f"  cap: mean={h.mean():.1f}deg median={np.median(h):.1f}deg  <5deg: {int((h<5).sum())}/{len(h)}")
    print(f"  collisions: {int(np.sum(coll))}/{len(coll)}   GARE ENTRE LES LIGNES (strict): {int(np.sum(strict))}/{len(strict)} ({100*np.mean(strict):.0f}%)")


if __name__ == "__main__":
    main()
