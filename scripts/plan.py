"""Planning latent CEM avec LeWM sur parking-v0.

Charge un checkpoint, encode (obs, goal) en latent, optimise une sequence d'actions
pour minimiser ||z_hat_H - z_goal||^2 via CEM, applique le plan en MPC.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.lewm import LeWM, LeWMConfig
from src.parking_env import make_parking, render_pixel, render_goal_pixel, goal_distance, HeuristicPDPolicy
from src.parking_metrics import final_pose_metrics, parked_success


DEFAULT_ACTION_SMOOTHING = 0.25
DEFAULT_ACTION_L2 = 0.01
DEFAULT_SMOOTH_L2 = 0.05
DEFAULT_TRAJECTORY_WEIGHT = 0.0
DEFAULT_POSE_COST_WEIGHT = 0.0
DEFAULT_LATERAL_WEIGHT = 1.0
DEFAULT_ALONG_WEIGHT = 0.5
DEFAULT_HEADING_WEIGHT = 0.25
DEFAULT_COLLISION_WEIGHT = 50.0
DEFAULT_POSE_RERANK_COUNT = 8


def load_model(ckpt_path: str, device: torch.device) -> tuple[LeWM, LeWMConfig]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = LeWMConfig(**ckpt["cfg"])
    model = LeWM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(device)


def smooth_action_sequences(actions: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Lisse les actions dans le temps sans changer la premiere commande."""
    if smoothing <= 0 or actions.shape[1] <= 1:
        return actions
    smoothing = float(np.clip(smoothing, 0.0, 0.95))
    out = actions.clone()
    for t in range(1, actions.shape[1]):
        out[:, t] = smoothing * out[:, t - 1] + (1.0 - smoothing) * actions[:, t]
    return out.clamp(-1.0, 1.0)


def smooth_action_sequence(actions: np.ndarray, smoothing: float) -> np.ndarray:
    if smoothing <= 0 or len(actions) <= 1:
        return actions.astype(np.float32, copy=True)
    smoothing = float(np.clip(smoothing, 0.0, 0.95))
    out = actions.astype(np.float32, copy=True)
    for t in range(1, len(out)):
        out[t] = smoothing * out[t - 1] + (1.0 - smoothing) * actions[t]
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def latent_plan_cost(
    model: LeWM,
    z_rollout: torch.Tensor,
    z_goal: torch.Tensor,
    trajectory_weight: float = 0.0,
) -> torch.Tensor:
    """Cout latent: dernier latent predit vers goal, avec option trajectoire."""
    cost = model.latent_goal_cost(z_rollout, z_goal)
    if trajectory_weight <= 0:
        return cost
    if z_goal.dim() == 1:
        z_goal = z_goal.unsqueeze(0)
    goal = z_goal[:, None, :].expand(z_rollout.shape[0], z_rollout.shape[1], z_rollout.shape[2])
    trajectory = (z_rollout - goal.detach()).square().sum(dim=2).mean(dim=1)
    return cost + trajectory_weight * trajectory


def snapshot_env_state(env) -> tuple[dict, dict]:
    state = env.unwrapped.__dict__
    try:
        return copy.deepcopy(state), {}
    except Exception:
        pass

    saved = {}
    preserved = {}
    for key, value in state.items():
        try:
            saved[key] = copy.deepcopy(value)
        except Exception:
            preserved[key] = value
    return saved, preserved


def restore_env_state(env, snapshot: tuple[dict, dict]):
    saved, preserved = snapshot
    state = env.unwrapped.__dict__
    for key in list(state.keys()):
        if key not in saved and key not in preserved:
            del state[key]
    state.update(saved)
    for key, value in preserved.items():
        state[key] = value
    action_type = state.get("action_type")
    if action_type is not None and hasattr(action_type, "env"):
        action_type.env = env.unwrapped
    road = state.get("road")
    controlled = state.get("controlled_vehicles")
    if road is not None and controlled and hasattr(road, "vehicles"):
        for i, vehicle in enumerate(controlled):
            if vehicle in road.vehicles:
                continue
            if road.vehicles:
                distances = [
                    float(np.linalg.norm(getattr(candidate, "position", np.zeros(2)) - vehicle.position))
                    for candidate in road.vehicles
                ]
                road.vehicles[int(np.argmin(distances))] = vehicle
        state["vehicle"] = controlled[0]


def set_auto_render(env, enabled: bool):
    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "enable_auto_render"):
        return None
    previous = unwrapped.enable_auto_render
    unwrapped.enable_auto_render = enabled
    return previous


def _vehicle_xy(env) -> np.ndarray:
    vehicle = env.unwrapped.controlled_vehicles[0]
    return np.asarray(vehicle.position, dtype=np.float32).copy()


def simulate_action_trajectory(env, actions: np.ndarray) -> np.ndarray:
    """Simule des actions et retourne les positions monde, sans modifier l'env courant."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"actions must have shape (N, 2), got {actions.shape}")
    if len(actions) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    saved_state = snapshot_env_state(env)
    previous_auto_render = set_auto_render(env, False)
    positions = []
    try:
        last_pos = _vehicle_xy(env)
        for action in actions:
            _, _, term, trunc, _ = env.step(action)
            last_pos = _vehicle_xy(env)
            positions.append(last_pos)
            if term or trunc:
                break
        while len(positions) < len(actions):
            positions.append(last_pos.copy())
    finally:
        if previous_auto_render is not None:
            set_auto_render(env, previous_auto_render)
        restore_env_state(env, saved_state)
    return np.stack(positions).astype(np.float32)


def pose_plan_cost(
    env,
    actions: np.ndarray,
    lateral_weight: float = DEFAULT_LATERAL_WEIGHT,
    along_weight: float = DEFAULT_ALONG_WEIGHT,
    heading_weight: float = DEFAULT_HEADING_WEIGHT,
    collision_weight: float = DEFAULT_COLLISION_WEIGHT,
) -> float:
    """Score strict d'un plan via simulation env restauree.

    Ce cout est volontairement reserve au rerank de quelques candidats: faire
    cette simulation pour toute la population CEM serait trop cher.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"actions must have shape (N, 2), got {actions.shape}")

    saved_state = snapshot_env_state(env)
    previous_auto_render = set_auto_render(env, False)
    try:
        for action in actions:
            _, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        metrics = final_pose_metrics(env)
        lateral = float(metrics.get("abs_lateral_offset_m", abs(float(metrics["lateral_offset_m"]))))
        along = float(metrics.get("abs_along_offset_m", abs(float(metrics["along_offset_m"]))))
        heading = float(metrics.get("angle_deg", metrics.get("axis_error_deg", 0.0))) / 90.0
        collision = 1.0 if bool(metrics.get("collided", False)) else 0.0
        return float(
            lateral_weight * lateral
            + along_weight * along
            + heading_weight * heading
            + collision_weight * collision
        )
    finally:
        if previous_auto_render is not None:
            set_auto_render(env, previous_auto_render)
        restore_env_state(env, saved_state)


@torch.no_grad()
def rerank_action_candidates(
    model: LeWM,
    env,
    z_init: torch.Tensor,
    z_goal: torch.Tensor,
    candidates: np.ndarray,
    action_l2: float = DEFAULT_ACTION_L2,
    smooth_l2: float = DEFAULT_SMOOTH_L2,
    trajectory_weight: float = DEFAULT_TRAJECTORY_WEIGHT,
    pose_cost_weight: float = DEFAULT_POSE_COST_WEIGHT,
    lateral_weight: float = DEFAULT_LATERAL_WEIGHT,
    along_weight: float = DEFAULT_ALONG_WEIGHT,
    heading_weight: float = DEFAULT_HEADING_WEIGHT,
    collision_weight: float = DEFAULT_COLLISION_WEIGHT,
) -> np.ndarray:
    """Rerank top CEM candidates avec cout latent + cout parking strict."""
    candidates = np.asarray(candidates, dtype=np.float32)
    if candidates.ndim != 3 or candidates.shape[2] != 2:
        raise ValueError(f"candidates must have shape (K, H, 2), got {candidates.shape}")
    if len(candidates) == 0:
        raise ValueError("candidates must not be empty")
    if pose_cost_weight <= 0:
        return candidates[0].copy()

    device = z_init.device
    actions = torch.from_numpy(candidates).to(device)
    z_batch = z_init.expand(actions.shape[0], -1)
    z_rollout = model.rollout_latents(z_batch, actions)
    latent_cost = latent_plan_cost(model, z_rollout, z_goal, trajectory_weight=trajectory_weight)
    if action_l2 > 0:
        latent_cost = latent_cost + action_l2 * actions.square().mean(dim=(1, 2))
    if smooth_l2 > 0 and actions.shape[1] > 1:
        latent_cost = latent_cost + smooth_l2 * (actions[:, 1:] - actions[:, :-1]).square().mean(dim=(1, 2))

    scores = latent_cost.detach().cpu().numpy().astype(np.float64)
    pose_scores = np.array(
        [
            pose_plan_cost(
                env,
                candidate,
                lateral_weight=lateral_weight,
                along_weight=along_weight,
                heading_weight=heading_weight,
                collision_weight=collision_weight,
            )
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    best = int(np.argmin(scores + pose_cost_weight * pose_scores))
    return candidates[best].copy()


@torch.no_grad()
def cem_plan(
    model: LeWM,
    z_init: torch.Tensor,
    z_goal: torch.Tensor,
    horizon: int,
    pop_size: int = 200,
    elites: int = 20,
    iters: int = 8,
    sigma_init: float = 1.0,
    action_dim: int = 2,
    warm_start: np.ndarray | None = None,
    action_l2: float = DEFAULT_ACTION_L2,
    smooth_l2: float = DEFAULT_SMOOTH_L2,
    action_smoothing: float = DEFAULT_ACTION_SMOOTHING,
    trajectory_weight: float = DEFAULT_TRAJECTORY_WEIGHT,
    seed: int | None = None,
    return_candidates: bool = False,
    candidate_count: int = DEFAULT_POSE_RERANK_COUNT,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Retourne la meilleure sequence d'actions trouvee (horizon, action_dim).
    warm_start: (horizon, action_dim) numpy pour initialiser mu (sinon zeros).
    """
    device = z_init.device
    if seed is not None:
        torch.manual_seed(seed)
    warm_candidate = None
    if warm_start is not None:
        warm = smooth_action_sequence(warm_start, action_smoothing)
        warm_candidate = warm.astype(np.float32)
        mu = torch.from_numpy(warm_candidate).to(device)
        sigma = torch.full((horizon, action_dim), 0.35, device=device)
    else:
        mu = torch.zeros(horizon, action_dim, device=device)
        sigma = torch.full((horizon, action_dim), sigma_init, device=device)

    elite_count = min(max(1, elites), pop_size)
    best_actions = mu
    best_cost = torch.tensor(float("inf"), device=device)
    last_candidates = mu.unsqueeze(0)

    for _ in range(iters):
        actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(pop_size, horizon, action_dim, device=device)
        actions = actions.clamp(-1.0, 1.0)
        actions = smooth_action_sequences(actions, action_smoothing)

        z_rollout = model.rollout_latents(z_init.expand(pop_size, -1), actions)
        cost = latent_plan_cost(model, z_rollout, z_goal, trajectory_weight=trajectory_weight)
        if action_l2 > 0:
            cost = cost + action_l2 * actions.square().mean(dim=(1, 2))
        if smooth_l2 > 0 and horizon > 1:
            cost = cost + smooth_l2 * (actions[:, 1:] - actions[:, :-1]).square().mean(dim=(1, 2))

        current_best = cost.argmin()
        if cost[current_best] < best_cost:
            best_cost = cost[current_best]
            best_actions = actions[current_best]

        topk = cost.topk(elite_count, largest=False).indices
        elite_actions = actions[topk]
        last_candidates = elite_actions[: max(1, min(candidate_count, elite_actions.shape[0]))]
        mu = elite_actions.mean(dim=0)
        sigma = elite_actions.std(dim=0, unbiased=False).clamp(min=0.05, max=1.0)

    best_np = best_actions.cpu().numpy()
    if not return_candidates:
        return best_np

    candidate_parts = [best_np[None]]
    if warm_candidate is not None:
        candidate_parts.append(warm_candidate[None])
    candidate_parts.append(last_candidates.cpu().numpy())
    candidates_np = np.concatenate(candidate_parts, axis=0).astype(np.float32, copy=False)
    return best_np, candidates_np



def pd_rollout_warm_start(env, horizon: int) -> np.ndarray:
    """Simule PD sur une copie profonde de l etat env, puis restaure."""
    saved_state = snapshot_env_state(env)
    previous_auto_render = set_auto_render(env, False)

    pd = HeuristicPDPolicy(noise=0.0)
    actions = []
    try:
        for _ in range(horizon):
            a = pd(env)
            actions.append(a.astype(np.float32))
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    finally:
        if previous_auto_render is not None:
            set_auto_render(env, previous_auto_render)
        restore_env_state(env, saved_state)

    arr = np.stack(actions) if actions else np.zeros((horizon, 2), dtype=np.float32)
    if arr.shape[0] < horizon:
        pad_src = arr[-1:] if len(arr) else np.zeros((1, 2), dtype=np.float32)
        pad = np.tile(pad_src, (horizon - arr.shape[0], 1))
        arr = np.concatenate([arr, pad], axis=0)
    return arr


def run_episode(
    model: LeWM,
    env,
    horizon: int = 5,
    mpc_apply: int = 5,
    max_env_steps: int = 60,
    pop_size: int = 200,
    elites: int = 20,
    iters: int = 8,
    device=None,
    warm_start_pd: bool = False,
    action_l2: float = DEFAULT_ACTION_L2,
    smooth_l2: float = DEFAULT_SMOOTH_L2,
    action_smoothing: float = DEFAULT_ACTION_SMOOTHING,
    trajectory_weight: float = DEFAULT_TRAJECTORY_WEIGHT,
    pose_cost_weight: float = DEFAULT_POSE_COST_WEIGHT,
    lateral_weight: float = DEFAULT_LATERAL_WEIGHT,
    along_weight: float = DEFAULT_ALONG_WEIGHT,
    heading_weight: float = DEFAULT_HEADING_WEIGHT,
    collision_weight: float = DEFAULT_COLLISION_WEIGHT,
    seed: int | None = None,
    return_planned_trajectories: bool = False,
) -> dict:
    obs, info = env.reset()
    # highway-env may try to auto-render during env.step even when no viewer exists.
    set_auto_render(env, False)
    goal_img = render_goal_pixel(env)
    goal_t = to_tensor(goal_img, device)
    z_goal = model.encode(goal_t)

    cur_dist, cur_ang = goal_distance(env)
    init_dist, init_ang = cur_dist, cur_ang

    frames = []
    actions_taken = []
    planned_trajectories = []
    plan_idx = 0
    for step in range(0, max_env_steps, mpc_apply):
        cur_img = render_pixel(env)
        cur_t = to_tensor(cur_img, device)
        z_cur = model.encode(cur_t)
        ws = pd_rollout_warm_start(env, horizon) if warm_start_pd else None
        plan_seed = None if seed is None else seed + plan_idx
        plan_result = cem_plan(
            model,
            z_cur,
            z_goal,
            horizon=horizon,
            pop_size=pop_size,
            elites=elites,
            iters=iters,
            warm_start=ws,
            action_l2=action_l2,
            smooth_l2=smooth_l2,
            action_smoothing=action_smoothing,
            trajectory_weight=trajectory_weight,
            seed=plan_seed,
            return_candidates=pose_cost_weight > 0,
            candidate_count=DEFAULT_POSE_RERANK_COUNT,
        )
        if pose_cost_weight > 0:
            _, candidates = plan_result
            plan = rerank_action_candidates(
                model,
                env,
                z_cur,
                z_goal,
                candidates,
                action_l2=action_l2,
                smooth_l2=smooth_l2,
                trajectory_weight=trajectory_weight,
                pose_cost_weight=pose_cost_weight,
                lateral_weight=lateral_weight,
                along_weight=along_weight,
                heading_weight=heading_weight,
                collision_weight=collision_weight,
            )
        else:
            plan = plan_result
        if return_planned_trajectories:
            planned_trajectories.append(simulate_action_trajectory(env, plan))
        plan_idx += 1

        for k in range(min(mpc_apply, max_env_steps - step)):
            a = plan[k]
            obs, r, term, trunc, info = env.step(a)
            actions_taken.append(a.copy())
            frames.append(render_pixel(env))
            if term or trunc:
                break
        if term or trunc:
            break

    final_dist, final_ang = goal_distance(env)
    pose_metrics = final_pose_metrics(env)
    result = {
        "init_dist": init_dist,
        "init_ang": init_ang,
        "final_dist": final_dist,
        "final_ang": final_ang,
        "pose_metrics": pose_metrics,
        "strict_success": parked_success(pose_metrics),
        "final_lateral": pose_metrics["abs_lateral_offset_m"],
        "final_along": pose_metrics["abs_along_offset_m"],
        "final_heading": pose_metrics["angle_deg"],
        "collided": pose_metrics["collided"],
        "n_steps": len(actions_taken),
        "frames": frames,
        "goal_img": goal_img,
        "actions": np.stack(actions_taken) if actions_taken else np.zeros((0, 2)),
    }
    if return_planned_trajectories:
        result["planned_trajectories"] = planned_trajectories
    return result


def random_policy(env):
    return np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)


def run_policy_episode(env, policy, max_env_steps: int = 60) -> dict:
    obs, info = env.reset()
    init_dist, init_ang = goal_distance(env)
    frames = []
    actions_taken = []
    goal_img = render_goal_pixel(env)
    for step in range(max_env_steps):
        action = policy(env).astype(np.float32)
        obs, r, term, trunc, info = env.step(action)
        actions_taken.append(action.copy())
        frames.append(render_pixel(env))
        if term or trunc:
            break
    final_dist, final_ang = goal_distance(env)
    pose_metrics = final_pose_metrics(env)
    return {
        "init_dist": init_dist,
        "init_ang": init_ang,
        "final_dist": final_dist,
        "final_ang": final_ang,
        "pose_metrics": pose_metrics,
        "strict_success": parked_success(pose_metrics),
        "final_lateral": pose_metrics["abs_lateral_offset_m"],
        "final_along": pose_metrics["abs_along_offset_m"],
        "final_heading": pose_metrics["angle_deg"],
        "collided": pose_metrics["collided"],
        "n_steps": len(actions_taken),
        "frames": frames,
        "goal_img": goal_img,
        "actions": np.stack(actions_taken) if actions_taken else np.zeros((0, 2)),
    }


def _sim_pose_cost(env, actions, *, w_pos=1.0, w_lat=0.0, w_along=0.0,
                   w_head=0.03, w_axis=0.0, w_speed=0.2, w_coll=25.0) -> float:
    """Simule un plan dans l'env restaure et score la VRAIE pose finale (vrai cap).

    w_lat/w_along scorent dans le repere du but: un w_lat fort force la voiture
    sur la ligne centrale de la place, ce qui impose une approche dans l'axe donc
    un cap aligne en arrivant.
    """
    saved = snapshot_env_state(env)
    prev = set_auto_render(env, False)
    try:
        for a in actions:
            _, _, term, trunc, _ = env.step(np.asarray(a, dtype=np.float32))
            if term or trunc:
                break
        m = final_pose_metrics(env)
        return (
            w_pos * float(m["dist"])
            + w_lat * float(m["abs_lateral_offset_m"])
            + w_along * float(m["abs_along_offset_m"])
            + w_head * float(m["abs_heading_error_deg"])
            + w_axis * float(m["angle_deg"])
            + w_speed * abs(float(m["speed_mps"]))
            + (w_coll if bool(m["collided"]) else 0.0)
        )
    finally:
        if prev is not None:
            set_auto_render(env, prev)
        restore_env_state(env, saved)


def cem_plan_sim(env, horizon=8, pop_size=32, elites=6, iters=3, sigma_init=0.6, seed=None,
                 **weights) -> np.ndarray:
    """MPC privilegie: CEM sur les actions, scorees par simulation du VRAI env (pose vraie).

    Ne fait intervenir aucun modele appris. Robuste par construction (le simulateur dit
    la verite), au prix d'un cout de calcul (~pop*iters*horizon pas d'env par plan).
    """
    rng = np.random.default_rng(seed)
    mu = np.zeros((horizon, 2), dtype=np.float32)
    sigma = np.full((horizon, 2), sigma_init, dtype=np.float32)
    best_a, best_c = None, float("inf")
    for _ in range(iters):
        cand = mu[None] + sigma[None] * rng.standard_normal((pop_size, horizon, 2)).astype(np.float32)
        cand = np.clip(cand, -1.0, 1.0)
        cand[0] = np.clip(mu, -1.0, 1.0)
        costs = np.array([_sim_pose_cost(env, c, **weights) for c in cand])
        order = np.argsort(costs)
        elite = cand[order[:elites]]
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 0.05
        if costs[order[0]] < best_c:
            best_c = float(costs[order[0]])
            best_a = cand[order[0]].copy()
    return best_a


def run_sim_mpc_episode(env, horizon=8, mpc_apply=3, max_env_steps=100, pop_size=32,
                        elites=6, iters=3, seed=None, **weights) -> dict:
    obs, info = env.reset()
    init_dist, init_ang = goal_distance(env)
    frames, actions_taken, planned = [], [], []
    goal_img = render_goal_pixel(env)
    step = 0
    done = False
    while step < max_env_steps and not done:
        plan = cem_plan_sim(
            env, horizon=horizon, pop_size=pop_size, elites=elites, iters=iters,
            seed=None if seed is None else seed + step, **weights,
        )
        planned.append(simulate_action_trajectory(env, plan))
        for k in range(min(mpc_apply, max_env_steps - step)):
            a = plan[k].astype(np.float32)
            obs, r, term, trunc, info = env.step(a)
            actions_taken.append(a.copy())
            frames.append(render_pixel(env))
            step += 1
            if term or trunc:
                done = True
                break
    final_dist, final_ang = goal_distance(env)
    pose_metrics = final_pose_metrics(env)
    return {
        "init_dist": init_dist,
        "init_ang": init_ang,
        "final_dist": final_dist,
        "final_ang": final_ang,
        "pose_metrics": pose_metrics,
        "strict_success": parked_success(pose_metrics),
        "final_lateral": pose_metrics["abs_lateral_offset_m"],
        "final_along": pose_metrics["abs_along_offset_m"],
        "final_heading": pose_metrics["angle_deg"],
        "collided": pose_metrics["collided"],
        "n_steps": len(actions_taken),
        "frames": frames,
        "goal_img": goal_img,
        "actions": np.stack(actions_taken) if actions_taken else np.zeros((0, 2)),
        "planned_trajectories": planned,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/last/ckpt_last.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--pop", type=int, default=200)
    parser.add_argument("--elites", type=int, default=20)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--mpc-apply", type=int, default=5)
    parser.add_argument("--max-env-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--warm-start-pd", action="store_true", help="initialise CEM autour de PD (au lieu de zeros)")
    parser.add_argument("--action-l2", type=float, default=DEFAULT_ACTION_L2)
    parser.add_argument("--smooth-l2", type=float, default=DEFAULT_SMOOTH_L2)
    parser.add_argument("--action-smoothing", type=float, default=DEFAULT_ACTION_SMOOTHING)
    parser.add_argument("--trajectory-weight", type=float, default=DEFAULT_TRAJECTORY_WEIGHT)
    parser.add_argument("--pose-cost-weight", type=float, default=DEFAULT_POSE_COST_WEIGHT, help="rerank top CEM candidates with strict simulated parking cost")
    parser.add_argument("--lateral-weight", type=float, default=DEFAULT_LATERAL_WEIGHT)
    parser.add_argument("--along-weight", type=float, default=DEFAULT_ALONG_WEIGHT)
    parser.add_argument("--heading-weight", type=float, default=DEFAULT_HEADING_WEIGHT)
    parser.add_argument("--collision-weight", type=float, default=DEFAULT_COLLISION_WEIGHT)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, cfg = load_model(args.ckpt, device)
    print(f"Loaded {args.ckpt} on {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = []
    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        np.random.seed(ep_seed)
        torch.manual_seed(ep_seed)
        env = make_parking(seed=args.seed + ep, img_size=64)
        out = run_episode(
            model, env,
            horizon=args.horizon,
            mpc_apply=args.mpc_apply,
            max_env_steps=args.max_env_steps,
            pop_size=args.pop, elites=args.elites, iters=args.iters,
            device=device,
            warm_start_pd=args.warm_start_pd,
            action_l2=args.action_l2,
            smooth_l2=args.smooth_l2,
            action_smoothing=args.action_smoothing,
            trajectory_weight=args.trajectory_weight,
            pose_cost_weight=args.pose_cost_weight,
            lateral_weight=args.lateral_weight,
            along_weight=args.along_weight,
            heading_weight=args.heading_weight,
            collision_weight=args.collision_weight,
            seed=ep_seed,
        )
        env.close()
        success = bool(out["strict_success"])
        out["success"] = success
        results.append(out)
        print(
            f"ep {ep}: init=({out['init_dist']:.2f}m, {out['init_ang']:.1f}°)"
            f" -> final=({out['final_dist']:.2f}m, {out['final_ang']:.1f}°)"
            f" lat={out['final_lateral']:.2f}m along={out['final_along']:.2f}m"
            f" heading={out['final_heading']:.1f}° collision={out['collided']}"
            f" strict_success={success}  ({out['n_steps']} steps)"
        )

    n_succ = sum(1 for r in results if r["success"])
    print(f"\nSuccess rate: {n_succ}/{len(results)} = {100 * n_succ / len(results):.1f}%")

    out_dir = Path(args.ckpt).parent / "plan_results"
    out_dir.mkdir(exist_ok=True, parents=True)
    np.savez(
        out_dir / "results.npz",
        init_dist=np.array([r["init_dist"] for r in results]),
        final_dist=np.array([r["final_dist"] for r in results]),
        init_ang=np.array([r["init_ang"] for r in results]),
        final_ang=np.array([r["final_ang"] for r in results]),
        final_lateral=np.array([r["final_lateral"] for r in results]),
        final_along=np.array([r["final_along"] for r in results]),
        final_heading=np.array([r["final_heading"] for r in results]),
        collided=np.array([r["collided"] for r in results]),
        success=np.array([r["success"] for r in results]),
    )
    print(f"saved {out_dir / 'results.npz'}")

    import imageio.v2 as imageio
    gif_dir = out_dir / "gifs"
    gif_dir.mkdir(exist_ok=True)
    # Sauve les 5 meilleurs episodes (plus petite distance finale au goal)
    indexed = list(enumerate(results))
    best = sorted(indexed, key=lambda kv: kv[1]["final_dist"])[:5]
    for rank, (i, r) in enumerate(best):
        if not r["frames"]:
            continue
        composed = []
        for f in r["frames"]:
            side = np.concatenate([f, np.full((f.shape[0], 4, 3), 255, dtype=np.uint8), r["goal_img"]], axis=1)
            composed.append(side)
        tag = "ok" if r["success"] else f"d{r['final_dist']:.1f}m"
        imageio.mimsave(gif_dir / f"top{rank:02d}_ep{i:02d}_{tag}.gif", composed, fps=10)
    print(f"saved top-5 GIFs (by final dist) in {gif_dir}")


if __name__ == "__main__":
    main()
