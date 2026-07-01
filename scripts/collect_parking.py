"""Collecte de trajectoires parking-v0 multi-processus pour saturer le CPU.

Dataset v2: episodes equilibres, metadata par episode, split train/val fixe.
Dataset v3: variantes d'environnement/initialisation et metadata enrichie.
"""

from __future__ import annotations

import argparse
import h5py
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.parking_metrics import final_pose_metrics, parked_success


DEFAULT_POLICY_MIX = "random,pd,pd_noisy,expert,expert_noisy,reverse,reverse_noisy,near_goal_correction,near_goal_noisy,final_alignment,final_alignment_noisy"
DEFAULT_POLICY_MIX_V3 = "pd,pd_noisy,expert,expert_noisy,reverse,reverse_noisy,near_goal_correction,near_goal_noisy,final_alignment,final_alignment_noisy"
DEFAULT_ENV_VARIANTS_V3 = "standard,wide_start,long_approach,near_goal,short_correction,reverse_entry,final_alignment,slot_offset,static_vehicles,crowded_static"
POLICY_ALIASES = {
    "reverse_expert": "reverse",
    "near_goal_expert": "near_goal_correction",
    "near_goal_pd": "near_goal_correction",
    "near_goal_reverse": "reverse_noisy",
}
KNOWN_POLICIES = set(DEFAULT_POLICY_MIX.split(",")) | set(POLICY_ALIASES)
ENV_VARIANT_ALIASES = {
    "default": "standard",
    "parking": "standard",
    "reverse": "reverse_entry",
    "align": "final_alignment",
    "final_align": "final_alignment",
    "static": "static_vehicles",
    "crowded": "crowded_static",
}
KNOWN_ENV_VARIANTS = {
    "standard",
    "wide_start",
    "long_approach",
    "near_goal",
    "short_correction",
    "reverse_entry",
    "final_alignment",
    "slot_offset",
    "static_vehicles",
    "crowded_static",
    "no_walls",
}


def random_action(env=None) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)


def parse_policy_mix(text: str) -> list[str]:
    policies = [p.strip() for p in text.split(",") if p.strip()]
    if not policies:
        raise ValueError("policy mix cannot be empty")
    unknown = sorted(set(policies) - KNOWN_POLICIES)
    if unknown:
        raise ValueError(f"unknown policies: {', '.join(unknown)}")
    return [POLICY_ALIASES.get(p, p) for p in policies]


def parse_env_variants(text: str) -> list[str]:
    variants = [v.strip() for v in text.split(",") if v.strip()]
    if not variants:
        raise ValueError("env variants cannot be empty")
    normalized = [ENV_VARIANT_ALIASES.get(v, v) for v in variants]
    unknown = sorted(set(normalized) - KNOWN_ENV_VARIANTS)
    if unknown:
        raise ValueError(f"unknown env variants: {', '.join(unknown)}")
    return normalized


def choose_policy_name(i: int, policies: list[str]) -> str:
    return policies[i % len(policies)]


def choose_env_variant(i: int, variants: list[str]) -> str:
    return variants[i % len(variants)]


def place_near_goal(
    env,
    rng: np.random.Generator,
    *,
    radius_min: float = 0.9,
    radius_max: float = 3.0,
    heading_noise: float = 1.0,
):
    from src.parking_env import _get_goal

    unwrapped = env.unwrapped
    vehicle = unwrapped.controlled_vehicles[0]
    goal = _get_goal(env)
    angle = float(rng.uniform(-np.pi, np.pi))
    radius = float(rng.uniform(radius_min, radius_max))
    offset = radius * np.array([np.cos(angle), np.sin(angle)])
    vehicle.position = goal.position.copy() + offset
    vehicle.heading = float(goal.heading + rng.uniform(-heading_noise, heading_noise))
    vehicle.speed = float(rng.uniform(-0.5, 0.5))


def place_wide_start(env, rng: np.random.Generator):
    from src.parking_env import _get_goal

    unwrapped = env.unwrapped
    vehicle = unwrapped.controlled_vehicles[0]
    goal = _get_goal(env)
    angle = float(rng.uniform(-np.pi, np.pi))
    radius = float(rng.uniform(5.0, 11.0))
    vehicle.position = goal.position.copy() + radius * np.array([np.cos(angle), np.sin(angle)])
    target_heading = float(np.arctan2(goal.position[1] - vehicle.position[1], goal.position[0] - vehicle.position[0]))
    vehicle.heading = target_heading + float(rng.uniform(-1.0, 1.0))
    vehicle.speed = float(rng.uniform(-0.4, 0.8))


def place_long_approach(env, rng: np.random.Generator):
    from src.parking_env import _get_goal, _goal_axes

    unwrapped = env.unwrapped
    vehicle = unwrapped.controlled_vehicles[0]
    goal = _get_goal(env)
    forward, left = _goal_axes(goal)
    along = float(rng.choice([-1.0, 1.0]) * rng.uniform(6.0, 11.0))
    lateral = float(rng.uniform(-3.0, 3.0))
    vehicle.position = goal.position.copy() + along * forward + lateral * left
    target_heading = float(np.arctan2(goal.position[1] - vehicle.position[1], goal.position[0] - vehicle.position[0]))
    vehicle.heading = target_heading + float(rng.uniform(-0.7, 0.7))
    vehicle.speed = float(rng.uniform(-0.2, 0.8))


def place_slot_offset(env, rng: np.random.Generator):
    from src.parking_env import _get_goal, _goal_axes

    unwrapped = env.unwrapped
    vehicle = unwrapped.controlled_vehicles[0]
    goal = _get_goal(env)
    forward, left = _goal_axes(goal)
    along = float(rng.uniform(-1.8, 2.2))
    lateral = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.9, 2.5))
    vehicle.position = goal.position.copy() + along * forward + lateral * left
    vehicle.heading = float(goal.heading + rng.uniform(-0.65, 0.65))
    vehicle.speed = float(rng.uniform(-0.2, 0.3))


def place_reverse_setup(env, rng: np.random.Generator):
    from src.parking_env import _get_goal, _goal_axes

    unwrapped = env.unwrapped
    vehicle = unwrapped.controlled_vehicles[0]
    goal = _get_goal(env)
    forward, left = _goal_axes(goal)
    along = rng.uniform(3.0, 6.0)
    lateral = rng.uniform(-2.0, 2.0)
    vehicle.position = goal.position.copy() + along * forward + lateral * left
    vehicle.heading = float(goal.heading + rng.uniform(-0.45, 0.45))
    vehicle.speed = float(rng.uniform(-0.2, 0.2))


def place_final_alignment(env, rng: np.random.Generator):
    from src.parking_env import _get_goal, _goal_axes

    unwrapped = env.unwrapped
    vehicle = unwrapped.controlled_vehicles[0]
    goal = _get_goal(env)
    forward, left = _goal_axes(goal)
    along = rng.uniform(-1.0, 1.0)
    lateral = rng.uniform(-1.1, 1.1)
    heading_error = rng.choice([-1.0, 1.0]) * rng.uniform(0.45, 1.35)
    vehicle.position = goal.position.copy() + along * forward + lateral * left
    vehicle.heading = float(goal.heading + heading_error)
    vehicle.speed = float(rng.uniform(-0.15, 0.15))


def prepare_initial_state(env, policy_name: str, rng: np.random.Generator):
    if policy_name in {"near_goal_correction", "near_goal_noisy"}:
        place_near_goal(env, rng, radius_min=1.0, radius_max=3.5, heading_noise=1.25)
    elif policy_name in {"final_alignment", "final_alignment_noisy"}:
        place_final_alignment(env, rng)
    elif policy_name in {"reverse", "reverse_noisy"}:
        place_reverse_setup(env, rng)


def prepare_env_variant(env, env_variant: str, rng: np.random.Generator):
    if env_variant == "standard":
        return
    if env_variant == "wide_start":
        place_wide_start(env, rng)
    elif env_variant == "long_approach":
        place_long_approach(env, rng)
    elif env_variant == "near_goal":
        place_near_goal(env, rng, radius_min=1.0, radius_max=3.5, heading_noise=1.25)
    elif env_variant == "short_correction":
        place_near_goal(env, rng, radius_min=0.35, radius_max=1.4, heading_noise=0.75)
    elif env_variant == "reverse_entry":
        place_reverse_setup(env, rng)
    elif env_variant == "final_alignment":
        place_final_alignment(env, rng)
    elif env_variant == "slot_offset":
        place_slot_offset(env, rng)
    elif env_variant in {"static_vehicles", "crowded_static", "no_walls"}:
        return
    else:
        raise ValueError(f"unknown env variant: {env_variant}")


def make_policy(policy_name: str):
    from src.parking_env import HeuristicPDPolicy, ParkingExpertPolicy

    if policy_name == "random":
        return lambda env: random_action()
    if policy_name == "pd":
        return HeuristicPDPolicy(noise=0.0)
    if policy_name == "pd_noisy":
        return HeuristicPDPolicy(noise=0.35)
    if policy_name == "expert":
        return ParkingExpertPolicy(noise=0.0)
    if policy_name == "expert_noisy":
        return ParkingExpertPolicy(noise=0.18)
    if policy_name == "reverse":
        return ParkingExpertPolicy(force_reverse=True, noise=0.03)
    if policy_name == "reverse_noisy":
        return ParkingExpertPolicy(force_reverse=True, noise=0.20)
    if policy_name == "near_goal_correction":
        return ParkingExpertPolicy(speed_cap=1.2, align_dist=4.0, noise=0.04)
    if policy_name == "near_goal_noisy":
        return ParkingExpertPolicy(speed_cap=1.2, align_dist=4.0, noise=0.24)
    if policy_name == "final_alignment":
        return ParkingExpertPolicy(speed_cap=0.9, align_dist=4.5, final_alignment=True, noise=0.0)
    if policy_name == "final_alignment_noisy":
        return ParkingExpertPolicy(speed_cap=0.9, align_dist=4.5, final_alignment=True, noise=0.18)
    raise ValueError(f"unknown policy: {policy_name}")


def summarize_episodes(episodes: list[dict], policy_order: list[str]) -> list[dict]:
    rows = []
    seen = set()
    names = list(policy_order) + sorted({e["policy_name"] for e in episodes} - set(policy_order))
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        eps = [e for e in episodes if e["policy_name"] == name]
        if not eps:
            continue
        init_dist = np.array([e["init_dist"] for e in eps], dtype=np.float32)
        final_dist = np.array([e["final_dist"] for e in eps], dtype=np.float32)
        final_ang = np.array([e["final_ang"] for e in eps], dtype=np.float32)
        success = np.array([e["success"] for e in eps], dtype=np.float32)
        strict_success = np.array([e.get("strict_success", 0) for e in eps], dtype=np.float32)
        abs_lateral = np.array([e.get("abs_lateral_offset_m", np.nan) for e in eps], dtype=np.float32)
        abs_along = np.array([e.get("abs_along_offset_m", np.nan) for e in eps], dtype=np.float32)
        abs_heading = np.array([e.get("abs_heading_error_deg", np.nan) for e in eps], dtype=np.float32)
        rows.append({
            "policy": name,
            "episodes": len(eps),
            "success_rate": float(success.mean()),
            "strict_success_rate": float(strict_success.mean()),
            "mean_init_dist": float(init_dist.mean()),
            "mean_final_dist": float(final_dist.mean()),
            "mean_final_ang": float(final_ang.mean()),
            "mean_abs_lateral_offset_m": float(np.nanmean(abs_lateral)),
            "mean_abs_along_offset_m": float(np.nanmean(abs_along)),
            "mean_abs_heading_error_deg": float(np.nanmean(abs_heading)),
            "near_1_5": float((final_dist <= 1.5).mean()),
            "near_3": float((final_dist <= 3.0).mean()),
            "near_6": float((final_dist <= 6.0).mean()),
        })
    return rows


def print_dataset_stats(episodes: list[dict], policies: list[str], splits: np.ndarray | None = None):
    rows = summarize_episodes(episodes, policies)
    print("Dataset stats by policy:")
    print("policy                    eps  succ strict  init_m  final_m  final_deg  lat_m along_m head_deg  <=1.5m   <=3m   <=6m")
    for row in rows:
        print(
            f"{row['policy']:<24} "
            f"{row['episodes']:>4d} "
            f"{100 * row['success_rate']:>5.1f}% "
            f"{100 * row['strict_success_rate']:>5.1f}% "
            f"{row['mean_init_dist']:>7.2f} "
            f"{row['mean_final_dist']:>8.2f} "
            f"{row['mean_final_ang']:>9.1f} "
            f"{row['mean_abs_lateral_offset_m']:>6.2f} "
            f"{row['mean_abs_along_offset_m']:>7.2f} "
            f"{row['mean_abs_heading_error_deg']:>8.1f} "
            f"{100 * row['near_1_5']:>6.1f}% "
            f"{100 * row['near_3']:>6.1f}% "
            f"{100 * row['near_6']:>6.1f}%"
        )
    if splits is not None:
        train = int(np.sum(splits == "train"))
        val = int(np.sum(splits == "val"))
        print(f"splits: train={train} val={val}")
    variant_names = [str(e.get("env_variant", "")) for e in episodes if e.get("env_variant")]
    if variant_names:
        print("Dataset stats by env_variant:")
        print("env_variant              eps  init_m  final_m  final_deg strict  lat_m along_m  obst  static")
        for name in sorted(set(variant_names)):
            eps = [e for e in episodes if e.get("env_variant") == name]
            init_dist = np.array([e["init_dist"] for e in eps], dtype=np.float32)
            final_dist = np.array([e["final_dist"] for e in eps], dtype=np.float32)
            final_ang = np.array([e["final_ang"] for e in eps], dtype=np.float32)
            strict_success = np.array([e.get("strict_success", 0) for e in eps], dtype=np.float32)
            abs_lateral = np.array([e.get("abs_lateral_offset_m", np.nan) for e in eps], dtype=np.float32)
            abs_along = np.array([e.get("abs_along_offset_m", np.nan) for e in eps], dtype=np.float32)
            obstacle = np.array([e.get("obstacle_count", 0) for e in eps], dtype=np.float32)
            static = np.array([e.get("static_vehicle_count", 0) for e in eps], dtype=np.float32)
            print(
                f"{name:<24} "
                f"{len(eps):>4d} "
                f"{init_dist.mean():>7.2f} "
                f"{final_dist.mean():>8.2f} "
                f"{final_ang.mean():>9.1f} "
                f"{100 * strict_success.mean():>5.1f}% "
                f"{np.nanmean(abs_lateral):>6.2f} "
                f"{np.nanmean(abs_along):>7.2f} "
                f"{obstacle.mean():>5.1f} "
                f"{static.mean():>6.1f}"
            )


def check_dataset_readable(h5_path: str | Path, seq_len: int):
    from src.data import ParkingTrajectoryDataset

    train = ParkingTrajectoryDataset(h5_path, seq_len=seq_len, split="train")
    val = ParkingTrajectoryDataset(h5_path, seq_len=seq_len, split="val")
    if len(train) == 0 or len(val) == 0:
        raise RuntimeError(f"readability check failed: train={len(train)} val={len(val)}")
    train_sample = train[0]
    val_sample = val[0]
    print(
        "readable: "
        f"train_windows={len(train)} val_windows={len(val)} "
        f"obs={tuple(train_sample['obs'].shape)} actions={tuple(val_sample['actions'].shape)}"
    )


def strip_episode_payload(episode: dict) -> dict:
    return {k: v for k, v in episode.items() if k not in {"obs", "actions", "goal_obs"}}


def write_v3_episode_metadata(h5_path: str | Path, episodes: list[dict]):
    """Append v3-only metadata without changing the generic v1/v2 loader."""
    string_fields = ["env_variant"]
    numeric_fields = [
        "obstacle_count",
        "static_vehicle_count",
        "strict_success",
        "lateral_offset_m",
        "along_offset_m",
        "abs_lateral_offset_m",
        "abs_along_offset_m",
        "heading_error_deg",
        "abs_heading_error_deg",
        "speed_mps",
        "collided",
    ]
    with h5py.File(h5_path, "a") as f:
        f.attrs["format_version"] = "parking_v3"
        for key in string_fields:
            values = [str(ep.get(key, "")) for ep in episodes]
            if any(values):
                name = f"episode_{key}"
                if name in f:
                    del f[name]
                f.create_dataset(name, data=np.array(values, dtype=h5py.string_dtype(encoding="utf-8")))
        for key in numeric_fields:
            values = [ep.get(key) for ep in episodes]
            if any(v is not None for v in values):
                int_fields = {"obstacle_count", "static_vehicle_count", "strict_success", "collided"}
                fill = 0 if key in int_fields else np.nan
                arr = np.array(
                    [fill if v is None else v for v in values],
                    dtype=np.int64 if key in int_fields else np.float32,
                )
                name = f"episode_{key}"
                if name in f:
                    del f[name]
                f.create_dataset(name, data=arr)


def _collect_chunk(args):
    """Worker function: collecte n_episodes dans ce process et retourne la liste."""
    if len(args) == 7:
        worker_id, start_index, n_episodes, max_steps, img_size, seed_base, policies = args
        env_variants = []
    else:
        worker_id, start_index, n_episodes, max_steps, img_size, seed_base, policies, env_variants = args

    from src.parking_env import (
        make_parking,
        render_pixel,
        render_goal_pixel,
        goal_distance,
        obstacle_count,
        static_vehicle_count,
    )

    rng = np.random.default_rng(seed_base + worker_id)
    random.seed(seed_base + worker_id * 7919)
    np.random.seed(seed_base + worker_id * 7919)

    env = None
    current_env_variant = None

    out = []
    try:
        for i in range(n_episodes):
            episode_id = start_index + i
            episode_seed = int(rng.integers(0, 2**31 - 1))
            policy_name = choose_policy_name(episode_id, policies)
            env_variant = choose_env_variant(episode_id, env_variants) if env_variants else "default"
            make_variant = "standard" if env_variant == "default" else env_variant
            if env is None or current_env_variant != make_variant:
                if env is not None:
                    env.close()
                env = make_parking(seed=seed_base + worker_id, img_size=img_size, env_variant=make_variant)
                current_env_variant = make_variant
            env.reset(seed=episode_seed)
            if env_variants:
                prepare_env_variant(env, make_variant, rng)
            else:
                prepare_initial_state(env, policy_name, rng)
            init_dist, init_ang = goal_distance(env)
            goal_img = render_goal_pixel(env)
            obs_list = [render_pixel(env)]
            action_list = []
            policy_fn = make_policy(policy_name)
            for step in range(max_steps):
                a = policy_fn(env)
                obs, reward, term, trunc, info = env.step(a)
                action_list.append(a.astype(np.float32))
                obs_list.append(render_pixel(env))
                if term or trunc:
                    break
            final_dist, final_ang = goal_distance(env)
            pose_metrics = final_pose_metrics(env)
            episode = {
                "episode_id": episode_id,
                "obs": np.stack(obs_list, axis=0),
                "actions": np.stack(action_list, axis=0),
                "goal_obs": goal_img,
                "policy_name": policy_name,
                "seed": episode_seed,
                "init_dist": init_dist,
                "init_ang": init_ang,
                "final_dist": final_dist,
                "final_ang": final_ang,
                "success": int(final_dist < 1.5 and final_ang < 15.0),
                "strict_success": int(parked_success(pose_metrics)),
                "lateral_offset_m": float(pose_metrics["lateral_offset_m"]),
                "along_offset_m": float(pose_metrics["along_offset_m"]),
                "abs_lateral_offset_m": float(pose_metrics["abs_lateral_offset_m"]),
                "abs_along_offset_m": float(pose_metrics["abs_along_offset_m"]),
                "heading_error_deg": float(pose_metrics["heading_error_deg"]),
                "abs_heading_error_deg": float(pose_metrics["abs_heading_error_deg"]),
                "speed_mps": float(pose_metrics["speed_mps"]),
                "collided": int(bool(pose_metrics["collided"])),
                "episode_len": len(action_list),
            }
            if env_variants:
                episode.update({
                    "env_variant": make_variant,
                    "obstacle_count": obstacle_count(env),
                    "static_vehicle_count": static_vehicle_count(env),
                })
            out.append(episode)
    finally:
        if env is not None:
            env.close()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--out", type=str, default="data/parking/train_v2.h5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--policy-mix", type=str, default=None)
    parser.add_argument("--dataset-version", choices=["v2", "v3"], default="v2")
    parser.add_argument("--env-variants", type=str, default=None)
    parser.add_argument(
        "--stream-write",
        action="store_true",
        help="write worker chunks directly to HDF5 to avoid keeping every frame in RAM",
    )
    parser.add_argument(
        "--episodes-per-chunk",
        type=int,
        default=256,
        help="task size used by --stream-write; smaller chunks reduce peak RAM",
    )
    parser.add_argument("--check-readable", action="store_true")
    parser.add_argument("--check-seq-len", type=int, default=3)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.episodes_per_chunk <= 0:
        raise ValueError("--episodes-per-chunk must be positive")

    if args.dataset_version == "v3" and args.out == "data/parking/train_v2.h5":
        args.out = "data/parking/train_v3.h5"
    policy_mix = args.policy_mix or (DEFAULT_POLICY_MIX_V3 if args.dataset_version == "v3" else DEFAULT_POLICY_MIX)
    policies = parse_policy_mix(policy_mix)
    env_variants = []
    if args.dataset_version == "v3" or args.env_variants:
        env_variants = parse_env_variants(args.env_variants or DEFAULT_ENV_VARIANTS_V3)
    print(f"workers: {args.workers} (cpu_count={mp.cpu_count()})")
    print(f"policy mix: {', '.join(policies)}")
    if env_variants:
        print(f"env variants: {', '.join(env_variants)}")
    if args.stream_write:
        print(f"stream write: on (episodes_per_chunk={args.episodes_per_chunk})")

    chunk_size = args.episodes_per_chunk if args.stream_write else (args.episodes + args.workers - 1) // args.workers
    chunks = []
    remaining = args.episodes
    start_index = 0
    chunk_id = 0
    while remaining > 0:
        n = min(chunk_size, remaining)
        if n <= 0:
            break
        chunks.append((chunk_id, start_index, n, args.max_steps, args.img_size, args.seed, policies, env_variants))
        remaining -= n
        start_index += n
        chunk_id += 1

    t0 = time.time()
    ctx = mp.get_context("spawn")
    n_processes = min(args.workers, len(chunks))
    if args.stream_write:
        from src.data import EpisodeH5Writer

        out_path = Path(args.out)
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        if out_path.exists():
            out_path.unlink()

        all_episode_summaries = []
        split_chunks = []
        string_fields = ["policy_name"]
        from src.data import DEFAULT_NUMERIC_FIELDS

        numeric_fields = list(DEFAULT_NUMERIC_FIELDS)
        if args.dataset_version == "v3":
            string_fields.append("env_variant")
            numeric_fields.extend(["obstacle_count", "static_vehicle_count"])
        with EpisodeH5Writer(
            tmp_path,
            total_episodes=args.episodes,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
            format_version=f"parking_{args.dataset_version}",
            string_fields=string_fields,
            numeric_fields=numeric_fields,
        ) as writer, ctx.Pool(processes=n_processes) as pool:
            with tqdm(total=args.episodes, desc="collect") as pbar:
                for chunk_episodes in pool.imap_unordered(_collect_chunk, chunks):
                    chunk_episodes.sort(key=lambda e: e["episode_id"])
                    split_chunks.append(writer.append(chunk_episodes))
                    all_episode_summaries.extend(strip_episode_payload(e) for e in chunk_episodes)
                    pbar.update(len(chunk_episodes))
        os.replace(tmp_path, out_path)
        all_episodes_for_stats = all_episode_summaries
        splits = np.concatenate(split_chunks, axis=0) if split_chunks else np.array([], dtype=object)
        n_total = sum(int(e["episode_len"]) for e in all_episodes_for_stats)
    else:
        with ctx.Pool(processes=n_processes) as pool:
            all_episodes = []
            with tqdm(total=args.episodes, desc="collect") as pbar:
                for chunk_episodes in pool.imap_unordered(_collect_chunk, chunks):
                    all_episodes.extend(chunk_episodes)
                    pbar.update(len(chunk_episodes))
        all_episodes.sort(key=lambda e: e["episode_id"])
        n_total = sum(len(e["actions"]) for e in all_episodes)

        from src.data import write_episodes
        print(f"Writing {len(all_episodes)} episodes to {args.out}")
        splits = write_episodes(args.out, all_episodes, val_fraction=args.val_fraction, split_seed=args.split_seed)
        if args.dataset_version == "v3":
            write_v3_episode_metadata(args.out, all_episodes)
        all_episodes_for_stats = all_episodes

    elapsed = time.time() - t0
    print(f"Collected {len(all_episodes_for_stats)} episodes ({n_total} transitions) in {elapsed:.1f}s "
          f"= {len(all_episodes_for_stats)/elapsed:.2f} ep/s")
    if args.stream_write:
        print(f"Wrote streamed dataset to {args.out}")
    print(f"Mean ep length: {n_total / len(all_episodes_for_stats):.1f}")
    print_dataset_stats(all_episodes_for_stats, policies, splits=splits)
    if args.check_readable:
        check_dataset_readable(args.out, seq_len=args.check_seq_len)


if __name__ == "__main__":
    main()
