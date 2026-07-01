"""Wrapper highway-env parking-v0 pour produire des frames pixel + goal pixel.

parking-v0 par defaut retourne un dict d'etats kinematics (6D). On ajoute un mode
ou l'observation est l'image rendue (top-down view) pour LeWM.

Strategie pour le goal_obs:
- A reset, on connait la place cible (desired_goal kinematics state).
- On simule un env "virtuel" ou la voiture est deja a la place cible, on rend, et on capture
  l'image - ca devient goal_obs.

Le rendu RGB de parking-v0 est 600x300 par defaut. On le crop au centre et resize a 64x64.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import highway_env  # noqa: F401  (registers env)
import numpy as np


PARKING_CONFIG = {
    "duration": 100,
    "screen_width": 256,
    "screen_height": 256,
    "centering_position": [0.5, 0.5],
    "scaling": 5.0,
    "policy_frequency": 5,
    "simulation_frequency": 15,
    "controlled_vehicles": 1,
}

PARKING_VARIANT_CONFIGS = {
    "default": {},
    "standard": {},
    "wide_start": {},
    "long_approach": {},
    "near_goal": {},
    "short_correction": {},
    "reverse_entry": {},
    "final_alignment": {},
    "slot_offset": {},
    "static_vehicles": {"vehicles_count": 4},
    "crowded_static": {"vehicles_count": 8},
    "no_walls": {"add_walls": False},
}


def _crop_resize(img: np.ndarray, out_size: int = 64) -> np.ndarray:
    """img: (H, W, 3) uint8 -> (out_size, out_size, 3) uint8 par downsampling lineaire."""
    h, w, _ = img.shape
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    img = img[top : top + side, left : left + side]
    factor = side // out_size
    if factor < 1:
        return img
    img = img[: factor * out_size, : factor * out_size]
    img = img.reshape(out_size, factor, out_size, factor, 3).mean(axis=(1, 3))
    return img.astype(np.uint8)


def show_initial_frame(seed: int = 0):
    """Helper interactif : retourne (current_64, goal_64, full_render) pour debug visuel."""
    env = make_parking(seed=seed, img_size=64)
    env.reset()
    full = env.render()
    cur = render_pixel(env)
    goal = render_goal_pixel(env)
    env.close()
    return cur, goal, full


def make_parking(
    seed: int | None = None,
    img_size: int = 64,
    env_variant: str = "default",
    config_overrides: dict | None = None,
) -> gym.Env:
    if env_variant not in PARKING_VARIANT_CONFIGS:
        raise ValueError(f"unknown parking env variant: {env_variant}")
    config = dict(PARKING_CONFIG)
    config.update(PARKING_VARIANT_CONFIGS[env_variant])
    if config_overrides:
        config.update(config_overrides)
    env = gym.make("parking-v0", render_mode="rgb_array", config=config)
    if seed is not None:
        env.reset(seed=seed)
    env.unwrapped._lewm_img_size = img_size
    env.unwrapped._lewm_env_variant = env_variant
    return env


def render_pixel(env: gym.Env) -> np.ndarray:
    """Frame courante en (img_size, img_size, 3) uint8."""
    img = env.render()
    out_size = getattr(env.unwrapped, "_lewm_img_size", 64)
    return _crop_resize(img, out_size=out_size)


def _get_goal(env: gym.Env):
    """Recupere le Landmark goal dans road.objects (parking-v0)."""
    from highway_env.vehicle.objects import Landmark
    for obj in env.unwrapped.road.objects:
        if isinstance(obj, Landmark):
            return obj
    raise RuntimeError("No Landmark goal found in road.objects")


def render_goal_pixel(env: gym.Env) -> np.ndarray:
    """Image qu'on aurait si la voiture etait deja a la place cible."""
    unwrapped = env.unwrapped
    vehicle = unwrapped.controlled_vehicles[0]
    saved_pos = vehicle.position.copy()
    saved_heading = vehicle.heading
    saved_speed = vehicle.speed

    goal = _get_goal(env)
    vehicle.position = goal.position.copy()
    vehicle.heading = goal.heading
    vehicle.speed = 0.0

    img = env.render()

    vehicle.position = saved_pos
    vehicle.heading = saved_heading
    vehicle.speed = saved_speed

    out_size = getattr(unwrapped, "_lewm_img_size", 64)
    return _crop_resize(img, out_size=out_size)


def goal_distance(env: gym.Env) -> tuple[float, float]:
    """Retourne (distance positionnelle, ecart angulaire en degres) entre vehicule et goal."""
    unwrapped = env.unwrapped
    v = unwrapped.controlled_vehicles[0]
    g = _get_goal(env)
    pos_dist = float(np.linalg.norm(v.position - g.position))
    cos_diff = np.cos(v.heading) * np.cos(g.heading) + np.sin(v.heading) * np.sin(g.heading)
    angle_diff_rad = float(np.arccos(np.clip(cos_diff, -1.0, 1.0)))
    angle_diff_deg = np.degrees(min(angle_diff_rad, np.pi - angle_diff_rad))
    return pos_dist, angle_diff_deg


def static_vehicle_count(env: gym.Env) -> int:
    """Number of non-controlled parked/static vehicles in the current scene."""
    controlled = set(env.unwrapped.controlled_vehicles)
    return sum(1 for vehicle in env.unwrapped.road.vehicles if vehicle not in controlled)


def obstacle_count(env: gym.Env) -> int:
    """Count solid road obstacles plus non-controlled vehicles.

    In highway-env parking-v0, walls are represented as Obstacle objects and
    parked cars are regular Vehicle objects. This keeps both visible blockers
    in one simple metadata field.
    """
    from highway_env.vehicle.objects import Obstacle

    road_obstacles = sum(1 for obj in env.unwrapped.road.objects if isinstance(obj, Obstacle))
    return road_obstacles + static_vehicle_count(env)


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _goal_axes(goal) -> tuple[np.ndarray, np.ndarray]:
    forward = np.array([np.cos(goal.heading), np.sin(goal.heading)])
    left = np.array([-np.sin(goal.heading), np.cos(goal.heading)])
    return forward, left


@dataclass
class HeuristicPDPolicy:
    """Politique simple PD pour viser la place cible.

    Action: [acceleration, steering] dans [-1, 1]^2 pour highway-env.
    """
    k_pos: float = 0.5
    k_heading: float = 1.0
    k_vel: float = 0.3
    noise: float = 0.0

    def __call__(self, env: gym.Env) -> np.ndarray:
        unwrapped = env.unwrapped
        v = unwrapped.controlled_vehicles[0]
        g = _get_goal(env)

        dx = g.position[0] - v.position[0]
        dy = g.position[1] - v.position[1]
        target_heading = np.arctan2(dy, dx)
        heading_err = np.arctan2(np.sin(target_heading - v.heading), np.cos(target_heading - v.heading))

        dist = np.linalg.norm([dx, dy])
        speed = float(v.speed)
        desired_speed = np.clip(self.k_pos * dist, 0, 5.0)
        accel = self.k_vel * (desired_speed - speed)

        if dist < 2.0:
            goal_heading_err = np.arctan2(np.sin(g.heading - v.heading), np.cos(g.heading - v.heading))
            steer = self.k_heading * goal_heading_err
            accel = -0.5 * speed
        else:
            steer = self.k_heading * heading_err

        if self.noise > 0:
            steer += np.random.randn() * self.noise
            accel += np.random.randn() * self.noise

        return np.clip(np.array([accel, steer], dtype=np.float32), -1.0, 1.0)


@dataclass
class ParkingExpertPolicy:
    """Heuristique plus riche pour generer de la data utile au planner.

    Elle choisit automatiquement marche avant/arriere selon la position relative
    du goal, ralentit pres du goal, puis aligne l orientation finale.
    """
    k_pos: float = 0.42
    k_heading: float = 1.35
    k_vel: float = 0.45
    speed_cap: float = 3.0
    align_dist: float = 2.2
    reverse_bias: float = 0.15
    noise: float = 0.0
    force_reverse: bool = False
    final_alignment: bool = False

    def __call__(self, env: gym.Env) -> np.ndarray:
        unwrapped = env.unwrapped
        v = unwrapped.controlled_vehicles[0]
        g = _get_goal(env)

        rel = g.position - v.position
        dist = float(np.linalg.norm(rel))
        forward = np.array([np.cos(v.heading), np.sin(v.heading)])
        local_x = float(np.dot(rel, forward))

        goal_heading_err = _wrap_angle(g.heading - v.heading)
        target_heading = float(np.arctan2(rel[1], rel[0]))
        forward_err = _wrap_angle(target_heading - v.heading)
        reverse_err = _wrap_angle(target_heading + np.pi - v.heading)

        use_reverse = self.force_reverse or local_x < -self.reverse_bias
        direction = -1.0 if use_reverse else 1.0
        heading_err = reverse_err if use_reverse else forward_err

        if self.final_alignment or dist < self.align_dist:
            goal_forward, goal_left = _goal_axes(g)
            from_goal = v.position - g.position
            along = float(np.dot(from_goal, goal_forward))
            lateral = float(np.dot(from_goal, goal_left))
            desired_speed = float(np.clip(-0.55 * along, -0.9, 0.9))
            if abs(along) < 0.25 and abs(lateral) > 0.35:
                desired_speed = 0.45 * np.sign(lateral)
            if abs(goal_heading_err) > 0.75:
                desired_speed *= 0.45
            if dist < 0.45 and abs(goal_heading_err) < 0.25:
                desired_speed = 0.0

            direction = -1.0 if desired_speed < -0.05 else 1.0
            heading_err = goal_heading_err + 0.28 * np.clip(lateral, -2.0, 2.0)
        else:
            desired_speed = direction * min(self.speed_cap, self.k_pos * dist)

        if dist < 0.6:
            desired_speed = 0.0

        steer = direction * self.k_heading * heading_err
        accel = self.k_vel * (desired_speed - float(v.speed))

        if self.noise > 0:
            steer += np.random.randn() * self.noise
            accel += np.random.randn() * self.noise

        return np.clip(np.array([accel, steer], dtype=np.float32), -1.0, 1.0)


@dataclass
class PoseParkController:
    """Controleur de pose privilegie (utilise la pose but de l'env).

    Contrairement a l'expert heuristique qui s'arrete a ~0.6 m sans regler le cap,
    ce controleur suit la ligne centrale de la place (comme du suivi de voie) pour
    aligner le cap en entrant, puis fait une regulation terminale serree pour viser
    dist < ~0.12 m et |cap| < ~5 deg. Stateless: se rebranche sur run_policy_episode.
    """
    lookahead: float = 2.0
    k_steer: float = 1.4
    k_speed: float = 0.55
    v_cruise: float = 3.0
    v_creep: float = 0.55
    tol_pos: float = 0.12
    tol_head: float = 0.08          # rad (~4.6 deg)
    heading_align_dist: float = 1.3  # en deca: on privilegie l'alignement final

    def __call__(self, env: gym.Env) -> np.ndarray:
        v = env.unwrapped.controlled_vehicles[0]
        g = _get_goal(env)
        forward, left = _goal_axes(g)
        rel = np.asarray(v.position, dtype=np.float64) - np.asarray(g.position, dtype=np.float64)
        along = float(rel @ forward)
        lateral = float(rel @ left)
        he = _wrap_angle(float(v.heading) - float(g.heading))
        dist = float(np.linalg.norm(rel))
        speed = float(v.speed)
        fwd_vec = np.array([np.cos(v.heading), np.sin(v.heading)])

        if dist < self.tol_pos and abs(he) < self.tol_head:
            return np.array([-0.6 * speed, 0.0], dtype=np.float32)

        if dist >= self.heading_align_dist:
            # Poursuite d'un point sur la ligne centrale, glissant vers la place.
            carrot_param = along - np.sign(along) * self.lookahead
            if np.sign(carrot_param) != np.sign(along):
                carrot_param = 0.0
            carrot = np.asarray(g.position, dtype=np.float64) + forward * carrot_param
            to_carrot = carrot - np.asarray(v.position, dtype=np.float64)
            gear = 1.0 if float(to_carrot @ fwd_vec) >= 0 else -1.0
            travel_heading = float(np.arctan2(to_carrot[1], to_carrot[0]))
            if gear < 0:
                steer_err = _wrap_angle(travel_heading + np.pi - float(v.heading))
            else:
                steer_err = _wrap_angle(travel_heading - float(v.heading))
            v_des = min(self.v_cruise, 1.2 * dist + 0.25)
            if abs(steer_err) > 0.5:
                v_des *= 0.5
        else:
            # Regulation terminale: aligner le cap sur le but, avancer/reculer le long de l'axe.
            facing = 1.0 if np.cos(he) >= 0 else -1.0
            gear = -1.0 if along > 0 else 1.0
            gear *= facing
            steer_err = _wrap_angle(-he) * facing
            v_des = self.v_creep

        target_speed = gear * v_des
        accel = self.k_speed * (target_speed - speed)
        steer = gear * self.k_steer * steer_err
        return np.clip(np.array([accel, steer], dtype=np.float32), -1.0, 1.0)
