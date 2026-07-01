"""Metrics explicites pour verifier qu'une voiture est vraiment garee."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


DEFAULT_PARKED_THRESHOLDS = {
    "max_dist_m": 1.5,
    "max_angle_deg": 15.0,
    "max_lateral_m": 0.45,
    "max_along_m": 1.0,
    "max_heading_deg": None,
    "max_speed_mps": None,
    "allow_collision": False,
}


def _as_xy(position) -> np.ndarray:
    arr = np.asarray(position, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 2:
        raise ValueError("position must contain at least x/y")
    return arr[:2]


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _axis_angle_deg(angle_rad: float) -> float:
    err = abs(_wrap_angle(angle_rad))
    if err > np.pi / 2:
        err = np.pi - err
    return float(np.degrees(abs(err)))


def _get_unwrapped(env):
    return getattr(env, "unwrapped", env)


def _get_vehicle(env):
    unwrapped = _get_unwrapped(env)
    vehicles = getattr(unwrapped, "controlled_vehicles", None)
    if vehicles:
        return vehicles[0]
    vehicle = getattr(unwrapped, "vehicle", None)
    if vehicle is not None:
        return vehicle
    raise RuntimeError("No controlled vehicle found")


def _get_goal(env):
    unwrapped = _get_unwrapped(env)
    road = getattr(unwrapped, "road", None)
    for obj in getattr(road, "objects", ()):
        if obj.__class__.__name__ == "Landmark" and hasattr(obj, "position"):
            return obj
    raise RuntimeError("No Landmark goal found in road.objects")


def _bool_attr(obj, names: tuple[str, ...]) -> bool:
    for name in names:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        if callable(value):
            value = value()
        try:
            if bool(value):
                return True
        except ValueError:
            if np.asarray(value).any():
                return True
    return False


def collision_metric(env) -> bool:
    """Retourne True si l'env expose simplement un crash/collision."""
    unwrapped = _get_unwrapped(env)
    names = ("crashed", "collided", "collision", "collision_occurred")
    candidates = []

    vehicles = getattr(unwrapped, "controlled_vehicles", None)
    if vehicles:
        candidates.extend(vehicles)
    vehicle = getattr(unwrapped, "vehicle", None)
    if vehicle is not None:
        candidates.append(vehicle)
    candidates.extend([unwrapped, env])

    return any(_bool_attr(obj, names) for obj in candidates if obj is not None)


def slot_alignment_metrics(vehicle, goal) -> dict[str, float]:
    """Projette la pose du vehicule dans le repere de la place cible."""
    vehicle_pos = _as_xy(vehicle.position)
    goal_pos = _as_xy(goal.position)
    vehicle_heading = float(getattr(vehicle, "heading", 0.0))
    goal_heading = float(getattr(goal, "heading", 0.0))

    forward = np.array([np.cos(goal_heading), np.sin(goal_heading)], dtype=np.float32)
    left = np.array([-np.sin(goal_heading), np.cos(goal_heading)], dtype=np.float32)
    rel = vehicle_pos - goal_pos

    along = float(np.dot(rel, forward))
    lateral = float(np.dot(rel, left))
    heading_error_rad = _wrap_angle(vehicle_heading - goal_heading)
    heading_error_deg = float(np.degrees(heading_error_rad))
    abs_heading_error_deg = float(abs(heading_error_deg))
    axis_error_deg = _axis_angle_deg(heading_error_rad)
    dist = float(np.linalg.norm(rel))

    return {
        "dist": dist,
        "distance_m": dist,
        "angle_deg": axis_error_deg,
        "axis_error_deg": axis_error_deg,
        "heading_error_rad": heading_error_rad,
        "heading_error_deg": heading_error_deg,
        "abs_heading_error_deg": abs_heading_error_deg,
        "along_offset_m": along,
        "lateral_offset_m": lateral,
        "abs_along_offset_m": abs(along),
        "abs_lateral_offset_m": abs(lateral),
    }


def final_pose_metrics(env) -> dict[str, float | bool]:
    """Mesure finale lisible pour eval: distance, angle, offsets, vitesse, collision."""
    vehicle = _get_vehicle(env)
    goal = _get_goal(env)
    metrics: dict[str, float | bool] = dict(slot_alignment_metrics(vehicle, goal))
    metrics["speed_mps"] = float(getattr(vehicle, "speed", 0.0))
    metrics["collided"] = collision_metric(env)
    return metrics


def _metric_value(metrics: Mapping[str, object], primary: str, fallback: str | None = None) -> float:
    if primary in metrics:
        return float(metrics[primary])
    if fallback is not None and fallback in metrics:
        return float(metrics[fallback])
    raise KeyError(primary)


def parked_success(metrics: Mapping[str, object], thresholds: Mapping[str, object] | None = None) -> bool:
    """Succes strict: proche, axe aligne, centre dans la place, sans collision."""
    cfg = dict(DEFAULT_PARKED_THRESHOLDS)
    if thresholds:
        cfg.update(thresholds)

    dist = _metric_value(metrics, "dist", "distance_m")
    angle = _metric_value(metrics, "angle_deg", "axis_error_deg")
    lateral = _metric_value(metrics, "abs_lateral_offset_m", "lateral_offset_m")
    along = _metric_value(metrics, "abs_along_offset_m", "along_offset_m")

    ok = (
        dist <= float(cfg["max_dist_m"])
        and angle <= float(cfg["max_angle_deg"])
        and abs(lateral) <= float(cfg["max_lateral_m"])
        and abs(along) <= float(cfg["max_along_m"])
    )

    max_heading_deg = cfg.get("max_heading_deg")
    if max_heading_deg is not None:
        heading = _metric_value(metrics, "abs_heading_error_deg", "heading_error_deg")
        ok = ok and abs(heading) <= float(max_heading_deg)

    max_speed_mps = cfg.get("max_speed_mps")
    if max_speed_mps is not None:
        speed = float(metrics.get("speed_mps", 0.0))
        ok = ok and abs(speed) <= float(max_speed_mps)

    if not bool(cfg["allow_collision"]) and bool(metrics.get("collided", False)):
        ok = False

    return bool(ok)


def user_strict_success(
    metrics: Mapping[str, float | bool],
    max_dist_m: float = 0.15,
    max_heading_deg: float = 5.0,
) -> bool:
    """Reussite stricte "utilisateur": position tres fine ET vrai cap tres fin.

    Contrairement a parked_success (qui plie le cap dans [0,90] et l'ignore par defaut),
    on teste ici le vrai ecart de cap absolu et une tolerance de position serree.
    """
    if bool(metrics.get("collided", False)):
        return False
    dist = float(metrics.get("dist", metrics.get("distance_m", 1e9)))
    heading = float(metrics.get("abs_heading_error_deg", 1e9))
    return dist <= float(max_dist_m) and heading <= float(max_heading_deg)
