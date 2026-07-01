import math

import numpy as np
import pytest

from src.parking_env import goal_distance, make_parking
from src.parking_metrics import (
    collision_metric,
    final_pose_metrics,
    parked_success,
    slot_alignment_metrics,
)


class Obj:
    def __init__(self, position, heading=0.0, speed=0.0, crashed=False):
        self.position = np.array(position, dtype=np.float32)
        self.heading = float(heading)
        self.speed = float(speed)
        self.crashed = crashed


def test_slot_alignment_metrics_projects_position_into_goal_frame():
    goal = Obj([10.0, 5.0], heading=0.0)
    vehicle = Obj([11.0, 5.25], heading=math.radians(10.0))

    metrics = slot_alignment_metrics(vehicle, goal)

    assert metrics["dist"] == pytest.approx(math.sqrt(1.0**2 + 0.25**2))
    assert metrics["along_offset_m"] == pytest.approx(1.0)
    assert metrics["lateral_offset_m"] == pytest.approx(0.25)
    assert metrics["abs_along_offset_m"] == pytest.approx(1.0)
    assert metrics["abs_lateral_offset_m"] == pytest.approx(0.25)
    assert metrics["heading_error_deg"] == pytest.approx(10.0)
    assert metrics["angle_deg"] == pytest.approx(10.0)


def test_slot_alignment_keeps_axis_error_separate_from_exact_heading():
    goal = Obj([0.0, 0.0], heading=0.0)
    opposite_direction = Obj([0.0, 0.0], heading=math.pi)
    diagonal = Obj([0.0, 0.0], heading=math.pi / 2)

    opposite = slot_alignment_metrics(opposite_direction, goal)
    diagonal_metrics = slot_alignment_metrics(diagonal, goal)

    assert opposite["angle_deg"] == pytest.approx(0.0)
    assert opposite["abs_heading_error_deg"] == pytest.approx(180.0)
    assert diagonal_metrics["angle_deg"] == pytest.approx(90.0)


def test_parked_success_requires_distance_alignment_lateral_and_no_collision():
    good = {
        "dist": 0.55,
        "angle_deg": 4.0,
        "abs_lateral_offset_m": 0.12,
        "abs_along_offset_m": 0.25,
        "collided": False,
    }

    assert parked_success(good)
    assert not parked_success({**good, "dist": 2.0})
    assert not parked_success({**good, "angle_deg": 35.0})
    assert not parked_success({**good, "abs_lateral_offset_m": 0.9})
    assert not parked_success({**good, "abs_along_offset_m": 1.8})
    assert not parked_success({**good, "collided": True})


def test_parked_success_can_require_exact_heading_direction():
    metrics = {
        "dist": 0.1,
        "angle_deg": 0.0,
        "abs_heading_error_deg": 180.0,
        "abs_lateral_offset_m": 0.0,
        "abs_along_offset_m": 0.0,
        "collided": False,
    }

    assert parked_success(metrics)
    assert not parked_success(metrics, {"max_heading_deg": 20.0})


def test_collision_metric_reads_controlled_vehicle_crashed_flag():
    class Unwrapped:
        controlled_vehicles = [Obj([0.0, 0.0], crashed=True)]

    class Env:
        unwrapped = Unwrapped()

    assert collision_metric(Env())


def test_final_pose_metrics_matches_goal_distance_on_real_parking_env():
    env = make_parking(seed=0, img_size=32)
    env.reset(seed=0)

    metrics = final_pose_metrics(env)
    dist, angle = goal_distance(env)
    env.close()

    assert metrics["dist"] == pytest.approx(dist)
    assert metrics["angle_deg"] == pytest.approx(float(angle))
    assert isinstance(metrics["collided"], bool)
    assert np.isfinite(
        [
            metrics["along_offset_m"],
            metrics["lateral_offset_m"],
            metrics["heading_error_deg"],
            metrics["speed_mps"],
        ]
    ).all()
