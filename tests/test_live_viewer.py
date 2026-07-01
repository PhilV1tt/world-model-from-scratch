import numpy as np

from scripts.live_viewer import (
    goal_world_state,
    predict_plan_world_points,
    world_points_to_tile_pixels,
)


class DummySurface:
    def pos2pix(self, x, y):
        return x * 10.0, 100.0 - y * 10.0


class DummyViewer:
    sim_surface = DummySurface()


class DummyUnwrapped:
    viewer = DummyViewer()


class DummyEnv:
    unwrapped = DummyUnwrapped()


def test_predict_plan_world_points_uses_highway_env_action_order():
    state = {
        "position": np.array([0.0, 0.0], dtype=np.float32),
        "heading": 0.0,
        "speed": 0.0,
    }
    # highway-env order: [acceleration, steering].
    accel_then_steer = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    points = predict_plan_world_points(state, accel_then_steer)

    assert len(points) == 3
    assert points[1][0] > 0.0
    assert abs(points[1][1]) < 1e-6
    assert points[2][1] > 0.0


def test_world_points_to_tile_pixels_scales_from_renderer_surface():
    points = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

    pixels = world_points_to_tile_pixels(DummyEnv(), points, frame_shape=(320, 640, 3))

    assert pixels == [(10, 160), (30, 120)]


def test_goal_world_state_finds_landmark_like_object():
    class Landmark:
        position = np.array([4.0, -2.0])
        heading = 1.5

    class Road:
        objects = [object(), Landmark()]

    class Unwrapped:
        road = Road()

    class Env:
        unwrapped = Unwrapped()

    goal = goal_world_state(Env())

    assert np.allclose(goal["position"], [4.0, -2.0])
    assert goal["heading"] == 1.5
