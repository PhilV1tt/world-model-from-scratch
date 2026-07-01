import numpy as np
import torch

from scripts.plan import (
    cem_plan,
    latent_plan_cost,
    pd_rollout_warm_start,
    pose_plan_cost,
    rerank_action_candidates,
    run_episode,
    simulate_action_trajectory,
    smooth_action_sequences,
)
from scripts.sweep_plan import summarize
from src.parking_env import goal_distance, make_parking, render_pixel


class DummyCostModel:
    def latent_goal_cost(self, z_rollout, z_goal):
        if z_goal.dim() == 2:
            z_goal = z_goal.unsqueeze(1)
        goal = z_goal[:, -1:, :].expand(z_rollout.shape[0], 1, z_rollout.shape[-1])
        return (z_rollout[:, -1:, :] - goal.detach()).square().sum(dim=(1, 2))


class LinearRolloutModel(DummyCostModel):
    def rollout_latents(self, z_init, action_seq):
        return z_init.unsqueeze(1) + action_seq.cumsum(dim=1)


class TinyPlanningModel(LinearRolloutModel):
    def encode(self, obs):
        return torch.zeros(obs.shape[0], 2, device=obs.device)


class Landmark:
    def __init__(self, position=(0.0, 0.0), heading=0.0):
        self.position = np.asarray(position, dtype=np.float32)
        self.heading = float(heading)


class FakeVehicle:
    def __init__(self, position=(0.0, 0.0), heading=0.0):
        self.position = np.asarray(position, dtype=np.float32)
        self.heading = float(heading)
        self.speed = 0.0
        self.crashed = False


class FakeRoad:
    def __init__(self, goal):
        self.objects = [goal]


class FakeParkingEnv:
    def __init__(self, start=(0.0, 2.0), heading=0.0, goal=(0.0, 0.0), goal_heading=0.0):
        self.vehicle = FakeVehicle(start, heading)
        self.goal = Landmark(goal, goal_heading)
        self.controlled_vehicles = [self.vehicle]
        self.road = FakeRoad(self.goal)
        self.steps = 0

    @property
    def unwrapped(self):
        return self

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        vehicle = self.controlled_vehicles[0]
        vehicle.position = vehicle.position + action[:2]
        vehicle.speed = float(np.linalg.norm(action[:2]))
        self.vehicle = vehicle
        self.steps += 1
        if action[0] > 8.0:
            vehicle.crashed = True
        return None, 0.0, False, False, {}


def test_latent_plan_cost_terminal_and_trajectory():
    model = DummyCostModel()
    z_rollout = torch.tensor([[[1.0, 0.0], [2.0, 0.0]], [[0.0, 0.0], [3.0, 4.0]]])
    z_goal = torch.tensor([[0.0, 0.0]])

    terminal = latent_plan_cost(model, z_rollout, z_goal, trajectory_weight=0.0)
    with_traj = latent_plan_cost(model, z_rollout, z_goal, trajectory_weight=0.5)

    assert torch.allclose(terminal, torch.tensor([4.0, 25.0]))
    assert torch.allclose(with_traj, torch.tensor([5.25, 31.25]))


def test_action_smoothing_keeps_first_action():
    actions = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [-1.0, 1.0]]])
    out = smooth_action_sequences(actions, smoothing=0.5)
    expected = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [-0.5, 0.5]]])
    assert torch.allclose(out, expected)


def test_cem_seed_reproducible():
    model = LinearRolloutModel()
    z_init = torch.zeros(1, 2)
    z_goal = torch.tensor([[0.5, -0.25]])

    plan_a = cem_plan(
        model,
        z_init,
        z_goal,
        horizon=3,
        pop_size=8,
        elites=3,
        iters=2,
        action_l2=0.0,
        smooth_l2=0.0,
        action_smoothing=0.0,
        seed=123,
    )
    plan_b = cem_plan(
        model,
        z_init,
        z_goal,
        horizon=3,
        pop_size=8,
        elites=3,
        iters=2,
        action_l2=0.0,
        smooth_l2=0.0,
        action_smoothing=0.0,
        seed=123,
    )

    assert np.allclose(plan_a, plan_b)


def test_cem_can_return_top_candidates():
    model = LinearRolloutModel()
    z_init = torch.zeros(1, 2)
    z_goal = torch.tensor([[0.5, -0.25]])

    best, candidates = cem_plan(
        model,
        z_init,
        z_goal,
        horizon=3,
        pop_size=8,
        elites=3,
        iters=2,
        action_l2=0.0,
        smooth_l2=0.0,
        action_smoothing=0.0,
        seed=123,
        return_candidates=True,
        candidate_count=2,
    )

    assert best.shape == (3, 2)
    assert candidates.shape == (3, 3, 2)
    assert np.allclose(candidates[0], best)


def test_cem_return_candidates_keeps_warm_start_candidate():
    model = LinearRolloutModel()
    z_init = torch.zeros(1, 2)
    z_goal = torch.tensor([[0.0, 0.0]])
    warm = np.array([[0.6, -0.2], [0.4, -0.1], [0.2, 0.0]], dtype=np.float32)

    _, candidates = cem_plan(
        model,
        z_init,
        z_goal,
        horizon=3,
        pop_size=8,
        elites=3,
        iters=2,
        warm_start=warm,
        action_l2=0.0,
        smooth_l2=0.0,
        action_smoothing=0.0,
        seed=123,
        return_candidates=True,
        candidate_count=2,
    )

    assert candidates.shape == (4, 3, 2)
    assert any(np.allclose(candidate, warm) for candidate in candidates)


def test_pd_warm_start_restores_env_state():
    env = make_parking(seed=0, img_size=64)
    env.reset()
    render_pixel(env)
    unwrapped = env.unwrapped
    vehicle = unwrapped.controlled_vehicles[0]
    before = (
        vehicle.position.copy(),
        float(vehicle.heading),
        float(vehicle.speed),
        getattr(unwrapped, "steps", None),
        getattr(unwrapped, "time", None),
        goal_distance(env),
    )

    warm = pd_rollout_warm_start(env, horizon=3)
    vehicle = env.unwrapped.controlled_vehicles[0]
    after = (
        vehicle.position.copy(),
        float(vehicle.heading),
        float(vehicle.speed),
        getattr(env.unwrapped, "steps", None),
        getattr(env.unwrapped, "time", None),
        goal_distance(env),
    )
    env.close()

    assert warm.shape == (3, 2)
    assert np.allclose(before[0], after[0])
    assert before[1:] == after[1:]


def test_pose_plan_cost_rewards_strict_slot_alignment_and_restores_env():
    env = FakeParkingEnv(start=(0.0, 2.0), goal=(0.0, 0.0), goal_heading=0.0)
    before_pos = env.controlled_vehicles[0].position.copy()
    before_steps = env.steps

    bad = np.zeros((2, 2), dtype=np.float32)
    good = np.array([[0.0, -1.0], [0.0, -1.0]], dtype=np.float32)
    bad_cost = pose_plan_cost(env, bad, lateral_weight=1.0, along_weight=0.0, heading_weight=0.0)
    good_cost = pose_plan_cost(env, good, lateral_weight=1.0, along_weight=0.0, heading_weight=0.0)

    assert good_cost < bad_cost
    assert np.allclose(env.controlled_vehicles[0].position, before_pos)
    assert env.steps == before_steps


def test_pose_plan_cost_penalizes_heading_and_collision():
    heading_env = FakeParkingEnv(start=(0.0, 0.0), heading=np.pi / 2, goal=(0.0, 0.0), goal_heading=0.0)
    heading_cost = pose_plan_cost(
        heading_env,
        np.zeros((1, 2), dtype=np.float32),
        lateral_weight=0.0,
        along_weight=0.0,
        heading_weight=2.0,
        collision_weight=0.0,
    )
    assert heading_cost == 2.0

    collision_env = FakeParkingEnv(start=(0.0, 0.0), goal=(0.0, 0.0), goal_heading=0.0)
    collision_cost = pose_plan_cost(
        collision_env,
        np.array([[9.0, 0.0]], dtype=np.float32),
        lateral_weight=0.0,
        along_weight=0.0,
        heading_weight=0.0,
        collision_weight=7.0,
    )
    assert collision_cost == 7.0
    assert not collision_env.controlled_vehicles[0].crashed


def test_rerank_action_candidates_keeps_latent_cost_but_can_choose_strict_pose():
    env = FakeParkingEnv(start=(0.0, 2.0), goal=(0.0, 0.0), goal_heading=0.0)
    model = LinearRolloutModel()
    z_init = torch.zeros(1, 2)
    z_goal = torch.zeros(1, 2)
    candidates = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, -1.0], [0.0, -1.0]],
        ],
        dtype=np.float32,
    )

    latent_only = rerank_action_candidates(model, env, z_init, z_goal, candidates, pose_cost_weight=0.0)
    strict = rerank_action_candidates(
        model,
        env,
        z_init,
        z_goal,
        candidates,
        action_l2=0.0,
        smooth_l2=0.0,
        pose_cost_weight=3.0,
        lateral_weight=1.0,
        along_weight=0.0,
        heading_weight=0.0,
        collision_weight=0.0,
    )

    assert np.allclose(latent_only, candidates[0])
    assert np.allclose(strict, candidates[1])
    assert np.allclose(env.controlled_vehicles[0].position, np.array([0.0, 2.0], dtype=np.float32))


def test_simulate_action_trajectory_restores_env_state():
    env = make_parking(seed=1, img_size=64)
    env.reset()
    render_pixel(env)
    vehicle = env.unwrapped.controlled_vehicles[0]
    before = (
        vehicle.position.copy(),
        float(vehicle.heading),
        float(vehicle.speed),
        getattr(env.unwrapped, "steps", None),
        getattr(env.unwrapped, "time", None),
        goal_distance(env),
    )

    actions = np.array([[0.3, 0.1], [0.2, -0.2], [0.0, 0.0]], dtype=np.float32)
    trajectory = simulate_action_trajectory(env, actions)
    vehicle = env.unwrapped.controlled_vehicles[0]
    after = (
        vehicle.position.copy(),
        float(vehicle.heading),
        float(vehicle.speed),
        getattr(env.unwrapped, "steps", None),
        getattr(env.unwrapped, "time", None),
        goal_distance(env),
    )
    env.close()

    assert trajectory.shape == (3, 2)
    assert trajectory.dtype == np.float32
    assert np.isfinite(trajectory).all()
    assert np.allclose(before[0], after[0])
    assert before[1:] == after[1:]


def test_run_episode_can_return_planned_trajectories():
    env = make_parking(seed=2, img_size=64)
    model = TinyPlanningModel()
    out = run_episode(
        model,
        env,
        horizon=3,
        mpc_apply=1,
        max_env_steps=2,
        pop_size=4,
        elites=2,
        iters=1,
        device=torch.device("cpu"),
        pose_cost_weight=1.0,
        lateral_weight=1.0,
        along_weight=0.5,
        heading_weight=0.25,
        collision_weight=10.0,
        return_planned_trajectories=True,
        seed=11,
    )
    env.close()

    assert "planned_trajectories" in out
    assert "strict_success" in out
    assert "pose_metrics" in out
    assert len(out["planned_trajectories"]) >= 1
    assert all(traj.shape == (3, 2) for traj in out["planned_trajectories"])
    assert all(traj.dtype == np.float32 for traj in out["planned_trajectories"])


def test_sweep_summary_has_required_metrics():
    stats = summarize(
        init_dist=[5.0, 6.0, 7.0],
        final_dist=[3.0, 1.0, 9.0],
        final_ang=[10.0, 5.0, 20.0],
        success=[0.0, 1.0, 0.0],
    )

    assert stats["mean_final_dist"] == np.float32(13.0 / 3.0)
    assert stats["median_final_dist"] == 3.0
    assert stats["best_final_dist"] == 1.0
    assert stats["worst_final_dist"] == 9.0
    assert stats["success_rate"] == np.float32(1.0 / 3.0)
    assert stats["mean_dist_delta"] == np.float32(5.0 / 3.0)
