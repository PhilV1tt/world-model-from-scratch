import h5py
import numpy as np
import pytest

from src.data import EpisodeH5Writer, ParkingTrajectoryDataset, write_episodes


def make_episode(seed: int, n: int = 6):
    rng = np.random.default_rng(seed)
    return {
        "episode_id": seed,
        "obs": rng.integers(0, 255, size=(n + 1, 64, 64, 3), dtype=np.uint8),
        "actions": rng.uniform(-1, 1, size=(n, 2)).astype(np.float32),
        "goal_obs": rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8),
        "policy_name": "pd",
        "seed": seed,
        "init_dist": 5.0,
        "final_dist": 1.0,
        "final_ang": 7.0,
        "success": 1,
        "episode_len": n,
    }


def test_dataset_v2_metadata_and_split(tmp_path):
    path = tmp_path / "train_v2.h5"
    write_episodes(path, [make_episode(i) for i in range(10)], val_fraction=0.2, split_seed=0)

    with h5py.File(path, "r") as f:
        assert f.attrs["format_version"] == "parking_v2"
        assert "episode_split" in f
        assert "episode_policy_name" in f
        assert "episode_seed" in f
        assert "episode_init_dist" in f
        assert "episode_final_dist" in f
        assert "episode_final_ang" in f
        assert "episode_success" in f
        assert "episode_len" in f
        assert set(f["episode_split"].asstr()[:]) == {"train", "val"}
        assert f["episode_policy_name"].asstr()[0] == "pd"
        assert int(f["episode_len"][0]) == 6
        assert f["obs"].chunks == (1, 64, 64, 3)
        assert f["goal_obs"].chunks == (1, 64, 64, 3)

    train = ParkingTrajectoryDataset(path, seq_len=3, split="train")
    val = ParkingTrajectoryDataset(path, seq_len=3, split="val")
    assert len(train) > 0
    assert len(val) > 0
    assert len(train) + len(val) == len(ParkingTrajectoryDataset(path, seq_len=3))


def test_dataset_v1_still_reads(tmp_path):
    path = tmp_path / "train_v1.h5"
    episodes = [make_episode(0), make_episode(1)]
    lengths = np.array([len(ep["actions"]) for ep in episodes], dtype=np.int32)
    with h5py.File(path, "w") as f:
        f.create_dataset("episode_lengths", data=lengths)
        f.create_dataset("obs", data=np.concatenate([ep["obs"] for ep in episodes], axis=0))
        f.create_dataset("actions", data=np.concatenate([ep["actions"] for ep in episodes], axis=0))
        f.create_dataset("goal_obs", data=np.stack([ep["goal_obs"] for ep in episodes], axis=0))

    ds = ParkingTrajectoryDataset(path, seq_len=3)
    assert len(ds) > 0
    sample = ds[0]
    assert sample["obs"].shape == (4, 3, 64, 64)
    assert sample["actions"].shape == (3, 2)


def test_streaming_writer_keeps_metadata_and_chunks(tmp_path):
    path = tmp_path / "streamed_v3.h5"
    episodes = [make_episode(i) for i in range(6)]
    for i, episode in enumerate(episodes):
        episode["episode_id"] = i
        episode["env_variant"] = "near_goal" if i % 2 else "standard"
        episode["obstacle_count"] = i
        episode["static_vehicle_count"] = i // 2

    with EpisodeH5Writer(
        path,
        total_episodes=len(episodes),
        val_fraction=0.33,
        split_seed=4,
        format_version="parking_v3",
        string_fields=["policy_name", "env_variant"],
        numeric_fields=[
            "seed",
            "init_dist",
            "init_ang",
            "final_dist",
            "final_ang",
            "success",
            "episode_len",
            "obstacle_count",
            "static_vehicle_count",
        ],
    ) as writer:
        first_splits = writer.append(episodes[:2])
        second_splits = writer.append(episodes[2:])

    with h5py.File(path, "r") as f:
        assert f.attrs["format_version"] == "parking_v3"
        assert int(f.attrs["total_episodes_written"]) == 6
        assert int(f.attrs["total_transitions_written"]) == 36
        assert f["obs"].chunks == (1, 64, 64, 3)
        assert f["goal_obs"].chunks == (1, 64, 64, 3)
        assert f["actions"].chunks == (4096, 2)
        assert f["episode_env_variant"].asstr()[1] == "near_goal"
        assert int(f["episode_obstacle_count"][5]) == 5
        assert set(f["episode_split"].asstr()[:]) == {"train", "val"}

    assert len(first_splits) == 2
    assert len(second_splits) == 4
    ds = ParkingTrajectoryDataset(path, seq_len=3, split="all")
    assert len(ds) > 0


def test_obs_indexing_accounts_for_extra_frame_per_episode(tmp_path):
    path = tmp_path / "indexed.h5"
    ep0_obs = np.zeros((4, 64, 64, 3), dtype=np.uint8)
    ep1_obs = np.ones((4, 64, 64, 3), dtype=np.uint8) * 100
    ep0_actions = np.zeros((3, 2), dtype=np.float32)
    ep1_actions = np.ones((3, 2), dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("episode_lengths", data=np.array([3, 3], dtype=np.int32))
        f.create_dataset("obs", data=np.concatenate([ep0_obs, ep1_obs], axis=0))
        f.create_dataset("actions", data=np.concatenate([ep0_actions, ep1_actions], axis=0))
        f.create_dataset("goal_obs", data=np.stack([ep0_obs[0], ep1_obs[0]], axis=0))

    ds = ParkingTrajectoryDataset(path, seq_len=1, frame_skip=1)
    ep1_first_window = next(i for i, idx in enumerate(ds.indices) if idx == (1, 0))
    sample = ds[ep1_first_window]
    assert float(sample["obs"][0].mean()) == pytest.approx(100.0 / 255.0)
    assert float(sample["actions"][0].mean()) == 1.0


def test_collect_smoke_writes_readable_train_val(tmp_path):
    pytest.importorskip("highway_env")
    pytest.importorskip("pygame")

    from scripts.collect_parking import _collect_chunk

    path = tmp_path / "collect_smoke.h5"
    policies = ["expert", "reverse", "near_goal_correction", "final_alignment"]
    episodes = _collect_chunk((0, 0, 4, 8, 32, 123, policies))
    write_episodes(path, episodes, val_fraction=0.5, split_seed=0)

    train = ParkingTrajectoryDataset(path, seq_len=2, split="train")
    val = ParkingTrajectoryDataset(path, seq_len=2, split="val")
    assert len(train) > 0
    assert len(val) > 0
    assert train[0]["obs"].shape == (3, 3, 32, 32)
    assert val[0]["actions"].shape == (2, 2)
