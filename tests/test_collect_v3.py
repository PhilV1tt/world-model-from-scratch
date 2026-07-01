import h5py
import pytest

from src.data import ParkingTrajectoryDataset, write_episodes


def test_parse_env_variants_aliases():
    from scripts.collect_parking import parse_env_variants

    assert parse_env_variants("default,reverse,static") == [
        "standard",
        "reverse_entry",
        "static_vehicles",
    ]
    with pytest.raises(ValueError, match="unknown env variants"):
        parse_env_variants("standard,does_not_exist")


def test_collect_v3_smoke_writes_metadata(tmp_path):
    pytest.importorskip("highway_env")
    pytest.importorskip("pygame")

    from scripts.collect_parking import _collect_chunk, write_v3_episode_metadata

    path = tmp_path / "train_v3_smoke.h5"
    policies = ["expert", "reverse", "near_goal_correction", "final_alignment"]
    variants = ["near_goal", "reverse_entry", "final_alignment", "static_vehicles"]
    episodes = _collect_chunk((0, 0, 4, 8, 32, 123, policies, variants))
    write_episodes(path, episodes, val_fraction=0.5, split_seed=0)
    write_v3_episode_metadata(path, episodes)

    with h5py.File(path, "r") as f:
        assert f.attrs["format_version"] == "parking_v3"
        assert "episode_env_variant" in f
        assert "episode_obstacle_count" in f
        assert "episode_static_vehicle_count" in f
        assert set(f["episode_env_variant"].asstr()[:]) == set(variants)
        assert int(f["episode_obstacle_count"][:].min()) >= 0
        assert int(f["episode_static_vehicle_count"][-1]) >= 1

    train = ParkingTrajectoryDataset(path, seq_len=1, split="train")
    val = ParkingTrajectoryDataset(path, seq_len=1, split="val")
    assert len(train) > 0
    assert len(val) > 0
