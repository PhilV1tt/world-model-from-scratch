"""Dataset HDF5 de trajectoires parking-v0 pour entrainer LeWM."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
DEFAULT_STRING_FIELDS = ["policy_name"]
DEFAULT_NUMERIC_FIELDS = [
    "seed",
    "init_dist",
    "init_ang",
    "final_dist",
    "final_ang",
    "success",
    "strict_success",
    "lateral_offset_m",
    "along_offset_m",
    "abs_lateral_offset_m",
    "abs_along_offset_m",
    "heading_error_deg",
    "abs_heading_error_deg",
    "speed_mps",
    "collided",
    "episode_len",
]
INT_FIELDS = {"seed", "success", "strict_success", "collided", "episode_len", "obstacle_count", "static_vehicle_count"}


def _decode_strings(values) -> np.ndarray:
    return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values])


def _deterministic_episode_split(n_episodes: int, val_fraction: float, seed: int) -> np.ndarray:
    splits = np.full(n_episodes, "train", dtype=object)
    if val_fraction <= 0 or n_episodes == 0:
        return splits
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(n_episodes * val_fraction)))
    n_val = min(n_episodes, n_val)
    val_idx = rng.choice(n_episodes, size=n_val, replace=False)
    splits[val_idx] = "val"
    return splits


def _dataset_name_for_episode_field(key: str) -> str:
    return "episode_len" if key == "episode_len" else f"episode_{key}"


def _numeric_episode_array(episodes: list[dict], key: str, lengths: np.ndarray) -> np.ndarray:
    if key == "episode_len":
        return lengths.astype(np.int64)
    values = [ep.get(key) for ep in episodes]
    fill = 0 if key in INT_FIELDS else np.nan
    arr = np.array([fill if v is None else v for v in values])
    return arr.astype(np.int64 if key in INT_FIELDS else np.float32)


class EpisodeH5Writer:
    """Append-only HDF5 writer for large parking datasets.

    The training dataset is still one HDF5 file, but collection can stream worker
    chunks into it instead of keeping every rendered frame in Python memory.
    """

    def __init__(
        self,
        h5_path: str | Path,
        *,
        total_episodes: int,
        val_fraction: float = 0.0,
        split_seed: int = 0,
        format_version: str = "parking_v2",
        string_fields: list[str] | None = None,
        numeric_fields: list[str] | None = None,
    ):
        self.h5_path = Path(h5_path)
        self.total_episodes = int(total_episodes)
        self.val_fraction = float(val_fraction)
        self.split_seed = int(split_seed)
        self.format_version = format_version
        self.string_fields = string_fields or list(DEFAULT_STRING_FIELDS)
        self.numeric_fields = numeric_fields or list(DEFAULT_NUMERIC_FIELDS)
        self.splits = _deterministic_episode_split(
            self.total_episodes,
            val_fraction=self.val_fraction,
            seed=self.split_seed,
        )

        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.h5_path, "w")
        self.f.attrs["format_version"] = self.format_version
        self.f.attrs["val_fraction"] = self.val_fraction
        self.f.attrs["split_seed"] = self.split_seed
        self.f.attrs["total_episodes_target"] = self.total_episodes
        self._created = False
        self._episode_count = 0
        self._obs_count = 0
        self._action_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _create_datasets(self, first_episode: dict):
        obs_shape = tuple(first_episode["obs"].shape[1:])
        goal_shape = tuple(first_episode["goal_obs"].shape)
        action_dim = int(first_episode["actions"].shape[-1])
        obs_chunks = (1,) + obs_shape
        goal_chunks = (1,) + goal_shape

        self.f.create_dataset("episode_lengths", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(1024,))
        self.f.create_dataset(
            "obs",
            shape=(0,) + obs_shape,
            maxshape=(None,) + obs_shape,
            dtype=np.uint8,
            compression="lzf",
            chunks=obs_chunks,
        )
        self.f.create_dataset(
            "actions",
            shape=(0, action_dim),
            maxshape=(None, action_dim),
            dtype=np.float32,
            chunks=(4096, action_dim),
        )
        self.f.create_dataset(
            "goal_obs",
            shape=(0,) + goal_shape,
            maxshape=(None,) + goal_shape,
            dtype=np.uint8,
            compression="lzf",
            chunks=goal_chunks,
        )
        self.f.create_dataset(
            "episode_split",
            shape=(0,),
            maxshape=(None,),
            dtype=STRING_DTYPE,
            chunks=(1024,),
        )
        for key in self.string_fields:
            self.f.create_dataset(
                _dataset_name_for_episode_field(key),
                shape=(0,),
                maxshape=(None,),
                dtype=STRING_DTYPE,
                chunks=(1024,),
            )
        for key in self.numeric_fields:
            dtype = np.int64 if key in INT_FIELDS else np.float32
            self.f.create_dataset(
                _dataset_name_for_episode_field(key),
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=(1024,),
            )
        self._created = True

    def append(self, episodes: list[dict]) -> np.ndarray:
        if not episodes:
            return np.array([], dtype=object)
        if not self._created:
            self._create_datasets(episodes[0])

        lengths = np.array([len(ep["actions"]) for ep in episodes], dtype=np.int32)
        obs_all = np.concatenate([ep["obs"] for ep in episodes], axis=0)
        actions_all = np.concatenate([ep["actions"] for ep in episodes], axis=0).astype(np.float32)
        goal_obs_all = np.stack([ep["goal_obs"] for ep in episodes], axis=0)
        episode_ids = np.array(
            [ep.get("episode_id", self._episode_count + i) for i, ep in enumerate(episodes)],
            dtype=np.int64,
        )
        if episode_ids.min() < 0 or episode_ids.max() >= self.total_episodes:
            raise ValueError("episode_id outside total_episodes range")
        split_values = self.splits[episode_ids]

        ep0 = self._episode_count
        ep1 = ep0 + len(episodes)
        obs0 = self._obs_count
        obs1 = obs0 + len(obs_all)
        action0 = self._action_count
        action1 = action0 + len(actions_all)

        self.f["episode_lengths"].resize((ep1,))
        self.f["episode_lengths"][ep0:ep1] = lengths
        self.f["obs"].resize((obs1,) + self.f["obs"].shape[1:])
        self.f["obs"][obs0:obs1] = obs_all
        self.f["actions"].resize((action1, self.f["actions"].shape[1]))
        self.f["actions"][action0:action1] = actions_all
        self.f["goal_obs"].resize((ep1,) + self.f["goal_obs"].shape[1:])
        self.f["goal_obs"][ep0:ep1] = goal_obs_all
        self.f["episode_split"].resize((ep1,))
        self.f["episode_split"][ep0:ep1] = split_values.astype(STRING_DTYPE)

        for key in self.string_fields:
            name = _dataset_name_for_episode_field(key)
            values = np.array([str(ep.get(key, "")) for ep in episodes], dtype=STRING_DTYPE)
            self.f[name].resize((ep1,))
            self.f[name][ep0:ep1] = values
        for key in self.numeric_fields:
            name = _dataset_name_for_episode_field(key)
            values = _numeric_episode_array(episodes, key, lengths)
            self.f[name].resize((ep1,))
            self.f[name][ep0:ep1] = values

        self._episode_count = ep1
        self._obs_count = obs1
        self._action_count = action1
        return split_values

    def close(self):
        if self.f:
            self.f.attrs["total_episodes_written"] = self._episode_count
            self.f.attrs["total_transitions_written"] = self._action_count
            self.f.close()
            self.f = None


class ParkingTrajectoryDataset(Dataset):
    """Echantillonne des sub-trajectoires de longueur seq_len + 1 frames consecutives.

    Format HDF5 :
      /episode_lengths : (N_ep,) int  -- longueur de chaque episode
      /obs             : (N_total, H, W, 3) uint8
      /actions         : (N_total, 2) float32
      /goal_obs        : (N_ep, H, W, 3) uint8

    seq_len = nombre de transitions (T) dans un sample.
    Le sample retourne T+1 frames + T actions.
    """

    def __init__(
        self,
        h5_path: str | Path,
        seq_len: int = 3,
        frame_skip: int = 1,
        img_size: int = 64,
        split: str | None = None,
        val_fraction: float = 0.0,
        split_seed: int = 0,
    ):
        self.h5_path = Path(h5_path)
        self.seq_len = seq_len
        self.frame_skip = frame_skip
        self.img_size = img_size
        self.split = split
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        self._h5 = None

        with h5py.File(self.h5_path, "r") as f:
            self.episode_lengths = f["episode_lengths"][:]
            self.action_starts = np.concatenate([[0], np.cumsum(self.episode_lengths)])
            self.obs_starts = np.concatenate([[0], np.cumsum(self.episode_lengths + 1)])
            if "episode_split" in f:
                self.episode_splits = _decode_strings(f["episode_split"][:])
            else:
                self.episode_splits = _deterministic_episode_split(
                    len(self.episode_lengths),
                    val_fraction=val_fraction,
                    seed=split_seed,
                )

        self.min_actions = seq_len * frame_skip
        self.indices = []
        for ep_idx, ep_len in enumerate(self.episode_lengths):
            if split and split != "all" and self.episode_splits[ep_idx] != split:
                continue
            if ep_len < self.min_actions:
                continue
            for t in range(ep_len - self.min_actions + 1):
                self.indices.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self.indices)

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __getitem__(self, idx: int) -> dict:
        self._open()
        ep_idx, t = self.indices[idx]
        obs_start = self.obs_starts[ep_idx] + t
        action_start = self.action_starts[ep_idx] + t

        offsets = np.arange(self.seq_len + 1) * self.frame_skip
        frame_idx = obs_start + offsets

        obs = self._h5["obs"][frame_idx]
        action_idx = action_start + offsets[:-1]
        actions = self._h5["actions"][action_idx]
        goal_obs = self._h5["goal_obs"][ep_idx]

        obs_t = torch.from_numpy(obs).permute(0, 3, 1, 2).float() / 255.0
        goal_t = torch.from_numpy(goal_obs).permute(2, 0, 1).float() / 255.0
        actions_t = torch.from_numpy(actions).float()

        return {"obs": obs_t, "actions": actions_t, "goal_obs": goal_t}


def write_episodes(
    h5_path: str | Path,
    episodes: list[dict],
    val_fraction: float = 0.0,
    split_seed: int = 0,
) -> np.ndarray:
    """Ecrit une liste d'episodes dans un fichier HDF5 (overwrite)."""
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    lengths = np.array([len(ep["actions"]) for ep in episodes], dtype=np.int32)
    obs_all = np.concatenate([ep["obs"] for ep in episodes], axis=0)
    actions_all = np.concatenate([ep["actions"] for ep in episodes], axis=0)
    goal_obs_all = np.stack([ep["goal_obs"] for ep in episodes], axis=0)
    splits = _deterministic_episode_split(len(episodes), val_fraction=val_fraction, seed=split_seed)

    with h5py.File(h5_path, "w") as f:
        f.attrs["format_version"] = "parking_v2"
        f.attrs["val_fraction"] = float(val_fraction)
        f.attrs["split_seed"] = int(split_seed)
        f.create_dataset("episode_lengths", data=lengths)
        obs_chunks = (1,) + obs_all.shape[1:]
        goal_chunks = (1,) + goal_obs_all.shape[1:]
        f.create_dataset("obs", data=obs_all, compression="lzf", chunks=obs_chunks)
        f.create_dataset("actions", data=actions_all)
        f.create_dataset("goal_obs", data=goal_obs_all, compression="lzf", chunks=goal_chunks)
        f.create_dataset("episode_split", data=splits.astype(STRING_DTYPE))

        string_fields = ["policy_name"]
        numeric_fields = list(DEFAULT_NUMERIC_FIELDS)
        for key in string_fields:
            values = [str(ep.get(key, "")) for ep in episodes]
            if any(values):
                f.create_dataset(f"episode_{key}", data=np.array(values, dtype=STRING_DTYPE))
        for key in numeric_fields:
            if key == "episode_len":
                values = lengths
            else:
                values = [ep.get(key) for ep in episodes]
            if any(v is not None for v in values):
                fill = 0 if key in {"seed", "success", "episode_len"} else np.nan
                arr = np.array([fill if v is None else v for v in values])
                if key in {"seed", "success", "episode_len"}:
                    arr = arr.astype(np.int64)
                else:
                    arr = arr.astype(np.float32)
                name = "episode_len" if key == "episode_len" else f"episode_{key}"
                f.create_dataset(name, data=arr)

    return splits
