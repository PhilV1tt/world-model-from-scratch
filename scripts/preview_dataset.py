"""Visualise des episodes d'un dataset HDF5: GIFs (frames + goal a droite) + grille de samples."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/parking/train.h5")
    parser.add_argument("--out", type=str, default="runs/preview_dataset")
    parser.add_argument("--n-episodes", type=int, default=6)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--upscale", type=int, default=4)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.data, "r") as f:
        lengths = f["episode_lengths"][:]
        starts = np.concatenate([[0], np.cumsum(lengths)])
        n_ep = len(lengths)
        print(f"dataset: {n_ep} episodes, {int(lengths.sum())} transitions")
        idx = np.random.default_rng(0).choice(n_ep, size=min(args.n_episodes, n_ep), replace=False)

        for k, ep_idx in enumerate(idx):
            start, length = int(starts[ep_idx]), int(lengths[ep_idx])
            obs = f["obs"][start : start + length + 1]
            goal = f["goal_obs"][ep_idx]

            sep = np.full((obs.shape[1], 4, 3), 255, dtype=np.uint8)
            frames = []
            for img in obs:
                composed = np.concatenate([img, sep, goal], axis=1)
                if args.upscale > 1:
                    composed = composed.repeat(args.upscale, axis=0).repeat(args.upscale, axis=1)
                frames.append(composed)
            gif_path = out / f"ep{k:02d}_real_len{length}.gif"
            imageio.mimsave(gif_path, frames, fps=args.fps)
            print(f"saved {gif_path}")

        n_grid = 16
        idx2 = np.random.default_rng(1).choice(n_ep, size=n_grid, replace=False)
        thumbs = []
        for ep_idx in idx2:
            goal = f["goal_obs"][ep_idx]
            thumbs.append(goal)
        rows = 4
        cols = 4
        h, w = thumbs[0].shape[:2]
        grid = np.zeros((rows * h + (rows - 1) * 2, cols * w + (cols - 1) * 2, 3), dtype=np.uint8) + 255
        for i, t in enumerate(thumbs):
            r, c = divmod(i, cols)
            grid[r * (h + 2) : r * (h + 2) + h, c * (w + 2) : c * (w + 2) + w] = t
        if args.upscale > 1:
            grid = grid.repeat(args.upscale, axis=0).repeat(args.upscale, axis=1)
        imageio.imwrite(out / "goals_grid.png", grid)
        print(f"saved {out / 'goals_grid.png'}")


if __name__ == "__main__":
    main()
