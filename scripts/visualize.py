"""Visualisation post-training: courbes de loss + GIFs de rollouts du planner."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_episode_gif(frames: list, goal_img, out_path: Path, fps: int = 10):
    """frames: liste d'images (H, W, 3) uint8. Concatene goal a droite."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composed = []
    for f in frames:
        side = np.concatenate([f, np.full((f.shape[0], 4, 3), 255, dtype=np.uint8), goal_img], axis=1)
        composed.append(side)
    imageio.mimsave(out_path, composed, fps=fps)
    print(f"saved {out_path}")


def plot_loss_curves(log_csv: Path, out_path: Path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = []
    with open(log_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    steps = [int(r["step"]) for r in rows]
    loss_total = [float(r["loss"]) for r in rows]
    loss_pred = [float(r["loss_pred"]) for r in rows]
    loss_sig = [float(r["loss_sigreg"]) for r in rows]
    loss_dec = [float(r["loss_decoder"]) for r in rows]
    has_dec = any(v > 0 for v in loss_dec)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pred + Total", "SIGReg" + (" + Decoder MSE" if has_dec else "")))
    fig.add_trace(go.Scatter(x=steps, y=loss_pred, name="loss_pred", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=loss_total, name="loss_total", line=dict(color="#2ca02c")), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=loss_sig, name="loss_sigreg", line=dict(color="#d62728")), row=1, col=2)
    if has_dec:
        fig.add_trace(go.Scatter(x=steps, y=loss_dec, name="loss_decoder", line=dict(color="#ff7f0e")), row=1, col=2)
    fig.update_layout(title=f"LeWM training curves ({log_csv.name})", template="plotly_white", height=420)
    fig.update_xaxes(title_text="step")
    fig.write_html(out_path)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="runs/last")
    parser.add_argument("--mode", choices=["loss", "all"], default="loss")
    args = parser.parse_args()

    run_dir = Path(args.run)
    log = run_dir / "train_log.csv"
    if log.exists():
        plot_loss_curves(log, run_dir / "loss_curves.html")
    else:
        print(f"no log at {log}")


if __name__ == "__main__":
    main()
