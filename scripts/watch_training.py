"""Dashboard live: loss training + meilleur rollout du dernier checkpoint.

Le monitor est volontairement leger. Le training garde le MPS, l'eval tourne sur CPU
avec peu d'episodes pour donner un feedback visuel sans bloquer l'apprentissage.
"""
from __future__ import annotations

import csv
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


RUN_DIR = ROOT / "runs" / "last"
LOG_CSV = RUN_DIR / "train_log.csv"
CKPT = RUN_DIR / "ckpt_last.pt"
DASH = ROOT / "runs" / "live_dashboard"
STATUS = DASH / "status.json"
BEST_GIF = DASH / "latest_best.gif"
MEDIAN_GIF = DASH / "latest_median.gif"
WORST_GIF = DASH / "latest_worst.gif"
HISTORY = DASH / "eval_history.csv"
ANIM_DIR = DASH / "animations"
TRAJECTORY_DIR = DASH / "trajectories"
POLL_SECONDS = 0.5
EVAL_EPISODES = 3
EVAL_PERIOD_SECONDS = 0.0
EVAL_DEVICE = "cpu"
EVAL_SEED = 1000
DISCOVERY_SECONDS = 2.0
EXTRA_ANIM_DIRS: list[Path] = []
COLLECT_LOG: Path | None = None
COLLECT_DATASET: Path | None = None


def tail_rows(path: Path, n: int = 500) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-n:]


def summarize_training() -> dict:
    rows = tail_rows(LOG_CSV, 500)
    if not rows:
        return {"has_log": False}
    last = rows[-1]
    losses = [float(r["loss"]) for r in rows if r.get("loss")]
    recent = losses[-50:] if losses else []
    return {
        "has_log": True,
        "step": int(last["step"]),
        "epoch": int(last["epoch"]),
        "loss": float(last["loss"]),
        "loss_pred": float(last["loss_pred"]),
        "loss_sigreg": float(last["loss_sigreg"]),
        "loss_decoder": float(last["loss_decoder"]) if last.get("loss_decoder") else None,
        "loss_ar": float(last["loss_ar"]) if last.get("loss_ar") else None,
        "val_loss": float(last["val_loss"]) if last.get("val_loss") else None,
        "grad_norm": float(last["grad_norm"]) if last.get("grad_norm") else None,
        "loss_ema": float(last["loss_ema"]) if last.get("loss_ema") else None,
        "steps_per_sec": float(last["steps_per_sec"]) if last.get("steps_per_sec") else None,
        "latent_mean": float(last["latent_mean"]) if last.get("latent_mean") else None,
        "latent_std": float(last["latent_std"]) if last.get("latent_std") else None,
        "lr": float(last["lr"]),
        "elapsed": float(last["elapsed"]),
        "recent_loss_mean": float(sum(recent) / len(recent)) if recent else None,
        "recent_loss_min": float(min(recent)) if recent else None,
        "points": [
            {
                "step": int(r["step"]),
                "loss": float(r["loss"]),
                "pred": float(r["loss_pred"]),
                "sig": float(r["loss_sigreg"]),
                "dec": float(r["loss_decoder"]) if r.get("loss_decoder") else None,
                "ar": float(r["loss_ar"]) if r.get("loss_ar") else None,
                "val": float(r["val_loss"]) if r.get("val_loss") else None,
                "lr": float(r["lr"]) if r.get("lr") else None,
                "ema": float(r["loss_ema"]) if r.get("loss_ema") else None,
            }
            for r in rows
        ],
    }


def summarize_eval_history() -> list[dict]:
    rows = tail_rows(HISTORY, 200)
    out = []
    for r in rows:
        try:
            out.append(
                {
                    "time": r["time"],
                    "step": int(float(r["step"])) if r.get("step") else None,
                    "epoch": int(float(r["epoch"])) if r.get("epoch") else None,
                    "best_final_dist": float(r["best_final_dist"]),
                    "best_final_ang": float(r["best_final_ang"]),
                    "success_count": int(float(r["success_count"])),
                    "episodes": int(float(r["episodes"])),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def eval_summary_paths() -> list[Path]:
    paths = []
    for pattern in ("eval_protocol/summary.json", "eval_*/summary.json", "eval*/summary.json"):
        paths.extend(RUN_DIR.glob(pattern))
    seen = {}
    for path in paths:
        if path.exists():
            seen[path.resolve()] = path
    return list(seen.values())


def latest_eval_summary_path() -> Path | None:
    paths = eval_summary_paths()
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def load_latest_eval_summary() -> dict | None:
    path = latest_eval_summary_path()
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        planners = raw.get("planners") or {}
        order = raw.get("planner_order") or list(planners.keys())
        return {
            "path": str(path),
            "mtime": path.stat().st_mtime,
            "ckpt": raw.get("ckpt"),
            "started_at": raw.get("started_at"),
            "finished_at": raw.get("finished_at"),
            "elapsed_sec": raw.get("elapsed_sec"),
            "seed": raw.get("seed"),
            "episodes": raw.get("episodes"),
            "planner_order": order,
            "planners": planners,
        }
    except (OSError, json.JSONDecodeError):
        return None


def sync_animations(limit: int = 12) -> list[str]:
    ANIM_DIR.mkdir(parents=True, exist_ok=True)
    sources = []
    eval_gif_dirs = [path.parent / "gifs" for path in eval_summary_paths()]
    for folder in [RUN_DIR / "plan_results" / "gifs", *eval_gif_dirs, ROOT / "runs" / "preview_train", *EXTRA_ANIM_DIRS]:
        if folder.exists():
            def gif_rank(path: Path):
                name = path.name.lower()
                label_rank = 0 if "best" in name or "top00" in name else 1 if "median" in name else 2 if "worst" in name else 3
                return (label_rank, -path.stat().st_mtime)

            sources.extend(sorted(folder.glob("*.gif"), key=gif_rank))

    copied = []
    for src in sources[:limit]:
        dst = ANIM_DIR / src.name
        try:
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(src, dst)
            copied.append(f"animations/{dst.name}")
        except OSError:
            continue
    return copied


def artifact_role(path: Path) -> str:
    name = path.name.lower()
    if "best" in name or "top00" in name:
        return "best"
    if "median" in name:
        return "median"
    if "worst" in name:
        return "worst"
    return "run"


def is_planned_artifact(path: str | Path) -> bool:
    text = str(path).lower()
    tokens = ("trajectory", "planned", "plan", "rollout", "model", "cem", "top", "latest")
    return any(token in text for token in tokens)


def artifact_rank(item: dict) -> tuple[int, int, str]:
    role_order = {"best": 0, "median": 1, "worst": 2, "run": 3}
    planned_rank = 0 if item.get("planned") else 1
    return planned_rank, role_order.get(item.get("role"), 3), item.get("name", "")


def trajectory_source_dirs() -> list[Path]:
    eval_dirs = [path.parent for path in eval_summary_paths()]
    dirs = [
        RUN_DIR / "plan_results" / "trajectories",
        RUN_DIR / "plan_results" / "rollouts",
        RUN_DIR / "planned_trajectory",
        RUN_DIR / "trajectories",
        *[path / "trajectories" for path in eval_dirs],
        *[path / "rollouts" for path in eval_dirs],
        *EXTRA_ANIM_DIRS,
    ]
    seen = {}
    for folder in dirs:
        try:
            resolved = folder.resolve()
        except OSError:
            continue
        if folder.exists():
            seen[resolved] = folder
    return list(seen.values())


def sync_trajectory_files(limit: int = 12) -> list[dict]:
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    sources = []
    for folder in trajectory_source_dirs():
        sources.extend(folder.glob("*.json"))

    def json_rank(path: Path):
        return (artifact_role(path), -path.stat().st_mtime)

    artifacts = []
    for src in sorted(sources, key=json_rank)[:limit]:
        dst = TRAJECTORY_DIR / src.name
        try:
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(src, dst)
            artifacts.append(
                {
                    "kind": "json",
                    "path": f"trajectories/{dst.name}",
                    "name": dst.name,
                    "role": artifact_role(dst),
                    "planned": True,
                }
            )
        except OSError:
            continue
    return artifacts


def trajectory_artifacts(animation_paths: list[str]) -> list[dict]:
    artifacts = sync_trajectory_files()
    gif_paths = []
    if BEST_GIF.exists():
        gif_paths.append("latest_best.gif")
    if MEDIAN_GIF.exists():
        gif_paths.append("latest_median.gif")
    if WORST_GIF.exists():
        gif_paths.append("latest_worst.gif")
    for path in animation_paths:
        if path not in gif_paths:
            gif_paths.append(path)

    for path in gif_paths:
        name = Path(path).name
        artifacts.append(
            {
                "kind": "gif",
                "path": path,
                "name": name,
                "role": artifact_role(Path(name)),
                "planned": is_planned_artifact(path),
            }
        )
    return sorted(artifacts, key=artifact_rank)


def file_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    return {"mtime": st.st_mtime, "size": st.st_size}


def file_signature(path: Path) -> tuple[float | None, int | None]:
    state = file_state(path)
    if state is None:
        return None, None
    return state["mtime"], state["size"]


def tail_text(path: Path, max_bytes: int = 128_000) -> str:
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def summarize_collection() -> dict | None:
    if COLLECT_LOG is None:
        return None
    text = tail_text(COLLECT_LOG)
    state = file_state(COLLECT_LOG)
    if not text and state is None:
        return {"has_log": False, "path": str(COLLECT_LOG)}

    matches = re.findall(r"collect:\s+(\d+)%.*?\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]", text)
    progress = matches[-1] if matches else None
    done_match = re.search(r"Collected\s+(\d+)\s+episodes\s+\((\d+)\s+transitions\)\s+in\s+([0-9.]+)s\s+=\s+([0-9.]+)\s+ep/s", text)
    wrote_match = re.search(r"Wrote streamed dataset to\s+(.+)", text)
    error_match = re.search(r"(Traceback|Error|error:\s.+)", text)

    current = total = percent = None
    bracket = None
    if progress:
        percent = int(progress[0])
        current = int(progress[1])
        total = int(progress[2])
        bracket = progress[3]
    if done_match:
        current = int(done_match.group(1))
        total = current if total is None else total
        percent = 100

    dataset_path = COLLECT_DATASET
    tmp_dataset_path = None
    if dataset_path is not None:
        tmp_dataset_path = dataset_path.with_name(dataset_path.name + ".tmp")
    dataset_state = file_state(dataset_path) if dataset_path is not None else None
    tmp_dataset_state = file_state(tmp_dataset_path) if tmp_dataset_path is not None else None

    return {
        "has_log": state is not None,
        "path": str(COLLECT_LOG),
        "log_file": state,
        "current": current,
        "total": total,
        "percent": (float(current) / float(total)) if current is not None and total else (percent / 100.0 if percent is not None else None),
        "tqdm_percent": percent,
        "tqdm": bracket,
        "done": done_match is not None or wrote_match is not None,
        "error": error_match.group(0)[:200] if error_match else None,
        "episodes_per_sec": float(done_match.group(4)) if done_match else None,
        "transitions": int(done_match.group(2)) if done_match else None,
        "elapsed_sec": float(done_match.group(3)) if done_match else None,
        "dataset": str(dataset_path) if dataset_path is not None else None,
        "dataset_file": dataset_state,
        "tmp_dataset_file": tmp_dataset_state,
        "wrote_dataset": wrote_match.group(1).strip() if wrote_match else None,
    }


def configure_paths(run_dir: Path, dashboard_dir: Path):
    global RUN_DIR, LOG_CSV, CKPT, DASH, STATUS, BEST_GIF, MEDIAN_GIF, WORST_GIF, HISTORY, ANIM_DIR, TRAJECTORY_DIR
    RUN_DIR = run_dir.resolve()
    LOG_CSV = RUN_DIR / "train_log.csv"
    CKPT = RUN_DIR / "ckpt_last.pt"
    DASH = dashboard_dir.resolve()
    STATUS = DASH / "status.json"
    BEST_GIF = DASH / "latest_best.gif"
    MEDIAN_GIF = DASH / "latest_median.gif"
    WORST_GIF = DASH / "latest_worst.gif"
    HISTORY = DASH / "eval_history.csv"
    ANIM_DIR = DASH / "animations"
    TRAJECTORY_DIR = DASH / "trajectories"


def discover_latest_run_dir() -> Path | None:
    runs_root = ROOT / "runs"
    if not runs_root.exists():
        return None

    candidates = []
    for log_path in runs_root.rglob("train_log.csv"):
        try:
            rel_parts = log_path.relative_to(runs_root).parts
        except ValueError:
            continue
        if any(part.endswith("_dashboard") or part in {"live_dashboard", "runpod_dashboard", "mini_dashboard"} for part in rel_parts):
            continue
        try:
            st = log_path.stat()
        except OSError:
            continue
        candidates.append((st.st_mtime, log_path.parent))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def ensure_live_link(link: Path, target: Path):
    if not target.exists():
        return
    try:
        if link.is_symlink():
            if link.resolve() == target.resolve():
                return
            link.unlink()
        elif link.exists():
            return
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError:
        pass


def write_status(extra: dict | None = None):
    DASH.mkdir(parents=True, exist_ok=True)
    eval_summary_path = latest_eval_summary_path()
    animations = sync_animations()
    status = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(RUN_DIR),
        "training": summarize_training(),
        "collection": summarize_collection(),
        "eval_history": summarize_eval_history(),
        "eval_summary": load_latest_eval_summary(),
        "animations": animations,
        "trajectory_artifacts": trajectory_artifacts(animations),
        "train_log": file_state(LOG_CSV),
        "eval_history_file": file_state(HISTORY),
        "eval_summary_file": file_state(eval_summary_path) if eval_summary_path else None,
        "best_gif_file": file_state(BEST_GIF),
        "median_gif_file": file_state(MEDIAN_GIF),
        "worst_gif_file": file_state(WORST_GIF),
        "checkpoint_mtime": CKPT.stat().st_mtime if CKPT.exists() else None,
        "best_gif": "latest_best.gif" if BEST_GIF.exists() else None,
        "median_gif": "latest_median.gif" if MEDIAN_GIF.exists() else None,
        "worst_gif": "latest_worst.gif" if WORST_GIF.exists() else None,
    }
    if extra:
        status.update(extra)
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def checkpoint_stable(path: Path, quiet_seconds: float = 1.5) -> bool:
    if not path.exists():
        return False
    try:
        st = path.stat()
    except OSError:
        return False
    return st.st_size > 0 and (time.time() - st.st_mtime) >= quiet_seconds


def evaluate_checkpoint(seed: int = 1000) -> dict:
    import imageio.v2 as imageio
    import numpy as np
    import torch

    from scripts.plan import load_model, run_episode
    from src.parking_env import make_parking

    DASH.mkdir(parents=True, exist_ok=True)
    tmp = DASH / "eval_ckpt.pt"
    shutil.copy2(CKPT, tmp)

    device = torch.device(EVAL_DEVICE)
    model, _ = load_model(str(tmp), device)
    results = []
    for ep in range(EVAL_EPISODES):
        env = make_parking(seed=seed + ep, img_size=64)
        out = run_episode(
            model,
            env,
            horizon=5,
            mpc_apply=5,
            max_env_steps=60,
            pop_size=80,
            elites=10,
            iters=3,
            device=device,
            warm_start_pd=True,
            seed=seed + ep,
        )
        env.close()
        out["success"] = out["final_dist"] < 1.5 and out["final_ang"] < 15.0
        results.append(out)

    ranked = sorted(results, key=lambda r: r["final_dist"])
    best = ranked[0]

    def save_rollout_gif(result: dict, path: Path):
        frames = []
        sep = np.full((64, 4, 3), 255, dtype=np.uint8)
        for frame in result["frames"]:
            composed = np.concatenate([frame, sep, result["goal_img"]], axis=1)
            composed = composed.repeat(4, axis=0).repeat(4, axis=1)
            frames.append(composed)
        if frames:
            imageio.mimsave(path, frames, fps=20)

    save_rollout_gif(best, BEST_GIF)
    save_rollout_gif(ranked[len(ranked) // 2], MEDIAN_GIF)
    save_rollout_gif(ranked[-1], WORST_GIF)

    training = summarize_training()
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "step": training.get("step"),
        "epoch": training.get("epoch"),
        "best_final_dist": float(best["final_dist"]),
        "best_final_ang": float(best["final_ang"]),
        "success_count": int(sum(1 for r in results if r["success"])),
        "episodes": len(results),
    }
    write_header = not HISTORY.exists()
    with HISTORY.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row


INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LeWM live</title>
  <link rel="icon" href="data:,">
  <script src="plotly.min.js" onerror="this.onerror=null;this.src='https://cdn.plot.ly/plotly-2.35.2.min.js'"></script>
  <style>
    :root {
      color-scheme: dark;
      --ink: #0f100d;
      --ink-2: #171813;
      --line: #34352c;
      --paper: #f4f1e8;
      --paper-2: #e8e2d4;
      --text: #f3f0e8;
      --muted: #aaa89d;
      --blue: #4ca3ff;
      --orange: #ff6b2c;
      --green: #30b66a;
      --red: #d94a3a;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px) 0 0 / 64px 64px,
        linear-gradient(180deg, #151610 0, #0f100d 38%, #0b0c0a 100%);
      color: var(--text);
      font: 13px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    h1, h2, p { margin: 0; }
    .mono, td.value, code { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }

    .top {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: end;
      padding: 16px 18px 14px;
      border-bottom: 1px solid var(--line);
      background: rgba(15, 16, 13, .88);
    }

    .kicker {
      color: var(--orange);
      font: 700 11px/1 ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      text-transform: uppercase;
    }

    h1 {
      margin-top: 6px;
      font-size: 30px;
      font-weight: 760;
      line-height: 1;
      letter-spacing: 0;
    }

    #run-label {
      display: block;
      margin-top: 7px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: min(860px, 70vw);
    }

    .status {
      display: flex;
      align-items: center;
      gap: 9px;
      min-width: 190px;
      justify-content: flex-end;
      color: var(--muted);
    }

    .led {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: #6f7167;
    }

    .led.live { background: var(--green); box-shadow: 0 0 18px rgba(48, 182, 106, .55); }
    .led.error { background: var(--red); box-shadow: 0 0 18px rgba(217, 74, 58, .55); }

    .app {
      display: grid;
      grid-template-columns: 292px minmax(0, 1fr);
      gap: 14px;
      padding: 14px;
    }

    aside {
      border: 1px solid var(--line);
      background: rgba(23, 24, 19, .92);
      min-width: 0;
    }

    .block-title {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      min-height: 38px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      color: var(--paper);
      font-size: 12px;
      font-weight: 750;
      text-transform: uppercase;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }

    th, td {
      padding: 7px 12px;
      border-bottom: 1px solid rgba(244, 241, 232, .09);
      text-align: left;
      vertical-align: top;
    }

    th {
      width: 47%;
      color: var(--muted);
      font-weight: 520;
    }

    td.value {
      overflow: hidden;
      color: var(--paper);
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    #stable-summary td.value {
      white-space: normal;
      line-height: 1.25;
    }

    .workspace {
      display: grid;
      gap: 14px;
      min-width: 0;
    }

    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(360px, .75fr);
      gap: 14px;
      min-height: 520px;
    }

    .panel {
      min-width: 0;
      border: 1px solid var(--line);
      background: var(--paper);
      color: var(--ink);
      overflow: hidden;
    }

    .panel.dark {
      background: rgba(23, 24, 19, .94);
      color: var(--text);
    }

    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 42px;
      padding: 10px 12px;
      border-bottom: 1px solid rgba(15,16,13,.16);
    }

    .dark .panel-head {
      border-bottom-color: var(--line);
    }

    h2 {
      font-size: 12px;
      font-weight: 780;
      text-transform: uppercase;
    }

    .hint {
      color: #706c62;
      font-size: 12px;
    }

    .dark .hint { color: var(--muted); }

    .plot {
      width: 100%;
      height: calc(100% - 42px);
      min-height: 280px;
    }

    .primary-plot .plot {
      min-height: 476px;
    }

    .cards {
      display: grid;
      grid-template-columns: repeat(7, minmax(120px, 1fr));
      gap: 1px;
      border: 1px solid var(--line);
      background: var(--line);
    }

    .card {
      min-width: 0;
      min-height: 78px;
      padding: 11px 12px;
      background: rgba(23, 24, 19, .96);
    }

    .card span {
      display: block;
      color: var(--muted);
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .card strong {
      display: block;
      overflow: hidden;
      margin-top: 8px;
      color: var(--paper);
      font: 760 21px/1 ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .mini-grid {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 14px;
    }

    .mini-grid .panel {
      height: 300px;
    }

    .rollout {
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 520px;
    }

    .spotlight {
      display: grid;
      gap: 10px;
      padding: 10px;
      align-content: start;
    }

    .spotlight figure:first-child img {
      height: 250px;
    }

    figure {
      margin: 0;
      border: 1px solid rgba(244, 241, 232, .15);
      background: #070807;
    }

    figure.predicted {
      border-color: rgba(76, 163, 255, .62);
      box-shadow: inset 0 0 0 1px rgba(76, 163, 255, .16);
    }

    figcaption {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      padding: 6px 8px;
      border-bottom: 1px solid rgba(244, 241, 232, .12);
      color: var(--muted);
      font-size: 11px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    figcaption span:last-child {
      overflow: hidden;
      text-overflow: ellipsis;
    }

    img {
      display: block;
      width: 100%;
      height: 146px;
      object-fit: contain;
      background: #090a09;
    }

    .route-canvas {
      display: grid;
      place-items: center;
      width: 100%;
      height: 146px;
      background:
        linear-gradient(90deg, rgba(76,163,255,.07) 1px, transparent 1px) 0 0 / 28px 28px,
        linear-gradient(180deg, rgba(76,163,255,.07) 1px, transparent 1px) 0 0 / 28px 28px,
        #070807;
    }

    .spotlight figure:first-child .route-canvas {
      height: 250px;
    }

    .route-canvas svg {
      display: block;
      width: 100%;
      height: 100%;
    }

    .route-kind {
      color: var(--blue);
      font-weight: 760;
    }

    .gallery .animations {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: 10px;
      padding: 10px;
      background: rgba(23, 24, 19, .94);
    }

    .muted { color: var(--muted); }

    @media (max-width: 1180px) {
      .app { grid-template-columns: 1fr; }
      .hero { grid-template-columns: 1fr; }
      aside { order: 2; }
      .cards { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .mini-grid { grid-template-columns: 1fr; }
    }

    @media (max-width: 700px) {
      .top { grid-template-columns: 1fr; align-items: start; }
      h1 { font-size: 23px; }
      .app { padding: 10px; }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .hero { min-height: auto; }
      .primary-plot .plot { min-height: 360px; }
      .mini-grid .panel { height: 260px; }
    }
  </style>
</head>
<body>
  <header class="top">
    <div>
      <p class="kicker">latent world model monitor</p>
      <h1>LeWM live planning</h1>
      <span id="run-label" class="mono">run</span>
    </div>
    <div class="status"><span id="led" class="led"></span><span id="updated" class="mono">waiting</span></div>
  </header>

  <div class="app">
    <aside>
      <div class="block-title"><span>training</span><span class="mono" id="step">.</span></div>
      <table>
        <tr><th>epoch</th><td id="epoch" class="value">.</td></tr>
        <tr><th>loss</th><td id="loss" class="value">.</td></tr>
        <tr><th>loss_pred</th><td id="loss-pred" class="value">.</td></tr>
        <tr><th>loss_sigreg</th><td id="loss-sig" class="value">.</td></tr>
        <tr><th>loss_decoder</th><td id="loss-dec" class="value">.</td></tr>
        <tr><th>loss_ar</th><td id="loss-ar" class="value">.</td></tr>
        <tr><th>val_loss</th><td id="val-loss" class="value">.</td></tr>
        <tr><th>loss_ema</th><td id="loss-ema" class="value">.</td></tr>
        <tr><th>grad_norm</th><td id="grad-norm" class="value">.</td></tr>
        <tr><th>steps/sec</th><td id="steps-sec" class="value">.</td></tr>
        <tr><th>latent_mean</th><td id="latent-mean" class="value">.</td></tr>
        <tr><th>latent_std</th><td id="latent-std" class="value">.</td></tr>
        <tr><th>recent_mean</th><td id="recent" class="value">.</td></tr>
        <tr><th>lr</th><td id="lr" class="value">.</td></tr>
        <tr><th>elapsed</th><td id="elapsed" class="value">.</td></tr>
      </table>
      <div class="block-title"><span>eval</span><span class="mono" id="success">.</span></div>
      <table>
        <tr><th>step</th><td id="eval-step" class="value">.</td></tr>
        <tr><th>best_dist</th><td id="best-dist" class="value">.</td></tr>
        <tr><th>best_ang</th><td id="best-ang" class="value">.</td></tr>
        <tr><th>status</th><td id="error" class="value">.</td></tr>
      </table>
      <div class="block-title"><span>stable eval</span><span class="mono" id="stable-tag">.</span></div>
      <table id="stable-summary">
        <tr><th>summary</th><td class="value">.</td></tr>
      </table>
      <div class="block-title"><span>data collect</span><span class="mono" id="collect-tag">.</span></div>
      <table>
        <tr><th>episodes</th><td id="collect-progress" class="value">.</td></tr>
        <tr><th>status</th><td id="collect-status" class="value">.</td></tr>
        <tr><th>dataset</th><td id="collect-dataset" class="value">.</td></tr>
      </table>
    </aside>

    <main class="workspace">
      <section class="hero">
        <section class="panel primary-plot">
          <div class="panel-head">
            <h2>train loss</h2>
            <span class="hint mono">raw / ema / val</span>
          </div>
          <div id="loss-chart" class="plot"></div>
        </section>

        <section class="panel dark rollout">
          <div class="panel-head">
            <h2>planned trajectory</h2>
            <span class="hint mono">best / median / worst</span>
          </div>
          <div id="spotlight" class="spotlight"></div>
        </section>
      </section>

      <section class="cards">
        <div class="card"><span>train ema</span><strong id="card-ema">.</strong></div>
        <div class="card"><span>val loss</span><strong id="card-val">.</strong></div>
        <div class="card"><span>learning rate</span><strong id="card-lr">.</strong></div>
        <div class="card"><span>grad norm</span><strong id="card-grad">.</strong></div>
        <div class="card"><span>steps / sec</span><strong id="card-speed">.</strong></div>
        <div class="card"><span>latent std</span><strong id="card-latent">.</strong></div>
        <div class="card"><span>eval success</span><strong id="card-success">.</strong></div>
        <div class="card"><span>data collect</span><strong id="card-collect">.</strong></div>
      </section>

      <section class="mini-grid">
        <section class="panel">
          <div class="panel-head"><h2>loss components</h2><span class="hint mono">pred / sig / aux</span></div>
          <div id="component-chart" class="plot"></div>
        </section>
        <section class="panel">
          <div class="panel-head"><h2>learning rate</h2><span class="hint mono">cosine schedule</span></div>
          <div id="lr-chart" class="plot"></div>
        </section>
        <section class="panel">
          <div class="panel-head"><h2>eval history</h2><span class="hint mono">distance / success</span></div>
          <div id="eval-chart" class="plot"></div>
        </section>
      </section>

      <section class="panel dark gallery">
        <div class="panel-head"><h2>trajectory archive</h2><span class="hint mono">planner gifs</span></div>
        <div id="animations" class="animations"></div>
      </section>
    </main>
  </div>

<script>
const $ = (id) => document.getElementById(id);
let lastStatus = null;
let lastTrainKey = '';
let lastComponentKey = '';
let lastLrKey = '';
let lastEvalKey = '';
let lastAnimationsKey = '';

const plotStyle = {
  paper_bgcolor: '#f4f1e8',
  plot_bgcolor: '#f4f1e8',
  font: { family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 12, color: '#111' },
};

function finite(v) {
  return Number.isFinite(Number(v));
}

function fmt(v, digits = 4) {
  if (!finite(v)) return '.';
  return Number(v).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function fmtInt(v) {
  if (!finite(v)) return '.';
  return Math.round(Number(v)).toLocaleString();
}

function fmtPct(v) {
  if (!finite(v)) return '.';
  return `${fmt(100 * Number(v), 1)}%`;
}

function fmtSeconds(v) {
  if (!finite(v)) return '.';
  const sec = Math.round(Number(v));
  const m = Math.floor(sec / 60);
  const s = String(sec % 60).padStart(2, '0');
  return `${m}:${s}`;
}

function fmtBytes(v) {
  if (!finite(v)) return '.';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let n = Number(v);
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i += 1;
  }
  return `${fmt(n, i === 0 ? 0 : 1)} ${units[i]}`;
}

function setText(id, value) {
  $(id).textContent = value;
}

function readNumber(row, key) {
  return row[key] === undefined || row[key] === '' ? null : Number(row[key]);
}

function parseCsv(text) {
  const lines = text.trim().split(/\\r?\\n/).filter(Boolean);
  if (lines.length < 2) return [];
  const headers = lines[0].split(',');
  return lines.slice(1).map((line) => {
    const cells = line.split(',');
    return Object.fromEntries(headers.map((h, i) => [h, cells[i]]));
  });
}

function summarizeTrainRows(rows) {
  if (!rows.length) return { has_log: false, points: [] };
  const tail = rows.slice(-500);
  const last = tail[tail.length - 1];
  const losses = tail.map((r) => Number(r.loss)).filter((v) => Number.isFinite(v));
  const recent = losses.slice(-50);
  return {
    has_log: true,
    step: Number(last.step),
    epoch: Number(last.epoch),
    loss: Number(last.loss),
    loss_pred: Number(last.loss_pred),
    loss_sigreg: Number(last.loss_sigreg),
    loss_decoder: readNumber(last, 'loss_decoder'),
    loss_ar: readNumber(last, 'loss_ar'),
    val_loss: readNumber(last, 'val_loss'),
    grad_norm: readNumber(last, 'grad_norm'),
    loss_ema: readNumber(last, 'loss_ema'),
    steps_per_sec: readNumber(last, 'steps_per_sec'),
    latent_mean: readNumber(last, 'latent_mean'),
    latent_std: readNumber(last, 'latent_std'),
    lr: Number(last.lr),
    elapsed: Number(last.elapsed),
    recent_loss_mean: recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : null,
    recent_loss_min: recent.length ? Math.min(...recent) : null,
    points: tail.map((r) => ({
      step: Number(r.step),
      loss: Number(r.loss),
      pred: Number(r.loss_pred),
      sig: Number(r.loss_sigreg),
      dec: readNumber(r, 'loss_decoder'),
      ar: readNumber(r, 'loss_ar'),
      val: readNumber(r, 'val_loss'),
      lr: readNumber(r, 'lr'),
      ema: readNumber(r, 'loss_ema'),
    })),
  };
}

async function fetchTrainLog() {
  const r = await fetch(`train_log.csv?ts=${Date.now()}`);
  if (!r.ok) return null;
  const text = await r.text();
  const rows = parseCsv(text);
  return { rows, state: { size: text.length, mtime: rows.length ? Number(rows[rows.length - 1].step) : 0 } };
}

function emptyPlot(id, text) {
  const node = $(id);
  if (window.Plotly) Plotly.purge(node);
  node.innerHTML = `<div style="padding:14px;color:#5f5a50">${text}</div>`;
}

function plotLoss(points) {
  if (!window.Plotly) return emptyPlot('loss-chart', 'Plotly loading');
  const rows = (points || []).filter((p) => finite(p.step));
  if (rows.length < 2) return emptyPlot('loss-chart', 'no train_log.csv data');
  const hasEma = rows.some((p) => finite(p.ema));
  const traces = [
    {
      x: rows.map((p) => p.step),
      y: rows.map((p) => p.loss),
      name: hasEma ? 'raw loss' : 'loss',
      type: 'scatter',
      mode: 'lines',
      line: { color: hasEma ? 'rgba(17, 17, 17, .28)' : '#111111', width: hasEma ? 1 : 2.2 },
    },
  ];
  if (hasEma) {
    traces.push({
      x: rows.map((p) => p.step),
      y: rows.map((p) => finite(p.ema) ? p.ema : null),
      name: 'EMA',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#111111', width: 2.5 },
    });
  }
  if (rows.some((p) => finite(p.val))) {
    traces.push({
      x: rows.map((p) => p.step),
      y: rows.map((p) => finite(p.val) ? p.val : null),
      name: 'val loss',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#ff6b2c', width: 1.9 },
    });
  }

  Plotly.react('loss-chart', traces, {
    ...plotStyle,
    margin: { l: 58, r: 24, t: 16, b: 46 },
    hovermode: 'x unified',
    legend: { orientation: 'h', x: 0, y: 1.08 },
    xaxis: { title: 'step', gridcolor: '#ded8ca', zeroline: false },
    yaxis: { title: 'loss', gridcolor: '#ded8ca', rangemode: 'tozero', zeroline: false },
  }, { responsive: true, displaylogo: false });
}

function plotComponents(points) {
  if (!window.Plotly) return emptyPlot('component-chart', 'Plotly loading');
  const rows = (points || []).filter((p) => finite(p.step));
  if (rows.length < 2) return emptyPlot('component-chart', 'no component data');
  const traces = [];
  const addTrace = (key, name, color, yaxis = undefined) => {
    const y = rows.map((p) => finite(p[key]) ? p[key] : null);
    if (y.some((v) => v !== null)) {
      traces.push({ x: rows.map((p) => p.step), y, name, type: 'scatter', mode: 'lines', yaxis, line: { color, width: 1.7 } });
    }
  };
  addTrace('pred', 'loss_pred', '#0f6b8f');
  addTrace('sig', 'loss_sigreg', '#d94a3a', 'y2');
  addTrace('dec', 'loss_decoder', '#7a3fb0', 'y2');
  addTrace('ar', 'loss_ar', '#a45b21', 'y2');
  if (!traces.length) return emptyPlot('component-chart', 'no component data');
  Plotly.react('component-chart', traces, {
    ...plotStyle,
    margin: { l: 48, r: 48, t: 12, b: 34 },
    hovermode: 'x unified',
    legend: { orientation: 'h', x: 0, y: 1.22 },
    xaxis: { title: 'step', gridcolor: '#ded8ca', zeroline: false },
    yaxis: { title: 'pred', gridcolor: '#ded8ca', rangemode: 'tozero', zeroline: false },
    yaxis2: { title: 'aux', overlaying: 'y', side: 'right', rangemode: 'tozero', zeroline: false },
  }, { responsive: true, displaylogo: false });
}

function plotLr(points) {
  if (!window.Plotly) return emptyPlot('lr-chart', 'Plotly loading');
  const rows = (points || []).filter((p) => finite(p.step) && finite(p.lr) && Number(p.lr) > 0);
  if (rows.length < 2) return emptyPlot('lr-chart', 'no LR data');
  Plotly.react('lr-chart', [{
    x: rows.map((p) => p.step),
    y: rows.map((p) => p.lr),
    name: 'lr',
    type: 'scatter',
    mode: 'lines',
    line: { color: '#285943', width: 2.2 },
  }], {
    ...plotStyle,
    margin: { l: 50, r: 18, t: 12, b: 34 },
    hovermode: 'x unified',
    xaxis: { title: 'step', gridcolor: '#ded8ca', zeroline: false },
    yaxis: { title: 'lr', gridcolor: '#ded8ca', type: 'log', zeroline: false },
  }, { responsive: true, displaylogo: false });
}

function plotEval(history) {
  if (!window.Plotly) return emptyPlot('eval-chart', 'Plotly loading');
  const rows = (history || []).filter((r) => finite(r.step));
  if (rows.length < 1) return emptyPlot('eval-chart', 'no eval history yet');
  const successRate = rows.map((r) => r.episodes ? 100 * r.success_count / r.episodes : 0);
  Plotly.react('eval-chart', [
    {
      x: rows.map((r) => r.step),
      y: rows.map((r) => r.best_final_dist),
      name: 'best dist',
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#111111', width: 2 },
    },
    {
      x: rows.map((r) => r.step),
      y: successRate,
      name: 'success %',
      type: 'scatter',
      mode: 'lines+markers',
      yaxis: 'y2',
      line: { color: '#30b66a', width: 1.6 },
    },
  ], {
    ...plotStyle,
    margin: { l: 50, r: 48, t: 12, b: 34 },
    hovermode: 'x unified',
    legend: { orientation: 'h', x: 0, y: 1.22 },
    xaxis: { title: 'step', gridcolor: '#ded8ca', zeroline: false },
    yaxis: { title: 'meters', gridcolor: '#ded8ca', rangemode: 'tozero', zeroline: false },
    yaxis2: { title: 'success %', overlaying: 'y', side: 'right', range: [0, 100], zeroline: false },
  }, { responsive: true, displaylogo: false });
}

function stableLabel(name) {
  if (name === 'model_pd') return 'model+PD';
  return name;
}

function pickStableEval(summary) {
  const planners = summary && summary.planners ? summary.planners : {};
  for (const name of ['model_pd', 'model', 'pd', 'random']) {
    if (planners[name]) return { name, stats: planners[name] };
  }
  return null;
}

function renderStableSummary(summary) {
  const table = $('stable-summary');
  if (!summary || !summary.planners || !Object.keys(summary.planners).length) {
    setText('stable-tag', '.');
    table.innerHTML = '<tr><th>summary</th><td class="value">no summary.json</td></tr>';
    return;
  }
  setText('stable-tag', finite(summary.episodes) ? `${summary.episodes} eps` : 'done');
  const order = summary.planner_order && summary.planner_order.length ? summary.planner_order : Object.keys(summary.planners);
  table.innerHTML = order
    .filter((name) => summary.planners[name])
    .map((name) => {
      const s = summary.planners[name];
      const parts = [
        `${fmtPct(s.success_rate)} ok`,
        `mean ${fmt(s.mean_final_dist, 2)}m`,
        `best ${fmt(s.best_final_dist, 2)}m`,
        `delta ${fmt(s.mean_dist_delta, 2)}m`,
      ];
      return `<tr><th>${stableLabel(name)}</th><td class="value">${parts.join(' | ')}</td></tr>`;
    })
    .join('');
}

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[ch]));
}

function fallbackArtifacts(status) {
  const paths = [];
  if (status.best_gif) paths.push({ kind: 'gif', path: status.best_gif, name: status.best_gif, role: 'best', planned: true });
  if (status.median_gif) paths.push({ kind: 'gif', path: status.median_gif, name: status.median_gif, role: 'median', planned: true });
  if (status.worst_gif) paths.push({ kind: 'gif', path: status.worst_gif, name: status.worst_gif, role: 'worst', planned: true });
  for (const p of status.animations || []) {
    if (!paths.some((item) => item.path === p)) {
      const lower = p.toLowerCase();
      const role = lower.includes('best') || lower.includes('top00') ? 'best' : lower.includes('median') ? 'median' : lower.includes('worst') ? 'worst' : 'run';
      const planned = /trajectory|planned|plan|rollout|model|cem|top|latest/.test(lower);
      paths.push({ kind: 'gif', path: p, name: p.split('/').pop(), role, planned });
    }
  }
  return paths;
}

function normalizePoint(point) {
  if (Array.isArray(point) && point.length >= 2) return [Number(point[0]), Number(point[1])];
  if (point && Array.isArray(point.position) && point.position.length >= 2) return [Number(point.position[0]), Number(point.position[1])];
  if (point && finite(point.x) && finite(point.y)) return [Number(point.x), Number(point.y)];
  return null;
}

function normalizeSeries(raw) {
  if (!Array.isArray(raw)) return [];
  return raw.map(normalizePoint).filter((point) => point && finite(point[0]) && finite(point[1]));
}

function findSeries(data, keys) {
  if (Array.isArray(data)) return normalizeSeries(data);
  if (!data || typeof data !== 'object') return [];
  for (const key of keys) {
    const series = normalizeSeries(data[key]);
    if (series.length >= 2) return series;
  }
  return [];
}

function routePolyline(points, bounds, width, height, pad) {
  const [minX, maxX, minY, maxY] = bounds;
  const dx = Math.max(1e-6, maxX - minX);
  const dy = Math.max(1e-6, maxY - minY);
  return points.map(([x, y]) => {
    const px = pad + ((x - minX) / dx) * (width - 2 * pad);
    const py = height - pad - ((y - minY) / dy) * (height - 2 * pad);
    return `${px.toFixed(1)},${py.toFixed(1)}`;
  }).join(' ');
}

function routeDot(point, bounds, width, height, pad) {
  const [minX, maxX, minY, maxY] = bounds;
  const dx = Math.max(1e-6, maxX - minX);
  const dy = Math.max(1e-6, maxY - minY);
  const [x, y] = point;
  return {
    x: pad + ((x - minX) / dx) * (width - 2 * pad),
    y: height - pad - ((y - minY) / dy) * (height - 2 * pad),
  };
}

function renderTrajectorySvg(data) {
  const planned = findSeries(data, ['planned', 'predicted', 'predicted_trajectory', 'trajectory', 'points', 'path', 'rollout', 'plan']);
  const actual = findSeries(data, ['actual', 'executed', 'observed', 'states', 'real']);
  if (planned.length < 2) return '<span class="muted">no trajectory points</span>';
  const all = [...planned, ...actual];
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  const width = 360;
  const height = 210;
  const pad = 22;
  const bounds = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const plannedLine = routePolyline(planned, bounds, width, height, pad);
  const actualLine = actual.length >= 2 ? routePolyline(actual, bounds, width, height, pad) : '';
  const start = routeDot(planned[0], bounds, width, height, pad);
  const end = routeDot(planned[planned.length - 1], bounds, width, height, pad);
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="planned trajectory">
      <rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>
      ${actualLine ? `<polyline points="${actualLine}" fill="none" stroke="rgba(244,241,232,.42)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"></polyline>` : ''}
      <polyline points="${plannedLine}" fill="none" stroke="#4ca3ff" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"></polyline>
      <polyline points="${plannedLine}" fill="none" stroke="#9fe2ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></polyline>
      <circle cx="${start.x.toFixed(1)}" cy="${start.y.toFixed(1)}" r="6" fill="#f4f1e8"></circle>
      <circle cx="${end.x.toFixed(1)}" cy="${end.y.toFixed(1)}" r="8" fill="#4ca3ff"></circle>
    </svg>
  `;
}

async function hydrateTrajectoryJson() {
  const nodes = document.querySelectorAll('[data-trajectory-json]');
  for (const node of nodes) {
    const path = node.getAttribute('data-trajectory-json');
    const canvas = node.querySelector('.route-canvas');
    if (!path || !canvas || node.dataset.loaded === '1') continue;
    node.dataset.loaded = '1';
    try {
      const response = await fetch(`${path}?ts=${Date.now()}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      canvas.innerHTML = renderTrajectorySvg(data);
    } catch (err) {
      canvas.innerHTML = '<span class="muted">trajectory file unreadable</span>';
    }
  }
}

function artifactFigure(artifact, index) {
  const role = artifact.role || (index === 0 ? 'best' : index === 1 ? 'median' : index === 2 ? 'worst' : 'run');
  const kind = artifact.planned ? 'planned' : 'rollout';
  const name = artifact.name || artifact.path || 'artifact';
  const klass = artifact.planned ? 'predicted' : '';
  const caption = `
    <figcaption>
      <span class="mono route-kind">${esc(role)}</span>
      <span class="mono">${esc(kind)} · ${esc(name)}</span>
    </figcaption>
  `;
  if (artifact.kind === 'json') {
    return `
      <figure class="${klass}" data-trajectory-json="${esc(artifact.path)}">
        ${caption}
        <div class="route-canvas"><span class="muted">loading trajectory</span></div>
      </figure>
    `;
  }
  return `
    <figure class="${klass}">
      ${caption}
      <img src="${esc(artifact.path)}?ts=${Date.now()}" alt="${esc(name)}">
    </figure>
  `;
}

function renderAnimations(status) {
  const artifacts = (status.trajectory_artifacts && status.trajectory_artifacts.length)
    ? status.trajectory_artifacts
    : fallbackArtifacts(status);
  const preferred = artifacts.filter((artifact) => artifact.planned);
  const spotlight = (preferred.length ? preferred : artifacts).slice(0, 3);
  const spotlightPaths = new Set(spotlight.map((artifact) => artifact.path));
  const archive = artifacts.filter((artifact) => !spotlightPaths.has(artifact.path));
  $('spotlight').innerHTML = spotlight.length
    ? spotlight.map(artifactFigure).join('')
    : '<div class="muted">no rollout artifacts</div>';
  $('animations').innerHTML = archive.length
    ? archive.map(artifactFigure).join('')
    : '<div class="muted">no extra artifacts</div>';
  hydrateTrajectoryJson();
}

function update(status) {
  lastStatus = status;
  const t = status.training || {};
  const e = status.eval || {};
  const c = status.collection || {};
  const trainFile = status.train_log || {};
  const evalFile = status.eval_history_file || {};
  const gifFile = status.best_gif_file || {};
  const medianGifFile = status.median_gif_file || {};
  const worstGifFile = status.worst_gif_file || {};
  const stableEval = pickStableEval(status.eval_summary);
  $('led').className = `led ${(t.has_log || c.has_log) ? 'live' : ''}`;
  setText('run-label', status.run_dir || 'run');
  setText('updated', status.updated_at || 'waiting');
  setText('step', fmtInt(t.step));
  setText('epoch', fmtInt(t.epoch));
  setText('loss', fmt(t.loss));
  setText('loss-pred', fmt(t.loss_pred));
  setText('loss-sig', fmt(t.loss_sigreg));
  setText('loss-dec', fmt(t.loss_decoder));
  setText('loss-ar', fmt(t.loss_ar));
  setText('val-loss', fmt(t.val_loss));
  setText('loss-ema', fmt(t.loss_ema));
  setText('grad-norm', fmt(t.grad_norm, 2));
  setText('steps-sec', fmt(t.steps_per_sec, 2));
  setText('latent-mean', fmt(t.latent_mean, 3));
  setText('latent-std', fmt(t.latent_std, 3));
  setText('recent', fmt(t.recent_loss_mean));
  setText('lr', finite(t.lr) ? Number(t.lr).toExponential(2) : '.');
  setText('elapsed', fmtSeconds(t.elapsed));
  setText('eval-step', fmtInt(e.step));
  setText('best-dist', finite(e.best_final_dist) ? `${fmt(e.best_final_dist, 3)} m` : '.');
  setText('best-ang', finite(e.best_final_ang) ? `${fmt(e.best_final_ang, 2)} deg` : '.');
  setText('success', finite(e.success_count) ? `${e.success_count}/${e.episodes}` : (stableEval ? `${fmtPct(stableEval.stats.success_rate)} ${stableLabel(stableEval.name)}` : '.'));
  setText('error', status.eval_running ? 'eval running' : (status.eval_error || '.'));
  const collectDone = c.done;
  const collectCount = finite(c.current) && finite(c.total) ? `${fmtInt(c.current)}/${fmtInt(c.total)}` : '.';
  const collectPercent = finite(c.percent) ? fmtPct(c.percent) : '.';
  const tmpSize = c.tmp_dataset_file && finite(c.tmp_dataset_file.size) ? fmtBytes(c.tmp_dataset_file.size) : null;
  const finalSize = c.dataset_file && finite(c.dataset_file.size) ? fmtBytes(c.dataset_file.size) : null;
  setText('collect-tag', c.has_log ? (collectDone ? 'done' : collectPercent) : '.');
  setText('collect-progress', collectCount);
  setText('collect-status', c.error ? c.error : (collectDone ? 'done' : (c.has_log ? 'running' : '.')));
  setText('collect-dataset', finalSize || tmpSize || '.');
  setText('card-ema', fmt(t.loss_ema || t.recent_loss_mean));
  setText('card-val', fmt(t.val_loss));
  setText('card-lr', finite(t.lr) ? Number(t.lr).toExponential(2) : '.');
  setText('card-grad', fmt(t.grad_norm, 2));
  setText('card-speed', fmt(t.steps_per_sec, 2));
  setText('card-latent', fmt(t.latent_std, 3));
  setText('card-success', finite(e.success_count) ? `${e.success_count}/${e.episodes}` : (stableEval ? `${fmtPct(stableEval.stats.success_rate)} ${stableLabel(stableEval.name)}` : '.'));
  setText('card-collect', c.has_log ? (collectDone ? 'done' : collectPercent) : '.');
  renderStableSummary(status.eval_summary);

  const trainKey = `${trainFile.mtime || 0}:${trainFile.size || 0}:${t.step || 0}`;
  const evalKey = `${evalFile.mtime || 0}:${evalFile.size || 0}:${(status.eval_history || []).length}`;
  const animationsKey = JSON.stringify([status.best_gif, status.median_gif, status.worst_gif, gifFile.mtime || 0, medianGifFile.mtime || 0, worstGifFile.mtime || 0, status.animations || [], status.trajectory_artifacts || []]);
  if (!window.Plotly || trainKey !== lastTrainKey) {
    plotLoss(t.points || []);
    if (window.Plotly) lastTrainKey = trainKey;
  }
  if (!window.Plotly || trainKey !== lastComponentKey) {
    plotComponents(t.points || []);
    if (window.Plotly) lastComponentKey = trainKey;
  }
  if (!window.Plotly || trainKey !== lastLrKey) {
    plotLr(t.points || []);
    if (window.Plotly) lastLrKey = trainKey;
  }
  if (!window.Plotly || evalKey !== lastEvalKey) {
    plotEval(status.eval_history || []);
    if (window.Plotly) lastEvalKey = evalKey;
  }
  if (animationsKey !== lastAnimationsKey) {
    renderAnimations(status);
    lastAnimationsKey = animationsKey;
  }
}

async function refresh() {
  try {
    const [statusResult, trainLog] = await Promise.all([
      fetch(`status.json?ts=${Date.now()}`).then((r) => r.ok ? r.json() : {}),
      fetchTrainLog(),
    ]);
    const status = statusResult || {};
    if (trainLog && trainLog.rows.length) {
      status.training = summarizeTrainRows(trainLog.rows);
      status.train_log = trainLog.state;
      status.updated_at = `step ${status.training.step}`;
    }
    update(status);
  } catch (err) {
    $('led').className = 'led error';
    setText('updated', 'status error');
    setText('error', err.message || String(err));
  }
}

window.addEventListener('resize', () => {
  if (lastStatus) {
    Plotly.Plots.resize($('loss-chart'));
    Plotly.Plots.resize($('component-chart'));
    Plotly.Plots.resize($('lr-chart'));
    Plotly.Plots.resize($('eval-chart'));
  }
});
setInterval(refresh, 1000);
refresh();
</script>
</body>
</html>
"""


def write_index():
    DASH.mkdir(parents=True, exist_ok=True)
    ensure_live_link(DASH / "train_log.csv", LOG_CSV)
    ensure_live_link(DASH / "plan_gifs", RUN_DIR / "plan_results" / "gifs")
    ensure_live_link(DASH / "preview_train", ROOT / "runs" / "preview_train")
    try:
        import plotly

        plotly_js = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
        if plotly_js.exists():
            shutil.copy2(plotly_js, DASH / "plotly.min.js")
    except Exception:
        pass
    (DASH / "index.html").write_text(INDEX_HTML, encoding="utf-8")


def main():
    global EVAL_EPISODES, EVAL_PERIOD_SECONDS, EVAL_DEVICE, EVAL_SEED, EXTRA_ANIM_DIRS, COLLECT_LOG, COLLECT_DATASET
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None, help="default: latest run with train_log.csv")
    parser.add_argument("--follow-latest", action="store_true", help="switch to the newest train_log.csv while running")
    parser.add_argument("--dashboard-dir", type=str, default=str(ROOT / "runs" / "live_dashboard"))
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-period", type=float, default=0.0, help="minimum seconds between checkpoint evals")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--eval-existing", action="store_true", help="evaluate an already-present checkpoint on startup")
    parser.add_argument("--seed-gif-dir", action="append", default=[], help="extra gif folder to display while waiting for eval")
    parser.add_argument("--collect-log", type=str, default=None, help="optional collection log to display while dataset generation is running")
    parser.add_argument("--collect-dataset", type=str, default=None, help="optional dataset path produced by --collect-log")
    args = parser.parse_args()

    requested_run_dir = Path(args.run_dir) if args.run_dir else None
    follow_latest = args.follow_latest or requested_run_dir is None
    initial_run_dir = discover_latest_run_dir() if follow_latest else requested_run_dir
    if initial_run_dir is None:
        initial_run_dir = ROOT / "runs" / "last"

    configure_paths(initial_run_dir, Path(args.dashboard_dir))
    EVAL_EPISODES = args.eval_episodes
    EVAL_PERIOD_SECONDS = args.eval_period
    EVAL_DEVICE = args.device
    EVAL_SEED = args.eval_seed
    EXTRA_ANIM_DIRS = [Path(p).resolve() for p in args.seed_gif_dir]
    COLLECT_LOG = Path(args.collect_log).resolve() if args.collect_log else None
    COLLECT_DATASET = Path(args.collect_dataset).resolve() if args.collect_dataset else None

    write_index()
    last_eval_mtime = 0.0 if args.eval_existing else CKPT.stat().st_mtime if CKPT.exists() else 0.0
    history = summarize_eval_history()
    last_eval = history[-1] if history else None
    eval_error = None
    last_status_key = None
    last_status_write = 0.0
    last_eval_wall = 0.0
    last_discovery = 0.0
    while True:
        now = time.time()
        if follow_latest and now - last_discovery >= DISCOVERY_SECONDS:
            latest_run_dir = discover_latest_run_dir()
            last_discovery = now
            if latest_run_dir and latest_run_dir.resolve() != RUN_DIR:
                configure_paths(latest_run_dir, DASH)
                write_index()
                last_eval_mtime = 0.0 if args.eval_existing else CKPT.stat().st_mtime if CKPT.exists() else 0.0
                history = summarize_eval_history()
                last_eval = history[-1] if history else None
                last_status_key = None

        eval_summary_path = latest_eval_summary_path()
        status_key = (
            file_signature(LOG_CSV),
            file_signature(HISTORY),
            file_signature(eval_summary_path) if eval_summary_path else (None, None),
            file_signature(BEST_GIF),
            file_signature(MEDIAN_GIF),
            file_signature(WORST_GIF),
            file_signature(CKPT),
        )
        extra = {"eval": last_eval, "eval_running": False}
        if eval_error:
            extra["eval_error"] = eval_error
        if status_key != last_status_key or now - last_status_write >= 2.0:
            write_status(extra)
            last_status_key = status_key
            last_status_write = now

        if EVAL_EPISODES > 0 and CKPT.exists():
            mtime = CKPT.stat().st_mtime
            log_state = file_state(LOG_CSV)
            log_active = bool(log_state and time.time() - log_state["mtime"] < 3.0)
            period_ok = EVAL_PERIOD_SECONDS <= 0 or now - last_eval_wall >= EVAL_PERIOD_SECONDS
            if mtime > last_eval_mtime and period_ok and not log_active and checkpoint_stable(CKPT):
                write_status({"eval": last_eval, "eval_running": True})
                try:
                    last_eval = evaluate_checkpoint(seed=EVAL_SEED)
                    eval_error = None
                except Exception as exc:
                    eval_error = repr(exc)
                last_eval_mtime = mtime
                last_eval_wall = now
                last_status_key = None

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
