"""Live pygame viewer pour parking-v0 - grille NxN d'episodes en parallele.

- Au depart: PD heuristic (en attendant le 1er ckpt) -> bordure jaune
- Des qu'un ckpt apparait dans runs/last/ckpt_last.pt: LeWM + CEM -> bordure verte
- Recharge ckpt automatiquement a chaque epoch
- Quitter: Cmd+W ou Esc
"""
from __future__ import annotations

import os as _os
_os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import sys
from pathlib import Path

import numpy as np
import pygame
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.parking_env import (
    make_parking, render_pixel, render_goal_pixel,
    goal_distance, HeuristicPDPolicy,
)
from src.lewm import LeWM, LeWMConfig

CKPT = ROOT / "runs" / "last" / "ckpt_last.pt"

GRID = 1          # demo: une scene en grand, plus lisible
TILE = 640
GAP = 10
TOP_BAR = 72
MAX_STEPS = 60
FPS = 30
ANIM_SUBSTEPS = 4
PLAN_HORIZON = 5
PLAN_POP = 256
PLAN_ELITES = 32
PLAN_ITERS = 6
TRAIL_MAX_POINTS = 80

BG = (10, 12, 14)
PANEL = (16, 19, 23)
LINE = (42, 49, 56)
MUTED = (150, 158, 162)
TEXT = (242, 239, 230)
GREEN = (118, 217, 140)
CYAN = (102, 200, 212)
AMBER = (229, 198, 93)


def to_t_batch(imgs: np.ndarray, device: torch.device) -> torch.Tensor:
    """imgs: (N, H, W, 3) uint8 -> (N, 3, H, W) float on device."""
    t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div_(255.0)
    return t.to(device)


@torch.no_grad()
def cem_plan_batch(model, z_init, z_goal, horizon=PLAN_HORIZON, pop=PLAN_POP, elites=PLAN_ELITES, iters=PLAN_ITERS, warm_start=None):
    """Plan en parallele pour B envs en un seul forward batche.
    z_init, z_goal: (B, D)  ->  return: (B, horizon, A) numpy.
    """
    B, D = z_init.shape
    device = z_init.device
    A = 2
    if warm_start is not None:
        mu = torch.from_numpy(warm_start.astype(np.float32)).to(device)
        sigma = torch.full((B, horizon, A), 0.45, device=device)
    else:
        mu = torch.zeros(B, horizon, A, device=device)
        sigma = torch.full((B, horizon, A), 1.0, device=device)
    for _ in range(iters):
        # (B, pop, horizon, A)
        actions = mu.unsqueeze(1) + sigma.unsqueeze(1) * torch.randn(B, pop, horizon, A, device=device)
        actions = actions.clamp(-1, 1)
        a_flat = actions.reshape(B * pop, horizon, A)
        z_flat = z_init.unsqueeze(1).expand(B, pop, D).reshape(B * pop, D)
        z_final = model.rollout_latents(z_flat, a_flat)[:, -1, :].reshape(B, pop, D)
        cost = ((z_final - z_goal.unsqueeze(1)) ** 2).sum(dim=2)  # (B, pop)
        topk = cost.topk(elites, largest=False, dim=1).indices  # (B, elites)
        idx = topk.unsqueeze(-1).unsqueeze(-1).expand(B, elites, horizon, A)
        elite = torch.gather(actions, 1, idx)  # (B, elites, horizon, A)
        mu = elite.mean(dim=1)
        sigma = elite.std(dim=1).clamp(min=0.05)
    return mu.cpu().numpy()


def maybe_reload(model, last_mtime, device):
    if not CKPT.exists():
        return model, last_mtime, False, -1
    try:
        mt = CKPT.stat().st_mtime
    except OSError:
        return model, last_mtime, False, -1
    if mt <= last_mtime + 0.5:
        return model, last_mtime, False, -1
    try:
        ckpt = torch.load(CKPT, map_location=device, weights_only=False)
        cfg = LeWMConfig(**ckpt["cfg"])
        nm = LeWM(cfg).to(device)
        nm.load_state_dict(ckpt["model"])
        nm.eval()
        epoch = int(ckpt.get("epoch", -1))
        return nm, mt, True, epoch
    except Exception:
        return model, last_mtime, False, -1


def array_to_surface(arr: np.ndarray) -> pygame.Surface:
    """arr: (H, W, 3) uint8 RGB -> pygame Surface, contiguous, no swapaxes weirdness."""
    arr = np.ascontiguousarray(arr)
    h, w, _ = arr.shape
    return pygame.image.frombuffer(arr.tobytes(), (w, h), "RGB")


def vehicle_state(env) -> dict:
    v = env.unwrapped.controlled_vehicles[0]
    return {
        "position": v.position.copy(),
        "heading": float(v.heading),
        "speed": float(v.speed),
    }


def set_vehicle_state(env, state: dict):
    v = env.unwrapped.controlled_vehicles[0]
    v.position = state["position"].copy()
    v.heading = float(state["heading"])
    v.speed = float(state["speed"])


def interp_heading(a: float, b: float, t: float) -> float:
    diff = np.arctan2(np.sin(b - a), np.cos(b - a))
    return float(a + t * diff)


def interp_state(a: dict, b: dict, t: float) -> dict:
    return {
        "position": (1.0 - t) * a["position"] + t * b["position"],
        "heading": interp_heading(a["heading"], b["heading"], t),
        "speed": (1.0 - t) * a["speed"] + t * b["speed"],
    }


def render_interpolated(env, start_state: dict | None, end_state: dict | None, alpha: float):
    if start_state is None or end_state is None:
        return env.render()
    current = vehicle_state(env)
    try:
        set_vehicle_state(env, interp_state(start_state, end_state, alpha))
        return env.render()
    finally:
        set_vehicle_state(env, current)


def predict_plan_world_points(state: dict, plan: np.ndarray) -> list[np.ndarray]:
    """Approx visuelle du plan choisi, sans step l'env affiche.

    Les actions highway-env sont dans l ordre [acceleration, steering].
    C est une projection visuelle isolee: elle sert a rendre lisible le plan
    pendant que CEM recalcule, sans toucher aux etats internes de highway-env.
    """
    pos = state["position"].copy()
    heading = float(state["heading"])
    speed = float(state["speed"])
    points = [pos.copy()]
    dt = 1.0 / 5.0
    for accel, steer in plan:
        speed = float(np.clip(speed + float(accel) * 2.5 * dt, -4.0, 6.0))
        heading = float(heading + float(steer) * 0.75 * dt)
        pos = pos + np.array([np.cos(heading), np.sin(heading)]) * speed * dt
        points.append(pos.copy())
    return points


def planned_world_points(env, plan: np.ndarray) -> list[np.ndarray]:
    return predict_plan_world_points(vehicle_state(env), plan)


def world_points_to_tile_pixels(env, points: list[np.ndarray] | None, frame_shape, start_idx: int = 0):
    if not points:
        return []
    viewer = getattr(env.unwrapped, "viewer", None)
    surface = getattr(viewer, "sim_surface", None)
    if surface is None or not hasattr(surface, "pos2pix"):
        return []

    h, w = frame_shape[:2]
    sx = TILE / max(1, w)
    sy = TILE / max(1, h)
    pixels = []
    for p in points[max(0, start_idx) :]:
        px, py = surface.pos2pix(float(p[0]), float(p[1]))
        pixels.append((int(px * sx), int(py * sy)))
    return pixels


def goal_world_state(env) -> dict | None:
    road = getattr(env.unwrapped, "road", None)
    if road is None:
        return None
    for obj in getattr(road, "objects", []):
        if obj.__class__.__name__ == "Landmark" and hasattr(obj, "position"):
            return {
                "position": obj.position.copy(),
                "heading": float(getattr(obj, "heading", 0.0)),
            }
    return None


def pd_warm_start(env, horizon: int) -> np.ndarray:
    """Plan PD repete pour guider CEM sans modifier l'environnement."""
    action = HeuristicPDPolicy(noise=0.0)(env).astype(np.float32)
    return np.repeat(action[None, :], horizon, axis=0)


def draw_path_overlay(screen, env, *, plan_points, trail_points, start_idx: int, x: int, y: int, frame_shape):
    overlay = pygame.Surface((TILE, TILE), pygame.SRCALPHA)

    goal = goal_world_state(env)
    if goal is not None:
        goal_px = world_points_to_tile_pixels(env, [goal["position"]], frame_shape)
        if goal_px:
            gx, gy = goal_px[0]
            heading = goal["heading"]
            nose = (int(gx + 26 * np.cos(heading)), int(gy + 26 * np.sin(heading)))
            pygame.draw.circle(overlay, (80, 255, 205, 90), (gx, gy), 22, 3)
            pygame.draw.line(overlay, (80, 255, 205, 130), (gx, gy), nose, 3)

    trail = world_points_to_tile_pixels(env, trail_points, frame_shape)
    if len(trail) >= 2:
        pygame.draw.lines(overlay, (92, 130, 150, 105), False, trail, 3)
        pygame.draw.circle(overlay, (126, 165, 185, 125), trail[-1], 4)

    plan = world_points_to_tile_pixels(env, plan_points, frame_shape, start_idx=start_idx)
    if len(plan) >= 2:
        pygame.draw.lines(overlay, (0, 40, 80, 185), False, plan, 9)
        pygame.draw.lines(overlay, (20, 176, 255, 230), False, plan, 5)
        for j, p in enumerate(plan[1:], start=1):
            radius = 7 if j == len(plan) - 1 else 4
            color = (55, 224, 255, 245) if j == len(plan) - 1 else (75, 194, 255, 205)
            pygame.draw.circle(overlay, color, p, radius)
    screen.blit(overlay, (x, y))


def draw_text(screen, font, text: str, x: int, y: int, color=TEXT):
    screen.blit(font.render(text, True, color), (x, y))


def draw_top_bar(screen, fonts, *, mode: str, detail: str, accent, device, n_envs: int, done: int, success: int):
    pygame.draw.rect(screen, BG, (0, 0, screen.get_width(), TOP_BAR))
    pygame.draw.line(screen, LINE, (0, TOP_BAR - 1), (screen.get_width(), TOP_BAR - 1), 1)

    pygame.draw.circle(screen, accent, (24, 25), 6)
    draw_text(screen, fonts["label"], "LeWM live", 42, 14, MUTED)
    draw_text(screen, fonts["title"], mode, 42, 30, TEXT)

    right = screen.get_width() - 18
    succ_pct = 100.0 * success / max(1, done) if done else 0.0
    stats = [
        f"{detail}",
        f"{n_envs} env",
        f"{success}/{done} success ({succ_pct:.0f}%)",
        str(device).upper(),
    ]
    x = right
    for item in reversed(stats):
        surf = fonts["label"].render(item, True, MUTED)
        rect = surf.get_rect()
        rect.topright = (x, 27)
        pad_x = 10
        chip = rect.inflate(pad_x * 2, 10)
        pygame.draw.rect(screen, PANEL, chip, border_radius=6)
        pygame.draw.rect(screen, LINE, chip, width=1, border_radius=6)
        screen.blit(surf, rect)
        x = chip.left - 8


def main():
    pygame.display.init()
    pygame.font.init()
    n_envs = GRID * GRID
    W = GRID * TILE + (GRID + 1) * GAP
    H = GRID * TILE + (GRID + 1) * GAP + TOP_BAR
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"LeWM live - {GRID}x{GRID} parallel rollouts  [waiting for ckpt]")
    clock = pygame.time.Clock()
    fonts = {
        "title": pygame.font.SysFont("Helvetica Neue", 18, bold=True),
        "label": pygame.font.SysFont("Helvetica Neue", 12),
    }

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = None
    last_mtime = 0.0
    epoch_seen = -1
    pd = HeuristicPDPolicy(noise=0.4)

    rng = np.random.default_rng()
    envs = []
    last_frames = []
    for i in range(n_envs):
        e = make_parking(seed=int(rng.integers(0, 2**31 - 1)), img_size=64)
        e.reset()
        # warm up le viewer interne de highway_env (sinon premier render peut crasher)
        try:
            f = e.render()
        except Exception:
            f = np.zeros((256, 256, 3), dtype=np.uint8)
        envs.append(e)
        last_frames.append(f)
    step_idx = [0] * n_envs
    ep_idx = [0] * n_envs
    plans = [None] * n_envs
    plan_paths = [None] * n_envs
    plan_idx = [0] * n_envs
    trail_paths = [[vehicle_state(e)["position"].copy()] for e in envs]
    anim_from = [vehicle_state(e) for e in envs]
    anim_to = [vehicle_state(e) for e in envs]
    anim_phase = ANIM_SUBSTEPS
    n_success = 0
    n_episodes_done = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        model, last_mtime, reloaded, ep = maybe_reload(model, last_mtime, device)
        if reloaded:
            epoch_seen = ep
            plans = [None] * n_envs
            plan_paths = [None] * n_envs
            plan_idx = [0] * n_envs

        if model is not None:
            mode = "TRAINED LeWM + CEM"
            detail = f"epoch {epoch_seen + 1 if epoch_seen >= 0 else '?'}"
            color = GREEN
        else:
            mode = "PD heuristic"
            detail = "waiting for ckpt"
            color = AMBER

        if anim_phase >= ANIM_SUBSTEPS:
            # Pick actions for each env (CEM batche pour les envs qui replanifient)
            actions = []
            if model is not None:
                need_plan = [i for i in range(n_envs) if plans[i] is None or plan_idx[i] >= len(plans[i])]
                if need_plan:
                    cur_imgs = np.stack([render_pixel(envs[i]) for i in need_plan])
                    goal_imgs = np.stack([render_goal_pixel(envs[i]) for i in need_plan])
                    z_cur = model.encode(to_t_batch(cur_imgs, device))
                    z_goal = model.encode(to_t_batch(goal_imgs, device))
                    warm = np.stack([pd_warm_start(envs[i], PLAN_HORIZON) for i in need_plan])
                    batch_plans = cem_plan_batch(model, z_cur, z_goal, warm_start=warm)
                    for k, i in enumerate(need_plan):
                        plans[i] = batch_plans[k]
                        plan_paths[i] = planned_world_points(envs[i], plans[i])
                        plan_idx[i] = 0
                for i in range(n_envs):
                    actions.append(plans[i][plan_idx[i]])
                    plan_idx[i] += 1
            else:
                for i in range(n_envs):
                    actions.append(pd(envs[i]))

            # Step all envs. Le rendu entre deux steps est interpole pour faire video.
            for i in range(n_envs):
                anim_from[i] = vehicle_state(envs[i])
                envs[i].step(actions[i])
                anim_to[i] = vehicle_state(envs[i])
                trail_paths[i].append(anim_to[i]["position"].copy())
                if len(trail_paths[i]) > TRAIL_MAX_POINTS:
                    del trail_paths[i][0 : len(trail_paths[i]) - TRAIL_MAX_POINTS]
                step_idx[i] += 1
                dist, ang = goal_distance(envs[i])
                if step_idx[i] >= MAX_STEPS:
                    if dist < 1.5 and ang < 15.0:
                        n_success += 1
                    n_episodes_done += 1
                    envs[i].reset(seed=int(rng.integers(0, 2**31 - 1)))
                    current = vehicle_state(envs[i])
                    anim_from[i] = current
                    anim_to[i] = current
                    trail_paths[i] = [current["position"].copy()]
                    step_idx[i] = 0
                    ep_idx[i] += 1
                    plans[i] = None
                    plan_paths[i] = None
                    plan_idx[i] = 0
            anim_phase = 0

        succ_pct = (100.0 * n_success / max(1, n_episodes_done)) if n_episodes_done > 0 else 0.0

        # Render grid
        screen.fill(BG)
        draw_top_bar(
            screen,
            fonts,
            mode=mode,
            detail=detail,
            accent=color,
            device=device,
            n_envs=n_envs,
            done=n_episodes_done,
            success=n_success,
        )

        for i in range(n_envs):
            row, col = divmod(i, GRID)
            x = GAP + col * (TILE + GAP)
            y = TOP_BAR + GAP + row * (TILE + GAP)
            try:
                alpha = min(1.0, (anim_phase + 1) / ANIM_SUBSTEPS)
                full = render_interpolated(envs[i], anim_from[i], anim_to[i], alpha)
                last_frames[i] = full
            except Exception:
                full = last_frames[i]  # garde la derniere frame valide
            surf = array_to_surface(full)
            surf = pygame.transform.smoothscale(surf, (TILE, TILE))
            screen.blit(surf, (x, y))
            draw_path_overlay(
                screen,
                envs[i],
                plan_points=plan_paths[i],
                trail_points=trail_paths[i],
                start_idx=plan_idx[i],
                x=x,
                y=y,
                frame_shape=full.shape,
            )
            pygame.draw.rect(screen, LINE, (x - 1, y - 1, TILE + 2, TILE + 2), width=1, border_radius=4)
            pygame.draw.rect(screen, color, (x - 1, y - 1, TILE + 2, TILE + 2), width=2, border_radius=4)
            prog = step_idx[i] / MAX_STEPS
            pygame.draw.rect(screen, LINE, (x, y + TILE - 4, TILE, 4))
            pygame.draw.rect(screen, color, (x, y + TILE - 4, int(TILE * prog), 4))

        pygame.display.set_caption(
            f"LeWM live - {mode} ({detail})  |  {n_envs} parallel envs  "
            f"| episodes done: {n_episodes_done}  | success {n_success}/{n_episodes_done} ({succ_pct:.0f}%)"
        )

        pygame.display.flip()
        anim_phase += 1
        clock.tick(FPS)

    for e in envs:
        try:
            e.close()
        except Exception:
            pass
    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
