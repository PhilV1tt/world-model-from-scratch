"""Viewer temps reel: la scene du parking + les trajectoires IMAGINEES par le modele.

Sur une seule figure (la vraie sim parking-v0) on superpose, a chaque pas:
  - le faisceau des trajectoires candidates que le planner imagine (lignes pales),
  - la trajectoire optimale imaginee que le modele retient (ligne vive),
  - la trace reellement parcourue par la voiture,
  - le but (place + cap).
A chaque replanification l'optimale imaginee se met a jour: on voit la trajectoire
optimale evoluer au fur et a mesure que la voiture avance.

Le planner est le modele du monde: il deroule ses latents (model.rollout_latents)
sous chaque sequence d'actions candidate et garde celles dont le latent final est le
plus proche du latent-but (qui encode la pose garee visee).

Modes:
  interactif (defaut)   : fenetre pygame. Esc pour quitter.
  --record out.mp4       : rendu hors-ecran (SDL dummy) -> video (verif sans ecran).

Exemples:
  python scripts/imagine_viewer.py --ckpt runs/lewm_fix_v1/ckpt_last.pt
  python scripts/imagine_viewer.py --variant static_vehicles
  SDL_VIDEODRIVER=dummy python scripts/imagine_viewer.py --record /tmp/imagine.mp4 --steps 320
"""
from __future__ import annotations

import argparse
import os as _os
_os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.parking_env import make_parking, render_pixel, render_goal_pixel, goal_distance, HeuristicPDPolicy
from src.parking_metrics import final_pose_metrics, user_strict_success
from src.lewm import LeWM, LeWMConfig
from scripts.live_viewer import to_t_batch, vehicle_state, render_interpolated, goal_world_state, array_to_surface
from scripts.plan import cem_plan_sim
from src.parking_control import BiarcPursuit

import pygame

TILE = 720
GAP = 12
TOP_BAR = 74
FPS = 30
ANIM_SUBSTEPS = 4
MAX_STEPS = 140

BG = (10, 12, 14); PANEL = (16, 19, 23); LINE = (42, 49, 56)
MUTED = (150, 158, 162); TEXT = (242, 239, 230)
GREEN = (118, 217, 140); CYAN = (90, 210, 235); AMBER = (229, 198, 93)


def make_fonts():
    try:
        pygame.font.init()
        return {
            "title": pygame.font.SysFont("Helvetica Neue", 18, bold=True),
            "label": pygame.font.SysFont("Helvetica Neue", 13),
            "big": pygame.font.SysFont("Helvetica Neue", 16, bold=True),
        }
    except Exception:
        class _Dummy:
            def render(self, *a, **k):
                return pygame.Surface((1, 1), pygame.SRCALPHA)
        return {k: _Dummy() for k in ("title", "label", "big")}


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = LeWMConfig(**ckpt["cfg"])
    model = LeWM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, int(ckpt.get("step", -1))


@torch.no_grad()
def cem_with_candidates(model, z_init, z_goal, horizon, pop, elites, iters, warm=None):
    """CEM latent. Retourne (best (H,2), elites (E,H,2)) pour visualiser l'imagination."""
    B, D = z_init.shape
    device = z_init.device
    A = 2
    mu = torch.zeros(B, horizon, A, device=device)
    sigma = torch.full((B, horizon, A), 1.0, device=device)
    if warm is not None:
        mu = torch.from_numpy(warm[None].astype(np.float32)).to(device)
        sigma[:] = 0.45
    last_elite = None
    for _ in range(iters):
        actions = (mu.unsqueeze(1) + sigma.unsqueeze(1) * torch.randn(B, pop, horizon, A, device=device)).clamp(-1, 1)
        a_flat = actions.reshape(B * pop, horizon, A)
        z_flat = z_init.unsqueeze(1).expand(B, pop, D).reshape(B * pop, D)
        z_final = model.rollout_latents(z_flat, a_flat)[:, -1, :].reshape(B, pop, D)
        cost = ((z_final - z_goal.unsqueeze(1)) ** 2).sum(dim=2)
        topk = cost.topk(elites, largest=False, dim=1).indices
        idx = topk.unsqueeze(-1).unsqueeze(-1).expand(B, elites, horizon, A)
        elite = torch.gather(actions, 1, idx)
        last_elite = elite
        mu = elite.mean(dim=1)
        sigma = elite.std(dim=1).clamp(min=0.05)
    return mu[0].cpu().numpy(), last_elite[0].cpu().numpy()


def rollout_world_points(state, plan):
    """Projette une sequence d'actions en points monde (approx cinematique bicyclette)."""
    pos = np.asarray(state["position"], dtype=np.float64).copy()
    heading = float(state["heading"]); speed = float(state["speed"])
    pts = [pos.copy()]; dt = 1.0 / 5.0
    for accel, steer in plan:
        speed = float(np.clip(speed + float(accel) * 2.5 * dt, -6.0, 6.0))
        heading = heading + float(steer) * 0.75 * dt
        pos = pos + np.array([np.cos(heading), np.sin(heading)]) * speed * dt
        pts.append(pos.copy())
    return pts


def world_to_tile(env, points, frame_shape):
    if not points:
        return []
    viewer = getattr(env.unwrapped, "viewer", None)
    surface = getattr(viewer, "sim_surface", None)
    if surface is None or not hasattr(surface, "pos2pix"):
        return []
    h, w = frame_shape[:2]
    sx, sy = TILE / max(1, w), TILE / max(1, h)
    out = []
    for p in points:
        px, py = surface.pos2pix(float(p[0]), float(p[1]))
        out.append((int(px * sx), int(py * sy)))
    return out


def draw_imagination(screen, env, best_world, cand_worlds, trail, past_optima, x, y, frame_shape):
    ov = pygame.Surface((TILE, TILE), pygame.SRCALPHA)

    # but
    goal = goal_world_state(env)
    if goal is not None:
        gp = world_to_tile(env, [goal["position"]], frame_shape)
        if gp:
            gx, gy = gp[0]; hd = goal["heading"]
            nose = (int(gx + 30 * np.cos(hd)), int(gy + 30 * np.sin(hd)))
            pygame.draw.circle(ov, (80, 255, 205, 110), (gx, gy), 24, 3)
            pygame.draw.line(ov, (80, 255, 205, 150), (gx, gy), nose, 3)

    # trace reelle
    tr = world_to_tile(env, trail, frame_shape)
    if len(tr) >= 2:
        pygame.draw.lines(ov, (120, 150, 170, 150), False, tr, 3)

    # optima passees (fade) -> montre l'evolution de la trajectoire imaginee
    n = len(past_optima)
    for k, pth in enumerate(past_optima):
        px = world_to_tile(env, pth, frame_shape)
        if len(px) >= 2:
            a = int(30 + 40 * (k + 1) / max(1, n))
            pygame.draw.lines(ov, (90, 160, 210, a), False, px, 2)

    # faisceau des trajectoires candidates imaginees (pales)
    for pth in cand_worlds:
        px = world_to_tile(env, pth, frame_shape)
        if len(px) >= 2:
            pygame.draw.lines(ov, (70, 130, 180, 60), False, px, 2)

    # trajectoire optimale imaginee / planifiee (vive)
    bp = world_to_tile(env, best_world, frame_shape)
    if len(bp) >= 2:
        pygame.draw.lines(ov, (0, 40, 80, 200), False, bp, 8)
        pygame.draw.lines(ov, (40, 200, 255, 245), False, bp, 4)
        for j, p in enumerate(bp[1:], 1):
            r = 7 if j == len(bp) - 1 else 4
            pygame.draw.circle(ov, (70, 225, 255, 245), p, r)

    screen.blit(ov, (x, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs" / "lewm_fix_v1" / "ckpt_last.pt"))
    ap.add_argument("--variant", default="standard")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--pop", type=int, default=256)
    ap.add_argument("--elites", type=int, default=24)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--show-candidates", type=int, default=24, help="nb de trajectoires candidates affichees")
    ap.add_argument("--drive", choices=["model", "biarc"], default="model",
                    help="pilote: 'model' = imagination latente du world model ; 'biarc' = manoeuvre analytique qui se gare parfaitement")
    ap.add_argument("--warm", action="store_true", help="warm-start du CEM par la politique PD (plan initial vers le but, raffine par le modele)")
    ap.add_argument("--finalize", action="store_true", help="finition de pose privilegiee pres du but (redresse entre les lignes)")
    ap.add_argument("--finalize-dist", type=float, default=1.6)
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--steps", type=int, default=320)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()

    recording = args.record is not None

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, step = load_model(args.ckpt, device)

    W = TILE + 2 * GAP
    H = TILE + 2 * GAP + TOP_BAR
    if recording:
        # NE PAS forcer SDL 'dummy': highway-env rend hors-ecran correctement sans fenetre,
        # alors que le driver 'dummy' rend du noir. On compose sur une Surface, sans display.
        pygame.init()
        screen = pygame.Surface((W, H))
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.record, fps=15, macro_block_size=None)
    else:
        pygame.display.init()
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("LeWM - trajectoires imaginees")
        writer = None
    fonts = make_fonts()
    clock = pygame.time.Clock()
    rng = np.random.default_rng(args.seed)
    pd = HeuristicPDPolicy(noise=0.0)

    def new_env(seed):
        e = make_parking(seed=seed, img_size=64, env_variant=args.variant)
        e.reset(seed=seed)
        # Forcer highway-env a rendre HORS-ECRAN: sinon il ouvre sa propre fenetre
        # "Highway-env" et vole l'affichage pygame, ecrasant nos overlays.
        try:
            e.unwrapped.config["offscreen_rendering"] = True
            e.unwrapped.viewer = None  # recree le viewer en mode offscreen au prochain render
        except Exception:
            pass
        try:
            e.render()
        except Exception:
            pass
        return e

    ep_seed = args.seed
    env = new_env(ep_seed)
    ctrl = BiarcPursuit(env) if args.drive == "biarc" else None
    best_world = []
    cand_worlds = []
    drive_action = np.zeros(2, dtype=np.float32)
    trail = [vehicle_state(env)["position"].copy()]
    past_optima = deque(maxlen=6)
    anim_from = vehicle_state(env)
    anim_to = vehicle_state(env)
    anim_phase = ANIM_SUBSTEPS
    plan_state = vehicle_state(env)
    step_idx = 0
    n_done = 0
    n_strict = 0
    last_m = final_pose_metrics(env)
    frames_written = 0
    phase = "modele imagine"
    hold_frames = 0
    running = True

    while running:
        if not recording:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

        if anim_phase >= ANIM_SUBSTEPS and hold_frames > 0:
            # fige sur l'etat final (voiture garee) pour montrer le succes, puis episode suivant
            hold_frames -= 1
            anim_from = anim_to = vehicle_state(env)
            if hold_frames == 0:
                ep_seed += 1
                env = new_env(ep_seed)
                ctrl = BiarcPursuit(env) if args.drive == "biarc" else None
                trail = [vehicle_state(env)["position"].copy()]
                past_optima.clear()
                anim_from = anim_to = plan_state = vehicle_state(env)
                step_idx = 0
                best_world = []; cand_worlds = []
            anim_phase = 0
        elif anim_phase >= ANIM_SUBSTEPS:
            # replanifie a CHAQUE pas -> on voit la trajectoire optimale evoluer
            plan_state = vehicle_state(env)
            dist_now, _ = goal_distance(env)
            cand_worlds = []
            if args.drive == "biarc":
                if not best_world:  # plan UNE fois par episode (replanifier chaque pas fait diverger)
                    ctrl._plan(np.asarray(plan_state["position"], dtype=float), float(plan_state["heading"]))
                    best_world = [tuple(p) for p in ctrl.path]
                drive_action = ctrl.act()
                phase = "manoeuvre planifiee (bi-arc)" if dist_now >= 1.6 else "creneau final"
            elif args.finalize and dist_now < args.finalize_dist:
                best_plan = cem_plan_sim(env, horizon=10, pop_size=28, elites=6, iters=3,
                                         seed=int(rng.integers(1 << 30)), w_pos=0.4, w_lat=1.0,
                                         w_along=0.5, w_head=0.2, w_axis=0.4, w_speed=0.25, w_coll=30.0)
                best_world = rollout_world_points(plan_state, best_plan)
                drive_action = best_plan[0]
                phase = "finition (pose privilegiee)"
            else:
                cur = render_pixel(env); goal = render_goal_pixel(env)
                warm = np.repeat(pd(env)[None, :], args.horizon, axis=0).astype(np.float32) if args.warm else None
                with torch.no_grad():
                    zc = model.encode(to_t_batch(cur[None], device))
                    zg = model.encode(to_t_batch(goal[None], device))
                    if zc.dim() == 3:
                        zc = zc.reshape(1, -1); zg = zg.reshape(1, -1)
                    best_plan, elite_plans = cem_with_candidates(
                        model, zc, zg, args.horizon, args.pop, args.elites, args.iters, warm=warm)
                show_elites = elite_plans[: args.show_candidates] if len(elite_plans) else elite_plans
                cand_worlds = [rollout_world_points(plan_state, e) for e in show_elites]
                best_world = rollout_world_points(plan_state, best_plan)
                drive_action = best_plan[0] if len(best_plan) else np.zeros(2, dtype=np.float32)
                phase = "modele imagine"
            if len(best_world) and (step_idx % 3 == 0):
                past_optima.append(best_world)

            anim_from = vehicle_state(env)
            env.step(np.asarray(drive_action, dtype=np.float32))
            anim_to = vehicle_state(env)
            trail.append(anim_to["position"].copy())
            if len(trail) > 120:
                del trail[0:len(trail) - 120]
            step_idx += 1
            last_m = final_pose_metrics(env)
            parked_between_lines = last_m["dist"] < 0.30 and last_m["angle_deg"] < 10.0 and not last_m.get("collided", False)
            fully_parked = last_m["dist"] < 0.16 and last_m["angle_deg"] < 6.0 and abs(last_m["speed_mps"]) < 0.3 and step_idx > 12
            done_ep = step_idx >= MAX_STEPS or bool(last_m.get("collided", False)) or fully_parked
            if done_ep:
                if parked_between_lines:
                    n_strict += 1
                n_done += 1
                hold_frames = 26  # fige ~1s sur l'etat final avant l'episode suivant
            anim_phase = 0

        # render
        screen.fill(BG)
        aligned_now = last_m["dist"] < 0.35 and last_m["angle_deg"] < 10.0 and not last_m.get("collided", False)
        accent = GREEN if aligned_now else (CYAN if last_m["dist"] < 1.0 else AMBER)
        pygame.draw.rect(screen, BG, (0, 0, W, TOP_BAR))
        pygame.draw.line(screen, LINE, (0, TOP_BAR - 1), (W, TOP_BAR - 1), 1)
        screen.blit(fonts["label"].render("LeWM - world model imagine ses trajectoires", True, MUTED), (16, 12))
        screen.blit(fonts["title"].render(f"{phase}   |   variant {args.variant}   H {args.horizon}   ckpt step {step}", True, TEXT), (16, 32))
        srate = 100.0 * n_strict / max(1, n_done)
        info = f"dist {last_m['dist']:.2f}m  axe {last_m['angle_deg']:.0f}deg  garees {n_strict}/{n_done} ({srate:.0f}%)"
        s = fonts["big"].render(info, True, accent)
        screen.blit(s, (W - s.get_width() - 16, 28))

        x, y = GAP, TOP_BAR + GAP
        try:
            alpha = min(1.0, (anim_phase + 1) / ANIM_SUBSTEPS)
            full = render_interpolated(env, anim_from, anim_to, alpha)
        except Exception:
            full = render_pixel(env)
        surf = pygame.transform.smoothscale(array_to_surface(np.ascontiguousarray(full)), (TILE, TILE))
        screen.blit(surf, (x, y))
        draw_imagination(screen, env, best_world, cand_worlds, trail, list(past_optima), x, y, np.asarray(full).shape)
        pygame.draw.rect(screen, accent, (x - 1, y - 1, TILE + 2, TILE + 2), width=2, border_radius=4)

        if recording:
            writer.append_data(pygame.surfarray.array3d(screen).swapaxes(0, 1))
            frames_written += 1
            if frames_written >= args.steps:
                running = False
        else:
            pygame.display.flip()
            clock.tick(FPS)
        anim_phase += 1

    if writer is not None:
        writer.close()
        print(f"wrote {frames_written} frames -> {args.record}  (strict {n_strict}/{n_done})")
    try:
        env.close()
    except Exception:
        pass
    pygame.quit()


if __name__ == "__main__":
    main()
