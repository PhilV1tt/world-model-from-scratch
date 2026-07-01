"""Manoeuvre de parking analytique: chemin de Dubins (courbure bornee) vers une
pose de pre-creneau sur l'axe de la place, segment droit jusqu'au centre, suivi
en pure-pursuit + creneau terminal (creep-and-stop).

Model-free et privilegie (utilise la pose but de l'env). Gagnant de la Phase 4:
100% de stationnement strict (dist<0.15 m, axe<5 deg) sur les graines de test.
Interface: BiarcPursuit(env).act() -> action [accel, steer]; .path (N,2) = chemin planifie.
"""
from __future__ import annotations
import math

import numpy as np

from src.parking_env import _get_goal, _goal_axes, _wrap_angle

L = 5.0                  # longueur vehicule; le modele bicyclette utilise L/2
LB = L / 2.0             # empattement effectif 2.5
MAX_STEER = math.pi / 4  # action steer=1 -> pi/4 rad
MAX_ACCEL = 5.0
DT = 0.2                 # dt de commande (5 Hz)


def _mod2pi(x):
    return x - 2 * math.pi * math.floor(x / (2 * math.pi))


def _dubins_LSL(a, b, d):
    tmp = d + math.sin(a) - math.sin(b)
    p2 = 2 + d * d - 2 * math.cos(a - b) + 2 * d * (math.sin(a) - math.sin(b))
    if p2 < 0:
        return None
    t = _mod2pi(-a + math.atan2(math.cos(b) - math.cos(a), tmp))
    p = math.sqrt(p2)
    q = _mod2pi(b - math.atan2(math.cos(b) - math.cos(a), tmp))
    return (t, p, q, ('L', 'S', 'L'))


def _dubins_RSR(a, b, d):
    tmp = d - math.sin(a) + math.sin(b)
    p2 = 2 + d * d - 2 * math.cos(a - b) + 2 * d * (math.sin(b) - math.sin(a))
    if p2 < 0:
        return None
    t = _mod2pi(a - math.atan2(math.cos(a) - math.cos(b), tmp))
    p = math.sqrt(p2)
    q = _mod2pi(-b + math.atan2(math.cos(a) - math.cos(b), tmp))
    return (t, p, q, ('R', 'S', 'R'))


def _dubins_LSR(a, b, d):
    p2 = -2 + d * d + 2 * math.cos(a - b) + 2 * d * (math.sin(a) + math.sin(b))
    if p2 < 0:
        return None
    p = math.sqrt(p2)
    tmp = math.atan2(-math.cos(a) - math.cos(b), d + math.sin(a) + math.sin(b)) - math.atan2(-2.0, p)
    t = _mod2pi(-a + tmp)
    q = _mod2pi(-_mod2pi(b) + tmp)
    return (t, p, q, ('L', 'S', 'R'))


def _dubins_RSL(a, b, d):
    p2 = -2 + d * d + 2 * math.cos(a - b) - 2 * d * (math.sin(a) + math.sin(b))
    if p2 < 0:
        return None
    p = math.sqrt(p2)
    tmp = math.atan2(math.cos(a) + math.cos(b), d - math.sin(a) - math.sin(b)) - math.atan2(2.0, p)
    t = _mod2pi(a - tmp)
    q = _mod2pi(b - tmp)
    return (t, p, q, ('R', 'S', 'L'))


def _dubins_RLR(a, b, d):
    tmp = (6.0 - d * d + 2 * math.cos(a - b) + 2 * d * (math.sin(a) - math.sin(b))) / 8.0
    if abs(tmp) > 1:
        return None
    p = _mod2pi(2 * math.pi - math.acos(tmp))
    t = _mod2pi(a - math.atan2(math.cos(a) - math.cos(b), d - math.sin(a) + math.sin(b)) + p / 2.0)
    q = _mod2pi(a - b - t + p)
    return (t, p, q, ('R', 'L', 'R'))


def _dubins_LRL(a, b, d):
    tmp = (6.0 - d * d + 2 * math.cos(a - b) + 2 * d * (-math.sin(a) + math.sin(b))) / 8.0
    if abs(tmp) > 1:
        return None
    p = _mod2pi(2 * math.pi - math.acos(tmp))
    t = _mod2pi(-a + math.atan2(-math.cos(a) + math.cos(b), d + math.sin(a) - math.sin(b)) + p / 2.0)
    q = _mod2pi(_mod2pi(b) - a - t + p)
    return (t, p, q, ('L', 'R', 'L'))


_WORDS = [_dubins_LSL, _dubins_RSR, _dubins_LSR, _dubins_RSL, _dubins_RLR, _dubins_LRL]


def dubins_shortest(q0, q1, rho):
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    D = math.hypot(dx, dy)
    d = D / rho
    theta = _mod2pi(math.atan2(dy, dx))
    a = _mod2pi(q0[2] - theta)
    b = _mod2pi(q1[2] - theta)
    best = None
    for w in _WORDS:
        res = w(a, b, d)
        if res is None:
            continue
        t, p, qq, types = res
        cost = t + p + qq
        if best is None or cost < best[0]:
            best = (cost, [(types[0], t), (types[1], p), (types[2], qq)])
    if best is None:
        return None, float('inf')
    return best[1], best[0] * rho


def sample_dubins(q0, segs, rho, step=0.25):
    pts = []
    x, y, th = q0
    for stype, slen in segs:
        L_real = slen * rho
        n = max(1, int(math.ceil(L_real / step)))
        ds = L_real / n
        for _ in range(n):
            if stype == 'S':
                x += ds * math.cos(th)
                y += ds * math.sin(th)
            elif stype == 'L':
                dth = ds / rho
                cx = x - rho * math.sin(th)
                cy = y + rho * math.cos(th)
                th2 = th + dth
                x = cx + rho * math.sin(th2)
                y = cy - rho * math.cos(th2)
                th = th2
            else:  # 'R'
                dth = ds / rho
                cx = x + rho * math.sin(th)
                cy = y - rho * math.cos(th)
                th2 = th - dth
                x = cx - rho * math.sin(th2)
                y = cy + rho * math.cos(th2)
                th = th2
            pts.append((x, y, th))
    return pts


class BiarcPursuit:
    def __init__(self, env, rho=5.8, lookahead=3.0, cruise_speed=3.0, back=16.0, replan_every=5):
        self.env = env
        self.rho = rho
        self.lookahead = lookahead
        self.cruise = cruise_speed
        self.back = back
        self.replan_every = replan_every
        self.goal = _get_goal(env)
        self.forward, self.left = _goal_axes(self.goal)
        self.path = None
        self.path_orient = None
        self.step_count = 0

    def _state(self):
        v = self.env.unwrapped.controlled_vehicles[0]
        pos = np.asarray(v.position, dtype=float)
        return pos, float(v.heading), float(v.speed)

    def _goal_frame(self, pos, heading):
        rel = pos - self.goal.position
        along = float(rel @ self.forward)
        lateral = float(rel @ self.left)
        he = _wrap_angle(heading - self.goal.heading)
        return along, lateral, he

    def _plan(self, pos, heading):
        gh = self.goal.heading
        toward_goal = self.goal.position - np.array([self.goal.position[0], 0.0])
        vdir = toward_goal / (np.linalg.norm(toward_goal) + 1e-9)
        fwd = np.array([math.cos(gh), math.sin(gh)])
        orient = gh if float(fwd @ vdir) >= 0 else gh + math.pi
        fdir = np.array([math.cos(orient), math.sin(orient)])

        best = None
        for seg_back in (self.back, self.back + 3.0, self.back - 2.0, self.back + 6.0):
            app = self.goal.position - seg_back * fdir
            if abs(app[0]) > 32.0 or abs(app[1]) > 19.0:
                continue
            segs, length = dubins_shortest((pos[0], pos[1], heading), (app[0], app[1], orient), self.rho)
            if segs is None:
                continue
            if best is None or length < best[0]:
                best = (length, segs, app, seg_back)
        if best is None:
            seg_back = self.back
            app = self.goal.position - seg_back * fdir
            segs, _ = dubins_shortest((pos[0], pos[1], heading), (app[0], app[1], orient), self.rho)
            best = (0.0, segs, app, seg_back)

        _, segs, app, seg_back = best
        pts = []
        if segs is not None:
            pts = sample_dubins((pos[0], pos[1], heading), segs, self.rho, step=0.25)
        n = max(1, int(seg_back / 0.25))
        for i in range(1, n + 1):
            p = app + fdir * (seg_back * i / n)
            pts.append((p[0], p[1], orient))
        if not pts:
            pts = [(self.goal.position[0], self.goal.position[1], orient)]
        self.path = np.array([[p[0], p[1]] for p in pts], dtype=float)
        self.path_orient = orient
        return self.path

    def _lookahead_point(self, pos, Ld):
        path = self.path
        d = np.linalg.norm(path - pos, axis=1)
        i0 = int(np.argmin(d))
        for i in range(i0, len(path)):
            if np.linalg.norm(path[i] - pos) >= Ld:
                return path[i], i0
        return path[-1], i0

    def _pursuit_steer(self, pos, heading, target):
        dvec = target - pos
        alpha = _wrap_angle(math.atan2(dvec[1], dvec[0]) - heading)
        ld = max(1e-3, np.linalg.norm(dvec))
        kappa = 2.0 * math.sin(alpha) / ld
        sin_beta = np.clip(LB * kappa, -0.999, 0.999)
        beta = math.asin(sin_beta)
        delta = math.atan(2.0 * math.tan(beta))
        return float(np.clip(delta / MAX_STEER, -1.0, 1.0))

    def _accel_for_speed(self, cur, des):
        return float(np.clip(2.5 * (des - cur) / MAX_ACCEL, -1.0, 1.0))

    def act(self, replan=False):
        pos, heading, speed = self._state()
        along, lateral, he = self._goal_frame(pos, heading)
        dist = math.hypot(along, lateral)

        if dist < 1.6:
            return self._terminal(pos, heading, speed, along, lateral, he)

        if self.path is None or replan:
            self._plan(pos, heading)
        self.step_count += 1

        rem = float(np.linalg.norm(self.path[-1] - pos))
        Ld = np.clip(0.9 * max(speed, 0.5) + 1.2, 1.6, self.lookahead)
        target, _ = self._lookahead_point(pos, Ld)
        steer = self._pursuit_steer(pos, heading, target)
        desired = min(self.cruise, 0.6 * rem + 0.4)
        accel = self._accel_for_speed(speed, desired)
        return np.array([accel, steer], dtype=np.float32)

    def _terminal(self, pos, heading, speed, along, lateral, he):
        target_axis_speed = np.clip(-1.0 * along, -0.9, 0.9)
        c = math.cos(he)
        if abs(c) < 0.25:
            c = 0.25 * (1.0 if c >= 0 else -1.0)
        desired_car_speed = float(np.clip(target_axis_speed / c, -0.9, 0.9))
        travel_sign = 1.0 if desired_car_speed >= 0 else -1.0
        axis_orient = self.goal.heading if c >= 0 else self.goal.heading + math.pi
        heading_err = _wrap_angle(axis_orient - heading)
        ct = math.atan2(-1.2 * lateral, 1.0)
        steer_cmd = heading_err + travel_sign * ct
        steer = float(np.clip(1.4 * steer_cmd / MAX_STEER, -1.0, 1.0))
        if abs(along) < 0.06 and abs(lateral) < 0.08:
            desired_car_speed = 0.0
        accel = self._accel_for_speed(speed, desired_car_speed)
        if math.hypot(along, lateral) < 0.15 and abs(speed) < 0.4:
            accel = float(np.clip(-speed / MAX_ACCEL - 0.15 * np.sign(speed), -1, 1))
            steer = 0.0 if abs(lateral) < 0.08 else steer
        return np.array([accel, steer], dtype=np.float32)
