"""Protokol eksperimen yang dipakai bersama oleh Q-learning dan DQN."""

import hashlib
import heapq
import json
import math
import os
from pathlib import Path

import numpy as np

EVAL_HEADINGS = (0.0, 22.5, 45.0, 67.5, 90.0)
EVAL_REPEATS = 2
GRID = 0.01
CLEARANCE = 0.08
DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


def obstacle_distance(x, y, obstacle):
    """Jarak bertanda terhadap jejak rintangan di lantai."""
    if obstacle["shape"] == "cyl":
        return np.hypot(x - obstacle["x"], y - obstacle["y"]) - obstacle["r"]
    if obstacle["shape"] == "box":
        dx, dy = x - obstacle["x"], y - obstacle["y"]
        c, s = math.cos(obstacle["yaw"]), math.sin(obstacle["yaw"])
        a = np.abs(c * dx + s * dy) - obstacle["sx"] / 2
        b = np.abs(-s * dx + c * dy) - obstacle["sy"] / 2
        return np.hypot(np.maximum(a, 0), np.maximum(b, 0)) + np.minimum(np.maximum(a, b), 0)
    points = obstacle["pts"]
    distances, sides = [], []
    for (ax, ay), (bx, by) in zip(points, points[1:] + points[:1]):
        dx, dy = bx - ax, by - ay
        px, py = x - ax, y - ay
        fraction = np.clip((px * dx + py * dy) / (dx * dx + dy * dy), 0, 1)
        distances.append(np.hypot(px - fraction * dx, py - fraction * dy))
        sides.append(dx * py - dy * px)
    inside = (np.min(sides, axis=0) >= 0) | (np.max(sides, axis=0) <= 0)
    return np.where(inside, -1, 1) * np.min(distances, axis=0)


class RouteReference:
    """Waktu minimum dalam model maju-putar dengan delapan arah."""

    def __init__(self, layout, arena, start, target_radius, actions, wheel_radius, track):
        self.speed = wheel_radius * max((left for left, right in actions if left == right), default=0)
        self.spin = wheel_radius / track * max((right - left for left, right in actions if left == -right), default=0)
        if self.speed <= 0 or self.spin <= 0:
            raise ValueError("Epsilon berbasis rute memerlukan aksi maju lurus dan putar di tempat.")
        self.start = tuple(round(value / GRID) for value in start)
        axis = np.arange(math.floor(arena / GRID + 1e-9) + 1) * GRID
        x, y = np.meshgrid(axis, axis, indexing="ij")
        blocked = (x < CLEARANCE) | (y < CLEARANCE) | (x > arena - CLEARANCE) | (y > arena - CLEARANCE)
        for obstacle in layout["obstacles"]:
            blocked |= obstacle_distance(x, y, obstacle) < CLEARANCE
        for hx, hy, radius in layout["hazards"]:
            blocked |= np.hypot(x - hx, y - hy) <= radius
        self.free = ~blocked
        tx, ty = layout["target"]
        remaining = np.hypot(x - tx, y - ty) - target_radius
        self.goal = (remaining <= 0) & self.free
        self.lower_bound = np.maximum(remaining, 0) / self.speed
        self.advance = tuple(GRID * math.hypot(dx, dy) / self.speed for dx, dy in DIRECTIONS)
        self.turn = (math.pi / 4) / self.spin
        self.arrivals = {}
        if not all(0 <= i < len(axis) for i in self.start) or not self.free[self.start] or not self.goal.any():
            raise ValueError("Start atau target tidak memiliki sel bebas untuk rute epsilon.")

    def _search(self, heading):
        start = (*self.start, heading)
        frontier = [(self.lower_bound[self.start], 0.0, start)]
        cost = {start: 0.0}
        previous = {}
        size = self.free.shape[0]
        while frontier:
            _, elapsed, state = heapq.heappop(frontier)
            if elapsed > cost[state]:
                continue
            x, y, direction = state
            if self.goal[x, y]:
                path = [state]
                while path[-1] != start:
                    path.append(previous[path[-1]])
                points = [(i * GRID, j * GRID) for i, j, _ in reversed(path)]
                return elapsed, points
            neighbors = [((x, y, (direction + delta) % 8), self.turn) for delta in (-1, 1)]
            dx, dy = DIRECTIONS[direction]
            nx, ny = x + dx, y + dy
            # Periksa dua petak sisi sebelum menyeberang secara diagonal.
            if 0 <= nx < size and 0 <= ny < size and self.free[nx, ny] and self.free[x, ny] and self.free[nx, y]:
                neighbors.append(((nx, ny, direction), self.advance[direction]))
            for nxt, duration in neighbors:
                new_cost = elapsed + duration
                if new_cost < cost.get(nxt, math.inf):
                    cost[nxt] = new_cost
                    previous[nxt] = state
                    heapq.heappush(frontier, (new_cost + self.lower_bound[nxt[:2]], new_cost, nxt))
        raise ValueError("Tidak ada rute bebas menuju target untuk epsilon berbasis rute.")

    def from_yaw(self, yaw):
        if not self.arrivals:
            # Simpan waktu dan lintasan untuk setiap arah pada kisi.
            for heading in range(8):
                self.arrivals[heading] = self._search(heading)
        candidates = []
        for heading, (duration, path) in self.arrivals.items():
            angle = math.remainder(heading * math.pi / 4 - yaw, math.tau)
            candidates.append((duration + abs(angle) / self.spin, heading, path))
        duration, _, path = min(candidates)
        return duration, path


class EpsilonSchedule:
    def __init__(self, env, actions, wheel_radius, track, interval, decay):
        self.mode = os.getenv("MM_EPS_MODE", "decay").strip().lower()
        self.start, self.decay, self.floor = decay
        self.constant = None
        self.route = None
        self.interval = interval
        if self.mode == "constant":
            self.constant = float(os.getenv("MM_EPSILON", "0.10"))
            if not 0 <= self.constant <= 1:
                raise ValueError("MM_EPSILON harus berada dalam [0, 1].")
        elif self.mode == "route":
            self.route = RouteReference(env.layout, env.arena, env.start, env.target_r, actions, wheel_radius, track)
        elif self.mode != "decay":
            raise ValueError("MM_EPS_MODE harus decay, constant, atau route.")

    def for_episode(self, episode, yaw):
        if self.mode == "constant":
            return self.constant, None, None
        if self.mode == "route":
            seconds, _ = self.route.from_yaw(yaw)
            decisions = max(1, math.ceil(seconds / self.interval))
            return 1 / decisions, seconds, decisions
        return max(self.floor, self.start * self.decay ** (episode - 1)), None, None

    def metadata(self):
        if self.mode == "route":
            return {"mode": self.mode, "grid_m": GRID, "clearance_m": CLEARANCE,
                    "decision_interval_s": self.interval, "forward_m_s": self.route.speed,
                    "spin_rad_s": self.route.spin}
        if self.mode == "constant":
            return {"mode": self.mode, "value": self.constant}
        return {"mode": self.mode, "initial": self.start, "factor": self.decay, "floor": self.floor}


def output_dir(default):
    dest = Path(os.getenv("MM_OUT") or default).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def checkpoint_metadata(path, episode):
    return {"name": path.name, "episode": episode, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def evaluate_policy(env, agent, run_episode, dest, *, seed, checkpoint, settle_time, fields=None):
    results, by_heading = [], {}
    for heading in EVAL_HEADINGS:
        outcomes = []
        for repeat in range(1, EVAL_REPEATS + 1):
            before = env.sim_time()
            outcome, steps, total, path = run_episode(
                env, agent, 0.0, False, True, start_yaw=math.radians(90 - heading))
            elapsed = env.sim_time() - before - settle_time
            results.append({"heading": heading, "repeat": repeat, "outcome": outcome,
                            "decisions": steps, "reward": round(total, 2), "sim_time": round(elapsed, 3),
                            "path_m": round(sum(math.dist(a, b) for a, b in zip(path, path[1:])), 4),
                            "hazard_steps": env.hazard_steps, "path": path})
            outcomes.append(outcome)
            print(f"EVAL {len(results)}/{len(EVAL_HEADINGS) * EVAL_REPEATS}: arah={heading:5.1f} "
                  f"{outcome:9s} decisions={steps} R={total:.1f} hazard_steps={env.hazard_steps}", flush=True)
        by_heading[str(heading)] = outcomes
    successes = sum(row["outcome"] == "goal" for row in results)
    layout = env.layout
    payload = {"success": successes, "n": len(results), "arena": env.arena, "start": env.start,
               "target": layout["target"], "target_r": env.target_r, "obstacles": layout["obstacles"],
               "hazards": layout["hazards"], "headings": EVAL_HEADINGS, "repeats": EVAL_REPEATS,
               "by_heading": by_heading, "world": os.getenv("MM_WORLD"),
               "seed": seed, "epsilon": 0.0, "checkpoint": checkpoint, "episodes": results}
    payload.update(fields or {})
    dest.write_text(json.dumps(payload), encoding="utf-8")
    print(f"EVAL success: {successes}/{len(results)} -> {dest}", flush=True)
