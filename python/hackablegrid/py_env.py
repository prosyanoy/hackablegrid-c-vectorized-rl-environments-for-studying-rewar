"""Pure-Python Gymnasium implementation of the same dynamics.

It serves as (a) the baseline for the throughput benchmark and (b) an independent
reference implementation used to cross-check the C++ engine in the tests.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_MOVES = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0), 4: (0, 0)}


class PyGridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, variant: str = "intended", grid_size: int = 8, max_steps: int = 64,
                 goal_reward: float = 1.0, step_penalty: float = 0.01,
                 pickup_reward: float = 0.3, respawn_delay: int = 1):
        self.misspecified = variant == "misspecified"
        self.G, self.max_steps = grid_size, max_steps
        self.goal_reward, self.step_penalty = goal_reward, step_penalty
        self.pickup_reward, self.respawn_delay = pickup_reward, respawn_delay
        self.observation_space = spaces.Box(0.0, 1.0, (3 * grid_size * grid_size + 1,), np.float32)
        self.action_space = spaces.Discrete(5)
        self._ep_true = self._ep_proxy = 0.0
        self._ep_pick = 0

    def _obs(self):
        G = self.G
        o = np.zeros(3 * G * G + 1, np.float32)
        o[self.ay * G + self.ax] = 1
        o[G * G + self.gy * G + self.gx] = 1
        if self.cd == 0:
            o[2 * G * G + self.py * G + self.px] = 1
        o[-1] = self.t / self.max_steps
        return o

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "state" in options:  # [ax, ay, gx, gy, px, py, cooldown, t]
            self.ax, self.ay, self.gx, self.gy, self.px, self.py, self.cd, self.t = map(
                int, options["state"])
        else:
            cells = self.np_random.choice(self.G * self.G, 3, replace=False)
            (self.ay, self.ax), (self.gy, self.gx), (self.py, self.px) = (
                divmod(int(c), self.G) for c in cells)
            self.cd, self.t = 0, 0
        self._ep_true = self._ep_proxy = 0.0
        self._ep_pick = 0
        return self._obs(), {}

    def step(self, action):
        dx, dy = _MOVES[int(action)]
        nx = min(max(self.ax + dx, 0), self.G - 1)
        ny = min(max(self.ay + dy, 0), self.G - 1)
        moved = (nx, ny) != (self.ax, self.ay)
        self.ax, self.ay = nx, ny
        self.t += 1
        true_r, bonus, picked = -self.step_penalty, 0.0, False
        if self.cd == 0 and moved and (nx, ny) == (self.px, self.py):
            picked, bonus, self.cd = True, self.pickup_reward, self.respawn_delay
            self._ep_pick += 1
        elif self.cd > 0:
            self.cd -= 1
        goal = (nx, ny) == (self.gx, self.gy)
        if goal:
            true_r += self.goal_reward
        proxy_r = true_r + bonus
        self._ep_true += true_r
        self._ep_proxy += proxy_r
        trunc = (not goal) and self.t >= self.max_steps
        info = {"true_reward": true_r, "proxy_reward": proxy_r, "picked": picked}
        if goal or trunc:
            info.update(ep_true_return=self._ep_true, ep_proxy_return=self._ep_proxy,
                        ep_length=self.t, ep_pickups=self._ep_pick, ep_goal=goal)
        return self._obs(), (proxy_r if self.misspecified else true_r), goal, trunc, info
