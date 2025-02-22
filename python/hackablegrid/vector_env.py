"""Gymnasium-compatible vector environment backed by the C++ engine."""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from . import _core

VARIANTS = {"intended": 0, "misspecified": 1}
NUM_ACTIONS = 5
# Columns of the state matrix returned by get_state()
STATE_COLS = ("ax", "ay", "gx", "gy", "px", "py", "cooldown", "t")


class HackableGridVectorEnv(VectorEnv):
    """Batched gridworld stepping ``num_envs`` instances in C++ threads.

    Sub-environments auto-reset in the same step (``autoreset_mode=SAME_STEP``): when
    ``terminated|truncated`` is set, the returned observation already belongs to the new
    episode, and ``info['ep_*']`` carry the finished episode's statistics.

    ``reward`` is the *training* reward of the chosen variant; ``info['true_reward']`` and
    ``info['proxy_reward']`` always report both signals so hacking can be measured.
    """

    metadata = {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP, "render_modes": []}

    def __init__(self, num_envs: int, variant: str = "intended", grid_size: int = 8,
                 max_steps: int = 64, num_threads: int = 1, seed: int = 0,
                 goal_reward: float = 1.0, step_penalty: float = 0.01,
                 pickup_reward: float = 0.3, respawn_delay: int = 1):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
        self.num_envs = num_envs
        self.variant = variant
        self._core = _core.NativeVecEnv(num_envs, grid_size, max_steps, VARIANTS[variant],
                                        num_threads, seed, goal_reward, step_penalty,
                                        pickup_reward, respawn_delay)
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.pickup_reward = pickup_reward
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.single_observation_space = spaces.Box(0.0, 1.0, (self._core.obs_dim,), np.float32)
        self.single_action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.spec = None
        self.render_mode = None

    @property
    def obs_dim(self) -> int:
        return self._core.obs_dim

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs = self._core.reset(seed)
        return obs.copy(), {}

    def step(self, actions):
        actions = np.ascontiguousarray(actions, dtype=np.int32)
        obs, rew, term, trunc, info = self._core.step(actions)
        info = {k: v.copy() for k, v in info.items()}
        return obs.copy(), rew.copy(), term.astype(bool), trunc.astype(bool), info

    def get_state(self) -> np.ndarray:
        """int32 array (num_envs, 8) with columns STATE_COLS."""
        return self._core.get_state()

    def set_state(self, state: np.ndarray) -> np.ndarray:
        return self._core.set_state(np.ascontiguousarray(state, dtype=np.int32)).copy()
