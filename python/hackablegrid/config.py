"""Runtime configuration, read from environment variables (see .env.example)."""
import os
from dataclasses import dataclass


def _env(name, default, cast):
    raw = os.environ.get(name)
    return default if raw in (None, "") else cast(raw)


@dataclass
class Settings:
    num_envs: int = 256
    num_threads: int = max(1, os.cpu_count() or 1)
    seed: int = 0
    total_steps: int = 1_500_000
    grid_size: int = 8
    pickup_reward: float = 0.5
    max_steps: int = 64
    out_dir: str = "results"

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            num_envs=_env("HG_NUM_ENVS", d.num_envs, int),
            num_threads=_env("HG_NUM_THREADS", d.num_threads, int),
            seed=_env("HG_SEED", d.seed, int),
            total_steps=_env("HG_TOTAL_STEPS", d.total_steps, int),
            grid_size=_env("HG_GRID_SIZE", d.grid_size, int),
            pickup_reward=_env("HG_PICKUP_REWARD", d.pickup_reward, float),
            max_steps=_env("HG_MAX_STEPS", d.max_steps, int),
            out_dir=_env("HG_OUT_DIR", d.out_dir, str),
        )
