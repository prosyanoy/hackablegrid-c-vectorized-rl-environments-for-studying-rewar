"""CLI: train one PPO agent.  python -m hackablegrid.train --variant misspecified"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import Settings
from .ppo import PPOConfig, train, save
from .vector_env import HackableGridVectorEnv


def main(argv=None):
    s = Settings.from_env()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=["intended", "misspecified"], default="intended")
    ap.add_argument("--total-steps", type=int, default=s.total_steps)
    ap.add_argument("--num-envs", type=int, default=s.num_envs)
    ap.add_argument("--threads", type=int, default=s.num_threads)
    ap.add_argument("--seed", type=int, default=s.seed)
    ap.add_argument("--out", type=Path, default=None, help="default: <HG_OUT_DIR>/<variant>")
    a = ap.parse_args(argv)
    torch.set_num_threads(max(1, a.threads))
    env = HackableGridVectorEnv(a.num_envs, a.variant, s.grid_size, s.max_steps, a.threads, a.seed,
                                pickup_reward=s.pickup_reward)
    cfg = PPOConfig(total_steps=a.total_steps, num_envs=a.num_envs, seed=a.seed)
    model, hist = train(env, cfg)
    out = a.out or Path(s.out_dir) / a.variant
    save(model, hist, cfg, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
