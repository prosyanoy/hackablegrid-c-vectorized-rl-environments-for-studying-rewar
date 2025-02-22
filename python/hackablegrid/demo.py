"""End-to-end experiment: train on both rewards, detect hacking, draw the exploit.

    python -m hackablegrid.demo            # full run (a few minutes on CPU)
    HG_TOTAL_STEPS=300000 python -m hackablegrid.demo   # quick smoke run
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import Settings
from .detector import detect, rollout
from .ppo import PPOConfig, train, save
from .vector_env import HackableGridVectorEnv
from .viz import animate_exploit, plot_trajectories, plot_training


def main():
    s = Settings.from_env()
    out = Path(s.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, s.num_threads))
    histories, reports, trajs = {}, {}, {}
    for variant in ("intended", "misspecified"):
        env = HackableGridVectorEnv(s.num_envs, variant, s.grid_size, s.max_steps,
                                    s.num_threads, s.seed, pickup_reward=s.pickup_reward)
        cfg = PPOConfig(total_steps=s.total_steps, num_envs=s.num_envs, seed=s.seed)
        model, hist = train(env, cfg)
        model.eval()
        save(model, hist, cfg, out / variant)
        histories[variant] = hist
        gen = torch.Generator().manual_seed(s.seed)
        trajs[variant] = rollout(lambda o: model.act(o, generator=gen), 128,
                                 grid_size=s.grid_size, max_steps=s.max_steps,
                                 pickup_reward=s.pickup_reward)
        reports[variant] = detect(trajs[variant], pickup_reward=s.pickup_reward)
    # Control: a uniformly random policy should not be flagged.
    rng = np.random.default_rng(s.seed)
    rnd = rollout(lambda o: rng.integers(0, 5, len(o)), 128, grid_size=s.grid_size,
                  max_steps=s.max_steps, pickup_reward=s.pickup_reward)
    reports["random"] = detect(rnd, pickup_reward=s.pickup_reward)

    for k, r in reports.items():
        print(f"\n== agent trained on: {k}\n{r.verdict}")
    (out / "detector_report.json").write_text(
        json.dumps({k: r.to_dict() for k, r in reports.items()}, indent=2))
    plot_training(histories, out / "training_curves.png")
    pick = lambda v: (trajs[v], reports[v].exploit_episodes[:3] or list(range(3)))
    plot_trajectories({"intended-reward agent": (trajs["intended"], list(range(3))),
                       "misspecified-reward agent (worst exploits)": pick("misspecified")},
                      s.grid_size, out / "exploit_trajectories.png")
    worst = reports["misspecified"].exploit_episodes[0]
    animate_exploit({"intended-reward agent": trajs["intended"][0],
                     "misspecified-reward agent": trajs["misspecified"][worst]},
                    s.grid_size, out / "exploit.gif", s.pickup_reward)
    print(f"\nartifacts written to {out}/")


if __name__ == "__main__":
    main()
