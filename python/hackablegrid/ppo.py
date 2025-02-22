"""Compact PPO (clipped objective, GAE) for the vectorized gridworld."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .vector_env import HackableGridVectorEnv, NUM_ACTIONS


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = NUM_ACTIONS, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def forward(self, obs):
        h = self.body(obs)
        return self.pi(h), self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False, generator=None) -> np.ndarray:
        logits, _ = self(torch.as_tensor(obs))
        if greedy:
            return logits.argmax(-1).numpy().astype(np.int32)
        probs = torch.softmax(logits, -1)
        return torch.multinomial(probs, 1, generator=generator).squeeze(-1).numpy().astype(np.int32)


@dataclass
class PPOConfig:
    total_steps: int = 1_500_000
    num_envs: int = 256
    rollout_len: int = 32
    epochs: int = 4
    minibatches: int = 8
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    seed: int = 0
    log_every: int = 10  # updates


def train(env: HackableGridVectorEnv, cfg: PPOConfig, log=print):
    """Train on ``env`` (whose ``variant`` decides the reward). Returns (model, history).

    History rows hold the mean *true* and *proxy* return of episodes finished in that
    update, so one run shows how the optimised proxy diverges from the true objective.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    n, T = env.num_envs, cfg.rollout_len
    model = ActorCritic(env.obs_dim, hidden=cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    updates = max(1, cfg.total_steps // (n * T))
    obs_buf = torch.zeros(T, n, env.obs_dim)
    act_buf = torch.zeros(T, n, dtype=torch.long)
    logp_buf, rew_buf, val_buf = (torch.zeros(T, n) for _ in range(3))
    done_buf, term_buf = torch.zeros(T, n), torch.zeros(T, n)

    obs, _ = env.reset(seed=cfg.seed)
    history, t0, steps = [], time.time(), 0
    for u in range(updates):
        for g in opt.param_groups:
            g["lr"] = cfg.lr * (1.0 - u / updates)
        ep = {"true": [], "proxy": [], "len": [], "goal": [], "pickups": []}
        for t in range(T):
            o = torch.as_tensor(obs)
            with torch.no_grad():
                logits, val = model(o)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = o, a, dist.log_prob(a), val
            obs, rew, term, trunc, info = env.step(a.numpy().astype(np.int32))
            rew_buf[t] = torch.as_tensor(rew)
            done = term | trunc
            done_buf[t] = torch.as_tensor(done, dtype=torch.float32)
            term_buf[t] = torch.as_tensor(term, dtype=torch.float32)
            if done.any():
                ep["true"].append(info["ep_true_return"][done])
                ep["proxy"].append(info["ep_proxy_return"][done])
                ep["len"].append(info["ep_length"][done])
                ep["goal"].append(info["ep_goal"][done])
                ep["pickups"].append(info["ep_pickups"][done])
            # Time-limit truncation: bootstrap from the value of the (reset) state is
            # inexact, so the cut-off is treated as an episode end like in classic PPO.
        steps += n * T
        with torch.no_grad():
            _, last_val = model(torch.as_tensor(obs))
        adv = torch.zeros(T, n)
        last = torch.zeros(n)
        for t in reversed(range(T)):
            nxt_val = last_val if t == T - 1 else val_buf[t + 1]
            nonterm = 1.0 - done_buf[t]
            delta = rew_buf[t] + cfg.gamma * nxt_val * nonterm - val_buf[t]
            last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
            adv[t] = last
        ret = adv + val_buf

        b_obs, b_act = obs_buf.reshape(-1, env.obs_dim), act_buf.reshape(-1)
        b_logp, b_adv, b_ret = logp_buf.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        N = b_obs.shape[0]
        mb = N // cfg.minibatches
        for _ in range(cfg.epochs):
            perm = torch.randperm(N)
            for s in range(0, N - mb + 1, mb):
                idx = perm[s:s + mb]
                logits, v = model(b_obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(b_act[idx]) - b_logp[idx]).exp()
                a_ = b_adv[idx]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pg = -torch.min(ratio * a_, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a_).mean()
                vl = 0.5 * (v - b_ret[idx]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

        if ep["true"]:
            cat = {k: np.concatenate(v) for k, v in ep.items()}
            row = {"update": u, "steps": steps, "episodes": int(len(cat["true"])),
                   "true_return": float(cat["true"].mean()),
                   "proxy_return": float(cat["proxy"].mean()),
                   "ep_length": float(cat["len"].mean()),
                   "success": float(cat["goal"].mean()),
                   "pickups": float(cat["pickups"].mean()),
                   "entropy": float(ent.detach()), "sps": steps / (time.time() - t0)}
            history.append(row)
            if u % cfg.log_every == 0 or u == updates - 1:
                log(f"[{env.variant}] upd {u + 1}/{updates} steps {steps:,} "
                    f"true {row['true_return']:+.2f} proxy {row['proxy_return']:+.2f} "
                    f"success {row['success']:.2f} pickups {row['pickups']:.1f} "
                    f"({row['sps']:,.0f} sps)")
    return model, history


def save(model: ActorCritic, history, cfg: PPOConfig, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "obs_dim": model.body[0].in_features,
                "hidden": cfg.hidden}, path / "model.pt")
    (path / "history.json").write_text(json.dumps({"config": asdict(cfg), "history": history}))


def load(path: Path) -> ActorCritic:
    ck = torch.load(path / "model.pt", weights_only=True)
    m = ActorCritic(ck["obs_dim"], hidden=ck["hidden"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()
