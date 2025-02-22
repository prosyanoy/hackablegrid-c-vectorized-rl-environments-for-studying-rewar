"""Detector and PPO tests."""
import numpy as np
import torch

from hackablegrid import HackableGridVectorEnv
from hackablegrid.detector import Trajectory, detect, rollout
from hackablegrid.ppo import PPOConfig, train


def _loop_policy(env_state_fn=None):
    """Scripted exploiter: walk to the pickup by reading the observation, then oscillate."""
    G = 8

    def act(obs):
        out = []
        for o in obs:
            a = int(np.argmax(o[:G * G])); p = o[2 * G * G:3 * G * G]
            ax, ay = a % G, a // G
            if p.sum() == 0:       # pickup on cooldown (we just collected it): step away
                out.append(1 if ay < G - 1 else 0)
                continue
            pi = int(np.argmax(p)); px, py = pi % G, pi // G
            out.append(3 if px > ax else 2 if px < ax else 1 if py > ay else 0)
        return np.array(out)
    return act


def test_detector_flags_scripted_exploiter():
    trajs = rollout(_loop_policy(), 64, pickup_reward=0.5)
    rep = detect(trajs, pickup_reward=0.5)
    assert rep.hacking_detected and rep.divergence_flag and rep.behaviour_flag
    assert rep.proxy_return > rep.honest_proxy_ceiling > rep.true_return
    assert rep.mean_pickups > 10


def test_detector_ignores_random_and_honest_agents():
    rng = np.random.default_rng(0)
    rep = detect(rollout(lambda o: rng.integers(0, 5, len(o)), 128, pickup_reward=0.5), pickup_reward=0.5)
    assert not rep.hacking_detected

    def honest(obs):  # walks straight to the goal
        G = 8
        acts = []
        for o in obs:
            a, g = int(np.argmax(o[:G * G])), int(np.argmax(o[G * G:2 * G * G]))
            ax, ay, gx, gy = a % G, a // G, g % G, g // G
            acts.append(3 if gx > ax else 2 if gx < ax else 1 if gy > ay else 0)
        return np.array(acts)
    rep = detect(rollout(honest, 128, pickup_reward=0.5), pickup_reward=0.5)
    assert rep.success_rate == 1.0 and not rep.hacking_detected
    assert abs(rep.true_return - rep.optimal_true_return) < 1e-4


def test_rollout_trajectories_are_consistent():
    trajs = rollout(_loop_policy(), 8)
    for t in trajs:
        assert t.reached_goal == bool((t.positions[-1] == t.goal).all())  # path ends where reported
        assert len(t.positions) == (len(t.positions) if t.reached_goal else 65)
        assert t.proxy_return >= t.true_return
        assert len(t.pickup_steps) >= 10 or t.reached_goal
    assert any(not t.reached_goal for t in trajs)
    t = Trajectory(np.array([[0, 0], [1, 0], [0, 0], [1, 0], [0, 0]]), (5, 5), (1, 0), [0], 0, 0, False, 0)
    assert t.loop_ratio == 1.0


def test_ppo_smoke_improves_intended_reward():
    torch.manual_seed(0)
    env = HackableGridVectorEnv(64, "intended", grid_size=5, max_steps=30, seed=0)
    cfg = PPOConfig(total_steps=150_000, num_envs=64, seed=0, rollout_len=32, lr=5e-4)
    _, hist = train(env, cfg, log=lambda *_: None)
    assert np.mean([h["true_return"] for h in hist[-5:]]) > np.mean([h["true_return"] for h in hist[:3]]) + 0.1
    assert hist[-1]["success"] > 0.5
