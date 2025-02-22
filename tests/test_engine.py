"""Correctness tests for the native engine: dynamics, determinism, seeding, API."""
import numpy as np
import pytest

from hackablegrid import HackableGridVectorEnv
from hackablegrid.py_env import PyGridEnv


def rollout_trace(num_threads, seed=7, n=257, steps=120):
    env = HackableGridVectorEnv(n, "misspecified", num_threads=num_threads, seed=seed)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(99)
    trace = [obs]
    for _ in range(steps):
        obs, r, te, tr, _ = env.step(rng.integers(0, 5, n))
        trace += [obs, r, te, tr]
    return trace


def test_same_seed_is_deterministic():
    a, b = rollout_trace(1), rollout_trace(1)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_thread_count_does_not_change_results():
    a, b = rollout_trace(1), rollout_trace(4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_different_seeds_differ():
    a, b = rollout_trace(1, seed=1)[0], rollout_trace(1, seed=2)[0]
    assert not np.array_equal(a, b)


def test_envs_in_batch_are_independent_streams():
    env = HackableGridVectorEnv(64, num_threads=1, seed=3)
    env.reset(seed=3)
    assert len({tuple(r) for r in env.get_state()[:, :6]}) > 32


def test_reset_seed_reproduces_layouts():
    env = HackableGridVectorEnv(32, seed=0)
    env.reset(seed=5)
    s1 = env.get_state()
    env.step(np.zeros(32, int))
    env.reset(seed=5)
    assert np.array_equal(s1, env.get_state())


def _matching_envs(n=64, steps=300, variant="misspecified"):
    nat = HackableGridVectorEnv(n, variant, seed=1)
    nat.reset(seed=1)
    state = nat.get_state()
    pys = [PyGridEnv(variant) for _ in range(n)]
    for i, e in enumerate(pys):
        e.reset(options={"state": state[i]})
    return nat, pys


@pytest.mark.parametrize("variant", ["intended", "misspecified"])
def test_matches_pure_python_reference(variant):
    nat, pys = _matching_envs(variant=variant)
    rng = np.random.default_rng(0)
    for _ in range(300):
        acts = rng.integers(0, 5, len(pys))
        obs, rew, term, trunc, info = nat.step(acts)
        for i, e in enumerate(pys):
            o, r, te, tr, inf = e.step(acts[i])
            assert r == pytest.approx(rew[i], abs=1e-6)
            assert (te, tr) == (term[i], trunc[i])
            assert inf["picked"] == bool(info["picked"][i])
            assert inf["true_reward"] == pytest.approx(info["true_reward"][i], abs=1e-6)
            if te or tr:
                assert inf["ep_pickups"] == info["ep_pickups"][i]
                assert inf["ep_true_return"] == pytest.approx(info["ep_true_return"][i], abs=1e-5)
                # re-sync the reference with the engine's auto-reset layout
                e.reset(options={"state": nat.get_state()[i]})
            else:
                assert np.array_equal(o, obs[i])


def test_loop_exploit_dominates_proxy_but_not_true_reward():
    env = HackableGridVectorEnv(1, "misspecified", max_steps=64, pickup_reward=0.5)
    env.reset(seed=0)
    env.set_state(np.array([[0, 0, 7, 7, 1, 0, 0, 0]]))
    tot_train = tot_true = 0.0
    picks = 0
    for t in range(40):
        _, r, *_ , info = env.step([3 if t % 2 == 0 else 2])
        tot_train += r[0]; tot_true += info["true_reward"][0]; picks += info["picked"][0]
    assert picks == 20
    assert tot_train == pytest.approx(tot_true + 10.0, abs=1e-4)
    assert tot_true < 0


def test_intended_variant_ignores_pickup():
    env = HackableGridVectorEnv(1, "intended")
    env.reset(seed=0)
    env.set_state(np.array([[0, 0, 7, 7, 1, 0, 0, 0]]))
    _, r, *_, info = env.step([3])
    assert info["picked"][0] == 1 and r[0] == pytest.approx(-0.01)


def test_goal_terminates_and_reports_episode_stats():
    env = HackableGridVectorEnv(1, "intended")
    env.reset(seed=0)
    env.set_state(np.array([[4, 4, 5, 4, 0, 0, 0, 0]]))
    _, r, term, trunc, info = env.step([3])
    assert term[0] and not trunc[0] and r[0] == pytest.approx(0.99)
    assert info["ep_goal"][0] and info["ep_length"][0] == 1


def test_timeout_truncates_and_autoresets():
    env = HackableGridVectorEnv(2, max_steps=10)
    env.reset(seed=0)
    env.set_state(np.array([[0, 0, 7, 7, 4, 4, 0, 0]] * 2))
    for t in range(10):
        _, _, term, trunc, info = env.step([4, 4])
    assert trunc.all() and not term.any()
    assert (info["ep_length"] == 10).all()
    assert (env.get_state()[:, 7] == 0).all()


def test_walls_clamp_movement():
    env = HackableGridVectorEnv(1)
    env.reset(seed=0)
    env.set_state(np.array([[0, 0, 7, 7, 4, 4, 0, 0]]))
    env.step([0]); env.step([2])
    assert tuple(env.get_state()[0, :2]) == (0, 0)


def test_observation_encoding():
    env = HackableGridVectorEnv(1, grid_size=5)
    env.reset(seed=0)
    obs = env.set_state(np.array([[1, 2, 3, 4, 0, 1, 0, 0]]))[0]
    plane = 25
    assert obs[2 * 5 + 1] == 1 and obs[plane + 4 * 5 + 3] == 1 and obs[2 * plane + 1 * 5 + 0] == 1
    assert obs.sum() == 3
    # after collection the pickup disappears from the observation until it respawns
    env.set_state(np.array([[0, 1, 3, 4, 1, 1, 0, 0]]))
    obs, *_ = env.step([3])
    assert obs[0, 2 * plane:3 * plane].sum() == 0


def test_gymnasium_vector_api():
    env = HackableGridVectorEnv(8)
    assert env.single_observation_space.shape == (193,)
    assert env.single_action_space.n == 5
    assert env.observation_space.shape == (8, 193)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
    assert rew.shape == term.shape == trunc.shape == (8,)
    assert term.dtype == bool


def test_step_returns_independent_copies():
    env = HackableGridVectorEnv(4)
    env.reset(seed=0)
    o1, *_ = env.step(np.ones(4, int))
    snap = o1.copy()
    env.step(np.zeros(4, int))
    assert np.array_equal(o1, snap)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        HackableGridVectorEnv(4, variant="nope")
    env = HackableGridVectorEnv(4)
    with pytest.raises(Exception):
        env.step(np.zeros(3, int))
    with pytest.raises(Exception):
        env.set_state(np.full((4, 8), 99))
