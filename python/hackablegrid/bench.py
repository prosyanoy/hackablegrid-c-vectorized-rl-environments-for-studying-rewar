"""Throughput benchmark: C++ engine vs pure-Python Gymnasium, and thread scaling.

    python -m hackablegrid.bench            # prints a Markdown table
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from gymnasium.vector import SyncVectorEnv

from .py_env import PyGridEnv
from .vector_env import HackableGridVectorEnv


def _time_steps(step, n_envs, secs, rng, make_actions):
    step(make_actions())  # warm-up
    steps, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < secs:
        for _ in range(10):
            step(make_actions())
        steps += 10 * n_envs
    return steps / (time.perf_counter() - t0)


def bench_native(n_envs, threads, secs):
    # Drive the native engine directly (zero-copy buffers) so the number reflects the
    # engine; the Gymnasium wrapper adds one observation copy per step on top of this.
    env = HackableGridVectorEnv(n_envs, "misspecified", num_threads=threads, seed=0)._core
    env.reset(0)
    rng = np.random.default_rng(0)
    pool = [rng.integers(0, 5, n_envs, dtype=np.int32) for _ in range(16)]
    counter = [0]

    def actions():
        counter[0] += 1
        return pool[counter[0] % 16]
    return _time_steps(env.step, n_envs, secs, rng, actions)


def bench_python(n_envs, secs):
    env = SyncVectorEnv([lambda: PyGridEnv("misspecified") for _ in range(n_envs)])
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    return _time_steps(env.step, n_envs, secs, rng, lambda: rng.integers(0, 5, n_envs))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--envs", type=int, nargs="+", default=[16, 256, 4096])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--plot", type=str, default=None, help="write a PNG chart to this path")
    a = ap.parse_args(argv)
    py = {n: bench_python(min(n, 256), a.secs) for n in a.envs}  # python cost is linear in n
    print("| num_envs | Gymnasium SyncVectorEnv (py) | " +
          " | ".join(f"C++ {t}T" for t in a.threads) + " | best speedup |")
    print("|---:|---:|" + "---:|" * len(a.threads) + "---:|")
    rows = {}
    for n in a.envs:
        rates = rows[n] = [bench_native(n, t, a.secs) for t in a.threads]
        print(f"| {n} | {py[n]:,.0f} | " + " | ".join(f"{r:,.0f}" for r in rates) +
              f" | {max(rates) / py[n]:.0f}x |")
    print("\nValues are environment steps per second (higher is better).")
    if a.plot:
        from .viz import plot_benchmark
        plot_benchmark(rows, py, a.threads, a.plot)


if __name__ == "__main__":
    main()
