"""Matplotlib visualisations of exploit trajectories and training curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .detector import Trajectory

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERMILION = "#0072B2", "#E69F00", "#009E73", "#D55E00"


def _draw(ax, traj: Trajectory, G: int, title: str):
    visits = np.zeros((G, G))
    for x, y in traj.positions:
        visits[y, x] += 1
    ax.imshow(visits, cmap="Blues", origin="upper", extent=(-.5, G - .5, G - .5, -.5), vmin=0)
    ax.set_xticks(np.arange(-.5, G, 1), minor=True)
    ax.set_yticks(np.arange(-.5, G, 1), minor=True)
    ax.grid(which="minor", color="#cccccc", lw=0.5)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    p = traj.positions + np.random.default_rng(0).uniform(-.12, .12, traj.positions.shape)
    ax.plot(p[:, 0], p[:, 1], "-", color="#555555", lw=1, alpha=.7)
    ax.scatter(p[0, 0], p[0, 1], s=70, color="black", marker="o", zorder=4, label="start")
    ax.scatter(*traj.goal, s=260, color=GREEN, marker="*", zorder=3, label="goal")
    ax.scatter(*traj.pickup, s=130, color=ORANGE, marker="D", zorder=3, label="pickup")
    ax.set_title(title, fontsize=9)


def plot_trajectories(groups: dict, G: int, path: Path, per_group: int = 3):
    """groups: {label: (list[Trajectory], indices)} -> one row per label."""
    fig, axes = plt.subplots(len(groups), per_group, figsize=(3.2 * per_group, 3.7 * len(groups)),
                             squeeze=False)
    for r, (label, (trajs, idx)) in enumerate(groups.items()):
        for c in range(per_group):
            ax = axes[r][c]
            if c >= len(idx):
                ax.axis("off")
                continue
            t = trajs[idx[c]]
            _draw(ax, t, G, f"{label}\ntrue {t.true_return:+.2f} | proxy {t.proxy_return:+.2f} | "
                            f"pickups {len(t.pickup_steps)}")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=3.0)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_training(histories: dict, path: Path):
    """histories: {label: history rows}. Left: true return; right: proxy return."""
    colors = {k: c for k, c in zip(histories, (BLUE, VERMILION, GREEN, ORANGE))}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    for name, key, ax in (("True objective return", "true_return", axes[0]),
                          ("Proxy (misspecified) return", "proxy_return", axes[1])):
        for label, h in histories.items():
            ax.plot([r["steps"] for r in h], [r[key] for r in h], color=colors[label], label=label, lw=1.8)
        ax.set_title(name)
        ax.set_xlabel("environment steps")
        ax.grid(alpha=.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, title="trained on")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def animate_exploit(groups: dict, G: int, path: Path, pickup_reward: float, step_penalty=0.01,
                    goal_reward=1.0, frames: int = 40, fps: int = 8):
    """groups: {label: Trajectory}. Side-by-side GIF with running true/proxy returns."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 4.0), squeeze=False)
    axes = axes[0]
    arts = []
    for ax, (label, t) in zip(axes, groups.items()):
        ax.set_xlim(-.5, G - .5); ax.set_ylim(G - .5, -.5); ax.set_aspect("equal")
        ax.set_xticks(np.arange(-.5, G, 1)); ax.set_yticks(np.arange(-.5, G, 1))
        ax.grid(color="#cccccc", lw=.5); ax.tick_params(length=0, labelbottom=False, labelleft=False)
        ax.scatter(*t.goal, s=300, color=GREEN, marker="*", zorder=2)
        ax.scatter(*t.pickup, s=150, color=ORANGE, marker="D", zorder=2)
        trail, = ax.plot([], [], "-", color="#777777", lw=1.2, alpha=.8, zorder=3)
        dot, = ax.plot([], [], "o", color=BLUE, ms=12, zorder=4)
        arts.append((ax, label, t, trail, dot))

    def update(f):
        out = []
        for ax, label, t, trail, dot in arts:
            k = min(f, len(t.positions) - 1)
            p = t.positions[:k + 1]
            trail.set_data(p[:, 0], p[:, 1]); dot.set_data([p[-1, 0]], [p[-1, 1]])
            n_pick = sum(1 for s in t.pickup_steps if s < k)
            true_r = -step_penalty * k + (goal_reward if t.reached_goal and k == len(t.positions) - 1 else 0)
            ax.set_title(f"{label}\nstep {k} | true {true_r:+.2f} | proxy {true_r + pickup_reward * n_pick:+.2f}", fontsize=9)
            out += [trail, dot]
        return out

    FuncAnimation(fig, update, frames=frames, blit=False).save(path, writer=PillowWriter(fps=fps), dpi=70)
    plt.close(fig)


def plot_benchmark(rows: dict, py: dict, threads: list, path: Path):
    """rows: {num_envs: [steps/s per thread count]}, py: {num_envs: steps/s}."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cols = [BLUE, VERMILION, GREEN, ORANGE]
    for (n, rates), c in zip(rows.items(), cols):
        ax.plot(threads, rates, "o-", color=c, lw=2, label=f"C++ engine, {n:,} envs")
    base = float(np.mean(list(py.values())))
    ax.axhline(base, color="#555555", ls="--", lw=1.5, label=f"pure-Python Gymnasium (~{base / 1e3:.0f}k steps/s)")
    ax.set_yscale("log"); ax.set_xscale("log", base=2); ax.set_xticks(threads); ax.set_xticklabels(threads)
    ax.set_xlabel("C++ threads"); ax.set_ylabel("environment steps / second")
    ax.set_title("HackableGrid throughput vs. pure-Python Gymnasium")
    ax.grid(alpha=.3, which="both"); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
