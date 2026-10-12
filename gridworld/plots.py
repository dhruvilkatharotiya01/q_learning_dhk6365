"""Training curves, Q-value heatmaps and the learned policy (matplotlib, saved to files)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files; works without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .env import ACTION_NAMES, GridWorldEnv  # noqa: E402
from .q_learning import History  # noqa: E402

ARROWS = {0: (0, -0.35), 1: (0, 0.35), 2: (0.35, 0), 3: (-0.35, 0)}  # dx, dy in plot coordinates


def _moving_average(x: list[float], window: int = 50) -> np.ndarray:
    window = max(1, min(window, len(x)))
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_training(hist: History, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    ax.plot(hist.returns, color="#c4cbe0", lw=0.8, label="episode return")
    ma = _moving_average(hist.returns)
    ax.plot(
        np.arange(len(ma)) + len(hist.returns) - len(ma), ma, color="#3b5998", lw=2, label="50-episode mean"
    )
    ax.set(title="Return per episode", xlabel="episode", ylabel="return")
    ax.legend(loc="lower right")

    ax = axes[1]
    ma = _moving_average(hist.lengths)
    ax.plot(hist.lengths, color="#c4cbe0", lw=0.8)
    ax.plot(np.arange(len(ma)) + len(hist.lengths) - len(ma), ma, color="#3b5998", lw=2)
    ax.set(title="Steps per episode", xlabel="episode", ylabel="steps", yscale="log")

    ax = axes[2]
    ax.plot(hist.epsilons, color="#e07b39", lw=2)
    ax.set(title="Exploration rate ε", xlabel="episode", ylabel="ε", ylim=(0, 1.05))
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _mark_special(ax, env: GridWorldEnv) -> None:
    for (r, c), label, color in (
        [(env.goal, "G", "#1a7f37")]
        + [(o, "H", "#b42318") for o in env.obstacles]
        + ([(env.black_hole, "B", "black")] if env.black_hole else [])
    ):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="white", zorder=2))
        ax.text(c, r, label, ha="center", va="center", color=color, weight="bold", fontsize=13, zorder=3)


def plot_q_values(q: np.ndarray, env: GridWorldEnv, path: Path) -> None:
    fig, axes = plt.subplots(1, env.n_actions, figsize=(4.2 * env.n_actions, 4))
    vmin, vmax = q.min(), q.max()
    for a, ax in enumerate(axes):
        im = ax.imshow(q[:, :, a], cmap="viridis", vmin=vmin, vmax=vmax)
        for r in range(env.size):
            for c in range(env.size):
                ax.text(c, r, f"{q[r, c, a]:.0f}", ha="center", va="center", color="white", fontsize=8)
        _mark_special(ax, env)
        ax.set(title=f"Q(s, {ACTION_NAMES[a]})", xticks=[], yticks=[])
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_policy(q: np.ndarray, env: GridWorldEnv, path_cells: list[tuple[int, int]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(q.max(axis=2), cmap="Blues", alpha=0.6)
    special = env.obstacles | {env.goal} | ({env.black_hole} if env.black_hole else set())
    for r in range(env.size):
        for c in range(env.size):
            if (r, c) in special:
                continue
            dx, dy = ARROWS[int(np.argmax(q[r, c]))]
            ax.arrow(c - dx / 2, r - dy / 2, dx, dy, head_width=0.15, color="#555", length_includes_head=True)
    ys, xs = zip(*path_cells, strict=True)
    ax.plot(xs, ys, color="#e07b39", lw=3, alpha=0.8, label=f"greedy path ({len(path_cells) - 1} steps)")
    _mark_special(ax, env)
    ax.set(title="Learned greedy policy", xticks=range(env.size), yticks=range(env.size))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
