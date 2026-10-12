"""Tabular Q-learning with epsilon-greedy exploration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .env import GridWorldEnv


@dataclass
class Config:
    episodes: int = 1000
    alpha: float = 0.1  # learning rate
    gamma: float = 0.99  # discount factor
    epsilon: float = 1.0  # initial exploration rate
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995  # multiplied after every episode
    seed: int = 0


@dataclass
class History:
    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)  # goal | black_hole | timeout
    converged_at: int | None = None  # first checked episode from which the greedy path stays optimal


def train(env: GridWorldEnv, cfg: Config | None = None, check_every: int = 10) -> tuple[np.ndarray, History]:
    """Learn a Q-table of shape (size, size, n_actions).

    Every ``check_every`` episodes the greedy policy is rolled out (on a copy of the
    environment) to record when it first becomes, and stays, optimal.
    """
    cfg = cfg or Config()
    optimum = shortest_path_length(env)
    probe = GridWorldEnv(env.size, env.goal, env.obstacles, env.black_hole, env.start, env.max_steps)
    rng = np.random.default_rng(cfg.seed)
    q = np.zeros((env.size, env.size, env.n_actions))
    hist = History()
    eps = cfg.epsilon

    for episode in range(cfg.episodes):
        obs, _ = env.reset(seed=cfg.seed + episode)
        s = tuple(obs)
        total, done, outcome = 0.0, False, "timeout"
        while not done:
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:  # break ties randomly so early greedy steps don't always go "up"
                best = np.flatnonzero(q[s] == q[s].max())
                a = int(rng.choice(best))
            obs, r, terminated, truncated, _ = env.step(a)
            s2 = tuple(obs)
            target = r + (0.0 if terminated else cfg.gamma * q[s2].max())
            q[s][a] += cfg.alpha * (target - q[s][a])
            total += r
            s = s2
            done = terminated or truncated
            if terminated:
                outcome = "goal" if s == env.goal else "black_hole"
        hist.returns.append(total)
        hist.lengths.append(env.steps)
        hist.epsilons.append(eps)
        hist.outcomes.append(outcome)
        eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)
        if check_every and (episode + 1) % check_every == 0:
            path, _, reached = greedy_path(probe, q)
            optimal = reached and len(path) - 1 == optimum
            if optimal and hist.converged_at is None:
                hist.converged_at = episode + 1
            elif not optimal:
                hist.converged_at = None
    return q, hist


def greedy_path(env: GridWorldEnv, q: np.ndarray) -> tuple[list[tuple[int, int]], float, bool]:
    """Follow the learned policy from the start. Returns (cells visited, return, reached goal)."""
    obs, _ = env.reset()
    path, total = [tuple(obs)], 0.0
    while True:
        obs, r, terminated, truncated, _ = env.step(int(np.argmax(q[tuple(obs)])))
        path.append(tuple(obs))
        total += r
        if terminated or truncated:
            return path, total, bool(terminated and tuple(obs) == env.goal)


def shortest_path_length(env: GridWorldEnv) -> int | None:
    """Breadth-first search over safe cells: the optimum the agent should learn."""
    from collections import deque

    blocked = env.obstacles | ({env.black_hole} if env.black_hole else set())
    queue, seen = deque([(env.start, 0)]), {env.start}
    while queue:
        (r, c), d = queue.popleft()
        if (r, c) == env.goal:
            return d
        for dr, dc in ((-1, 0), (1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < env.size and 0 <= n[1] < env.size and n not in blocked and n not in seen:
                seen.add(n)
                queue.append((n, d + 1))
    return None
