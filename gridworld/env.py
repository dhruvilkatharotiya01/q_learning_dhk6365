"""Grid world environment with the Gymnasium API.

The agent starts in the top-left cell and must reach the goal. Obstacle
("hell") cells cost -10 and block the move, the black hole costs -50 and ends
the episode, the goal gives +100 and ends the episode, and every other step
costs -1, so the shortest safe path has the highest return.

The environment only needs NumPy. If Gymnasium is installed it subclasses
``gymnasium.Env`` and exposes proper ``action_space``/``observation_space``.
Rendering with Pygame is optional (see ``render.py``).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

try:  # optional: only needed to plug the env into Gymnasium tooling
    import gymnasium as gym

    _Base = gym.Env
except ImportError:  # pragma: no cover - exercised when gymnasium is absent
    gym = None
    _Base = object

Cell = tuple[int, int]

ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}  # up, down, right, left
ACTION_NAMES = ("Up", "Down", "Right", "Left")

REWARD_GOAL = 100
REWARD_BLACK_HOLE = -50
REWARD_OBSTACLE = -10
REWARD_STEP = -1


class GridWorldEnv(_Base):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        size: int = 6,
        goal: Cell = (5, 5),
        obstacles: Iterable[Cell] = ((1, 1), (2, 3), (3, 1), (4, 4), (0, 5)),
        black_hole: Cell | None = (5, 2),
        start: Cell = (0, 0),
        max_steps: int = 200,
        render_mode: str | None = None,
    ) -> None:
        self.size = size
        self.goal = tuple(goal)
        self.obstacles = {tuple(o) for o in obstacles}
        self.black_hole = tuple(black_hole) if black_hole is not None else None
        self.start = tuple(start)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.n_actions = len(ACTIONS)
        self._check_layout()
        self.state: Cell = self.start
        self.steps = 0
        self._renderer = None
        if gym is not None:
            self.action_space = gym.spaces.Discrete(self.n_actions)
            self.observation_space = gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int64)

    def _check_layout(self) -> None:
        cells = [self.goal, self.start, *self.obstacles] + ([self.black_hole] if self.black_hole else [])
        for r, c in cells:
            if not (0 <= r < self.size and 0 <= c < self.size):
                raise ValueError(f"Cell {(r, c)} is outside the {self.size}x{self.size} grid")
        if self.start in self.obstacles or self.goal in self.obstacles:
            raise ValueError("Start and goal must not be obstacles")

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if gym is not None:
            super().reset(seed=seed)
        self.state = self.start
        self.steps = 0
        return np.array(self.state), {}

    def step(self, action: int):
        if action not in ACTIONS:
            raise ValueError(f"Invalid action {action}")
        dr, dc = ACTIONS[int(action)]
        r = min(max(self.state[0] + dr, 0), self.size - 1)  # walls keep the agent inside the grid
        c = min(max(self.state[1] + dc, 0), self.size - 1)
        nxt = (r, c)
        self.steps += 1

        terminated = False
        if nxt == self.goal:
            reward, terminated = REWARD_GOAL, True
        elif nxt in self.obstacles:
            reward, nxt = REWARD_OBSTACLE, self.state  # bounce back
        elif nxt == self.black_hole:
            reward, terminated = REWARD_BLACK_HOLE, True
        else:
            reward = REWARD_STEP
        self.state = nxt
        truncated = not terminated and self.steps >= self.max_steps
        if self.render_mode == "human":
            self.render()
        return np.array(self.state), reward, terminated, truncated, {"action": ACTION_NAMES[int(action)]}

    def render(self) -> None:
        if self._renderer is None:
            from .render import PygameRenderer  # imported lazily: pygame is optional

            self._renderer = PygameRenderer(self)
        self._renderer.draw()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
