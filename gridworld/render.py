"""Pygame rendering with the original sprites (optional dependency)."""

from __future__ import annotations

import time
from pathlib import Path

import pygame

ASSETS = Path(__file__).resolve().parent.parent / "assets"
CELL = 100


def _load(name: str, size: tuple[int, int]) -> pygame.Surface:
    return pygame.transform.scale(pygame.image.load(str(ASSETS / name)), size)


class PygameRenderer:
    def __init__(self, env, fps: float = 10.0) -> None:
        pygame.init()
        self.env = env
        self.delay = 1.0 / fps
        side = CELL * env.size
        self.screen = pygame.display.set_mode((side, side))
        pygame.display.set_caption("Grid world Q-learning")
        cell = (CELL, CELL)
        self.background = _load("bgMain.jpeg", (side, side))
        self.background.set_alpha(128)
        self.agent = _load("agent.jpeg", cell)
        self.goal = _load("goal.jpeg", cell)
        self.black_hole = _load("bh.jpeg", cell)
        self.obstacles = [_load(f"obs{i}.jpeg", cell) for i in range(1, 6)]

    def _blit(self, image: pygame.Surface, cell: tuple[int, int]) -> None:
        self.screen.blit(image, (cell[1] * CELL, cell[0] * CELL))

    def draw(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit
        self.screen.fill((255, 255, 255))
        self.screen.blit(self.background, (0, 0))
        self._blit(self.goal, self.env.goal)
        if self.env.black_hole:
            self._blit(self.black_hole, self.env.black_hole)
        for i, cell in enumerate(sorted(self.env.obstacles)):
            self._blit(self.obstacles[i % len(self.obstacles)], cell)
        self._blit(self.agent, self.env.state)
        pygame.display.flip()
        time.sleep(self.delay)

    def close(self) -> None:
        pygame.quit()
