import tempfile
import unittest
from pathlib import Path

import numpy as np

from gridworld.env import REWARD_BLACK_HOLE, REWARD_GOAL, REWARD_OBSTACLE, REWARD_STEP, GridWorldEnv
from gridworld.q_learning import Config, greedy_path, shortest_path_length, train

UP, DOWN, RIGHT, LEFT = range(4)


class TestEnv(unittest.TestCase):
    def setUp(self):
        self.env = GridWorldEnv()
        self.env.reset()

    def test_reset_returns_start_and_info(self):
        obs, info = self.env.reset(seed=1)
        self.assertEqual(tuple(obs), (0, 0))
        self.assertEqual(info, {})

    def test_wall_keeps_agent_inside(self):
        obs, r, terminated, truncated, _ = self.env.step(UP)
        self.assertEqual((tuple(obs), r, terminated, truncated), ((0, 0), REWARD_STEP, False, False))

    def test_obstacle_bounces_back_with_penalty(self):
        self.env.step(RIGHT)  # (0, 1)
        obs, r, terminated, _, _ = self.env.step(DOWN)  # (1, 1) is an obstacle
        self.assertEqual((tuple(obs), r, terminated), ((0, 1), REWARD_OBSTACLE, False))

    def test_black_hole_ends_episode(self):
        env = GridWorldEnv(start=(4, 2))
        env.reset()
        _, r, terminated, _, _ = env.step(DOWN)
        self.assertEqual((r, terminated), (REWARD_BLACK_HOLE, True))

    def test_goal_ends_episode(self):
        env = GridWorldEnv(start=(5, 4))
        env.reset()
        _, r, terminated, _, _ = env.step(RIGHT)
        self.assertEqual((r, terminated), (REWARD_GOAL, True))

    def test_episode_is_truncated(self):
        env = GridWorldEnv(max_steps=3)
        env.reset()
        results = [env.step(UP) for _ in range(3)]
        self.assertEqual([res[3] for res in results], [False, False, True])

    def test_invalid_layout_and_action(self):
        with self.assertRaises(ValueError):
            GridWorldEnv(goal=(9, 9))
        with self.assertRaises(ValueError):
            GridWorldEnv(obstacles=[(0, 0)])
        with self.assertRaises(ValueError):
            self.env.step(7)

    def test_shortest_path(self):
        self.assertEqual(shortest_path_length(self.env), 10)
        self.assertIsNone(
            shortest_path_length(
                GridWorldEnv(size=3, goal=(2, 2), obstacles=[(1, 2), (2, 1)], black_hole=None)
            )
        )


class TestQLearning(unittest.TestCase):
    def test_learns_optimal_policy(self):
        env = GridWorldEnv()
        q, hist = train(env, Config(episodes=300, alpha=0.3, seed=0))
        path, ret, reached = greedy_path(env, q)
        self.assertTrue(reached)
        self.assertEqual(len(path) - 1, shortest_path_length(env))
        self.assertEqual(ret, REWARD_GOAL + (len(path) - 2) * REWARD_STEP)
        self.assertIsNotNone(hist.converged_at)
        self.assertEqual(len(hist.returns), 300)

    def test_same_seed_same_result(self):
        q1, _ = train(GridWorldEnv(), Config(episodes=50, seed=3))
        q2, _ = train(GridWorldEnv(), Config(episodes=50, seed=3))
        np.testing.assert_array_equal(q1, q2)

    def test_terminal_states_are_never_updated(self):
        env = GridWorldEnv()
        q, _ = train(env, Config(episodes=200, seed=0))
        self.assertTrue(np.all(q[env.goal] == 0))
        self.assertTrue(np.all(q[env.black_hole] == 0))


class TestCLI(unittest.TestCase):
    def test_main_writes_results(self):
        from main import main

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(["--episodes", "150", "--alpha", "0.3", "--out", tmp]), 0)
            for name in ("q_table.npy", "summary.json", "training.png", "q_values.png", "policy.png"):
                self.assertTrue((Path(tmp) / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
