import pygame
import numpy as np
import gymnasium as gym
import time
import os

script_dir = os.path.dirname(__file__)

AGENT_IMG_PATH = os.path.join(script_dir, "agent.jpeg")
GOAL_IMG_PATH = os.path.join(script_dir, "goal.jpeg")
BACKGROUND_IMG = os.path.join(script_dir, "bgMain.jpeg")
OBSTACLE_IMG1 = os.path.join(script_dir, "obs1.jpeg")
OBSTACLE_IMG2 = os.path.join(script_dir, "obs2.jpeg")
OBSTACLE_IMG3 = os.path.join(script_dir, "obs3.jpeg")
OBSTACLE_IMG4 = os.path.join(script_dir, "obs4.jpeg")
OBSTACLE_IMG5 = os.path.join(script_dir, "obs5.jpeg")
BLACK_HOLE_IMG = os.path.join(script_dir, "bh.jpeg")

class DKGridWorldEnv(gym.Env):
    """
    Custom environment for grid-world based tasks implemented with Pygame and Gym libraries.
    This environment supports the basic operations typical in reinforcement learning experiments,
    such as step, reset, and render operations.

    Attributes:
        grid_size (int): The size of the grid (number of cells in one row or column).
        render_flag (bool): Indicates whether the environment should be rendered with each step.
        action_space (gym.spaces): Defines the range of allowable actions.
        observation_space (gym.spaces): Defines the shape and bounds of the state space.
        state (np.array): Current agent position in the grid.
        reward (int): Current reward after the last action taken.
        goal (np.array): The position of the goal the agent aims to reach.
        done (bool): Indicates whether the episode has ended.
        hell_states (list of np.array): List of 'hell' positions which provide negative rewards.
        black_hole_state (np.array): Position of a black hole that terminates the episode.
    """

    def __init__(self, grid_size=6, render=False):
        """
        Initializes a new grid world environment.

        Args:
            grid_size (int): Size of the grid, default is 6.
            render (bool): Flag to render the game's GUI, default is False.
        """
        pygame.init()
        super(DKGridWorldEnv, self).__init__()
        self.render_flag = render
        self.grid_size = grid_size
        self.cell_size = 100
        self.state = np.array([0, 0])
        self.reward = 0
        self.goal = np.array([5, 5])
        self.done = False
        self.hell_states = []
        self.black_hole_state = None

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32)

        # Setup Pygame display
        self.screen = pygame.display.set_mode((self.cell_size * grid_size, self.cell_size * grid_size))

        self.agent_image = pygame.image.load(AGENT_IMG_PATH)
        self.agent_original = pygame.transform.scale(self.agent_image, (self.cell_size, self.cell_size))
        self.goal_image = pygame.transform.scale(pygame.image.load(GOAL_IMG_PATH), (self.cell_size, self.cell_size))
        self.background_image = pygame.transform.scale(pygame.image.load(BACKGROUND_IMG), (self.cell_size * grid_size, self.cell_size * grid_size))
        self.obstacle_images = [
            pygame.transform.scale(pygame.image.load(OBSTACLE_IMG1), (self.cell_size, self.cell_size)),
            pygame.transform.scale(pygame.image.load(OBSTACLE_IMG2), (self.cell_size, self.cell_size)),
            pygame.transform.scale(pygame.image.load(OBSTACLE_IMG3), (self.cell_size, self.cell_size)),
            pygame.transform.scale(pygame.image.load(OBSTACLE_IMG4), (self.cell_size, self.cell_size)),
            pygame.transform.scale(pygame.image.load(OBSTACLE_IMG5), (self.cell_size, self.cell_size))
        ]
        self.blackHole_image = pygame.transform.scale(pygame.image.load(BLACK_HOLE_IMG), (self.cell_size, self.cell_size))

    def reset(self):
        """
        Resets the environment to its initial state and returns the initial state.

        Returns:
            tuple: A tuple containing the initial state and an empty info dictionary.
        """
        self.state = np.array([0, 0])
        self.reward = 0
        self.done = False
        return self.state, {}

    def add_hell_states(self, hell_state_coordinates):
        """
        Adds positions that will be considered as 'hell' states which give negative rewards when visited.

        Args:
            hell_state_coordinates (list of tuples): Coordinates of hell states to add.
        """
        self.hell_states.extend([np.array(hell) for hell in hell_state_coordinates])

    def set_black_hole_state(self, black_hole_coordinate):
        """
        Sets a black hole state which ends the episode when visited.

        Args:
            black_hole_coordinate (tuple): The coordinates of the black hole state.
        """
        self.black_hole_state = np.array(black_hole_coordinate)

    def step(self, action):
        """
        Updates the environment based on the action taken. Adjusts the agent's position, calculates reward, and updates the state of the environment.

        Args:
            action (int): An integer representing the action taken by the agent.

        Returns:
            tuple: A tuple containing the new state, the reward obtained, whether the episode is done, and an empty info dictionary.
        """
        old_state = self.state.copy()
        if action == 0 and self.state[0] > 0:  # Up
            self.state[0] -= 1
            self.agent_image = pygame.transform.rotate(self.agent_original, 90)
        elif action == 1 and self.state[0] < self.grid_size - 1:  # Down
            self.state[0] += 1
            self.agent_image = pygame.transform.flip(self.agent_original, True, False)
            self.agent_image = pygame.transform.rotate(self.agent_original, -90)
        elif action == 2 and self.state[1] < self.grid_size - 1:  # Right
            self.state[1] += 1
            self.agent_image = self.agent_original
        elif action == 3 and self.state[1] > 0:  # Left
            self.state[1] -= 1
            self.agent_image = pygame.transform.flip(self.agent_original, True, False)

        if np.array_equal(self.state, [0, 0]):
            self.agent_image = self.agent_original

        """
        Agent Effects and Rewards
            - Goal State: When the agent reaches the goal state (`self.goal`), the reward is increased by 100, and the episode is marked as done (`self.done = True`).
            - Hell State: When the agent hits a hell state (an obstacle `self.hell_states`), the reward is decreased by 10.
            - Black Hole State: When the agent reaches the black hole state (`self.black_hole_state`), the reward is decreased by 50, and the episode is marked as done (`self.done = True`).
        """

        if np.array_equal(self.state, self.goal):
            self.reward = 100
            self.done = True
        elif any(np.array_equal(self.state, hell) for hell in self.hell_states):
            self.reward = -10
            self.state = old_state  # Remain in the same position
        elif np.array_equal(self.state, self.black_hole_state):
            self.reward = -50
            self.done = True
        else:
            self.reward = -1  # Standard step penalty

        return self.state, self.reward, self.done, {}
    
    def render(self):
        """
        Renders the current state of the environment to the screen. This method updates the display and draws the agent,
        goal, black hole, and hell states on the grid.
        """
        if not self.render_flag:
            return
        # Process Pygame events to keep the window responsive
        for event in pygame.event.get():
            if event.type is pygame.QUIT:
                pygame.quit()
                raise SystemExit  # Optionally raise an error or quit the application
            
        temp_background = self.background_image.copy()
        temp_background.set_alpha(128)
        self.screen.blit(temp_background, (0, 0))

        # Draw elements
        self.screen.blit(self.goal_image, (self.goal[1] * self.cell_size, self.goal[0] * self.cell_size))
        if self.black_hole_state is not None:
            self.screen.blit(self.blackHole_image, (self.black_hole_state[1] * self.cell_size, self.black_hole_state[0] * self.cell_size))

        self.screen.blit(self.agent_image, (self.state[1] * self.cell_size, self.state[0] * self.cell_size))
        for index, hell in enumerate(self.hell_states):
            obstacle_image = self.obstacle_images[index % len(self.obstacle_images)]
            self.screen.blit(obstacle_image, (hell[1] * self.cell_size, hell[0] * self.cell_size))

        pygame.display.flip()  # Update the display to show the new positions
        time.sleep(0.1)

def create_env(goal_coordinates, hell_state_coordinates, black_hole_coordinate, render=True):
    """
    Factory function to create a new DKGridWorld environment with specified configurations.

    Args:
        goal_coordinates (tuple): Coordinates of the goal state.
        hell_state_coordinates (list of tuples): List of hell state coordinates.
        black_hole_coordinate (tuple): Coordinates of the black hole state.
        render (bool): If true, the environment will be rendered.

    Returns:
        DKGridWorldEnv: The newly created environment instance.
    """
    env = DKGridWorldEnv(grid_size=6, render=render)
    env.goal = np.array(goal_coordinates)
    env.add_hell_states(hell_state_coordinates)
    env.set_black_hole_state(black_hole_coordinate)
    return env