import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pygame

def train_q_learning(env, no_episodes, epsilon, epsilon_min, epsilon_decay, alpha, gamma, q_table_save_path="q_table.npy"):
    """
    Train a Q-learning model on the specified environment.

    Args:
        env: The environment to train the agent in.
        no_episodes (int): The number of episodes to train the agent for.
        epsilon (float): Initial exploration rate.
        epsilon_min (float): Minimum exploration rate.
        epsilon_decay (float): Decay rate for exploration rate.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        q_table_save_path (str): Path to save the Q-table.

    Returns:
        tuple: A tuple containing lists of rewards per episode and epsilon values per episode.
    """
    pygame.init()
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    rewards_per_episode = []
    epsilon_values = []
    best_total_reward = float('-inf')

    for episode in range(no_episodes):
        state, _ = env.reset()
        state = tuple(state)
        total_reward = 0
        episode_actions = []
        episode_states = []
        while True:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, _ = env.step(action)
            next_state = tuple(next_state)

            # Track state and action for replaying the best episode
            episode_states.append(state)
            episode_actions.append(action)
            total_reward += reward

            # UPDATE---Q-TABLE
            q_table[state][action] = q_table[state][action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
            
            state = next_state

            if env.render_flag:
                env.render()

            if done:
                if total_reward > best_total_reward:
                    best_total_reward = total_reward
                rewards_per_episode.append(total_reward)
                epsilon_values.append(epsilon)
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"epsilon ----------  {epsilon}")
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    print(f"Rendering the best episode with Total Reward: {best_total_reward}")
    env.reset()
    env.close()
    pygame.quit()
    np.save(q_table_save_path, q_table)
    print("Saved the Q-table and finished rendering the best episode.")
    return rewards_per_episode, epsilon_values

def plot_rewards(rewards):
    """
    Plot the rewards per episode as a line graph.

    Args:
        rewards (list): List of total rewards per episode.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Total Reward per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Training Progress: Total Rewards per Episode')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_epsilon(epsilon_values):
    """
    Plot the epsilon values per episode as a line graph.

    Args:
        epsilon_values (list): List of epsilon values per episode.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(epsilon_values, label='Epsilon Value per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay Over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_q_table(hell_state_coordinates, goal_coordinates, black_hole_state, actions=["Up", "Down", "Right", "Left"], q_values_path="q_table.npy"):
    """
    Visualize the Q-table values using heatmaps for each action.

    Args:
        hell_state_coordinates (list): List of coordinates for 'hell' states.
        goal_coordinates (tuple): Coordinates for the goal state.
        black_hole_state (tuple): Coordinates for the black hole state.
        actions (list): List of actions corresponding to Q-table dimensions.
        q_values_path (str): Path to the saved Q-table.
    """
    try:
        q_table = np.load(q_values_path)
        _, axes = plt.subplots(1, 4, figsize=(20, 5))
        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()
            mask = np.zeros_like(heatmap_data, dtype=bool)
            mask[goal_coordinates[0], goal_coordinates[1]] = True
            for hell_state in hell_state_coordinates:
                mask[hell_state[0], hell_state[1]] = True
            mask[black_hole_state[0], black_hole_state[1]] = True
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})
            ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green', ha='center', va='center', weight='bold', fontsize=10)
            for hell_state in hell_state_coordinates:
                ax.text(hell_state[1] + 0.5, hell_state[0] + 0.5, 'H', color='red', ha='center', va='center', weight='bold', fontsize=10)
            ax.text(black_hole_state[1] + 0.5, black_hole_state[0] + 0.5, 'B', color='black', ha='center', va='center', weight='bold', fontsize=10)
            ax.set_title(f'Action: {action}')
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")