from padm_env import create_env
from q_learning import train_q_learning, visualize_q_table, plot_rewards, plot_epsilon

# Parameters for the training and environment setup
train = True
render_during_training = False  # Set to True if you want to see the environment during training
visualize_results = True

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.95  # Decay rate for exploration
no_episodes = 1000 # Number of episodes

# Define the environment layout
goal_coordinates = (5, 5)
hell_state_coordinates = [(1, 1), (2, 3), (3, 1), (4, 4), (0, 5)]
black_hole_state = (5, 2)

if train:
    # Create and configure the environment
    env = create_env(goal_coordinates=goal_coordinates, hell_state_coordinates=hell_state_coordinates, black_hole_coordinate=black_hole_state, render=render_during_training)
    # Train the Q-learning agent
    rewards, epsilon_values = train_q_learning(env=env, no_episodes=no_episodes, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, alpha=learning_rate, gamma=gamma)
    # Visualize training results
    plot_rewards(rewards)
    plot_epsilon(epsilon_values)
    env.close()

if visualize_results:
    # Visualize the learned Q-table
    visualize_q_table(hell_state_coordinates=hell_state_coordinates, goal_coordinates=goal_coordinates, black_hole_state=black_hole_state, q_values_path="q_table.npy")