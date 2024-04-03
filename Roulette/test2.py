from RouletteEnv import *
import matplotlib.pyplot as plt

# Create the environment
env = RouletteEnv()

# Set the parameters for the experiment
num_episodes = 100000
base_learning_rate = 0.3
learning_rate_exponent = 0.5
epsilon_start = 1.0
epsilon_end = 0.1
decay_rate = 0.0001
gamma = 0.99
delta_start = 1.0

# Run the experiments
_, Q_learning_abs_errors, Q_learning_avg_action_values = Q_learning(env, num_episodes, base_learning_rate, learning_rate_exponent, epsilon_start, epsilon_end, decay_rate, gamma)
print("Q Learning Completed!")
# env = RouletteEnv()
# _, _, double_Q_learning_abs_errors, double_Q_learning_avg_action_values = double_Q_learning(env, num_episodes, base_learning_rate, learning_rate_exponent, epsilon_start, epsilon_end, decay_rate, gamma)
# print("Double Q Learning Completed!")
# env = RouletteEnv()
# _, softmax_smoothed_abs_errors, softmax_smoothed_avg_action_values = smoothed_Q_learning(env, num_episodes, base_learning_rate, learning_rate_exponent, epsilon_start, epsilon_end, decay_rate, gamma, "softmax")
# print("Softmax Smooth Q Learning Completed!")
# env = RouletteEnv()
# _, clipped_max_smoothed_abs_errors, clipped_max_smoothed_avg_action_values = smoothed_Q_learning(env, num_episodes, base_learning_rate, learning_rate_exponent, epsilon_start, epsilon_end, decay_rate, gamma, "clipped_max")
# print("Clipped max Smooth Q Learning Completed!")
# env = RouletteEnv()
# _, clipped_softmax_smoothed_abs_errors, clipped_softmax_smoothed_avg_action_values = smoothed_Q_learning(env, num_episodes, base_learning_rate, learning_rate_exponent, epsilon_start, epsilon_end, decay_rate, gamma, "clipped_softmax")
# print("Clipped Softmax Smooth Q Learning Completed!")
# Create subplots
fig, axs = plt.subplots(2, figsize=(12, 16))

# Plot the results for expected profit
axs[0].plot(range(num_episodes), Q_learning_avg_action_values, label='Q-learning', color='blue')
# axs[0].plot(range(num_episodes), double_Q_learning_avg_action_values, label='Double Q-learning', color='orange')
# axs[0].plot(range(num_episodes), softmax_smoothed_avg_action_values, label='Softmax Smoothed Q-learning', color='green')
# axs[0].plot(range(num_episodes), clipped_max_smoothed_avg_action_values, label='Clipped max Smoothed Q-learning', color='red')
# axs[0].plot(range(num_episodes), clipped_softmax_smoothed_avg_action_values, label='Clipped Softmax Smoothed Q-learning', color='purple')
# axs[0].plot(range(num_episodes), gmm_smoothed_avg_action_values, label='Gaussian Mixture Model Smoothed Q-learning', color='cyan')
axs[0].set_xlabel('Number of Trials')
axs[0].set_ylabel('Expected Profit')
axs[0].legend()

# Plot the results for abs_errors
axs[1].plot(range(num_episodes), Q_learning_abs_errors, label='Q-learning', color='blue')
# axs[1].plot(range(num_episodes), double_Q_learning_abs_errors, label='Double Q-learning', color='orange')
# axs[1].plot(range(num_episodes), softmax_smoothed_abs_errors, label='Softmax Smoothed Q-learning', color='green')
# axs[1].plot(range(num_episodes), clipped_max_smoothed_abs_errors, label='Clipped max Smoothed Q-learning', color='red')
# axs[1].plot(range(num_episodes), clipped_softmax_smoothed_abs_errors, label='Clipped Softmax Smoothed Q-learning', color='purple')
# axs[1].plot(range(num_episodes), gmm_smoothed_abs_errors, label='Gaussian Mixture Model Smoothed Q-learning', color='cyan')
axs[1].set_xlabel('Number of Episodes')
axs[1].set_ylabel('Abs Errors')
axs[1].legend()

# Show the plots
plt.show()
