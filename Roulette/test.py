import matplotlib.pyplot as plt
import numpy as np
from RouletteEnv import *
# Use the implementations provided above here
# from your_implementation_file import Q_learning, double_Q_learning, smoothed_Q_learning, softmax, clipped_max

def run_experiments(env, num_experiments, num_episodes, base_learning_rate, alpha, gamma, epsilon):
    # Initialize averages and other tracking variables
    Q_avg_total = np.zeros(env.action_space.n)
    double_Q_avg_total = np.zeros(env.action_space.n)
    softmax_Q_avg_total = np.zeros(env.action_space.n)
    clipped_max_Q_avg_total = np.zeros(env.action_space.n)

    # Initialize tracking variables for new plots
    Q_left_actions = []
    double_Q_left_actions = []
    softmax_Q_left_actions = []
    clipped_max_Q_left_actions = []
    Q_diffs = []
    double_Q_diffs = []
    softmax_Q_diffs = []
    clipped_max_Q_diffs = []

    for _ in range(num_experiments):
        # Run all four methods
        Q_table, Q_left_action, Q_diff = Q_learning(env, num_episodes, base_learning_rate, alpha, gamma, epsilon)
        Q1_table, Q2_table, double_Q_left_action, double_Q_diff = double_Q_learning(env, num_episodes, base_learning_rate, alpha, gamma, epsilon)
        softmax_Q_table, softmax_Q_left_action, softmax_Q_diff = smoothed_Q_learning(env, num_episodes,
                                                                                     base_learning_rate, gamma,
                                                                                     "softmax", epsilon)
        clipped_max_Q_table, clipped_max_Q_left_action, clipped_max_Q_diff = smoothed_Q_learning(env, num_episodes,
                                                                                                 base_learning_rate,
                                                                                                 gamma, "clipped_max",
                                                                                                 epsilon)

        # Average action values for each action
        Q_avg = np.mean(Q_table, axis=0)
        double_Q_avg = np.mean((Q1_table + Q2_table) / 2, axis=0)
        softmax_Q_avg = np.mean(softmax_Q_table, axis=0)
        clipped_max_Q_avg = np.mean(clipped_max_Q_table, axis=0)

        # Add to totals
        Q_avg_total += Q_avg
        double_Q_avg_total += double_Q_avg
        softmax_Q_avg_total += softmax_Q_avg
        clipped_max_Q_avg_total += clipped_max_Q_avg

        # Add to tracking variables
        Q_left_actions.append(Q_left_action)
        double_Q_left_actions.append(double_Q_left_action)
        softmax_Q_left_actions.append(softmax_Q_left_action)
        clipped_max_Q_left_actions.append(clipped_max_Q_left_action)
        Q_diffs.append(Q_diff)
        double_Q_diffs.append(double_Q_diff)
        softmax_Q_diffs.append(softmax_Q_diff)
        clipped_max_Q_diffs.append(clipped_max_Q_diff)

    # Divide by number of experiments to get averages
    Q_avg_total /= num_experiments
    double_Q_avg_total /= num_experiments
    softmax_Q_avg_total /= num_experiments
    clipped_max_Q_avg_total /= num_experiments

    return Q_avg_total, double_Q_avg_total, softmax_Q_avg_total, clipped_max_Q_avg_total, \
           Q_left_actions, double_Q_left_actions, softmax_Q_left_actions, clipped_max_Q_left_actions, \
           Q_diffs, double_Q_diffs, softmax_Q_diffs, clipped_max_Q_diffs

def plot_results(Q_avg, double_Q_avg, softmax_Q_avg, clipped_max_Q_avg, Q_left_actions, double_Q_left_actions, softmax_Q_left_actions, clipped_max_Q_left_actions, Q_diffs, double_Q_diffs, softmax_Q_diffs, clipped_max_Q_diffs):
    plt.figure(figsize=(10, 6))
    plt.plot(Q_avg, label='Q-learning')
    plt.plot(double_Q_avg, label='Double Q-learning')
    plt.plot(softmax_Q_avg, label='Smoothed Q-learning (Softmax)')
    plt.plot(clipped_max_Q_avg, label='Smoothed Q-learning (Clipped Max)')
    plt.xlabel('Action')
    plt.ylabel('Average action value')
    plt.title('The average action values for different Q-learning methods when playing roulette')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(Q_left_actions, label='Q-learning')
    plt.plot(double_Q_left_actions, label='Double Q-learning')
    plt.plot(softmax_Q_left_actions, label='Smoothed Q-learning (Softmax)')
    plt.plot(clipped_max_Q_left_actions, label='Smoothed Q-learning (Clipped Max)')
    plt.xlabel('Episode')
    plt.ylabel('Frequency of Left Action from State A')
    plt.title('Maximisation Bias Example')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(Q_diffs, label='Q-learning')
    plt.plot(double_Q_diffs, label='Double Q-learning')
    plt.plot(softmax_Q_diffs, label='Smoothed Q-learning (Softmax)')
    plt.plot(clipped_max_Q_diffs, label='Smoothed Q-learning (Clipped Max)')
    plt.xlabel('Episode')
    plt.ylabel('|Q - Q*|')
    plt.title('Convergence of Q Value Estimation')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    num_experiments = 10
    num_episodes = 100000
    base_learning_rate = 1.0
    learning_rate_exponent = 0.5
    epsilon = 0.1
    gamma = 0.9

    env = RouletteEnv()

    Q_avg, double_Q_avg, softmax_Q_avg, clipped_max_Q_avg, \
    Q_left_actions, double_Q_left_actions, softmax_Q_left_actions, clipped_max_Q_left_actions, \
    Q_diffs, double_Q_diffs, softmax_Q_diffs, clipped_max_Q_diffs = run_experiments(env, num_experiments, num_episodes, base_learning_rate, learning_rate_exponent, epsilon, gamma)

    plot_results(Q_avg, double_Q_avg, softmax_Q_avg, clipped_max_Q_avg,
                 Q_left_actions, double_Q_left_actions, softmax_Q_left_actions, clipped_max_Q_left_actions,
                 Q_diffs, double_Q_diffs, softmax_Q_diffs, clipped_max_Q_diffs)

