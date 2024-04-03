from gridworld import GridWorld
from Q_learning import QLearning
from DoubleQ_learning import DoubleQLearning
from SmoothQ_learning import SmoothQLearning
import numpy as np
import matplotlib.pyplot as plt


def pad_rewards(rewards, max_length=None):
    if max_length is None:
        max_length = max(len(r) for r in rewards)
    padded_rewards = []
    for reward in rewards:
        padded_reward = reward + [0]*(max_length - len(reward))
        padded_rewards.append(padded_reward)
    return padded_rewards


def run_experiment(episodes=1000, experiments=100):
    Q_avg_rewards = []
    Q_max_action_vals = []
    DoubleQ_avg_rewards = []
    DoubleQ_max_action_vals = []
    SmoothQ_avg_rewards_softmax = []
    SmoothQ_max_action_vals_softmax = []
    SmoothQ_avg_rewards_clipped = []
    SmoothQ_max_action_vals_clipped = []

    for _ in range(experiments):
        env = GridWorld()

        QLearner = QLearning(env)
        DoubleQLearner = DoubleQLearning(env)
        # SmoothQLearner_softmax = SmoothQLearning(env, smooth_method="softmax")
        # SmoothQLearner_clipped = SmoothQLearning(env, smooth_method="clipped_max")

        Q_avg_rewards.append(QLearner.learn(episodes))
        Q_max_action_vals.append(QLearner.max_action_val_start)

        DoubleQ_avg_rewards.append(DoubleQLearner.learn(episodes))
        DoubleQ_max_action_vals.append(DoubleQLearner.max_action_val_start)

        # SmoothQ_avg_rewards_softmax.append(SmoothQLearner_softmax.learn(episodes))
        # SmoothQ_max_action_vals_softmax.append(SmoothQLearner_softmax.max_action_val_start)

        # SmoothQ_avg_rewards_clipped.append(SmoothQLearner_clipped.learn(episodes))
        # SmoothQ_max_action_vals_clipped.append(SmoothQLearner_clipped.max_action_val_start)

    # Find maximum reward list length
    max_reward_length = max(len(r) for r in Q_avg_rewards + DoubleQ_avg_rewards)

    # Padding rewards to the same length
    Q_avg_rewards = pad_rewards(Q_avg_rewards, max_reward_length)
    DoubleQ_avg_rewards = pad_rewards(DoubleQ_avg_rewards, max_reward_length)
    # SmoothQ_avg_rewards_softmax = np.array(pad_rewards(SmoothQ_avg_rewards_softmax))
    # SmoothQ_avg_rewards_clipped = np.array(pad_rewards(SmoothQ_avg_rewards_clipped))

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))



    # Plotting average rewards per time step on subplot 1
    axs[0].plot(np.mean(Q_avg_rewards, axis=0), label="Q-learning")
    axs[0].plot(np.mean(DoubleQ_avg_rewards, axis=0), label="Double Q-learning")
    # axs[0].plot(np.mean(SmoothQ_avg_rewards_softmax, axis=0), label="Smooth Q-learning Softmax")
    # axs[0].plot(np.mean(SmoothQ_avg_rewards_clipped, axis=0), label="Smooth Q-learning Clipped")
    axs[0].set_xlabel('Time Steps')
    axs[0].set_ylabel('Average Reward')
    axs[0].set_title('Average Rewards per Time Step')
    axs[0].legend()

    # Plotting maximal action value in the starting state on subplot 2
    axs[1].plot(np.mean(Q_max_action_vals, axis=0), label="Q-learning")
    axs[1].plot(np.mean(DoubleQ_max_action_vals, axis=0), label="Double Q-learning")
    # axs[1].plot(np.mean(SmoothQ_max_action_vals_softmax, axis=0), label="Smooth Q-learning Softmax")
    # axs[1].plot(np.mean(SmoothQ_max_action_vals_clipped, axis=0), label="Smooth Q-learning Clipped")
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Maximal Action Value in Starting State')
    axs[1].set_title('Maximal Action Value in the Starting State S')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

run_experiment()
