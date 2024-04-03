import matplotlib.pyplot as plt
import numpy as np

from gridworld import GridWorld
from Q_learning import QLearning
from DoubleQ_learning import DoubleQLearning
from SmoothQ_learning import SmoothQLearning

class QLearningComparison:
    def __init__(self, env, algorithms, episodes=50000):
        self.env = env
        self.algorithms = algorithms
        self.episodes = episodes
        self.Q_star = np.random.uniform(size=(env.grid_size, env.grid_size, 4))  # Assume this is the true Q*
        self.state_A = (0, 0)  # Define state A
        self.action_Left = 0  # Define the Left action

    def run(self):
        rewards = {}
        q_differences = {}
        max_q_values = {}
        left_fractions = {}
        learning_speeds_1 = {}
        learning_speeds_2 = {}
        for alg_name, alg in self.algorithms.items():
            print(f"Running {alg_name}...")
            rewards[alg_name] = alg.learn(self.episodes)
            left_fractions[alg_name] = self.calculate_left_fraction(alg)
            learning_speeds_1[alg_name] = self.calculate_learning_speed_1(rewards[alg_name])
            if hasattr(alg, 'QB'):  # Check if this is a Double Q-learning algorithm
                q_differences[alg_name] = self.calculate_q_difference(alg.QA, alg.QB)
                max_q_values[alg_name] = max(np.max(alg.QA[(0, 0)]), np.max(alg.QB[(0, 0)]))
                learning_speeds_2[alg_name] = self.calculate_learning_speed_2(alg.QA_history, alg.QB_history)
            else:
                q_differences[alg_name] = self.calculate_q_difference(alg.Q)
                max_q_values[alg_name] = np.max(alg.Q[(0, 0)])
                learning_speeds_2[alg_name] = self.calculate_learning_speed_2(alg.Q_history)

        return rewards, q_differences, max_q_values, left_fractions, learning_speeds_1, learning_speeds_2

    def calculate_learning_speed_1(self, rewards):
        target_reward = np.max(rewards) * 0.9  # Let's say 90% of the maximum reward obtained
        steps_to_target = np.argmax(np.cumsum(rewards) > target_reward)
        return steps_to_target

    def calculate_learning_speed_2(self, Q_history1, Q_history2=None):
        if Q_history2 is None:
            Q_diff = np.diff(Q_history1, axis=0)  # Calculate the difference between consecutive Q values
            return np.abs(Q_diff).mean()  # Calculate the mean of the absolute differences
        else:
            Q_diff1 = np.diff(Q_history1, axis=0)
            Q_diff2 = np.diff(Q_history2, axis=0)
            return (np.abs(Q_diff1).mean() + np.abs(Q_diff2).mean()) / 2  # Calculate the mean for Double Q-learning

    # def calculate_learning_speed_1(self, rewards):
    #     target_reward = np.max(rewards) * 0.9  # Let's say 90% of the maximum reward obtained
    #     steps_to_target = np.argmax(np.cumsum(rewards) > target_reward)
    #     return steps_to_target / self.episodes  # Normalize by the number of episodes
    #
    # def calculate_learning_speed_2(self, Q_history1, Q_history2=None):
    #     if Q_history2 is None:
    #         Q_diff = np.abs(Q_history1 - self.Q_star)  # Calculate the difference between the final Q values and Q*
    #         return Q_diff.mean()  # Calculate the mean of the absolute differences
    #     else:
    #         Q_diff1 = np.abs(Q_history1 - self.Q_star)
    #         Q_diff2 = np.abs(Q_history2 - self.Q_star)
    #         return (Q_diff1.mean() + Q_diff2.mean()) / 2  # Calculate the mean for Double Q-learning

    def calculate_q_difference(self, Q1, Q2=None):
        if Q2 is None:
            return np.abs(Q1 - self.Q_star).mean()
        else:
            return (np.abs(Q1 - self.Q_star).mean() + np.abs(Q2 - self.Q_star).mean()) / 2

    def plot_results(self, rewards, q_differences, max_q_values, left_fractions, learning_speeds_1, learning_speeds_2):
        fig, axs = plt.subplots(6, 1, figsize=(10, 30))
        for alg_name in self.algorithms.keys():
            axs[0].plot(np.cumsum(rewards[alg_name]) / np.arange(1, self.episodes+1), label=alg_name)
            axs[1].plot([q_differences[alg_name]] * self.episodes, label=alg_name)
            axs[2].plot([max_q_values[alg_name]] * self.episodes, label=alg_name)
            axs[3].plot([left_fractions[alg_name]] * self.episodes, label=alg_name)
            axs[4].plot([learning_speeds_1[alg_name]] * self.episodes, label=alg_name)
            axs[5].plot([learning_speeds_2[alg_name]] * self.episodes, label=alg_name)

        axs[0].set_title("Average Reward per Step")
        axs[0].legend()
        axs[1].set_title("Average Difference between Q and Q*")
        axs[1].legend()
        axs[2].set_title("Max Action Value in Starting State")
        axs[2].legend()
        axs[3].set_title("Fraction of Times the Action Taken from State A is Left")
        axs[3].legend()
        axs[4].set_title("Calculate_learning_speed with Q_history1 and Q_history2")
        axs[4].legend()
        axs[5].set_title("Calculate_learning_speed with rewards")
        axs[5].legend()

        plt.show()

    def calculate_left_fraction(self, alg):
        if self.state_A in alg.action_history:
            action_history_A = alg.action_history[self.state_A]
            left_count = action_history_A.count(self.action_Left)
            total_count = len(action_history_A)
            return left_count / total_count
        else:
            return 0

grid_world = GridWorld()

algorithms = {
    'QLearning': QLearning(grid_world),
    'DoubleQLearning': DoubleQLearning(grid_world),
    'SmoothQLearning_Softmax': SmoothQLearning(grid_world, smooth_method='softmax'),
    'SmoothQLearning_ClippedMax': SmoothQLearning(grid_world, smooth_method='clipped_max')
}

comparison = QLearningComparison(grid_world, algorithms)
rewards, q_differences, max_q_values, left_fractions, learning_speeds_1, learning_speeds_2 = comparison.run()
comparison.plot_results(rewards, q_differences, max_q_values, left_fractions, learning_speeds_1, learning_speeds_2)

