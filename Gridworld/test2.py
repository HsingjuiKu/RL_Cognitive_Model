import matplotlib.pyplot as plt
import numpy as np

from gridworld import GridWorld
from Q_learning import QLearning
from DoubleQ_learning import DoubleQLearning
from SmoothQ_learning import SmoothQLearning


class QLearningComparison:
    def __init__(self, env, algorithms, episodes=500):
        self.env = env
        self.algorithms = algorithms
        self.episodes = episodes
        self.Q_star = np.random.uniform(size=(env.grid_size, env.grid_size, 4))  # Assume this is the true Q*
        self.state_A = (0, 0)  # Define state A
        self.action_Left = 0  # Define the Left action
        self.q_diff_history = {}
        self.max_action_value_history = {}
        self.action_left_fraction_history = {}

    def run(self):
        rewards = {}
        # q_differences = {}
        # max_q_values = {}
        # left_fractions = {}
        learning_speeds_1 = {}
        learning_speeds_2 = {}

        for alg_name, alg in self.algorithms.items():
            print(f"Running {alg_name}...")
            rewards[alg_name] = alg.learn(self.episodes)
            # left_fractions[alg_name] = self.calculate_left_fraction(alg)
            learning_speeds_1[alg_name] = self.calculate_learning_speed_1(rewards[alg_name])

            if isinstance(alg, DoubleQLearning):  # Check if this is a Double Q-learning algorithm
                # q_differences[alg_name] = self.calculate_q_difference(alg.QA, alg.QB)
                # max_q_values[alg_name] = max(np.max(alg.QA[self.state_A]), np.max(alg.QB[self.state_A]))
                learning_speeds_2[alg_name] = self.calculate_learning_speed_2(alg.QA_history, alg.QB_history)

                # Store the history of variables for Double Q-learning
                self.q_diff_history[alg_name] = [(np.abs(QA - QB)).mean() for QA, QB in
                                                 zip(alg.QA_history, alg.QB_history)]
                self.max_action_value_history[alg_name] = [max(np.max(QA[self.state_A]), np.max(QB[self.state_A])) for
                                                           QA, QB in zip(alg.QA_history, alg.QB_history)]
                self.action_left_fraction_history[alg_name] = [
                    np.mean(np.array(alg.action_history.get(self.state_A, [])) == self.action_Left) for _ in
                    alg.QA_history]

            else:
                # q_differences[alg_name] = self.calculate_q_difference(alg.Q)
                # max_q_values[alg_name] = np.max(alg.Q[self.state_A])
                learning_speeds_2[alg_name] = self.calculate_learning_speed_2(alg.Q_history)

                # Store the history of variables for other algorithms
                self.q_diff_history[alg_name] = [np.abs(Q - self.Q_star).mean() for Q in alg.Q_history]
                self.max_action_value_history[alg_name] = [np.max(Q[self.state_A]) for Q in alg.Q_history]
                self.action_left_fraction_history[alg_name] = [
                    np.mean(np.array(alg.action_history.get(self.state_A, [])) == self.action_Left) for _ in
                    alg.Q_history]

        return rewards, learning_speeds_1, learning_speeds_2

    def calculate_learning_speed_1(self, rewards):
        max_rewards_so_far = np.maximum.accumulate(rewards)
        target_rewards = max_rewards_so_far * 0.4  # you can adjust this value
        target_reached = np.cumsum(rewards) > target_rewards
        if np.any(target_reached):
            steps_to_target = np.argmax(target_reached)
        else:
            steps_to_target = self.episodes  # use the total number of episodes as default speed
        return steps_to_target

    # def calculate_learning_speed_1(self, rewards):
    #     target_reward = np.max(rewards) * 0.9  # Let's say 90% of the maximum reward obtained
    #     steps_to_target = np.argmax(np.cumsum(rewards) > target_reward)
    #     return steps_to_target
    #
    # def calculate_learning_speed_1(self, rewards):
    #     speeds = []
    #     for episode_rewards in rewards:
    #         target_reward = np.max(episode_rewards) * 0.95
    #         steps_to_target = np.argmax(np.cumsum(episode_rewards) > target_reward)
    #         speeds.append(steps_to_target)
    #     return speeds

    # def calculate_learning_speed_2(self, Q_history1, Q_history2=None):
    #     if Q_history2 is None:
    #         Q_diff = np.diff(Q_history1, axis=0)  # Calculate the difference between consecutive Q values
    #         return np.abs(Q_diff).mean()  # Calculate the mean of the absolute differences
    #     else:
    #         Q_diff1 = np.diff(Q_history1, axis=0)
    #         Q_diff2 = np.diff(Q_history2, axis=0)
    #         return (np.abs(Q_diff1).mean() + np.abs(Q_diff2).mean()) / 2  # Calculate the mean for Double Q-learning

    def calculate_learning_speed_2(self, Q_history1, Q_history2=None):
        Q_diff1 = np.diff(Q_history1, axis=0)
        Q_diff1_mean = np.abs(Q_diff1).mean()

        if Q_history2 is not None:
            Q_diff2 = np.diff(Q_history2, axis=0)
            Q_diff2_mean = np.abs(Q_diff2).mean()
            return (Q_diff1_mean + Q_diff2_mean) / 2  # Calculate the mean for Double Q-learning
        else:
            return Q_diff1_mean  # If there's only one Q history, return its mean difference

    def calculate_left_fraction(self, alg):
        if self.state_A in alg.action_history:
            action_history_A = alg.action_history[self.state_A]
            left_count = action_history_A.count(self.action_Left)
            total_count = len(action_history_A)
            return left_count / total_count
        else:
            return 0

    def plot_results(self, rewards, learning_speeds_1, learning_speeds_2):
        fig, axs = plt.subplots(6, 1, figsize=(10, 30))
        for alg_name in self.algorithms.keys():
            axs[0].plot(np.cumsum(rewards[alg_name]) / np.arange(1, self.episodes + 1), label=alg_name)

            # Plot the history of variables instead of constant values
            axs[1].plot(self.q_diff_history[alg_name], label=alg_name)
            axs[2].plot(self.max_action_value_history[alg_name], label=alg_name)
            axs[3].plot(self.action_left_fraction_history[alg_name], label=alg_name)
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
        axs[4].set_title("Learning Speed based on Reward")
        axs[4].legend()
        axs[5].set_title("Learning Speed based on Q Change")
        axs[5].legend()

        plt.show()


grid_world = GridWorld()

algorithms = {
    'QLearning': QLearning(grid_world),
    'DoubleQLearning': DoubleQLearning(grid_world),
    'SmoothQLearning_Softmax': SmoothQLearning(grid_world, smooth_method='softmax'),
    'SmoothQLearning_ClippedMax':  SmoothQLearning(grid_world, smooth_method='clipped_max')
}
comparison = QLearningComparison(grid_world, algorithms)
rewards, learning_speeds_1, learning_speeds_2 = comparison.run()
comparison.plot_results(rewards, learning_speeds_1, learning_speeds_2)
print("End...")