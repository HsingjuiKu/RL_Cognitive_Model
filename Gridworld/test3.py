import matplotlib.pyplot as plt
import numpy as np

from gridworld import GridWorld
from Q_learning import QLearning
from DoubleQ_learning import DoubleQLearning
from SmoothQ_learning import SmoothQLearning

# Define a class for comparing different Q-Learning algorithms
class QLearningComparison:
    def __init__(self, env, algorithms, episodes=50000):
        self.env = env  # environment
        self.algorithms = algorithms  # the algorithms to be compared
        self.episodes = episodes  # number of episodes for each algorithm
        self.Q_star = np.random.uniform(size=(env.grid_size, env.grid_size, 4))  # Assume this is the true Q*
        self.state_A = (0, 0)  # Define state A
        self.action_Left = 0  # Define the Left action
        self.q_diff_history = {}  # A dictionary to store the average difference between Q and Q*
        self.max_action_value_history = {}  # A dictionary to store the maximum action value in the starting state
        self.action_left_fraction_history = {}  # A dictionary to store the fraction of times the action taken from State A is left

    # Define a function to run the algorithms and store results
    def run(self):
        rewards = {}  # Initialize a dictionary to store the total reward of each algorithm

        for alg_name, alg in self.algorithms.items():
            print(f"Running {alg_name}...")
            rewards[alg_name] = alg.learn(self.episodes)

            # Different calculations for Double Q-learning and other algorithms
            if isinstance(alg, DoubleQLearning):  # Check if this is a Double Q-learning algorithm
                # Store the history of variables for Double Q-learning
                self.q_diff_history[alg_name] = [(np.abs(QA - QB)).mean() for QA, QB in zip(alg.QA_history, alg.QB_history)]
                self.max_action_value_history[alg_name] = [max(np.max(QA[self.state_A]), np.max(QB[self.state_A])) for QA, QB in zip(alg.QA_history, alg.QB_history)]
                self.action_left_fraction_history[alg_name] = [np.mean(np.array(alg.action_history.get(self.state_A, [])) == self.action_Left) for _ in alg.QA_history]

            else:
                # Store the history of variables for other algorithms
                self.q_diff_history[alg_name] = [np.abs(Q - self.Q_star).mean() for Q in alg.Q_history]
                self.max_action_value_history[alg_name] = [np.max(Q[self.state_A]) for Q in alg.Q_history]
                self.action_left_fraction_history[alg_name] = [np.mean(np.array(alg.action_history.get(self.state_A, [])) == self.action_Left) for _ in alg.Q_history]

        return rewards

    # Define a function to plot the results
    def plot_results(self, rewards):
        # Plot the average reward per step
        plt.figure(figsize=(10, 5))
        for alg_name in self.algorithms.keys():
            plt.plot(np.cumsum(rewards[alg_name]) / np.arange(1, self.episodes + 1), label=alg_name)
        plt.title("Average Reward per Step")
        plt.legend()
        plt.show()

        # Plot the average difference between Q and Q*
        plt.figure(figsize=(10, 5))
        for alg_name in self.algorithms.keys():
            plt.plot(self.q_diff_history[alg_name], label=alg_name)
        plt.title("Average Difference between Q and Q*")
        plt.legend()
        plt.show()

        # Plot the maximum action value in the starting state
        plt.figure(figsize=(10, 5))
        for alg_name in self.algorithms.keys():
            plt.plot(self.max_action_value_history[alg_name], label=alg_name)
        plt.title("Max Action Value in Starting State")
        plt.legend()
        plt.show()

        # Plot the fraction of times the action taken from State A is left
        plt.figure(figsize=(10, 5))
        for alg_name in self.algorithms.keys():
            plt.plot(self.action_left_fraction_history[alg_name], label=alg_name)
        plt.title("Fraction of Times the Action Taken from State A is Left")
        plt.legend()
        plt.show()


# Initialize the environment
grid_world = GridWorld()

# Initialize the algorithms
algorithms = {
    'QLearning': QLearning(grid_world),
    'DoubleQLearning': DoubleQLearning(grid_world),
    'SmoothQLearning_Softmax': SmoothQLearning(grid_world, smooth_method='softmax'),
    'SmoothQLearning_ClippedMax':  SmoothQLearning(grid_world, smooth_method='clipped_max')
}

# Initialize the comparison
comparison = QLearningComparison(grid_world, algorithms)

# Run the comparison
rewards = comparison.run()

# Plot the results
comparison.plot_results(rewards)

print("End...")