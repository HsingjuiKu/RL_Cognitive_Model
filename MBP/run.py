import numpy as np
import matplotlib.pyplot as plt
import math
import collections


"""
第一个图像 是 "平均绝对误差" 的图像。误差是指在每一次探索后，Q矩阵的预测值与其真实值之间的差距。
在每一个方法中，计算所有实验的平均绝对误差，并画出随着实验次数增加，这个误差如何变化。
计算方法是先计算每一次探索结束后Q矩阵在特定状态'a'和行动'l'下的值与真实值的差的绝对值，
然后计算所有这些值的平均值。

第二个图像 是 "平均行动" 的图像。这个图像描述的是在给定的实验和探索次数下，智能体选择行动'l'的平均次数。
对每个方法，都计算了所有实验中智能体选择行动'l'的平均次数，并画出这个数量随着实验次数增加的变化。
计算方法是先确定每一次实验中智能体选择行动'l'的次数，然后计算所有这些值的平均值。
"""
"""
两个图像的计算方法与之前的代码相比，都增加了一个维度：在每个episode结束时，
而不仅仅是在每次探索结束时，都计算了这两种数据。这样就可以观察到在每次运行和在所有运行中，
智能体的行动选择和学习进程如何随着时间（即episode）的推进而变化。

第一个图像是在所有运行和所有方法中，每个episode的平均Q值误差。
这个图像可以显示出每种方法在学习环境和优化其行动选择上的效率。

第二个图像是在所有运行和所有方法中，每个episode的平均"left"行动选择比例。
这个图像可以显示出每种方法在平衡探索和利用之间的策略。
"""

# Define the Environment class
class Environment:
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_state = 'A'
        return self.current_state

    def step(self, action):
        if self.current_state == 'A':
            if action == 'left':
                self.current_state = 'B'
            else:
                self.current_state = 'C'
            return self.current_state, 0, self.current_state == 'C'
        elif self.current_state == 'B':
            # From B there are 8 actions, each of which takes the model to a terminal state (here represented as state 'D')
            # with a reward sampled from a Gaussian distribution with mean -0.1 and variance 1
            reward = np.random.normal(-0.1, 1)
            self.current_state = 'D'
            return self.current_state, reward, True
        else:
            # This should not be reached during normal execution
            return None, None, None



# Define the QLearning class
class QLearning:
    def __init__(self, states, actions, alpha, gamma, epsilon):
        # Initialize Q-learning parameters
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Initialize Q-value table
        self.Q = {state: {action: 0 for action in actions} for state in states}
        # Initialize action count dictionary for data collection
        self.action_count = collections.defaultdict(int)
        # Initialize Q-value history list for data collection
        self.q_values_history = []

    def select_action(self, state):
        # Select an action based on current state and epsilon-greedy policy
        available_actions = self.actions if state != 'B' else ['action'+str(i) for i in range(8)]
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(available_actions)
        else:
            action = max({a: self.Q[state][a] for a in available_actions}, key=self.Q[state].get)
        # If state is 'A' and action is 'left', increment action count
        if state == 'A' and action == 'left':
            self.action_count[action] += 1
        return action

    def update(self, state, action, reward, next_state, done, t):
        # Update the Q-value for the current state-action pair based on the Q-learning update rule
        max_q_next = 0 if done else max(self.Q[next_state].values())
        self.Q[state][action] += self.alpha(t) * (reward + self.gamma * max_q_next - self.Q[state][action])
        # If episode is done, append the mean Q-value to history
        if done:
            self.q_values_history.append(np.mean([v for q in self.Q.values() for v in q.values()]))

class DoubleQLearning(QLearning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q1 = self.Q
        self.Q2 = {state: {action: 0 for action in self.actions} for state in self.states}
        self.error_history = []

    def select_action(self, state):
        available_actions = self.actions if state != 'B' else ['action' + str(i) for i in range(8)]
        sum_q = {action: self.Q1[state][action] + self.Q2[state][action] for action in available_actions}
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(available_actions)
        else:
            action = max(sum_q, key=sum_q.get)
        if state == 'A' and action == 'left':
            self.action_count[action] += 1
        return action

    def update(self, state, action, reward, next_state, done, t):
        q_old = self.Q1[state][action] if np.random.uniform() < 0.5 else self.Q2[state][action]
        if np.random.uniform() < 0.5:
            max_q_next = 0 if done else max(self.Q2[next_state].values())
            self.Q1[state][action] += self.alpha(t) * (reward + self.gamma * max_q_next - self.Q1[state][action])
        else:
            max_q_next = 0 if done else max(self.Q1[next_state].values())
            self.Q2[state][action] += self.alpha(t) * (reward + self.gamma * max_q_next - self.Q2[state][action])
        if state == 'A' and action == 'left':
            self.error_history.append(abs((self.Q1[state][action] + self.Q2[state][action]) / 2 - q_old))
        if done:
            self.q_values_history.append(np.mean([v for s in self.states for v in self.Q1[s].values()] +
                                                 [v for s in self.states for v in self.Q2[s].values()]))


class SmoothQLearning(QLearning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_count = collections.defaultdict(int)  # For data collection
        self.q_values_history = []  # For data collection
        self.error_history = []

    def select_action(self, state):
        available_actions = self.actions if state != 'B' else ['action'+str(i) for i in range(8)]
        q_values = {action: self.Q[state][action] for action in available_actions}
        max_q_value = max(q_values.values())
        epsilon_non_zero = self.epsilon if self.epsilon != 0 else 1e-10  # ensure epsilon is non-zero
        # Subtract the max Q value from all Q values to prevent overflow
        adjusted_q_values = [q_value - max_q_value for q_value in q_values.values()]
        probabilities = [math.exp(q_value / epsilon_non_zero) for q_value in adjusted_q_values]
        probabilities = [prob / sum(probabilities) for prob in probabilities]
        action = np.random.choice(available_actions, p=probabilities)
        if state == 'A' and action == 'left':
            self.action_count[action] += 1  # Data collection
        return action

    def update(self, state, action, reward, next_state, done, t):
        q_old = self.Q[state][action]
        max_q_next = 0 if done else max(self.Q[next_state].values())
        self.Q[state][action] += self.alpha(t) * (reward + self.gamma * max_q_next - self.Q[state][action])
        if state == 'A' and action == 'left':
            self.error_history.append(abs(self.Q[state][action] - q_old))
        if done:
            self.q_values_history.append(
                np.mean([v for q in self.Q.values() for v in q.values()]))  # Record only at the end of each episode


class SoftmaxQLearning(SmoothQLearning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_epsilon(self, t):
        self.epsilon = 0.1 + 0.1 * (t - 1)


class ClippedMaxQLearning(SmoothQLearning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_epsilon(self, t):
        self.epsilon = math.exp(-0.02 * t)

alpha = lambda t: 0.1 / (1 + 0.001 * t)
gamma = 0.99
epsilon = 0.1
timesteps = 100
episodes = 100
runs = 100

states = ['A', 'B', 'C', 'D']  # We don't need to include 'E' to 'L' in the states as they are not different states but different actions in state 'B'
actions = ['left', 'right'] + ['action'+str(i) for i in range(8)]  # We add 8 new actions corresponding to the 8 actions from state 'B'

# 初始化环境
env = Environment()

# 创建学习者实例
learners = [
    QLearning(states, actions, alpha, gamma, epsilon),
    DoubleQLearning(states, actions, alpha, gamma, epsilon),
    SoftmaxQLearning(states, actions, alpha, gamma, epsilon),
    ClippedMaxQLearning(states, actions, alpha, gamma, epsilon)
]
# 进行实验
results = {}
for learner in learners:
    action_history = []
    q_values_histories = []
    for run in range(runs):
        learner.action_count = collections.defaultdict(int) # 计数器重置
        action_history_per_run = []
        q_values_per_episode = []
        for episode in range(episodes):
            state = env.reset()
            actions_per_episode = []
            for t in range(timesteps):
                action = learner.select_action(state)
                next_state, reward, done = env.step(action)
                learner.update(state, action, reward, next_state, done, t)
                state = next_state
                # 每一步的行动记录下来
                actions_per_episode.append(1 if action == 'left' else 0)
                if done:
                    action_history_per_run.append(np.mean(actions_per_episode))
                    q_values_per_episode.append(np.abs(learner.q_values_history[-1] - np.mean(learner.q_values_history)))
                    break
            # 如果达不到done状态，同样记录这个episode的行动历史
            if not done:
                action_history_per_run.append(np.mean(actions_per_episode))
        action_history.append(action_history_per_run)
        q_values_histories.append(q_values_per_episode)

    results[type(learner).__name__] = (np.array(action_history), q_values_histories)

# Data processing and plotting
plt.figure(figsize=(10, 6))
for learner_name, data in results.items():
    action_history, q_values_histories = data
    mean_q_values_histories = np.mean(q_values_histories, axis=0)

    plt.plot(mean_q_values_histories, label=learner_name)
plt.xlabel('Number of Episodes')
plt.ylabel('Mean Absolute Q-value Errors')
plt.title('Mean Absolute Q-value Errors of Different Q-Learning Algorithms over Episodes')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for learner_name, data in results.items():
    action_history, _ = data
    mean_action_history = np.mean(action_history, axis=0)

    plt.plot(mean_action_history, label=learner_name)
plt.xlabel('Number of Episodes')
plt.ylabel('Proportion of Left Actions')
plt.title('Proportion of Left Actions of Different Q-Learning Algorithms over Episodes')
plt.legend()
plt.show()


