from random import random

"""
Q(S_t, A_t) ← Q(S_t, A_t) + α [R_t+1 + γ max_a Q(S_t+1, a) - Q(S_t, A_t)]

我们用以下符号进行解释：

S_t：当前状态
A_t：在当前状态下采取的动作
R_t+1：在执行动作A_t后获得的奖励
S_t+1：执行动作A_t后进入的新状态
α：学习率（0 ≤ α ≤ 1），用于控制我们应该在多大程度上考虑新信息，而忽略旧信息
γ：折扣因子（0 ≤ γ ≤ 1），用于确定我们应该多重视未来奖励
max_a Q(S_t+1, a)：预期的最大未来奖励，对于在新状态S_t+1中所有可能的动作a，我们看看哪一个能给我们最大的Q值
"""

class QLearning:
    def __init__(self, states, actions, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.states = states
        self.actions = actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.q_table = {s: {a: 0 for a in actions} for s in states}

    def update_q_value(self, current_state, action, reward, next_state):
        max_future_q = max(self.q_table[next_state].values())
        current_q = self.q_table[current_state][action]
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
        self.q_table[current_state][action] = new_q

    def choose_action(self, current_state):
        if random.uniform(0, 1) < self.epsilon:  # explore
            action = random.choice(self.actions)
        else:  # exploit
            action = max(self.q_table[current_state], key=self.q_table[current_state].get)
        return action


class DoubleQLearning(QLearning):
    def __init__(self, states, actions, alpha=0.5, gamma=0.9, epsilon=0.1):
        super().__init__(states, actions, alpha, gamma, epsilon)
        self.q_table2 = {s: {a: 0 for a in actions} for s in states}

    def update_q_value(self, current_state, action, reward, next_state):
        if random.uniform(0, 1) < 0.5:  # update q_table1
            max_future_q = max(self.q_table2[next_state].values())
            current_q = self.q_table[current_state][action]
            new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
            self.q_table[current_state][action] = new_q
        else:  # update q_table2
            max_future_q = max(self.q_table[next_state].values())
            current_q = self.q_table2[current_state][action]
            new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
            self.q_table2[current_state][action] = new_q



class SmoothedQLearning(QLearning):
    def __init__(self, states, actions, alpha=0.5, gamma=0.9, epsilon=0.1, beta=0.5):
        super().__init__(states, actions, alpha, gamma, epsilon)
        self.beta = beta  # smoothing parameter

    def update_q_value(self, current_state, action, reward, next_state):
        max_future_q = max(self.q_table[next_state].values())
        current_q = self.q_table[current_state][action]
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
        self.q_table[current_state][action] = (1 - self.beta) * self.q_table[current_state][action] + self.beta * new_q

