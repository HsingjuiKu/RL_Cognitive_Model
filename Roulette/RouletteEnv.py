# Import necessary libraries
import numpy as np
import gym
from gym import spaces
import math

# Define a Roulette environment class, inherited from gym.Env
class RouletteEnv(gym.Env):
    def __init__(self):
        super(RouletteEnv, self).__init__()  # Inherited gym.Env
        self.action_space = spaces.Discrete(171)  # Define a discrete action space with 171 possible actions
        self.observation_space = spaces.Discrete(1)  # Define an observation space with 1 possible state
        self.current_step = 0  # Initialize a counter for the current step

    def step(self, action):
        if action == 170:  # If the action is to stop playing
            return 0, 0, True, {}

        winning_number = np.random.choice(37)  # Choose a random winning number from 0 to 36
        reward = -1  # Assume player bets $1 each time and initialize the reward as -1

        # If player bets on a number and wins, they get a reward of 35
        if action < 36 and action == winning_number:
            reward = 35

        # If player bets on red or black and wins, they get a reward of 1
        elif action == 36 or action == 37:
            if (action == 36 and winning_number % 2 == 1) or (action == 37 and winning_number % 2 == 0):
                reward = 1

        # Bet on even or odd
        elif action == 38 or action == 39:
            if (action == 38 and winning_number % 2 == 0) or (action == 39 and winning_number % 2 == 1):
                reward = 1

        # Bet on low (1-18) or high (19-36)
        elif action == 40 or action == 41:
            if (action == 40 and 1 <= winning_number <= 18) or (action == 41 and 19 <= winning_number <= 36):
                reward = 1

        # Bet on first 12, second 12, or third 12
        elif action >= 42 and action <= 44:
            if (action == 42 and 1 <= winning_number <= 12) or \
                    (action == 43 and 13 <= winning_number <= 24) or \
                    (action == 44 and 25 <= winning_number <= 36):
                reward = 2

        # Bet on column
        elif action >= 45 and action <= 47:
            if (action == 45 and winning_number % 3 == 1) or \
                    (action == 46 and winning_number % 3 == 2) or \
                    (action == 47 and winning_number % 3 == 0):
                reward = 2

        # Bet on two adjacent numbers
        elif action >= 48 and action <= 82:
            if winning_number in [(action - 48), (action - 47)]:
                reward = 17

        # Bet on two numbers at the column intersection
        elif action >= 83 and action <= 94:
            columns = [(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36), (1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34),
                       (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)]
            column_pairs = [(columns[0], columns[1]), (columns[1], columns[2])]
            for pair in column_pairs:
                if winning_number in pair[0] and winning_number in pair[1]:
                    reward = 17

        return 0, reward, False, {}

    def reset(self):
        # Reset the environment to its initial state
        self.current_step = 0
        return 0



# Define a function for selecting an action using epsilon-greedy strategy
def epsilon_greedy(Q, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice(Q.shape[1])  # Randomly select an action
    else:
        action = np.argmax(Q[state])  # Select the action with the highest estimated reward
    return action

# Define a function for calculating a polynomial learning rate
def polynomial_learning_rate(base, exponent, t):
    t = max(t, 1)  # Ensure t is not zero
    return base / (t ** exponent)

def decay_epsilon(epsilon_start, epsilon_end, decay_rate, i_episode):
    return epsilon_end + (epsilon_start - epsilon_end) * math.exp(-decay_rate * i_episode)


# The Q_learning function implements the standard Q-learning algorithm.
# It takes as arguments the environment, the number of episodes, the base learning rate, learning rate exponent, epsilon, and the discount factor gamma.

def Q_learning(env, num_episodes, base_learning_rate, learning_rate_exponent, epsilon_start, epsilon_end, decay_rate, gamma):
    Q_table = np.zeros([env.observation_space.n, env.action_space.n])
    n_table = np.zeros_like(Q_table)
    MaxQ_table = np.full([env.observation_space.n, env.action_space.n], 0)
    abs_errors = []
    avg_action_values = []
    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        alpha = polynomial_learning_rate(base_learning_rate, learning_rate_exponent, i_episode)
        epsilon = decay_epsilon(epsilon_start, epsilon_end, decay_rate, i_episode)
        actions = []
        while not done:
            action = epsilon_greedy(Q_table, state, epsilon)
            n_table[state, action] += 1
            # alpha = 1 / n_table[state, action]
            actions.append(action)
            next_state, reward, done, info = env.step(action)
            old_value = Q_table[state][action]
            Q_table[state][action] = (1 - alpha) * old_value + alpha * (reward + gamma * np.max(Q_table[next_state]))
            state = next_state
        abs_errors.append(np.mean(np.abs(Q_table - MaxQ_table)))
        avg_action_values.append(np.mean(Q_table[state, :]))
    return Q_table, abs_errors, avg_action_values


def double_Q_learning(env, num_episodes, base_learning_rate, learning_rate_exponent, epsilon_start, epsilon_end, decay_rate, gamma):
    Q1_table = np.zeros([env.observation_space.n, env.action_space.n])
    Q2_table = np.zeros([env.observation_space.n, env.action_space.n])
    n1_table = np.zeros_like(Q1_table)
    n2_table = np.zeros_like(Q2_table)
    MaxQ_table = np.full([env.observation_space.n, env.action_space.n], 0)
    abs_errors = []
    avg_action_values = []
    for i_episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        alpha = polynomial_learning_rate(base_learning_rate, learning_rate_exponent, i_episode)
        epsilon = decay_epsilon(epsilon_start, epsilon_end, decay_rate, i_episode)
        actions = []
        while not done:
            action = epsilon_greedy(Q1_table + Q2_table, state, epsilon)
            actions.append(action)
            next_state, reward, done, info = env.step(action)
            if np.random.binomial(1, 0.5) == 1:
                n1_table[state, action] += 1
                # alpha = 1 / n1_table[state, action]
                best_next_action = np.argmax(Q1_table[next_state])
                Q1_table[state][action] = (1 - alpha) * Q1_table[state][action] + alpha * (reward + gamma * Q2_table[next_state][best_next_action])
            else:
                n2_table[state, action] += 1
                alpha = 1 / n2_table[state, action]
                best_next_action = np.argmax(Q2_table[next_state])
                Q2_table[state][action] = (1 - alpha) * Q2_table[state][action] + alpha * (reward + gamma * Q1_table[next_state][best_next_action])
            state = next_state
        abs_errors.append(np.mean(np.abs((Q1_table + Q2_table) / 2 - MaxQ_table)))
        avg_action_values.append(np.mean((Q1_table + Q2_table)[state, :] / 2))
    return Q1_table, Q2_table, abs_errors, avg_action_values

def smoothed_Q_learning(env, num_episodes, base_learning_rate, learning_rate_exponent, epsilon_start, epsilon_end, decay_rate, gamma, selection_method):
    Q_table = np.zeros([env.observation_space.n, env.action_space.n])
    n_table = np.zeros_like(Q_table)
    MaxQ_table = np.full([env.observation_space.n, env.action_space.n], 0)


    abs_errors = []
    avg_action_values = []

    def softmax(Q_state, beta):
        Q_normalized = (Q_state - np.mean(Q_state)) / (
                    np.std(Q_state) + 1e-10)  # Added epsilon to avoid division by zero
        exps = np.exp(beta * Q_normalized)
        probs = exps / np.sum(exps)
        probs = probs / np.sum(probs)  # Ensure probabilities sum to 1
        if np.isnan(probs).any():
            probs = np.ones_like(Q_state) / len(Q_state)
        return probs

    def clipped_softmax(Q_state, beta):
        Q_clipped = np.ones_like(Q_state) * (-np.inf)
        top_k_indices = np.argsort(Q_state)[-3:]
        Q_clipped[top_k_indices] = Q_state[top_k_indices] - np.max(Q_state)  # Subtract the maximum value
        exps = np.exp(beta * Q_clipped)
        return exps / np.sum(exps)

    def argmaxR(Q_state):
        max_value_indices = np.argwhere(Q_state == np.amax(Q_state)).flatten().tolist()
        return np.random.choice(max_value_indices)

    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        beta = 0.1 + 0.1 * (i_episode + 1)
        mu = np.exp(-0.02 * i_episode)
        actions = []
        epsilon = decay_epsilon(epsilon_start, epsilon_end, decay_rate, i_episode)
        alpha = polynomial_learning_rate(base_learning_rate, learning_rate_exponent, i_episode)

        while not done:
            if np.random.rand() > epsilon:
                action = argmaxR(Q_table[state])
            else:
                if selection_method == 'softmax':
                    action = np.random.choice(np.arange(env.action_space.n), p=softmax(Q_table[state], beta))
                elif selection_method == 'clipped_max':
                    A = len(Q_table[state])
                    action_distribution = np.full(A, mu / (A - 1))
                    action_distribution[argmaxR(Q_table[state])] = 1 - mu
                    action = np.random.choice(np.arange(env.action_space.n), p=action_distribution)
                elif selection_method == 'clipped_softmax':
                    action = np.random.choice(np.arange(env.action_space.n), p=clipped_softmax(Q_table[state], beta))



            actions.append(action)
            n_table[state, action] += 1
            # alpha = 1 / n_table[state, action]
            next_state, reward, done, _ = env.step(action)

            if selection_method == 'softmax':
                action_distribution = softmax(Q_table[next_state], beta)
            elif selection_method == 'clipped_max':
                A = len(Q_table[next_state])
                action_distribution = np.ones(A) * mu / (A - 1) if A != 1 else np.ones(A)
                as_ = argmaxR(Q_table[next_state])
                action_distribution[as_] = 1 - mu
            elif selection_method == 'clipped_softmax':
                action_distribution = clipped_softmax(Q_table[next_state], beta)

            expected_return = reward + gamma * np.sum(Q_table[next_state] * action_distribution)
            Q_table[state][action] = Q_table[state][action] + alpha * (expected_return - Q_table[state][action])
            state = next_state

        abs_errors.append(np.mean(np.abs(Q_table - MaxQ_table)))
        avg_action_values.append(np.mean(Q_table[state, :]))

    return Q_table, abs_errors, avg_action_values
