import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld

def Q_learning(env, num_episodes, max_steps):
    Q_table = np.zeros((env.grid_size, env.grid_size, 4))
    N_table = np.zeros((env.grid_size, env.grid_size, 4))  # Keep track of the counts
    rewards = np.zeros(num_episodes)
    max_action_values = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        # alpha = 1 / (episode + 1)  # Decaying learning rate
        while not done and steps < max_steps:
            # epsilon = env.get_epsilon(state)
            epsilon = 1 / np.sqrt(episode + 1)  # Decaying epsilon-greedy
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, 4)  # Explore
            else:
                action = np.argmax(Q_table[state])  # Exploit
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            N_table[state][action] += 1  # Increase the count
            alpha = 1 / N_table[state][action]  # Decaying learning rate based on the count
            target = reward + env.gamma * np.max(Q_table[next_state])
            Q_table[state][action] = Q_table[state][action] + alpha * (target - Q_table[state][action])
            state = next_state
        rewards[episode] = total_reward
        max_action_values.append(np.max(Q_table[env.start]))
    average_rewards = np.cumsum(rewards) / (np.arange(num_episodes) + 1)
    return average_rewards, max_action_values

def double_Q_learning(env, num_episodes, max_steps):
    Q1_table = np.zeros((env.grid_size, env.grid_size, 4))
    Q2_table = np.zeros((env.grid_size, env.grid_size, 4))
    N_table = np.zeros((env.grid_size, env.grid_size, 4))  # Keep track of the counts
    rewards = np.zeros(num_episodes)
    max_action_values = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < max_steps:
            # epsilon = env.get_epsilon(state)
            epsilon = 1 / np.sqrt(episode + 1)  # Decaying epsilon
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, 4)  # Explore
            else:
                action = np.argmax(Q1_table[state] + Q2_table[state])  # Exploit
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            N_table[state][action] += 1  # Increase the count
            alpha = 1 / N_table[state][action]  # Decaying learning rate based on the count
            if np.random.uniform(0, 1) < 0.5:
                target = reward + env.gamma * Q2_table[next_state][np.argmax(Q1_table[next_state])]
                Q1_table[state][action] = Q1_table[state][action] + alpha * (target - Q1_table[state][action])
            else:
                target = reward + env.gamma * Q1_table[next_state][np.argmax(Q2_table[next_state])]
                Q2_table[state][action] = Q2_table[state][action] + alpha * (target - Q2_table[state][action])
            state = next_state
        rewards[episode] = total_reward
        max_action_values.append(np.max(Q1_table[env.start] + Q2_table[env.start]))
    average_rewards = np.cumsum(rewards) / (np.arange(num_episodes) + 1)
    return average_rewards, max_action_values


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def clipped_max(Q_values, tau):
    A = len(Q_values)
    qt = np.ones(A) * tau / (A - 1)
    qt[np.argmax(Q_values)] = 1 - tau
    return qt

def clipped_softmax(Q_values, beta):
    sorted_indices = np.argsort(Q_values)[-3:]
    clipped_Q_values = np.full(Q_values.shape, -np.inf)
    clipped_Q_values[sorted_indices] = Q_values[sorted_indices]
    e_x = np.exp(beta * clipped_Q_values)
    return e_x / e_x.sum()

def smoothed_Q_learning(env, num_episodes, max_steps, smoothing_strategy="softmax"):
    Q_table = np.zeros((env.grid_size, env.grid_size, 4))
    N_table = np.zeros((env.grid_size, env.grid_size, 4))  # Keep track of the counts
    rewards = np.zeros(num_episodes)
    max_action_values = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0

        beta = 0.1 + 0.1 * episode  # Decaying beta
        tau = np.exp(-0.02 * episode)  # Decaying tau

        while not done and steps < max_steps:
            epsilon = 1 / np.sqrt(episode + 1)  # Decaying epsilon
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, 4)  # Explore
            else:
            #     Big Q Action Selection
            #     action = np.argmax(Q_table[state])  # Exploit
            #     Small Q Action Selection
                if smoothing_strategy == "softmax":
                    qt = softmax(Q_table[state])
                elif smoothing_strategy == "clipped_softmax":
                    qt = clipped_softmax(Q_table[state], beta)
                elif smoothing_strategy == "clipped_max":
                    qt = clipped_max(Q_table[state], tau)

                action = np.argmax(qt * Q_table[state])  # Exploit based on max q-value

            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            N_table[state][action] += 1  # Increase the count
            alpha = 1 / N_table[state][action]  # Decaying learning rate based on the count

            if smoothing_strategy == "softmax":
                qt = softmax(Q_table[next_state])
            elif smoothing_strategy == "clipped_softmax":
                qt = clipped_softmax(Q_table[next_state], beta)
            elif smoothing_strategy == "clipped_max":
                qt = clipped_max(Q_table[next_state], tau)

            target = reward + env.gamma * np.sum(qt * Q_table[next_state])
            Q_table[state][action] = Q_table[state][action] + alpha * (target - Q_table[state][action])

            state = next_state

        rewards[episode] = total_reward
        max_action_values.append(np.max(Q_table[env.start]))

    average_rewards = np.cumsum(rewards) / (np.arange(num_episodes) + 1)

    return average_rewards, max_action_values



num_experiments = 10000
num_episodes = 10000
max_steps = 5

Q_rewards = []
double_Q_rewards = []
Q_max_action_values = []
double_Q_max_action_values = []
smoothed_Q_softmax_rewards = []
smoothed_Q_softmax_max_action_values = []
smoothed_Q_clipped_max_rewards = []
smoothed_Q_clipped_max_max_action_values = []
smoothed_Q_clipped_softmax_rewards = []
smoothed_Q_clipped_softmax_max_action_values = []


for _ in range(num_experiments):
    env = GridWorld()
    rewards, max_action_values = Q_learning(env, num_episodes, max_steps)
    Q_rewards.append(rewards)
    Q_max_action_values.append(max_action_values)

    env = GridWorld()
    rewards, max_action_values = double_Q_learning(env, num_episodes, max_steps)
    double_Q_rewards.append(rewards)
    double_Q_max_action_values.append(max_action_values)

    env = GridWorld()
    rewards, max_action_values = smoothed_Q_learning(env, num_episodes, max_steps, smoothing_strategy="softmax")
    smoothed_Q_softmax_rewards.append(rewards)
    smoothed_Q_softmax_max_action_values.append(max_action_values)

    env = GridWorld()
    rewards, max_action_values = smoothed_Q_learning(env, num_episodes, max_steps, smoothing_strategy="clipped_max")
    smoothed_Q_clipped_max_rewards.append(rewards)
    smoothed_Q_clipped_max_max_action_values.append(max_action_values)

    env = GridWorld()
    rewards, max_action_values = smoothed_Q_learning(env, num_episodes, max_steps, smoothing_strategy="clipped_softmax")
    smoothed_Q_clipped_softmax_rewards.append(rewards)
    smoothed_Q_clipped_softmax_max_action_values.append(max_action_values)

Q_rewards = np.mean(np.array(Q_rewards), axis=0)
double_Q_rewards = np.mean(np.array(double_Q_rewards), axis=0)
Q_max_action_values = np.mean(Q_max_action_values, axis=0)
double_Q_max_action_values = np.mean(double_Q_max_action_values, axis=0)
smoothed_Q_softmax_rewards = np.mean(np.array(smoothed_Q_softmax_rewards), axis=0)
smoothed_Q_softmax_max_action_values = np.mean(smoothed_Q_softmax_max_action_values, axis=0)
smoothed_Q_clipped_max_rewards = np.mean(np.array(smoothed_Q_clipped_max_rewards), axis=0)
smoothed_Q_clipped_max_max_action_values = np.mean(smoothed_Q_clipped_max_max_action_values, axis=0)
smoothed_Q_clipped_softmax_rewards = np.mean(np.array(smoothed_Q_clipped_softmax_rewards), axis=0)
smoothed_Q_clipped_softmax_max_action_values = np.mean(np.array(smoothed_Q_clipped_softmax_max_action_values), axis=0)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(Q_rewards, label="Q-learning", color="blue")
plt.plot(double_Q_rewards, label="Double Q-learning", color="orange")
plt.plot(smoothed_Q_softmax_rewards, label="Smoothed Q-learning (Softmax)", color="green")
plt.plot(smoothed_Q_clipped_max_rewards, label="Smoothed Q-learning (Clipped Max)", color="red")
plt.plot(smoothed_Q_clipped_softmax_rewards, label="Smoothed Q-learning (Clipped Softmax)", color="purple")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(Q_max_action_values, label="Q-learning", color="blue")
plt.plot(double_Q_max_action_values, label="Double Q-learning", color="orange")
plt.plot(smoothed_Q_softmax_max_action_values, label="Smoothed Q-learning (Softmax)", color="green")
plt.plot(smoothed_Q_clipped_max_max_action_values, label="Smoothed Q-learning (Clipped Max)", color="red")
plt.plot(smoothed_Q_clipped_softmax_max_action_values, label="Smoothed Q-learning (Clipped Softmax)", color="purple")
plt.xlabel("Steps")
plt.ylabel("Max Action Value at Start State")
plt.legend()

plt.show()