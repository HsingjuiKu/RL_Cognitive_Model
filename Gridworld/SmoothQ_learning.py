import numpy as np

# Define the Smooth Q-Learning class
class SmoothQLearning:
    def __init__(self, env, alpha=0.9, smooth_method="softmax"):
        self.env = env
        self.alpha = alpha
        self.Q = np.zeros((env.grid_size, env.grid_size, 4))
        self.smooth_method = smooth_method
        self.action_history = {}
        self.Q_history = [self.Q.copy()]
        self.abs_error = []
        self.action_l_count = []

        # Initialize the max possible reward for optimistic initialization
        self.initial_MaxQ = env.goal_reward + (env.max_steps - 1) * env.reward

        self.max_action_val_start = []

    def action(self, state, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.choice(4)
        else:
            action = np.argmax(self.Q[state])
        self.action_history.setdefault(state, []).append(action)
        return action

    def smooth(self, state, epsilon):
        if self.smooth_method == "softmax":
            if np.all(self.Q[state] == 0):
                return np.zeros_like(self.Q[state])
            else:
                Q_norm = epsilon * (self.Q[state] / np.max(np.abs(self.Q[state])))
                Q_norm -= np.max(Q_norm)
                return np.exp(Q_norm) / np.sum(np.exp(Q_norm))
        elif self.smooth_method == "clipped_max":
            qt = np.ones(4) * epsilon / 3
            qt[np.argmax(self.Q[state])] = 1 - epsilon
            return qt
        elif self.smooth_method == "clipped_softmax":
            k = 3
            clipped_Q = np.full(4, -np.inf)
            top_k_indices = np.argpartition(self.Q[state], -k)[-k:]
            clipped_Q[top_k_indices] = self.Q[state][top_k_indices]
            clipped_Q -= np.max(clipped_Q)
            return np.exp(clipped_Q) / np.sum(np.exp(clipped_Q))

    def learn(self, episodes=10000):
        rewards = []  # Initialize the list of rewards
        max_action_vals_start = []

        # Repeat for a certain number of episodes
        for episode in range(episodes):
            state = self.env.reset()  # Reset the environment and get the initial state
            done = False  # Initialize "done" to False
            total_reward = 0  # Initialize the total reward
            episode_rewards = []  # Initialize the list of rewards for this episode

            # Continue the loop until an episode is done
            while not done:
                epsilon = self.env.get_epsilon(state)  # Get the value of epsilon
                action = self.action(state, epsilon)  # Select an action

                # Get the new state, the reward, and whether the episode is done
                new_state, reward, done = self.env.step(action)
                total_reward += reward  # Update the total reward
                episode_rewards.append(reward)  # Add the reward to the list of rewards for this episode

                qt = self.smooth(new_state, epsilon)
                self.Q[state][action] += self.alpha * (
                            reward + self.env.gamma * np.sum(qt * self.Q[new_state]) - self.Q[state][action])
                state = new_state

                max_action_vals_start.append(np.max(self.Q[(0, 0)]))

            total_reward /= self.env.max_steps
            episode_rewards = [r / len(episode_rewards) for r in
                               episode_rewards]  # Compute average rewards for each time step in this episode
            rewards.append(episode_rewards)  # Add the rewards of this episode to the list of rewards

            self.Q_history.append(self.Q.copy())

            # Compute the average absolute error between max Q value and initial_MaxQ
            self.abs_error.append(np.mean(np.abs(np.max(self.Q, axis=2) - self.initial_MaxQ)))

            self.max_action_val_start.append(np.max(self.Q[(0, 0)]))

        self.max_action_val_start.append(np.mean(max_action_vals_start))
        return rewards  # Return the list of rewards