import numpy as np


class DoubleQLearning:
    def __init__(self, env, alpha=0.9):
        self.env = env
        self.alpha = alpha
        self.QA = np.zeros((env.grid_size, env.grid_size, 4))
        self.QB = np.zeros((env.grid_size, env.grid_size, 4))
        self.action_history = {}
        self.QA_history = [self.QA.copy()]
        self.QB_history = [self.QB.copy()]
        self.abs_error = []
        self.action_l_count = []

        # Initialize the max possible reward for optimistic initialization
        self.initial_MaxQ = env.goal_reward + (env.max_steps - 1) * env.reward

        self.max_action_val_start = []

    def action(self, state, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.choice(4)
        else:
            action = np.argmax(self.QA[state] + self.QB[state])
        self.action_history.setdefault(state, []).append(action)
        return action

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

                if np.random.rand() < 0.5:
                    self.QA[state][action] += self.alpha * (
                            reward + self.env.gamma * self.QB[new_state][np.argmax(self.QA[new_state])] -
                            self.QA[state][action])
                else:
                    self.QB[state][action] += self.alpha * (
                            reward + self.env.gamma * self.QA[new_state][np.argmax(self.QB[new_state])] -
                            self.QB[state][action])
                state = new_state

                max_action_vals_start.append(max(np.max(self.QA[(0, 0)]), np.max(self.QB[(0, 0)])))

            total_reward /= self.env.max_steps
            episode_rewards = [r / len(episode_rewards) for r in
                               episode_rewards]  # Compute average rewards for each time step in this episode
            rewards.append(episode_rewards)  # Add the rewards of this episode to the list of rewards

            self.QA_history.append(self.QA.copy())
            self.QB_history.append(self.QB.copy())

            # Compute the average absolute error between max Q value and initial_MaxQ
            MaxQ = np.maximum(self.QA, self.QB)
            self.abs_error.append(np.mean(np.abs(MaxQ - self.initial_MaxQ)))

            max_action_val = max(np.max(self.QA[(0, 0)]), np.max(self.QB[(0, 0)]))
            self.max_action_val_start.append(max_action_val)

        self.max_action_val_start.append(np.mean(max_action_vals_start))
        return rewards  # Return the list of rewards