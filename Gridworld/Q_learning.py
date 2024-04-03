import numpy as np


class QLearning:
    def __init__(self, env, alpha=0.9):
        self.env = env  # Set the environment
        self.alpha = alpha  # Set the learning rate
        self.Q = np.zeros((env.grid_size, env.grid_size, 4))  # Initialize the Q function
        self.action_history = {}  # Initialize the action history
        self.Q_history = []  # Initialize the history of Q values
        self.max_action_val_start = []

        # Calculate the MaxQ values
        self.MaxQ = np.max(self.Q, axis=2)
        self.MaxQ_history = [self.MaxQ.copy()]

        # Initialize the max possible reward for optimistic initialization
        self.initial_MaxQ = env.goal_reward + (env.max_steps - 1) * env.reward

        # Initialize data for tracking average abolute error and average action 'l'
        self.abs_error = []
        self.action_l_count = []

    def action(self, state, epsilon):
        # If a random number is less than epsilon, select a random action
        if np.random.rand() < epsilon:
            action = np.random.choice(4)
        # Otherwise, select the action with the highest expected reward from Q
        else:
            action = np.argmax(self.Q[state])
        self.action_history.setdefault(state, []).append(action)  # Store the action
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

                # Update Q
                self.Q[state][action] += self.alpha * (
                        reward + self.env.gamma * np.max(self.Q[new_state]) - self.Q[state][action])
                state = new_state  # Update the state

                max_action_vals_start.append(np.max(self.Q[(0, 0)]))

            # Update the MaxQ values and store it
            self.MaxQ = np.max(self.Q, axis=2)
            self.MaxQ_history.append(self.MaxQ.copy())

            # Store the Q values at the end of each episode
            self.Q_history.append(self.Q.copy())

            # Compute the average absolute error and add it to abs_error
            self.abs_error.append(np.mean(np.abs(self.MaxQ - self.initial_MaxQ)))

            episode_rewards = [r / len(episode_rewards) for r in
                               episode_rewards]  # Compute average rewards for each time step in this episode
            rewards.append(episode_rewards)  # Add the rewards of this episode to the list of rewards

            self.max_action_val_start.append(np.max(self.Q[(0, 0)]))

        self.max_action_val_start.append(np.mean(max_action_vals_start))
        return rewards  # Return the list of rewards