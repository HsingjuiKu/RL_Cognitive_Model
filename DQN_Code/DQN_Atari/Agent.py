# Inspired from https://github.com/raillab/dqn
from gym import spaces
import numpy as np

from model_nature import DQN as DQN_nature
from model_neurips import DQN as DQN_neurips
from replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F

import random


class DQNAgent:
    def __init__(self,
                 observation_space: spaces.Box,  # The space of possible observations from environment
                 action_space: spaces.Discrete,  # The space of possible actions
                 replay_buffer: ReplayBuffer,  # Buffer for storing experiences
                 dqn_variant,  # Variant of DQN to use, either "dqn" or "ddqn"
                 lr,  # Learning rate for optimizer
                 batch_size,  # Batch size for training
                 gamma,  # Discount factor for future rewards
                 device=torch.device("cuda:1"),  # Device for PyTorch to use
                 dqn_type="neurips"):  # Type of DQN to use

        # Initialize agent variables
        self.memory = replay_buffer
        self.batch_size = batch_size
        self.dqn_variant = dqn_variant
        self.gamma = gamma
        self.action_space = action_space
        # Set default epsilon and delta values based on dqn_variant
        if self.dqn_variant == "ssdqn" or self.dqn_variant == "cssdqn":
            self.epsilon = 0.1
        elif self.dqn_variant == "scdqn":
            self.delta = 0.1

        # Choose DQN type
        if(dqn_type=="neurips"):
            DQN = DQN_neurips
        else:
            DQN = DQN_nature
        # Create policy and target networks
        self.policy_network = DQN(observation_space, action_space).to(device)
        self.target_network = DQN(observation_space, action_space).to(device)

        # Initialize target network with policy network's weights
        self.update_target_network()
        self.target_network.eval()
        # Define optimizer
        self.optimiser = torch.optim.RMSprop(self.policy_network.parameters(), lr=lr)
        self.device = device

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        device = self.device
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = np.array(states) / 255.0
        next_states = np.array(next_states) / 255.0
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        with torch.no_grad():
            if self.dqn_variant == "ddqn":
                _, max_next_action = self.policy_network(next_states).max(1)
                max_next_q_values = self.target_network(next_states).gather(1, max_next_action.unsqueeze(1)).squeeze()
            else:
                next_q_values = self.target_network(next_states)
                max_next_q_values, _ = next_q_values.max(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        input_q_values = self.policy_network(states)
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = F.smooth_l1_loss(input_q_values, target_q_values)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        del states
        del next_states
        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())



    def act(self, state: np.ndarray, reward=None, done=None):
        """
        Select an action using ε-greedy strategy from the Q-network given the state
        :param state: the current state
        :param reward: the previous reward. Used for tracking rewards over episodes.
        :param done: flag indicating if the episode is done.
        :return: the action to take
        """
        epsilon = 0.1  # Probability of selecting a random action

        device = self.device
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # ε-greedy action selection
        if random.random() < epsilon:
            return random.randint(0, self.action_space.n - 1)

        with torch.no_grad():
            q_values = self.policy_network(state)
            _, action = q_values.max(1)
            return action.item()

    def run_policy_and_compute_actual_rewards(self, env, n_episodes=10):
        """
        Run the current policy for n_episodes and compute the mean cumulative reward
        :param env: the environment to run the policy on
        :param n_episodes: the number of episodes to run the policy for
        :return: the mean cumulative reward
        """
        total_rewards = 0.0
        for _ in range(n_episodes):
            state = env.reset()
            done = False
            episode_rewards = 0.0
            while not done:
                action = self.act(state)
                state, reward, done, _ = env.step(action)
                episode_rewards += reward
            total_rewards += episode_rewards
        mean_rewards = total_rewards / n_episodes
        return mean_rewards

    def compute_q_value(self, state: np.ndarray, action: int):
        """
        Compute the Q-value for a given state-action pair
        :param state: the current state
        :param action: the action taken
        :return: the Q-value of the state-action pair
        """
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_network(state)
            # q_value = q_values[0][action]
            q_value=torch.max(q_values)
            return q_value.item()


    def get_predicted_values(self):
        # Returns the list of predicted Q values
        return self.predicted_values

    def get_actual_values(self):
        # Returns the list of actual Q values
        return self.actual_values

    def get_predicted_values_min(self):
        # Returns the list of min predicted Q values
        return self.predicted_values_min

    def get_predicted_values_max(self):
        # Returns the list of max predicted Q values
        return self.predicted_values_max

    def get_moving_avg_values(self):
        # Returns the list of avg predicted Q values
        return self.moving_avg_values
