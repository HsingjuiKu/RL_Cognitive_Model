import os
import gym
import numpy as np
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from collections import deque
import atari_py
import matplotlib.pyplot as plt

# Define the Deep Q-learning agent
class DQNAgent:
    def __init__(self, env_name='BreakoutDeterministic-v4'):
        # Suppress TensorFlow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Initialize the environment and the model
        self.env = gym.make(env_name)
        self.model = self._build_model()

        # Initialize the target model and sync it with the model weights
        self.target_model = clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        # Initialize the replay memory
        self.replay_memory = deque(maxlen=1000000)

        # Initialize lists to hold rewards and max Q-values
        self.rewards = []
        self.max_q_values = []

    # Preprocess the frame (convert it to grayscale and resize)
    @staticmethod
    def preprocess_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return np.expand_dims(resized, axis=-1)

    # Create the Q-learning model
    def _build_model(self):
        model = Sequential()
        model.add(Convolution2D(16, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 1)))
        model.add(Convolution2D(32, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))

        # Compile the model using an Adam optimizer and a mean squared error loss function
        optimizer = Adam(lr=0.00025)
        loss_fn = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer=optimizer, loss=loss_fn)
        return model

    # Train the model for a number of episodes
    def train(self, episodes=100000, steps=1000, save_frequency=100):
        for episode in range(episodes):
            observation, _ = self.env.reset()
            observation = self.preprocess_frame(observation)

            episode_rewards = []
            episode_q_values = []

            for step in range(steps):
                # Select an action using the model's Q-values or randomly
                old_q = self.model.predict(np.expand_dims(observation, axis=0))
                action = np.argmax(old_q[0]) if np.random.rand() > 0.1 else self.env.action_space.sample()

                new_observation, reward, done, _, info = self.env.step(action)
                new_observation = self.preprocess_frame(new_observation)

                # Store the experience in the replay memory
                self.replay_memory.append((observation, action, reward, new_observation, done))

                # Keep track of rewards and Q-values
                episode_rewards.append(reward)
                episode_q_values.append(np.max(old_q[0]))

                # Perform experience replay if there are enough experiences in the memory
                if len(self.replay_memory) > 1000:
                    minibatch = random.sample(self.replay_memory, 32)
                    for obs, action, reward, new_obs, done in minibatch:
                        update_target = reward if done else reward + 0.99 * np.max(
                            self.target_model.predict(np.expand_dims(new_obs, axis=0))[0])
                        target = self.model.predict(np.expand_dims(obs, axis=0))
                        target[0][action] = update_target
                        self.model.train_on_batch(np.expand_dims(obs, axis=0), target)

                    # Update the target network periodically
                    if step % 1000 == 0:
                        self.target_model.set_weights(self.model.get_weights())

                observation = new_observation
                if done:
                    break

            # Compute average reward and Q-value for this episode
            avg_reward = np.mean(episode_rewards)
            avg_q_value = np.mean(episode_q_values)
            self.rewards.append(avg_reward)
            self.max_q_values.append(avg_q_value)

            # Save model weights periodically
            if episode % save_frequency == 0:
                self.model.save_weights('dqn_weights.h5f', overwrite=True)

    # Plot the training results
    def plot_results(self):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.rewards)
        plt.title('Average reward per episode')

        plt.subplot(1, 2, 2)
        plt.plot(self.max_q_values)
        plt.title('Average max predicted Q-value')

        plt.show()


# Create an instance of the agent and start training
agent = DQNAgent()
agent.train()
agent.plot_results()
