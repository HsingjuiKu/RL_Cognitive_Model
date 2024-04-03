import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
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

# Create Atari Breakout environment
env = gym.make('BreakoutDeterministic-v4')


# Define preprocess function
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (84, 84))
    #     cropped = resized[18:102, :]
    return np.expand_dims(resized, axis=-1)  # Add an extra dimension for the channels


# Create neural network model
model = Sequential()
model.add(Convolution2D(16, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 1)))
model.add(Convolution2D(32, (4, 4), strides=(2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
# print(model.summary())


# Create target model and sync weights
target_model = clone_model(model)
target_model.set_weights(model.get_weights())

# Use an optimizer
# 自适应学习率
optimizer = Adam(lr=0.00025)
loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(optimizer=optimizer, loss=loss_fn)
target_model.compile(optimizer=optimizer, loss=loss_fn)

# Storage space for experience replay
replay_memory = deque(maxlen=1000000)

# Lists to hold reward and max Q values
rewards = []
max_q_values = []

# Train the network
for episode in range(1000):  # for example, we can train 1000 episodes
    observation, _ = env.reset()
    observation = preprocess_frame(observation)

    episode_rewards = []
    episode_q_values = []

    for step in range(100):  # maximum 1000 steps per episode
        old_q = model.predict(np.expand_dims(observation, axis=0))
        action = np.argmax(old_q[0]) if np.random.rand() > 0.1 else env.action_space.sample()  # epsilon-greedy strategy

        # result = env.step(action)  # Change this line

        new_observation, reward, done, _, info = env.step(action)
        new_observation = preprocess_frame(new_observation)

        # Store experience
        replay_memory.append((observation, action, reward, new_observation, done))

        # Track rewards and max Q value
        episode_rewards.append(reward)
        episode_q_values.append(np.max(old_q[0]))

        # Experience replay
        if len(replay_memory) > 1000:
            minibatch = random.sample(replay_memory, 32)
            for obs, action, reward, new_obs, done in minibatch:
                update_target = reward if done else reward + 0.99 * np.max(
                    target_model.predict(np.expand_dims(new_obs, axis=0))[0])
                target = model.predict(np.expand_dims(obs, axis=0))
                target[0][action] = update_target
                model.train_on_batch(np.expand_dims(obs, axis=0), target)

            # Update target network every once in a while
            if step % 1000 == 0:
                target_model.set_weights(model.get_weights())

        observation = new_observation
        if done:
            break

    # Calculate average reward and max Q value for this episode
    avg_reward = np.mean(episode_rewards)
    avg_q_value = np.mean(episode_q_values)
    rewards.append(avg_reward)
    max_q_values.append(avg_q_value)

    # Save model weights every 100 episodes
    if episode % 100 == 0:
        model.save_weights('dqn_weights.h5f', overwrite=True)

# Plot reward and max Q values
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.title('Average reward per episode')

plt.subplot(1, 2, 2)
plt.plot(max_q_values)
plt.title('Average max predicted Q-value')

plt.show()
