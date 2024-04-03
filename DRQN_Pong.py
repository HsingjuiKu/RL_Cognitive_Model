import gym
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, LSTM
from tensorflow.keras.optimizers import Adam
import random
from tensorflow.keras.layers import Reshape
from collections import deque


# Create Atari Pong environment
env = gym.make('PongDeterministic-v4')

# Define preprocess function
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return np.expand_dims(resized, axis=-1)  # Add an extra dimension for the channels

# Create DRQN model
model = Sequential()
model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 1)))
model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Reshape((1, -1)))
model.add(LSTM(512, return_sequences=False, activation='tanh'))
model.add(Dense(env.action_space.n, activation='linear'))
print(model.summary())

# Use an optimizer
optimizer = Adam(lr=0.00025)
loss_fn = tf.keras.losses.MeanSquaredError()

model.compile(optimizer=optimizer, loss=loss_fn)



#Initialize environment. v4 means no action repeat
env = gym.make('PongNoFrameskip-v4')
# Storage space for experience replay
replay_memory = deque(maxlen=1000000)

# Train the network
for episode in range(1000):  # for example, we can train 1000 episodes
    observation, _ = env.reset()
    observation = preprocess_frame(observation)
    # Initial hidden state for LSTM
    hidden_state = np.zeros((1, 512))
    for step in range(1000):  # maximum 1000 steps per episode
        # q_values, hidden_state = model.predict([np.expand_dims(observation, axis=0), hidden_state])
        q_values = model.predict(np.expand_dims(observation, axis=0))
        action = np.argmax(q_values[0]) if np.random.rand() > 0.1 else env.action_space.sample()  # epsilon-greedy strategy

        new_observation, reward, done, info, value = env.step(action)
        new_observation = preprocess_frame(new_observation)

        # Store experience
        replay_memory.append((observation, action, reward, new_observation, done, hidden_state))

        observation = new_observation
        if done:
            break

    # Experience replay
    if len(replay_memory) > 1000:
        minibatch = random.sample(replay_memory, 32)
        for obs, action, reward, new_obs, hidden_state in minibatch:
            q_values_next, _ = model.predict([np.expand_dims(new_obs, axis=0), hidden_state])
            update_target = reward if done else reward + 0.99 * np.max(q_values_next[0])
            q_values, _ = model.predict([np.expand_dims(obs, axis=0), hidden_state])
            q_values[0][action] = update_target
            model.train_on_batch([np.expand_dims(obs, axis=0), hidden_state], q_values)

    # Save model weights every 100 episodes
    if episode % 100 == 0:
        model.save_weights('drqn_weights.h5f', overwrite=True)

