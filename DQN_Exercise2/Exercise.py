import gym
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torchvision.transforms as T

# Create Atari Breakout environment
env = gym.make('BreakoutDeterministic-v4')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define preprocess function
transform = T.Compose([T.ToPILImage(),
                       T.Grayscale(),
                       T.Resize((84, 84)),
                       T.ToTensor()])


def preprocess_frame(frame):
    return transform(frame).unsqueeze(0).to(device)


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(self.feature_size(h, w), 256)
        self.fc2 = nn.Linear(256, outputs)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)

    def feature_size(self, h, w):
        return self.conv2(self.conv1(torch.zeros(1, 1, h, w))).view(1, -1).size(1)


policy_net = DQN(84, 84, env.action_space.n).to(device)
target_net = DQN(84, 84, env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.00025)

# Storage space for experience replay
replay_memory = deque(maxlen=1000000)

# Train the network
for episode in range(1000):  # for example, we can train 1000 episodes
    state = preprocess_frame(env.reset())
    for step in range(1000):  # maximum 1000 steps per episode
        state = state.to(device)
        with torch.no_grad():
            action_values = policy_net(state)

        action = torch.argmax(
            action_values).item() if random.random() > 0.1 else env.action_space.sample()  # epsilon-greedy strategy

        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_frame(next_state)

        # Store experience
        replay_memory.append((state, action, reward, next_state, done))

        # Experience replay
        if len(replay_memory) > 1000:
            minibatch = random.sample(replay_memory, 32)
            for state, action, reward, next_state, done in minibatch:
                update_target = reward if done else reward + 0.99 * torch.max(target_net(next_state))
                action_values = policy_net(state)
                action_values[0][action] = update_target

                loss = nn.functional.mse_loss(policy_net(state), action_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network every once in a while
            if step % 1000 == 0:
                target_net.load_state_dict(policy_net.state_dict())

        state = next_state
        if done:
            break

    # Save model weights every 100 episodes
    if episode % 100 == 0:
        torch.save(policy_net.state_dict(), 'dqn_weights.pth')
