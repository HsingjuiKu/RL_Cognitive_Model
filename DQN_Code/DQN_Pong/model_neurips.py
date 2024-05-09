from gym import spaces
import torch.nn as nn
import torch.nn.functional as F


# Class structure loosely inspired by https://towardsdatascience.com/beating-video-games-with-deep-q-networks-7f73320b9592
class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    neurips DQN paper.
    """

    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        assert type(
            observation_space) == spaces.Box, 'observation_space must be of type Box'
        assert len(
            observation_space.shape) == 3, 'observation space must have the form channels x width x height'
        assert type(
            action_space) == spaces.Discrete, 'action_space must be of type Discrete'

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=observation_space.shape[0], out_channels=16, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
        #     nn.ReLU()
        # )
        #
        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=32 * 9 * 9, out_features=256),
        #     nn.ReLU(),
        #     nn.Linear(in_features=256, out_features=action_space.n)
        # )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # LSTM layer, you may need to adjust the input size and hidden size
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)  # Flatten and reduce dimension
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)  # LSTM layer

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=action_space.n)  # in_features changed to match LSTM's hidden size
        )

        # Adjusted input features of the fully-connected layer
        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=64 * 7 * 7, out_features=512),
        #     nn.ReLU(),
        #     nn.Linear(in_features=512, out_features=action_space.n)
        # )
        # Adjusted input features of the fully-connected layer
        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=256, out_features=action_space.n)  # in_features changed to match LSTM's hidden size
        # )

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)  # Flatten conv output
        fc1_out = F.relu(self.fc1(conv_out)).view(x.size()[0], 1, -1)  # Pass through fc1 and reshape for LSTM
        lstm_out, _ = self.lstm(fc1_out)  # LSTM layer
        lstm_out = lstm_out[:, -1, :]  # take the output of the last LSTM step
        return self.fc2(lstm_out)
