import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPAttention(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MLPAttention, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean_layer = nn.Linear(128, action_dim)
        self.std_layer = nn.Linear(128, action_dim)

    def forward(self, state, q_values):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        std = torch.exp(self.std_layer(x))  # Ensure std is positive

        # Normalize Q values to be used as weights for action probabilities
        q_weights = F.softmax(q_values, dim=-1)

        # Create a normal distribution and sample actions
        normal_dist = torch.distributions.Normal(mean, std)
        actions = normal_dist.sample()

        # Compute log probabilities of the actions
        log_probs = normal_dist.log_prob(actions)

        # Weight log probabilities by Q values
        weighted_probs = torch.exp(log_probs) * q_weights

        return actions, weighted_probs

