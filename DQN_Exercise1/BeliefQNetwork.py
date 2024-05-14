import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """ Multi-layer Perceptron for encoding state and other components. """
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class BeliefQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(BeliefQNetwork, self).__init__()
        # Enhanced state and experience encoders using MLPs
        self.state_encoder = MLP(state_dim, hidden_dim)
        self.action_encoder = MLP(action_dim, hidden_dim)
        self.reward_encoder = MLP(1, hidden_dim)
        self.next_state_encoder = MLP(state_dim, hidden_dim)

        # Attention mechanism with Multi-Head Attention (optional)
        self.query_transform = nn.Linear(hidden_dim, hidden_dim)
        self.key_transform = nn.Linear(hidden_dim * 4, hidden_dim)

        # More complex action probability network
        self.action_prob_network = MLP(hidden_dim * 2, action_dim)

        # More complex Q-value prediction network
        self.q_value_network = MLP(hidden_dim * 2, action_dim)

    def forward(self, state, actions, rewards, next_states):
        state_enc = self.state_encoder(state)
        actions_enc = self.action_encoder(actions)
        rewards_enc = self.reward_encoder(rewards)
        next_states_enc = self.next_state_encoder(next_states)

        experiences = torch.cat([state_enc, actions_enc, rewards_enc, next_states_enc], dim=-1)

        query = self.query_transform(state_enc)
        keys = self.key_transform(experiences)
        attention_scores = torch.matmul(query, keys.T)
        attention_weights = F.softmax(attention_scores, dim=-1)

        context_vector = torch.sum(attention_weights.unsqueeze(-1) * experiences, dim=1)

        combined_representation = torch.cat([context_vector, state_enc], dim=-1)
        action_probs = F.softmax(self.action_prob_network(combined_representation), dim=-1)
        q_values = self.q_value_network(combined_representation)

        return action_probs, q_values