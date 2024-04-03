import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import parl
from parl.algorithms import DQN

class MLP(parl.Model):
    """ Linear network to solve Cartpole problem.
    Args:
        input_dim (int): Dimension of observation space.
        output_dim (int): Dimension of action space.
    """

    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        hidden_dim1 = 256
        hidden_dim2 = 256
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x