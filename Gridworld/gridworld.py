import numpy as np

class GridWorld:
    def __init__(self, grid_size=3, gamma=0.9, penalty=-12, reward=10, goal_reward=5, max_steps=5):
        self.grid_size = grid_size
        self.gamma = gamma
        self.penalty = penalty
        self.reward = reward
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.state_visits = np.zeros((grid_size, grid_size))
        self.reset()
        self.start = (0, 0)  # Define the start position

    def reset(self):
        self.x = 0
        self.y = 0
        self.steps = 0
        self.state_visits[:] = 0
        return (self.x, self.y)

    def step(self, action):
        self.state_visits[self.x, self.y] += 1

        if action == 0 and self.y != 0:
            self.y -= 1
        elif action == 1 and self.x != self.grid_size-1:
            self.x += 1
        elif action == 2 and self.y != self.grid_size-1:
            self.y += 1
        elif action == 3 and self.x != 0:
            self.x -= 1

        self.steps += 1

        if self.x == self.grid_size - 1 and self.y == self.grid_size - 1:
            return (self.x, self.y), self.goal_reward, True
        elif self.steps >= self.max_steps:
            return (self.x, self.y), 0, True
        else:
            return (self.x, self.y), np.random.choice([self.penalty, self.reward], p=[0.5, 0.5]), False

    def get_epsilon(self, state):
        x, y = state
        n_s = self.state_visits[x, y]
        return 1 / np.sqrt(n_s + 1)
