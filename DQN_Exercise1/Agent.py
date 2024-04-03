from random import random
import parl
import paddle
import math
import numpy as np

class DQNAgent(parl.Agent):
    """
    Agent of DQN_Exercise1.
    """

    def __int__(self, algorithm, memory, cfg):
        super(DQNAgent, self).__int__(algorithm)
        self.n_actions = cfg['n_actions']
        self.epsilon = cfg['epsilon_start']
        self.sample_count = 0
        self.epsilon_start = cfg['epsilon_start']
        self.epsilon_end = cfg['epsilon_end']
        self.epsilon_decay = cfg['epsilon_decay']
        self.batch_size = cfg['batch_size']
        self.global_step = 0
        self.update_target_steps = 600
        self.memory = memory #replay buffer

    def sample_action(self, state):
        self.sample_count += 1
        # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1.* self.sample_count / self.epsilon_decay)
        if random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = self.predict_action(state)
        return action

    def predict_action(self, state):
        state = paddle.to_tensor(state, dtype='float32')
        q_values = self.alg.predict(state)
        action = q_values.argmax().numpy()[0]
        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

        action_batch = np.expand_dims(action_batch, axis=-1)
        reward_batch = np.expand_dims(reward_batch, axis=-1)
        done_batch = np.expand_dims(done_batch, axis=-1)
        state_batch = paddle.to_tensor(state_batch, dtype='float32')
        action_batch = paddle.to_tensor(action_batch, dtype='int32')
        reward_batch = paddle.to_tensor(reward_batch, dtype='float32')
        next_state_batch = paddle.to_tensor(next_state_batch, dtype='float32')
        done_batch = paddle.to_tensor(done_batch, dtype='float32')
        loss = self.alg.learn(state_batch, action_batch, reward_batch, next_state_batch, done_batch)



