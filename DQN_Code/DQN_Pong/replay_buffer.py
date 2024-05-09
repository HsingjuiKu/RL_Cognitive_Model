import numpy as np


class ReplayBuffer:
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, size):
        """
        Initialise a buffer of a given size for storing transitions
        :param size: the maximum number of transitions that can be stored
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        :param state: the agent's initial state
        :param action: the action taken by the agent
        :param reward: the reward the agent received
        :param next_state: the subsequent state
        :param done: whether the episode terminated
        """
        data = (state, action, reward, next_state, done)

        # If the next index is greater than or equal to the length of the storage,
        # it means that the storage is not full yet, so we append the new data at the end.
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            # If the next index is less than the length of the storage,
            # it means that the storage is full, so we overwrite the oldest data with the new data.
            self._storage[self._next_idx] = data

        # We update the next index so that it always points to the place where the next data will be inserted.
        # The modulo operation ensures that the next index is always within the bounds of the storage,
        # creating a circular or looping effect.
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        '''
        The function of this code is to extract the corresponding experience from the store based on the given index,
        and then organise them into arrays of state, action, reward, next_state, and done or not done, respectively.
        :param indices:
        :return: np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
        '''

        # Initialize empty lists to hold states, actions, rewards, next_states, and dones.
        states, actions, rewards, next_states, dones = [], [], [], [], []

        # For each index, extract the corresponding experience from storage,
        # and append each component of the experience to the appropriate list.
        for i in indices:
            data = self._storage[i]
            state, action, reward, next_state, done = data
            states.append(np.array(state, copy=False))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)

        # Convert each list to a numpy array before returning.
        # These arrays will be used in the learning algorithm.
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        :param batch_size: the number of transitions to sample
        :return: a mini-batch of sampled transitions
        """
        indices = np.random.randint(0, len(self._storage) - 1, size=batch_size)
        return self._encode_sample(indices)