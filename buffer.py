import random
import numpy as np

class ReplayMemory:
    def __init__(self):
        self.buffer = []
        self.position = 0
        self.batch = None

    def push(self, data_tuple):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.buffer)

class FullReplayMemory(ReplayMemory):
    def __init__(self, capacity, obs_shape):
        super(FullReplayMemory, self).__init__()
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.obs_default = np.zeros(obs_shape, dtype=np.uint8)

    def push(self, data_tuple):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = data_tuple
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size, resample=True):
        if resample:
            self.batch = random.sample(self.buffer, batch_size)
        return map(np.stack, zip(*self.batch))

    def reset(self):
        self.buffer = []
        self.position = 0
        self.batch = None

    def last_obs(self, n=3, axis=0):
        obs = []
        another = False
        for i in range(self.position-1, self.position-n-1, -1):
            if i<0:
                obs.insert(0, self.obs_default)
            else:
                if self.buffer[i][3]:
                    another = True
                if another:
                    obs.insert(0, self.obs_default)
                else:
                    obs.insert(0, self.buffer[i][0])
        assert len(obs) == n
        samples = np.stack(obs, axis=-1)
        return samples