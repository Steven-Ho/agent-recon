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
    def __init__(self, capacity):
        super(FullReplayMemory, self).__init__()
        self.capacity = capacity

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
