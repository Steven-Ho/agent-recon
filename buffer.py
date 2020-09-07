import random
import numpy as np

class ReplayMemory:
    def __init__(self):
        self.buffer = []
        self.position = 0
        self.count = 0
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
    def __init__(self, capacity, obs_shape, kws, shapes, dtypes):
        super(FullReplayMemory, self).__init__()
        self.capacity = int(capacity)
        self.obs_shape = obs_shape[:-1]
        self.obs_default = np.zeros(self.obs_shape, dtype=np.uint8)
        self.buffer = dict()
        self.kws = kws
        for i in range(len(kws)):
            self.buffer[kws[i]] = np.zeros((self.capacity,) + shapes[i], dtype=dtypes[i])
        self.n = obs_shape[-1]

    def push(self, **kwargs):
        for key in kwargs:
            if len(self.buffer) < self.capacity:
                self.buffer[key][self.position] = kwargs[key]
        self.position = int((self.position + 1) % self.capacity)
        self.count = max(self.count, self.position)

    def sample(self, batch_size):
        indices = np.random.choice(self.count, batch_size, replace=False)
        batch = dict()
        for kw in self.kws:
            if kw not in ['obs', 'new_obs']:
                batch[kw] = self.buffer[kw][indices]
            else:
                obs = np.zeros([batch_size] + self.obs_shape + [self.n])
                obs[:,:,:,-1] = self.buffer[kw][indices]
                valid = [True for _ in range(batch_size)]
                for i in range(-1, -4, -1):
                    for j in range(indices.shape[0]):
                        if indices[j]+i<0:
                            obs[j,:,:,i-1] = self.obs_default
                        else:
                            if self.buffer['done'][indices[j]+i] or indices[j]+i==self.position-1:
                                valid[j] = False
                            if not valid[j]:
                                obs[j,:,:,i-1] = self.obs_default
                            else:
                                obs[j,:,:,i-1] = self.buffer[kw][indices[j]+i]
                batch[kw] = obs/255.

        return [batch[kw] for kw in self.kws]

    def reset(self):
        for key in self.kws:
            self.buffer[key] = []
        self.position = 0
        self.count
        self.batch = None

    def last_obs(self, axis=0):
        obs = []
        another = False
        for i in range(self.position-1, self.position-self.n, -1):
            if i<0:
                obs.insert(0, self.obs_default)
            else:
                if self.buffer['done'][i]:
                    another = True
                if another:
                    obs.insert(0, self.obs_default)
                else:
                    obs.insert(0, self.buffer['obs'][i])
        assert len(obs) == self.n-1
        samples = np.stack(obs, axis=-1)
        return samples