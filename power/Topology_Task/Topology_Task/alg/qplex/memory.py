from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from functools import partial

from common.imports import *

def to_torch(object: object, device: th.device):
    return th.from_numpy(object).to(device)
    
class Buffer:
    def __init__(self, agent, envs, args, state_dim, device):
        self.obs_space = envs.observation_space[agent]
        self.act_space = envs.action_space[agent]

        self.to_torch = partial(to_torch, device=device)

        self.capacity = args.buffer_size
        self.n_envs = args.n_envs
        self.batch_size = args.batch_size
        self.sampling_seed = args.seed
        self.idx, self.full = 0, False

        self.capacity //= self.n_envs    # Fix capacity based on n° of envs

        base_shape = (self.capacity, self.n_envs,)
        self.obs = np.zeros(base_shape + (self.obs_space.shape[-1],), dtype=self.obs_space.dtype)
        self.action = np.zeros((base_shape), dtype=self.act_space.dtype)
        self.reward = np.zeros((base_shape), dtype=np.float32)
        self.next_obs = deepcopy(self.obs)
        self.done = np.zeros((base_shape), dtype=np.int8)
        self.state = np.zeros(base_shape + (state_dim,), dtype=self.obs_space.dtype)    # TODO do a separate state buffer to avoid duplicating info when time allows
        self.next_state = np.zeros(base_shape + (state_dim,), dtype=self.obs_space.dtype)    # TODO do a separate state buffer to avoid duplicating info when time allows

        self.attributes = ['obs', 'action', 'reward', 'next_obs', 'done', 'state', 'next_state']

    def _set_sampling(self):
        self.sampling_seed += 1
        np.random.seed(self.sampling_seed)

    def store(self, obs, action, reward, next_obs, done, state, next_state):
        if self.idx == self.capacity: 
            self.full, self.idx = True, 0

        self.obs[self.idx] = obs
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.done[self.idx] = done
        self.state[self.idx] = state
        self.next_state[self.idx] = next_state

        self.idx += 1    

    def sample(self):
        self._set_sampling()    # Set seed for sequential sampling in MARL (i.e. sample same idxs for each agent)

        idxs = self.get_sample_idxs()

        batch = {a: getattr(self, a)[idxs] for a in self.attributes}

        # Fix shapes for training
        batch['action'] = batch['action'][..., np.newaxis]
        batch['reward'] = batch['reward'][..., np.newaxis]
        batch['done'] = batch['done'][..., np.newaxis]

        return {k: self.to_torch(v) for k, v in batch.items()}

    def get_sample_idxs(self):
        # Choice avoids sampling with replacement
        sample_size = min(self.batch_size, self.size)

        # (experience, env_id) idxs
        return (np.random.choice(np.arange(0, self.size), size=sample_size, replace=False), 
                np.random.randint(0, high=self.n_envs, size=sample_size)) 

    def clear(self) -> None:
        self.idx, self.full = 0, False
    
    @property
    def size(self) -> int:
        return self.capacity if self.full else self.idx