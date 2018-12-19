import random
from collections import deque, namedtuple

import numpy as np
import torch


class ReplayBuffer:

    # initialize ReplayBuffer
    def __init__(self, params):
        self.device = params.device
        self.action_size = params.action_size
        self.memory = deque(maxlen=params.buffer_size)
        self.batch_size = params.batch_size
        self.experience = namedtuple("Experience", field_names=["state",
                                                                "action",
                                                                "reward",
                                                                "next_state",
                                                                "done"])

    # add a new experience to memory
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    # randomly sample a batch of experiences from memory
    def sample(self, completion):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return (states, actions, rewards, next_states, dones)

    # return the size of the internal memory
    def __len__(self):
        return len(self.memory)



class PrioritizedReplayBuffer(object):
    """
    https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb
    """

    def __init__(self, params, prob_alpha=0.6, prob_beta=0.5):
        self.prob_alpha = prob_alpha
        self.prob_beta = prob_beta
        self.device = params.device
        self.buffer_size = params.buffer_size
        self.batch_size = params.batch_size
        self.device = params.device
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)  # long array
        self.experience = namedtuple("Experience", field_names=["state",
                                                                "action",
                                                                "reward",
                                                                "next_state",
                                                                "done"])

    def add(self, state, action, reward, next_state, done):

        # if self.buffer is empty return 1.0, else max
        max_priority = self.priorities.max() if self.buffer else 1.0
        exp = self.experience(state, action, reward, next_state, done)

        # if buffer has rooms left
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(exp)
        else:
            self.buffer[self.pos] = exp

        # assign max priority
        self.priorities[self.pos] = max_priority

        # update index
        self.pos = (self.pos + 1) % self.buffer_size

    def sample(self, completion):

        beta = self.prob_beta + (1 - self.prob_beta) * completion

        # if buffer is maxed out..
        if len(self.buffer) == self.buffer_size:
            # all priorities are the same as self.priorities
            priorities = self.priorities
        else:
            # all priorities are up to self.pos cuz it's not full yet
            priorities = self.priorities[:self.pos]

        # $ P(i) = (p_i^\alpha) / \Sigma_k p_k^\alpha $
        probabilities_a = priorities ** self.prob_alpha
        sum_probabilties_a = probabilities_a.sum()
        P_i = probabilities_a / sum_probabilties_a

        sampled_indices = np.random.choice(len(self.buffer), self.batch_size, p=P_i)
        experiences = [self.buffer[idx] for idx in sampled_indices]

        # $ w_i = ( 1/N * 1/P(i) ) ** \beta $
        # $ w_i = ( N * P(i) ** (-1 * \beta) ) $
        N = len(self.buffer)
        weights = (N * P_i[sampled_indices]) ** (-1 * beta)

        #  For stability reasons, we always normalize weights by 1/ maxi wi so
        #  that they only scale the update downwards.
        weights = weights / weights.max()

        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float().to(self.device)
        next_states = torch.from_numpy(
            np.vstack([exp.next_state for exp in experiences if exp is not None])).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([exp.done for exp in experiences if exp is not None]).astype(np.uint8)).float().to(self.device)
        weights = torch.from_numpy(np.vstack(weights)).float()

        return states, actions, rewards, next_states, dones, sampled_indices, weights

    def update_priorities(self, batch_indicies, batch_priorities):
        for idx, priority in zip(batch_indicies, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)