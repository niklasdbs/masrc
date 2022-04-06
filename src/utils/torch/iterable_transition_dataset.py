from typing import Optional

import torch.utils.data

from utils.rl.replay.replay_buffer import ReplayBuffer

""" 
Will not work with num_workers > 0
"""
class IterableTransitionDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 replay_buffer : ReplayBuffer, #todo create base class
                 seed=0):
        super(IterableTransitionDataset).__init__()
        self.replay_buffer = replay_buffer
        self.iterator = self.replay_buffer.sample_iterator(seed)

    def __iter__(self):
        return self.iterator
