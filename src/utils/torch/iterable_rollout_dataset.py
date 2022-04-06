from typing import Optional

import torch.utils.data

from utils.rl.replay.episode_replay_buffer import EpisodeReplayBuffer

""" 
Will not work with num_workers > 0
"""
class IterableRolloutDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 replay_buffer : EpisodeReplayBuffer, #todo create base class
                 sequence_length: Optional[int] = None,
                 over_sample_ends = False,
                 seed=0):
        super(IterableRolloutDataset).__init__()
        self.replay_buffer = replay_buffer
        self.iterator = self.replay_buffer.sample_episodes_iterator(sequence_length, over_sample_ends, seed)


    def __iter__(self):
        return self.iterator

