from typing import Optional

import torch.utils.data

from utils.rl.replay.complete_episode_replay_buffer import CompleteEpisodeReplayBuffer
from utils.rl.replay.episode_replay_buffer import EpisodeReplayBuffer

""" 
Will not work with num_workers > 0
"""


class IterableEpisodeDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 replay_buffer: CompleteEpisodeReplayBuffer,  # todo create base class
                 max_sequence_length: Optional[int] = None,
                 over_sample_ends: bool = False,
                 seed=0):
        super(IterableEpisodeDataset).__init__()
        self.replay_buffer = replay_buffer
        self.iterator = self.replay_buffer.sample_episodes_iterator(max_sequence_length, over_sample_ends, seed)

    def __iter__(self):
        return self.iterator
