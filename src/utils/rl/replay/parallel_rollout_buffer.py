from collections import defaultdict
from typing import Optional

import random

import numpy as np
import torch
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataloader import default_collate

from utils.rl.replay.episode_collate import EpisodeCollate
from utils.rl.replay.transition import Transition


class ParallelRolloutBuffer:
    def __init__(self):
        self.buffer = defaultdict(list)
        self.episode_collate = EpisodeCollate(1)

    def __len__(self):
        return len(self.buffer)

    def add_transition(self, transition : Transition, index : int = 0):
        self.buffer[index].append(transition)

    def clear(self):
        self.buffer.clear()


    def get_batch(self):
        batch = [default_convert(_make_batch_row_transition(trajectory)) for trajectory in self.buffer.values()]
        return self.episode_collate(batch)

    # def sample_episodes_iterator(self, sequence_length: Optional[int] = None, over_sample_ends=False, seed=0):
    #     random = np.random.RandomState(seed)
    #     while True:
    #         episode_index = self._on_policy_pointer % self.num_episodes()
    #         self._on_policy_pointer += 1
    #
    #         converted_episode = self._make_batch_row_transition(episode)
    #         yield converted_episode

def _make_batch_row_transition(episode):
    states = []
    actions = []
    rewards = []
    dones = []
    infos = []
    next_states = []

    for transition in episode:
        states.append(transition.state)
        actions.append(transition.action)
        rewards.append(np.float32(transition.reward))
        dones.append(transition.done)
        infos.append(transition.info["dt"])  # todo handle case without dt
        next_states.append(transition.next_state)

    return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "infos": infos,
            "next_states": next_states
    }
