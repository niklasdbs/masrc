import datetime
import pathlib
import uuid
from typing import Optional, TypeVar, Generic

import numpy as np
from omegaconf import DictConfig

from utils.rl.replay.clde_transition import CLDETransition
from utils.rl.replay.transition import Transition

T = TypeVar('T', CLDETransition, Transition)


class CompleteEpisodeReplayBuffer(Generic[T]):
    def __init__(self, config: DictConfig, transition_type=CLDETransition):
        assert config.replay_whole_episodes
        self.row_creator = \
            _make_batch_row_clde \
                if transition_type == CLDETransition \
                else _make_batch_row_transition

        self.episodes: [[T]] = []
        # self.current_index = 0
        self.size_limit: Optional[int] = int(float(config.replay_size))  # number of episodes
        self.steps = 0  # steps are all the steps that have ever been added to the replay buffer
        self.on_policy_replay = config.get("on_policy_replay", False)
        self._on_policy_pointer = 0

    def total_steps(self):
        """returns the number of steps that have been added to the replay buffer
        (this may be larger then the number of transitions that are currently stored in the buffer)"""
        return self.steps

    def num_transitions(self):
        """ returns the number of transitions that are currently stored in the replay buffer"""
        return sum(self._length(episode) for episode in self.episodes)

    def num_episodes(self):
        return len(self.episodes)

    def __len__(self):
        return self.num_episodes()

    def _length(self, episode):
        """calculate the length of an episode"""
        return len(episode)

    def add_episode(self, episode: [T]):
        assert self._length(episode) > 0
        self.steps += self._length(episode)
        while len(self.episodes) > self.size_limit:
            del_episode = self.episodes.pop(0)
            self.steps -= self._length(del_episode)

        self.episodes.append(episode)
        # self.episodes[self.current_index] = episode
        # self.current_index = (self.current_index + 1) % self.size_limit

    def save_episodes_to_file(self, directory, episodes, current_step):
        directory = pathlib.Path(directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

        for episode in episodes:
            filename = directory / f'{timestamp}-{current_step}-{uuid.uuid1()}.npz'

            np.savez_compressed(filename, episode)

    def sample_episodes_iterator(self, sequence_length: Optional[int] = None, over_sample_ends=False, seed=0):
        random = np.random.RandomState(seed)
        while True:
            if self.on_policy_replay:
                episode_index = self._on_policy_pointer % self.num_episodes()
                self._on_policy_pointer += 1
            else:
                episode_index = random.randint(0, self.num_episodes())
            episode = self.episodes[episode_index]
            if sequence_length:  # zero or None
                episode_length = self._length(episode)
                max_index_to_satisfy_seq_length = episode_length - sequence_length

                if max_index_to_satisfy_seq_length < 1:
                    # take whole episode
                    pass
                else:
                    if over_sample_ends:
                        index = min(random.randint(0, episode_length), max_index_to_satisfy_seq_length)
                    else:
                        index = random.randint(0, max_index_to_satisfy_seq_length + 1)

                    episode = episode[index: index + sequence_length]

            converted_episode = self.row_creator(episode)
            yield converted_episode


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


def _make_batch_row_clde(episode):
    local_observations = []
    global_observation = []
    actions = []
    action_masks = []
    rewards = []
    dones = []
    infos = []

    for transition in episode:
        local_observations.append(transition.local_observations)
        global_observation.append(transition.global_observation)
        actions.append(transition.actions)
        action_masks.append(transition.action_mask)
        rewards.append(transition.rewards)
        dones.append(transition.dones)
        infos.append(transition.infos)  # todo handle case without dt

    return {
            "local_observations": local_observations,
            "global_observation": global_observation,
            "actions": actions,
            "action_masks": action_masks,
            "rewards": rewards,
            "dones": dones,
            "infos": infos
    }
