import datetime
import pathlib
import uuid
from typing import List, Optional

import numpy as np

from utils.rl.replay.transition import Transition


class EpisodeReplayBuffer:
    def __init__(self, config):
        self.episodes = []
        #self.current_index = 0
        self.size_limit : Optional[int] = int(float(config.replay_size)) #number of steps
        self.steps = 0 #steps are all the steps that have ever been added to the replay buffer


    def total_steps(self):
        """returns the number of steps that have been added to the replay buffer
        (this may be larger then the number of transitions that are currently stored in the buffer)"""
        return self.steps

    def num_transitions(self):
        """ returns the number of transitions that are currently stored in the replay buffer"""
        return sum(self._length(episode) for episode in self.episodes)

    def num_episodes(self):
        return len(self.episodes)

    def _length(self, episode):
        """calculate the length of an episode"""
        return len(episode)

    def add_episode(self, episode : List[Transition]):
        self.steps += self._length(episode)
        while self.steps > self.size_limit:
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

    def sample_episodes_iterator(self, sequence_length : int  = None, over_sample_ends = False, seed=0):
        random = np.random.RandomState(seed)
        while True:
            episode_index = random.randint(0, self.num_episodes())
            episode = self.episodes[episode_index]
            if sequence_length:
                episode_length = self._length(episode)
                max_index_to_satisfy_seq_length = episode_length - sequence_length

                if max_index_to_satisfy_seq_length < 1:
                    #episode to short => skipp
                    continue

                if over_sample_ends:
                    index = min(random.randint(0, episode_length), max_index_to_satisfy_seq_length)
                else:
                    index = random.randint(0, max_index_to_satisfy_seq_length+1)

                episode = episode[index: index + sequence_length]
                #episode = [list(x) for x in episode ]

            states = []
            actions = []
            next_states = []
            rewards = []
            dones = []
            infos = []

            for transition in episode:
                states.append(transition.state)
                actions.append(transition.action)
                next_states.append(transition.next_state)
                rewards.append(transition.reward)
                dones.append(transition.done)
                infos.append(transition.info["dt"]) #todo handle case without dt

            states = np.asarray(states)
            actions = np.asarray(actions)
            rewards = np.asarray(rewards)
            next_states = np.asarray(next_states)
            dones = np.asarray(dones)
            infos = np.asarray(infos) #todo handle infos

            yield states, actions, next_states, rewards, dones, infos