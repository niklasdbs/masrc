import datetime
import pathlib
import uuid
from collections import deque
from typing import List, Optional

import numpy as np

from utils.rl.replay.transition import Transition


class ReplayBuffer:
    def __init__(self, config):
        self.size_limit : Optional[int] = int(float(config.replay_size)) #number of steps
        self.steps = 0 #steps are all the steps that have ever been added to the replay buffer
        self.buffer = deque(maxlen=self.size_limit if self.size_limit else 99999999999999999) #todo this is a bit ugly


    def total_steps(self):
        """returns the number of steps that have been added to the replay buffer
        (this may be larger then the number of transitions that are currently stored in the buffer)"""
        return self.steps

    def num_transitions(self):
        """ returns the number of transitions that are currently stored in the replay buffer"""
        return len(self.buffer)

    def add_transition(self, transition : Transition, *args, **kwargs):
        self.steps += 1
        self.buffer.append(transition)

    def __len__(self):
        return len(self.buffer)

    def save_to_file(self, directory, transitions, current_step):
        directory = pathlib.Path(directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

        filename = directory / f'{timestamp}-{current_step}-{uuid.uuid1()}.npz'

        np.savez_compressed(filename, transitions)

    def sample_iterator(self, seed=0):
        random = np.random.RandomState(seed)
        while True:
            transition_index = random.randint(0, self.num_transitions())
            transition = self.buffer[transition_index]

            if transition.additional_information is None:
            #todo handle cases where dt is not present
                yield transition.state, transition.action, transition.next_state, np.float32(transition.reward), transition.done, transition.info["dt"]
            else:
                yield transition.state, transition.action, transition.next_state, np.float32(transition.reward), transition.done, transition.info["dt"], transition.additional_information