import numpy as np
from omegaconf import DictConfig

from utils.rl.schedules.schedule import Schedule


class EpsilonSchedule(Schedule):

    def __init__(self, epsilon_initial: float, epsilon_decay: float, epsilon_min: float, epsilon_decay_start: float):
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_decay_start = epsilon_decay_start
        self.epsilon_initial = epsilon_initial
        self.epsilon = epsilon_initial
        self.current_step: int = 0

    def __call__(self, *args, **kwargs):
        return self.epsilon

    def step(self, n_steps: int = 1):
        self.current_step += n_steps

        if self.current_step >= self.epsilon_decay_start:
            self.epsilon = max(self.epsilon_min,
                               self.epsilon_initial - self.epsilon_decay * (self.current_step - self.epsilon_decay_start))



# https://github.com/oxwhirl/pymarl/blob/master/src/components/epsilon_schedules.py
class DecayThanFlatEpsilonSchedule(Schedule):
    def __init__(self,
                 epsilon_initial: float,
                 epsilon_min: float,
                 epsilon_decay_start: float,
                 steps_till_min_epsilon: int,
                 decay="exp"):
        self.current_step: int = 0
        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_min = epsilon_min
        self.epsilon_decay_start = epsilon_decay_start
        self.decay = decay
        self.steps_till_min_epsilon = steps_till_min_epsilon
        self.delta = (self.epsilon_initial - self.epsilon_min) / self.steps_till_min_epsilon

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.steps_till_min_epsilon / np.log(self.epsilon_min) if self.epsilon_min > 0 else 1

    def __call__(self, *args, **kwargs):
        return self.epsilon

    def step(self, n_steps: int = 1):
        self.current_step += n_steps

        if self.current_step >= self.epsilon_decay_start:
            if self.decay in ["linear"]:
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon_initial - self.delta * (self.current_step - self.epsilon_decay_start))
            elif self.decay in ["exp"]:
                self.epsilon = min(self.epsilon_initial,
                                   max(self.epsilon_min,
                                       np.exp(- (self.current_step - self.epsilon_decay_start) / self.exp_scaling)))


def init_from_config(config: DictConfig) -> DecayThanFlatEpsilonSchedule:
    return DecayThanFlatEpsilonSchedule(
            epsilon_initial=config.epsilon_initial,
            epsilon_min=config.epsilon_min,
            epsilon_decay_start=config.epsilon_decay_start,
            steps_till_min_epsilon=config.steps_till_min_epsilon,
            decay=config.epsilon_decay
    )
