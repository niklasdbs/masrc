from typing import Dict, Any, Union

import numpy as np
import torch
from numpy import ndarray
from omegaconf import DictConfig
from torch.distributions import Categorical

from utils.rl.action_selectors.action_selector import ActionSelector
from utils.rl.schedules import epsilon_schedule
from utils.rl.schedules.epsilon_schedule import EpsilonSchedule


class EpsilonGreedyActionSelector(ActionSelector):

    def __init__(self, config: DictConfig):
        try:
            self._epsilon = epsilon_schedule.init_from_config(config)
        except:
            #todo ugly fallback
            self._epsilon = EpsilonSchedule(epsilon_initial=config.epsilon,
                                            epsilon_decay_start=config.epsilon_decay_start,
                                            epsilon_decay=config.epsilon_decay,
                                            epsilon_min=config.epsilon_min)

    @property
    def epsilon(self):
        return self._epsilon()

    def get_metrics_for_logging(self)-> Dict[str, Any]:
        return {
                "epsilon": self.epsilon
        }

    def select_actions(self, q_values: torch.Tensor, test: bool = False) -> ndarray:
        # todo mask
        #possible shape (BxAxQ)
        if test:
            picked_actions = q_values.argmax(dim=-1).cpu().numpy()
        else:
            random_numbers = torch.rand_like(q_values[:, :, 0])
            pick_random = (random_numbers < self.epsilon).long()
            avail_actions = torch.ones_like(q_values)
            random_actions = Categorical(avail_actions.float()).sample().long()
            picked_actions = pick_random * random_actions + (1 - pick_random) * q_values.argmax(dim=-1)
            picked_actions_np = picked_actions.cpu().numpy()
            return picked_actions_np
            # should_choice_random = np.random.choice(2, q_values.size(0), p=[1-self.epsilon, self.epsilon])
            # picked_actions = np.array([(np.random.random_integers(low=0, high=q_values.size(1)-1) if should_choice_random[i] else q_values[i].argmax().item()) for i in range(q_values.size(0))])

        return picked_actions

    def after_env_step(self, n_steps:int=1):
        self._epsilon.step(n_steps)
