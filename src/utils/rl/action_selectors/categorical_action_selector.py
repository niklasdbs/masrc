from typing import Dict, Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.distributions import Categorical

from utils.rl.action_selectors.action_selector import ActionSelector
from utils.rl.schedules import epsilon_schedule
from utils.rl.schedules.epsilon_schedule import EpsilonSchedule


class CategoricalActionSelector(ActionSelector):

    def __init__(self, config: DictConfig):
        try:
            self._epsilon = epsilon_schedule.init_from_config(config)
        except:
            #todo ugly fallback
            self._epsilon = EpsilonSchedule(epsilon_initial=config.epsilon,
                                            epsilon_decay_start=config.epsilon_decay_start,
                                            epsilon_decay=config.epsilon_decay,
                                            epsilon_min=config.epsilon_min)

        self.test_max_likelihood = config.test_max_likelihood

    @property
    def epsilon(self):
        return self._epsilon()

    def get_metrics_for_logging(self)-> Dict[str, Any]:
        return {
                "epsilon": self.epsilon
        }

    def select_actions(self, probabilities: torch.Tensor, test: bool = False) -> torch.Tensor:
        # todo mask
        avail_actions = torch.ones_like(probabilities)

        epsilon = 0.0 if test else self.epsilon
        if not test:
            number_of_actions_for_epsilon = probabilities.size(-1)
            probabilities = ((1 - epsilon) * probabilities) + (
                        torch.ones_like(probabilities) * epsilon / number_of_actions_for_epsilon)

        masked_probabilities = probabilities  # todo we would need to clone if we want to use the tensor in other places (as we modify in-place)
        masked_probabilities[avail_actions == 0.0] = 0.0

        if test and self.test_max_likelihood:
            picked_actions = masked_probabilities.max(dim=-1)[1]
        else:
            picked_actions = Categorical(masked_probabilities).sample().long()

        return picked_actions

    def __call__(self, *args, **kwargs):
        return self.select_actions(*args, **kwargs)

    def after_env_step(self, n_steps: int = 1):
        self._epsilon.step(n_steps)
