from abc import ABC, abstractmethod
from typing import Dict, Any

import torch


class ActionSelector(ABC):
    @abstractmethod
    def after_env_step(self, n_steps : int =1):
        pass

    @abstractmethod
    def select_actions(self, model_output: torch.Tensor, test: bool = False) -> torch.Tensor:
        pass

    def __call__(self, *args, **kwargs):
        return self.select_actions(*args, **kwargs)

    def get_metrics_for_logging(self)-> Dict[str, Any]:
        return {}