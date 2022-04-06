from functools import partial
from typing import Callable

import torch
import torch.nn
from omegaconf import DictConfig


def identity(x):
    return x

#todo use in all places
def build_reward_transformation_fn_from_config(config: DictConfig) -> Callable[[torch.Tensor], torch.Tensor]:
    if config.reward_clipping:
        if config.reward_clipping == "clip" or isinstance(config.reward_clipping, bool):
            transform = partial(torch.clip, min=0, max=1)
        elif config.reward_clipping == "tanh":
            transform = torch.tanh
        else:
            raise Exception("Unkown reward clipping function")
        return transform
    else:
        return identity
