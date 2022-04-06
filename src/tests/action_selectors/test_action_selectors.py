import numpy as np
import torch
from hydra import initialize, compose

from utils.rl.action_selectors import EpsilonGreedyActionSelector, CategoricalActionSelector


def test_eps_greedy_action_selector():
    with initialize(config_path="../../config"):
        # config is relative to a module
        cfg = compose(config_name="config", overrides=["experiment_name=test"])
        action_selector = EpsilonGreedyActionSelector(cfg)
        assert action_selector.epsilon == 1.0

        test_tensor = torch.tensor(np.array([[[-1.0, 3.0, -33, 99, 3.0]]]))

        action = action_selector.select_actions(test_tensor, test=True)
        assert action.item() == 3


def test_eps_categorical_action_selector():
    with initialize(config_path="../../config"):
        # config is relative to a module
        cfg = compose(config_name="config", overrides=["experiment_name=test"])
        action_selector = CategoricalActionSelector(cfg)
        action_selector.test_max_likelihood = True
        assert action_selector.epsilon == 1.0

        test_tensor = torch.tensor(np.array([[[-1.0, 3.0, -33, 99, 3.0]]]))

        action = action_selector.select_actions(test_tensor, test=True)
        assert action.item() == 3

        action_selector.test_max_likelihood = False
        test_tensor = torch.tensor(np.array([[[0, 0, 0, 0, 1, 0]]]))
        action = action_selector.select_actions(test_tensor, test=True)
        assert action.item() == 4
