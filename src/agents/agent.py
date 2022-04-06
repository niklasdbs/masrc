"""
Module contains the base class for creating agents.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple, Union, Dict

import gym.spaces
import torch

from omegaconf import DictConfig
from utils.rl.replay.episode_replay_buffer import EpisodeReplayBuffer
from utils.rl.replay.replay_buffer import ReplayBuffer


class Agent(ABC):
    """The base class for creating agents. Cannot be instantiated."""

    def __init__(self, action_space: gym.Space, observation_space: gym.Space, graph, config: DictConfig, **kwargs):
        """
        Initializes a new instance.

        :param action_space: The agents action space.
        :param observation_space: The agents observation space.
        """
        self.device: torch.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.action_space = action_space
        self.observation_space = observation_space
        self.test = False
        self.graph = graph

        # todo maybe move replay buffer elsewhere, not all agents need one
        self.replay_whole_episodes = config.replay_whole_episodes
        self.buffer = EpisodeReplayBuffer(config) if self.replay_whole_episodes else ReplayBuffer(config)
        self.current_step = 0

        self._hidden_state = None

    @abstractmethod
    def act(self, state, **kwargs) -> Tuple[Union[int, dict], Union[None, Any]]:
        """
        Returns an action based on the state of the environment.

        :param state: The current state of the environment.
        :return: The action to be executed and maybe additional information like log prob, ...
        """

    def after_env_step(self, n_steps: int = 1):
        self.current_step += n_steps

    def log_weight_histogram(self, logger, global_step: int, prefix: str = ""):
        pass

    def get_agent_metrics_for_logging(self) -> Dict[str, Any]:
        return {}

    def save_model(self, path: Path, step: int, agent_number):
        """
        Saves the current model of the agent.

        :return: None.
        """

    def load_model(self, path: Path, step: int, agent_number):
        """
        Loads the pretrained model.

        :param filename: Filepath to the saved model.
        :return: None.
        """

    def set_test(self, test: bool):
        """
        Sets the test value of the agent. If true, the agent stops exploring and only exploits.

        :param test: Boolean value of the test parameter.
        :return: None.
        """
        self.test = test

    def reset_hidden_state(self):
        """
        Resets the hidden state, necessary for rnn agents
        @return:
        """
        self._hidden_state = None
