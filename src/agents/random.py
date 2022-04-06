"""
Module contains functionality for a random agent.
"""

from enum import Enum

from gym.spaces import Discrete, Box

from agents.agent import Agent
from omegaconf import DictConfig


class Strategy(Enum):
    """
    Describes the strategy of the random agent.
        RANDOM_EDGE -> choose random edge from current position
        RANDOM_ROUTE -> choose a random route to an edge with a parking spot
    """
    RANDOM_EDGE = 0
    RANDOM_ROUTE = 1


class RandomAgent(Agent):
    """A baseline agent taking random actions."""

    def __init__(self, action_space: Discrete, observation_space: Box, config: DictConfig, graph):
        """
        Initializes a new instance.

        :param action_space: The action space of this agent.
        :param observation_space: The observation space of this agent (unused).
        :param params: The parameters for this agent (only strategy is necessary).
        """
        super().__init__(action_space, observation_space, graph, config)
        self.strategy: Strategy = getattr(Strategy, config.strategy)

    def act(self, state) -> (int, None):
        # type dict means we are using an parallel environment
        if type(state) is dict:
            actions: dict = {}
            for agent_id, value in state.items():
                if value['needs_to_act'] == 1:
                    actions[agent_id] = self._get_action()
            return actions, None
        else:
            return self._get_action(), None

    def _get_action(self) -> int:
        """
        Returns: an random action
        """
        if self.strategy == Strategy.RANDOM_ROUTE:
            return self.action_space.sample()
        elif self.strategy == Strategy.RANDOM_EDGE:
            pass  # todo implement
        else:
            raise NotImplementedError
