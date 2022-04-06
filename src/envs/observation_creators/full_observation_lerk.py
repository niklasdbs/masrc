from abc import ABC
import datetime
from typing import Union, Any

import networkx as nx
import numpy as np
from gym import spaces

from omegaconf import DictConfig
from envs.agent_state import AgentState
from envs.enums import ParkingStatus
from envs.observation_creators.observation_base import Observation


class FullObservationLerk(Observation):

    def __init__(self, config: DictConfig, graph: nx.Graph, number_of_resources: int, number_of_agents: int):
        super().__init__(config, graph, number_of_resources, number_of_agents)

        self._shape = (number_of_resources, 5)

    def observation_space(self):
        return spaces.Tuple([spaces.Box(0, 1, shape=self._shape), spaces.Box(0, 1, shape=self._shape), spaces.Box(0, 1, shape=self._shape),
                             spaces.Box(0, 1, shape=self._shape)])  # todo

    def create_observation(self, env, current_agent_state: AgentState) -> Union[np.ndarray, Any]:
        return env, self.shortest_path_lengths, current_agent_state

    def shape(self):
        return self._shape
