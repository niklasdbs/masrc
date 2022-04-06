from typing import Any

import networkx as nx
import numpy as np
from gym import spaces
from omegaconf import DictConfig

from envs.agent_state import AgentState
from envs.enums import ParkingStatus
from envs.observation_creators.common_observation_logic import CommonObservationLogic
from envs.observation_creators.resource_observations.basic_resource_observation import BasicResourceObservation
from envs.resource import Resource


class FullObservationDDQN(CommonObservationLogic):

    def __init__(self, config: DictConfig, graph: nx.Graph, number_of_resources: int, number_of_agents: int):
        super().__init__(config, graph, number_of_resources, number_of_agents)
        self.resource_encoder = BasicResourceObservation(config, graph, number_of_resources, number_of_agents)

    @property
    def observation_space(self):
        return self.resource_encoder.observation_space

    def create_observation(self, env, current_agent_state: AgentState) -> np.ndarray:
        """Creates the observation based on the graph.
        """

        return self.resource_encoder.create_observation(env, current_agent_state)
