import networkx as nx
import numpy as np
from gym import spaces

from omegaconf import DictConfig
from envs.agent_state import AgentState
from envs.observation_creators.observation_base import Observation


class ZeroObservation(Observation):

    def __init__(self, config : DictConfig, graph: nx.Graph, number_of_resources : int, number_of_agents : int):
        super().__init__(config, graph, number_of_resources, number_of_agents)

        self._shape = (1,1)

    def observation_space(self):
        return spaces.Box(0, 1, shape=self._shape)

    def create_observation(self, env, current_agent_state: AgentState):
        return np.zeros(self._shape, dtype=np.float32)
