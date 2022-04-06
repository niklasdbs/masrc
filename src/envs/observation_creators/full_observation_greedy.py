import networkx as nx
from gym import spaces
from omegaconf import DictConfig

from envs.agent_state import AgentState
from envs.observation_creators.observation_base import Observation


class FullObservationGreedy(Observation):

    def __init__(self, config : DictConfig, graph: nx.Graph, number_of_resources : int, number_of_agents : int):
        super().__init__(config, graph, number_of_resources, number_of_agents)
        self._shape = (number_of_resources, 9)#todo

    @property
    def observation_space(self):
        return spaces.Tuple([spaces.Box(0, 1, shape=self._shape), spaces.Box(0, 1, shape=self._shape)])#todo

    def create_observation(self, env, current_agent_state: AgentState):
        return env, self.shortest_path_lengths



