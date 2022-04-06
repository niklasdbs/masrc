import networkx as nx
import numpy as np
from gym import spaces
from omegaconf import DictConfig

from envs.agent_state import AgentState
from envs.observation_creators.joint_observation_ddqn import JointObservationDDQN


class FlatJointObservation(JointObservationDDQN):

    def __init__(self, config: DictConfig, graph: nx.Graph, number_of_resources: int, number_of_agents: int):
        super().__init__(config, graph, number_of_resources, number_of_agents)
        self._shape_wrapped = (self._shape[0]*self._shape[1],self._shape[0], self._shape[1])
        #only the first dimension is relevant (the others are needed to unflatten for certain kind of agents)

    @property
    def observation_space(self):
        # todo the box is also has negative values
        return spaces.Box(0, 1, shape=self._shape_wrapped)

    def create_observation(self, env, _: AgentState) -> np.ndarray:
        return super().create_observation(env, _).flatten()
