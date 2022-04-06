import networkx as nx
import numpy as np
from gym import spaces

from omegaconf import DictConfig
from envs.agent_state import AgentState
from envs.enums import ParkingStatus
from envs.observation_creators.common_observation_logic import CommonObservationLogic
from envs.observation_creators.observation_base import Observation
from envs.observation_creators.resource_observations.basic_resource_observation import BasicResourceObservation


class FullObservationGRCNSharedAgent(CommonObservationLogic):

    def __init__(self, config: DictConfig, graph: nx.Graph, number_of_resources: int, number_of_agents: int):
        super().__init__(config, graph, number_of_resources, number_of_agents)

        self.resource_encoder = BasicResourceObservation(config, graph, number_of_resources, number_of_agents)
        self._observation_space = spaces.Dict({
                "resource_observations" : self.resource_encoder.observation_space,
                "distance_to_action": spaces.Box(0, 1, shape=(1,)),
                "current_agent_id": spaces.Discrete(number_of_agents),
        })


    @property
    def observation_space(self):
        return self._observation_space

    def create_observation(self, env, current_agent_state: AgentState):
        resource_observations = self.resource_encoder.create_observation(env, current_agent_state)
        distance_to_action = self.distance_agent_to_action(env, current_agent_state)

        return {
                "resource_observations":resource_observations,
                "distance_to_action": distance_to_action,
                "current_agent_id": current_agent_state.agent_id,
        }

    def distance_agent_to_action(self, env, current_agent_state: AgentState):
        distance_to_action = np.array([self.distance(edge[0],
                                                     edge[1],
                                                     current_agent_state.position_node_source,
                                                     current_agent_state.position_node,
                                                     current_agent_state.position_on_edge)
                                       for edge in env.edge_id_to_edge_mapping.values()], dtype=np.float32)
        distance_to_action = distance_to_action/self.distance_normalization
        return distance_to_action
