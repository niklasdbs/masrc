import datetime

import networkx as nx
import numpy as np
from gym import spaces

from omegaconf import DictConfig
from envs.agent_state import AgentState
from envs.enums import ParkingStatus
from envs.observation_creators.observation_base import Observation
from envs.resource import Resource


class JointObservationDDQN(Observation):

    def __init__(self, config: DictConfig, graph: nx.Graph, number_of_resources: int, number_of_agents: int):
        super().__init__(config, graph, number_of_resources, number_of_agents)

        self.distance_normalization = config.distance_normalization
        number_of_features_per_resource = 4 + (5 * number_of_agents)

        self._shape = (number_of_resources, number_of_features_per_resource)

    @property
    def observation_space(self):
        # todo the box is also has negative values
        return spaces.Box(0, 1, shape=self._shape)

    def create_observation(self, env, _: AgentState) -> np.ndarray:
        """Creates the observation based on the graph.

        Structure of an observation:

            index | Description
            ------|-----------------------------------------------------------
            0     | One-hot encoding for parking status (Free)
            1     | One-hot encoding for parking status (Occupied)
            2     | One-hot encoding for parking status (Violation)
            3     | One-hot encoding for parking status (Fined)
            4     | Walking time (distance of agent to parking spot)
            5     | The current date and time (normalized)
            6     | The time of arrival of the agent (normalized)
            7     | An indicator for free or violation, ranging from -1 to +2
            8     | Normalized distance of the agent to the resource
        """
        state = np.zeros(self._shape, np.float32)
        length_of_day_in_seconds = env.end_of_current_day - env.start_time
        normalized_current_time = (env.current_time - env.start_time) / length_of_day_in_seconds

        for resource in env.resources:
            resource: Resource
            resource_status: ParkingStatus = resource.status
            # if resource_status == ParkingStatus.OCCUPIED and env.current_time + walking_time > (resource.arrival_time + resource.max_parking_duration_seconds):
            #     resource_status = ParkingStatus.IN_VIOLATION  #resource status needs to be set to in violation if occupied and potential violation time > walking time
            state[resource.ident, resource_status] = 1  # one hot encoding of the status of a resource

            offset = 5
            for agent_state in env.agent_states:
                agent_state: AgentState
                agent_offset = agent_state.agent_id * offset
                distance = self.distance(resource.source_node,
                                         resource.target_node,
                                         agent_state.position_node_source,
                                         agent_state.position_node,
                                         agent_state.position_on_edge)

                walking_time = distance / self.walking_speed

                agent_arrival_time = normalized_current_time + (walking_time / length_of_day_in_seconds)

                state[resource.ident, 4 + agent_offset] = walking_time / 3600
                state[resource.ident, 5 + agent_offset] = normalized_current_time
                state[resource.ident, 6 + agent_offset] = agent_arrival_time
                state[resource.ident, 7 + agent_offset] = min(2,
                                                 (env.current_time + walking_time - resource.arrival_time - resource.max_parking_duration_seconds)
                                                 / resource.max_parking_duration_seconds) if resource.status == ParkingStatus.OCCUPIED \
                                                                                             or resource.status == ParkingStatus.IN_VIOLATION else -1
                state[resource.ident, 8 + agent_offset] = distance / self.distance_normalization

        return state
