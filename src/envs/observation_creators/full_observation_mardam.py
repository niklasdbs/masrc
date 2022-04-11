import networkx as nx
import numpy as np
from gym import spaces

from omegaconf import DictConfig
from envs.agent_state import AgentState
from envs.enums import ParkingStatus
from envs.observation_creators.common_observation_logic import CommonObservationLogic
from envs.observation_creators.observation_base import Observation


class FullObservationMardam(CommonObservationLogic):

    def __init__(self, config: DictConfig, graph: nx.Graph, number_of_resources: int, number_of_agents: int):
        super().__init__(config, graph, number_of_resources, number_of_agents)
        self.agent_observation_feature_size = 0
        # self.agent_observation_feature_size += number_of_agents  # one hot encoding of vehicle index
        self.agent_observation_feature_size += 2  # x and y position of agent
        self.agent_observation_feature_size += 2  # x and y position of agent target
        self.agent_observation_feature_size += 1  # time until arrival

        self.resource_observation_feature_size = 0
        self.resource_observation_feature_size += 4  # one hot encoding of the state of a resource
        self.resource_observation_feature_size += 2  # position of the resource

        self.use_other_agent_features = config.get("use_other_agent_features", False)

        if self.use_other_agent_features:
            self.resource_observation_feature_size += 5 * self.number_of_agents

        self._observation_space = spaces.Tuple([
            spaces.Box(0, 1, shape=(number_of_agents, self.agent_observation_feature_size)),
            spaces.Box(0, 1, shape=(number_of_resources, self.resource_observation_feature_size)),
            spaces.Discrete(number_of_agents),
        ])

    @property
    def observation_space(self):
        return self._observation_space

    def create_observation(self, env, current_agent_state: AgentState):
        # customer states and vehicle states
        # customers = resources
        # vehicle = agent

        # customer state:
        # state of resource
        # positions
        #

        # vehicle state:
        # index of vehicle: one_hot
        # position of vehicle
        # time until arrival at destination

        if current_agent_state is None:
            current_agent_index = min((ags.agent_id for ags in env.agent_states if ags.current_action is None))
        else:
            current_agent_index = current_agent_state.agent_id
        length_of_day_in_seconds = env.end_of_current_day - env.start_time
        normalized_datetime = (env.current_time - env.start_time) / length_of_day_in_seconds

        graph: nx.Graph = env.graph

        agent_observations = self.create_agents_observations(env, graph)
        resource_observations = self.create_resource_observations(current_agent_state,
                                                                  env,
                                                                  graph,
                                                                  length_of_day_in_seconds,
                                                                  normalized_datetime)

        return agent_observations, resource_observations, current_agent_index

    def create_agents_observations(self, env, graph):
        agent_observations = np.zeros((self.number_of_agents, self.agent_observation_feature_size), dtype=np.float32)
        for agent_s in env.agent_states:
            # agent_observations[agent_s.agent_id, agent_s.agent_id] = 1  # one hot encode the index of the vehicle
            current_position = agent_s.position_node
            current_position_node = graph.nodes[current_position]
            x = (current_position_node["x"] - self._min_x) / (self._max_x - self._min_x)
            y = (current_position_node["y"] - self._min_y) / (self._max_y - self._min_y)

            agent_observations[agent_s.agent_id,  0] = x
            agent_observations[agent_s.agent_id,  1] = y

            if agent_s.current_route:
                destination = agent_s.current_route[-1]
                destination_node = graph.nodes[destination]
                x = (destination_node["x"] - self._min_x) / (self._max_x - self._min_x)
                y = (destination_node["y"] - self._min_y) / (self._max_y - self._min_y)

                agent_observations[agent_s.agent_id, 2] = x
                agent_observations[agent_s.agent_id, 3] = y
                distance_to_target = self.distance(agent_s.current_route[-2],
                                                   agent_s.current_route[-1],
                                                   agent_s.position_node_source,
                                                   agent_s.position_node,
                                                   agent_s.position_on_edge)

                time_until_arrival = distance_to_target / self.walking_speed
                agent_observations[agent_s.agent_id, 4] = time_until_arrival / 3600
        return agent_observations

    def create_resource_observations(self,
                                     current_agent_state,
                                     env,
                                     graph,
                                     length_of_day_in_seconds,
                                     normalized_datetime):

        resource_observations = np.zeros((self.number_of_resources, self.resource_observation_feature_size),
                                         dtype=np.float32)

        for resource in env.resources:
            resource_observations[resource.ident, resource.status] = 1  # one hot encoding of parking status
            x = (resource.x - self._min_x) / (self._max_x - self._min_x)
            y = (resource.y - self._min_y) / (self._max_y - self._min_y)

            resource_observations[resource.ident, 4] = x
            resource_observations[resource.ident, 5] = y

            if self.use_other_agent_features:
                for agent_state in env.agent_states:
                    agent_state : AgentState
                    offset = 6 + (5* agent_state.agent_id)
                    distance = self.distance(resource.source_node,
                                             resource.target_node,
                                             agent_state.position_node_source,
                                             agent_state.position_node,
                                             agent_state.position_on_edge)

                    walking_time = distance / self.walking_speed

                    agent_arrival_time = normalized_datetime + (walking_time / length_of_day_in_seconds)
                    resource_observations[resource.ident, offset] = walking_time / 3600
                    resource_observations[resource.ident, offset + 1] = normalized_datetime
                    resource_observations[resource.ident, offset + 2] = agent_arrival_time
                    resource_observations[resource.ident, offset + 3] = min(2,
                                                      (
                                                              env.current_time + walking_time - resource.arrival_time - resource.max_parking_duration_seconds)
                                                      / resource.max_parking_duration_seconds) if resource.status == ParkingStatus.OCCUPIED \
                                                                                                  or resource.status == ParkingStatus.IN_VIOLATION else -1
                    resource_observations[resource.ident, offset + 4] = distance / self.distance_normalization


        return resource_observations
