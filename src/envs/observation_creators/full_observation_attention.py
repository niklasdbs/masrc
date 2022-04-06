import networkx as nx
import numpy as np
from gym import spaces

from omegaconf import DictConfig
from envs.agent_state import AgentState
from envs.enums import ParkingStatus
from envs.observation_creators.common_observation_logic import CommonObservationLogic
from envs.observation_creators.observation_base import Observation
from envs.observation_creators.resource_observations.basic_resource_observation import BasicResourceObservation


class FullObservationAttention(CommonObservationLogic):

    def __init__(self, config: DictConfig, graph: nx.Graph, number_of_resources: int, number_of_agents: int):
        super().__init__(config, graph, number_of_resources, number_of_agents)
        self.agent_observation_feature_size = 0
        self.agent_observation_feature_size += number_of_agents  # one hot encoding of vehicle index
        self.agent_observation_feature_size += 2  # x and y position of agent
        self.agent_observation_feature_size += 1  # time until arrival
        self.agent_observation_feature_size += 2  # x and y destination of agent

        self.resource_encoder = BasicResourceObservation(config, graph, number_of_resources, number_of_agents)

        # todo the box value range
        self._observation_space = spaces.Tuple([
                spaces.Box(0, 1, shape=(number_of_agents, self.agent_observation_feature_size)),
                self.resource_encoder.observation_space,
                spaces.Box(-1, 2, shape=(number_of_agents-1, *self.resource_encoder.observation_space.shape)),#todo do not hardcode
                spaces.Box(0, 1, shape=(number_of_resources, self.number_of_agents * (self.number_of_agents + 4))),
                spaces.Discrete(number_of_agents),
        ])

    @property
    def observation_space(self):
        return self._observation_space

    def create_observation(self, env, current_agent_state: AgentState):
        current_agent_index = current_agent_state.agent_id

        agent_observations = self.create_agent_observations(env, current_agent_state)
        resource_observations = self.resource_encoder.create_observation(env, current_agent_state)
        other_agent_resource_observations = self.create_other_agent_resource_observations(env, current_agent_state)
        distance_observations = self.create_observation_for_distances(env, current_agent_state)

        return agent_observations, resource_observations, other_agent_resource_observations, distance_observations, current_agent_index

    def create_other_agent_resource_observations(self, env, current_agent_state: AgentState):
        observations = []
        for agent_state in env.agent_states:
            if current_agent_state.agent_id == agent_state.agent_id:
                continue

            obs = self.resource_encoder.create_observation(env, agent_state)
            observations.append(obs)

        return np.stack(observations)



    def create_agent_observations(self, env, current_agent_state: AgentState):
        graph: nx.Graph = env.graph

        agent_observations = np.zeros((self.number_of_agents, self.agent_observation_feature_size), dtype=np.float32)
        for agent_s in env.agent_states:
            agent_observations[agent_s.agent_id, agent_s.agent_id] = 1  # one hot encode the index of the vehicle

            current_position = agent_s.position_node
            x = (graph.nodes[current_position]["x"] - self._min_x) / (self._max_x - self._min_x)
            y = (graph.nodes[current_position]["y"] - self._min_y) / (self._max_y - self._min_y)

            agent_observations[agent_s.agent_id, self.number_of_agents + 0] = x
            agent_observations[agent_s.agent_id, self.number_of_agents + 1] = y

            if agent_s.current_action:
                destination = agent_s.current_route[-2], agent_s.current_route[-1]
                x = (graph.nodes[destination]["x"] - self._min_x) / (self._max_x - self._min_x)
                y = (graph.nodes[destination]["y"] - self._min_y) / (self._max_y - self._min_y)

                agent_observations[agent_s.agent_id, self.number_of_agents + 2] = x
                agent_observations[agent_s.agent_id, self.number_of_agents + 3] = y

                time_until_arrival = self.shortest_path_lengths[agent_s.current_route[-2]][agent_s.current_route[-1]][
                                         agent_s.position_node] / self.walking_speed
                agent_observations[agent_s.agent_id, self.number_of_agents + 4] = time_until_arrival / 3600

        return agent_observations

    def create_observation_for_distances(self, env, current_agent_state: AgentState):
        # create route positions
        length_of_day_in_seconds = env.end_of_current_day - env.start_time
        normalized_datetime = (env.current_time - env.start_time) / length_of_day_in_seconds

        routes = self.get_resource_ids_in_routes_for_agents(env)

        number_of_agents = self.number_of_agents
        number_of_other_agent_features = self.number_of_agents + 4
        obs = np.zeros((len(env.resources), len(env.agent_states) * number_of_other_agent_features), dtype=np.float32)

        # todo maybe skip current agent
        for agent_state in env.agent_states:
            agent_state: AgentState

            if agent_state.agent_id == current_agent_state.agent_id:
                continue

            agent_offset = agent_state.agent_id * number_of_other_agent_features

            for resource in env.resources:
                routes_for_agent = routes[agent_state.agent_id]

                if resource.ident in routes_for_agent:
                    distance = self.shortest_path_lengths[resource.source_node][resource.target_node][
                        agent_state.position_node]
                    walking_time = distance / self.walking_speed
                    agent_arrival_time = normalized_datetime + (walking_time / length_of_day_in_seconds)
                    current_time = normalized_datetime

                    obs[resource.ident, agent_offset + agent_state.agent_id] = 1
                    obs[resource.ident, agent_offset + number_of_agents + 0] = distance / self.distance_normalization
                    obs[resource.ident, agent_offset + number_of_agents + 1] = walking_time / 3600
                    obs[resource.ident, agent_offset + number_of_agents + 2] = agent_arrival_time
                    obs[resource.ident, agent_offset + number_of_agents + 3] = current_time

        return obs
