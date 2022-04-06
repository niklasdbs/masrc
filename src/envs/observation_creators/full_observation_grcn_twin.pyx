import networkx as nx
import numpy as np
cimport numpy as np
from gym import spaces

from omegaconf import DictConfig
from envs.agent_state import AgentState
from envs.enums import ParkingStatus
from envs.observation_creators.common_observation_logic cimport CommonObservationLogic
from envs.observation_creators.resource_observations.basic_resource_observation cimport BasicResourceObservation

np.import_array()


cdef class FullObservationGRCNTwin(CommonObservationLogic):
    cdef BasicResourceObservation resource_encoder
    cdef int agent_observation_feature_size
    cdef object _observation_space

    def __init__(self, config: DictConfig, graph: nx.Graph, number_of_resources: int, number_of_agents: int):
        super().__init__(config, graph, number_of_resources, number_of_agents)

        self.resource_encoder = BasicResourceObservation(config, graph, number_of_resources, number_of_agents)
        self.agent_observation_feature_size = 5 + self.number_of_agents
        self._observation_space = spaces.Dict({
                "resource_observations" : self.resource_encoder.observation_space,
                "other_agent_resource_observations": spaces.Box(-1, 2, shape=(number_of_agents-1, *self.resource_encoder.observation_space.shape)),
                "distance_to_action": spaces.Box(0, 1, shape=(1,)),
                "current_agent_id": spaces.Discrete(number_of_agents),
                "current_agent_observations": spaces.Box(0, 1, shape=(self.agent_observation_feature_size,)),
                "other_agent_observations": spaces.Box(0,1,shape=(number_of_agents-1, self.agent_observation_feature_size)),
                "distance_to_action_other_agents": spaces.Box(0, 1, shape=(number_of_agents-1,1)),
                "other_agent_ids": spaces.MultiDiscrete(np.repeat(number_of_agents, repeats=number_of_agents-1)),
        })


    @property
    def observation_space(self):
        return self._observation_space

    cpdef create_observation(self, env, current_agent_state: AgentState):
        resource_observations = self.resource_encoder.create_observation(env, current_agent_state)
        other_agent_resource_observations = self.create_other_agent_resource_observations(env, current_agent_state)
        current_agent_observations = self.create_agent_observations(env, current_agent_state)
        other_agent_observations = self.create_other_agent_observations(env, current_agent_state)
        distance_to_action = self.distance_agent_to_action(env, current_agent_state)
        distance_to_action_other_agents = self.distance_to_action_other_agents(env, current_agent_state)
        other_agent_ids = self.other_agent_ids(env, current_agent_state)

        return {
                "resource_observations":resource_observations,
                "other_agent_resource_observations": other_agent_resource_observations,
                "other_agent_observations": other_agent_observations,
                "distance_to_action": distance_to_action,
                "current_agent_id": current_agent_state.agent_id,
                "current_agent_observations" : current_agent_observations,
                "distance_to_action_other_agents" : distance_to_action_other_agents,
                "other_agent_ids": other_agent_ids
        }

    cdef distance_agent_to_action(self, env, agent_state: AgentState):
        distance_to_action = np.array([self.distance(edge[0],
                                                     edge[1],
                                                     agent_state.position_node_source,
                                                     agent_state.position_node,
                                                     agent_state.position_on_edge)
                                       for edge in env.edge_id_to_edge_mapping.values()], dtype=np.float32)
        distance_to_action = distance_to_action/self.distance_normalization
        return distance_to_action


    cdef distance_to_action_other_agents(self, env, current_agent_state: AgentState):
        distances = []
        for agent_state in env.agent_states:
            if current_agent_state.agent_id == agent_state.agent_id:
                continue

            distances.append(self.distance_agent_to_action(env, agent_state))

        return np.stack(distances)

    cdef other_agent_ids(self, env, current_agent_state: AgentState):
        ids = np.empty(shape=(self.number_of_agents-1, 1), dtype=np.int)
        i=0
        for agent_state in env.agent_states:
            if current_agent_state.agent_id == agent_state.agent_id:
                continue
            ids[i] = agent_state.agent_id
            i+=1
        return ids

    cdef create_agent_observations(self, env, agent_s: AgentState):
        graph: nx.Graph = env.graph

        agent_observations = np.zeros(self.agent_observation_feature_size, dtype=np.float32)
        agent_observations[agent_s.agent_id] = 1  # one hot encode the index of the vehicle

        current_position = agent_s.position_node
        x = (graph.nodes[current_position]["x"] - self._min_x) / (self._max_x - self._min_x)
        y = (graph.nodes[current_position]["y"] - self._min_y) / (self._max_y - self._min_y)

        agent_observations[self.number_of_agents + 0] = x
        agent_observations[self.number_of_agents + 1] = y

        if agent_s.current_action:
            destination = agent_s.current_route[-1]
            x = (graph.nodes[destination]["x"] - self._min_x) / (self._max_x - self._min_x)
            y = (graph.nodes[destination]["y"] - self._min_y) / (self._max_y - self._min_y)

            agent_observations[self.number_of_agents + 2] = x
            agent_observations[self.number_of_agents + 3] = y

            distance_to_target = self.distance(agent_s.current_route[-2],
                                                    agent_s.current_route[-1],
                                                     agent_s.position_node_source,
                                                     agent_s.position_node,
                                                     agent_s.position_on_edge)
            # time_until_arrival = self.shortest_path_lengths[agent_s.current_route[-2]][agent_s.current_route[-1]][
            #                          agent_s.position_node] / self.walking_speed

            time_until_arrival = distance_to_target / self.walking_speed

            agent_observations[self.number_of_agents + 4] = time_until_arrival / 3600

        return agent_observations

    cdef create_other_agent_observations(self, env, current_agent_state: AgentState):
        observations = []
        for agent_state in env.agent_states:
            if current_agent_state.agent_id == agent_state.agent_id:
                continue

            obs = self.create_agent_observations(env, agent_state)
            observations.append(obs)

        return np.stack(observations)


    cdef create_other_agent_resource_observations(self, env, current_agent_state: AgentState):
        observations = []
        for agent_state in env.agent_states:
            if current_agent_state.agent_id == agent_state.agent_id:
                continue

            obs = self.resource_encoder.create_observation(env, agent_state)
            observations.append(obs)

        return np.stack(observations)

