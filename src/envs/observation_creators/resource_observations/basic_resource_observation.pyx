import networkx as nx
import numpy as np
cimport numpy as np
from gym import spaces
from omegaconf import DictConfig

np.import_array()

cdef class BasicResourceObservation(CommonObservationLogic):
    def __init__(self, config: DictConfig, graph: nx.Graph, number_of_resources: int, number_of_agents: int):
        super().__init__(config,
                         graph,
                         number_of_resources,
                         number_of_agents)  # todo currently we will create shortest paths multiple times

        self.add_other_agents_targets_to_resource = config.add_other_agents_targets_to_resource
        self.add_x_y_position_of_resource = config.add_x_y_position_of_resource

        self._encoders = [
                ResourceStatusEncoder(do_not_use_fined_status=config.do_not_use_fined_status,
                                      optimistic_in_violation=config.optimistic_in_violation),
                ResourceAgentInformationEncoder(self.distance_normalization)
        ]

        if self.add_x_y_position_of_resource:
            self._encoders.append(ResourcePositionEncoder(min_x=self._min_x,
                                                          max_x=self._max_x,
                                                          min_y=self._min_y,
                                                          max_y=self._max_y))

        self.number_of_features_per_resource = sum((enc.n_features for enc in self._encoders))

        if self.add_other_agents_targets_to_resource:
            self.add_route_positions = config.add_route_positions
            self.add_current_position_of_other_agents = config.add_current_position_of_other_agents

            other_agent_information_encoders = [
                    ResourceAgentIDOneHotEncoder(number_of_agents=number_of_agents),
                    ResourceAgentInformationEncoder(distance_normalization=self.distance_normalization),
                    ResourceTargetPositionEncoder()
            ]

            if self.add_current_position_of_other_agents:
                other_agent_information_encoders.append(ResourceAgentPositionEncoding())

            if self.add_route_positions:
                other_agent_information_encoders.append(ResourceRoutePositionEncoder())

            self._other_agent_information_encoder = ResourceOtherAgentFeatures(
                    number_of_agents=number_of_agents,
                    encoders=other_agent_information_encoders

            )

            self.number_of_features_per_resource += self._other_agent_information_encoder.n_features

        self._observation_space = spaces.Box(-1,
                                             2,
                                             shape=(self.number_of_resources, self.number_of_features_per_resource))

    @property
    def observation_space(self):
        return self._observation_space

    cpdef create_observation(self, env, current_agent_state: AgentState):
        cdef np.ndarray[np.float32_t, ndim=2] obs_array = np.zeros((self.number_of_resources, self.number_of_features_per_resource), dtype=np.float32)
        cdef np.float32_t[:,:] obs_view = obs_array
        cdef np.float32_t[:] obs_slice
        cdef int offset = 0
        cdef int index = 0
        cdef CommonInformation common_information = self.build_common_information(env)

        for resource in env.resources:
            resource: Resource

            common_information_current_agent = self.build_common_information_per_agent(env,
                                                                                       resource,
                                                                                       current_agent_state,
                                                                                       common_information)
            offset = 0
            index = resource.ident
            obs_slice = obs_view[index]
            for encoder in self._encoders:

                offset = encoder.encode(obs_slice,
                                        offset,
                                        env,
                                        current_agent_state,
                                        resource,
                                        common_information,
                                        common_information_current_agent)

            if self.add_other_agents_targets_to_resource:
                for agent_state in env.agent_states:
                    agent_state: AgentState

                    common_information_agent = self.build_common_information_per_agent(env,
                                                                                       resource,
                                                                                       agent_state,
                                                                                       common_information)

                    offset = self._other_agent_information_encoder.encode(obs_slice,
                                                                          offset,
                                                                          env,
                                                                          agent_state,
                                                                          resource,
                                                                          common_information,
                                                                          common_information_agent)

        return obs_array

    cdef CommonInformation build_common_information(self, env):
        cdef float length_of_day_in_seconds = env.end_of_current_day - env.start_time
        cdef float normalized_current_time = (env.current_time - env.start_time) / length_of_day_in_seconds

        resource_ids_in_routes_per_agent = self.get_resource_ids_in_routes_for_agents(env) \
            if self.add_other_agents_targets_to_resource and self.add_route_positions else None

        return CommonInformation(length_of_day_in_seconds, normalized_current_time, resource_ids_in_routes_per_agent)

    cdef CommonInformationPerAgent build_common_information_per_agent(self,
                                           env,
                                           resource: Resource,
                                           agent_state: AgentState,
                                           common_information: CommonInformation):
        cdef float distance = self.distance(resource_source=resource.source_node,
                                 resource_target=resource.target_node,
                                 agent_position_source=agent_state.position_node_source,
                                 agent_position_target=agent_state.position_node,
                                 agent_position_on_edge=agent_state.position_on_edge)
        cdef float walking_time = distance / self.walking_speed
        cdef float normalized_agent_arrival_time = common_information.normalized_current_time + (
                walking_time / common_information.length_of_day_in_seconds)

        return CommonInformationPerAgent(distance, walking_time, normalized_agent_arrival_time)
