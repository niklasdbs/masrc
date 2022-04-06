from envs.agent_state import AgentState
from envs.observation_creators.common_observation_logic cimport CommonObservationLogic
from envs.observation_creators.resource_observations.resource_encoders.resource_encoder cimport ResourceEncoder
from envs.observation_creators.resource_observations.resource_encoders.common_information cimport CommonInformation
from envs.observation_creators.resource_observations.resource_encoders.common_information_per_agent cimport CommonInformationPerAgent
from envs.observation_creators.resource_observations.resource_encoders.resource_agent_position_encoder cimport \
    ResourceAgentPositionEncoding
from envs.observation_creators.resource_observations.resource_encoders.resource_agent_id_encoder cimport \
    ResourceAgentIDOneHotEncoder
from envs.observation_creators.resource_observations.resource_encoders.resource_agent_information_encoder cimport \
    ResourceAgentInformationEncoder


from envs.observation_creators.resource_observations.resource_encoders.resource_other_agent_features_encoder cimport \
    ResourceOtherAgentFeatures
from envs.observation_creators.resource_observations.resource_encoders.resource_position_encoder cimport \
    ResourcePositionEncoder
from envs.observation_creators.resource_observations.resource_encoders.resource_route_position_encoder cimport \
    ResourceRoutePositionEncoder
from envs.observation_creators.resource_observations.resource_encoders.resource_status_encoder cimport \
    ResourceStatusEncoder
from envs.observation_creators.resource_observations.resource_encoders.resource_target_position_encoder cimport \
    ResourceTargetPositionEncoder
from envs.resource import Resource

cdef class BasicResourceObservation(CommonObservationLogic):
    cdef bint add_other_agents_targets_to_resource
    cdef bint add_x_y_position_of_resource
    cdef list _encoders
    cdef int number_of_features_per_resource
    cdef bint add_route_positions
    cdef bint add_current_position_of_other_agents
    cdef ResourceOtherAgentFeatures _other_agent_information_encoder
    cdef object _observation_space


    cpdef create_observation(self, env, object current_agent_state: AgentState)

    cdef CommonInformation build_common_information(self, env)

    cdef CommonInformationPerAgent build_common_information_per_agent(self,
                                           object env,
                                           object resource: Resource,
                                           object agent_state: AgentState,
                                           CommonInformation common_information: CommonInformation)
