cimport numpy as np

from envs.agent_state import AgentState
from envs.observation_creators.resource_observations.resource_encoders.resource_encoder cimport ResourceEncoder, \
    CommonInformation, CommonInformationPerAgent
from envs.resource import Resource


cdef class ResourceAgentIDOneHotEncoder(ResourceEncoder):
    cdef public int n_features

    cpdef int encode(self,
               np.float32_t[:] obs_array,
               offset: int,
               env,
               agent_state: AgentState,
               resource: Resource,
               common_information: CommonInformation,
               common_information_per_agent: CommonInformationPerAgent)