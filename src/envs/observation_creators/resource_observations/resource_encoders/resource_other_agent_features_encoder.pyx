cdef class ResourceOtherAgentFeatures(ResourceEncoder):
    def __init__(self, number_of_agents: int, encoders):
        self._encoders = encoders
        self.n_features = number_of_agents * sum((enc.n_features for enc in self._encoders))

    cpdef int encode(self,
               np.float32_t[:] obs_array,
               offset: int,
               env,
               agent_state: AgentState,
               resource: Resource,
               common_information: CommonInformation,
               common_information_per_agent: CommonInformationPerAgent):
        for encoder in self._encoders:
            offset = encoder.encode(obs_array,
                                    offset,
                                    env,
                                    agent_state,
                                    resource,
                                    common_information,
                                    common_information_per_agent)
        return offset