cdef class ResourceAgentIDOneHotEncoder(ResourceEncoder):
    def __init__(self, number_of_agents: int):
        self.n_features = number_of_agents

    cpdef int encode(self,
               np.float32_t[:] obs_array,
               offset: int,
               env,
               agent_state: AgentState,
               resource: Resource,
               common_information: CommonInformation,
               common_information_per_agent: CommonInformationPerAgent):
        obs_array[offset + agent_state.agent_id] = 1

        return offset + self.n_features