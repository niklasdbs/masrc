cdef class ResourceAgentPositionEncoding(ResourceEncoder):
    def __init__(self):
        self.n_features = 1

    cpdef int encode(self,
               np.float32_t[:] obs_array,
               offset: int,
               env,
               agent_state: AgentState,
               resource: Resource,
               common_information: CommonInformation,
               common_information_per_agent: CommonInformationPerAgent):
        if agent_state.position_node_source == resource.source_node \
                and agent_state.position_node == resource.target_node:
            obs_array[offset] = 1

        return offset + self.n_features