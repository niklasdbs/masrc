cdef class ResourceTargetPositionEncoder(ResourceEncoder):
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
        obs_array[offset] = 1 if agent_state.current_action == resource.ident else 0

        return offset + self.n_features