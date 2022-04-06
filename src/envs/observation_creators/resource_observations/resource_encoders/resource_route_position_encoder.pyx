cdef class ResourceRoutePositionEncoder(ResourceEncoder):
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
        obs_array[offset] = 1 if resource.ident in common_information.resource_ids_in_routes_per_agent[
            agent_state.agent_id] else 0

        return offset + self.n_features