cdef class ResourcePositionEncoder:
    def __init__(self, min_x, max_x, min_y, max_y):
        self.n_features = 2
        self._min_y = min_y
        self._min_x = min_x
        self._max_y = max_y
        self._max_x = max_x

    cpdef int encode(self,
               np.float32_t[:] obs_array,
               offset: int,
               env,
               agent_state: AgentState,
               resource: Resource,
               common_information: CommonInformation,
               common_information_per_agent: CommonInformationPerAgent):
        x = (resource.x - self._min_x) / (self._max_x - self._min_x)
        y = (resource.y - self._min_y) / (self._max_y - self._min_y)

        obs_array[offset] = x
        obs_array[offset + 1] = y

        return offset + self.n_features