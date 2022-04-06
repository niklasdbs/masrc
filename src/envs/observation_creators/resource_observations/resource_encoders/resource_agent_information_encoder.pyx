from envs.enums import ParkingStatus

cdef class ResourceAgentInformationEncoder(ResourceEncoder):
    def __init__(self, distance_normalization: float):
        self.n_features = 5
        self.distance_normalization = distance_normalization

    cpdef int encode(self,
               np.float32_t[:] obs_array,
               offset: int,
               env,
               agent_state: AgentState,
               resource: Resource,
               common_information: CommonInformation,
               common_information_per_agent: CommonInformationPerAgent):
        obs_array[offset + 0] = common_information_per_agent.walking_time / 3600
        obs_array[offset + 1] = common_information.normalized_current_time
        obs_array[offset + 2] = common_information_per_agent.normalized_agent_arrival_time
        obs_array[offset + 3] = min(2,
                                    (
                                            env.current_time + common_information_per_agent.walking_time - resource.arrival_time - resource.max_parking_duration_seconds)
                                    / resource.max_parking_duration_seconds) if resource.status == ParkingStatus.OCCUPIED \
                                                                                or resource.status == ParkingStatus.IN_VIOLATION else -1
        obs_array[offset + 4] = common_information_per_agent.distance / self.distance_normalization

        return offset + self.n_features