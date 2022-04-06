from envs.enums import ParkingStatus

cdef class ResourceStatusEncoder(ResourceEncoder):
    def __init__(self, do_not_use_fined_status: bint, optimistic_in_violation: bint):
        self.do_not_use_fined_status = do_not_use_fined_status
        self.optimistic_in_violation = optimistic_in_violation
        self.n_features = 4 + (1 if self.optimistic_in_violation else 0)

    cpdef int encode(self,
                     np.float32_t[:] obs_array,
                       offset: int,
                       env,
                       agent_state: AgentState,
                       resource: Resource,
                       common_information: CommonInformation,
                       common_information_per_agent: CommonInformationPerAgent):
        resource_status: int = resource.status
        if self.do_not_use_fined_status:
            if resource_status == ParkingStatus.FINED:
                resource_status = ParkingStatus.OCCUPIED

        obs_array[offset + resource_status] = 1

        if self.optimistic_in_violation:
            obs_array[offset + 4] = 1 if resource_status == ParkingStatus.OCCUPIED and env.current_time + common_information_per_agent.walking_time > (
                        resource.arrival_time + resource.max_parking_duration_seconds) else 0 # set flag if occupied and potential violation time > walking time

        return offset + self.n_features