cdef class CommonInformationPerAgent:
    def __cinit__(self, distance, walking_time, normalized_agent_arrival_time):
        self.distance = distance
        self.walking_time = walking_time
        self.normalized_agent_arrival_time = normalized_agent_arrival_time
