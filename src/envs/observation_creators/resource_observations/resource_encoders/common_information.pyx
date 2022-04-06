cdef class CommonInformation:
    def __cinit__(self, length_of_day_in_seconds, normalized_current_time, resource_ids_in_routes_per_agent):
        self.length_of_day_in_seconds = length_of_day_in_seconds
        self.normalized_current_time = normalized_current_time
        self.resource_ids_in_routes_per_agent = resource_ids_in_routes_per_agent
