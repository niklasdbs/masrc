cdef class CommonInformation:
    cdef public int length_of_day_in_seconds
    cdef public float normalized_current_time
    cdef public list resource_ids_in_routes_per_agent