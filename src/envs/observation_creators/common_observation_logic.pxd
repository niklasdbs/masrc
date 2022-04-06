from omegaconf import DictConfig

from envs.agent_state import AgentState
from envs.observation_creators.observation_base cimport Observation


cdef class CommonObservationLogic(Observation):
    """
    This class capsules logic that is used in many observations
    """
    cdef public float distance_normalization
    cdef public float _min_y, _min_x, _max_y, _max_x


    cdef list get_resource_ids_in_routes_for_agents(self, env)