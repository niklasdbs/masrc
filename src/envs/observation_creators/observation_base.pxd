from typing import Any, Union

import gym
import numpy as np

from omegaconf import DictConfig
from envs.agent_state import AgentState
# from envs.potop_env import PotopEnv
from envs.resource import Resource


cdef class Observation:
    cdef public float walking_speed
    cdef public int number_of_resources
    cdef public int number_of_agents
    cdef public dict shortest_path_lengths #dictionary from [resource_source][resource_target][current_position]
    cdef public dict length_of_edges

    # @property
    # cdef gym.Space observation_space(self)
    #@abstractmethod
    cdef create_observation(self, env, current_agent_state :  AgentState)

    #@lru_cache(maxsize=1000000)
    cpdef float distance(self, resource_source, resource_target, agent_position_source, agent_position_target, agent_position_on_edge=*)