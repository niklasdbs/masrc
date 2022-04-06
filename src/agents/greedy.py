from collections import defaultdict
from itertools import chain
from typing import Any

import numpy as np

from agents.agent import Agent
from omegaconf import DictConfig

from envs.enums import ParkingStatus
from envs.potop_env import PotopEnv


class Greedy(Agent):
    """A baseline agent implementing the greedy algorithm."""

    def __init__(self, action_space: Any, observation_space: Any, graph, config: DictConfig):
        super().__init__(action_space, observation_space, graph, config)
        self.reserved_resources = defaultdict(list)
        self.current_plan_for_agents = defaultdict(list)
        self.assign_only_one_resource_to_greedy = config.assign_only_one_resource_to_greedy
        self.decision_epoch: int = config.decision_epoch
        self.walking_speed = config.speed_in_kmh / 3.6
        self.use_wait_action = config.use_wait_action
        if self.use_wait_action:
            if config.wait_action_time != 1:
                raise Exception("need to set wait_action_time to 1")

    def act(self, state, **kwargs) -> (int, None):
        # only one resource should be assigned to the agent
        env, shortest_path_lengths = state

        # find the current agent
        current_agent = env.agent_selection
        current_time = env.current_time

        if len(self.current_plan_for_agents[current_agent]) > 0:
            return env.resource_id_to_edge_id_mapping[self.current_plan_for_agents[current_agent].pop(0)], None

        if self.use_wait_action:
            # wait until next decision epoch, wait_action_time must be set to 1
            if self.decision_epoch and current_time % self.decision_epoch != 0:
                return PotopEnv.WAIT_ACTION, None


        self.reserved_resources[current_agent].clear()

        reserved_resources = set(chain(*self.reserved_resources.values()))

        agent_states_that_need_to_decide = {agent: env.agent_states[agent] for agent in env.agents if
                                            env.agent_states[agent].current_action is None}

        resource_id_assignments = []  # this will also contain resources that have been already assigned to other agents and therefore need to be skipped

        resources_in_violation = (
            (current_time - resource.arrival_time - resource.max_parking_duration_seconds, resource)
            for resource in env.resources
            if resource.status == ParkingStatus.IN_VIOLATION)

        sorted_resources_in_violation = sorted(resources_in_violation, key=lambda t: t[0])

        # best_resource = None
        p_best_resource = -np.inf

        for overstayed_time, resource in sorted_resources_in_violation:
            if overstayed_time < 0: #happens not very often, but the reson is that there are some resource events that span multiple days
                overstayed_time = 7200

            if resource.ident in reserved_resources:
                continue

            p_max = -np.inf
            best_agent = None
            for agent, agent_state_from_decision in agent_states_that_need_to_decide.items():
                p = self._calculate_probability(overstayed_time, resource, agent_state_from_decision.position_node,
                                                shortest_path_lengths)
                if p > p_max:
                    p_max = p
                    best_agent = agent

            if best_agent == current_agent:
                if p_max > p_best_resource:
                    p_best_resource = p_max
                    # best_resource = resource.ident
                resource_id_assignments.append(resource.ident)

        next_plan = []
        for resource_id in resource_id_assignments:
            self.reserved_resources[current_agent].append(resource_id)
            next_plan.append(resource_id)

            if self.assign_only_one_resource_to_greedy:
                break

        if len(next_plan) == 0:
            if self.use_wait_action:
                return PotopEnv.WAIT_ACTION, None
            else:
                return self.action_space.sample(), None

        self.current_plan_for_agents[current_agent] = next_plan

        return env.resource_id_to_edge_id_mapping[self.current_plan_for_agents[current_agent].pop(0)], None

    def _calculate_probability(self, overstayed_time, resource, agent_position, shortest_path_lengths):
        distance = shortest_path_lengths[resource.source_node][resource.target_node][agent_position]
        walking_time = distance / self.walking_speed

        return -(overstayed_time + walking_time)
