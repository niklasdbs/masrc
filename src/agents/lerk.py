from collections import defaultdict

import random

from agents.agent import Agent
from typing import Any

import numpy as np

from omegaconf import DictConfig
from envs.enums import ParkingStatus
from envs.utils import get_distance_matrix
from envs.potop_env import PotopEnv


class Lerk(Agent):

    def __init__(self, action_space: Any, observation_space: Any, graph, config: DictConfig) -> None:
        """
        Initializes a new instance.

        :param action_space: The agents action space.
        :param observation_space: The agents observation space.
        """
        super().__init__(action_space, observation_space, graph, config)
        self.time_prob_halved = config.time_prob_halved
        self.population_size = config.population_size
        self.lerk_iterations = config.lerk_iterations
        self.local_rate = config.local_rate
        self.threshold = config.threshold
        self.walking_speed = config.speed_in_kmh / 3.6

        self.env = None
        self.shortest_path_lengths = None
        self.current_agent_states = []

        self.decision_epoch: int = config.decision_epoch
        self.num_agents: int = config.number_of_agents

        self.reserved_spots = []
        self.current_plan_for_agents = defaultdict(list)

        self.distance_matrix = get_distance_matrix(graph)

    def act(self, state: dict) -> (dict, None):
        """
        Returns an action based on the state of the environment.

        :param state: The current states of the environment.

        :return: The action to be executed.
        """
        # todo cuurent agent state for each agent individual
        observations = [st['observation'] for st in state.values()]
        self.env, self.shortest_path_lengths, _ = observations[0]
        self.current_agent_states = [current_agent_state for _, _, current_agent_state in observations]

        available_agents = [True if st['needs_to_act'] == 1 else False for st in state.values()]
        resource_id_to_edge_id_mapping = self.env.resource_id_to_edge_id_mapping

        list_of_available_officers = []
        list_of_current_parking_violations = []
        real_agent_ids = []

        actions: dict = {}

        for index, available in enumerate(available_agents):
            if available:
                actions[index] = PotopEnv.WAIT_ACTION

        masked_agent_id = 0  # id of the agents inside lerk (starting from 0)
        # agent_id is the real ID of an agent
        for agent_id, available in enumerate(available_agents):
            if available:
                if len(self.current_plan_for_agents[agent_id]) > 0:
                    next_spot = self.current_plan_for_agents[agent_id].pop(0)
                    actions[agent_id] = resource_id_to_edge_id_mapping[next_spot]
                    self.reserved_spots.remove(next_spot)
                # use all agents ?
                else:
                    list_of_available_officers.append(masked_agent_id)
                    masked_agent_id += 1
                    real_agent_ids.append(agent_id)
        if masked_agent_id == 0:
            return actions, None

        # wait until next decision epoch, wait_action_time must be set to 1
        if self.decision_epoch and self.env.current_time % self.decision_epoch != 0:
            # todo is this for loop redundant?
            for agent_id in real_agent_ids:
                actions[agent_id] = PotopEnv.WAIT_ACTION
            return actions, None

        for resource in self.env.resources:
            if resource.status == ParkingStatus.IN_VIOLATION:
                if resource.ident not in self.reserved_spots:
                    list_of_current_parking_violations.append(resource.ident)
                    self.reserved_spots.append(resource.ident)

        lerk = self.find_best_lerk(list_of_available_officers, list_of_current_parking_violations)

        # loop through list till we find an officer, than the next parking_spot gets assigned to
        # the officer as their next action
        first_node = False
        masked_agent_id = 0
        for (is_officer, ID, _) in lerk:
            ID = int(ID)
            if is_officer == 1:
                first_node = True
                masked_agent_id = ID
                continue
            elif first_node & available_agents[real_agent_ids[masked_agent_id]]:
                first_node = False
                actions[real_agent_ids[masked_agent_id]] = resource_id_to_edge_id_mapping[ID]
                self.reserved_spots.remove(ID)
            else:
                self.current_plan_for_agents[real_agent_ids[masked_agent_id]].append(ID)

        return actions, None

    def find_best_lerk(self, list_of_available_officers, list_of_current_parking_violations):
        lerk_population = []  # list of tuples (lerk, rating)

        for i in range(self.population_size):
            lerk_population.append((self.create_lerk(list_of_available_officers, list_of_current_parking_violations), 0))

        for _ in range(self.lerk_iterations):
            for index, (lerk, rating) in enumerate(lerk_population):
                lerk_population[index] = (lerk, self._rate_lerk(lerk))
            lerk_population.sort(key=lambda tup: tup[1])

            new_lerk_population = []
            for index, (lerk, rating) in enumerate(lerk_population):
                if index < self.population_size * 0.2:
                    new_lerk_population.append((lerk, rating))
                elif index < self.population_size * 0.8:
                    mom_lerk = list(random.choice(lerk_population)[0].copy())
                    dad_lerk = list(random.choice(lerk_population)[0].copy())
                    child_lerk = []

                    mom_lerk.sort(key=lambda tup: (not tup[0], tup[1]))
                    dad_lerk.sort(key=lambda tup: (not tup[0], tup[1]))
                    leader = None
                    for ind, node in enumerate(mom_lerk):
                        if node[2] == -1:
                            leader = node
                        else:
                            if ind % 2 == 1:
                                if dad_lerk[ind][2] == -1:
                                    child_lerk.append(
                                        (dad_lerk[ind][0], dad_lerk[ind][1], random.uniform(0, 1)))
                                else:
                                    child_lerk.append(dad_lerk[ind])
                            else:
                                child_lerk.append(node)

                    child_lerk.sort(key=lambda tup: tup[2])
                    child_lerk.insert(0, leader)
                    if random.uniform(0, 1) > self.local_rate:
                        child_lerk = np.array(child_lerk)
                        child_lerk = self._optimize_lerk(child_lerk)
                    new_lerk_population.append((child_lerk, 0))

                else:
                    new_lerk_population.append((self.create_lerk(list_of_available_officers, list_of_current_parking_violations), 0))
                # print("population", new_lerk_population)
            lerk_population = new_lerk_population

        for index, (lerk, rating) in enumerate(lerk_population):
            lerk_population[index] = (lerk, self._rate_lerk(lerk))

        lerk_population.sort(key=lambda tup: tup[1])
        return lerk_population[0][0]

    @staticmethod
    def create_lerk(available_officers, violations):
        """
        creates Leader-based Random Keys Encoding Scheme (LERK)
        Args:
            available_officers: idle officers
            violations: ID of parking spots currently in violation

        Returns: List of Tuples
            ----------------------------
            (is_officer, ID of parking spot/ officer, leader(-1)/value between 0-1)
            ----------------------------

            is_officer is True for officers and False for parking spots

        """
        officers = available_officers.copy()
        lerk = []
        leader = random.choice(officers)
        officers.remove(leader)

        for officer in officers:
            lerk.append((1, officer, random.uniform(0, 1)))
        for violation_spot_id in violations:
            lerk.append((0, violation_spot_id, random.uniform(0, 1)))
        lerk.sort(key=lambda tup: tup[2])
        lerk.insert(0, (1, leader, -1))
        lerk = np.array(lerk)
        return lerk

    def _optimize_lerk(self, lerk):
        """
        sort the sub routs of each officer greedily
        Args:
            lerk:

        Returns:

        """
        optimized_lerk = []
        agent_id = -1
        associated_spots_id = []
        associated_values = []
        for is_officer, ID, value in lerk:
            ID = int(ID)
            if value == -1:  # -1 stands for leader
                agent_id = ID
                optimized_lerk.append((is_officer, ID, value))
                continue

            if is_officer == 1 or ([is_officer, ID, value] == lerk[-1]).all():
                if not is_officer == 1 and ([is_officer, ID, value] == lerk[-1]).all():
                    associated_spots_id.append(ID)
                    associated_values.append(value)
                # find spot with the highest catch probability
                closest_spot_id = None

                agent_location_edge_id = None
                while associated_spots_id:
                    distance = None
                    for spot_id in associated_spots_id:
                        violation_time = self.env.current_time - self.env.resources[spot_id].arrival_time - self.env.resources[
                            spot_id].max_parking_duration_seconds
                        if agent_location_edge_id is None:
                            # todo we now know the position, so we can change this
                            # first position of officer is unknown(by the agent),
                            # so we get the closest spot from status
                            dist = self.shortest_path_lengths[self.env.resources[spot_id].source_node][self.env.resources[spot_id].target_node][
                                       self.current_agent_states[agent_id].position_node] + violation_time
                        else:
                            # distance matrix = edges x spots
                            # distance from the edge the agent moved to and new spots
                            dist = self.distance_matrix[agent_location_edge_id][spot_id] + violation_time
                        if distance is None or dist < distance:
                            closest_spot_id = spot_id
                            distance = dist
                    optimized_lerk.append((0, closest_spot_id, associated_values.pop(0)))
                    associated_spots_id.remove(closest_spot_id)
                    # search for the edge the spot is laying on and save it as the new position
                    # of the officer
                    # todo get from env resource_id to edge id
                    for index, e in enumerate(self.distance_matrix[:, closest_spot_id]):
                        if e == 0.0:
                            agent_location_edge_id = index
                            break
                if is_officer == 1:
                    agent_id = ID
                    # add the next spot to the route
                    optimized_lerk.append((is_officer, ID, value))
                continue

            associated_spots_id.append(ID)
            associated_values.append(value)

        return np.array(optimized_lerk)

    def _rate_lerk(self, lerk):
        """

        Args:
            lerk:

        Returns:

        """
        officer_id = -1
        time_passed = 0
        agent_location_edge_id = None
        sum_prob = 0
        sum_catch = 0
        for is_officer, ID, value in lerk:
            ID = int(ID)
            if is_officer == 1:
                officer_id = ID
                time_passed = 0
                agent_location_edge_id = None
                continue
            if agent_location_edge_id is None:
                # first position of officer is unknown(by the agent),
                # so we get the closest spot from status
                distance = self.shortest_path_lengths[self.env.resources[ID].source_node][self.env.resources[ID].target_node][
                    self.current_agent_states[officer_id].position_node]
                time_passed += distance / self.walking_speed
                # time_passed += self.state[officer_id][ID][1] * 3600  # in seconds
            else:
                # distance matrix = edges x spots
                # distance from the edge the agent moved to and new spots
                time_passed += self.distance_matrix[agent_location_edge_id][ID] * 3600  # in seconds
            # search for the edge the spot is laying on and save it as the new position
            # of the officer
            for index, e in enumerate(self.distance_matrix[:, ID]):
                if e == 0.0:
                    agent_location_edge_id = index
                    break

            total_violation_time: int = self.env.current_time - (self.env.resources[ID].arrival_time + self.env.resources[ID].max_parking_duration_seconds)

            overstay_time = (time_passed + total_violation_time) / self.time_prob_halved

            # todo divided by zero error ?? when can overstaytime be -1 --> investigate
            overstay_probability = 1 - (overstay_time / (overstay_time + 1))
            # print('overstay probability:', overstay_probability)
            if overstay_probability >= self.threshold:
                sum_catch += 1
            else:
                sum_catch += 0
            sum_prob += overstay_probability

        if sum_catch == 0:
            return 0.0

        return sum_prob / sum_catch
