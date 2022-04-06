"""
Module contains functionality for an ACO agent.
"""
import pickle
from time import time

import numpy as np

from agents.agent import Agent
from envs.utils import IN_VIOLATION_ENCODING

DISTANCE_MATRIX = "../data/distance_matrix.pickle"


# pylint:disable=R0902, R0914
class ACO(Agent):
    """
    Agent class implementing the ant colony optimization method.

    See also:
        Travelling Officer Problem: Managing Car Parking Violations Efficiently Using Sensor Data.
    """

    def __init__(self, action_space, observation_space, params):
        super().__init__(action_space, observation_space)
        self.speed = params["speed"]
        # self.env = env
        self.max_time = np.inf
        self.ants = params["ants"]
        self.computation_time = params["computation_time"]
        self.prob_alpha = params["prob_alpha"]
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.default_pheromon = 0.01
        self.evaporation_rate = params["evaporation_rate"]
        self.mapping = params["spot_to_edge_mapping"]

        try:
            with open(DISTANCE_MATRIX, "rb") as file:
                self.distances = pickle.load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError from error

    def act(self, state):
        """
        Returns an action based on the state of the environment.

        :param state: The current state of the environment.
        :return: The action to be executed.
        """
        _, position, device_ordering, device_states, _, resource_edges = state
        violations = [d for d in device_ordering if device_states[d] == IN_VIOLATION_ENCODING]

        if len(violations) == 0:
            return [self.action_space.n - 1]

        best_solution = None
        best_score = -np.inf
        phero = {}
        self.default_pheromon = 1 / len(violations)

        start = time()
        # while start + self.computation_time > time():
        results = []
        for _ in range(self.ants):
            path, scores = self._run_ant(state, violations, phero)
            score = sum(scores)
            if score >= best_score:
                best_solution = path
                best_score = score
            results.append((path, scores))

            # update pheromones
            # for path, scores in results:
            norm = sum(
                [phero.get(first_distance, {}).get(second_distance, self.default_pheromon) for
                 first_distance in (position, *violations) for second_distance in
                 (position, *violations)])
            for i, _ in enumerate(path):
                first_distance = position if i == 0 else path[i - 1]
                second_distance = path[i]
                old = phero.get(first_distance, {}).get(second_distance, self.default_pheromon)
                phero.setdefault(first_distance, {})[second_distance] = \
                    (1 - self.evaporation_rate) * old + scores[i] / norm

            if time() - start > self.computation_time:
                break

        if best_solution is None:
            return [-1]
        # return [self.env.edge_ordering.index(resource_edges[best_solution[0]])]
        return self.mapping[str(resource_edges[best_solution[0]])]

    def _run_ant(self, state, violations, pheromones):
        start, position, _, _, violation_times, resource_edges = state

        scores = []
        path = []
        mask = np.ones(len(violations), dtype=bool)
        pos = position
        current_time = start

        while sum(mask) > 0:
            travel_times = np.array(
                [self.distances[resource_edges[device][0]][pos] / self.speed for i, device in
                 enumerate(violations)])
            time_in_violation = np.array([current_time - violation_times[d] for d in violations])
            assert sum(time_in_violation < 0) == 0

            tag_prob = np.exp(- (travel_times + time_in_violation) / self.prob_alpha)

            phero = np.array(
                [pheromones.get(pos, {}).get(d, self.default_pheromon) for d in violations])
            prob = np.power(phero, self.alpha) + np.power(tag_prob, self.beta)
            prob *= mask
            prob = prob / prob.sum()

            action = np.random.choice(len(violations), p=prob)
            mask[action] = 0
            path.append(violations[action])
            scores.append(tag_prob[action])

            resource_edge = resource_edges[violations[action]]
            travel_time = 1  # should be path_length / speed
            current_time += travel_times[action] + travel_time
            pos = resource_edge[1]

            if current_time - start > self.max_time:
                break

        return path, scores
