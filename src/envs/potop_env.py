"""
Module contains functions for the simulation environment.
"""
import datetime
import heapq
import logging
from collections import defaultdict
from heapq import heappush, heappop
from typing import Union, Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
from gym import spaces
from omegaconf import DictConfig
from pettingzoo import AECEnv

from datasets.datasets import DataSplit
from envs.agent_state import AgentState
from envs.enums import ParkingStatus, EventType
from envs.event import AgentEvent, ResourceEvent
from envs.observation_creators import observation
from envs.observation_creators.observation_base import Observation
from envs.renderer import Renderer
from envs.resource import Resource
from envs.top_agent_selector import top_agent_selector
from envs.utils import get_num_spots, get_edges_w_spots, get_avg_walking_time

EVENT_TYPE_CONVERSION = {"Arrival": EventType.ARRIVAL,
                         "Departure": EventType.DEPARTURE,
                         "Violation": EventType.VIOLATION}


class PotopEnv(AECEnv):
    """
    OpenAI Gym environment which models the TOP (Travelling Officer Problem)
    based on the Melbourne parking sensors dataset.
    """

    metadata = {'render.modes': ['human']}

    WAIT_ACTION: int = -1  # only use the wait action after the env has been created #todo this may be dangerous, when having multiple instances with different graphs

    def __init__(self, event_log: pd.DataFrame, graph: nx.Graph, shortest_path_lookup: dict, data_split: DataSplit,
                 config: DictConfig) \
            -> None:
        super().__init__()
        logging.info('Creating simulation environment.')

        self._calculate_advanced_stats = data_split in [getattr(DataSplit, d) for d in
                                                        config.calculate_advanced_statistics]
        if self._calculate_advanced_stats:
            self._metric_aggregation_map: {str: Any} = {
                    "time_until_fine": np.mean,
                    "violation_durations": np.mean,
                    "violation_durations_non_fined_resources": np.mean
            }
            self._advanced_metrics = defaultdict(list)  # contains the advanced metrics aggregated per day
            self._advanced_metrics_per_soft_reset = defaultdict(list)  # contains advanced metrics for the current day

        self.final_advanced_metrics = {}

        self._create_observations_between_steps = config.create_observation_between_steps
        self.move_other_agents_between_edges = config.get("move_other_agents_between_edges", True)

        self.random_start_point = config.random_start_point
        self._num_agents = config.number_of_agents
        self._gamma = config.gamma
        self.speed_in_ms = config.speed_in_kmh / 3.6
        self.shared_reward = config.get("shared_reward", False)
        self.possible_agents = [i for i in range(self._num_agents)]
        self.agents = self.possible_agents[:]

        # event heap for every day (if some days of the year are not sue for the simualtion the lists for these days will be empty).
        # Do not modify this heap directly only use copys (i.e. current_heap)!
        self._event_heap: [[ResourceEvent]] = [[] for _ in range(365)]
        # the raw event log that contains events for all days
        self._event_log: pd.DataFrame = event_log
        self.graph: nx.Graph = graph
        self.shortest_path_lookup: dict[tuple[Any, Any], dict] = shortest_path_lookup
        self.resources: [Resource] = []
        self.agent_states: [AgentState] = []

        # actions correspond to edge ids (int). We only have edge ids for edges with resources on it.

        # maps marker_id of spots to the id of the resource (identified by an int)
        self.spots_mapping: dict[Any, int] = {}

        # The possible actions are all edges with parking spots. It has the format int : (u,v)
        self.edge_id_to_edge_mapping: dict[int, tuple[Any, Any]] = get_edges_w_spots(self.graph)

        # maps all the edges with resources to the id of the edge. Format (u,v) : int
        self.edge_to_edge_id_mapping: dict[tuple[Any, Any], int] = {edge: edge_id for edge_id, edge in
                                                                    self.edge_id_to_edge_mapping.items()}

        # maps an edge to (u,v) to a list of all resource ids located on that edge
        self.edge_to_resource_id_mapping: dict[tuple[Any, Any], [int]] = {}

        # maps the resource id (int) to the edge id (int)
        self.resource_id_to_edge_id_mapping: dict[int, int] = {}

        self._init_resources(graph)

        self.data_split: DataSplit = data_split
        self.year: int = config.year
        self.current_day: int = 0
        self.start_hour: int = config.start_hour
        self.end_hour: int = config.end_hour

        # initialise the environment at specific timestamp
        self.current_time: int = 0

        self.starting_positions = {agent_id : self._get_initial_position(self.graph) for agent_id in range(self._num_agents)}
        self._initialize_agent_states()

        # whether the order of days should be shuffled (validation and test can not be shuffled)
        self.shuffle_days = config.shuffle_days
        if self.data_split == DataSplit.TEST or self.data_split == DataSplit.VALIDATION:
            self.shuffle_days = False

        # this setting will remove the day when doing the reset and will only consider the day again when all other days have been simulated
        self._remove_days = (self.data_split == DataSplit.TRAINING and not self.shuffle_days) \
                            or (self.data_split == DataSplit.TEST or self.data_split == DataSplit.VALIDATION)

        # contains the days that are yet to be simulated
        self._days_to_simulate: [int] = []

        self._init_days_to_simulate()

        num_spots: int = get_num_spots(self.graph)

        self._observation_creator: Observation = getattr(observation, config.observation)(config, self.graph, num_spots,
                                                                                          self._num_agents)

        self._state_observation_creator: Observation = getattr(observation, config.state_observation)(config, self.graph, num_spots,self._num_agents)

        self._event_log_to_heap(event_log)
        self.current_event_heap: [Union[AgentEvent, ResourceEvent]] = None
        self.use_wait_action: bool = config.use_wait_action
        self.wait_action_time: int = config.wait_action_time
        PotopEnv.WAIT_ACTION = len(self.edge_id_to_edge_mapping)  # wait action is the last action

        # Attributes required by OpenAI Gym
        self.action_spaces = {
                agent: spaces.Discrete(len(self.edge_id_to_edge_mapping) + (1 if self.use_wait_action else 0)) for agent
                in
                self.possible_agents}

        self.observation_spaces = {agent: self._observation_creator.observation_space for agent in
                                   self.possible_agents}

        self.state_observation_space = self._state_observation_creator.observation_space

        self.renderer = Renderer(config)
        self._render_steps_of_action = (data_split == DataSplit.TEST or data_split == DataSplit.VALIDATION) and config.render and config.render_steps_of_actions

        self.fined_resources: int = 0  # number of resource find until current time from beginning of day
        self.cumulative_resources_in_violation: int = 0  # number of resource in violation until current time from beginning of day


        self._length_of_edges = {}#cache those for better performance
        for source, target in graph.edges():
            if source not in self._length_of_edges:
                self._length_of_edges[source] = {}
            self._length_of_edges[source][target] = graph[source][target]["length"] #assumes that resources are located at the end of an edge


        logging.debug('Action space: %s', self.action_space(0))
        logging.debug('Observation space: %s', self.observation_space(0))
        logging.info('Number of parking spots: %i', num_spots)
        # these are time consuming, so do not calculate every time
        # logging.info('Number of events: %i', len(list(itertools.chain(*self._event_heap))))
        # logging.info('Number of violations: %i',
        #              sum((1 for e in itertools.chain(*self._event_heap)
        #                   if e.event_type == EventType.VIOLATION
        #                   and config.start_hour <= datetime.datetime.fromtimestamp(e.time).hour < config.end_hour)))
        # logging.info('Average no of violations per day: %i',
        #              sum((1 for e in itertools.chain(*self._event_heap)
        #                   if e.event_type == EventType.VIOLATION
        #                   and config.start_hour <= datetime.datetime.fromtimestamp( e.time).hour < config.end_hour))
        #              / len(self._days_to_simulate))
        logging.debug('Average walking time per edge: %f',
                      get_avg_walking_time(self.graph))

        logging.info('Simulation environment created.')

    def observation_space(self, agent):
        """
        Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the observation_spaces dict
        """
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """
        Takes in agent and returns the action space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the action_spaces dict
        """
        return self.action_spaces[agent]


    def _init_resources(self, graph):
        self.spots_mapping = {}
        self.resources: [Resource] = []
        self.edge_to_resource_id_mapping = defaultdict(list)
        self.resource_id_to_edge_id_mapping: dict[int, int] = {}

        i = 0
        for start_node, end_node, data in graph.edges(data=True):
            if "spots" in data:
                for spot in data["spots"]:
                    resource = Resource(ident=i,
                                        status=ParkingStatus.FREE,
                                        source_node=start_node,
                                        target_node=end_node,
                                        position_on_edge=0,
                                        arrival_time=0,
                                        max_parking_duration_seconds=0,
                                        time_last_violation=0,
                                        x=spot["x"],
                                        y=spot["y"]
                                        )

                    self.resources.append(resource)

                    self.spots_mapping[spot["id"]] = i
                    self.edge_to_resource_id_mapping[(start_node, end_node)].append(i)
                    self.resource_id_to_edge_id_mapping[i] = self.edge_to_edge_id_mapping[(start_node, end_node)]
                    i += 1

    def _init_days_to_simulate(self):
        self._days_to_simulate = []
        for day in range(1, 366):
            if self._is_in_data_split(day, self.data_split):
                self._days_to_simulate.append(day)

    def _get_initial_position(self, graph: nx.Graph):
        """Returns the initial position of an agent."""
        if self.random_start_point:
            start = np.random.choice(list(graph.nodes))
        else:
            start = min(list(graph.nodes))
        logging.debug(f"Starting position: {start}", )
        return start

    @staticmethod
    def _is_in_data_split(day_of_year, data_split: DataSplit):
        """Returns true, if the selected date is included in the selected dataplit."""
        if data_split is DataSplit.TRAINING:
            return day_of_year % 13 > 1
        elif data_split is DataSplit.VALIDATION:
            return day_of_year % 13 == 1
        elif data_split is DataSplit.TEST:
            return day_of_year % 13 == 0
        raise AssertionError

    def _get_relevant_events(self, event_log):
        spot_ids = set(self.spots_mapping.keys())
        days_to_simulate = set(self._days_to_simulate)
        event_log["DayOfYear"] = event_log["Time"].dt.dayofyear
        event_log = event_log[event_log["DayOfYear"].isin(days_to_simulate) & event_log["StreetMarker"].isin(spot_ids)]
        return event_log.copy()

    def _event_log_to_heap(self, event_log):

        relevant_events = self._get_relevant_events(event_log)

        relevant_events["MappedType"] = relevant_events["Type"].map(EVENT_TYPE_CONVERSION)
        relevant_events["TimeStamp"] = relevant_events["Time"].view(np.int64) // 10 ** 9
        relevant_events["SpotID"] = relevant_events["StreetMarker"].map(self.spots_mapping)
        # assert relevant_events["TimeStamp"].eq(relevant_events["Time"].map(lambda time: int(time.timestamp()))).all()

        relevant_events["ResourceEvent"] = pd.Series((ResourceEvent(time_stamp, max_seconds, spot_id, event_type)
                                                      for time_stamp, max_seconds, spot_id, event_type
                                                      in zip(relevant_events["TimeStamp"],
                                                             relevant_events["MaxSeconds"],
                                                             relevant_events["SpotID"],
                                                             relevant_events["MappedType"])),
                                                     index=relevant_events.index)
        # the above is faster than apply
        resource_events = relevant_events.groupby("DayOfYear")

        for day_of_year, group in resource_events["ResourceEvent"]:
            self._event_heap[day_of_year - 1] = list(group)
            heapq.heapify(self._event_heap[day_of_year - 1])

    def state(self):
        return self._state_observation_creator.create_observation(self, None)

    def observe(self, agent):
        # return the observation at the last step for a specific agent
        return self.observations[agent]

    def last(self, agent=None, observe=True):
        if agent is None:
            agent = self.agent_selection
            assert self.agent_states[
                       agent].current_action is None, "currently between the actions of an agent, a valid observation may not be created"

        observation = self.observe(agent) if observe else None
        return observation, self.rewards[agent], self.discounted_rewards[agent], self.dones[agent], self.infos[agent]

    def step(self, action) -> bool:
        """
        @param action: action for that should act
        @return: return whether the call to step advanced the environment
        """
        # the output of step should not be used, because it may advances multiple agents at the same time, always use the last() from the iterator
        agent = self.agent_selection

        if self.dones[agent]:
            self.agents.remove(agent)

            _dones_order = [agent for agent in self.agents if self.dones[agent]]
            if _dones_order:
                self.agent_selection = _dones_order[0]

            return False

        assert self.agent_states[agent].current_action is None

        # update the agent_state class
        self.agent_states[agent].current_action = action
        self.agent_states[agent].time_last_action = self.current_time

        new_agent_events = self._create_events_from_agent_action(agent, action)
        [heappush(self.current_event_heap, ae) for ae in new_agent_events]

        need_to_advance_env = self._agent_selector.is_last()  # now every agent has selected an action => need to execute those actions

        if need_to_advance_env:
            agents_that_completed_the_action = []
            while True:
                need_to_simulate_more_events, agent_that_completed_action = self._simulate_next_event()
                if agent_that_completed_action == -1:
                    # this case means that all agents are finished because the end of the day is reached
                    agents_that_completed_the_action = self.agents
                    # set all agents done
                    for agent_c in self.agents:
                        self.dones[agent_c] = True
                elif agent_that_completed_action is not None:
                    agents_that_completed_the_action.append(agent_that_completed_action)

                if not need_to_simulate_more_events:
                    break

            assert len(agents_that_completed_the_action) > 0

            if self._create_observations_between_steps:
                for agent in self.agents:
                    if agent in agents_that_completed_the_action:
                        continue

                    # create observation for other agents between the steps
                    self.observations[agent] = self._observation_creator \
                        .create_observation(env=self, current_agent_state=self.agent_states[agent])

                    # also update rewards and infos
                    self.rewards[agent] = self.agent_states[agent].reward_since_last_action
                    self.discounted_rewards[agent] = self.agent_states[agent].discounted_reward_since_last_action
                    self.infos[agent] = {"dt": self.current_time - self.agent_states[agent].time_last_action}

            for agent_c in agents_that_completed_the_action:
                # update agent selector with agents that need to select a new agent in the next step
                self._agent_selector.set_agent_needs_to_select_new_action(agent_c)

                agent_state = self.agent_states[agent_c]
                self.rewards[agent_c] = agent_state.reward_since_last_action
                self.discounted_rewards[agent_c] = agent_state.discounted_reward_since_last_action
                self.infos[agent_c] = {"dt": self.current_time - agent_state.time_last_action}

                # reset agent state
                agent_state.reset_after_current_action_completed()

                # create observation for agent when the agent state has been reset
                self.observations[agent_c] = self._observation_creator.create_observation(env=self,
                                                                                          current_agent_state=agent_state)

        else:
            pass

        # select next agent
        self.agent_selection = self._agent_selector.next()
        return need_to_advance_env

    def _create_events_from_agent_action(self, agent: Any, action: Any) -> [AgentEvent]:
        agent_state = self.agent_states[agent]

        current_position = agent_state.position_node

        if self.use_wait_action and action == PotopEnv.WAIT_ACTION:
            return [AgentEvent(self.current_time + self.wait_action_time,
                               agent_state.position_node,
                               agent_state.position_node_source,
                               agent_state.position_on_edge, agent,
                               completes_action=True)]

        target_edge = self.edge_id_to_edge_mapping[action]

        route = self.shortest_path_lookup[target_edge][current_position] + [
                target_edge[1]]  # use the plus operator to create a new list a new list is created
        assert route[0] == current_position

        agent_state.current_route = route  # set the route of an agent here

        previous_node = current_position
        current_time: int = self.current_time
        events = []

        agent_event = None
        for node in route[1:]:
            assert previous_node != node

            edge_length = self._length_of_edges[previous_node][node]
            travel_time = int(edge_length / self.speed_in_ms)

            current_time += travel_time

            agent_event = AgentEvent(current_time,
                                     node,
                                     previous_node,
                                     travel_time,
                                     agent)

            events.append(agent_event)
            previous_node = node

        # last event needs to indicate that an agent has to act again
        agent_event.completes_action = True

        return events

    def _simulate_next_event(self, before_agents_start=False):
        # resources are passed at the end of an edge

        event = heappop(self.current_event_heap)

        if event.time > self.end_of_current_day:
            return False, -1  # -1 indicates that all agents should finish now

        next_event_after_event_to_simulate = self.current_event_heap[0] if len(self.current_event_heap) > 0 else None

        # need to simulate until an agent event, if there are multiple agent events that happen at the same time we need to simulate until we are at the last agent event
        # we can ensure that agent events come before resource events (or vice versa)
        need_to_simulate_more_events = isinstance(event, ResourceEvent) or (not event.completes_action or (
                isinstance(next_event_after_event_to_simulate, AgentEvent) and (
                next_event_after_event_to_simulate.time - event.time) == 0))

        if before_agents_start:
            assert isinstance(event, ResourceEvent), "no agent events, before the agents start working"
            # events before the officer starts working!

            # if we use < the agent will start at the time of the event before the start hour
            need_to_simulate_more_events = next_event_after_event_to_simulate and next_event_after_event_to_simulate.time < self.start_time

        time_diff = event.time - self.current_time
        assert time_diff >= 0, "encountered negative time!"

        # update current time
        self.current_time = event.time

        agent_that_completed_action = None

        if isinstance(event, AgentEvent):
            agent_state = self.agent_states[event.agent]
            previous_position = agent_state.position_node
            previous_position_on_edge = agent_state.position_on_edge

            time_diff_since_beginning_of_action = event.time - agent_state.time_last_action
            assert time_diff_since_beginning_of_action >= 0, "encountered negative time!"

            # update position of agent
            agent_state.position_node = event.position_node
            agent_state.position_node_source = event.position_node_source
            agent_state.position_on_edge = event.position_on_edge

            edge = (agent_state.position_node_source, agent_state.position_node)

            if self.move_other_agents_between_edges:
                # move agents to the position between the edges
                # if they would move past an instersection this would be an agent event and therefore it can not happen here
                for other_agent_state in self.agent_states:
                    if other_agent_state == agent_state:
                        continue

                    other_agent_state.position_on_edge += int(time_diff*self.speed_in_ms)

            if previous_position != event.position_node:  # wait action does not change the position, so do not collect resources again
                # collect resources
                # resources are positioned at the end of an edge
                number_of_fined_resources = 0
                for r_id in self.edge_to_resource_id_mapping[edge]:
                    resource: Resource = self.resources[r_id]
                    if resource.status == ParkingStatus.IN_VIOLATION:
                        number_of_fined_resources += 1
                        resource.status = ParkingStatus.FINED

                        if self._calculate_advanced_stats:
                            time_until_fine = self.current_time - resource.time_last_violation
                            self._advanced_metrics_per_soft_reset["time_until_fine"].append(time_until_fine)

                self.fined_resources += number_of_fined_resources  # for stats

                if not self.shared_reward:
                    reward = number_of_fined_resources
                    discounted_reward = (self._gamma ** time_diff_since_beginning_of_action) * reward

                    agent_state.reward_since_last_action += reward
                    agent_state.discounted_reward_since_last_action += discounted_reward

                    # reset of agent_state things related to action will be done at a different place (after the rewards have been reed)
                else:
                    for agent_state_other in self.agent_states:
                        time_diff_since_beginning_of_action_other = event.time - agent_state_other.time_last_action

                        agent_state_other.reward_since_last_action += number_of_fined_resources
                        agent_state_other.discounted_reward_since_last_action += \
                            (self._gamma ** time_diff_since_beginning_of_action_other) * number_of_fined_resources

            if event.completes_action:
                agent_that_completed_action = event.agent

            if self._render_steps_of_action:
                self.render("internal_step", show=False)

        elif isinstance(event, ResourceEvent):
            resource_to_update: Resource = self.resources[event.resource_id]
            if event.event_type == EventType.VIOLATION and (resource_to_update.status == ParkingStatus.IN_VIOLATION
                                                            or resource_to_update.status == ParkingStatus.FINED
                                                            or resource_to_update.status == ParkingStatus.FREE):
                # these events are duplicate or due to some events going over mutliple days => we can ignore them
                pass
            else:
                if self._calculate_advanced_stats and not before_agents_start:
                    if resource_to_update.status == ParkingStatus.IN_VIOLATION or resource_to_update.status == ParkingStatus.FINED:
                        assert event.event_type != EventType.VIOLATION
                        time_in_violation = event.time - resource_to_update.time_last_violation
                        self._advanced_metrics_per_soft_reset["violation_durations"].append(time_in_violation)

                        if resource_to_update.status != ParkingStatus.FINED:
                            self._advanced_metrics_per_soft_reset["violation_durations_non_fined_resources"].append(
                                    time_in_violation)

                if event.event_type == EventType.ARRIVAL:
                    resource_to_update.status = ParkingStatus.OCCUPIED
                    resource_to_update.arrival_time = event.time
                    resource_to_update.max_parking_duration_seconds = event.max_seconds
                elif event.event_type == EventType.DEPARTURE:
                    resource_to_update.arrival_time = 0
                    resource_to_update.status = ParkingStatus.FREE
                    resource_to_update.max_parking_duration_seconds = event.max_seconds
                elif event.event_type == EventType.VIOLATION:
                    resource_to_update.status = ParkingStatus.IN_VIOLATION
                    resource_to_update.time_last_violation = event.time
                    if not before_agents_start:
                        self.cumulative_resources_in_violation += 1

        if before_agents_start and not need_to_simulate_more_events:
            # set env time to start time
            self.current_time = self.start_time

        return need_to_simulate_more_events, agent_that_completed_action

    def _next_day(self, with_remove=True) -> bool:
        """
        After the working time for the day is exceeded, set the time to 07:00
        on the next day and restart the agent positions.
        """

        if self._calculate_advanced_stats:
            if self.cumulative_resources_in_violation > 0:  # reset before simulation
                violation_catched_quota = self.fined_resources / self.cumulative_resources_in_violation
                self._advanced_metrics["violation_catched_quota"].append(violation_catched_quota)
                self._advanced_metrics["fined_resources"].append(self.fined_resources)
                self._advanced_metrics[
                    "cumulative_resources_in_violation"].append(self.cumulative_resources_in_violation)
                for key, items in self._advanced_metrics_per_soft_reset.items():
                    self._advanced_metrics[key] = self._metric_aggregation_map[key](items)
                self._advanced_metrics_per_soft_reset = defaultdict(list)

        # all days have been simulated => indicate that a reset should be done
        if len(self._days_to_simulate) == 0:
            if self._calculate_advanced_stats:
                for key, items in self._advanced_metrics.items():
                    self.final_advanced_metrics[key] = self._metric_aggregation_map.get(key, np.mean)(items)

            return False

        self.fined_resources = 0
        self.cumulative_resources_in_violation = 0

        # get date of the next day
        next_day: int = np.random.choice(self._days_to_simulate) if self.shuffle_days else self._days_to_simulate[0]
        if with_remove:
            self._days_to_simulate.remove(next_day)
        self.current_day = next_day

        # Set new time
        self.start_time = int(
                datetime.datetime.strptime(("%03d %d %02d:00" % (self.current_day, self.year, self.start_hour)),
                                           "%j %Y %H:%M").timestamp())
        self.current_time = int(
                datetime.datetime.strptime(("%03d %d %02d:00" % (self.current_day, self.year, 0)),
                                           "%j %Y %H:%M").timestamp())
        self.end_of_current_day = int(
                datetime.datetime.strptime(("%03d %d %02d:00" % (self.current_day, self.year, self.end_hour)),
                                           "%j %Y %H:%M").timestamp())

        self.current_event_heap = self._event_heap[self.current_day - 1].copy()
        assert len(self.current_event_heap) > 0
        return True

    def _handle_events_before_the_agents_starts(self):
        while self._simulate_next_event(before_agents_start=True)[0]:
            pass

    def _reset_resources(self):
        for resource in self.resources:
            resource.max_parking_duration_seconds = 0
            resource.time_last_violation = 0
            resource.status = ParkingStatus.FREE
            resource.arrival_time = 0

    def reset(self, reset_days=False, only_do_single_episode=False) -> Union[bool, dict]:
        """Resets the state of the environment."""  # todo add better documentation
        logging.debug('Resetting simulation environment.')

        # Reset resources
        self._reset_resources()

        if reset_days:
            self._reset_days_in_reset()

        # this means that we have simulated a whole year (episode)
        if not self._next_day(with_remove=self._remove_days):
            if only_do_single_episode:
                return False
            else:
                self._reset_days_in_reset()

        self._handle_events_before_the_agents_starts()

        self.agents = self.possible_agents[:]
        self.rewards : dict[Any, float] = {agent: 0.0 for agent in self.agents}
        self.discounted_rewards : dict[Any, float] = {agent: 0.0 for agent in self.agents}
        self.starting_positions = {agent: self._get_initial_position(self.graph) for agent in self.agents}
        self.agent_states = []
        self._initialize_agent_states()

        self.observations = {agent: self._observation_creator.create_observation(self, self.agent_states[agent]) for
                             agent in self.agents}
        self.dones : dict[Any, bool] = {agent: False for agent in self.agents}
        self.infos = {agent: {"dt": 0} for agent in self.agents}

        self._agent_selector = top_agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        return self.observations  # todo may be it would be better create a copy at this point

    def _initialize_agent_states(self):
        for agent_id, starting_position in self.starting_positions.items():
            previous_node = next(self.graph.predecessors(starting_position))
            pos_on_edge = int(self.graph[previous_node][starting_position]["length"])
            agent_state = AgentState(agent_id=agent_id,
                                     position_node=starting_position,
                                     position_node_source=previous_node,
                                     position_on_edge=pos_on_edge)

            self.agent_states.append(agent_state)

    @property
    def unwrapped(self):
        return self

    def _reset_days_in_reset(self):
        self._init_days_to_simulate()
        self._next_day(with_remove=self._remove_days)
        if self._calculate_advanced_stats:
            self._advanced_metrics = defaultdict(list)

    def render(self, mode='human', show=False, additional_info=None) -> []:
        return self.renderer.render(self, mode, show, additional_info)

    def close(self) -> None:
        """Closes the running environment instance."""
        self.graph = None
        logging.info('Simulation environment closed.')