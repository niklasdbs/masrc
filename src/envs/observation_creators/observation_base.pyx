from envs.utils import get_edges_w_spots
import networkx as nx

cdef class Observation:
    def __init__(self, config : DictConfig, graph: nx.Graph, number_of_resources : int, number_of_agents : int):
        self.walking_speed = config.speed_in_kmh/3.6 #the walking speed in m/s (maybe move this to agent state?)
        self.number_of_resources : int = number_of_resources
        self.number_of_agents : int = number_of_agents
        self.shortest_path_lengths = {} # dictionary from [resource_source][resource_target][current_position]
        for source, target in get_edges_w_spots(graph).values():
            resource_edge = (source, target)
            if source not in self.shortest_path_lengths:
                self.shortest_path_lengths[source] = {}


            self.shortest_path_lengths[source][target] = dict(nx.shortest_path_length(graph, target=resource_edge[0], weight="length"))

        self.length_of_edges = {}
        for source, target, data in graph.edges(data=True):
            if source not in self.length_of_edges:
                self.length_of_edges[source] = {}
            self.length_of_edges[source][target] = data["length"] #assumes that resources are located at the end of an edge


    @property
    #@abstractmethod
    def observation_space(self):
        return None

    #@abstractmethod
    cdef create_observation(self, env, current_agent_state :  AgentState):
        return None

    cpdef float distance(self, resource_source, resource_target, agent_position_source, agent_position_target, agent_position_on_edge=None):
        """
        assumes that resources are located at the end of an edge

        Args:
            resource_source:
            resource_target:
            agent_position_source:
            agent_position_target:
            agent_position_on_edge:
        """

        if agent_position_target is None:
            agent_position_on_edge = 0

        current_edge_length = self.length_of_edges[agent_position_source][agent_position_target]
        time_to_target_of_edge = current_edge_length - agent_position_on_edge

        return self.shortest_path_lengths[resource_source][resource_target][agent_position_target] + time_to_target_of_edge + self.length_of_edges[resource_source][resource_target]
