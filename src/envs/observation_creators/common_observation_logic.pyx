import networkx as nx

cdef class CommonObservationLogic(Observation):
    """
    This class capsules logic that is used in many observations
    """

    def __init__(self, config: DictConfig, graph: nx.Graph, number_of_resources: int, number_of_agents: int):
        super().__init__(config, graph, number_of_resources, number_of_agents)
        self.distance_normalization = config.distance_normalization

        y = nx.get_node_attributes(graph, "y")
        x = nx.get_node_attributes(graph, "x")

        self._min_y = min(y.values())
        self._min_x = min(x.values())
        self._max_y = max(y.values())
        self._max_x = max(x.values())


    cdef list get_resource_ids_in_routes_for_agents(self, env):
        routes = []
        for agent_state in env.agent_states:
            agent_state: AgentState
            resource_ids_in_route = set()
            routes.append(resource_ids_in_route)

            if agent_state.current_route is None or len(agent_state.current_route) < 1:
                continue

            previous_node = agent_state.current_route[0]
            for node in agent_state.current_route[1:]:
                if (previous_node, node) in env.edge_to_resource_id_mapping:
                    resource_ids_in_route.update(env.edge_to_resource_id_mapping[(previous_node, node)])

                previous_node = node
        return routes
