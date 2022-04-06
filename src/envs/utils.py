"""
Module contains several utility methods for the simulation environment.
"""

import logging
from typing import Any, Tuple

import functools
import networkx as nx
import numpy as np
from numpy import ndarray
from shapely.geometry import LineString



def load_graph_from_file(filename: str) -> nx.DiGraph:
    """Loads graph with assigned parking spots from file."""
    graph: nx.DiGraph = nx.read_gpickle(filename)

    logging.info('Graph loaded from file: %s', filename)
    logging.info('Graph info: %s', nx.info(graph))

    return graph


def ensure_strongly_connected(graph):
    """Ensures that the graph is a single strongly connected component."""
    if nx.number_strongly_connected_components(graph) > 1:
        queue = sorted(nx.strongly_connected_components(graph),
                       key=len, reverse=True)[1:]
        while queue:
            max_scc = max(nx.strongly_connected_components(graph), key=len)
            component = queue.pop(0)
            connected = False
            for node in iter(component):
                # check neighbors that point to the node in this component
                nghbs = {n for n, _ in graph.in_edges(node)}
                intersection = nghbs.intersection(max_scc)
                if len(intersection) > 0:
                    add_reverted_edge(graph, list(intersection),
                                      node, node_to_ngh=True)
                    connected = True
                    break
                # check neighbors that are reachable from the node in this
                # component
                nghbs = {n for _, n in graph.out_edges(node)}
                intersection = nghbs.intersection(max_scc)
                if len(intersection) > 0:
                    add_reverted_edge(graph, list(
                        intersection), node, node_to_ngh=False)
                    connected = True
                    break
            if connected:
                continue
            queue.append(component)
    logging.info("Graph is now strongly connected.")
    return graph


def add_reverted_edge(graph, neighbors, node, node_to_ngh):
    """Adds a reverted edge."""
    neighbor = neighbors[np.random.randint(len(neighbors))]
    src = node if node_to_ngh else neighbor
    target = neighbor if node_to_ngh else node
    edge_data = {"id": graph[target][src]["id"] + "-r",
                 "length": graph[target][src]["length"]}
    # revert geometry if exists
    if 'geometry' in edge_data:
        geometry = np.flip(
            np.array(graph[target][src][0]["geometry"].xy), axis=1)
        edge_data["geometry"] = LineString(
            list(zip(geometry[0, :], geometry[1, :])))
    graph.add_edge(src, target, **edge_data)


def prune_subgraphs(graph):
    """"Removes (weekly connected) subgraphs without any parking spots."""
    # extract subgraphs
    sub_graphs = [graph.subgraph(c).copy()
                  for c in nx.weakly_connected_components(graph)]
    # delete all subgraphs except the biggest one
    for subgraph in sub_graphs[1:]:
        # remove subgraphs without parking spots
        if get_num_spots(subgraph) == 0:
            graph.remove_nodes_from(subgraph)
        else:
            # either way remove the subgraph
            logging.info("Deleting graph with %i spots.", get_num_spots(graph))
            graph.remove_nodes_from(subgraph)
    return graph


def get_subgraphs(graph):
    """Returns the (weekly connected) subgraphs and some info."""
    # extract subgraphs
    sub_graphs = [graph.subgraph(c).copy()
                  for c in nx.weakly_connected_components(graph)]
    for i, subgraph in enumerate(sub_graphs):
        logging.info("Subgraph %i has:", i)
        logging.info("\tNodes: %i", len(subgraph.nodes(data=True)))
        logging.info("\tEdges: %i", len(subgraph.edges()))
        logging.info("\tSpots: %i", get_num_spots(subgraph))


def get_num_spots(graph):
    """Returns the number of parking spots in a graph."""
    spots = 0
    for _, _, data in graph.edges(data=True):
        if "spots" in data:
            for _ in data["spots"]:
                spots += 1
    return spots


def get_spots(graph) -> [Tuple[Any,Any]]:
    """Returns all parking spots"""
    spots = []
    for start_node, end_node, data in graph.edges(data=True):
        if "spots" in data:
            for spot in data["spots"]:
                spots.append((start_node, end_node))
    return spots


def get_avg_walking_time(graph):
    """Returns the average walking time of all edges."""
    # todo is this even correct?
    time = 0
    i = 0
    for _, _, data in graph.edges(data=True):
        time += data["length"]
        i += 1
    return time / i

@functools.lru_cache(maxsize=None)
def get_edges_w_spots(graph):
    """Returns all edges that contain parking spots in a graph."""
    edges = {}
    i = 0
    for node_one, node_two, data in graph.edges(data=True):
        if "spots" in data:
            edges[i] = (node_one, node_two)
            i += 1
    return edges


def precompute_shortest_paths(graph: nx.DiGraph) -> dict:
    """Compute the shortest path from each node in the graph to each other node in the graph (target comes first)

    Returns:
        lookup, a dictionary of dictionaries of paths target should be specified first

    Usage:
        lookup[4][7] returns the shortest path from node 7 to node 4

        If you look up the path from a node to an edge, do the following:
            1. Loop up route from node to edge[0] (=start of edge)
                `path = lookup[edge[0]][node]`
            2. Append edge[1] (=end of edge) to the resulting list
                `path.append(edge[1])`

        If you search for the route from node to edge[1] directly you might get a route that doesn't
            pass the desired edge!
    """
    shortest_paths = {}
    resource_edges = get_spots(graph)
    for edge in resource_edges:
        shortest_paths[edge] = dict(nx.shortest_path(graph, target=edge[0], weight='length'))
    return shortest_paths

@functools.lru_cache(maxsize=None)
def get_distance_matrix(graph : nx.Graph) -> ndarray:
    """
    returns dist matrix; assumes that resource are positioned at the end of an edge
    shape: edge of action x resource
    """


    spots = get_spots(graph)
    edges_with_spots = list(get_edges_w_spots(graph).values())
    shortest_paths = {e: dict(nx.shortest_path_length(graph, target=e[0], weight='length')) for e in edges_with_spots}

    distance_matrix = np.zeros((len(edges_with_spots), len(spots)), dtype=np.float32)

    for edge_index, edge in enumerate(edges_with_spots):
        for resource_index, resource_edge in enumerate(spots):
            distance = shortest_paths[resource_edge][edge[1]]+ graph[resource_edge[0]][resource_edge[1]]["length"]
            distance_matrix[edge_index, resource_index] = distance

    return distance_matrix
