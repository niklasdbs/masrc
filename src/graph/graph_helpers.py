"""
Stub that imports the osm graph and adds parking spots to its edges
"""
import logging

import networkx as nx
import numpy as np
import osmnx
import osmnx as ox
import pandas as pd
import shapely

from omegaconf import DictConfig
from envs.enums import ParkingStatus
from hydra.utils import to_absolute_path



def assing_parking_spots(graph, spots):
    """Assigns parking spots to closest edges"""

    for _, row in spots.iterrows():
        # check if parking spot has location information
        if np.isnan(row["y"]) or np.isnan(row["x"]):
            continue

        g = osmnx.truncate.truncate_graph_dist(graph, ox.get_nearest_node(graph, (row["y"], row["x"])), 500, retain_all=True)
        nearest_edge = ox.get_nearest_edge(g, (row["y"], row["x"]))

        spot_object = {
            "id": row["marker_id"],
            "status": ParkingStatus.FREE,
            "arrivalTime": 0,
            "x" : row["x"],
            "y" : row["y"]
        }

        # if its the first parking spot at this nearest_edge, add spots attribute
        # else add spot object to spot list
        if 'spots' in graph.edges[nearest_edge]:
            spots = graph.edges[nearest_edge]["spots"].copy()
            spots.append(spot_object)
            graph.edges[nearest_edge].update({"spots": spots})
        else:
            graph.edges[nearest_edge].update({"spots": [spot_object]})

    return graph



def create_graph(config : DictConfig, filename):
    """Creates a graph using a OpenStreetMaps file and sensor data."""
    logging.info(f"Creating graph for districts: {config.area}")

    # load parking spots into data frame
    events = pd.read_pickle(to_absolute_path(config.path_to_event_log), compression="gzip")
    events = events[["StreetMarker", "AreaName"]]#only keep the necessary columns
    events = events.drop_duplicates()#we are only interested in the mapping => remove redundant events

    assert events["StreetMarker"].is_unique

    locations = pd.read_csv(to_absolute_path(config.path_to_bay_locations))
    events = events[events["AreaName"].isin(config.area)]
    spots = pd.merge(events, locations, left_on="StreetMarker", right_on="marker_id")
    logging.info(f"Number of parking spots in selected district: {len(spots)}")

    spots["x"] = spots["the_geom"].apply(lambda x: shapely.wkt.loads(x).centroid.x)
    spots["y"] = spots["the_geom"].apply(lambda x: shapely.wkt.loads(x).centroid.y)


    delta_degree = config.delta_degree

    east = spots["x"].max()
    west = spots["x"].min()

    north = spots["y"].max()
    south = spots["y"].min()

    dx = max(0.01, north - south) * delta_degree
    dy = max(0.01, east - west) * delta_degree


    north = north+dx
    south = south-dx
    west = west-dy
    east = east+dy

    graph = ox.graph_from_bbox(
        south=south, north=north, west=west, east=east, network_type='walk')

    # only look at the important part of the graph
    # spot is the data frame with the parking spots
    # if one wants to use a subset e.g. downtown
    # just just create a dataframe withe only those spots
    # graph = shrink.shrink_by_df(spots, None)
    graph = assing_parking_spots(graph, spots)
    # graph = fix_dead_end_nodes(graph)
    # get_subgraphs(graph)
    # graph = prune_subgraphs(graph)
    # graph = ensure_strongly_connected(graph)
    graph = nx.DiGraph(graph)
    assert nx.is_strongly_connected(graph)

    # save graph as .gpickle
    nx.write_gpickle(graph, filename)
