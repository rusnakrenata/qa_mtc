import osmnx as ox
import geopandas as gpd
import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def get_city_graph(city_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download and return the city road network as node and edge DataFrames.

    Args:
        city_name: Name of the city (e.g., 'Košice, Slovakia')

    Returns:
        nodes: DataFrame of nodes
        edges: DataFrame of edges
    """
    try:
        ## FOR TESTING
        # Coordinates of Hlavná ulica (Košice center)
        #kosice_center = (48.7208, 21.2575)  # (lat, lon)
        
        # Coordinates of the roundabout near Moldavská cesta (OC Optima)
        #moldavska_roundabout = (48.7019, 21.2446)  # (lat, lon)

        # Get a drivable road network within 1.5 km radius of city center
        #G = ox.graph_from_point(kosice_center, dist=500, network_type='drive')

        G = ox.graph_from_place(city_name, network_type='drive')
        G.graph['crs'] = 'epsg:4326'
        nodes, edges = ox.graph_to_gdfs(G)
        nodes = nodes.reset_index()
        edges = edges.reset_index()
        logger.info(f"Downloaded city graph for '{city_name}' with {len(nodes)} nodes and {len(edges)} edges.")
        return nodes, edges
    except Exception as e:
        logger.error(f"Error downloading city graph for '{city_name}': {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame()
