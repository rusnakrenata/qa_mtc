import osmnx as ox
import geopandas as gpd
import pandas as pd
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def get_city_graph(city_name: str, center_coords: Optional[Tuple[float, float]] = None, radius_km: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download and return the city road network as node and edge DataFrames.

    Args:
        city_name: Name of the city (e.g., 'Košice, Slovakia')
        center_coords: Optional tuple of (lat, lon) coordinates for center point
        radius_km: Radius in kilometers around the center point (default: 1.0 km)

    Returns:
        nodes: DataFrame of nodes
        edges: DataFrame of edges
    """
    try:
        if center_coords is not None and radius_km is not None:
            # Generate subset of city around specified coordinates
            lat, lon = center_coords
            logger.info(f"Downloading city subset for '{city_name}' around coordinates ({lat}, {lon}) with radius {radius_km}km")
            
            # Convert km to meters for ox.graph_from_point
            radius_meters = radius_km * 1000 
            
            G = ox.graph_from_point(center_coords, dist=radius_meters, network_type='drive') #type: ignore
            G.graph['crs'] = 'epsg:4326'
            nodes, edges = ox.graph_to_gdfs(G)
            nodes = nodes.reset_index()
            edges = edges.reset_index()
            
            logger.info(f"Downloaded city subset with {len(nodes)} nodes and {len(edges)} edges.")
        else:
            # Get the whole city
            logger.info(f"Downloading entire city graph for '{city_name}'")
            
            G = ox.graph_from_place(city_name, network_type='drive')
            G.graph['crs'] = 'epsg:4326'
            nodes, edges = ox.graph_to_gdfs(G)
            nodes = nodes.reset_index()
            edges = edges.reset_index()
            
            logger.info(f"Downloaded entire city graph with {len(nodes)} nodes and {len(edges)} edges.")
        
        return nodes, edges
    except Exception as e:
        logger.error(f"Error downloading city graph for '{city_name}': {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame()
