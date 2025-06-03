import osmnx as ox
import geopandas as gpd
import pandas as pd

def get_city_graph(city_name):
    import osmnx as ox
    G = ox.graph_from_place(city_name, network_type='drive')
    G.graph['crs'] = 'epsg:4326'
    nodes, edges = ox.graph_to_gdfs(G)
    nodes = nodes.reset_index()
    edges = edges.reset_index()
    return nodes, edges
