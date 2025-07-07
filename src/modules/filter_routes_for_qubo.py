import pandas as pd
import numpy as np
import logging
import networkx as nx
from igraph import Graph
import leidenalg

logger = logging.getLogger(__name__)

def select_vehicles_by_leiden_joined_clusters(
    congestion_df: pd.DataFrame,
    target_size: float,
    resolution: float = 0.1
) -> tuple:
    """
    Cluster vehicles using Leiden community detection and join clusters until the total number of vehicles
    reaches or exceeds the target_size. Clusters are added in order of descending total congestion.
    Args:
        congestion_df: DataFrame with columns ['edge_id', 'vehicle1', 'vehicle2', 'congestion_score']
        target_size: Minimum number of vehicles to select (e.g., N_VEHICLES//4)
        resolution: Resolution parameter for Leiden algorithm (default 1.0)
    Returns:
        Tuple: (List of selected vehicle IDs from the joined clusters, DataFrame of affected edges and their congestion score)
    """
    # Prepare nodes and edges
    edges = list(zip(congestion_df['vehicle1'], congestion_df['vehicle2']))
    weights = list(congestion_df['congestion_score'])
    nodes = set(congestion_df['vehicle1']).union(congestion_df['vehicle2'])
    node_to_idx = {n: i for i, n in enumerate(sorted(nodes))}
    idx_to_node = {i: n for n, i in node_to_idx.items()}
    g = Graph()
    g.add_vertices(len(nodes))
    g.add_edges([(node_to_idx[u], node_to_idx[v]) for u, v in edges])
    g.es['weight'] = weights
    # Leiden partitioning
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution
    )
    # Build clusters as lists of vehicle IDs and compute their total congestion
    cluster_info = []
    for idx, community in enumerate(part):
        subgraph = g.subgraph(community)
        total_weight = sum(subgraph.es['weight']) if subgraph.ecount() > 0 else 0.0
        vehicle_ids = [idx_to_node[i] for i in community]
        cluster_info.append((total_weight, vehicle_ids))
    # Sort clusters by total congestion (descending)
    cluster_info.sort(reverse=True, key=lambda x: x[0])
    # Join clusters until target_size is reached
    selected_vehicle_ids = set()
    for _, vehicle_ids in cluster_info:
        selected_vehicle_ids.update(vehicle_ids)
        if len(selected_vehicle_ids) >= target_size:
            break
    selected_vehicle_ids_sorted = sorted(selected_vehicle_ids)
    logger.info(f"Number of selected vehicle indexes in joined Leiden clusters: {len(selected_vehicle_ids_sorted)}")

    # Filter congestion_df for edges where both vehicles are in the selected set
    selected_list = list(selected_vehicle_ids_sorted)
    affected_edges_df = congestion_df[
        congestion_df['vehicle1'].isin(selected_list) & congestion_df['vehicle2'].isin(selected_list)
    ][['edge_id', 'congestion_score']]
    # Group by edge_id and sum congestion_score (in case of duplicates)
    if isinstance(affected_edges_df, pd.DataFrame) and not affected_edges_df.empty:
        affected_edges_df = affected_edges_df.groupby('edge_id', as_index=False)['congestion_score'].sum()

    return selected_vehicle_ids_sorted, affected_edges_df


def select_vehicles_simple(
    congestion_df: pd.DataFrame
) -> tuple:
    """
    Simple vehicle filtering without any clustering algorithm.
    Selects vehicles up to target_size and returns affected edges.
    Args:
        congestion_df: DataFrame with columns ['edge_id', 'vehicle1', 'vehicle2', 'congestion_score']
        target_size: Number of vehicles to select
    Returns:
        Tuple: (List of selected vehicle IDs, DataFrame of affected edges and their congestion score)
    """
    # Get all unique vehicles
    all_vehicles = set(congestion_df['vehicle1']).union(set(congestion_df['vehicle2']))
    all_vehicles_sorted = sorted(all_vehicles)
    
    # Select vehicles up to target_size
    selected_vehicle_ids = all_vehicles_sorted
    logger.info(f"Number of selected vehicles (simple): {len(selected_vehicle_ids)}")

    # Filter congestion_df for edges where both vehicles are in the selected set
    selected_list = list(selected_vehicle_ids)
    affected_edges_df = congestion_df[
        congestion_df['vehicle1'].isin(selected_list) & congestion_df['vehicle2'].isin(selected_list)
    ][['edge_id', 'congestion_score']]
    # Group by edge_id and sum congestion_score (in case of duplicates)
    if isinstance(affected_edges_df, pd.DataFrame) and not affected_edges_df.empty:
        affected_edges_df = affected_edges_df.groupby('edge_id', as_index=False)['congestion_score'].sum()

    return selected_vehicle_ids, affected_edges_df

