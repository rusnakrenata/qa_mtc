import pandas as pd
import numpy as np
import logging
from typing import List, Any, Optional, Tuple
import networkx as nx
import community as community_louvain  # pip install python-louvain
from igraph import Graph
import leidenalg

logger = logging.getLogger(__name__)


def select_top_congested_vehicles(w: np.ndarray, vehicle_ids: List[Any], num_selected: int) -> List[Any]:
    """
    Fast selection: Select vehicles with the highest total congestion score.
    Args:
        w: 4D numpy array of shape (n, n, t, t) with congestion weights.
        vehicle_ids: List of vehicle IDs.
        num_selected: Number of vehicles to select.
    Returns:
        List of selected vehicle IDs.
    """
    n = len(vehicle_ids)
    # Sum over all other vehicles and all route pairs
    total_scores = np.sum(w, axis=(1,2,3))  # shape: (n,)
    top_indices = np.argpartition(-total_scores, num_selected)[:num_selected]
    selected = sorted(top_indices.tolist())
    return [vehicle_ids[i] for i in selected]


def select_dense_vehicle_subset(w: np.ndarray, vehicle_ids: List[Any], num_selected: int) -> List[Any]:
    """
    (Legacy, slow) Greedy selection of densest subset based on pairwise interactions.
    Only use for small n.
    """
    n = len(vehicle_ids)
    t = len(w[0][0])
    interaction_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            interaction_matrix[i, j] = np.sum(w[i][j])
    scores = interaction_matrix.sum(axis=1)
    selected = [int(np.argmax(scores))]
    remaining = set(range(n)) - set(selected)
    while len(selected) < num_selected and remaining:
        best_candidate = None
        best_score = -1
        for r in remaining:
            density = sum(interaction_matrix[r, s] for s in selected)
            if density > best_score:
                best_score = density
                best_candidate = r
        if best_candidate is not None:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
    selected.sort()
    return [vehicle_ids[i] for i in selected]


def filter_vehicles_by_congested_edges_and_limit(
    congestion_df: pd.DataFrame,
    max_vehicles: int
) -> List[Any]:
    """
    Select up to max_vehicles that contribute to the most congested edges.
    Returns:
        filtered_vehicle_ids: List of selected vehicle IDs (in QUBO order)
    """
    edge_scores = congestion_df.groupby('edge_id')['congestion_score'].sum()
    edge_scores = pd.Series(edge_scores)
    sorted_edges = edge_scores.sort_values(ascending=False).index.tolist()  # type: ignore
    selected_vehicles = []
    selected_set = set()
    for edge_id in sorted_edges:
        edge_rows = congestion_df[congestion_df['edge_id'] == edge_id]
        vehicles = set(edge_rows['vehicle1']).union(set(edge_rows['vehicle2']))
        for v in vehicles:
            if v not in selected_set and len(selected_vehicles) < max_vehicles:
                selected_vehicles.append(v)
                selected_set.add(v)
            if len(selected_vehicles) >= max_vehicles:
                break
        if len(selected_vehicles) >= max_vehicles:
            break
    logger.info(f"Selected {len(selected_vehicles)} vehicles for QUBO.")
    return selected_vehicles


def filter_edges_by_cumulative_congestion(
    congestion_df: pd.DataFrame,
    target_fraction: float = 0.8
) -> pd.DataFrame:
    """
    Select the smallest set of edges whose cumulative congestion accounts for at least target_fraction of total congestion.
    Args:
        congestion_df: DataFrame with columns ['edge_id', 'congestion_score']
        target_fraction: Target fraction of total congestion to cover (e.g., 0.8 for 80%)
    Returns:
        DataFrame of selected edges with columns ['edge_id', 'congestion_score', 'congestion_score_rel', 'cumulative_congestion']
    """
    # Group by edge_id and sum congestion_score
    grouped = congestion_df.groupby('edge_id', as_index=False)['congestion_score'].sum()
    # Compute total congestion
    congestion_score_total = grouped['congestion_score'].sum()
    # Compute relative congestion per edge
    grouped = grouped.assign(congestion_score_rel=grouped['congestion_score'] / congestion_score_total)
    # Sort by congestion_score_rel descending
    grouped = grouped.sort_values('congestion_score_rel', ascending=False)
    # Compute cumulative sum
    grouped['cumulative_congestion'] = grouped['congestion_score_rel'].cumsum()
    # Select edges until cumulative sum >= target_fraction
    selected_edges = grouped[grouped['cumulative_congestion'] <= target_fraction]
    # Always include the first edge that crosses the threshold
    if not selected_edges.empty:
        cum_cong = np.asarray(selected_edges['cumulative_congestion'])
        if cum_cong[-1] < target_fraction and len(selected_edges) < len(grouped):
            next_edge = grouped.iloc[len(selected_edges)]
            selected_edges = pd.concat([selected_edges, next_edge.to_frame().T], ignore_index=True)
    # Ensure return type is always DataFrame
    if not isinstance(selected_edges, pd.DataFrame):
        selected_edges = pd.DataFrame(selected_edges)
    return selected_edges.reset_index(drop=True)


def select_vehicles_on_selected_edges(
    congestion_df: pd.DataFrame,
    selected_edges_df: pd.DataFrame
) -> list:
    """
    Given selected edges, return unique vehicle IDs that traverse these edges.
    Args:
        congestion_df: DataFrame with columns ['edge_id', 'vehicle1', 'vehicle2', ...]
        selected_edges_df: DataFrame with column 'edge_id' (from filter_edges_by_cumulative_congestion)
    Returns:
        List of unique vehicle IDs traversing the selected edges.
    """
    selected_edge_ids = list(selected_edges_df['edge_id'])
    filtered = congestion_df[congestion_df['edge_id'].isin(selected_edge_ids)]
    vehicles = set(filtered['vehicle1']).union(set(filtered['vehicle2']))
    return list(vehicles)


def select_vehicles_by_cumulative_congestion(
    congestion_df: pd.DataFrame,
    target_fraction: float = 0.8
) -> list:
    """
    Select a set of vehicles whose cumulative congestion accounts for at least target_fraction of total vehicle congestion.
    Args:
        congestion_df: DataFrame with columns ['vehicle1', 'vehicle2', 'congestion_score']
        target_fraction: Target fraction of total vehicle congestion to cover (e.g., 0.8 for 80%)
    Returns:
        List of selected vehicle IDs.
    """
    # Ensure input is a DataFrame
    congestion_df = pd.DataFrame(congestion_df)
    # Stack vehicle1 and vehicle2 as 'vehicle' and sum congestion_score
    v1 = pd.DataFrame(congestion_df[['vehicle1', 'congestion_score']]).rename(columns={'vehicle1': 'vehicle'})
    v2 = pd.DataFrame(congestion_df[['vehicle2', 'congestion_score']]).rename(columns={'vehicle2': 'vehicle'})
    all_vehicles = pd.concat([v1, v2], ignore_index=True)
    vehicle_congestion = all_vehicles.groupby('vehicle', as_index=False)['congestion_score'].sum()
    # Compute total congestion
    total_congestion = vehicle_congestion['congestion_score'].sum()
    # Compute relative congestion per vehicle
    vehicle_congestion = vehicle_congestion.assign(rel=vehicle_congestion['congestion_score'] / total_congestion)
    # Sort by relative congestion descending
    vehicle_congestion = vehicle_congestion.sort_values(by='rel', ascending=False)
    # Compute cumulative sum
    vehicle_congestion = vehicle_congestion.assign(cumulative_rel=vehicle_congestion['rel'].cumsum())
    # Select vehicles until cumulative sum >= target_fraction
    selected = vehicle_congestion[vehicle_congestion['cumulative_rel'] <= target_fraction]
    # Always include the first vehicle that crosses the threshold
    if not selected.empty:
        cum_rel = np.asarray(selected['cumulative_rel'])
        if cum_rel[-1] < target_fraction and len(selected) < len(vehicle_congestion):
            next_vehicle = vehicle_congestion.iloc[len(selected)]
            selected = pd.concat([selected, next_vehicle.to_frame().T], ignore_index=True)
    vehicle_congestion['vehicle'] = vehicle_congestion['vehicle'].astype(int)
    selected_vehicle_ids = [int(v) for v in selected['vehicle'].tolist()]
    selected_vehicle_ids_sorted = sorted(selected_vehicle_ids)
    logger.info(f"Number of selected vehicle indexes: {len(selected_vehicle_ids_sorted)}")
    return selected_vehicle_ids_sorted


def select_vehicles_by_louvain_clustering(
    congestion_df: pd.DataFrame,
    min_cluster_size: int = 1
) -> list:
    """
    Cluster vehicles using Louvain community detection based on congestion interactions
    and select the cluster with the highest total inter-vehicle congestion.
    Args:
        congestion_df: DataFrame with columns ['vehicle1', 'vehicle2', 'congestion_score']
        min_cluster_size: Minimum size of cluster to consider (default 1)
    Returns:
        List of selected vehicle IDs from the densest cluster.
    """
    # Build weighted graph
    G = nx.Graph()
    for _, row in congestion_df.iterrows():
        v1 = int(row['vehicle1'])
        v2 = int(row['vehicle2'])
        w = float(row['congestion_score'])
        if G.has_edge(v1, v2):
            G[v1][v2]['weight'] += w
        else:
            G.add_edge(v1, v2, weight=w)
    # Louvain partitioning
    partition = community_louvain.best_partition(G, weight='weight')
    # Group nodes by cluster
    clusters = {}
    for node, cluster_id in partition.items():
        clusters.setdefault(cluster_id, []).append(node)
    # Compute total congestion for each cluster
    cluster_scores = {}
    for cluster_id, nodes in clusters.items():
        subgraph = G.subgraph(nodes)
        total_weight = sum(d['weight'] for u, v, d in subgraph.edges(data=True))
        if len(nodes) >= min_cluster_size:
            cluster_scores[cluster_id] = total_weight
    # Select densest cluster
    if not cluster_scores:
        return []
    densest_cluster = max(list(cluster_scores), key=lambda k: cluster_scores[k])
    selected_vehicle_ids = sorted(clusters[densest_cluster])
    logger.info(f"Number of selected vehicle indexes in densest cluster: {len(selected_vehicle_ids)}")
    return selected_vehicle_ids


def select_vehicles_by_leiden_clustering(
    congestion_df: pd.DataFrame,
    min_cluster_size: int = 1,
    resolution: float = 1.0
) -> list:
    """
    Cluster vehicles using Leiden community detection based on congestion interactions
    and select the cluster with the highest total inter-vehicle congestion.
    Args:
        congestion_df: DataFrame with columns ['vehicle1', 'vehicle2', 'congestion_score']
        min_cluster_size: Minimum size of cluster to consider (default 1)
        resolution: Resolution parameter for Leiden algorithm (default 1.0)
    Returns:
        List of selected vehicle IDs from the densest cluster.
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
    # Build clusters as lists of vehicle IDs
    clusters = [[idx_to_node[i] for i in community] for community in part]
    # Compute total congestion for each cluster
    cluster_scores = {}
    for idx, community in enumerate(part):
        if len(community) >= min_cluster_size:
            subgraph = g.subgraph(community)
            total_weight = sum(subgraph.es['weight']) if subgraph.ecount() > 0 else 0.0
            cluster_scores[idx] = total_weight
    # Select densest cluster
    if cluster_scores:
        densest_cluster = max(list(cluster_scores), key=lambda k: cluster_scores[k])
        selected_vehicle_ids = sorted([idx_to_node[i] for i in part[densest_cluster]])
    else:
        # Fallback: pick the largest cluster
        largest_cluster = max(range(len(part)), key=lambda k: len(part[k]))
        selected_vehicle_ids = sorted([idx_to_node[i] for i in part[largest_cluster]])
    logger.info(f"Number of selected vehicle indexes in chosen Leiden cluster: {len(selected_vehicle_ids)}")
    return selected_vehicle_ids


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

