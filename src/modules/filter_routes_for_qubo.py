import pandas as pd
import numpy as np
import logging
from typing import List, Any, Optional, Tuple

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

