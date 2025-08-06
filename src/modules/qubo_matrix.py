from collections import defaultdict
from typing import List, Dict, Tuple, Any
import logging
import pandas as pd
import numpy as np
from congestion_weights import congestion_weights
import time

logger = logging.getLogger(__name__)

def qubo_matrix(
    route_alternatives: int,
    weights_df: pd.DataFrame,
    duration_penalty_df: pd.DataFrame,
    vehicle_ids_filtered: List[int],
    vehicle_routes_df: pd.DataFrame,
    comp_type: str = "hybrid"
) -> Tuple[Dict[Tuple[int, int], float], int, float]:
    """
    Construct QUBO matrix to minimize congestion.

    Parameters:
    - route_alternatives: Number of route alternatives per vehicle.
    - weights_df: DataFrame with pairwise congestion weights.
    - duration_penalty_df: DataFrame containing duration penalties for each vehicle-route pair.
    - vehicle_ids_filtered: List of vehicle IDs selected for QUBO optimization.
    - vehicle_routes_df: DataFrame containing valid routes for each vehicle.
    - comp_type: Optimization mode (default: 'hybrid', alternative: 'hybrid_cqm').

    Returns:
    - Q: QUBO matrix as a dictionary.
    - route_alternatives: Echoes the number of route alternatives.
    - lambda_penalty: Penalty parameter applied for enforcing constraints.
    """
    start_time = time.time()

    logger.info("Starting QUBO vehicle filtering...")
    n_filtered = len(vehicle_ids_filtered)
    logger.info(f"Vehicle filtering complete. {n_filtered} vehicles selected. Time elapsed: {time.time() - start_time:.2f}s")

    # Compute congestion weights
    logger.info("Computing congestion weights for filtered vehicles...")
    weights_start = time.time()
    congestion_w, max_w = congestion_weights(weights_df, n_filtered, route_alternatives, vehicle_ids_filtered, vehicle_routes_df)
    logger.info(f"Using max_weight strategy: max_w={max_w}")
    logger.info(f"Congestion weights computed. Time elapsed: {time.time() - weights_start:.2f}s")

    duration_penalty_df = duration_penalty_df[duration_penalty_df['vehicle'].isin(vehicle_ids_filtered)]

    # Map vehicle and route IDs to indices
    vehicle_id_to_idx = {int(v): i for i, v in enumerate(vehicle_ids_filtered)}
    route_ids = sorted(vehicle_routes_df['route_id'].unique())
    route_id_to_idx = {int(r): k for k, r in enumerate(route_ids)}

    # Penalty matrix construction
    penalty_matrix = np.zeros((n_filtered, len(route_ids)))
    for _, row in duration_penalty_df.iterrows():
        i = vehicle_id_to_idx[int(row['vehicle'])]
        k = route_id_to_idx[int(row['route'])]
        penalty_matrix[i, k] = row['penalty']

    penalties = [penalty_matrix[i, k] for i in range(n_filtered) for k in range(route_alternatives)]

    # Constructing QUBO matrix
    logger.info("Constructing QUBO objective terms...")
    qubo_start = time.time()
    Q = defaultdict(float)
    for i in range(n_filtered):
        for j in range(i + 1, n_filtered):
            for k1 in range(route_alternatives):
                for k2 in range(route_alternatives):
                    q1 = i * route_alternatives + k1
                    q2 = j * route_alternatives + k2
                    Q[(q1, q2)] += congestion_w[i][j][k1][k2]

    logger.info(f"QUBO objective constructed: {len(Q)} terms. Time: {time.time() - qubo_start:.2f}s")

    valid_pairs = set(zip(vehicle_routes_df['vehicle_id'], vehicle_routes_df['route_id']))

    dynamic_penalties, q_indices, not_real_routes_indices = [], [], []
    for i, vehicle_id in enumerate(vehicle_ids_filtered):
        for route_num in range(1, route_alternatives + 1):
            q = i * route_alternatives + (route_num - 1)
            if (vehicle_id, route_num) in valid_pairs:
                row_values = [Q.get((q, j), 0) for j in range(n_filtered * route_alternatives)]
                col_values = [Q.get((idx, q), 0) for idx in range(n_filtered * route_alternatives)]
                row_sum = sum(max(row_values[x], row_values[x+1]) for x in range(0, len(row_values)-1, 2))
                col_sum = sum(max(col_values[x], col_values[x+1]) for x in range(0, len(col_values)-1, 2))
                dynamic_penalties.append(row_sum + col_sum)
            else:
                not_real_routes_indices.append(q)
            q_indices.append(q)

    lambda_penalty = max(dynamic_penalties) if dynamic_penalties else 1.0
    logger.info(f"Lambda_penalty computed: {lambda_penalty}")

    # Apply one-hot constraints unless hybrid_cqm mode
    if comp_type != "hybrid_cqm":
        logger.info("Applying one-hot constraints to QUBO...")
        for q in q_indices:
            if q in not_real_routes_indices:
                Q[(q, q)] += lambda_penalty
            else:
                Q[(q, q)] += lambda_penalty * (-1) + penalties[q]

        for i in range(n_filtered):
            for k1 in range(route_alternatives):
                q1 = i * route_alternatives + k1
                for k2 in range(route_alternatives):
                    if k1 != k2:
                        q2 = i * route_alternatives + k2
                        Q[(q1, q2)] += lambda_penalty
    else:
        logger.info("Hybrid CQM mode: no one-hot constraints applied.")
        for q in q_indices:
            if q not in not_real_routes_indices:
                Q[(q, q)] += penalties[q]

    logger.info(f"QUBO matrix constructed: {len(Q)} terms, {n_filtered} vehicles. Total time: {time.time() - start_time:.2f}s")

    return dict(Q), route_alternatives, lambda_penalty