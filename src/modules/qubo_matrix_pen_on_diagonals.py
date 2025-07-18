from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
import logging
import pandas as pd
from normalize_congestion_weights import normalize_congestion_weights
from congestion_weights import congestion_weights
from penalized_congestion_weights import penalized_congestion_weights
from filter_routes_for_qubo import *
import time
import math

logger = logging.getLogger(__name__)

def qubo_matrix(
    n_vehicles: int,
    t: int,
    congestion_df: pd.DataFrame,
    w_df: pd.DataFrame,
    pd_df: pd.DataFrame,
    vehicle_routes_df: pd.DataFrame,
    lambda_strategy: str = "normalized",
    fixed_lambda: Optional[float] = None,
    filtering_percentage: float = 0.25
) -> Tuple[Dict[Tuple[int, int], float], List[Any], pd.DataFrame, float]:
    """
    Constructs the QUBO dictionary for the traffic assignment problem with 4D weights.
    Args:
        n_vehicles: Number of vehicles
        t: Number of route alternatives per vehicle
        congestion_df: Congestion DataFrame (for filtering)
        w_df: Congestion weights DataFrame
        pd_df: 
        vehicle_routes_df: Vehicle routes DataFrame
        lambda_strategy: "normalized" or "max_weight"
        fixed_lambda: λ if using normalized strategy
    Returns:
        Q: QUBO matrix as a dictionary {(q1, q2): value}
        vehicle_ids_filtered: The filtered list of vehicle IDs used in QUBO
    """
    start_time = time.time()
    logger.info("Starting QUBO vehicle filtering...")
    vehicle_ids_filtered, affected_edges_df = select_vehicles_by_leiden_joined_clusters(congestion_df,target_size=n_vehicles*filtering_percentage, resolution=0.7)
    #select_vehicles_simple(congestion_df)  #select_vehicles_by_cumulative_congestion(congestion_df, filtering_percentage or 1.0)
    n_filtered = len(vehicle_ids_filtered)
    logger.info(f"Vehicle filtering complete. {n_filtered} vehicles selected. Time elapsed: {time.time() - start_time:.2f}s")

    logger.info("Computing congestion weights for filtered vehicles...")
    weights_start = time.time()
    if lambda_strategy == "normalized":
        w, max_w = normalize_congestion_weights(w_df, n_filtered, t, vehicle_ids_filtered, vehicle_routes_df)
        logger.info(f"Using normalized weights max_w={max_w}")
    elif lambda_strategy == "penalized":
        w, max_w, penalty_matrix = penalized_congestion_weights(w_df, pd_df, n_filtered, t, vehicle_ids_filtered, vehicle_routes_df)
        logger.info(f"Using penalized weights max_w={max_w}")
    else:
        w, max_w = congestion_weights(w_df, n_filtered, t, vehicle_ids_filtered, vehicle_routes_df)
        logger.info(f"Using max_weight strategy max_w={max_w}")
    logger.info(f"Congestion weights computed. Time elapsed: {time.time() - weights_start:.2f}s")

    # Build set of valid (vehicle, route) pairs
    valid_pairs = set(zip(vehicle_routes_df['vehicle_id'], vehicle_routes_df['route_id']))

    logger.info("Constructing QUBO matrix...")
    qubo_start = time.time()
    Q = defaultdict(float)
    for i in range(n_filtered):
        for j in range(i + 1, n_filtered):
            for k1 in range(t):
                for k2 in range(t):
                    q1 = i * t + k1
                    q2 = j * t + k2
                    Q[(q1, q2)] += w[i][j][k1][k2]

    # Step 1: Compute all dynamic penalties and store them
    dynamic_penalties = []
    q_indices = []
    not_real_routes_indices = []
    for i in range(n_filtered):
        vi = vehicle_ids_filtered[i]
        # Go through ALL possible routes (1 to t)
        for k in range(1, t + 1):
            q = i * t + (k - 1)
            if (vi, k) in valid_pairs:
                # Real route: use dynamic penalty
                row_sum = sum(Q.get((q, j), 0) for j in range(n_filtered * t))
                col_sum = sum(Q.get((idx, q), 0) for idx in range(n_filtered * t))
                lambda_penalty_dynamic = (row_sum + col_sum)
                dynamic_penalties.append(lambda_penalty_dynamic)
                q_indices.append(q)
            else:
                # Non-real route: use high penalty 
                q_indices.append(q)
                not_real_routes_indices.append(q)

    # Step 2: Find the maximum penalty for real routes
    lambda_penalty = max(dynamic_penalties) 
    logger.info(f"lambda_penalty={lambda_penalty}")

    # Step 3: Apply the max penalty to all diagonal elements
    for q in q_indices:
        i = q // t  # vehicle index
        k = q % t   # route index (0-based)
        
        if q in not_real_routes_indices:
            # High penalty for non-real route
            Q[(q, q)] += 2 * lambda_penalty
        else:
            # Penalty for real route (encourages one-hot)
            route_penalty = penalty_matrix[i, k] if penalty_matrix is not None else 0
            Q[(q, q)] += -lambda_penalty + route_penalty / n_filtered

    # Step 4: Off-diagonal penalties for same vehicle (discourage multiple routes)
    # Step 4: Symmetric off-diagonal penalties (same vehicle, multiple route choices)
    for i in range(n_filtered):
        for k1 in range(t):
            q1 = i * t + k1
            for k2 in range(k1 + 1, t):
                q2 = i * t + k2

                # Get penalties from penalty_matrix
                penalty1 = penalty_matrix[i, k1] if penalty_matrix is not None else 0
                penalty2 = penalty_matrix[i, k2] if penalty_matrix is not None else 0
                avg_penalty = (penalty1 + penalty2) / (2*n_filtered)

                # Apply symmetric penalty
                penalty_value = lambda_penalty + avg_penalty
                Q[(q1, q2)] += penalty_value
                Q[(q2, q1)] += penalty_value




    logger.info(f"QUBO matrix constructed: {len(Q)} nonzero entries, {n_filtered} vehicles. Time elapsed: {time.time() - qubo_start:.2f}s")
    logger.info(f"Total QUBO matrix function time: {time.time() - start_time:.2f}s")
    return dict(Q), vehicle_ids_filtered, affected_edges_df, lambda_penalty

