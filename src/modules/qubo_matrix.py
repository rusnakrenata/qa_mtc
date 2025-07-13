from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
import logging
import pandas as pd
from normalize_congestion_weights import normalize_congestion_weights
from congestion_weights import congestion_weights
from filter_routes_for_qubo import *
import time
import math

logger = logging.getLogger(__name__)

def qubo_matrix(
    n_vehicles: int,
    t: int,
    congestion_df: pd.DataFrame,
    w_df: pd.DataFrame,
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
    else:
        w, max_w = congestion_weights(w_df, n_filtered, t, vehicle_ids_filtered, vehicle_routes_df)
        logger.info(f"Using max_weight strategy max_w={max_w}")
    logger.info(f"Congestion weights computed. Time elapsed: {time.time() - weights_start:.2f}s")

    # Build set of valid (vehicle, route) pairs
    valid_pairs = set(zip(vehicle_routes_df['vehicle_id'], vehicle_routes_df['route_id']))

    logger.info("Constructing QUBO matrix...")
    qubo_start = time.time()
    Q = defaultdict(float)

    # Step 0: Fill the objective terms
    for i in range(n_filtered):
        for j in range(i + 1, n_filtered):
            for k1 in range(t):
                for k2 in range(t):
                    q1 = i * t + k1
                    q2 = j * t + k2
                    Q[(q1, q2)] += w[i][j][k1][k2]

    # Step 1: Compute dynamic penalties for each route (real only)
    dynamic_penalties = []
    q_real_indices = []
    q_fake_indices = []
    for i in range(n_filtered):
        vi = vehicle_ids_filtered[i]
        for k in range(t):
            q = i * t + k
            if (vi, k + 1) in valid_pairs:
                # Valid route → calculate dynamic penalty
                row_sum = sum(Q.get((q, j), 0) for j in range(n_filtered * t))
                col_sum = sum(Q.get((j, q), 0) for j in range(n_filtered * t))
                lambda_dynamic = row_sum + col_sum
                dynamic_penalties.append(lambda_dynamic)
                q_real_indices.append(q)
            else:
                # Invalid route → track separately
                q_fake_indices.append(q)

    # Step 2: Use max dynamic penalty for all valid routes
    scale = n_filtered*t/10000
    if scale < 1:
        gamma = 1
    else:
        gamma = scale
        
    lambda_penalty = max(dynamic_penalties)*gamma # gamma = n_filtered*t/10000 scaling parameter
    logger.info(f"lambda_penalty={lambda_penalty}")

    # Step 3: Add one-hot penalty per vehicle
    for i in range(n_filtered):
        vi = vehicle_ids_filtered[i]
        real_qs = []
        for k in range(t):
            q = i * t + k
            if (vi, k + 1) in valid_pairs:
                real_qs.append(q)
            else:
                # Apply a very large penalty to make fake routes invalid
                Q[(q, q)] += 2 * lambda_penalty  # strong positive penalty

        if len(real_qs) == 0:
            continue  # skip if no real routes

        # Apply full quadratic penalty: λ * (∑ x_q - 1)^2
        for q1 in real_qs:
            # this can be simplified and the linear term can be moved to diagonal
            #Q[(q1, q1)] += lambda_penalty  # x_q^2 terms
            #Q[(q1, 0)] += -2 * lambda_penalty  # linear -2*x_q
            Q[(q1, q1)] += - lambda_penalty
            for q2 in real_qs:
                if q1 < q2:
                    Q[(q1, q2)] += 2 * lambda_penalty  # 2*x_q1*x_q2



    # Optional sanity check:
    max_obj_term = max(Q[q1, q2] for (q1, q2) in Q if q1 != q2)
    print(f"Max objective term: {max_obj_term:.2f}")
    if lambda_dynamic > max_obj_term:
        print("Penalty sufficient!")
    else:
        print(f"CHANGE THE CALCULATION OF GAMMA: {gamma:2f}")



    logger.info(f"QUBO matrix constructed: {len(Q)} nonzero entries, {n_filtered} vehicles. Time elapsed: {time.time() - qubo_start:.2f}s")
    logger.info(f"Total QUBO matrix function time: {time.time() - start_time:.2f}s")
    return dict(Q), vehicle_ids_filtered, affected_edges_df, lambda_penalty
