from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
import logging
import pandas as pd
import numpy as np
from congestion_weights import congestion_weights
from filter_routes_for_qubo import *
import time
import math
import random

logger = logging.getLogger(__name__)

def qubo_matrix(
    t: int,
    w_df: pd.DataFrame,
    pd_df: pd.DataFrame,
    vehicle_ids_filtered: List[int],
    vehicle_routes_df: pd.DataFrame,
    lambda_strategy: str = "normalized",
) -> Tuple[Dict[Tuple[int, int], float], List[Any], pd.DataFrame, float, float, float]:
    """
    Constructs the QUBO dictionary for the traffic assignment problem with 4D weights.
    Args:
        t: Number of route alternatives per vehicle
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
    n_filtered = len(vehicle_ids_filtered)
    logger.info(f"Vehicle filtering complete. {n_filtered} vehicles selected. Time elapsed: {time.time() - start_time:.2f}s")

    logger.info("Computing congestion weights for filtered vehicles...")
    weights_start = time.time()
    w, max_w = congestion_weights(w_df, n_filtered, t, vehicle_ids_filtered, vehicle_routes_df)
    logger.info(f"Using max_weight strategy max_w={max_w}")
    logger.info(f"Congestion weights computed. Time elapsed: {time.time() - weights_start:.2f}s")

    pd_df = pd.DataFrame(pd_df[pd_df['vehicle'].isin(vehicle_ids_filtered)])

    # Map vehicle and route IDs to their indices
    vehicle_id_to_idx = {int(v): i for i, v in enumerate(vehicle_ids_filtered)}
    route_ids = sorted(vehicle_routes_df['route_id'].unique())
    route_id_to_idx = {int(r): k for k, r in enumerate(route_ids)}

    #print(vehicle_id_to_idx)

    # Initialize penalty matrix
    penalty_matrix = np.zeros((len(vehicle_ids_filtered), len(route_ids)))
    

    # Fill penalty matrix
    for _, row in pd_df.iterrows():
        i = vehicle_id_to_idx[int(row['vehicle'])]
        k = route_id_to_idx[int(row['route'])]
        penalty_matrix[i, k] = row['penalty']

    # Transform penalty matrix to list penalties
    penalties = []
    for i in range(n_filtered):
        for k in range(2):
            penalties.append(penalty_matrix[i, k])
 
    logger.info("Constructing QUBO objective terms...")
    qubo_start = time.time()
    Q = defaultdict(float)
    for i in range(n_filtered):
        for j in range(i + 1, n_filtered):
            for k1 in range(t):
                for k2 in range(t):
                    q1 = i * t + k1
                    q2 = j * t + k2
                    Q[(q1, q2)] += w[i][j][k1][k2]   #+random.uniform(0, 50.0)  # Add small random noise to avoid degeneracy
    logger.info(f"QUBO objective constructed: {len(Q)} terms. Time: {time.time() - qubo_start:.2f}s")
    
    # Build set of valid (vehicle, route) pairs
    valid_pairs = set(zip(vehicle_routes_df['vehicle_id'], vehicle_routes_df['route_id']))

    # Step 1: Compute all dynamic penalties and store them
    dynamic_penalties = []
    q_indices = []
    q_indices_adj = []
    not_real_routes_indices = []
    for i in range(n_filtered):
        vi = vehicle_ids_filtered[i]
        # Go through ALL possible routes (1 to t)
        for k in range(1, t + 1):
            q = i * t + (k - 1)
            if (vi, k) in valid_pairs:
                # Real route: use dynamic penalty
                #row_sum = sum(Q.get((q, j), 0) for j in range(n_filtered * t))
                #col_sum = sum(Q.get((idx, q), 0) for idx in range(n_filtered * t))

                # Get all row values for variable q
                row_values = [Q.get((q, j), 0) for j in range(n_filtered * t)]
                # Get all column values for variable q  
                col_values = [Q.get((idx, q), 0) for idx in range(n_filtered * t)]

                # Calculate sum of max of pairs for row values
                row_sum = sum(max(row_values[i], row_values[i+1]) for i in range(0, len(row_values)-1, 2))

                # Calculate sum of max of pairs for column values
                col_sum = sum(max(col_values[i], col_values[i+1]) for i in range(0, len(col_values)-1, 2))

                lambda_penalty_dynamic = ( row_sum + col_sum )
                dynamic_penalties.append( lambda_penalty_dynamic )
                q_indices.append(q)
            else:
                # Non-real route: use high penalty
                q_indices.append(q)
                not_real_routes_indices.append(q)

    ### Na diagonále musí byť konštantná hodnota, aby sa zabránilo degenerácii


    # Step 2: Find the maximum penalty for real routes
    lambda_penalty = max(dynamic_penalties) 
    logger.info(f"lambda_penalty={lambda_penalty}")

    # Step 3: Apply the max penalty to all diagonal elements
    for q in q_indices:
        Q[(q, q)] += lambda_penalty * (1 - 2)+penalties[q] if q not in not_real_routes_indices else lambda_penalty 

    for i in range(n_filtered):
        # Go through ALL possible routes (1 to t)
        for k1 in range(1, t + 1):
            q1 = i * t + (k1 - 1)
            for k2 in range(1, t ):
                if k1 != t:
                    q2 = q1 +k2
                else:
                    q2 = q1 -1
                Q[(q1, q2)] += lambda_penalty#abs(Q.get((q1, q1), 0)) #lambda_penalty 
    #print(Q)
    logger.info(f"QUBO matrix constructed: {len(Q)} nonzero entries, {n_filtered} vehicles. Time elapsed: {time.time() - qubo_start:.2f}s")
    logger.info(f"Total QUBO matrix function time: {time.time() - start_time:.2f}s")
    return dict(Q), t, lambda_penalty