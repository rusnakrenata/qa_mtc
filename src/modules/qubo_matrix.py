from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
import logging
import pandas as pd
from normalize_congestion_weights import normalize_congestion_weights
from congestion_weights import congestion_weights
from filter_routes_for_qubo import filter_vehicles_by_congested_edges_and_limit
import time

logger = logging.getLogger(__name__)

def qubo_matrix(
    n: int,
    t: int,
    congestion_df: pd.DataFrame,
    w_df: pd.DataFrame,
    vehicle_ids: List[Any],
    lambda_strategy: str = "normalized",
    fixed_lambda: float = 1.0,
    filtering_percentage: Optional[float] = None,
    max_qubo_size: Optional[int] = None
) -> Tuple[Dict[Tuple[int, int], float], List[Any]]:
    """
    Constructs the QUBO dictionary for the traffic assignment problem with 4D weights.

    Args:
        n: Total number of vehicles
        t: Number of route alternatives per vehicle
        congestion_df: Congestion DataFrame (for filtering)
        w_df: Congestion weights DataFrame
        vehicle_ids: Full list of vehicle IDs
        lambda_strategy: "normalized" or "max_weight"
        fixed_lambda: λ if using normalized strategy
        filtering_percentage: Float (e.g., 0.1) to select a subset of vehicles
        max_qubo_size: Optional upper bound on number of QUBO variables

    Returns:
        Q: QUBO matrix as a dictionary {(q1, q2): value}
        vehicle_ids_filtered: The filtered list of vehicle IDs used in QUBO
    """
    start_time = time.time()
    logger.info("Starting QUBO vehicle filtering...")
    if filtering_percentage is not None or max_qubo_size is not None:
        vehicle_count = len(vehicle_ids)
        max_vehicles = int(vehicle_count * (filtering_percentage or 1.0))
        if max_qubo_size is not None:
            max_vehicles = min(max_vehicles, max_qubo_size // t)
        vehicle_ids_filtered = filter_vehicles_by_congested_edges_and_limit(
            congestion_df,
            max_vehicles
        )
    else:
        vehicle_ids_filtered = vehicle_ids
    n_filtered = len(vehicle_ids_filtered)
    logger.info(f"Vehicle filtering complete. {n_filtered} vehicles selected. Time elapsed: {time.time() - start_time:.2f}s")

    logger.info("Computing congestion weights for filtered vehicles...")
    weights_start = time.time()
    if lambda_strategy == "normalized":
        w = normalize_congestion_weights(w_df, n_filtered, t, vehicle_ids_filtered)
        lambda_penalty = fixed_lambda
        logger.info(f"Using normalized weights with fixed lambda_penalty={lambda_penalty}")
    else:
        w, max_w = congestion_weights(w_df, n_filtered, t, vehicle_ids_filtered)
        lambda_penalty = max_w
        logger.info(f"Using max_weight strategy with lambda_penalty={lambda_penalty:.6f}")
    logger.info(f"Congestion weights computed. Time elapsed: {time.time() - weights_start:.2f}s")

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

    for i in range(n_filtered):
        for k in range(t):
            q = i * t + k
            Q[(q, q)] += lambda_penalty * (1 - 2)  # x^2 - 2x
        for k1 in range(t):
            for k2 in range(k1 + 1, t):
                q1 = i * t + k1
                q2 = i * t + k2
                Q[(q1, q2)] += 2 * lambda_penalty

    logger.info(f"QUBO matrix constructed: {len(Q)} nonzero entries, {n_filtered} vehicles. Time elapsed: {time.time() - qubo_start:.2f}s")
    logger.info(f"Total QUBO matrix function time: {time.time() - start_time:.2f}s")
    return dict(Q), vehicle_ids_filtered
