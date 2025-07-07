import logging
import pandas as pd
import numpy as np
from typing import List, Any, Tuple, Set

logger = logging.getLogger(__name__)

def get_invalid_vehicle_route_pairs(vehicle_routes_df: pd.DataFrame,  t: int) -> Set[Tuple[Any, int]]:
    """
    Returns a set of (vehicle_id, route_id) pairs that are invalid (i.e., route_id does not exist for that vehicle).
    """
    vehicle_ids = vehicle_routes_df['vehicle_id'].unique()
    invalid_pairs = set()
    for vid in vehicle_ids:
        routes = set(vehicle_routes_df[vehicle_routes_df['vehicle_id'] == vid]['route_id'])
        for k in range(1, t + 1):
            if k not in routes:
                invalid_pairs.add((int(vid), int(k)))
    return invalid_pairs

def normalize_congestion_weights(
    weights_df: pd.DataFrame,
    n: int,
    t: int,
    vehicle_ids: List[Any],
    vehicle_routes_df: pd.DataFrame
) -> Tuple[List[List[List[List[float]]]], float]:
    """
    Constructs the 4D normalized congestion weight matrix with strong penalization for non-existent routes.
    All nonzero weights are normalized to [0, 1] and rounded to 4 digits.
    Args:
        weights_df: DataFrame with columns ['vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'weighted_congestion_score']
        n: Number of vehicles
        t: Number of route alternatives per vehicle
        vehicle_ids: List of vehicle IDs
        vehicle_routes_df: DataFrame with columns ['vehicle_id', 'route_id']
    Returns:
        w: 4D list of normalized congestion weights
        w_max: Maximum normalized weight value (1.0 if any nonzero, else 0.0)
    """
    # 1. Find invalid (vehicle, route) pairs
    invalid_pairs = get_invalid_vehicle_route_pairs(vehicle_routes_df, t)
    logger.info(f"Number of invalid pairs: {len(invalid_pairs)}")

    # 2. Build lookup for weights
    vehicle_ids_set = set(int(v) for v in vehicle_ids)
    weights_df = weights_df[
        weights_df['vehicle1'].apply(lambda x: int(x) in vehicle_ids_set) &
        weights_df['vehicle2'].apply(lambda x: int(x) in vehicle_ids_set)
    ]  # type: ignore
    weights_lookup = {}
    for _, row in weights_df.iterrows():
        i = vehicle_ids.index(int(row['vehicle1']))
        j = vehicle_ids.index(int(row['vehicle2']))
        k1 = int(row['vehicle1_route']) - 1
        k2 = int(row['vehicle2_route']) - 1
        weights_lookup[(i, j, k1, k2)] = row['weighted_congestion_score']

    # 3. Build lookup for valid (vehicle, route)
    valid_pairs = set(zip(vehicle_routes_df['vehicle_id'], vehicle_routes_df['route_id']))
    logger.info(f"Number of valid pairs: {len(valid_pairs)}")

    # 4. Construct w (raw, before normalization)
    w = np.zeros((n, n, t, t), dtype=np.float64)
    for i, vi in enumerate(vehicle_ids):
        for j, vj in enumerate(vehicle_ids):
            for k1 in range(t):
                for k2 in range(t):
                    key = (i, j, k1, k2)
                    pair1 = (vi, k1 + 1)
                    pair2 = (vj, k2 + 1)
                    if key in weights_lookup:
                        w[i, j, k1, k2] = weights_lookup[key]
                    elif (pair1 in valid_pairs) and (pair2 in valid_pairs):
                        w[i, j, k1, k2] = 0.0
                    elif (pair1 in invalid_pairs) or (pair2 in invalid_pairs):
                        w[i, j, k1, k2] = 0.0

    # 5. Normalize all nonzero weights to [0, 1] and round to 4 digits
    nonzero = (w != 0) 
    if np.any(nonzero):
        min_w = w[nonzero].min()
        max_w = w[nonzero].max()
        scale = max_w - min_w + 1e-9
        w[nonzero] = (w[nonzero] - min_w) / scale
        w[nonzero] = np.round(w[nonzero], 7)
        w_max = 1.0
    else:
        w_max = 0.0
    # Penalty values (R) remain as R, not normalized
    logger.info(f"Normalized weights: min={min_w if np.any(nonzero) else 0.0}, max={w_max if np.any(nonzero) else 0.0}, invalid pairs: {len(invalid_pairs)}")
    return w.tolist(), w_max
