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
                invalid_pairs.add((vid, k))
    return invalid_pairs

def penalized_congestion_weights(
    weights_df: pd.DataFrame,
    dp_df: pd.DataFrame,
    n: int,
    t: int,
    vehicle_ids: List[Any],
    vehicle_routes_df: pd.DataFrame
) -> Tuple[List[List[List[List[float]]]], float, np.ndarray]:
    """
    Constructs the 4D congestion weight matrix with strong penalization for non-existent routes.
    Args:
        weights_df: DataFrame with columns ['vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'weighted_congestion_score']
        n_vehicles: Number of vehicles
        t: Number of route alternatives per vehicle
        vehicle_ids: List of vehicle IDs
        vehicle_routes_df: DataFrame with columns ['vehicle_id', 'route_id']
    Returns:
        w: 4D list of congestion weights
        w_max: Maximum weight value
    """
    # 1. Find invalid (vehicle, route) pairs
    invalid_pairs = get_invalid_vehicle_route_pairs(vehicle_routes_df,  t)

    # 2. Calculate max weight
    w_max = float(weights_df['weighted_congestion_score'].max()) if not weights_df.empty else 1.0

    # 3. Build lookup for weights
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

    # 4. Build lookup for valid (vehicle, route)
    valid_pairs = set(zip(vehicle_routes_df['vehicle_id'], vehicle_routes_df['route_id']))

    # Map vehicle and route IDs to their indices
    vehicle_id_to_idx = {int(v): i for i, v in enumerate(vehicle_ids)}
    route_ids = sorted(vehicle_routes_df['route_id'].unique())
    route_id_to_idx = {int(r): k for k, r in enumerate(route_ids)}

    print(vehicle_id_to_idx)

    # Initialize penalty matrix
    penalty_matrix = np.zeros((len(vehicle_ids), len(route_ids)))
    
    dp_df = pd.DataFrame(dp_df[dp_df['vehicle'].isin(vehicle_ids)])


    # Fill penalty matrix
    for _, row in dp_df.iterrows():
        i = vehicle_id_to_idx[int(row['vehicle'])]
        k = route_id_to_idx[int(row['route'])]
        penalty_matrix[i, k] = row['penalty']

    num_vehicles = len(vehicle_ids)

    # 5. Construct w
    w = np.zeros((n, n, t, t), dtype=np.float64)
    for i, vi in enumerate(vehicle_ids):
        for j, vj in enumerate(vehicle_ids):
            for k1 in range(t):
                for k2 in range(t):
                    key = (i, j, k1, k2)
                    pair1 = (vi, k1 + 1)
                    pair2 = (vj, k2 + 1)
                    w[i, j, k1, k2] = (penalty_matrix[i, k1] + penalty_matrix[j, k2])/num_vehicles
                    if key in weights_lookup:
                        w[i, j, k1, k2] += weights_lookup[key]
                    elif (pair1 in valid_pairs) and (pair2 in valid_pairs):
                        w[i, j, k1, k2] += 0.0 
                    elif (pair1 in invalid_pairs) or (pair2 in invalid_pairs):
                        w[i, j, k1, k2] += 0.0
    '''
    nonzero = (w != 0) 
    if np.any(nonzero):
        min_w = w[nonzero].min()
        max_w = w[nonzero].max()
        scale = max_w - min_w + 1e-9
        #w[nonzero] = (w[nonzero] - min_w) / scale
        #w[nonzero] = np.round(w[nonzero], 7)
        w_max = 1.0
    else:
        w_max = 0.0
    '''
    
    # Penalty values (R) remain as R, not normalized
    # logger.info(f"Normalized weights: min={min_w if np.any(nonzero) else 0.0}, max={w_max if np.any(nonzero) else 0.0}, invalid pairs: {len(invalid_pairs)}")
    return w.tolist(), w_max, penalty_matrix
 
