import logging
import pandas as pd
import numpy as np
from typing import List, Any, Tuple

logger = logging.getLogger(__name__)

def congestion_weights(
    weights_df: pd.DataFrame,
    n: int,
    t: int,
    vehicle_ids: List[Any]
) -> Tuple[List[List[List[List[float]]]], float]:
    """
    Converts a DataFrame of congestion weights into a 4D matrix w[i][j][k1][k2].
    Used when lambda_strategy = 'max_weight'.

    Args:
        weights_df: DataFrame with columns ['vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'weighted_congestion_score']
        n: Number of vehicles
        t: Number of route alternatives per vehicle
        vehicle_ids: List of vehicle IDs

    Returns:
        w: 4D list of congestion weights
        max_w: Maximum weight value
    """
    vehicle_ids_index = {vid: idx for idx, vid in enumerate(vehicle_ids)}
    w = np.zeros((n, n, t, t), dtype=np.float64)

    weights_df = weights_df.copy()
    weights_df['i'] = weights_df['vehicle1'].map(vehicle_ids_index)  # type: ignore
    weights_df['j'] = weights_df['vehicle2'].map(vehicle_ids_index)  # type: ignore
    weights_df['k1'] = weights_df['vehicle1_route'].astype(int) - 1
    weights_df['k2'] = weights_df['vehicle2_route'].astype(int) - 1

    valid = (
        weights_df['i'].notnull() & weights_df['j'].notnull() &
        (weights_df['k1'] >= 0) & (weights_df['k1'] < t) &
        (weights_df['k2'] >= 0) & (weights_df['k2'] < t)
    )
    weights_df = weights_df.loc[valid].copy()

    idx = (weights_df['i'].astype(int), weights_df['j'].astype(int),
           weights_df['k1'].astype(int), weights_df['k2'].astype(int))
    w[idx] = weights_df['weighted_congestion_score'].astype(float)
    idx_sym = (weights_df['j'].astype(int), weights_df['i'].astype(int),
               weights_df['k2'].astype(int), weights_df['k1'].astype(int))
    w[idx_sym] = weights_df['weighted_congestion_score'].astype(float)

    nonzero = w > 0
    min_w = w[nonzero].min() if np.any(nonzero) else 0.0
    max_w = w[nonzero].max() if np.any(nonzero) else 0.0
    nonzero_count = np.count_nonzero(w)

    logger.info(f"|i| = {n}, |j| = {n}, |k1| = {t}, |k2| = {t}")
    logger.info(f"min_w = {min_w:.6f}, max_w = {max_w:.6f}, non-zero weights = {nonzero_count}")

    return w.tolist(), max_w
