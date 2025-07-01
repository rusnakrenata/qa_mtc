import logging
import pandas as pd
import numpy as np
from typing import List, Any

logger = logging.getLogger(__name__)

def normalize_congestion_weights(
    weights_df: pd.DataFrame,
    n: int,
    t: int,
    vehicle_ids: List[Any]
) -> List[List[List[List[float]]]]:
    """
    Converts a DataFrame of congestion weights into a normalized 4D matrix w[i][j][k1][k2].
    Normalization is done over all weights into the [0, 1] range.

    Args:
        weights_df: DataFrame with columns ['vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'weighted_congestion_score']
        n: Number of vehicles
        t: Number of route alternatives per vehicle
        vehicle_ids: List of vehicle IDs

    Returns:
        w: 4D list of normalized congestion weights
    """
    vehicle_ids_index = {vid: idx for idx, vid in enumerate(vehicle_ids)}
    w = np.zeros((n, n, t, t), dtype=np.float64)

    # Map vehicle IDs to indices
    weights_df = weights_df.copy()
    weights_df['i'] = weights_df['vehicle1'].map(vehicle_ids_index)  # type: ignore
    weights_df['j'] = weights_df['vehicle2'].map(vehicle_ids_index)  # type: ignore
    weights_df['k1'] = weights_df['vehicle1_route'].astype(int) - 1
    weights_df['k2'] = weights_df['vehicle2_route'].astype(int) - 1

    # Filter valid rows
    valid = (
        weights_df['i'].notnull() & weights_df['j'].notnull() &
        (weights_df['k1'] >= 0) & (weights_df['k1'] < t) &
        (weights_df['k2'] >= 0) & (weights_df['k2'] < t)
    )
    weights_df = weights_df.loc[valid].copy()

    # Assign scores
    idx = (weights_df['i'].astype(int), weights_df['j'].astype(int),
           weights_df['k1'].astype(int), weights_df['k2'].astype(int))
    w[idx] = weights_df['weighted_congestion_score'].astype(float)
    # Symmetric assignment
    idx_sym = (weights_df['j'].astype(int), weights_df['i'].astype(int),
               weights_df['k2'].astype(int), weights_df['k1'].astype(int))
    w[idx_sym] = weights_df['weighted_congestion_score'].astype(float)

    # Normalize nonzero values
    nonzero = w > 0
    if np.any(nonzero):
        min_w = w[nonzero].min()
        max_w = w[nonzero].max()
        scale = max_w - min_w + 1e-9
        w[nonzero] = (w[nonzero] - min_w) / scale
    else:
        min_w = max_w = 0.0

    nonzero_count = np.count_nonzero(w)
    logger.info(f"|i| = {n}, |j| = {n}, |k1| = {t}, |k2| = {t}")
    logger.info(f"min_w = {min_w:.6f}, max_w = {max_w:.6f}, non-zero weights = {nonzero_count}")

    return w.tolist()
