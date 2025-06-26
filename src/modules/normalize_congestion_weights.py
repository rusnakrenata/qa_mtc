import logging
import pandas as pd
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
    #print("Vehicle ids:" , vehicle_ids)
    #print("Vehicle ids index:" , vehicle_ids_index)
    w = [[[[0.0 for _ in range(t)] for _ in range(t)] for _ in range(n)] for _ in range(n)]
    values = []

    for _, row in weights_df.iterrows():
        i = vehicle_ids_index.get(row['vehicle1'])
        j = vehicle_ids_index.get(row['vehicle2'])
        k1 = int(row['vehicle1_route']) - 1
        k2 = int(row['vehicle2_route']) - 1

        if i is None or j is None or not (0 <= k1 < t) or not (0 <= k2 < t):
            continue

        score = row['weighted_congestion_score']
        #print("i,j,k1,k2,score: ", i,j,k1,k2, score)
        values.append(score)
        w[i][j][k1][k2] = float(score)
        w[j][i][k2][k1] = float(score)  # symmetric

    if values:
        min_w = min(values)
        max_w = max(values)
        scale = max_w - min_w + 1e-9
        for i in range(n):
            for j in range(n):
                for k1 in range(t):
                    for k2 in range(t):
                        if w[i][j][k1][k2] > 0:
                            w[i][j][k1][k2] = (w[i][j][k1][k2] - min_w) / scale
                        else:
                            w[i][j][k1][k2] = 0.0
    else:
        min_w = max_w = 0.0

    nonzero_count = sum(
        1 for i in range(n) for j in range(n) for k1 in range(t) for k2 in range(t)
        if w[i][j][k1][k2] > 0
    )

    logger.info(f"|i| = {n}, |j| = {n}, |k1| = {t}, |k2| = {t}")
    logger.info(f"min_w = {min_w:.6f}, max_w = {max_w:.6f}, non-zero weights = {nonzero_count}")

    return w
