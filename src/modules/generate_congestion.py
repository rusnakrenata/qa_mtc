import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import text as sa_text
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import logging
from typing import Any

logger = logging.getLogger(__name__)

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def process_group_nondiectional(group_df, dist_thresh, speed_diff_thresh):
    if len(group_df) < 2:
        return pd.DataFrame()

    # Use numpy arrays directly instead of pandas operations
    vehicle_ids = group_df['vehicle_id'].values
    route_ids = group_df['route_id'].values
    lats = group_df['lat'].values
    lons = group_df['lon'].values
    speeds = group_df['speed'].values
    edge_id = group_df['edge_id'].iloc[0]  # Same for all in group
    
    n = len(group_df)
    results = []
    
    # Vectorized pairwise computations
    for i in range(n):
        if i + 1 >= n:
            break
            
        # Vectorized distance calculation for all remaining pairs
        j_indices = np.arange(i + 1, n)
        distances = haversine_np(
            lats[i], lons[i], 
            lats[j_indices], lons[j_indices]
        )
        
        # Vectorized average speed calculation
        avg_speeds = (speeds[i] + speeds[j_indices]) / 2.0
        
        # Vectorized congestion score calculation with multiple distance factors
        distance_factors = np.array([0.5, 1.0, 1.5, 2.0])
        
        # Broadcast for all factor combinations
        distances_broadcast = distances[:, np.newaxis]  # Shape: (n_pairs, 1)
        factors_broadcast = distance_factors[np.newaxis, :]  # Shape: (1, n_factors)
        avg_speeds_broadcast = avg_speeds[:, np.newaxis]  # Shape: (n_pairs, 1)
        
        # Compute scores for all factor-pair combinations
        scores = np.maximum(
            (avg_speeds_broadcast - distances_broadcast / factors_broadcast) / avg_speeds_broadcast,
            0
        )
        
        # Take max across factors for each pair
        max_scores = np.max(scores, axis=1) * dist_thresh
        
        # Filter positive scores
        valid_mask = max_scores > 0
        if not valid_mask.any():
            continue
            
        # Create result arrays
        valid_j_indices = j_indices[valid_mask]
        valid_scores = max_scores[valid_mask]
        
        # Append results in batch
        for j_idx, score in zip(valid_j_indices, valid_scores):
            results.append({
                'edge_id': edge_id,
                'vehicle_id_a': vehicle_ids[i],
                'route_id_a': route_ids[i],
                'vehicle_id_b': vehicle_ids[j_idx],
                'route_id_b': route_ids[j_idx],
                'congestion_score': score
            })
    
    if not results:
        return pd.DataFrame()
        
    return pd.DataFrame(results)


def process_group(group_df, dist_thresh, speed_diff_thresh):
    if len(group_df) < 2:
        return pd.DataFrame()

    vehicle_ids = group_df['vehicle_id'].values
    route_ids = group_df['route_id'].values
    lats = group_df['lat'].values
    lons = group_df['lon'].values
    speeds = group_df['speed'].values
    edge_id = group_df['edge_id'].iloc[0]

    # Get direction vector from cardinal
    cardinal = group_df['cardinal'].iloc[0].upper()
    cardinal_map = {
        'N':  np.array([0, 1]),
        'S':  np.array([0, -1]),
        'E':  np.array([1, 0]),
        'W':  np.array([-1, 0]),
        'NE': np.array([1, 1]),
        'NW': np.array([-1, 1]),
        'SE': np.array([1, -1]),
        'SW': np.array([-1, -1])
    }

    if cardinal not in cardinal_map:
        return pd.DataFrame()

    edge_unit_vec = cardinal_map[cardinal]
    edge_unit_vec = edge_unit_vec / np.linalg.norm(edge_unit_vec)

    # Project all positions onto direction vector
    positions = np.stack([lons, lats], axis=1)
    projections = positions @ edge_unit_vec  # shape (n,)

    n = len(group_df)
    results = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # skip self

            # Only consider congestion if vehicle j is BEHIND vehicle i
            if projections[i] <= projections[j]:
                continue

            distance = haversine_np(lats[i], lons[i], lats[j], lons[j])
            avg_speed = (speeds[i] + speeds[j]) / 2.0

            distance_factors = np.array([0.5, 1.0, 1.5, 2.0])
            scores = np.maximum(
                (avg_speed - distance / distance_factors) / avg_speed,
                0
            )
            max_score = np.max(scores) * dist_thresh

            if max_score > 0:
                results.append({
                    'edge_id': edge_id,
                    'vehicle_id_a': vehicle_ids[i],
                    'route_id_a': route_ids[i],
                    'vehicle_id_b': vehicle_ids[j],
                    'route_id_b': route_ids[j],
                    'congestion_score': max_score
                })

    return pd.DataFrame(results) if results else pd.DataFrame()




def generate_congestion(
    session: Any,
    run_config_id: int,
    iteration_id: int,
    dist_thresh: float,
    speed_diff_thresh: float
) -> pd.DataFrame:
    """
    Compute and store pairwise congestion scores for all vehicle-route pairs in the database.
    Uses parallel processing for efficiency.

    Args:
        session: SQLAlchemy session
        run_config_id: Run configuration ID
        iteration_id: Iteration number
        dist_thresh: Distance threshold (meters)
        speed_diff_thresh: Speed difference threshold (km/h)

    Returns:
        DataFrame with columns [edge_id, vehicle1, vehicle1_route, vehicle2, vehicle2_route, congestion_score]
    """
    try:
        logger.info("Loading route_points from DB at: %s", datetime.now())
        start = datetime.now()
        query = sa_text("""
            SELECT edge_id, vehicle_id, route_id, lat, lon, speed, time, cardinal
            FROM trafficOptimization.route_points
            WHERE run_configs_id = :run_config_id AND iteration_id = :iteration_id
        """)
        df = pd.read_sql_query(query, session.bind, params={
            'run_config_id': run_config_id,
            'iteration_id': iteration_id
        })
        logger.info("Bucketing route_points for spatial-temporal filtering at: %s", datetime.now())
        df['lat_bucket'] = (df['lat'] * 100).astype(int)
        df['lon_bucket'] = (df['lon'] * 100).astype(int)
        df['time_bucket'] = (df['time'] // 10).astype(int)
        df['bucket'] = (
            df['edge_id'].astype(str) + "_" +
            df['cardinal'].astype(str) + "_" +
            df['lat_bucket'].astype(str) + "_" +
            df['lon_bucket'].astype(str) + "_" +
            df['time_bucket'].astype(str)
        )
        group_list = [group for _, group in df.groupby('bucket')]
        logger.info(f"Starting parallel processing of {len(group_list)} buckets at: {datetime.now()}")
        results = []
        with ProcessPoolExecutor(max_workers=min(16, multiprocessing.cpu_count())) as executor:
            futures = [executor.submit(process_group, group.copy(), dist_thresh, speed_diff_thresh) for group in group_list]
            for future in as_completed(futures):
                result = future.result()
                if not result.empty:
                    results.append(result)
        if not results:
            logger.warning("No congestion pairs detected.")
            return pd.DataFrame(columns=pd.Index([
                'edge_id', 'vehicle1', 'vehicle1_route',
                'vehicle2', 'vehicle2_route', 'congestion_score'
            ]))
        logger.info("Aggregating results and preparing insert at: %s", datetime.now())
        all_congestion = pd.concat(results, ignore_index=True)
        grouped = all_congestion.groupby(
            ['edge_id', 'vehicle_id_a', 'vehicle_id_b', 'route_id_a', 'route_id_b']
        )['congestion_score'].sum().reset_index()
        grouped.rename(columns={
            'vehicle_id_a': 'vehicle1',
            'vehicle_id_b': 'vehicle2',
            'route_id_a': 'vehicle1_route',
            'route_id_b': 'vehicle2_route'
        }, inplace=True)
        grouped['run_configs_id'] = run_config_id
        grouped['iteration_id'] = iteration_id
        grouped['created_at'] = datetime.now()
        logger.info("Inserting congestion results into DB at: %s", datetime.now())
        try:
            grouped.to_sql(
                'congestion_map',
                con=session.bind,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=5000
            )
        except Exception as e:
            session.rollback()
            logger.error(f"Insertion failed: {e}", exc_info=True)
            raise
        session.commit()
        logger.info("Congestion calculation and insert completed at: %s", datetime.now())
        logger.info("Total runtime: %s", datetime.now() - start)
        return grouped
    except Exception as e:
        session.rollback()
        logger.error(f"Error in generate_congestion: {e}", exc_info=True)
        return pd.DataFrame(columns=pd.Index([
            'edge_id', 'vehicle1', 'vehicle1_route',
            'vehicle2', 'vehicle2_route', 'congestion_score'
        ]))

