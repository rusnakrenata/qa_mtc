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


def process_group(group_df, time_step):
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

            distance_factor = 4.0

            score = np.maximum(
                (avg_speed - distance / distance_factor) / avg_speed,
                0
            )
            
            max_score = score * time_step

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
    time_step: int = 10,
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
            futures = [executor.submit(process_group, group.copy(), time_step) for group in group_list]
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

