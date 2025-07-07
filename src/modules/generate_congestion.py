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

def process_group(group_df, dist_thresh, speed_diff_thresh):
    if len(group_df) < 2:
        return pd.DataFrame()

    merged = group_df.merge(group_df, on=['edge_id', 'time', 'cardinal'], suffixes=('_a', '_b'))
    merged = merged[merged['vehicle_id_a'] < merged['vehicle_id_b']]
    if merged.empty:
        return pd.DataFrame()

    merged['distance'] = haversine_np(
        merged['lat_a'].values, merged['lon_a'].values,
        merged['lat_b'].values, merged['lon_b'].values
    )
    merged['speed_diff'] = (merged['speed_a'] - merged['speed_b']).abs()

    merged['congestion_score'] = np.where(
        (merged['distance'] < dist_thresh) & (merged['speed_diff'] < speed_diff_thresh),
        1 / ((1 + merged['distance']) * (1 + merged['speed_diff'])),
        0
    )

    filtered = merged[merged['congestion_score'] > 0]
    if filtered.empty:
        return pd.DataFrame()

    return filtered[[
        'edge_id',
        'vehicle_id_a', 'route_id_a',
        'vehicle_id_b', 'route_id_b',
        'congestion_score'
    ]]

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

