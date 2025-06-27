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
    CongestionMap: Any,
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
        CongestionMap: SQLAlchemy model for congestion_map table
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
        logger.error(f"Error in generate_congestion: {e}", exc_info=True)
        return pd.DataFrame(columns=pd.Index([
            'edge_id', 'vehicle1', 'vehicle1_route',
            'vehicle2', 'vehicle2_route', 'congestion_score'
        ]))









# import pandas as pd
# from datetime import datetime
# from sqlalchemy import text as sa_text

# def generate_congestion(session, CongestionMap, run_config_id, iteration_id, dist_thresh, speed_diff_thresh):


#     # Step 2: Set user-defined thresholds in MySQL
#     session.execute(sa_text("SET @dist_thresh := :dist_thresh"), {'dist_thresh': dist_thresh})
#     session.execute(sa_text("SET @speed_diff_thresh := :speed_diff_thresh"), {'speed_diff_thresh': speed_diff_thresh})
#     session.execute(sa_text("SET @iteration := :iteration_id"), {'iteration_id': iteration_id})
#     session.execute(sa_text("SET @run_configs_id := :run_configs_id"), {'run_configs_id': run_config_id})

#     print(f"Running congestion calculation SELECT at:", datetime.now())
#     # Step 3: Run congestion calculation query
#     result = session.execute(sa_text("""
#         SELECT
#             edge_id,
#             vehicle1, 
#             vehicle1_route,
#             vehicle2, 
#             vehicle2_route,
#             SUM(CASE 
#                 WHEN distance < @dist_thresh AND speed_diff < @speed_diff_thresh THEN 
#                     (1 / ((1 + distance) * (1 + speed_diff)))
#                 ELSE 0
#             END) AS weighted_congestion_score
#         FROM (
#             SELECT
#                 a.edge_id,
#                 a.vehicle_id as vehicle1,
#                 b.vehicle_id as vehicle2,
#                 a.route_id as vehicle1_route,
#                 b.route_id as vehicle2_route,
#                 6371 * 2 * ASIN(SQRT(
#                     POW(SIN(RADIANS(b.lat - a.lat) / 2), 2) +
#                     COS(RADIANS(a.lat)) * COS(RADIANS(b.lat)) *
#                     POW(SIN(RADIANS(b.lon - a.lon) / 2), 2)
#                 )) * 1000 AS distance,
#                 ABS(a.speed - b.speed) AS speed_diff
#             FROM trafficOptimization.route_points a
#             INNER JOIN trafficOptimization.route_points b
#                 ON a.run_configs_id = @run_configs_id
#                 AND b.run_configs_id = @run_configs_id
#                 AND a.iteration_id = @iteration
#                 AND b.iteration_id = @iteration
#                 AND a.edge_id = b.edge_id
#                 AND a.time = b.time
#                 AND a.cardinal = b.cardinal
#                 AND a.vehicle_id < b.vehicle_id
#                 AND ABS(a.lat - b.lat) < 0.01
#                 AND ABS(a.lon - b.lon) < 0.01
#         ) AS pairwise
#         GROUP BY edge_id, vehicle1, vehicle2, vehicle1_route, vehicle2_route
#     """))

#     t1 = datetime.now()
#     print("Congestion calculation SELECT completed at:", datetime.now())
#     congestion_data = result.fetchall()

#     if not congestion_data:
#         return pd.DataFrame(columns=[
#             'edge_id', 'vehicle1', 'vehicle1_route',
#             'vehicle2', 'vehicle2_route', 'congestion_score'
#         ])

#     print("Convert results into rows for executemany at:", datetime.now())
#     # Step 4: Convert results into rows for executemany
#     insert_rows = [{
#         'run_configs_id': run_config_id,
#         'iteration_id': iteration_id,
#         'edge_id': row.edge_id,
#         'vehicle1': row.vehicle1,
#         'vehicle2': row.vehicle2,
#         'vehicle1_route': row.vehicle1_route,
#         'vehicle2_route': row.vehicle2_route,
#         'congestion_score': row.weighted_congestion_score,
#         'created_at': datetime.now()
#     } for row in congestion_data]

#     print("Insert_rows prepared for executemany at:", datetime.now())
#     # Step 5: Use raw INSERT for fast executemany
#     insert_sql = sa_text("""
#         INSERT INTO congestion_map (
#             run_configs_id, iteration_id, edge_id,
#             vehicle1, vehicle2, vehicle1_route, vehicle2_route, congestion_score, created_at
#         ) VALUES (
#             :run_configs_id, :iteration_id, :edge_id,
#             :vehicle1, :vehicle2, :vehicle1_route, :vehicle2_route, :congestion_score, :created_at
#         )
#     """)

#     session.execute(insert_sql, insert_rows)
#     session.commit()
#     print("Congestion data inserted in:", datetime.now() - t1)

#     # Step 6: Return as DataFrame
#     return pd.DataFrame(congestion_data, columns=[
#         'edge_id', 'vehicle1', 'vehicle1_route',
#         'vehicle2', 'vehicle2_route', 'congestion_score'
#     ])