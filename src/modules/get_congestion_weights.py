from queue import Empty
import pandas as pd
import sqlalchemy as sa
import logging
from typing import Any
import itertools
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from datetime import datetime

logger = logging.getLogger(__name__)

def compute_vehicle_pair_rows(args):
    v1, v2, df1, df2 = args
    rows = []
    for row1 in df1.itertuples(index=False):
        for row2 in df2.itertuples(index=False):
            rows.append({
                'vehicle_id_1': v1,
                'vehicle_id_2': v2,
                'route_id_1': row1.route_id,
                'route_id_2': row2.route_id,
                'scaling_factor': 1 + (row1.penalty + row2.penalty) / 2
            })
    return rows

def pair_generator(routes_by_vehicle, vehicle_ids):
    for v1, v2 in itertools.combinations(vehicle_ids, 2):
        yield (v1, v2, routes_by_vehicle[v1], routes_by_vehicle[v2])

'''
def get_congestion_weights(
    session: Any,
    run_configs_id: int,
    iteration_id: int,
    method: str = 'duration',
    base_penalty: float = 0.1
) -> pd.DataFrame:
    """
    Computes weighted congestion scores based on both congestion and relative duration penalty.

    Args:
        session: SQLAlchemy session
        run_configs_id: Run configuration ID
        iteration_id: Iteration ID
        method: Metric for shortest route ('duration', 'distance', 'duration_in_traffic')
        base_penalty: Default congestion penalty if no data exists

    Returns:
        DataFrame with: vehicle1, vehicle2, vehicle1_route, vehicle2_route, weighted_congestion_score
    """
    valid_methods = {'duration', 'distance', 'duration_in_traffic'}
    if method not in valid_methods:
        raise ValueError(f"Invalid method: '{method}'. Must be one of {valid_methods}.")

    try:
        # Load vehicle routes
        routes_query = sa.text(f"""
            SELECT vehicle_id, route_id, {method} AS metric
            FROM vehicle_routes
            WHERE run_configs_id = :run_configs_id AND iteration_id = :iteration_id
        """)
        routes_df = pd.DataFrame(
            session.execute(routes_query, {
                'run_configs_id': run_configs_id,
                'iteration_id': iteration_id
            }).fetchall(),
            columns=pd.Index(['vehicle_id', 'route_id', 'metric'])
        )

        if routes_df.empty:
            logger.warning("No vehicle_routes data found.")
            return pd.DataFrame(columns=pd.Index([
                'vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'weighted_congestion_score'
             ]))

        # Compute min duration (or metric) per vehicle
        routes_df['min_metric'] = routes_df.groupby('vehicle_id')['metric'].transform('min')
        routes_df['penalty'] = routes_df['metric'] - routes_df['min_metric']


        # Pre-grouped routes for efficiency
        routes_by_vehicle = dict(tuple(routes_df.groupby('vehicle_id')))
        vehicle_ids = list(routes_by_vehicle.keys())
        print('Vehicle_ids:', datetime.now())

        # Prepare arguments for each combination
        pair_rows = []
        with ProcessPoolExecutor(max_workers=min(16, multiprocessing.cpu_count())) as executor:
            for result in executor.map(compute_vehicle_pair_rows, pair_generator(routes_by_vehicle, vehicle_ids), chunksize=500):
                pair_rows.extend(result)

        pairs = pd.DataFrame(pair_rows)
        print('Pairs calculated', datetime.now())

        # Load congestion_map
        cong_query = sa.text("""
            SELECT vehicle1, vehicle2, vehicle1_route, vehicle2_route, SUM(congestion_score/2) as congestion_score
            FROM trafficOptimization.congestion_map
            WHERE run_configs_id = :run_configs_id AND iteration_id = :iteration_id
            GROUP BY vehicle1, vehicle2, vehicle1_route, vehicle2_route
        """)
        cong_df = pd.DataFrame(session.execute(cong_query, {
            'run_configs_id': run_configs_id,
            'iteration_id': iteration_id
        }).fetchall(), columns=pd.Index(['vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'congestion_score']))

        print('Cong df calculated')
        # Merge with route pairs
        merged = pairs.merge(
            cong_df,
            left_on=['vehicle_id_1', 'vehicle_id_2', 'route_id_1', 'route_id_2'],
            right_on=['vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route'],
            how='left'
        )
        print('Cong merged', datetime.now())

        merged['congestion_score'] = merged['congestion_score'].fillna(base_penalty)
        merged['weighted_congestion_score'] = merged['congestion_score'] * merged['scaling_factor']


        print('Before result', datetime.now())
        # Final output
        result_df = merged[['vehicle_id_1', 'vehicle_id_2', 'route_id_1', 'route_id_2', 'weighted_congestion_score']]
        result_df.columns = ['vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'weighted_congestion_score']

        logger.info(f"Computed {len(result_df)} congestion weights using method='{method}'")
        return pd.DataFrame(result_df)

    except Exception as e:
        session.rollback()
        logger.error(f"Error computing congestion weights: {e}", exc_info=True)
        return pd.DataFrame(columns=pd.Index([
        'vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'weighted_congestion_score'
    ]))
'''







def get_congestion_weights(
    session: Any,
    run_configs_id: int,
    iteration_id: int
    ):
    """
    Fetches pairwise vehicle congestion weights from SQL query and returns as a DataFrame.

    Args:
        session: SQLAlchemy session
        run_configs_id: ID of the run config
        iteration_id: Iteration number

    Returns:
        weights_df: DataFrame with columns:
            ['vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'weighted_congestion_score']
    """
    try:
        # Set parameters
        session.execute(sa.text("SET @iteration := :iteration_id"), {'iteration_id': iteration_id})
        session.execute(sa.text("SET @run_configs_id := :run_configs_id"), {'run_configs_id': run_configs_id})
        # Query
        sql = sa.text("""


            SELECT
                vehicle1,
                vehicle2,
                vehicle1_route,
                vehicle2_route,
                sum(congestion_score/2) as weighted_congestion_score
            FROM trafficOptimization.congestion_map            
                WHERE iteration_id = @iteration
                AND run_configs_id = @run_configs_id
        group by vehicle1, vehicle2, vehicle1_route, vehicle2_route
        """)
        result = session.execute(sql)
        weights_df = pd.DataFrame(result.fetchall(), columns=pd.Index([
            'vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'weighted_congestion_score'
        ]))
        logger.info(f"Fetched {len(weights_df)} congestion weight records for run_config_id={run_configs_id}, iteration_id={iteration_id}.")


        sql2 = sa.text("""

        WITH routes_with_min AS (
                SELECT vehicle_id,
                    MIN(duration)  AS min_duration
                FROM vehicle_routes
                WHERE run_configs_id = @run_configs_id AND iteration_id = @iteration
                group by vehicle_id
            )
        Select
            a.vehicle_id as vehicle,
            a.route_id as route,
            duration - min_duration as penalty
        FROM vehicle_routes as a
            join routes_with_min as b on a.vehicle_id=b.vehicle_id
        WHERE run_configs_id = @run_configs_id AND iteration_id = @iteration

        """
        )
        result = session.execute(sql2)
        duration_penalty_df = pd.DataFrame(result.fetchall(), columns=pd.Index([
            'vehicle', 'route', 'penalty'
        ]))


        return weights_df,duration_penalty_df
    except Exception as e:
        session.rollback()
        logger.error(f"Error fetching congestion weights: {e}", exc_info=True)
        return pd.DataFrame({col: pd.Series(dtype='float64') for col in ['vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'weighted_congestion_score']})
