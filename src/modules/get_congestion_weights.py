import pandas as pd
import sqlalchemy as sa

def get_congestion_weights(session, run_configs_id, iteration_id,
                                     dist_thresh=10.0, speed_diff_thresh=5.0):
    """
    Fetches pairwise vehicle congestion weights from SQL query and returns as a DataFrame.

    Args:
        session: SQLAlchemy session
        run_configs_id: ID of the run config
        iteration_id: iteration number
        dist_thresh: max distance (in km) for congestion to be considered
        speed_diff_thresh: max speed difference (in km/h) for congestion

    Returns:
        weights_df: DataFrame with columns:
            ['vehicle_1', 'vehicle_2', 'vehicle_1_route', 'vehicle_2_route', 'weighted_congestion_score']
    """

    # Set parameters
    session.execute(sa.text("SET @dist_thresh := :dist_thresh"), {'dist_thresh': dist_thresh})
    session.execute(sa.text("SET @speed_diff_thresh := :speed_diff_thresh"), {'speed_diff_thresh': speed_diff_thresh})
    session.execute(sa.text("SET @iteration := :iteration_id"), {'iteration_id': iteration_id})
    session.execute(sa.text("SET @run_configs_id := :run_configs_id"), {'run_configs_id': run_configs_id})

    # Query
    sql = sa.text("""
        SELECT
            vehicle_1,
            vehicle_2,
            vehicle_1_route,
            vehicle_2_route,
            SUM(CASE 
                WHEN distance < @dist_thresh AND speed_diff < @speed_diff_thresh THEN 
                    (1 / ((1 + distance) * (1 + speed_diff)))
                ELSE 0
            END) AS weighted_congestion_score
        FROM (
            SELECT
                a.edge_id,
                a.vehicle_id as vehicle_1,
                b.vehicle_id as vehicle_2,
                a.route_id as vehicle_1_route,
                b.route_id as vehicle_2_route,
                6371 * 2 * ASIN(SQRT(
                    POW(SIN(RADIANS(b.lat - a.lat) / 2), 2) +
                    COS(RADIANS(a.lat)) * COS(RADIANS(b.lat)) *
                    POW(SIN(RADIANS(b.lon - a.lon) / 2), 2)
                )) AS distance,
                ABS(a.speed - b.speed) AS speed_diff
            FROM trafficOptimization.route_points a
            JOIN trafficOptimization.route_points b
                ON a.time = b.time
                AND a.edge_id = b.edge_id
                AND a.cardinal = b.cardinal
                AND a.vehicle_id < b.vehicle_id
            WHERE a.iteration_id = @iteration
            AND b.iteration_id = @iteration
            AND a.run_configs_id = @run_configs_id
            AND b.run_configs_id = @run_configs_id
        ) AS pairwise
        GROUP BY vehicle_1, vehicle_2, vehicle_1_route, vehicle_2_route
    """)

    result = session.execute(sql)
    weights_df = pd.DataFrame(result.fetchall(), columns=[
        'vehicle_1', 'vehicle_2', 'vehicle_1_route', 'vehicle_2_route', 'weighted_congestion_score'
    ])

    return weights_df
