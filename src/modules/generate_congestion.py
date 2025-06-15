import pandas as pd
from sqlalchemy import text as sa_text

def generate_congestion(session, CongestionMap, run_config_id, iteration_id, dist_thresh, speed_diff_thresh):
    # Delete previous congestion data for the iteration
    session.query(CongestionMap).filter_by(iteration_id=iteration_id).delete()
    session.commit()

    # Set MySQL user variables
    session.execute(sa_text("SET @dist_thresh := :dist_thresh"), {'dist_thresh': dist_thresh})
    session.execute(sa_text("SET @speed_diff_thresh := :speed_diff_thresh"), {'speed_diff_thresh': speed_diff_thresh})
    session.execute(sa_text("SET @iteration := :iteration_id"), {'iteration_id': iteration_id})
    session.execute(sa_text("SET @run_configs_id := :run_configs_id"), {'run_configs_id': run_config_id})

    # Execute SQL query
    result = session.execute(sa_text("""
        SELECT
            edge_id,
            vehicle1, 
            vehicle1_route,
            vehicle2, 
            vehicle2_route,
            SUM(CASE 
                WHEN distance < @dist_thresh AND speed_diff < @speed_diff_thresh THEN 
                    (1 / ((1 + distance) * (1 + speed_diff)))
                ELSE 0
            END) AS weighted_congestion_score
        FROM (
            SELECT
                a.edge_id,
                a.vehicle_id as vehicle1,
                b.vehicle_id as vehicle2,
                a.route_id as vehicle1_route,
                b.route_id as vehicle2_route,
                6371 * 2 * ASIN(SQRT(
                    POW(SIN(RADIANS(b.lat - a.lat) / 2), 2) +
                    COS(RADIANS(a.lat)) * COS(RADIANS(b.lat)) *
                    POW(SIN(RADIANS(b.lon - a.lon) / 2), 2)
                )) * 1000 AS distance, #because result of Harvesine formula is in km
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
        GROUP BY edge_id;
    """), {
        'dist_thresh': dist_thresh,
        'speed_diff_thresh': speed_diff_thresh,
        'iteration_id': iteration_id,
        'run_configs_id': run_config_id
    })

    congestion_data = result.fetchall()

    for row in congestion_data:
        congestion_map = CongestionMap(
            run_configs_id=run_config_id,
            iteration_id=iteration_id,
            edge_id=row.edge_id,
            vehicle1 = row.vehicle1,
            vehicle2 = row.vehicle2,
            vehicle1_route = row.vehicle1_route,
            vehicle2_route = row.vehicle2_route,
            congestion_score=row.weighted_congestion_score
        )
        session.add(congestion_map)

    session.commit()

    # Return results as DataFrame
    return pd.DataFrame(congestion_data, columns=['edge_id', 'vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'congestion_score'])
