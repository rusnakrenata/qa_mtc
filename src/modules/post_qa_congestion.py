import pandas as pd
from sqlalchemy import text as sa_text

def post_qa_congestion(session, run_config_id, iteration_id, dist_thresh=10.0, speed_diff_thresh=2.0):
    """
    Recomputes congestion based on the QA-selected vehicle-route assignments stored in the `qa_results` table.
    
    Parameters:
        session: SQLAlchemy session
        run_config_id: ID of the run configuration
        iteration_id: Iteration number
        dist_thresh: Maximum distance (km) for route overlap to count as congestion
        speed_diff_thresh: Maximum allowed speed difference (km/h)
    
    Returns:
        DataFrame with columns ['edge_id', 'congestion_score']
    """
    # Step 1: Fetch QA result from DB
    qa_result = session.execute(sa_text("""
        SELECT assignment, vehicle_ids
        FROM qa_results
        WHERE run_configs_id = :run_config_id AND iteration_id = :iteration_id
        ORDER BY created_at DESC
        LIMIT 1
    """), {
        'run_config_id': run_config_id,
        'iteration_id': iteration_id
    }).fetchone()

    if not qa_result:
        raise ValueError(f"No QA result found for run_config_id={run_config_id}, iteration_id={iteration_id}")

    assignment = qa_result.assignment
    vehicle_ids = qa_result.vehicle_ids

    # Step 2: Prepare (vehicle_id, route_id) pairs
    vehicle_route_pairs = [(vehicle_ids[i], int(route_id) + 1) for i, route_id in enumerate(assignment)]

    # Step 3: Create temporary table to hold selected routes
    session.execute(sa_text("DROP TEMPORARY TABLE IF EXISTS temp_selected_routes"))
    session.execute(sa_text("""
        CREATE TEMPORARY TABLE temp_selected_routes (
            vehicle_id BIGINT,
            route_id INT
        )
    """))

    # Step 4: Insert vehicle-route pairs
    for vehicle_id, route_id in vehicle_route_pairs:
        session.execute(sa_text("""
            INSERT INTO temp_selected_routes (vehicle_id, route_id) VALUES (:vehicle_id, :route_id)
        """), {'vehicle_id': vehicle_id, 'route_id': route_id})
    session.commit()

    # Step 5: Recompute congestion from route_points matching the QA assignment
    result = session.execute(sa_text("""
        SELECT
            a.edge_id,
            SUM(CASE 
                WHEN distance < :dist_thresh AND speed_diff < :speed_diff_thresh THEN 
                    (1 / ((1 + distance) * (1 + speed_diff)))
                ELSE 0
            END) AS congestion_score
        FROM (
            SELECT
                a.edge_id,
                6371 * 2 * ASIN(SQRT(
                    POW(SIN(RADIANS(b.lat - a.lat) / 2), 2) +
                    COS(RADIANS(a.lat)) * COS(RADIANS(b.lat)) *
                    POW(SIN(RADIANS(b.lon - a.lon) / 2), 2)
                )) AS distance,
                ABS(a.speed - b.speed) AS speed_diff
            FROM route_points a
            JOIN route_points b
                ON a.time = b.time
                AND a.edge_id = b.edge_id
                AND a.cardinal = b.cardinal
                AND a.vehicle_id < b.vehicle_id
            JOIN temp_selected_routes ta ON a.vehicle_id = ta.vehicle_id AND a.route_id = ta.route_id
            JOIN temp_selected_routes tb ON b.vehicle_id = tb.vehicle_id AND b.route_id = tb.route_id
            WHERE a.run_configs_id = :run_config_id
              AND b.run_configs_id = :run_config_id
              AND a.iteration_id = :iteration_id
              AND b.iteration_id = :iteration_id
        ) AS pairwise
        GROUP BY edge_id
    """), {
        'dist_thresh': dist_thresh,
        'speed_diff_thresh': speed_diff_thresh,
        'run_config_id': run_config_id,
        'iteration_id': iteration_id
    })

    return pd.DataFrame(result.fetchall(), columns=['edge_id', 'congestion_score'])
