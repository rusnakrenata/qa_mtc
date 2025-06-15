import pandas as pd
from sqlalchemy import text as sa_text

def post_qa_congestion(session, run_config_id, iteration_id):
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
        SELECT edge_id, sum(congestion_score) as congestion_score
        FROM 
        (
        SELECT distinct edge_id, vehicle1 as vehicle, congestion_score
        FROM congestion_map cm 
        INNER JOIN temp_selected_routes sr
        ON CONCAT(cm.vehicle1,vehicle1_route) IN (SELECT CONCAT(vehicle_id,route_id) FROM selected_routes)
        WHERE run_configs_id = :run_config_id AND iteration_id = :iteration_id
        UNION ALL
        SELECT distinct edge_id, vehicle2 as vehicle, congestion_score
        FROM congestion_map cm 
        INNER JOIN temp_selected_routes sr
        ON CONCAT(cm.vehicle2,vehicle2_route) IN (SELECT CONCAT(vehicle_id,route_id) FROM selected_routes)
        WHERE run_configs_id = :un_config_id AND iteration_id = :iteration_id
        ) a
        group by edge_id
    """), {
        'run_config_id': run_config_id,
        'iteration_id': iteration_id
    })

    return pd.DataFrame(result.fetchall(), columns=['edge_id', 'congestion_score'])
