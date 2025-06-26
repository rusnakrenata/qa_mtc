import pandas as pd
from sqlalchemy import text as sa_text
import logging
from typing import Any, List

logger = logging.getLogger(__name__)

def post_qa_congestion(
    session: Any,
    run_config_id: int,
    iteration_id: int,
    all_vehicle_ids: List[Any],
    optimized_vehicle_ids: List[Any],
    qa_assignment: List[int],
    method: str = "duration"
) -> pd.DataFrame:
    """
    Recomputes congestion based on the QA-selected vehicle-route assignments and shortest routes for non-optimized vehicles.
    Args:
        session: SQLAlchemy session
        run_config_id: ID of the run configuration
        iteration_id: Iteration number
        all_vehicle_ids: List of all vehicle IDs in the simulation
        optimized_vehicle_ids: List of vehicle IDs used in QUBO/QA
        qa_assignment: List of selected route indices (0-based) for optimized vehicles
        method: 'distance' or 'duration' for non-optimized vehicles
    Returns:
        DataFrame with columns ['edge_id', 'congestion_score']
    """
    try:
        # Step 1: Build vehicle-route assignment for all vehicles
        vehicle_route_pairs = []

        # For optimized vehicles, use QA assignment
        for idx, vehicle_id in enumerate(optimized_vehicle_ids):
            route_id = int(qa_assignment[idx]) + 1  # assuming assignment is 0-based
            vehicle_route_pairs.append((vehicle_id, route_id))

        # For non-optimized vehicles, assign by shortest (distance/duration)
        non_optimized = set(all_vehicle_ids) - set(optimized_vehicle_ids)
        if non_optimized:
            sql = sa_text(f"""
                SELECT vehicle_id, route_id
                FROM vehicle_routes
                WHERE run_configs_id = :run_config_id AND iteration_id = :iteration_id
                AND {method} = (
                    SELECT MIN({method}) FROM vehicle_routes
                    WHERE run_configs_id = :run_config_id AND iteration_id = :iteration_id AND vehicle_id = vehicle_routes.vehicle_id
                )
            """)
            result = session.execute(sql, {
                'run_config_id': run_config_id,
                'iteration_id': iteration_id
            })
            for row in result.fetchall():
                if row.vehicle_id in non_optimized:
                    vehicle_route_pairs.append((row.vehicle_id, row.route_id))

        # Step 2: Create temporary table to hold selected routes
        session.execute(sa_text("DROP TEMPORARY TABLE IF EXISTS temp_selected_routes"))
        session.execute(sa_text("""
            CREATE TEMPORARY TABLE temp_selected_routes (
                vehicle_id BIGINT,
                route_id INT
            )
        """))

        # Step 3: Insert vehicle-route pairs
        for vehicle_id, route_id in vehicle_route_pairs:
            session.execute(sa_text("""
                INSERT INTO temp_selected_routes (vehicle_id, route_id) VALUES (:vehicle_id, :route_id)
            """), {'vehicle_id': vehicle_id, 'route_id': route_id})
        session.commit()

        # Step 4: Recompute congestion from route_points matching the assignment
        result = session.execute(sa_text("""
            SELECT edge_id, sum(congestion_score) as congestion_score
            FROM 
            (
            SELECT distinct edge_id, vehicle1 as vehicle, congestion_score
            FROM congestion_map cm 
            INNER JOIN temp_selected_routes sr
            ON cm.vehicle1 = sr.vehicle_id AND cm.vehicle1_route = sr.route_id
            WHERE cm.run_configs_id = :run_config_id AND cm.iteration_id = :iteration_id
            UNION ALL
            SELECT distinct edge_id, vehicle2 as vehicle, congestion_score
            FROM congestion_map cm 
            INNER JOIN temp_selected_routes sr
            ON cm.vehicle2 = sr.vehicle_id AND cm.vehicle2_route = sr.route_id
            WHERE cm.run_configs_id = :run_config_id AND cm.iteration_id = :iteration_id
            ) a
            group by edge_id
        """), {
            'run_config_id': run_config_id,
            'iteration_id': iteration_id
        })
        logger.info(f"Recomputed QA congestion for run_config_id={run_config_id}, iteration_id={iteration_id}.")
        return pd.DataFrame(list(result.fetchall()), columns=pd.Index(['edge_id', 'congestion_score']))
    except Exception as e:
        logger.error(f"Error in post_qa_congestion: {e}", exc_info=True)
        raise e
        
