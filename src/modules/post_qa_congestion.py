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
        print("Optimized vehicles: ", optimized_vehicle_ids)

        # For non-optimized vehicles, assign by shortest (distance/duration)
        non_optimized = set(all_vehicle_ids) - set(optimized_vehicle_ids)
        print("Non-optimized vehicles: ", non_optimized)
        if non_optimized:
            sql = sa_text(f"""
                WITH shortest_routes AS (
                    SELECT vehicle_id, MIN({method}) AS min_value
                    FROM vehicle_routes
                    WHERE run_configs_id = :run_config_id AND iteration_id = :iteration_id
                    GROUP BY vehicle_id
                ),
                selected_routes AS (
                    SELECT vr.vehicle_id, max(vr.route_id) as route_id 
                    FROM vehicle_routes vr
                    JOIN shortest_routes sr
                    ON vr.vehicle_id = sr.vehicle_id AND vr.{method} = sr.min_value
                    WHERE vr.run_configs_id = :run_config_id AND vr.iteration_id = :iteration_id
                    GROUP BY vr.vehicle_id
                )
                SELECT vehicle_id, route_id FROM selected_routes
            """)
            result = session.execute(sql, {
                'run_config_id': run_config_id,
                'iteration_id': iteration_id
            })
            for row in result.fetchall():
                if row.vehicle_id in non_optimized:
                    vehicle_route_pairs.append((row.vehicle_id, row.route_id))

        # Use the same connection for all temp table operations
        with session.connection() as conn:
            conn.execute("""
                CREATE TEMPORARY TABLE IF NOT EXISTS temp_selected_routes (
                    vehicle_id INT,
                    route_id INT
                )
            """)
            # Insert data into temp_selected_routes
            for vehicle_id, route_id in vehicle_route_pairs:
                conn.execute(sa_text("""
                    INSERT INTO temp_selected_routes (vehicle_id, route_id) VALUES (:vehicle_id, :route_id)
                """), {'vehicle_id': vehicle_id, 'route_id': route_id})

            # Now run your main query using conn
            result = conn.execute(sa_text(f"""
                WITH selected_routes AS (
                    SELECT vehicle_id, route_id FROM temp_selected_routes
                ),
                cm_routes AS (
                    SELECT vehicle1 as vehicle, vehicle1_route as vehicle_route, edge_id, congestion_score
                    FROM congestion_map
                    WHERE run_configs_id = :run_config_id AND iteration_id = :iteration_id
                    UNION ALL
                    SELECT vehicle2 as vehicle, vehicle2_route, edge_id, congestion_score
                    FROM congestion_map
                    WHERE run_configs_id = :run_config_id AND iteration_id = :iteration_id
                )
                SELECT edge_id, sum(congestion_score) as congestion_score
                FROM (
                    SELECT cm.edge_id, cm.vehicle AS vehicle, coalesce(cm.congestion_score / 2,0) AS congestion_score
                    FROM selected_routes sr
                    LEFT JOIN cm_routes cm
                    ON cm.vehicle = sr.vehicle_id AND cm.vehicle_route = sr.route_id
                ) AS derived
                WHERE edge_id is not null
                group by edge_id;
            """), {'run_config_id': run_config_id, 'iteration_id': iteration_id})

        logger.info(f"Recomputed QA congestion for run_config_id={run_config_id}, iteration_id={iteration_id}.")
        return pd.DataFrame(list(result.fetchall()), columns=pd.Index(['edge_id', 'congestion_score']))
    except Exception as e:
        session.rollback()
        logger.error(f"Error in post_qa_congestion: {e}", exc_info=True)
        raise e
        
