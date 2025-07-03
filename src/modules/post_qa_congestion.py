import pandas as pd
from sqlalchemy import text as sa_text
import logging
from typing import Any, List

logger = logging.getLogger(__name__)

def post_qa_congestion(
    session: Any,
    run_configs_id: int,
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
        vehicle_route_pairs = []

        # For optimized vehicles, use QA assignment
        num_routes = len(qa_assignment) // len(optimized_vehicle_ids)
        for idx, vehicle_id in enumerate(optimized_vehicle_ids):
            assignment = qa_assignment[num_routes * idx : num_routes * (idx + 1)]
            if assignment.count(1) != 1:
                # raise ValueError(f"Invalid assignment {assignment} for vehicle {vehicle_id}: not one-hot.")
                logger.warning(f"Invalid assignment for vehicle {vehicle_id}: not one-hot. Defaulting to first route.")
                route_id = 1  # Default to first route
            else:
                route_id = assignment.index(1) + 1  # 1-based route_id
            vehicle_route_pairs.append((vehicle_id, route_id))
        logger.info(f"Number of optimized vehicles: {len(optimized_vehicle_ids)}")

        # For non-optimized vehicles, assign by shortest (distance/duration)
        non_optimized = set(all_vehicle_ids) - set(optimized_vehicle_ids)
        logger.info(f"Number of non-optimized vehicles: {len(non_optimized)}")
        if non_optimized:
            sql = sa_text(f"""
                WITH shortest_routes AS (
                    SELECT vehicle_id, MIN({method}) AS min_value
                    FROM vehicle_routes
                    WHERE run_configs_id = :run_configs_id AND iteration_id = :iteration_id
                    GROUP BY vehicle_id
                ),
                selected_routes AS (
                    SELECT vr.vehicle_id, max(vr.route_id) as route_id 
                    FROM vehicle_routes vr
                    JOIN shortest_routes sr
                    ON vr.vehicle_id = sr.vehicle_id AND vr.{method} = sr.min_value
                    WHERE vr.run_configs_id = :run_configs_id AND vr.iteration_id = :iteration_id
                    GROUP BY vr.vehicle_id
                )
                SELECT vehicle_id, route_id FROM selected_routes
            """)
            result = session.execute(sql, {
                'run_configs_id': run_configs_id,
                'iteration_id': iteration_id
            })
            for row in result.fetchall():
                if row.vehicle_id in non_optimized:
                    vehicle_route_pairs.append((row.vehicle_id, row.route_id))

        # Use a regular table for selected routes
        session.execute(sa_text("""
            CREATE TABLE IF NOT EXISTS selected_routes (
                vehicle_id INT,
                route_id INT
            )
        """))
        # Clear previous data
        session.execute(sa_text("TRUNCATE TABLE selected_routes"))
        for vehicle_id, route_id in vehicle_route_pairs:
            session.execute(sa_text("""
                INSERT INTO selected_routes (vehicle_id, route_id) VALUES (:vehicle_id, :route_id)
            """), {'vehicle_id': vehicle_id, 'route_id': route_id})
        session.commit()

        # After building vehicle_route_pairs, check for completeness and uniqueness
        vehicle_id_counts = {}
        for vehicle_id, route_id in vehicle_route_pairs:
            vehicle_id_counts[vehicle_id] = vehicle_id_counts.get(vehicle_id, 0) + 1
        missing_vehicles = set(all_vehicle_ids) - set(vehicle_id_counts.keys())
        duplicated_vehicles = [vid for vid, count in vehicle_id_counts.items() if count > 1]
        if missing_vehicles:
            logger.warning(f"Missing vehicles in vehicle_route_pairs: {missing_vehicles}")
        if duplicated_vehicles:
            logger.warning(f"Duplicated vehicles in vehicle_route_pairs: {duplicated_vehicles}")
        if not missing_vehicles and not duplicated_vehicles:
            logger.info(f"All {len(all_vehicle_ids)} vehicles have exactly one route assigned.")

        # Now run your main query using session
        result = session.execute(sa_text(f"""
            WITH cm_routes AS (
                SELECT vehicle1 as vehicle, vehicle1_route as vehicle_route, edge_id, congestion_score
                FROM congestion_map
                WHERE run_configs_id = :run_configs_id AND iteration_id = :iteration_id
                UNION ALL
                SELECT vehicle2 as vehicle, vehicle2_route, edge_id, congestion_score
                FROM congestion_map
                WHERE run_configs_id = :run_configs_id AND iteration_id = :iteration_id
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
        """), {'run_configs_id': run_configs_id, 'iteration_id': iteration_id})
        rows = list(result.fetchall())
        # Optionally, drop the table at the end:
        # session.execute(sa_text("DROP TABLE selected_routes"))
        # session.commit()

        logger.info(f"Recomputed QA congestion for run_configs_id={run_configs_id}, iteration_id={iteration_id}.")
        return pd.DataFrame(rows, columns=pd.Index(['edge_id', 'congestion_score']))
    except Exception as e:
        session.rollback()
        logger.error(f"Error in post_qa_congestion: {e}", exc_info=True)
        return pd.DataFrame()
        
