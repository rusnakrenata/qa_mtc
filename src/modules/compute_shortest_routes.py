import pandas as pd
from sqlalchemy import text as sa_text
import logging
from typing import Any, Tuple, List, Type
from datetime import datetime
from models import ShortestRouteDur, ShortestRouteDis  # Assume you create this second model

logger = logging.getLogger(__name__)

def compute_shortest_routes(
    session: Any,
    run_config_id: int,
    iteration_id: int,
    method: str = "duration"
) -> Tuple[pd.DataFrame, List[Any]]:
    """
    Compute congestion for shortest routes and return:
    - congestion_df: DataFrame with ['edge_id', 'congestion_score']
    - route_objs: List of ORM route objects stored in shortest_routes_{method} table
    """
    assert method in ("duration", "distance"), "Method must be 'duration' or 'distance'"
    
    # Select model class based on method
    route_model: Type = ShortestRouteDur if method == "duration" else ShortestRouteDis
    table_name = route_model.__tablename__

    try:
        # Step 1: Select (vehicle_id, route_id) pairs
        route_sql = sa_text(f"""
            WITH 
            filtered_vehicle_routes AS (
                SELECT vehicle_id, route_id, {method}
                FROM vehicle_routes
                WHERE run_configs_id = :run_config_id 
                AND iteration_id = :iteration_id
            ),
            shortest_routes AS (
                SELECT vehicle_id, MIN({method}) AS min_value
                FROM filtered_vehicle_routes
                GROUP BY vehicle_id
            ),
            shortest_selected_routes AS (
                SELECT fvr.vehicle_id, MAX(fvr.route_id) AS route_id
                FROM filtered_vehicle_routes fvr
                JOIN shortest_routes sr
                ON fvr.vehicle_id = sr.vehicle_id AND fvr.{method} = sr.min_value
                GROUP BY fvr.vehicle_id
            )
            SELECT vehicle_id, route_id
            FROM shortest_selected_routes;
        """)
        route_result = session.execute(route_sql, {
            'run_config_id': run_config_id,
            'iteration_id': iteration_id
        })
        route_pairs = route_result.fetchall()

        logger.info(f"Selected shortest-{method} routes for {len(route_pairs)} vehicles.")

        # Step 2: Delete old records and insert new ones
        session.query(route_model).filter(
            route_model.run_configs_id == run_config_id,
            route_model.iteration_id == iteration_id
        ).delete()

        route_objs = [
            route_model(
                run_configs_id=run_config_id,
                iteration_id=iteration_id,
                vehicle_id=row.vehicle_id,
                route_id=row.route_id,
                created_at=datetime.utcnow()
            )
            for row in route_pairs
        ]
        session.add_all(route_objs)
        session.commit()

        # Step 3: Calculate congestion
        congestion_sql = sa_text(f"""
            SELECT 
                cm.edge_id, 
                SUM(cm.congestion_score) AS congestion_score
            FROM congestion_map cm
            JOIN {table_name} sr1 
                ON sr1.vehicle_id = cm.vehicle1 AND sr1.route_id = cm.vehicle1_route
            JOIN {table_name} sr2 
                ON sr2.vehicle_id = cm.vehicle2 AND sr2.route_id = cm.vehicle2_route
            WHERE cm.run_configs_id = :run_config_id 
            AND cm.iteration_id = :iteration_id
            GROUP BY cm.edge_id;
        """)
        congestion_result = session.execute(congestion_sql, {
            'run_config_id': run_config_id,
            'iteration_id': iteration_id
        })
        congestion_df = pd.DataFrame(congestion_result.fetchall(), columns=["edge_id", "congestion_score"])

        return congestion_df, route_objs
    except Exception as e:
        logger.error(f"Error computing shortest-{method} routes: {e}", exc_info=True)
        return pd.DataFrame(columns=["edge_id", "congestion_score"]), []
