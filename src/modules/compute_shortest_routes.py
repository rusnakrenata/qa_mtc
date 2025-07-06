# method - duration/distance
# from sql to compare the results with qubo results
# return congestion_df which will be used as input to plot_congestion_heatmap.py

import pandas as pd
from sqlalchemy import text as sa_text
import logging
from typing import Any

logger = logging.getLogger(__name__)

def compute_shortest_routes(
    session: Any,
    run_config_id: int,
    iteration_id: int,
    method: str = "duration"
) -> pd.DataFrame:
    """
    Returns congestion_df for shortest routes based on congestion_map table.

    Args:
        session: SQLAlchemy session
        run_config_id: Run configuration ID
        iteration_id: Iteration ID
        method: "duration" or "distance"

    Returns:
        congestion_df: DataFrame with columns ['edge_id', 'congestion_score']
    """
    assert method in ("duration", "distance"), "Method must be 'duration' or 'distance'"
    try:
        sql = sa_text(f"""
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
            ),
            filtered_congestion AS (
                SELECT edge_id, vehicle1, vehicle1_route, vehicle2, vehicle2_route, congestion_score
                FROM trafficOptimization.congestion_map
                WHERE run_configs_id = :run_config_id 
                AND iteration_id = :iteration_id
            )
            SELECT 
                cm.edge_id, 
                SUM(cm.congestion_score) AS congestion_score
            FROM filtered_congestion cm
            JOIN shortest_selected_routes sr1 
                ON sr1.vehicle_id = cm.vehicle1 AND sr1.route_id = cm.vehicle1_route
            JOIN shortest_selected_routes sr2 
                ON sr2.vehicle_id = cm.vehicle2 AND sr2.route_id = cm.vehicle2_route
            GROUP BY cm.edge_id;
        """)
        result = session.execute(sql, {
            'run_config_id': run_config_id,
            'iteration_id': iteration_id
        })
        df = pd.DataFrame(result.fetchall(), columns=pd.Index(["edge_id", "congestion_score"]))
        logger.info(f"Computed shortest routes ({method}) for run_config_id={run_config_id}, iteration_id={iteration_id}.")
        return df
    except Exception as e:
        logger.error(f"Error computing shortest routes: {e}", exc_info=True)
        return pd.DataFrame(columns=pd.Index(["edge_id", "congestion_score"]))
