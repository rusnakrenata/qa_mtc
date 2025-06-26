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
        WITH shortest_routes AS (
        SELECT vehicle_id, MIN({method}) AS min_value
        FROM vehicle_routes
        WHERE run_configs_id = :run_config_id AND iteration_id = :iteration_id
        GROUP BY vehicle_id
        ),
        selected_routes AS (
            SELECT vr.route_id, vr.vehicle_id
            FROM vehicle_routes vr
            JOIN shortest_routes sr
            ON vr.vehicle_id = sr.vehicle_id AND vr.{method} = sr.min_value
            WHERE vr.run_configs_id = :run_config_id AND vr.iteration_id = :iteration_id
        )
        SELECT edge_id, SUM(congestion_score) AS congestion_score
        FROM (
            SELECT DISTINCT cm.edge_id, cm.vehicle1 AS vehicle, cm.congestion_score / 2 AS congestion_score
            FROM congestion_map cm
            JOIN selected_routes sr
            ON cm.vehicle1 = sr.vehicle_id AND cm.vehicle1_route = sr.route_id
            WHERE cm.run_configs_id = :run_config_id AND cm.iteration_id = :iteration_id
            UNION ALL
            SELECT DISTINCT cm.edge_id, cm.vehicle2 AS vehicle, cm.congestion_score / 2 AS congestion_score
            FROM congestion_map cm
            JOIN selected_routes sr
            ON cm.vehicle2 = sr.vehicle_id AND cm.vehicle2_route = sr.route_id
            WHERE cm.run_configs_id = :run_config_id AND cm.iteration_id = :iteration_id
        ) AS derived
        GROUP BY edge_id;
        """)
        result = session.execute(sql, {
            'run_config_id': run_config_id,
            'iteration_id': iteration_id
        })
        df = pd.DataFrame(result.fetchall(), columns=["edge_id", "congestion_score"])
        logger.info(f"Computed shortest routes ({method}) for run_config_id={run_config_id}, iteration_id={iteration_id}.")
        return df
    except Exception as e:
        logger.error(f"Error computing shortest routes: {e}", exc_info=True)
        return pd.DataFrame(columns=["edge_id", "congestion_score"])
