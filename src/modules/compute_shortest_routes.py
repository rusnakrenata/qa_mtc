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
        SELECT vehicle_id, MIN(duration) AS min_value
        FROM vehicle_routes
        WHERE run_configs_id = :run_config_id AND iteration_id = :iteration_id
        GROUP BY vehicle_id
        ),
        selected_routes AS (
            SELECT vr.vehicle_id, max(vr.route_id) as route_id 
            FROM vehicle_routes vr
            JOIN shortest_routes sr
            ON vr.vehicle_id = sr.vehicle_id AND vr.duration = sr.min_value
            WHERE vr.run_configs_id = :run_config_id AND vr.iteration_id = :iteration_id
            group by vr.vehicle_id
        ),
        cm_routes as (
            SELECT * FROM 
		        (
		        SELECT vehicle1 as vehicle, vehicle1_route as vehicle_route, edge_id,congestion_score  
		        FROM congestion_map 
		        WHERE run_configs_id = :run_config_id AND iteration_id = :iteration_id
		        union all
		        SELECT vehicle2 as vehicle, vehicle2_route, edge_id,congestion_score 
		        FROM congestion_map 
		        WHERE run_configs_id = :run_config_id AND iteration_id = :iteration_id
		        ) as cong
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
