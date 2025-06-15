# method - duration/distance
# from sql to compare the results with qubo results
# return congestion_df which will be used as input to plot_congestion_heatmap.py

import pandas as pd
from sqlalchemy import text as sa_text

def compute_shortest_routes(session, run_config_id, iteration_id, method="duration"):
    """
    Returns congestion_df for shortest routes based on congestion_map table.

    Parameters:
        session: SQLAlchemy session
        run_config_id: run_config.id
        iteration_id: iteration_id
        method: "duration" or "distance"

    Returns:
        congestion_df: DataFrame with columns ['edge_id', 'congestion_score']
    """
    assert method in ("duration", "distance"), "Method must be 'duration' or 'distance'"

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
        SELECT edge_id, sum(congestion_score) as congestion_score
        FROM 
        (
        SELECT distinct edge_id, vehicle1 as vehicle, congestion_score/2 as congestion_score
        FROM congestion_map cm 
        INNER JOIN selected_routes sr
        ON CONCAT(cm.vehicle1,vehicle1_route) IN (SELECT CONCAT(vehicle_id,route_id) FROM selected_routes)
        WHERE run_configs_id = :run_config_id AND iteration_id = :iteration_id
        UNION ALL
        SELECT distinct edge_id, vehicle2 as vehicle, congestion_score/2 as congestion_score
        FROM congestion_map cm 
        INNER JOIN selected_routes sr
        ON CONCAT(cm.vehicle2,vehicle2_route) IN (SELECT CONCAT(vehicle_id,route_id) FROM selected_routes)
        WHERE run_configs_id = :run_config_id AND iteration_id = :iteration_id
        ) a
        group by edge_id
    """)

    result = session.execute(sql, {
        'run_config_id': run_config_id,
        'iteration_id': iteration_id
    })

    df = pd.DataFrame(result.fetchall(), columns=["edge_id", "congestion_score"])
    return df
