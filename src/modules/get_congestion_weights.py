import pandas as pd
import sqlalchemy as sa

def get_congestion_weights(session,  run_configs_id, iteration_id ):
    """
    Fetches pairwise vehicle congestion weights from SQL query and returns as a DataFrame.

    Args:
        session: SQLAlchemy session
        run_configs_id: ID of the run config
        iteration_id: iteration number


    Returns:
        weights_df: DataFrame with columns:
            ['vehicle_1', 'vehicle_2', 'vehicle_1_route', 'vehicle_2_route', 'weighted_congestion_score']
    """

    # Set parameters
    session.execute(sa.text("SET @iteration := :iteration_id"), {'iteration_id': iteration_id})
    session.execute(sa.text("SET @run_configs_id := :run_configs_id"), {'run_configs_id': run_configs_id})

    # Query
    sql = sa.text("""
        SELECT
            vehicle1,
            vehicle2,
            vehicle1_route,
            vehicle2_route,
            congestion_score/2 as weighted_congestion_score
        FROM trafficOptimization.congestion_map            
            WHERE iteration_id = @iteration
            AND run_configs_id = @run_configs_id
    """)

    result = session.execute(sql)
    weights_df = pd.DataFrame(result.fetchall(), columns=[
        'vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'weighted_congestion_score'
    ])

    return weights_df
