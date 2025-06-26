import pandas as pd
import sqlalchemy as sa
import logging
from typing import Any

logger = logging.getLogger(__name__)

def get_congestion_weights(
    session: Any,
    run_configs_id: int,
    iteration_id: int
) -> pd.DataFrame:
    """
    Fetches pairwise vehicle congestion weights from SQL query and returns as a DataFrame.

    Args:
        session: SQLAlchemy session
        run_configs_id: ID of the run config
        iteration_id: Iteration number

    Returns:
        weights_df: DataFrame with columns:
            ['vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'weighted_congestion_score']
    """
    try:
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
        logger.info(f"Fetched {len(weights_df)} congestion weight records for run_config_id={run_configs_id}, iteration_id={iteration_id}.")
        return weights_df
    except Exception as e:
        logger.error(f"Error fetching congestion weights: {e}", exc_info=True)
        return pd.DataFrame(columns=[
            'vehicle1', 'vehicle2', 'vehicle1_route', 'vehicle2_route', 'weighted_congestion_score'
        ])
