import logging
from typing import Any

logger = logging.getLogger(__name__)

def get_or_create_run_config(
    session: Any,
    city_id: int,
    RunConfig: Any,
    nr_vehicles: int,
    k_routes: int,
    min_length: int,
    max_length: int,
    time_step: int,
    time_window: int
) -> Any:
    """
    Get or create a run configuration in the database.

    Args:
        session: SQLAlchemy session
        city_id: City ID
        RunConfig: SQLAlchemy RunConfig model
        nr_vehicles: Number of vehicles
        k_routes: Number of route alternatives
        min_length: Minimum route length
        max_length: Maximum route length
        time_step: Time step
        time_window: Time window

    Returns:
        run_config: The existing or newly created RunConfig object
    """
    try:
        existing_run = session.query(RunConfig).filter_by(
            city_id=city_id,
            n_cars=nr_vehicles,
            k_alternatives=k_routes,
            min_length=min_length,
            max_length=max_length,
            time_step=time_step,
            time_window=time_window
        ).first()
        if existing_run:
            logger.info(f"Run config already exists (run_id={existing_run.id}), skipping insertion.")
            return existing_run
        else:
            run_config = RunConfig(
                city_id=city_id,
                n_cars=nr_vehicles,
                k_alternatives=k_routes,
                min_length=min_length,
                max_length=max_length,
                time_step=time_step,
                time_window=time_window
            )
            session.add(run_config)
            session.commit()
            logger.info(f"Run configuration saved (run_id={run_config.id}).")
            return run_config
    except Exception as e:
        logger.error(f"Error in get_or_create_run_config: {e}", exc_info=True)
        session.rollback()
        return None
