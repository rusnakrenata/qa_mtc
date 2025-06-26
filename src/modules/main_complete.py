# ---------- IMPORT MODULES ----------
import pandas as pd
import numpy as np
from get_city_graph import get_city_graph
from get_city_data_from_db import get_city_data_from_db
from store_city_to_db import store_city_to_db
from get_or_create_run_config import get_or_create_run_config
from create_iteration import create_iteration
from generate_vehicles import generate_vehicles
from generate_vehicle_routes import generate_vehicle_routes
from generate_congestion import generate_congestion
from plot_congestion_heatmap import plot_congestion_heatmap, plot_congestion_heatmap_interactive
from get_congestion_weights import get_congestion_weights
from normalize_congestion_weights import normalize_congestion_weights
from congestion_weights import congestion_weights
from qubo_matrix import qubo_matrix
from compute_shortest_routes import compute_shortest_routes
from filter_routes_for_qubo import filter_vehicles_by_congested_edges_and_limit
from post_qa_congestion import post_qa_congestion
from qa_testing import qa_testing
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Any, List, Tuple, Optional


# ---------- CONFIGURATION ----------
from sqlalchemy.orm import sessionmaker
from models import * #City, Node, Edge, RunConfig, Iteration, Vehicle, VehicleRoute, CongestionMap, RoutePoint  # adjust to your actual model imports

from config import CITY_NAME, N_VEHICLES, K_ALTERNATIVES, MIN_LENGTH, MAX_LENGTH, TIME_STEP, TIME_WINDOW, DIST_THRESH, SPEED_DIFF_THRESH, LAMBDA_STRATEGY, LAMBDA_VALUE, COMP_TYPE, ROUTE_METHOD

# Named constants
OFFSET_DEG = 0.0000025
QUBO_OUTPUT_DIR = Path("files")
QUBO_MATRIX_FILENAME = QUBO_OUTPUT_DIR / "qubo_matrix.csv"
CONGESTION_WEIGHTS_FILENAME = QUBO_OUTPUT_DIR / "congestion_weights.csv"
CONGESTION_HEATMAP_FILENAME = QUBO_OUTPUT_DIR / "congestion_heatmap.html"
SHORTEST_DUR_HEATMAP_FILENAME = QUBO_OUTPUT_DIR / "shortest_routes_dur_congestion_heatmap.html"
SHORTEST_DIS_HEATMAP_FILENAME = QUBO_OUTPUT_DIR / "shortest_routes_dis_congestion_heatmap.html"
POST_QA_HEATMAP_FILENAME = QUBO_OUTPUT_DIR / "post_qa_congestion_heatmap.html"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def create_db_session() -> Any:
    """Create a new SQLAlchemy session."""
    Session = sessionmaker(bind=engine, autocommit=False)
    return Session()


def get_or_create_city(session) -> Any:
    """Get or create the city in the database."""
    city = session.query(City).filter_by(name=CITY_NAME).first()
    if not city:
        nodes, edges = get_city_graph(CITY_NAME)
        city = store_city_to_db(session, CITY_NAME, nodes, edges, City, Node, Edge)
    return city


def get_or_create_run_config_for_city(session, city) -> Any:
    """Get or create a run configuration for the city."""
    return get_or_create_run_config(
        session, city.id, RunConfig, N_VEHICLES, K_ALTERNATIVES,
        MIN_LENGTH, MAX_LENGTH, TIME_STEP, TIME_WINDOW
    )


def create_simulation_iteration(session, run_config) -> Optional[int]:
    """Create a new simulation iteration."""
    iteration_id = create_iteration(session, run_config.id, None, Iteration)
    if iteration_id is None:
        logger.warning("No new iteration created. Exiting workflow.")
    return iteration_id


def generate_and_store_vehicles(session, run_config, iteration_id) -> Any:
    """Generate vehicles and store them in the database."""
    logger.info("Generate vehicles at: %s", datetime.now())
    return generate_vehicles(
        session, Vehicle, run_config.id, iteration_id,
        get_city_data_from_db(session, run_config.city_id)[1], N_VEHICLES, MIN_LENGTH, MAX_LENGTH
    )


def generate_and_store_routes(session, run_config, iteration_id, vehicles_gdf, edges) -> None:
    """Generate vehicle routes and store them in the database."""
    logger.info("Generate vehicle routes at: %s", datetime.now())
    generate_vehicle_routes(
        session, VehicleRoute, RoutePoint,
        run_config.id, iteration_id,
        vehicles_gdf, edges, K_ALTERNATIVES, TIME_STEP, TIME_WINDOW
    )


def compute_and_store_congestion(session, run_config, iteration_id) -> pd.DataFrame:
    """Compute congestion and store in the database."""
    logger.info("Compute congestion at: %s", datetime.now())
    return generate_congestion(
        session, CongestionMap,
        run_config.id, iteration_id,
        DIST_THRESH, SPEED_DIFF_THRESH
    )


def build_and_save_qubo_matrix(session, run_config, iteration_id, vehicles_gdf, congestion_df, weights_df) -> Tuple[Any, List[Any]]:
    """Build QUBO matrix and save to CSV."""
    vehicle_ids = vehicles_gdf["vehicle_id"].tolist()
    Q, filtered_vehicle_ids = qubo_matrix(
        N_VEHICLES, K_ALTERNATIVES, congestion_df, weights_df, vehicle_ids, lambda_strategy="normalized", fixed_lambda=1.0, filtering_percentage=0.1, max_qubo_size=None
    )
    N_FILTERED = len(filtered_vehicle_ids)
    logger.info("Filtered vehicles number: %d", N_FILTERED)

    def qubo_dict_to_dataframe(Q, size):
        matrix = np.zeros((size, size))
        for (i, j), v in Q.items():
            matrix[i][j] = v
            if i != j:
                matrix[j][i] = v  # ensure symmetry for display
        return pd.DataFrame(matrix)

    size = N_FILTERED * K_ALTERNATIVES
    Q_df = qubo_dict_to_dataframe(Q, size)
    QUBO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Q_df.to_csv(QUBO_MATRIX_FILENAME, index=False)
    return Q, filtered_vehicle_ids


def visualize_congestion(edges, congestion_df, shortest_routes_dur_df, shortest_routes_dis_df, post_qa_congestion_df) -> None:
    """Visualize congestion and save heatmaps."""
    plot_map = plot_congestion_heatmap_interactive(edges, congestion_df, offset_deg=OFFSET_DEG)
    if plot_map is not None:
        plot_map.save(CONGESTION_HEATMAP_FILENAME)
    else:
        logger.warning("No congestion map data to plot.")

    plot_map_dur = plot_congestion_heatmap_interactive(edges, shortest_routes_dur_df, offset_deg=OFFSET_DEG)
    if plot_map_dur is not None:
        plot_map_dur.save(SHORTEST_DUR_HEATMAP_FILENAME)
    else:
        logger.warning("No duration-based shortest route map data to plot.")

    plot_map_dis = plot_congestion_heatmap_interactive(edges, shortest_routes_dis_df, offset_deg=OFFSET_DEG)
    if plot_map_dis is not None:
        plot_map_dis.save(SHORTEST_DIS_HEATMAP_FILENAME)
    else:
        logger.warning("No distance-based shortest route map data to plot.")

    plot_map_post_qa = plot_congestion_heatmap_interactive(edges, post_qa_congestion_df, offset_deg=OFFSET_DEG)
    if plot_map_post_qa is not None:
        plot_map_post_qa.save(POST_QA_HEATMAP_FILENAME)
    else:
        logger.warning("No post-qa map data to plot.")


def main() -> None:
    """Main workflow for traffic simulation and QUBO optimization."""
    start_time = datetime.now()
    try:
        with create_db_session() as session:
            logger.info("Starting workflow at: %s", start_time)
            city = get_or_create_city(session)
            nodes, edges = get_city_data_from_db(session, city.id)
            run_config = get_or_create_run_config_for_city(session, city)
            iteration_id = create_simulation_iteration(session, run_config)
            if iteration_id is None:
                return
            vehicles_gdf = generate_and_store_vehicles(session, run_config, iteration_id)
            generate_and_store_routes(session, run_config, iteration_id, vehicles_gdf, edges)
            congestion_df = compute_and_store_congestion(session, run_config, iteration_id)
            weights_df = get_congestion_weights(session, run_config.id, iteration_id)
            weights_df.to_csv(CONGESTION_WEIGHTS_FILENAME, index=False)

            # Filtering for QUBO
            all_vehicle_ids = vehicles_gdf["vehicle_id"].tolist()

            # QUBO matrix
            Q, filtered_vehicle_ids = build_and_save_qubo_matrix(session, run_config, iteration_id, vehicles_gdf, congestion_df, weights_df)

            # QA testing
            qa_result = qa_testing(
                Q=Q,
                run_config_id=run_config.id,
                iteration_id=iteration_id,
                session=session,
                n=len(filtered_vehicle_ids),
                t=K_ALTERNATIVES,
                weights=weights_df.to_dict(orient='records'),
                vehicle_ids=filtered_vehicle_ids,
                lambda_strategy=LAMBDA_STRATEGY,
                lambda_value=LAMBDA_VALUE,
                comp_type=COMP_TYPE,
                num_reads=10
            )
            qa_assignment = qa_result['assignment']
            logger.info(f"QA assignment: {qa_assignment}")

            # Post-QA congestion
            post_qa_congestion_df = post_qa_congestion(
                session=session,
                run_config_id=run_config.id,
                iteration_id=iteration_id,
                all_vehicle_ids=all_vehicle_ids,
                optimized_vehicle_ids=filtered_vehicle_ids,
                qa_assignment=qa_assignment,
                method=ROUTE_METHOD
            )
            logger.info(f"Post-QA congestion result: {post_qa_congestion_df}")

             

            shortest_routes_dur_df = compute_shortest_routes(session, run_config.id, iteration_id, method="duration")
            shortest_routes_dis_df = compute_shortest_routes(session, run_config.id, iteration_id, method="distance")
            visualize_congestion(edges, congestion_df, shortest_routes_dur_df, shortest_routes_dis_df, post_qa_congestion_df)
            logger.info("Workflow completed successfully!")
    except Exception as e:
        logger.error("Workflow failed: %s", str(e), exc_info=True)
    finally:
        end_time = datetime.now()
        logger.info("Total computational time: %s", end_time - start_time)


if __name__ == "__main__":
    main()
