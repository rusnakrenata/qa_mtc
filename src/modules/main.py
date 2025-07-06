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
from plot_congestion_heatmap import plot_congestion_heatmap_interactive
from get_congestion_weights import get_congestion_weights
from qubo_matrix import qubo_matrix
from compute_shortest_routes import compute_shortest_routes
from post_qa_congestion import post_qa_congestion
from qa_testing import qa_testing
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Any, List, Tuple, Optional
from sqlalchemy import text as sa_text
import geopandas as gpd


# ---------- CONFIGURATION ----------
from sqlalchemy.orm import sessionmaker
from models import * #City, Node, Edge, RunConfig, Iteration, Vehicle, VehicleRoute, CongestionMap, RoutePoint  # adjust to your actual model imports

from config import *

# Named constants
OFFSET_DEG = 0.0000025
QUBO_OUTPUT_DIR = Path("files")
QUBO_MATRIX_FILENAME = QUBO_OUTPUT_DIR / "qubo_matrix.csv"
CONGESTION_WEIGHTS_FILENAME = QUBO_OUTPUT_DIR / "congestion_weights.csv"
CONGESTION_HEATMAP_FILENAME = QUBO_OUTPUT_DIR / "congestion_heatmap.html"
AFFECTED_EDGES_HEATMAP_FILENAME = QUBO_OUTPUT_DIR / "affected_edges_heatmap.html"
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
    # Check if we're looking for a city subset or full city
    if CENTER_COORDS is not None:
        # Look for existing city subset with matching coordinates
        lat, lon = CENTER_COORDS  # type: ignore
        city = session.query(City).filter_by(
            name=CITY_NAME,
            center_lat=lat,
            center_lon=lon,
            is_subset=True
        ).first()
        
        if not city:
            # Create new city subset
            nodes, edges = get_city_graph(CITY_NAME, center_coords=CENTER_COORDS, radius_km=1.0)
            city = store_city_to_db(
                session, CITY_NAME, nodes, edges, City, Node, Edge,
                center_coords=CENTER_COORDS, radius_km=1.0
            )
    else:
        # Look for existing full city
        city = session.query(City).filter_by(
            name=CITY_NAME,
            is_subset=False
        ).first()
        
        if not city:
            # Create new full city
            nodes, edges = get_city_graph(CITY_NAME)
            city = store_city_to_db(session, CITY_NAME, nodes, edges, City, Node, Edge)
    
    return city


def get_or_create_run_config_for_city(session, city) -> Any:
    """Get or create a run configuration for the city."""
    return get_or_create_run_config(
        session, city.city_id, RunConfig, N_VEHICLES, K_ALTERNATIVES,
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


def generate_and_store_routes(session, run_config, iteration_id, vehicles_gdf, edges) -> pd.DataFrame:
    """Generate vehicle routes and store them in the database."""
    logger.info("Generate vehicle routes at: %s", datetime.now())
    vehicle_routes_df = generate_vehicle_routes(
        session, VehicleRoute, RoutePoint,
        run_config.id, iteration_id,
        vehicles_gdf, edges, K_ALTERNATIVES, TIME_STEP, TIME_WINDOW
    )
    return vehicle_routes_df


def compute_and_store_congestion(session, run_config, iteration_id) -> pd.DataFrame:
    """Compute congestion and store in the database."""
    logger.info("Compute congestion at: %s", datetime.now())
    congestion_df = generate_congestion(
        session, CongestionMap,
        run_config.id, iteration_id,
        DIST_THRESH, SPEED_DIFF_THRESH
    )
    return congestion_df  # Do not groupby here


def get_k_alternatives(session, run_configs_id, iteration_id):
    sql = sa_text("""
        SELECT MAX(route_id) as max_route_id
        FROM vehicle_routes
        WHERE run_configs_id = :run_configs_id AND iteration_id = :iteration_id
    """)
    result = session.execute(sql, {'run_configs_id': run_configs_id, 'iteration_id': iteration_id})
    max_route_id = result.scalar()
    return int(max_route_id) if max_route_id is not None else 1


def build_and_save_qubo_matrix(
    vehicle_routes_df: pd.DataFrame,
    congestion_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    session: Any,
    run_configs_id: int,
    iteration_id: int,
    t: int,
    lambda_strategy: str,
    fixed_lambda: float,
    filtering_percentage: float,
    target_size: float,
    R: float = R_VALUE
) -> Tuple[Any, List[Any], pd.DataFrame]:
    """Build QUBO matrix and save run stats."""
    Q, filtered_vehicle_ids, affected_edges_df = qubo_matrix(
        t, congestion_df, weights_df, vehicle_routes_df,
        lambda_strategy=lambda_strategy,
        fixed_lambda=fixed_lambda,
        filtering_percentage=filtering_percentage,
        target_size=target_size,
        R=R
    )
    # Find the matrix size
    max_index = 0
    if Q:
        max_index = max(max(k[0], k[1]) for k in Q.keys()) + 1

    # Create the matrix and fill it
    Q_matrix = np.zeros((max_index, max_index))
    for (q1, q2), value in Q.items():
        Q_matrix[q1, q2] = value

    # Create DataFrame with proper column and row labels
    Q_df = pd.DataFrame(Q_matrix)
    
    # Add column numbers as headers (formatted to 9 characters)
    Q_df.columns = [f'{0:18g}'] + [f'{i:9g}' for i in range(1, max_index)]
    Q_df.index = [f'{i:9g}' for i in range(max_index)]
    
   
    # Save with custom formatting - 9 characters per number
    Q_df.to_csv(QUBO_MATRIX_FILENAME, index=True, header=True, float_format='%9g')
    N_FILTERED = len(filtered_vehicle_ids)
    logger.info("Filtered vehicles number: %d", N_FILTERED)
    stats = QuboRunStats(
        run_configs_id=run_configs_id,
        iteration_id=iteration_id,
        filtering_percentage=filtering_percentage,
        n_vehicles=N_VEHICLES,
        n_filtered_vehicles=N_FILTERED
    )
    session.add(stats)
    session.commit()
    return Q, filtered_vehicle_ids, affected_edges_df


def visualize_and_save_congestion(
    edges: gpd.GeoDataFrame,
    congestion_df: pd.DataFrame,
    affected_edges_df: pd.DataFrame,
    shortest_routes_dur_df: pd.DataFrame,
    shortest_routes_dis_df: pd.DataFrame,
    post_qa_congestion_df: pd.DataFrame
) -> None:
    """Visualize congestion and save heatmaps."""
    all_scores = pd.concat([
        shortest_routes_dur_df['congestion_score'],
        shortest_routes_dis_df['congestion_score'],
        post_qa_congestion_df['congestion_score']
    ])
    vmin = float(all_scores.min())
    vmax = float(all_scores.max())
    plot_map = plot_congestion_heatmap_interactive(edges, congestion_df, offset_deg=OFFSET_DEG)
    if plot_map is not None:
        plot_map.save(CONGESTION_HEATMAP_FILENAME)
    plot_map_affected_edges = plot_congestion_heatmap_interactive(edges, affected_edges_df, offset_deg=OFFSET_DEG, vmin=vmin, vmax=vmax)
    if plot_map_affected_edges is not None:
        plot_map_affected_edges.save(AFFECTED_EDGES_HEATMAP_FILENAME)
    plot_map_dur = plot_congestion_heatmap_interactive(edges, shortest_routes_dur_df, offset_deg=OFFSET_DEG, vmin=vmin, vmax=vmax)
    if plot_map_dur is not None:
        plot_map_dur.save(SHORTEST_DUR_HEATMAP_FILENAME)
    plot_map_dis = plot_congestion_heatmap_interactive(edges, shortest_routes_dis_df, offset_deg=OFFSET_DEG, vmin=vmin, vmax=vmax)
    if plot_map_dis is not None:
        plot_map_dis.save(SHORTEST_DIS_HEATMAP_FILENAME)
    plot_map_post_qa = plot_congestion_heatmap_interactive(edges, post_qa_congestion_df, offset_deg=OFFSET_DEG, vmin=vmin, vmax=vmax)
    if plot_map_post_qa is not None:
        plot_map_post_qa.save(POST_QA_HEATMAP_FILENAME)


def save_congestion_summary(
    session: Any,
    edges: pd.DataFrame,
    congestion_df: pd.DataFrame,
    post_qa_congestion_df: pd.DataFrame,
    shortest_routes_dur_df: pd.DataFrame,
    shortest_routes_dis_df: pd.DataFrame,
    run_config: RunConfig,
    iteration_id: int
) -> None:
    """Save congestion summary to the database."""
    congestion_df_grouped = congestion_df.groupby('edge_id', as_index=False).agg({'congestion_score': 'sum'})
    merged = pd.DataFrame({'edge_id': edges.drop_duplicates(subset='edge_id')['edge_id']})
    merged = merged.merge(congestion_df_grouped[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_all'}), on='edge_id', how='left')  # type: ignore
    merged = merged.merge(post_qa_congestion_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_post_qa'}), on='edge_id', how='left')  # type: ignore
    merged = merged.merge(shortest_routes_dur_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_shortest_dur'}), on='edge_id', how='left')  # type: ignore
    merged = merged.merge(shortest_routes_dis_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_shortest_dis'}), on='edge_id', how='left')  # type: ignore
    merged = merged.fillna(0)
    records = [
        CongestionSummary(
            run_configs_id=run_config.id,
            iteration_id=iteration_id,
            edge_id=int(row['edge_id']),
            congestion_all=float(row['congestion_all']),
            congestion_post_qa=float(row['congestion_post_qa']),
            congestion_shortest_dur=float(row['congestion_shortest_dur']),
            congestion_shortest_dis=float(row['congestion_shortest_dis'])
        )
        for _, row in merged.iterrows()
    ]
    session.add_all(records)
    session.commit()


def run_congestion_results_sql(session, run_configs_id, iteration_id):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sql_file = os.path.join(base_dir, 'sql', 'congestion_results.sql')
    with open(sql_file, 'r') as f:
        sql = f.read()
    # Split queries (naive split on ';')
    queries = [q.strip() for q in sql.split(';') if q.strip()]
    params = (run_configs_id, iteration_id)
    for i, query in enumerate(queries):
        print(f'--- Query {i+1} ---')
        try:
            df = pd.read_sql(query, session.bind, params=params)
            print(df)
        except Exception as e:
            print(f'Error running query {i+1}: {e}')


def check_bqm_against_solver_limits(Q):
    import dimod
    from dwave.system import LeapHybridBQMSampler
    bqm = dimod.BQM.from_qubo(Q)
    num_variables = len(bqm.variables)
    num_linear = len(bqm.linear)
    num_quadratic = len(bqm.quadratic)
    num_biases = num_linear + num_quadratic
    sampler = LeapHybridBQMSampler()
    max_vars = sampler.properties["maximum_number_of_variables"]
    max_biases = sampler.properties["maximum_number_of_biases"]
    print("Number of variables:", num_variables)
    print("Number of linear biases:", num_linear)
    print("Number of quadratic biases:", num_quadratic)
    print("Total number of biases:", num_biases)
    print("Solver maximum_number_of_variables:", max_vars)
    print("Solver maximum_number_of_biases:", max_biases)
    if num_variables > max_vars:
        raise ValueError("Too many variables for this solver!")
    if num_biases > max_biases:
        raise ValueError("Too many biases for this solver!")


def main() -> None:
    """Main workflow for traffic simulation and QUBO optimization."""
    start_time = datetime.now()
    try:
        with create_db_session() as session:
            logger.info("Starting workflow at: %s", start_time)
            city = get_or_create_city(session)
            nodes, edges = get_city_data_from_db(session, city.city_id)
            if not isinstance(edges, gpd.GeoDataFrame):
                edges = gpd.GeoDataFrame(edges)
            run_config = get_or_create_run_config_for_city(session, city)
            iteration_id = create_simulation_iteration(session, run_config)
            if iteration_id is None:
                return
            vehicles_gdf = generate_and_store_vehicles(session, run_config, iteration_id)
            vehicle_routes_df = generate_and_store_routes(session, run_config, iteration_id, vehicles_gdf, edges)
            congestion_df = compute_and_store_congestion(session, run_config, iteration_id)
            weights_df = get_congestion_weights(session, run_config.id, iteration_id)
            weights_df.to_csv(CONGESTION_WEIGHTS_FILENAME, index=False)

            # Filtering for QUBO
            all_vehicle_ids = vehicles_gdf["vehicle_id"].tolist()

            # QUBO matrix
            t = get_k_alternatives(session, run_config.id, iteration_id)
            Q, filtered_vehicle_ids, affected_edges_df = build_and_save_qubo_matrix(
                vehicle_routes_df, congestion_df, weights_df, session, run_config.id, iteration_id, t,
                LAMBDA_STRATEGY, LAMBDA_VALUE, FILTERING_PERCENTAGE, N_VEHICLES//10,
                R_VALUE
            )

            # Before QA testing, check BQM/QUBO limits
            check_bqm_against_solver_limits(Q)

            # QA testing
            qa_result = qa_testing(
                Q=Q,
                run_configs_id=run_config.id,
                iteration_id=iteration_id,
                session=session,
                n=len(filtered_vehicle_ids),
                t=t,
                vehicle_ids=filtered_vehicle_ids,
                lambda_strategy=LAMBDA_STRATEGY,
                lambda_value=LAMBDA_VALUE,
                comp_type=COMP_TYPE,
                num_reads=10,
                vehicle_routes_df=vehicle_routes_df
            )
            qa_assignment = qa_result['assignment']
            logger.info(f"QA assignment: {qa_assignment}")

            # Post-QA congestion
            post_qa_congestion_df = post_qa_congestion(
                session=session,
                run_configs_id=run_config.id,
                iteration_id=iteration_id,
                all_vehicle_ids=all_vehicle_ids,
                optimized_vehicle_ids=filtered_vehicle_ids,
                qa_assignment=qa_assignment,
                method=ROUTE_METHOD
            )
            post_qa_congestion_df = post_qa_congestion_df.groupby('edge_id', as_index=False).agg({'congestion_score': 'sum'})
            if not isinstance(post_qa_congestion_df, pd.DataFrame):
                post_qa_congestion_df = pd.DataFrame(post_qa_congestion_df)
            logger.info(f"Post-QA congestion result: {post_qa_congestion_df}")

            shortest_routes_dur_df = compute_shortest_routes(session, run_config.id, iteration_id, method="duration")
            shortest_routes_dis_df = compute_shortest_routes(session, run_config.id, iteration_id, method="distance")
            # I changed the congestion_df to affected_edges_df - from leiden clustering
            visualize_and_save_congestion(edges, congestion_df, affected_edges_df, shortest_routes_dur_df, shortest_routes_dis_df, post_qa_congestion_df)

            save_congestion_summary(session, edges, congestion_df, post_qa_congestion_df, shortest_routes_dur_df, shortest_routes_dis_df, run_config, iteration_id)
            run_congestion_results_sql(session, run_config.id, iteration_id)

            logger.info("Workflow completed successfully!")
    except Exception as e:
        logger.error("Workflow failed: %s", str(e), exc_info=True)
    finally:
        end_time = datetime.now()
        logger.info("Total computational time: %s", end_time - start_time)


if __name__ == "__main__":
    main()
