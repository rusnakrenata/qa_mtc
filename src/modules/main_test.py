# ---------- IMPORT MODULES ----------
import pandas as pd
import numpy as np
from get_city_graph import get_city_graph
from get_city_data_from_db import get_city_data_from_db
from store_city_to_db import store_city_to_db
from get_or_create_run_config import get_or_create_run_config
from create_iteration import create_iteration
from generate_vehicles_random import generate_vehicles
from generate_vehicles_attraction import generate_vehicles_attraction
from generate_vehicle_routes import generate_vehicle_routes
from generate_congestion import generate_congestion
from plot_congestion_heatmap import plot_congestion_heatmap_interactive
from get_congestion_weights import get_congestion_weights
from qubo_matrix import qubo_matrix
from compute_shortest_routes import compute_shortest_routes
from compute_random_routes import compute_random_routes
from post_qa_congestion import post_qa_congestion
from qa_testing import qa_testing
from gurobi import solve_qubo_with_gurobi
from post_gurobi_congestion import post_gurobi_congestion
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
QUBO_OUTPUT_DIR = Path("files_csv")
MAPS_OUTPUT_DIR = Path("files_html")
QUBO_MATRIX_FILENAME = QUBO_OUTPUT_DIR / "qubo_matrix.csv"
CONGESTION_WEIGHTS_FILENAME = QUBO_OUTPUT_DIR / "congestion_weights.csv"
# Removed static HTML filenames

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
            radius_km=RADIUS_KM,
            is_subset=True
        ).first()
        
        if not city:
            # Create new city subset
            nodes, edges = get_city_graph(CITY_NAME, center_coords=CENTER_COORDS, radius_km =RADIUS_KM)
            city = store_city_to_db(
                session, CITY_NAME, nodes, edges, City, Node, Edge,
                center_coords=CENTER_COORDS, radius_km=RADIUS_KM
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


def create_simulation_iteration(session, run_configs_id) -> Optional[int]:
    """Create a new simulation iteration."""
    iteration_id = create_iteration(session, run_configs_id, None, Iteration)
    if iteration_id is None:
        logger.warning("No new iteration created. Exiting workflow.")
    return iteration_id



logger = logging.getLogger(__name__)

def generate_and_store_vehicles(
    session,
    run_configs,
    iteration_id,
    attraction_point=None,        # (lat, lon) tuple or None
    d_alternatives=None           # int or None
) -> Any:
    """Generate vehicles and store them in the database."""
    logger.info("Generate vehicles at: %s", datetime.now())
    
    # Load edges GeoDataFrame
    _, edges_gdf = get_city_data_from_db(session, run_configs.city_id)
    
    if attraction_point is not None and d_alternatives is not None:
        # Use attraction-aware version
        logger.info("Using attraction-based vehicle generation.")
        return generate_vehicles_attraction(
            session=session,
            Vehicle=Vehicle,
            run_config_id=run_configs.run_configs_id,
            iteration_id=iteration_id,
            edges_gdf=edges_gdf,
            nr_vehicles=N_VEHICLES,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH,
            attraction_point=attraction_point,
            d_alternatives=d_alternatives
        )
    else:
        # Use default version
        logger.info("Using random vehicle generation.")
        return generate_vehicles(
            session=session,
            Vehicle=Vehicle,
            run_config_id=run_configs.run_configs_id,
            iteration_id=iteration_id,
            edges_gdf=edges_gdf,
            nr_vehicles=N_VEHICLES,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH
        )



def generate_and_store_routes(session, run_configs_id, iteration_id, vehicles_gdf, edges) -> pd.DataFrame:
    """Generate vehicle routes and store them in the database."""
    logger.info("Generate vehicle routes at: %s", datetime.now())
    vehicle_routes_df = generate_vehicle_routes(
        session, VehicleRoute, RoutePoint,
        run_configs_id, iteration_id,
        vehicles_gdf, edges, K_ALTERNATIVES, TIME_STEP, TIME_WINDOW
    )
    return vehicle_routes_df


def compute_and_store_congestion(session, run_configs_id, iteration_id) -> pd.DataFrame:
    """Compute congestion and store in the database."""
    logger.info("Compute congestion at: %s", datetime.now())
    congestion_df = generate_congestion(
        session,
        run_configs_id, iteration_id,
        DIST_THRESH, SPEED_DIFF_THRESH
    )
    return congestion_df  # Do not groupby h


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
    duration_penalty_df: pd.DataFrame,
    session: Any,
    run_configs_id: int,
    iteration_id: int,
    t: int,
    lambda_strategy: str,
    fixed_lambda: Optional[float] = None,
    filtering_percentage: float = 0.25
) -> Tuple[Any, List[Any], pd.DataFrame, float]:
    """Build QUBO matrix and save run stats."""
    Q, filtered_vehicle_ids, affected_edges_df, n_Q, t_Q, lambda_penalty = qubo_matrix(
        N_VEHICLES, t, congestion_df, weights_df, duration_penalty_df, vehicle_routes_df,
        lambda_strategy=lambda_strategy,
        fixed_lambda=fixed_lambda,
        filtering_percentage=filtering_percentage
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
        filtering_percentage=N_FILTERED/N_VEHICLES,
        n_vehicles=N_VEHICLES,
        n_filtered_vehicles=N_FILTERED
    )
    session.add(stats)
    session.commit()
    return Q, filtered_vehicle_ids, affected_edges_df, n_Q, t_Q, lambda_penalty


def visualize_and_save_congestion(
    edges: gpd.GeoDataFrame,
    congestion_df: pd.DataFrame,
    affected_edges_df: pd.DataFrame,
    shortest_routes_dur_df: pd.DataFrame,
    shortest_routes_dis_df: pd.DataFrame,
    post_qa_congestion_df: pd.DataFrame,
    post_gurobi_congestion_df: pd.DataFrame,
    random_routes_df: pd.DataFrame,
    congestion_heatmap_filename: Path,
    affected_edges_heatmap_filename: Path,
    shortest_dur_heatmap_filename: Path,
    shortest_dis_heatmap_filename: Path,
    post_qa_heatmap_filename: Path,
    random_routes_heatmap_filename: Path,
    post_gurobi_heatmap_file: Path
) -> None:
    """Visualize congestion and save heatmaps."""
    all_scores = pd.concat([
        shortest_routes_dur_df['congestion_score'],
        shortest_routes_dis_df['congestion_score'],
        post_qa_congestion_df['congestion_score'],
        random_routes_df['congestion_score'],
        post_gurobi_congestion_df['congestion_score']
    ])
    vmin = float(all_scores.min())
    vmax = float(all_scores.max())
    plot_map = plot_congestion_heatmap_interactive(edges, congestion_df, offset_deg=OFFSET_DEG)
    if plot_map is not None:
        plot_map.save(congestion_heatmap_filename)
    plot_map_affected_edges = plot_congestion_heatmap_interactive(edges, affected_edges_df, offset_deg=OFFSET_DEG, vmin=vmin, vmax=vmax)
    if plot_map_affected_edges is not None:
        plot_map_affected_edges.save(affected_edges_heatmap_filename)
    plot_map_dur = plot_congestion_heatmap_interactive(edges, shortest_routes_dur_df, offset_deg=OFFSET_DEG, vmin=vmin, vmax=vmax)
    if plot_map_dur is not None:
        plot_map_dur.save(shortest_dur_heatmap_filename)
    plot_map_dis = plot_congestion_heatmap_interactive(edges, shortest_routes_dis_df, offset_deg=OFFSET_DEG, vmin=vmin, vmax=vmax)
    if plot_map_dis is not None:
        plot_map_dis.save(shortest_dis_heatmap_filename)
    plot_map_post_qa = plot_congestion_heatmap_interactive(edges, post_qa_congestion_df, offset_deg=OFFSET_DEG, vmin=vmin, vmax=vmax)
    if plot_map_post_qa is not None:
        plot_map_post_qa.save(post_qa_heatmap_filename)
    plot_map_random = plot_congestion_heatmap_interactive(edges, random_routes_df, offset_deg=OFFSET_DEG, vmin=vmin, vmax=vmax)
    if plot_map_random is not None:
        plot_map_random.save(random_routes_heatmap_filename)
    plot_map_post_gurobi = plot_congestion_heatmap_interactive(edges, post_gurobi_congestion_df, offset_deg=OFFSET_DEG, vmin=vmin, vmax=vmax)
    if plot_map_post_gurobi is not None:
        plot_map_post_gurobi.save(post_gurobi_heatmap_file)


def save_congestion_summary(
    session: Any,
    edges: pd.DataFrame,
    congestion_df: pd.DataFrame,
    post_qa_congestion_df: pd.DataFrame,
    shortest_routes_dur_df: pd.DataFrame,
    shortest_routes_dis_df: pd.DataFrame,
    random_routes_df: pd.DataFrame,
    post_gurobi_df: pd.DataFrame,
    run_configs_id: RunConfig,
    iteration_id: int
) -> None:
    """Save congestion summary to the database."""
    congestion_df_grouped = congestion_df.groupby('edge_id', as_index=False).agg({'congestion_score': 'sum'})
    merged = pd.DataFrame({'edge_id': edges.drop_duplicates(subset='edge_id')['edge_id']})
    merged = merged.merge(congestion_df_grouped[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_all'}), on='edge_id', how='left')  # type: ignore
    merged = merged.merge(post_qa_congestion_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_post_qa'}), on='edge_id', how='left')  # type: ignore
    merged = merged.merge(shortest_routes_dur_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_shortest_dur'}), on='edge_id', how='left')  # type: ignore
    merged = merged.merge(shortest_routes_dis_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_shortest_dis'}), on='edge_id', how='left')  # type: ignore
    merged = merged.merge(random_routes_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_random'}), on='edge_id', how='left')  # type: ignore
    merged = merged.merge(post_gurobi_df[['edge_id', 'congestion_score']].rename(columns={'congestion_score': 'congestion_post_gurobi'}), on='edge_id', how='left')  # type: ignore
    merged = merged.fillna(0)
    records = [
        CongestionSummary(
            run_configs_id=run_configs_id,
            iteration_id=iteration_id,
            edge_id=int(row['edge_id']),
            congestion_all=float(row['congestion_all']),
            congestion_post_qa=float(row['congestion_post_qa']),
            congestion_shortest_dur=float(row['congestion_shortest_dur']),
            congestion_shortest_dis=float(row['congestion_shortest_dis']),
            congestion_random=float(row['congestion_random']),
            congestion_post_gurobi=float(row['congestion_post_gurobi'])
        )
        for _, row in merged.iterrows()
    ]
    session.add_all(records)
    session.commit()


def save_dist_dur_summary(session, run_configs_id, iteration_id):
    """Save distance and duration summary to the database using dist_dur_results.sql."""
    import os
    from models import DistDurSummary
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sql_file = os.path.join(base_dir, 'sql', 'dist_dur_results.sql')
    with open(sql_file, 'r') as f:
        sql = f.read()
    # Replace %s placeholders with :run_configs_id and :iteration_id for SQLAlchemy
    sql = sql.replace('%s', ':run_configs_id', 1)
    sql = sql.replace('%s', ':iteration_id', 1)
    result = session.execute(sa_text(sql), {'run_configs_id': run_configs_id, 'iteration_id': iteration_id})
    row = result.fetchone()
    if row is not None:
        summary = DistDurSummary(
            run_configs_id=run_configs_id,
            iteration_id=iteration_id,
            shortest_dist=row[0],
            shortest_dur=row[1],
            post_qa_dist=row[2],
            post_qa_dur=row[3],
            rnd_dist=row[4],
            rnd_dur=row[5],
            post_gurobi_dist=row[6],
            post_gurobi_dur=row[7]

        )
        session.add(summary)
        session.commit()
    else:
        logger.warning("No distance/duration summary results found for run_configs_id=%s, iteration_id=%s", run_configs_id, iteration_id)


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
    from dwave.system import LeapHybridBQMSampler, LeapHybridCQMSampler

    dwave_constraints_check= True
    bqm = dimod.BQM.from_qubo(Q)
    num_variables = len(bqm.variables)
    num_linear = len(bqm.linear)
    num_quadratic = len(bqm.quadratic)
    num_biases = num_linear + num_quadratic

    sampler = LeapHybridCQMSampler()
    max_vars_cqm = sampler.properties["maximum_number_of_variables"]
    max_biases_cqm = sampler.properties["maximum_number_of_biases"]

    sampler = LeapHybridBQMSampler()
    max_vars_bqm = sampler.properties["maximum_number_of_variables"]
    max_biases_bqm = sampler.properties["maximum_number_of_biases"]
    print("Number of variables:", num_variables)
    print("Number of linear biases:", num_linear)
    print("Number of quadratic biases:", num_quadratic)
    print("Total number of biases:", num_biases)
    print("BQM Solver maximum_number_of_variables:", max_vars_bqm)
    print("BQM Solver maximum_number_of_biases:", max_biases_bqm)
    print("CQM Solver maximum_number_of_variables:", max_vars_cqm)
    print("CQM Solver maximum_number_of_biases:", max_biases_cqm)
    if num_variables > max(max_vars_bqm, max_vars_cqm):
        dwave_constraints_check = False
        logger.warning("Too many variables for this solver!")
    if num_biases > max(max_biases_bqm, max_biases_cqm):
        dwave_constraints_check = False
        logger.warning("Too many biases for this solver!")
    return dwave_constraints_check


def compute_objective_value(Q: np.ndarray, routes: dict, filtered_vehicle_ids: list, t: int) -> float:
    """
    Compute the QUBO objective value: x^T Q x using a sparse QUBO dictionary.

    Parameters:
    - Q: sparse QUBO dict with (i, j): value
    - routes: dict {vehicle_id: route_index (zero-based)}
    - filtered_vehicle_ids: list of vehicle_ids used in QUBO
    - t: number of route alternatives per vehicle

    Returns:
    - float: objective value
    """
    active_indices = [
        i * t + routes[vehicle_id]
        for i, vehicle_id in enumerate(filtered_vehicle_ids)
        if vehicle_id in routes
    ]

    obj = 0.0

    # Diagonal terms
    for i in active_indices:
        obj += Q.get((i, i), 0.0)

    # Off-diagonal terms (only i < j to avoid double counting)
    for idx1 in range(len(active_indices)):
        i = active_indices[idx1]
        for idx2 in range(idx1 + 1, len(active_indices)):
            j = active_indices[idx2]
            obj += Q.get((i, j), Q.get((j, i), 0.0))

    return float(obj)


def load_iteration_data(session, run_configs_id: int, iteration_id: int):
    """Load existing city, vehicle, route, and congestion data from the database."""
    if CENTER_COORDS is not None:
        # Look for existing city subset with matching coordinates
        lat, lon = CENTER_COORDS  # type: ignore
        city = session.query(City).filter_by(
            name=CITY_NAME,
            center_lat=lat,
            center_lon=lon,
            radius_km=RADIUS_KM,
            is_subset=True
        ).first()
        
    else:
        # Look for existing full city
        city = session.query(City).filter_by(
            name=CITY_NAME,
            is_subset=False
        ).first()

    print(f"Loaded city: {city.name} (ID: {city.city_id})")
    nodes, edges = get_city_data_from_db(session, city.city_id)
    
    vehicles_gdf = pd.read_sql(
        sa_text("SELECT * FROM vehicles WHERE run_configs_id = :rc AND iteration_id = :it"),
        session.bind,
        params={"rc": run_configs_id, "it": iteration_id}
    )

    vehicle_routes_df = pd.read_sql(
        sa_text("SELECT * FROM vehicle_routes WHERE run_configs_id = :rc AND iteration_id = :it"),
        session.bind,
        params={"rc": run_configs_id, "it": iteration_id}
    )

    congestion_df = pd.read_sql(
        sa_text("SELECT * FROM congestion_map WHERE run_configs_id = :rc AND iteration_id = :it"),
        session.bind,
        params={"rc": run_configs_id, "it": iteration_id}
    )

    return edges, vehicles_gdf, vehicle_routes_df, congestion_df


def main(RC_ID, IT_ID) -> None:
    """Main workflow for traffic simulation and QUBO optimization."""
    start_time = datetime.now()
    try:
        with create_db_session() as session:
            edges, vehicles_gdf, vehicle_routes_df, congestion_df = load_iteration_data(session, RC_ID, IT_ID)
            weights_df, duration_penalty_df = get_congestion_weights(session, RC_ID, IT_ID)
            weights_df.to_csv(CONGESTION_WEIGHTS_FILENAME, index=False)

            # Filtering for QUBO
            all_vehicle_ids = vehicles_gdf["vehicle_id"].tolist()

            # QUBO matrix
            t = get_k_alternatives(session, RC_ID, IT_ID)

            new_iteration_id = create_simulation_iteration(session, RC_ID)
            if new_iteration_id is None:
                logger.warning("No new iteration created. Aborting optimization.")
                return
            Q, filtered_vehicle_ids, affected_edges_df, n_Q, t_Q, lambda_penalty = build_and_save_qubo_matrix(
                vehicle_routes_df, congestion_df, weights_df, duration_penalty_df, session, RC_ID, new_iteration_id, t,
                LAMBDA_STRATEGY, LAMBDA_VALUE, FILTERING_PERCENTAGE
            )

            # Before QA testing, check BQM/QUBO limits
            '''
            dwave_constraints_check = check_bqm_against_solver_limits(Q)
            if not dwave_constraints_check:
                logger.warning("BQM/QUBO constraints check failed! Exiting workflow.")
                return
            '''

            # QA testing
            qa_result = qa_testing(
                Q=Q,
                run_configs_id=RC_ID,
                iteration_id=new_iteration_id,
                session=session,
                n=len(filtered_vehicle_ids),
                t=t,
                vehicle_ids=filtered_vehicle_ids,
                lambda_strategy=LAMBDA_STRATEGY,
                lambda_value=lambda_penalty,
                comp_type=COMP_TYPE,
                num_reads=10,
                vehicle_routes_df=vehicle_routes_df,
                dwave_constraints_check=0#dwave_constraints_check
            )
            qa_assignment = qa_result['assignment']
            qa_annealing_time = qa_result['duration']

            # Post-QA congestion
            post_qa_congestion_df, selected_routes = post_qa_congestion(
                session=session,
                run_configs_id=RC_ID,
                iteration_id=new_iteration_id,
                all_vehicle_ids=all_vehicle_ids,
                optimized_vehicle_ids=filtered_vehicle_ids,
                qa_assignment=qa_assignment,
                method=ROUTE_METHOD
            )

            result, obj_val = solve_qubo_with_gurobi(Q, n_Q, t_Q, RC_ID, new_iteration_id, session, time_limit_seconds=qa_annealing_time)

            post_gurobi_congestion_df, gurobi_routes = post_gurobi_congestion(
                session=session,
                run_configs_id=RC_ID,
                iteration_id=new_iteration_id,
                all_vehicle_ids=all_vehicle_ids,
                optimized_vehicle_ids=filtered_vehicle_ids,
                gurobi_assignment=result,
                t=t,
                method=ROUTE_METHOD
            )


            post_qa_congestion_df = post_qa_congestion_df.groupby('edge_id', as_index=False).agg({'congestion_score': 'sum'})
            post_qa_congestion_df = pd.DataFrame(post_qa_congestion_df)

            post_gurobi_congestion_df = post_gurobi_congestion_df.groupby('edge_id', as_index=False).agg({'congestion_score': 'sum'})
            post_gurobi_congestion_df = pd.DataFrame(post_gurobi_congestion_df)

            #random_routes_df, random_routes = compute_random_routes(session, RC_ID, IT_ID, store_results=False)
            #shortest_routes_dur_df, shortest_routes_dur = compute_shortest_routes(session, RC_ID, IT_ID, method="duration",store_results=False)
            #shortest_routes_dis_df, shortest_routes_dis = compute_shortest_routes(session, RC_ID, IT_ID, method="distance",store_results=False)


            selected_routes_dict = {
                sr.vehicle_id: sr.route_id - 1  # Subtract 1 to make route index zero-based!
                for sr in selected_routes
            }


            qa_obj_val = compute_objective_value(Q, selected_routes_dict, filtered_vehicle_ids, t)
            print(f"QA Objective Value: {qa_obj_val}")


            gurobi_routes_dict = {
                sr.vehicle_id: sr.route_id - 1  # Subtract 1 to make route index zero-based!
                for sr in gurobi_routes
            }
        

            gurobi_obj_val = compute_objective_value(Q, gurobi_routes_dict, filtered_vehicle_ids, t)
            print(f"Gurobi Objective Value: {gurobi_obj_val}")
            
            

            objective_values = [
            ("qa", qa_obj_val),
            ("gurobi", gurobi_obj_val),
            ("random", 0),
            ("shortest_duration", 0),
            ("shortest_distance", 0),
        ]

            # First, clear old entries for this run+iteration (if needed)
            session.query(ObjectiveValue).filter_by(
                run_configs_id=RC_ID,
                iteration_id=new_iteration_id
            ).delete()

            # Then add new records
            records = [
                ObjectiveValue(
                    run_configs_id=RC_ID,
                    iteration_id=new_iteration_id,
                    method=comp,
                    objective_value=value,
                )
                for comp, value in objective_values
            ]

            session.add_all(records)
            session.commit()
            logger.info(f"Stored objective values for run_config {RC_ID}, iteration {new_iteration_id}.")




            logger.info("Workflow completed successfully!")
    except Exception as e:
        logger.error("Workflow failed: %s", str(e), exc_info=True)
    finally:
        end_time = datetime.now()
        logger.info("Total computational time: %s", end_time - start_time)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QUBO traffic optimization pipeline.")
    parser.add_argument("RC_ID", type=int, help="ID of the existing run_config")
    parser.add_argument("IT_ID", type=int, help="ID of the existing iteration to start from")
    args = parser.parse_args()

    main(args.RC_ID, args.IT_ID)
# This script can be run from the command line with:
# python main_test.py <run_config_id> <iteration_id>
