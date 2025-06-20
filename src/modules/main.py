# ---------- IMPORT MODULES ----------
import pandas as pd
import numpy as np
from get_city_graph import get_city_graph
from get_city_data_from_db import get_city_data_from_db
from store_city_to_db import store_city_to_db
from get_or_create_run_config import get_or_create_run_config
from create_iteration import create_iteration
from generate_vehicles import generate_vehicles
from generate_vehicle_routes_STree import generate_vehicle_routes
from generate_congestion import generate_congestion
from plot_congestion_heatmap import plot_congestion_heatmap, plot_congestion_heatmap_interactive
from filter_routes_for_qubo import filter_routes_for_qubo
from get_congestion_weights import get_congestion_weights
from normalize_congestion_weights import normalize_congestion_weights
from congestion_weights import congestion_weights
from qubo_matrix import qubo_matrix
from compute_shortest_routes import compute_shortest_routes
from kill_alldb_processes import kill_alldb_processes
import datetime

# ---------- CONFIGURATION ----------
from sqlalchemy.orm import sessionmaker
from models import * #City, Node, Edge, RunConfig, Iteration, Vehicle, VehicleRoute, CongestionMap, RoutePoint  # adjust to your actual model imports

API_KEY = 'AIzaSyCawuGvoiyrHOh3RyJdq7yzFCcG5smrZCI'
CITY_NAME = "Košice, Slovakia"
N_VEHICLES = 25000
K_ALTERNATIVES = 3
MIN_LENGTH = 200
MAX_LENGTH = 6000
TIME_STEP = 10
TIME_WINDOW = 800
DIST_THRESH = 10
SPEED_DIFF_THRESH = 2


# ---------- WORKFLOW ----------
def main():

    #kill_alldb_processes()

    Session = sessionmaker(bind=engine, autocommit=False)
    session = Session()
    print("Starting workflow at:", datetime.now())    
    
    # Step 1: Get or generate city
    city = session.query(City).filter_by(name=CITY_NAME).first()
    if not city:
        nodes, edges = get_city_graph(CITY_NAME)
        city = store_city_to_db(session, CITY_NAME, nodes, edges, City, Node, Edge)
    nodes, edges = get_city_data_from_db(session, city.id)

    # Step 2: Create or fetch run configuration
    run_config = get_or_create_run_config(
        session, city.id, RunConfig, N_VEHICLES, K_ALTERNATIVES,
        MIN_LENGTH, MAX_LENGTH, TIME_STEP, TIME_WINDOW
    )

    # Step 3: Create iteration
    iteration_id = create_iteration(session, run_config.id, None, Iteration)
    if iteration_id is None:
        return

    # Step 4: Generate vehicles
    print("Generate vehicles at:", datetime.now())
    vehicles_gdf = generate_vehicles(
        session, Vehicle, run_config.id, iteration_id,
        edges, N_VEHICLES, MIN_LENGTH, MAX_LENGTH
    )

    # Step 5: Generate vehicle routes
    print("Generate vehicle routes at:", datetime.now())
    generate_vehicle_routes(
        session, VehicleRoute, RoutePoint,
        API_KEY, run_config.id, iteration_id,
        vehicles_gdf, edges, K_ALTERNATIVES, TIME_STEP, TIME_WINDOW
    )

    # Step 6: Compute congestion
    print("Compute congestion at:", datetime.now())
    congestion_df = generate_congestion(
        session, CongestionMap,
        run_config.id, iteration_id,
        DIST_THRESH, SPEED_DIFF_THRESH
    )

    # Step 7: Get congestion weights
    weights_df = get_congestion_weights(session, run_config.id, iteration_id)
    vehicle_ids = vehicles_gdf["vehicle_id"].tolist()

    # Step 8: QUBO matrix construction
    Q, filtered_vehicle_ids = qubo_matrix(
        N_VEHICLES, K_ALTERNATIVES, weights_df, vehicle_ids, lambda_strategy="normalized", fixed_lambda=1.0, filtering_percentage=0.1, max_qubo_size=None )

    N_FILTERED = len(filtered_vehicle_ids)
    print("Filtered vehicles number:", N_FILTERED)

    # Step 9: Save QUBO matrix to CSV
    def qubo_dict_to_dataframe(Q, size):
        matrix = np.zeros((size, size))
        for (i, j), v in Q.items():
            matrix[i][j] = v
            if i != j:
                matrix[j][i] = v  # ensure symmetry for display
        return pd.DataFrame(matrix)

    size = N_FILTERED * K_ALTERNATIVES
    Q_df = qubo_dict_to_dataframe(Q, size)
    Q_df.to_csv("files/qubo_matrix.csv", index=False)

    # Step 10: Visualize original congestion
    plot_map = plot_congestion_heatmap_interactive(edges, congestion_df, offset_deg=0.0000025)
    plot_map.save("files/congestion_heatmap.html")

    # Step 11: Visualize congestion for shortest routes (duration)
    shortest_routes_dur_df = compute_shortest_routes(session, run_config.id, iteration_id, method="duration")
    plot_map_dur = plot_congestion_heatmap_interactive(edges, shortest_routes_dur_df, offset_deg=0.0000025)
    plot_map_dur.save("files/shortest_routes_dur_congestion_heatmap.html")

    # Step 12: Visualize congestion for shortest routes (distance)
    shortest_routes_dis_df = compute_shortest_routes(session, run_config.id, iteration_id, method="distance")
    plot_map_dis = plot_congestion_heatmap_interactive(edges, shortest_routes_dis_df, offset_deg=0.0000025)
    plot_map_dis.save("files/shortest_routes_dis_congestion_heatmap.html")

    session.close()
    print("Workflow completed successfully!")
    
if __name__ == "__main__":
    main()
