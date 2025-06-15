# ---------- IMPORT MODULES ----------
import pandas as pd
import numpy as np
from get_city_graph import get_city_graph
from get_city_data_from_db import get_city_data_from_db
from store_city_to_db import store_city_to_db
from get_or_create_run_config import get_or_create_run_config
from create_iteration import create_iteration
from generate_vehicles import generate_vehicles
from generate_vehicle_routes_opti import generate_vehicle_routes
from generate_congestion import generate_congestion
from plot_congestion_heatmap import plot_congestion_heatmap, plot_congestion_heatmap_interactive
from filter_routes_for_qubo import filter_routes_for_qubo
from get_congestion_weights import get_congestion_weights
from normalize_congestion_weights import normalize_congestion_weights
from congestion_weights import congestion_weights
from qubo_matrix import qubo_matrix
from compute_shortest_routes import compute_shortest_routes
import datetime

# ---------- CONFIGURATION ----------
from sqlalchemy.orm import sessionmaker
from models import * #City, Node, Edge, RunConfig, Iteration, Vehicle, VehicleRoute, CongestionMap, RoutePoint  # adjust to your actual model imports

API_KEY = 'AIzaSyCawuGvoiyrHOh3RyJdq7yzFCcG5smrZCI'
CITY_NAME = "Košice, Slovakia"
N_VEHICLES = 25000
K_ALTERNATIVES = 3
MIN_LENGTH = 200
MAX_LENGTH = 10000
TIME_STEP = 10
TIME_WINDOW = 1200
DIST_THRESH = 10
SPEED_DIFF_THRESH = 2


# ---------- WORKFLOW ----------
def main():
    
    Session = sessionmaker(bind=engine)
    session = Session()
    print("Starting workflow at:", datetime.now())    
    
    # Step 1: Get or generate city
    city = session.query(City).filter_by(name=CITY_NAME).first()
    if not city:
        nodes, edges = get_city_graph(CITY_NAME)
        city = store_city_to_db(session, CITY_NAME, nodes, edges, City, Node, Edge)
    else:
        nodes, edges = get_city_data_from_db(session, city.id)

    # Step 2: Create or fetch run configuration
    run_config = get_or_create_run_config(
        session, city.id, RunConfig, N_VEHICLES, K_ALTERNATIVES,
        MIN_LENGTH, MAX_LENGTH, TIME_STEP, TIME_WINDOW
    )
    print("Iterations at start:", datetime.now())
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
    routes_df = generate_vehicle_routes(
        session, VehicleRoute, RoutePoint,
        API_KEY, run_config.id, iteration_id, vehicles_gdf,
        edges, K_ALTERNATIVES, TIME_STEP, TIME_WINDOW
    )

 
    # Step 6: Compute congestion
    print("Compute congestion at:", datetime.now())

    congestion_df = generate_congestion(
        session, CongestionMap,
        run_config.id, iteration_id,
        DIST_THRESH, SPEED_DIFF_THRESH
    )
    
    plot_map = plot_congestion_heatmap_interactive(edges, congestion_df,offset_deg=0.000025)
    plot_map
    plot_map.save("files/congestion_heatmap.html")


    # Step 7: Filter routes for QUBO
    filtered_vehicles = filter_routes_for_qubo(congestion_df, threshold_percentile=0.9)
    #print(filtered_vehicles)
    N_FILTERED = len(filtered_vehicles)
    print("Number of elements:", N_FILTERED)

    # Step 8: Compute wights from congestion
    weights_df = get_congestion_weights(session, run_config.id, iteration_id)
    print(weights_df)
    

    #weights_normalized = normalize_congestion_weights(weights_df, N_FILTERED, K_ALTERNATIVES, filtered_vehicles)
    #weights_wo_normalization, max_weight = congestion_weights(weights_df, N_FILTERED, K_ALTERNATIVES, filtered_vehicles)
    #print(weights_normalized)

    # Step 9: QUBO
    Q, weights, _ = qubo_matrix(N_FILTERED, K_ALTERNATIVES, weights_df, filtered_vehicles, lambda_strategy="normalized", fixed_lambda=1.0)
 
    #print(Q)

    # Step 10: Save QUBO matrix to CSV
    # Note: The QUBO matrix is a dictionary of dictionaries, where Q[i][j] is the weight for the interaction between i and j.
    #QA testing qa_testing.py(N_FILTERED, K_ALTERNATIVES, weights_df, filtered_vehicles, run_config.id, iteration_id, session, lambda_strategy="normalized", fixed_lambda=1.0, comp_type='hybrid', num_reads=10):
    # post_qa_congestion.py(session, run_config.id, iteration_id, dist_thresh=10.0, speed_diff_thresh=2.0) -- need to check the vehicle_ids matching back to original
    # qa_congestion_df = post_qa_congestion(session, run_config.id, iteration_id, dist_thresh=10.0, speed_diff_thresh=2.0)
    # plot_congestion_heatmap(edges_gdf, qa_congestion_df)



    def qubo_dict_to_dataframe(Q, size):
        matrix = np.zeros((size, size))
        for (i, j), v in Q.items():
            matrix[i][j] = v
            if i != j:
                matrix[j][i] = v  # ensure symmetry for display
        return pd.DataFrame(matrix)

    # Store Q to csv
    size = N_FILTERED * K_ALTERNATIVES
    Q_df = qubo_dict_to_dataframe(Q, size)
    #print(Q_df.round(3))
    Q_df.to_csv("files/qubo_matrix.csv", index=False)

    

    shortes_routes_dur_df = compute_shortest_routes(session, run_config.id, iteration_id, method="duration")
    plot_map_dur = plot_congestion_heatmap_interactive(edges, shortes_routes_dur_df,offset_deg=0.000025)
    plot_map_dur
    plot_map_dur.save("files/shortest_routes_dur_congestion_heatmap.html")


    shortes_routes_dis_df = compute_shortest_routes(session, run_config.id, iteration_id, method="distance")
    plot_map_dis = plot_congestion_heatmap_interactive(edges, shortes_routes_dis_df,offset_deg=0.000025)
    plot_map_dis
    plot_map_dis.save("files/shortest_routes_dis_congestion_heatmap.html")

    session.close()
    print("Workflow completed successfully!")

if __name__ == "__main__":
    main()
