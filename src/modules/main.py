# ---------- IMPORT MODULES ----------
from get_city_graph import get_city_graph
from get_city_data_from_db import get_city_data_from_db
from store_city_to_db import store_city_to_db
from get_or_create_run_config import get_or_create_run_config
from create_iteration import create_iteration
from generate_vehicles import generate_vehicles
from generate_vehicle_routes import generate_vehicle_routes
from generate_congestion import generate_congestion
from plot_congestion_heatmap import plot_congestion_heatmap

# ---------- CONFIGURATION ----------
from sqlalchemy.orm import sessionmaker
from models import * #City, Node, Edge, RunConfig, Iteration, Vehicle, VehicleRoute, CongestionMap, RoutePoint  # adjust to your actual model imports

API_KEY = 'AIzaSyCawuGvoiyrHOh3RyJdq7yzFCcG5smrZCI'
CITY_NAME = "Košice, Slovakia"
N_VEHICLES = 500
K_ALTERNATIVES = 3
MIN_LENGTH = 0
MAX_LENGTH = 10000
TIME_STEP = 10
TIME_WINDOW = 600
DIST_THRESH = 10
SPEED_DIFF_THRESH = 2

Session = sessionmaker(bind=engine)
session = Session()

# ---------- WORKFLOW ----------
def main():
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

    # Step 3: Create iteration
    iteration_id = create_iteration(session, run_config.id, None, Iteration)
    if iteration_id is None:
        return

    # Step 4: Generate vehicles
    vehicles_gdf = generate_vehicles(
        session, Vehicle, run_config.id, iteration_id,
        edges, N_VEHICLES, MIN_LENGTH, MAX_LENGTH
    )

    # Step 5: Generate vehicle routes
    routes = generate_vehicle_routes(
        session, VehicleRoute, RoutePoint,
        API_KEY, run_config.id, iteration_id, vehicles_gdf,
        edges, K_ALTERNATIVES, TIME_STEP, TIME_WINDOW
    )

    # Step 6: Compute congestion
    congestion_df = generate_congestion(
        session, CongestionMap,
        run_config.id, iteration_id,
        DIST_THRESH, SPEED_DIFF_THRESH
    )

    # Step 7: Plot heatmap
    plot_congestion_heatmap(edges, congestion_df)

if __name__ == "__main__":
    main()
