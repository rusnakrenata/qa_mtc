import sys
import os
import pandas as pd
import numpy as np
import logging

# If running from project root, add src/modules to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'modules'))

from sqlalchemy.orm import sessionmaker
from src.modules.models import engine, CongestionMap
from src.modules.get_congestion_weights import get_congestion_weights
from src.modules.filter_routes_for_qubo import filter_vehicles_by_congested_edges_and_limit
from src.modules.qubo_matrix_pen_on_diagonals import qubo_matrix
from src.modules.generate_congestion import generate_congestion
from src.modules.config import K_ALTERNATIVES, LAMBDA_STRATEGY, LAMBDA_VALUE, DIST_THRESH, SPEED_DIFF_THRESH

logging.basicConfig(level=logging.INFO)

def test_pipeline(run_config_id: int, iteration_id: int):
    # --- Setup DB session ---
    Session = sessionmaker(bind=engine, autocommit=False)
    session = Session()

    # 1. Select 2 vehicles
    vehicles = pd.read_sql(
        f"SELECT DISTINCT vehicle_id FROM vehicle_routes WHERE run_configs_id = {run_config_id} AND iteration_id = {iteration_id} LIMIT 2",
        session.bind)
    vehicle_ids = vehicles['vehicle_id'].tolist()
    print('Selected vehicle IDs:', vehicle_ids)
    if len(vehicle_ids) < 2:
        print('Not enough vehicles found for this run_config_id and iteration_id.')
        return

    # 2. Calculate congestion
    congestion_df = generate_congestion(session, CongestionMap, run_config_id, iteration_id, DIST_THRESH, SPEED_DIFF_THRESH)
    print('Congestion DF:')
    print(congestion_df.head())

    # 3. Calculate weights
    weights_df = get_congestion_weights(session, run_config_id, iteration_id)
    print('Weights DF:')
    print(weights_df.head())

    # 4. Filtering (should just return the 2 selected vehicles)
    filtered_vehicle_ids = [vid for vid in vehicle_ids]  # For 2 vehicles, no further filtering
    print('Filtered vehicle IDs:', filtered_vehicle_ids)

    # 5. QUBO matrix
    n = len(filtered_vehicle_ids)
    t = K_ALTERNATIVES
    Q, _ = qubo_matrix(n, t, congestion_df, weights_df, filtered_vehicle_ids, lambda_strategy=LAMBDA_STRATEGY, fixed_lambda=LAMBDA_VALUE)
    print('QUBO matrix (dict, first 10 entries):')
    for i, (k, v) in enumerate(Q.items()):
        if i >= 10: break
        print(k, v)

    session.close()

if __name__ == "__main__":
    # Set your test run_config_id and iteration_id here
    test_pipeline(run_config_id=1, iteration_id=1) 