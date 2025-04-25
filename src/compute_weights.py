from collections import defaultdict
from itertools import combinations
from sqlalchemy.orm import sessionmaker
from db_tables import *

def calculate_weights_from_congestion_scores(congestion_scores):
    """
    Computes adjusted weights w(i,j,k) by subtracting self-overlap from congestion scores.

    Args:
        congestion_scores: dict {(i, k): score}

    Returns:
        weights[i][j][k]: shared weight from congestion (ignoring self-use)
    """
    weights = defaultdict(lambda: defaultdict(dict))
    
    all_keys = congestion_scores.keys()
    cars = sorted(set(i for i, _ in all_keys))
    ks = sorted(set(k for _, k in all_keys))

    for i, j in combinations(cars, 2):
        for k in ks:
            if (i, k) in congestion_scores and (j, k) in congestion_scores:
                score_i = max(0, congestion_scores[(i, k)] - 1)
                score_j = max(0, congestion_scores[(j, k)] - 1)
                avg = (score_i + score_j) / 2
                weights[i][j][k] = avg
                weights[j][i][k] = avg
            else:
                weights[i][j][k] = 0
                weights[j][i][k] = 0

    return weights

def store_in_db_weights(congestion_scores, run_id, iteration_id):
    """
    Store congestion weights in the database.

    Args:
        weights: nested dict of form weights[i][j][k] where i, j are logical car_ids
        run_id: ID of the run configuration
        iteration_id: Iteration number for the car generation
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    weights = calculate_weights_from_congestion_scores(congestion_scores)
    
    # Step 1: Build mapping from logical car_id -> Car.id (DB primary key)
    car_rows = session.query(Car).filter_by(run_configs_id=run_id, iteration_id=iteration_id).all()
    car_id_map = {car.car_id: car.id for car in car_rows}

    if not car_id_map:
        print(f"No car_id mapping found for run_id={run_id}, iteration_id={iteration_id}")
        session.close()
        return

    # Step 2: Store weights
    count = 0
    for i, j_dict in weights.items():
        for j, k_dict in j_dict.items():
            for k, w in k_dict.items():
                if i not in car_id_map or j not in car_id_map:
                    print(f"Skipping pair ({i}, {j}) due to missing mapping.")
                    continue
                row = CongestionWeight(
                    run_id=run_id,
                    iteration_id=iteration_id,
                    car_i_id=car_id_map[i],
                    car_j_id=car_id_map[j],
                    route_index=k,
                    weight=w
                )
                session.add(row)
                count += 1

    session.commit()
    session.close()
    print(f" Stored {count} congestion weights (run_id={run_id}, iteration_id={iteration_id}).")
    return weights

