import random
import networkx as nx
from sqlalchemy.orm import sessionmaker
from db_tables import *

def generate_car_od_pairs(G, n, max_dist_m, min_dist_m):
    if G.graph['crs'] != 'epsg:4326':
        raise ValueError("Graph must be in EPSG:4326 (unprojected WGS84)")

    nodes = list(G.nodes)
    cars = []
    i = 0
    attempts = 0
    max_attempts = n * 10  # safeguard to avoid infinite loop

    while i < n and attempts < max_attempts:
        src = random.choice(nodes)
        attempts += 1

        # Search beyond max_dist to get enough nodes, but filter later
        lengths = nx.single_source_dijkstra_path_length(G, src, cutoff=max_dist_m * 10, weight='length')

        candidate_dsts = [
            node for node, dist in lengths.items()
            if node != src and min_dist_m <= dist <= max_dist_m
        ]

        if not candidate_dsts:
            continue

        dst = random.choice(candidate_dsts)

        src_coords = (round(G.nodes[src]['y'], 6), round(G.nodes[src]['x'], 6))
        dst_coords = (round(G.nodes[dst]['y'], 6), round(G.nodes[dst]['x'], 6))

        cars.append({
            "car_id": i,
            "src_node": src,
            "dst_node": dst,
            "src_coords": src_coords,
            "dst_coords": dst_coords
        })
        i += 1

    if i < n:
        print(f" Only generated {i} out of {n} cars after {attempts} attempts.")
    else:
        print(f" Successfully generated {n} cars after {attempts} attempts.")

    return cars



def store_in_db_cars(G, n, max_dist_m, min_dist_m, run_id): 
    Session = sessionmaker(bind=engine)
    session = Session()

    # Get latest iteration number
    last_iteration = session.query(Car).filter_by(run_configs_id=run_id).order_by(desc(Car.iteration_id)).first()
    next_iteration = 1 if not last_iteration else last_iteration.iteration_id + 1

    # Generate new car OD pairs
    cars = generate_car_od_pairs(G, n, max_dist_m, min_dist_m)

    for c in cars:
        car = Car(
            car_id = c["car_id"],
            run_configs_id=run_id,
            iteration_id=next_iteration,  
            src_node=c["src_node"],
            dst_node=c["dst_node"],
            src_lat=c["src_coords"][0],
            src_lon=c["src_coords"][1],
            dst_lat=c["dst_coords"][0],
            dst_lon=c["dst_coords"][1]
        )
        session.add(car)

    session.commit()
    session.close()

    print(f"Stored  run #{run_id}, and {len(cars)} cars.")