from collections import defaultdict
from sqlalchemy.orm import sessionmaker
from db_tables import *

def compute_route_overlap_congestion(car_routes, run_id, iteration_id, precision=5, traffic_light_weight = 0.25):
    Session = sessionmaker(bind=engine)
    session = Session()

    # Get the city ID from the run configuration
    run_config = session.query(RunConfig).filter_by(id=run_id).first()
    city_id = run_config.city_id

    # Load all traffic lights for that city
    traffic_lights = session.query(TrafficLight).filter_by(city_id=city_id).all()
    session.close()

    # Build penalty lookup: (rounded_lat, rounded_lon) -> penalty (normalized)
    light_penalty = {}
    for light in traffic_lights:
        key = (round(light.lat, precision), round(light.lon, precision))
        penalty = (light.red_cycle / 60.0) * traffic_light_weight  # Normalize: 60s red = 1.0 penalty
        light_penalty[key] = penalty

    # --- Step 1: Unique routes per car (directional) ---
    point_to_cars = defaultdict(set)
    unique_routes_per_car = defaultdict(set)

    for car in car_routes:
        car_id = car['car_id']
        for route in car.get('routes', []):
            rounded_route = tuple((round(lat, precision), round(lon, precision)) for lat, lon in route['geometry'])
            unique_routes_per_car[car_id].add(rounded_route)

    # --- Step 2: Frequency of cars per point ---
    for car_id, unique_routes in unique_routes_per_car.items():
        seen_points = set()
        for route in unique_routes:
            for point in route:
                if point not in seen_points:
                    point_to_cars[point].add(car_id)
                    seen_points.add(point)

    point_freq = {pt: len(cars) for pt, cars in point_to_cars.items()}

    # --- Step 3: Congestion score = overlap + red light penalties ---
    congestion_scores = {}
    for car in car_routes:
        car_id = car['car_id']
        for idx, route in enumerate(car.get('routes', [])):
            total = 0
            for lat, lon in route['geometry']:
                key = (round(lat, precision), round(lon, precision))
                base = point_freq.get(key, 0)
                penalty = light_penalty.get(key, 0)
                total += base + penalty
            avg_congestion = total / max(len(route['geometry']), 1)
            congestion_scores[(car_id, idx)] = avg_congestion

    return congestion_scores, point_freq





def store_in_db_congestion_scores(car_routes, run_id, iteration_id):
    Session = sessionmaker(bind=engine)
    session = Session()

    congestion_scores, _ = compute_route_overlap_congestion(car_routes, run_id, iteration_id, precision=5)
    for (car_logical_id, route_index), score in congestion_scores.items():
        # Find the internal Car.id for this car_id
        car_obj = session.query(Car).filter_by(
            run_configs_id=run_id,
            iteration_id=iteration_id,
            car_id=car_logical_id
        ).first()

        if not car_obj:
            print(f" Could not find Car with car_id={car_logical_id}, run={run_id}, iteration={iteration_id}")
            continue

        row = CongestionScore(
            car_id=car_obj.id,
            route_index=route_index,
            run_id=run_id,
            iteration_id=iteration_id,
            score=score
        )
        session.add(row)

    session.commit()
    session.close()
    print(f" Stored {len(congestion_scores)} congestion scores.")
    return congestion_scores
