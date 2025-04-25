from collections import defaultdict
from sqlalchemy.orm import sessionmaker
from db_tables import *

def compute_route_overlap_congestion(car_routes, precision=5):
    """
    Calculates congestion score based on overlapping route points,
    considering direction and avoiding duplicate route contributions per car.

    Returns:
        congestion_scores: dict {(car_id, route_index): congestion_score}
        point_freq: dict {point: number of unique cars using the point}
    """
    point_to_cars = defaultdict(set)

    # Step 1: Collect unique (directed) routes per car
    unique_routes_per_car = defaultdict(set)
    for car in car_routes:
        car_id = car['car_id']
        for route in car.get('routes', []):
            rounded_route = tuple((round(lat, precision), round(lon, precision)) for lat, lon in route['geometry'])
            unique_routes_per_car[car_id].add(rounded_route)

    # Step 2: Register points used in unique routes (directional)
    for car_id, unique_routes in unique_routes_per_car.items():
        seen_points = set()
        for route in unique_routes:
            for point in route:
                if point not in seen_points:
                    point_to_cars[point].add(car_id)
                    seen_points.add(point)

    # Step 3: Compute frequency per point
    point_freq = {pt: len(cars) for pt, cars in point_to_cars.items()}

    # Step 4: Compute average congestion score per car/route
    congestion_scores = {}
    for car in car_routes:
        car_id = car['car_id']
        for idx, route in enumerate(car.get('routes', [])):
            total = 0
            for lat, lon in route['geometry']:
                key = (round(lat, precision), round(lon, precision))
                total += point_freq.get(key, 0)
            avg_congestion = total / max(len(route['geometry']), 1)
            congestion_scores[(car_id, idx)] = avg_congestion

    return congestion_scores, point_freq




def store_in_db_congestion_scores(car_routes, run_id, iteration_id):
    Session = sessionmaker(bind=engine)
    session = Session()

    congestion_scores, _ = compute_route_overlap_congestion(car_routes, precision=5)
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
