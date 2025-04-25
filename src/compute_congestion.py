from collections import defaultdict
from sqlalchemy.orm import sessionmaker
from db_tables import *

def compute_route_overlap_congestion(car_routes, run_id, iteration_id, precision=5, red_light_penalty_scaling=0.25):
    """
    Calculates congestion score considering overlapping routes and traffic light penalties.
    """
    from collections import defaultdict
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=engine)
    session = Session()

    point_to_cars = defaultdict(set)

    # Step 1: Unique route paths per car
    unique_routes_per_car = defaultdict(set)
    for car in car_routes:
        car_id = car['car_id']
        for route in car.get('routes', []):
            rounded_route = tuple((round(lat, precision), round(lon, precision)) for lat, lon in route['geometry'])
            unique_routes_per_car[car_id].add(rounded_route)

    # Step 2: Build point-to-car mapping
    for car_id, routes in unique_routes_per_car.items():
        for route in routes:
            for point in set(route):
                point_to_cars[point].add(car_id)

    # Step 3: Point frequency
    point_freq = {pt: len(cars) for pt, cars in point_to_cars.items()}

    # Step 4: Compute congestion score
    congestion_scores = {}
    for car in car_routes:
        car_id = car['car_id']
        for idx, route in enumerate(car.get('routes', [])):
            total = 0
            for lat, lon in route['geometry']:
                key = (round(lat, precision), round(lon, precision))
                total += point_freq.get(key, 0)

            avg_congestion = total / max(len(route['geometry']), 1)

            # 🚦 Fetch red light wait time from DB for this car and route
            car_obj = session.query(Car).filter_by(
                run_configs_id=run_id,
                iteration_id=iteration_id,
                car_id=car_id
            ).first()

            if not car_obj:
                print(f" Car not found: {car_id}")
                continue

            car_route = session.query(CarRoute).filter_by(
                car_id=car_obj.id,
                iteration_id=iteration_id,
                route_index=idx
            ).first()

            red_wait = car_route.total_red_light_wait if car_route else 0
            red_penalty = (red_wait/60) * red_light_penalty_scaling  # e.g., 30 sec * 0.01 = 0.3 penalty

            congestion_scores[(car_id, idx)] = avg_congestion + red_penalty

    session.close()
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
