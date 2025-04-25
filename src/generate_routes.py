import requests
import polyline
import time
from db_tables import *
from sqlalchemy.orm import sessionmaker
from shapely.geometry import Point
from geopy.distance import geodesic

def get_routes_from_google(origin, destination, api_key):
    base_url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "mode": "driving",
        "alternatives": "true",
        "departure_time": "now",
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def count_unique_traffic_lights_on_route(route_polyline, run_id, iteration_id):
    Session = sessionmaker(bind=engine)
    session = Session()

    # You can optionally filter lights by city_id if needed later
    lights = session.query(TrafficLight).all()
    traffic_light_points = [(l.id, l.lat, l.lon, l.red_cycle) for l in lights]
    session.close()

    visited_lights = set()
    total_estimated_wait = 0

    for latlon in route_polyline:
        for light_id, lat, lon, red_cycle in traffic_light_points:
            if light_id in visited_lights:
                continue
            if geodesic(latlon, (lat, lon)).meters <= 15:
                visited_lights.add(light_id)
                total_estimated_wait += red_cycle
                break

    return {
        "light_count": len(visited_lights),
        "light_ids": list(visited_lights),
        "total_red_light_wait": total_estimated_wait
    }


def collect_routes(cars, api_key, run_id, iteration_id, K_ALTERNATIVES=3):
    all_car_routes = []

    for car in cars:
        origin = car['src_coords']
        destination = car['dst_coords']
        response = get_routes_from_google(origin, destination, api_key)

        car_routes = []
        if response and 'routes' in response:
            for route in response['routes'][:K_ALTERNATIVES]:
                poly = polyline.decode(route['overview_polyline']['points'])
                leg = route['legs'][0]
                duration = leg['duration']['value']
                distance = leg['distance']['value']
                traffic_time = leg.get('duration_in_traffic', {}).get('value', duration)

                light_info = count_unique_traffic_lights_on_route(poly, run_id, iteration_id)

                car_routes.append({
                    "geometry": poly,
                    "duration": duration,
                    "distance": distance,
                    "duration_in_traffic": traffic_time,
                    "traffic_light_count": light_info["light_count"],
                    "traffic_light_ids": light_info["light_ids"],
                    "total_red_light_wait": light_info["total_red_light_wait"],
                    "contains_traffic_light": light_info["light_count"] > 0
                })

            if len(car_routes) < K_ALTERNATIVES and len(car_routes) > 0:
                car_routes += [car_routes[0]] * (K_ALTERNATIVES - len(car_routes))

        if not car_routes:
            print(f" No routes returned for Car {car['car_id']}")

        all_car_routes.append({
            "car_id": car['car_id'],
            "origin": car['src_coords'],
            "destination": car['dst_coords'],
            "routes": car_routes
        })

        time.sleep(1)

    return all_car_routes


def store_in_db_car_routes(cars, api_key, K_ALTERNATIVES, run_id, iteration_id):
    Session = sessionmaker(bind=engine)
    session = Session()

    all_car_routes = collect_routes(cars, api_key, run_id, iteration_id, K_ALTERNATIVES)

    for route_data in all_car_routes:
        car_id = route_data["car_id"]

        car_obj = session.query(Car).filter_by(
            run_configs_id=run_id,
            iteration_id=iteration_id,
            car_id=car_id
        ).first()

        if not car_obj:
            print(f"Could not find Car (car_id={car_id}, run={run_id}, iter={iteration_id})")
            continue

        for idx, route in enumerate(route_data["routes"]):
            car_route = CarRoute(
                car_id=car_obj.id,
                iteration_id=iteration_id,
                route_index=idx,
                geometry=route["geometry"],
                duration=route["duration"],
                distance=route["distance"],
                duration_in_traffic=route["duration_in_traffic"],
                traffic_light_count=route["traffic_light_count"],
                contains_traffic_light=route["contains_traffic_light"],
                total_red_light_wait=route["total_red_light_wait"]
            )
            session.add(car_route)

    session.commit()
    session.close()
    print("All car routes stored.")
    return all_car_routes
