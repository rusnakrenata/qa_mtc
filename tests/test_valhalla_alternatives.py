from numpy import True_
import requests
import polyline
import random
from geopy.distance import geodesic

# Košice bounding box (approximate)
LAT_MIN, LAT_MAX = 48.65, 48.78
LON_MIN, LON_MAX = 21.18, 21.32

VALHALLA_URL = "http://147.232.204.254:8002/route"

def random_point():
    lat = random.uniform(LAT_MIN, LAT_MAX)
    lon = random.uniform(LON_MIN, LON_MAX)
    return {"lat": lat, "lon": lon}

def get_routes(origin, destination, max_alternates=2, exclude_highways=False):
    payload = {
        "locations": [origin, destination],
        "costing": "auto",
        "alternates": max_alternates > 1,
        "number_of_alternates": max_alternates - 1 if max_alternates > 1 else 0
    }
    if exclude_highways:
        payload["costing_options"] = {
            "auto": {
                "exclude_trunk": True
            }
        }
    response = requests.post(VALHALLA_URL, json=payload)
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return []
    data = response.json()
    routes = [data.get("trip")] + data.get("alternates", [])
    return routes

def print_routes(routes, label):
    print(f"\n--- {label} ---")
    seen_shapes = set()
    for idx, route in enumerate(routes):
        if not route or "legs" not in route or not route["legs"]:
            print(f"Route {idx+1}: No valid route found or missing 'legs'.")
            continue
        summary = route.get("summary", {})
        shape = route["legs"][0].get("shape")
        if not shape or shape in seen_shapes:
            continue
        seen_shapes.add(shape)
        coords = polyline.decode(shape, precision=6)
        print(f"Route {idx+1}: {summary.get('length', '?')} km, {summary.get('time', '?')} sec, shape hash: {hash(shape)}")
        print(f"  Start: {coords[0]}, End: {coords[-1]}")

def main():
    # Find two random points at least 3 km apart
    min_distance_km = 3.0
    while True:
        origin = random_point()
        destination = random_point()
        dist = geodesic((origin["lat"], origin["lon"]), (destination["lat"], destination["lon"])).km
        if dist >= min_distance_km:
            break
    print(f"Random origin: {origin}")
    print(f"Random destination: {destination}")
    print(f"Great-circle distance: {dist:.2f} km")

    # Standard
    standard_routes = get_routes(origin, destination, max_alternates=2)
    # Exclude highways
    penalized_routes = get_routes(origin, destination, max_alternates=2, exclude_highways=True)

    print_routes(standard_routes, "Standard Valhalla")
    print_routes(penalized_routes, "Exclude Highways (motorway, trunk, primary)")

    # Compare all unique shapes
    all_shapes = set()
    for route in standard_routes + penalized_routes:
        shape = route["legs"][0]["shape"]
        all_shapes.add(shape)
    print(f"\nTotal unique alternatives found: {len(all_shapes)}")

if __name__ == "__main__":
    main()