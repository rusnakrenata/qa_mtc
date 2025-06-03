import math
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import polyline
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from pyproj import Transformer
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import osmnx as ox

def get_point_at_percentage(line_wkt, percentage):
    line = LineString([(21.2159377, 48.7126189), (21.2159939, 48.7125398), (21.2162822, 48.7121463)])
    total_length = line.length
    target_length = total_length * percentage
    point_at_percentage = line.interpolate(target_length)
    return point_at_percentage.x, point_at_percentage.y

def create_geodataframe_from_coords(coords):
    points = [Point(coord['lng'], coord['lat']) for coord in coords]
    gdf = gpd.GeoDataFrame(geometry=points)
    gdf.set_crs("EPSG:4326", allow_override=True, inplace=True)
    return gdf

def create_linestring_from_polyline(polyline_points):
    line = LineString(polyline_points)
    gdf = gpd.GeoDataFrame(geometry=[line])
    gdf.set_crs("EPSG:4326", allow_override=True, inplace=True)
    return gdf

def get_point_on_line(line, percentage):
    if not 0 <= percentage <= 1:
        raise ValueError("Percentage must be between 0 and 1.")
    total_length = line.length
    target_distance = total_length * percentage
    return line.interpolate(target_distance)

# def get_routes_from_google(origin, destination, api_key, max_nr_of_alternative_routes):
#     base_url = "https://maps.googleapis.com/maps/api/directions/json"
#     params = {
#         "origin": f"{origin[0]},{origin[1]}",
#         "destination": f"{destination[0]},{destination[1]}",
#         "mode": "driving",
#         "alternatives": "true",
#         "departure_time": "now",
#         "key": api_key
#     }
#     response = requests.get(base_url, params=params)
#     if response.status_code == 200:
#         return response.json().get("routes", [])[:max_nr_of_alternative_routes]
#     else:
#         print(response.text)
#         return None

def calculate_initial_bearing(start_lat, start_lng, end_lat, end_lng):
    lat1 = math.radians(start_lat)
    lat2 = math.radians(end_lat)
    diff_long = math.radians(end_lng - start_lng)
    x = math.sin(diff_long) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diff_long))
    initial_bearing = math.atan2(x, y)
    return (math.degrees(initial_bearing) + 360) % 360

def bearing_to_cardinal(bearing):
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ix = round(bearing / 45) % 8
    return directions[ix]

def find_closest_osm_edge(lat, lng, edges_gdf, edge_tree, transformer=None):
    x, y = transformer.transform(lng, lat)
    point = Point(x, y)
    index = edge_tree.nearest(point)
    edge_row = edges_gdf.iloc[index]
    return {
        'id': edge_row.get('id', None),
        'geometry': edge_row.geometry,
        'distance_meters': 0  # can add actual distance if needed
    }

def animate_vehicles(G, vehicle_paths, interval=10):
    fig, ax = ox.plot_graph(G, node_color='black', node_size=5, edge_linewidth=0.5, bgcolor='white', show=False, close=False, ax=None)
    scatters = [ax.scatter(path['path'][0][0], path['path'][0][1], c='red', s=20, label=f"Vehicle {path['id']}") for path in vehicle_paths]

    def update(frame):
        for idx, vehicle in enumerate(vehicle_paths):
            if frame < len(vehicle['path']):
                scatters[idx].set_offsets(vehicle['path'][frame])
        return scatters

    total_frames = max(len(v['path']) for v in vehicle_paths)
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval * 1000, blit=True, repeat=False)
    plt.legend()
    plt.show()

def compute_spatial_density_with_speed(df, dist_thresh=10, speed_diff_thresh=2):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs='EPSG:4326').to_crs(epsg=3857)
    results = []
    grouped = gdf.groupby(['time', 'edge_id', 'cardinal'])

    for (time, edge_id, cardinal), group in grouped:
        if len(group) < 2:
            results.append({
                'time': time,
                'edge_id': edge_id,
                'cardinal': cardinal,
                'vehicle_count': group['vehicle_id'].nunique(),
                'congested_pair_count': 0,
                'avg_speed': group['speed'].mean() if 'speed' in group else None,
                'avg_pairwise_distance': None
            })
            continue

        coords = np.array([(geom.x, geom.y) for geom in group.geometry])
        speeds = np.array(group['speed'])
        vehicle_ids = np.array(group['vehicle_id'])

        dist_matrix = squareform(pdist(coords))
        speed_matrix = squareform(pdist(speeds[:, None]))
        valid_mask = ~np.equal.outer(vehicle_ids, vehicle_ids) & np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)
        congested_mask = (dist_matrix < dist_thresh) & (speed_matrix < speed_diff_thresh) & valid_mask

        results.append({
            'time': time,
            'edge_id': edge_id,
            'cardinal': cardinal,
            'vehicle_count': group['vehicle_id'].nunique(),
            'congested_pair_count': np.sum(congested_mask),
            'avg_speed': speeds.mean(),
            'avg_pairwise_distance': dist_matrix[valid_mask].mean() if valid_mask.any() else None
        })

    return pd.DataFrame(results)

def normalize_valhalla_route(route, route_index=0):
    if "summary" not in route:
        route = route.get("trip", route)
    return {
        "index": route_index,
        "summary": route.get("summary", {}),
        "leg": route["legs"][0] if route.get("legs") else None,
        "distance_km": route.get("summary", {}).get("length"),
        "duration_sec": route.get("summary", {}).get("time"),
    }

async def async_get_routes_from_valhalla(session, origin, destination, max_nr_of_alternative_routes):
    base_url = "http://77.93.155.81:8002/route"
    payload = {
        "locations": [{"lat": origin[1], "lon": origin[0]}, {"lat": destination[1], "lon": destination[0]}],
        "costing": "auto",
        "alternates": max_nr_of_alternative_routes > 1,
        "number_of_alternates": max_nr_of_alternative_routes - 1 if max_nr_of_alternative_routes > 1 else 0
    }

    async with session.post(base_url, json=payload) as response:
        if response.status == 200:
            data = await response.json()
            raw_routes = [data.get("trip")] + data.get("alternates", [])
            return [normalize_valhalla_route(route, idx) for idx, route in enumerate(raw_routes)]
        else:
            print(f"Error: {response.status} - {await response.text()}")
            return None

def convert_valhalla_leg_to_google_like_steps(leg):
    full_coords = polyline.decode(leg['shape'], precision=6)
    steps = []
    for maneuver in leg['maneuvers']:
        step = {
            'polyline': {'points': full_coords[maneuver['begin_shape_index']:maneuver['end_shape_index'] + 1]},
            'duration': {'value': maneuver['time']},
            'distance': {'value': int(maneuver['length'] * 1000)}
        }
        steps.append(step)
    return steps
