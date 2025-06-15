from concurrent.futures import ProcessPoolExecutor
import asyncio
import aiohttp
import nest_asyncio
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree
from pyproj import Transformer
from datetime import datetime
from scipy.spatial import KDTree
import numpy as np
import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from utils import (
    create_linestring_from_polyline,
    get_point_on_line,
    convert_valhalla_leg_to_google_like_steps,
    async_get_routes_from_valhalla,
    find_closest_osm_edge,
    find_approx_nearest_edge,
    calculate_initial_bearing,
    bearing_to_cardinal
)

nest_asyncio.apply()

BATCH_SIZE = 1000

def get_points_in_time_window(steps, time_step, time_window):
    cumulative_times = []
    time_acc = 0
    for step in steps:
        duration = step['duration']['value']
        cumulative_times.append((time_acc, time_acc + duration))
        time_acc += duration

    points = []
    step_idx = 0
    step_start, step_end = cumulative_times[step_idx]

    for step_time in range(time_step, time_window, time_step):
        while step_time >= step_end and step_idx + 1 < len(cumulative_times):
            step_idx += 1
            step_start, step_end = cumulative_times[step_idx]

        step = steps[step_idx]
        duration = step['duration']['value']
        if duration == 0:
            continue

        polyline_points = step['polyline']['points']
        gdf_line = create_linestring_from_polyline(polyline_points)
        fraction = (step_time - step_start) / duration
        point_on_line = get_point_on_line(gdf_line.geometry[0], fraction)

        points.append({
            'location': point_on_line,
            'time': step_time,
            'speed': step['distance']['value'] / duration
        })

    return points

def process_vehicle_route(vehicle_data):
    vehicle, vehicle_idx, route_data, edges_proj_dict, coords, time_step, time_window = vehicle_data
    route_points_records = []
    vehicle_routes_records = []

    edges_proj = edges_proj_dict['edges']
    edge_geometries = [geom if isinstance(geom, BaseGeometry) else geom.__geo_interface__ for geom in edges_proj['geometry'].values.tolist()]
    edge_tree = STRtree(edge_geometries)
    transformer = Transformer.from_crs("EPSG:4326", edges_proj_dict['crs'], always_xy=True)

    origin = (vehicle['origin_geometry'].x, vehicle['origin_geometry'].y)

    for route_id, route in enumerate(route_data, start=1):
        summary = route['summary']

        vehicle_routes_records.append({
            "vehicle_id": vehicle['vehicle_id'],
            "route_id": route_id,
            "duration": summary['time'],
            "distance": summary['length'] * 1000
        })

        point_id = 0
        steps = convert_valhalla_leg_to_google_like_steps(route['leg'])
        points = get_points_in_time_window(steps, time_step, time_window)

        for point in points:
            point_id += 1
            lat = point['location'].y
            lon = point['location'].x
            time_val = point['time']
            speed = point['speed']

            previous_location = points[point_id - 2]['location'] if point_id > 1 else Point(origin[0], origin[1])

            edge = find_closest_osm_edge(lon, lat, edges_proj, edge_tree, transformer=transformer)

            edge_id = edge['id']

            bearing = calculate_initial_bearing(previous_location.y, previous_location.x, lat, lon)
            cardinal = bearing_to_cardinal(bearing)

            route_points_records.append({
                "vehicle_id": vehicle['vehicle_id'],
                "route_id": route_id,
                "point_id": point_id,
                "edge_id": edge_id,
                "cardinal": cardinal,
                "speed": speed,
                "lat": lat,
                "lon": lon,
                "time": time_val
            })

    return vehicle_routes_records, route_points_records

def generate_vehicle_routes(session, VehicleRoute, RoutePoint,
    api_key, run_config_id, iteration_id, vehicles, edges_gdf,
    max_nr_of_alternative_routes, time_step, time_window):

    async def batched_valhalla_fetch(vehicles, max_concurrent=20):
        semaphore = asyncio.Semaphore(max_concurrent)
        async with aiohttp.ClientSession() as http_session:
            async def fetch(vehicle):
                async with semaphore:
                    origin = (vehicle['origin_geometry'].x, vehicle['origin_geometry'].y)
                    dest = (vehicle['destination_geometry'].x, vehicle['destination_geometry'].y)
                    return await async_get_routes_from_valhalla(http_session, origin, dest, max_nr_of_alternative_routes)

            tasks = [fetch(vehicle) for _, vehicle in vehicles.iterrows()]
            return await asyncio.gather(*tasks)

    loop = asyncio.get_event_loop()
    routes_data_list = loop.run_until_complete(batched_valhalla_fetch(vehicles))

    edges_proj = edges_gdf.to_crs(epsg=3857)
    coords = np.array([[pt.x, pt.y] for pt in edges_proj.geometry.centroid])
    edges_proj_dict = {'edges': edges_proj, 'crs': edges_proj.crs}

    vehicle_data_list = [
        (vehicle, idx, routes_data_list[idx], edges_proj_dict, coords, time_step, time_window)
        for idx, vehicle in vehicles.iterrows()
        if routes_data_list[idx]  # skip if no route found
    ]

    with ProcessPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(process_vehicle_route, vehicle_data_list))

    all_vehicle_routes = []
    all_route_points = []
    for vehicle_routes, route_points in results:
        all_vehicle_routes.extend(vehicle_routes)
        all_route_points.extend(route_points)

    for i in range(0, len(all_vehicle_routes), BATCH_SIZE):
        batch = all_vehicle_routes[i:i + BATCH_SIZE]
        session.bulk_save_objects([
            VehicleRoute(
                vehicle_id=rec['vehicle_id'],
                run_configs_id=run_config_id,
                iteration_id=iteration_id,
                route_id=rec['route_id'],
                duration=rec['duration'],
                distance=rec['distance'],
                duration_in_traffic=None
            ) for rec in batch
        ])
        session.commit()

    for i in range(0, len(all_route_points), BATCH_SIZE):
        batch = all_route_points[i:i + BATCH_SIZE]
        session.bulk_save_objects([
            RoutePoint(
                vehicle_id=rec['vehicle_id'],
                run_configs_id=run_config_id,
                iteration_id=iteration_id,
                route_id=rec['route_id'],
                point_id=rec['point_id'],
                edge_id=rec['edge_id'],
                cardinal=rec['cardinal'],
                speed=rec['speed'],
                lat=rec['lat'],
                lon=rec['lon'],
                time=rec['time']
            ) for rec in batch
        ])
        session.commit()

    routes_df = pd.DataFrame(all_route_points)
    return routes_df
