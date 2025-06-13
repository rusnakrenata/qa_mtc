import asyncio
import aiohttp
import nest_asyncio
import pandas as pd
from asyncio import Semaphore
from shapely.geometry import Point
from shapely.strtree import STRtree
from shapely.geometry.base import BaseGeometry
from pyproj import Transformer
from datetime import datetime

from utils import (
    create_linestring_from_polyline,
    get_point_on_line,
    convert_valhalla_leg_to_google_like_steps,
    find_closest_osm_edge,
    calculate_initial_bearing,
    bearing_to_cardinal,
    async_get_routes_from_valhalla
)

nest_asyncio.apply()

def generate_vehicle_routes(
    session, VehicleRoute, RoutePoint,
    api_key, run_config_id, iteration_id, vehicles, edges_gdf, max_nr_of_alternative_routes, time_step, time_window
):
    def get_points_in_time_window(steps):
        iter_step = 0
        time_start = 0
        points = []
        for step_time in range(time_step, time_window, time_step):
            for index in range(len(steps))[iter_step:]:
                step = steps[index]
                if time_start <= step_time < time_start + step['duration']['value']:
                    polyline_points = step['polyline']['points']
                    gdf_line = create_linestring_from_polyline(polyline_points)
                    point_on_line = get_point_on_line(gdf_line.geometry[0], (step_time - time_start) / step['duration']['value'])
                    points.append({
                        'location': point_on_line,
                        'time': step_time,
                        'speed': step['distance']['value'] / step['duration']['value'] if step['duration']['value'] != 0 else 0
                    })
                    break
                if step_time + time_step > time_start + step['duration']['value']:
                    time_start += step['duration']['value']
                    iter_step = index + 1
        return points

    print("Starting async route fetching...")
    print(datetime.now())
    route_points_records = []

    # async def fetch_all_routes(vehicles):
    #     async with aiohttp.ClientSession() as http_session:
    #         tasks = []
    #         for _, vehicle in vehicles.iterrows():
    #             origin = (vehicle['origin_geometry'].x, vehicle['origin_geometry'].y)
    #             destination = (vehicle['destination_geometry'].x, vehicle['destination_geometry'].y)
    #             tasks.append(async_get_routes_from_valhalla(http_session, origin, destination, max_nr_of_alternative_routes))
    #         return await asyncio.gather(*tasks)

    # loop = asyncio.get_event_loop()
    # routes_data_list = loop.run_until_complete(fetch_all_routes(vehicles))

    async def batched_valhalla_fetch(vehicles, max_concurrent=20):
        semaphore = Semaphore(max_concurrent)
        async with aiohttp.ClientSession() as session:
            async def fetch(vehicle):
                async with semaphore:
                    origin = (vehicle['origin_geometry'].x, vehicle['origin_geometry'].y)
                    dest = (vehicle['destination_geometry'].x, vehicle['destination_geometry'].y)
                    return await async_get_routes_from_valhalla(session, origin, dest, max_nr_of_alternative_routes)

            tasks = [fetch(vehicle) for _, vehicle in vehicles.iterrows()]
            return await asyncio.gather(*tasks)

    # Usage
    loop = asyncio.get_event_loop()
    routes_data_list = loop.run_until_complete(batched_valhalla_fetch(vehicles, max_concurrent=20))



    print("Valhalla routes generated.")
    print(datetime.now())

    routes = []
    edges_proj = edges_gdf.to_crs(epsg=3857)
    edge_geometries = [geom if isinstance(geom, BaseGeometry) else geom.__geo_interface__ for geom in edges_proj['geometry'].values.tolist()]
    edge_tree = STRtree(edge_geometries)
    transformer = Transformer.from_crs("EPSG:4326", edges_proj.crs, always_xy=True)

    for vehicle_idx, vehicle in vehicles.iterrows():
        routes_data = routes_data_list[vehicle_idx]
        if not routes_data:
            print(f"No routes found for vehicle {vehicle['vehicle_id']}")
            continue

        origin = (vehicle['origin_geometry'].x, vehicle['origin_geometry'].y)

        for route_id, route in enumerate(routes_data, start=1):
            summary = route['summary']
            vehicle_route = VehicleRoute(
                vehicle_id=vehicle['vehicle_id'],
                run_configs_id=run_config_id,
                iteration_id=iteration_id,
                route_id=route_id,
                duration=summary['time'],
                distance=summary['length'] * 1000,
                duration_in_traffic=None
            )
            session.add(vehicle_route)

            point_id = 0
            steps = convert_valhalla_leg_to_google_like_steps(route['leg'])
            #print("convert_valhalla_leg_to_google_like_steps done.")
            #print(datetime.now())
            points = get_points_in_time_window(steps)
            #print("get_points_in_time_window done.")
            #print(datetime.now())
            

            for point in points:
                point_id += 1
                lat = point['location'].y
                lon = point['location'].x
                time_val = point['time']
                speed = point['speed']

                previous_location = points[point_id - 2]['location'] if point_id > 1 else Point(origin[0], origin[1])
                edge = find_closest_osm_edge(lon, lat, edges_proj, edge_tree, transformer=transformer)
                #print("find_closest_osm_edge done.")
                #print(datetime.now())
                edge_id = edge['id']
                bearing = calculate_initial_bearing(previous_location.y, previous_location.x, lat, lon)
                #print("calculate_initial_bearing done.")
                #print(datetime.now())
                cardinal = bearing_to_cardinal(bearing)
                #print("bearing_to_cardinal done.")
                #print(datetime.now())

                route_point = RoutePoint(
                    vehicle_id=vehicle_route.vehicle_id,
                    run_configs_id=vehicle_route.run_configs_id,
                    iteration_id=vehicle_route.iteration_id,
                    route_id=vehicle_route.route_id,
                    point_id=point_id,
                    edge_id=edge_id,
                    cardinal=cardinal,
                    speed=speed,
                    lat=lat,
                    lon=lon,
                    time=time_val
                )
                session.add(route_point)

                route_points_records.append({
                    "vehicle_id": vehicle_route.vehicle_id,
                    "route_id": vehicle_route.route_id,
                    "point_id": point_id,
                    "edge_id": edge_id,
                    "cardinal": cardinal,
                    "speed": speed,
                    "lat": lat,
                    "lon": lon,
                    "time": time_val
                })

            routes.append(vehicle_route)

    session.commit()
    print("Routes stored in DB done.")
    print(datetime.now())
    routes_df = pd.DataFrame(route_points_records)
    #print(routes_df)
    return routes_df #, routes
