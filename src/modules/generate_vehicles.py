import random
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
from sklearn.cluster import KMeans

def generate_vehicles(session, Vehicle, run_config_id, iteration_id, edges_gdf, nr_vehicles, min_length, max_length):
    vehicles = []
    vehicle_records = []
    vehicle_id = 0

    for _ in range(nr_vehicles):
        valid_vehicle = False
        retries = 0
        max_retries = 100

        while not valid_vehicle and retries < max_retries:
            retries += 1

            origin_edge = edges_gdf.sample(n=1).iloc[0]
            origin_position_on_edge = random.random()
            origin_line = origin_edge['geometry']
            origin_point = origin_line.interpolate(origin_position_on_edge, normalized=True)

            destination_edge = edges_gdf.sample(n=1).iloc[0]
            destination_position_on_edge = random.random()
            destination_line = destination_edge['geometry']
            destination_point = destination_line.interpolate(destination_position_on_edge, normalized=True)

            distance = geodesic((origin_point.y, origin_point.x), (destination_point.y, destination_point.x)).meters

            if min_length <= distance <= max_length:
                valid_vehicle = True
                vehicle_id += 1

                vehicle = {
                    'vehicle_id': vehicle_id,
                    'origin_edge_id': origin_edge['id'],
                    'origin_position_on_edge': origin_position_on_edge,
                    'origin_geometry': origin_point,
                    'destination_edge_id': destination_edge['id'],
                    'destination_position_on_edge': destination_position_on_edge,
                    'destination_geometry': destination_point
                }
                vehicles.append(vehicle)

                vehicle_record = Vehicle(
                    vehicle_id=vehicle_id,
                    run_configs_id=run_config_id,
                    iteration_id=iteration_id,
                    origin_edge_id=origin_edge['id'],
                    origin_position_on_edge=origin_position_on_edge,
                    origin_geometry=origin_point,
                    destination_edge_id=destination_edge['id'],
                    destination_position_on_edge=destination_position_on_edge,
                    destination_geometry=destination_point
                )
                vehicle_records.append(vehicle_record)

        if not valid_vehicle:
            print(f"⚠️ Skipped a vehicle after {max_retries} retries due to distance constraints.")

    # Efficient bulk insert
    session.bulk_save_objects(vehicle_records)
    session.commit()

    # Create GeoDataFrame for return
    vehicles_df = pd.DataFrame(vehicles)
    vehicles_df['origin_geometry'] = vehicles_df['origin_geometry'].apply(lambda x: Point(x.x, x.y))
    vehicles_df['destination_geometry'] = vehicles_df['destination_geometry'].apply(lambda x: Point(x.x, x.y))
    vehicles_gdf = gpd.GeoDataFrame(vehicles_df, geometry='origin_geometry', crs='EPSG:4326')

    return vehicles_gdf

