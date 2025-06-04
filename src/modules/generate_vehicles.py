import random
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
from sklearn.cluster import KMeans

def generate_vehicles(session, Vehicle, run_config_id, iteration_id, edges_gdf, nr_vehicles, min_length, max_length):
    vehicles = []
    vehicle_id = 0

    # def sample_spatially_diverse_edge(edges_gdf, n_clusters=500):
    #     edges_proj = edges_gdf.to_crs(epsg=3857)
    #     centroids = edges_proj.geometry.centroid
    #     coords = np.array([[p.x, p.y] for p in centroids])
    #     kmeans = KMeans(n_clusters=min(n_clusters, len(edges_gdf)), n_init='auto').fit(coords)
    #     edges_gdf['cluster'] = kmeans.labels_
    #     sampled_cluster = random.choice(edges_gdf['cluster'].unique())
    #     return edges_gdf[edges_gdf['cluster'] == sampled_cluster].sample(n=1).iloc[0]

    for _ in range(nr_vehicles):
        valid_vehicle = False
        retries = 0
        max_retries = 100

        while not valid_vehicle and retries < max_retries:
            retries += 1

            #print(edges_gdf)
            origin_edge = edges_gdf.sample(n=1).iloc[0]# sample_spatially_diverse_edge(edges_gdf)
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

                vehicledb = Vehicle(
                    vehicle_id=vehicle_id,
                    run_configs_id=run_config_id,
                    iteration_id=iteration_id,
                    origin_edge_id=vehicle['origin_edge_id'],
                    origin_position_on_edge=vehicle['origin_position_on_edge'],
                    origin_geometry=vehicle['origin_geometry'],
                    destination_edge_id=vehicle['destination_edge_id'],
                    destination_position_on_edge=vehicle['destination_position_on_edge'],
                    destination_geometry=vehicle['destination_geometry']
                )
                vehicles.append(vehicle)
                session.add(vehicledb)

        if not valid_vehicle:
            print(f"⚠️  Skipped a vehicle after {max_retries} retries due to distance constraints.")

    session.commit()

    vehicles_df = pd.DataFrame(vehicles)
    vehicles_df['origin_geometry'] = vehicles_df['origin_geometry'].apply(lambda x: Point(x.x, x.y))
    vehicles_df['destination_geometry'] = vehicles_df['destination_geometry'].apply(lambda x: Point(x.x, x.y))
    vehicles_gdf = gpd.GeoDataFrame(vehicles_df, geometry='origin_geometry', crs='EPSG:4326')

    return vehicles_gdf
