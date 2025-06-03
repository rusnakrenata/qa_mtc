from sqlalchemy import text
import pandas as pd
import geopandas as gpd
from shapely import wkt

def get_city_data_from_db(session, city_id):
    nodes_query = session.execute(
        text(f"SELECT id, geometry AS geometry FROM nodes WHERE city_id = {city_id}")
    )
    edges_query = session.execute(
        text(f"SELECT id, geometry AS geometry FROM edges WHERE city_id = {city_id}")
    )

    nodes_df = pd.DataFrame(nodes_query.fetchall(), columns=["id", "geometry"])
    edges_df = pd.DataFrame(edges_query.fetchall(), columns=["id", "geometry"])

    nodes_df['geometry'] = nodes_df['geometry'].apply(wkt.loads)
    edges_df['geometry'] = edges_df['geometry'].apply(wkt.loads)

    nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry='geometry', crs='EPSG:4326')
    edges_gdf = gpd.GeoDataFrame(edges_df, geometry='geometry', crs='EPSG:4326')

    return nodes_gdf, edges_gdf
