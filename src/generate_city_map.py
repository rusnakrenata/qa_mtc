# generate_map_with_lights.py
import osmnx as ox
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from db_tables import *

def get_city_graph(city_name):
    G = ox.graph_from_place(city_name, network_type='drive')
    G.graph['crs'] = 'epsg:4326'
    return G

def store_in_db_city(city_name):
    Session = sessionmaker(bind=engine)
    session = Session()

    city = session.query(City).filter_by(name=city_name).first()

    if city:
        print(f" City '{city_name}' already exists in the database.")
        session.close()
        G = get_city_graph(city_name)
        return G

    print(f" Generating and storing new city '{city_name}'...")
    G = get_city_graph(city_name)

    node_count = len(G.nodes)
    edge_count = len(G.edges)

    city = City(
        name=city_name,
        node_count=node_count,
        edge_count=edge_count,
        created_at=datetime.utcnow()
    )
    session.add(city)

    city_added = session.query(City).filter_by(name=city_name).first()

    # Store traffic lights from OSM
    lights = ox.features_from_place(city_name, tags={'highway': 'traffic_signals'})
    for _, row in lights.iterrows():
        if 'geometry' in row:
            lat = row.geometry.y
            lon = row.geometry.x
            traffic_light = TrafficLight(
                city_id=city_added.id,
                lat=lat,
                lon=lon,
                created_at=datetime.utcnow()
            )
            session.add(traffic_light)

    session.commit()
    session.close()

    print(f" City '{city_name}' saved with {node_count} nodes and {edge_count} edges and traffic lights.")
    return G
