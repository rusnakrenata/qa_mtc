from datetime import datetime
import pandas as pd

def store_city_to_db(session, city_name, nodes, edges, City, Node, Edge):
    node_count = len(nodes)
    edge_count = len(edges)

    city = City(
        name=city_name,
        node_count=node_count,
        edge_count=edge_count,
        created_at=datetime.utcnow()
    )
    session.add(city)
    session.commit()

    for _, node in nodes.iterrows():
        node_data = {
            'city_id': city.id,
            'osmid': node.get('osmid'),
            'x': node['x'] if not pd.isna(node['x']) else None,
            'y': node['y'] if not pd.isna(node['y']) else None,
            'street_count': node.get('street_count'),
            'highway': node.get('highway'),
            'railway': node.get('railway'),
            'junction': node.get('junction'),
            'geometry': str(node['geometry']) if node['geometry'] is not None else None
        }
        session.add(Node(**node_data))

    session.commit()

    for _, edge in edges.iterrows():
        edge_data = {
            'city_id': city.id,
            'u': edge.get('u'),
            'v': edge.get('v'),
            'length': str(edge['length']) if not pd.isna(edge.get('length')) else None,
            'geometry': str(edge['geometry']) if edge['geometry'] is not None else None
        }
        session.add(Edge(**edge_data))

    session.commit()
    return city
