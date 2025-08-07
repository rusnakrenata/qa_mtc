from typing import Any
from models import City, Node, Edge
from get_city_graph import get_city_graph
from store_city_to_db import store_city_to_db
from get_city_data_from_db import get_city_data_from_db

def get_or_create_city(
    session,
    city_name: str,
    center_coords: tuple | None = None,
    radius_km: float | None = None,
    attraction_point: tuple | None = None,
    d_alternatives: int | None = None
) -> Any:
    """
    Get or create a city (full or subset) in the database.
    """

    # Prepare filter with string keys
    filters = {
        "name": city_name,
        "is_subset": center_coords is not None or attraction_point is not None or radius_km is not None or d_alternatives is not None
    }

    if center_coords:
        filters["center_lat"] = center_coords[0]
        filters["center_lon"] = center_coords[1]
    if radius_km is not None:
        filters["radius_km"] = radius_km
    if attraction_point:
        filters["attraction_lat"] = attraction_point[0]
        filters["attraction_lon"] = attraction_point[1]
    if d_alternatives is not None:
        filters["d_alternatives"] = d_alternatives

    # Use getattr to dynamically apply all filters
    filter_conditions = [getattr(City, key) == value for key, value in filters.items()]
    city = session.query(City).filter(*filter_conditions).first()
    nodes, edges = get_city_data_from_db(session, city.city_id)

    if not city:
        nodes, edges = get_city_graph(city_name, center_coords=center_coords, radius_km=radius_km)
        city = store_city_to_db(
            session=session,
            city_name=city_name,
            nodes=nodes,
            edges=edges,
            City=City,
            Node=Node,
            Edge=Edge,
            center_coords=center_coords,
            radius_km=radius_km,
            attraction_point=attraction_point,
            d_alternatives=d_alternatives
        )

    session.close()
    return city, edges
