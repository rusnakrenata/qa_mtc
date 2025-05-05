import networkx as nx
from collections import defaultdict
from itertools import combinations
from sqlalchemy.orm import sessionmaker
from db_tables import *


# Round a (lat, lon) point to a specified precision to normalize coordinates
def normalize_point(p, precision=5):
    return (round(p[0], precision), round(p[1], precision))

# Normalize a segment (defined by two points) for consistency
# Direction is preserved here (not symmetric)
def normalize_segment(segment, precision=5):
    a = normalize_point(segment[0], precision)
    b = normalize_point(segment[1], precision)
    return (a, b)

# Convert car route polylines into segment lists
def transform_routes_to_segments(car_routes, run_id, iteration_id):
    def to_segments(points):
        # Convert a sequence of points into consecutive segments
        return [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    
    transformed = {}
    for car in car_routes:
        car_id = car['car_id']
        # For each route of the car, convert geometry to segments
        routes = [to_segments(route['geometry']) for route in car['routes']]
        transformed[car_id] = routes
    return transformed

# Build a graph where cars are nodes and edges represent overlap in route segments
def build_overlap_graph(car_routes_segments, precision=5):
    segment_to_cars = defaultdict(set)

    # Invert the data to know which cars use which segments
    for car_id, routes in car_routes_segments.items():
        for route in routes:
            for segment in route:
                seg = normalize_segment(segment, precision)
                segment_to_cars[seg].add(car_id)

    overlap_graph = nx.Graph()
    overlap_counts = defaultdict(int)

    # Count how many times each pair of cars share segments
    for segment, cars in segment_to_cars.items():
        for car1, car2 in combinations(cars, 2):
            pair = tuple(sorted((car1, car2)))
            overlap_counts[pair] += 1

    # Add an edge for each pair of overlapping cars, weighted by shared segments
    for (car1, car2), weight in overlap_counts.items():
        overlap_graph.add_edge(car1, car2, weight=weight)

    return overlap_graph

# Compute cluster assignments (segments) for each car
def compute_segment_assignments(car_routes, run_id, iteration_id):
    """
    Computes segment (cluster) assignments for each car based on route segment overlaps.

    Args:
        car_routes: list of car route dicts (with car_id and routes[geometry])
        run_id: ID of the run configuration
        iteration_id: ID of the iteration within the run

    Returns:
        car_to_segment: dict mapping logical car_id → segment_id
    """
    if not car_routes:
        print(f"No car routes provided for run_id={run_id}, iteration_id={iteration_id}")
        return {}

    car_routes_segments = transform_routes_to_segments(car_routes, run_id, iteration_id)
    overlap_graph = build_overlap_graph(car_routes_segments)

    if len(overlap_graph.nodes) == 0:
        print(f"No overlap graph could be constructed for run_id={run_id}, iteration_id={iteration_id}")
        return {}

    clusters = list(nx.connected_components(overlap_graph))

    car_to_segment = {}
    for segment_id, cluster in enumerate(clusters):
        for car_id in cluster:
            car_to_segment[car_id] = segment_id

    print(f"Segment assignments complete for run_id={run_id}, iteration_id={iteration_id} — {len(car_to_segment)} cars assigned to {len(clusters)} segments.")
    return car_to_segment



def store_in_db_segment_assignments(car_routes, run_id, iteration_id):
    Session = sessionmaker(bind=engine)
    session = Session()

    car_to_segment = compute_segment_assignments(car_routes, run_id, iteration_id)

    # Map logical car_id to DB Car.id
    db_car_map = {
        c.car_id: c.id for c in session.query(Car).filter_by(
            run_configs_id=run_id, iteration_id=iteration_id
        ).all()
    }

    added = 0
    for logical_id, segment_id in car_to_segment.items():
        car_db_id = db_car_map.get(logical_id)
        if not car_db_id:
            print(f" Warning: Car {logical_id} not found in DB.")
            continue

        assignment = SegmentAssignment(
            car_id=car_db_id,
            run_id=run_id,
            iteration_id=iteration_id,
            segment_id=segment_id
        )
        session.add(assignment)
        added += 1

    session.commit()
    session.close()
    print(f"Stored {added} segment assignments for run {run_id}, iteration {iteration_id}.")
    return car_to_segment


