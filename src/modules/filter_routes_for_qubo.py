#select only 0.9 percentile of the vehicles that cause congestion

def filter_routes_for_qubo(routes_df, congestion_df, threshold=0.9):
    high_congestion_edges = congestion_df[congestion_df['congestion_score'] > congestion_df['congestion_score'].quantile(threshold)]
    congested_edge_ids = set(high_congestion_edges['edge_id'])

    filtered_vehicles = []
    for vehicle_id, group in routes_df.groupby('vehicle_id'):
        used_edges = set(group['edge_id'])
        if used_edges & congested_edge_ids:
            filtered_vehicles.append(vehicle_id)
    return filtered_vehicles
