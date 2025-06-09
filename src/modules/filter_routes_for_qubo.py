#select only 0.9 percentile of the vehicles that cause congestion

# def filter_routes_for_qubo(routes_df, congestion_df, threshold=0.9):
#     high_congestion_edges = congestion_df[congestion_df['congestion_score'] > congestion_df['congestion_score'].quantile(threshold)]
#     congested_edge_ids = set(high_congestion_edges['edge_id'])

#     filtered_vehicles = []
#     for vehicle_id, group in routes_df.groupby('vehicle_id'):
#         used_edges = set(group['edge_id'])
#         if used_edges & congested_edge_ids:
#             filtered_vehicles.append(vehicle_id)
#     return filtered_vehicles


def filter_routes_for_qubo(routes_df, congestion_df, top_percent=0.1):
    # Merge congestion scores onto route points
    merged = routes_df.merge(congestion_df, on='edge_id', how='left')
    merged['congestion_score'] = merged['congestion_score'].fillna(0)

    # Compute total congestion per vehicle
    vehicle_scores = merged.groupby('vehicle_id')['congestion_score'].sum().reset_index()
    
    # Select top X%
    top_pct = int(len(vehicle_scores) * top_percent)
    top_vehicles = vehicle_scores.sort_values(by='congestion_score', ascending=False).head(top_pct)

    filtered_vehicles = []
    filtered_vehicles = top_vehicles['vehicle_id'].tolist()

    return filtered_vehicles