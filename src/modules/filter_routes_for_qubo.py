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


def filter_routes_for_qubo(routes_df, congestion_df, threshold_percentile=0.9):
    """
    Filters vehicles that contribute significantly to congestion based on a percentile threshold.

    Parameters:
        routes_df: DataFrame with route points
        congestion_df: DataFrame with congestion scores (edge_id, congestion_score)
        threshold_percentile: percentile above which vehicles are considered congesting (e.g., 0.9)

    Returns:
        List of filtered vehicle_ids
    """
    # Merge congestion scores onto route points
    merged = routes_df.merge(congestion_df, on='edge_id', how='left')
    merged['congestion_score'] = merged['congestion_score'].fillna(0)

    # Compute total congestion exposure per vehicle
    vehicle_scores = merged.groupby('vehicle_id')['congestion_score'].sum().reset_index()

    # Compute threshold value using percentile
    threshold_value = vehicle_scores['congestion_score'].quantile(threshold_percentile)

    # Select all vehicles whose congestion exceeds or equals the threshold
    filtered_vehicles = vehicle_scores[vehicle_scores['congestion_score'] >= threshold_value]['vehicle_id'].tolist()

    return filtered_vehicles
