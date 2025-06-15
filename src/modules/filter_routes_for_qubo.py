import pandas as pd

def filter_routes_for_qubo(congestion_df, threshold_percentile=0.9):
    """
    Filters vehicle IDs that contribute most to congestion based on pairwise interactions.
    Each congestion score is divided by 2 to split the contribution equally between the two vehicles.

    Parameters:
        congestion_df: DataFrame with columns [vehicle1, vehicle2, congestion_score]
        threshold_percentile: Percentile threshold (e.g., 0.9 for top 10%)

    Returns:
        List of filtered vehicle IDs
    """
    # Divide score equally between both vehicles
    v1_scores = congestion_df[['vehicle1', 'congestion_score']].copy()
    v1_scores['congestion_score'] /= 2
    v1_scores = v1_scores.rename(columns={'vehicle1': 'vehicle_id'})

    v2_scores = congestion_df[['vehicle2', 'congestion_score']].copy()
    v2_scores['congestion_score'] /= 2
    v2_scores = v2_scores.rename(columns={'vehicle2': 'vehicle_id'})

    all_scores = pd.concat([v1_scores, v2_scores], axis=0)

    # Sum all congestion scores per vehicle
    vehicle_scores = all_scores.groupby('vehicle_id')['congestion_score'].sum().reset_index()

    # Compute threshold
    threshold_value = vehicle_scores['congestion_score'].quantile(threshold_percentile)

    # Filter vehicle IDs above the threshold
    filtered_vehicles = vehicle_scores[
        vehicle_scores['congestion_score'] >= threshold_value
    ]['vehicle_id'].tolist()

    print(f"Threshold ({threshold_percentile:.2%}): {threshold_value:.6f}")
    print(f"Filtered vehicles ({len(filtered_vehicles)} total): {filtered_vehicles}")

    return filtered_vehicles
