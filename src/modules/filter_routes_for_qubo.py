import pandas as pd
import numpy as np

def select_dense_vehicle_subset(w, vehicle_ids, num_selected):
    n = len(vehicle_ids)
    t = len(w[0][0])
    interaction_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            interaction_matrix[i, j] = sum(
                w[i][j][k1][k2] for k1 in range(t) for k2 in range(t)
            )

    scores = interaction_matrix.sum(axis=1)
    selected = [int(np.argmax(scores))]
    remaining = set(range(n)) - set(selected)

    while len(selected) < num_selected and remaining:
        best_candidate = None
        best_score = -1
        for r in remaining:
            density = sum(interaction_matrix[r, s] for s in selected)
            if density > best_score:
                best_score = density
                best_candidate = r
        selected.append(best_candidate)
        remaining.remove(best_candidate)

    selected.sort()
    return [vehicle_ids[i] for i in selected]


def filter_routes_for_qubo(
    t, vehicle_ids, w,
    filtering_percentage=0.10,
    max_qubo_size=None
):
    """
    Filters vehicle routes for QUBO using percentage-based selection
    and optional upper bound on QUBO size.

    Parameters:
    - congestion_df: unused, placeholder for future enhancements
    - t: number of route alternatives per vehicle
    - vehicle_ids: full list of vehicle IDs
    - w: 4D congestion weight matrix
    - filtering_percentage: proportion of vehicles to keep (e.g., 0.1 for 10%)
    - max_qubo_size: optional hard cap on number of QUBO variables

    Returns:
    - List of selected vehicle IDs
    """
    vehicle_count = len(vehicle_ids)
    max_vehicles = int(vehicle_count * filtering_percentage)

    if max_qubo_size is not None:
        max_vehicles = min(max_vehicles, max_qubo_size // t)

    if max_vehicles == 0:
        raise ValueError("Too few vehicles selected. Adjust filtering_percentage or max_qubo_size.")

    print(f"Selecting up to {max_vehicles} vehicles from {vehicle_count} ({100 * filtering_percentage:.1f}%)")
    return select_dense_vehicle_subset(w, vehicle_ids, max_vehicles)

















# import pandas as pd

# def filter_routes_for_qubo(congestion_df, threshold_percentile=0.9):
#     """
#     Filters vehicle IDs that contribute most to congestion based on pairwise interactions.
#     Each congestion score is divided by 2 to split the contribution equally between the two vehicles.

#     Parameters:
#         congestion_df: DataFrame with columns [vehicle1, vehicle2, congestion_score]
#         threshold_percentile: Percentile threshold (e.g., 0.9 for top 10%)

#     Returns:
#         List of filtered vehicle IDs
#     """
#     # Divide score equally between both vehicles
#     v1_scores = congestion_df[['vehicle1', 'congestion_score']].copy()
#     v1_scores['congestion_score'] /= 2
#     v1_scores = v1_scores.rename(columns={'vehicle1': 'vehicle_id'})

#     v2_scores = congestion_df[['vehicle2', 'congestion_score']].copy()
#     v2_scores['congestion_score'] /= 2
#     v2_scores = v2_scores.rename(columns={'vehicle2': 'vehicle_id'})

#     all_scores = pd.concat([v1_scores, v2_scores], axis=0)

#     # Sum all congestion scores per vehicle
#     vehicle_scores = all_scores.groupby('vehicle_id')['congestion_score'].sum().reset_index()

#     # Compute threshold
#     threshold_value = vehicle_scores['congestion_score'].quantile(threshold_percentile)

#     # Filter vehicle IDs above the threshold
#     filtered_vehicles = vehicle_scores[
#         vehicle_scores['congestion_score'] >= threshold_value
#     ]['vehicle_id'].tolist()

#     print(f"Threshold ({threshold_percentile:.2%}): {threshold_value:.6f}")
#     print(f"Filtered vehicles ({len(filtered_vehicles)} total): {filtered_vehicles}")

#     return filtered_vehicles
