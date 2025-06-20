from collections import defaultdict
from normalize_congestion_weights import normalize_congestion_weights
from congestion_weights import congestion_weights
from filter_routes_for_qubo import filter_routes_for_qubo

def qubo_matrix(
    n,
    t,
    w_df,
    vehicle_ids,
    lambda_strategy="normalized",
    fixed_lambda=1.0,
    filtering_percentage=None,
    max_qubo_size=None
):
    """
    Constructs the QUBO dictionary for the traffic assignment problem with 4D weights.

    Parameters:
    - n: total number of vehicles
    - t: number of route alternatives per vehicle
    - w_df: congestion weights DataFrame
    - vehicle_ids: full list of vehicle IDs
    - lambda_strategy: "normalized" or "max_weight"
    - fixed_lambda: λ if using normalized strategy
    - filtering_percentage: float (e.g., 0.1) to select a subset of vehicles
    - max_qubo_size: optional upper bound on number of QUBO variables

    Returns:
    - Q: QUBO matrix as a dictionary {(q1, q2): value}
    - w: final 4D weight matrix used in QUBO
    - vehicle_ids_filtered: the filtered list of vehicle IDs used in QUBO
    """

    # Step 1: Compute weights according to the selected strategy
    if lambda_strategy == "normalized":
        w_full = normalize_congestion_weights(w_df, n, t, vehicle_ids)
        lambda_penalty = fixed_lambda
    else:
        w_full, max_w = congestion_weights(w_df, n, t, vehicle_ids)
        lambda_penalty = max_w
        print(f"Using lambda_penalty = max_w = {lambda_penalty:.6f}")

    # Step 2: Filter vehicles based on full weight matrix
    if filtering_percentage is not None or max_qubo_size is not None:
        vehicle_ids_filtered = filter_routes_for_qubo(
            t=t,
            vehicle_ids=vehicle_ids,
            w=w_full,
            filtering_percentage=filtering_percentage or 1.0,
            max_qubo_size=max_qubo_size
        )
    else:
        vehicle_ids_filtered = vehicle_ids

    n_filtered = len(vehicle_ids_filtered)

    # Step 3: Extract weights for the filtered vehicles only
    if lambda_strategy == "normalized":
        w = normalize_congestion_weights(w_df, n_filtered, t, vehicle_ids_filtered)
    else:
        w, _ = congestion_weights(w_df, n_filtered, t, vehicle_ids_filtered)


    # Step 4: QUBO matrix construction
    Q = defaultdict(float)
    for i in range(n_filtered):
        for j in range(i + 1, n_filtered):
            for k1 in range(t):
                for k2 in range(t):
                    q1 = i * t + k1
                    q2 = j * t + k2
                    Q[(q1, q2)] += w[i][j][k1][k2]

    for i in range(n_filtered):
        for k in range(t):
            q = i * t + k
            Q[(q, q)] += lambda_penalty * (1 - 2)  # x^2 - 2x

        for k1 in range(t):
            for k2 in range(k1 + 1, t):
                q1 = i * t + k1
                q2 = i * t + k2
                Q[(q1, q2)] += 2 * lambda_penalty

    return dict(Q), vehicle_ids_filtered
