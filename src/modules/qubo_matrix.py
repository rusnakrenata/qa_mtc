from collections import defaultdict
from normalize_congestion_weights import normalize_congestion_weights
from congestion_weights import congestion_weights

def qubo_matrix(n, t, w_df, vehicle_ids, lambda_strategy="normalized", fixed_lambda=1.0):
    """
    Constructs the QUBO dictionary for the traffic assignment problem with 4D weights.

    Parameters:
    - n: number of vehicles
    - t: number of route alternatives per vehicle
    - w_df: congestion weights DataFrame
    - vehicle_ids: list of filtered vehicle IDs
    - lambda_strategy: "normalized" or "max_weight"
    - fixed_lambda: λ if using normalized strategy

    Returns:
    - QUBO matrix Q as a dictionary {(q1, q2): value}
    """
    Q = defaultdict(float)

    # Step 1: Convert DataFrame to 4D matrix
    if lambda_strategy == "normalized":
        lambda_penalty = fixed_lambda
        w = normalize_congestion_weights(w_df, n, t, vehicle_ids)
    else:
        w, max_w = congestion_weights(w_df, n, t, vehicle_ids)
        lambda_penalty = max_w
        print(f"Using lambda_penalty = max_w = {lambda_penalty:.6f}")

    w_c, max_w_c = congestion_weights(w_df, n, t, vehicle_ids)

    # Step 2: Congestion cost terms (based on partial route overlaps)
    for i in range(n):
        for j in range(i + 1, n):  # only i < j to avoid duplicates
            for k1 in range(t):
                for k2 in range(t):
                    q1 = i * t + k1
                    q2 = j * t + k2
                    Q[(q1, q2)] += w[i][j][k1][k2]

    # Step 3: One-hot assignment constraints for each vehicle
    for i in range(n):
        for k in range(t):
            q = i * t + k
            Q[(q, q)] += lambda_penalty * (1 - 2)  # x^2 - 2x

        for k1 in range(t):
            for k2 in range(k1 + 1, t):
                q1 = i * t + k1
                q2 = i * t + k2
                Q[(q1, q2)] += 2 * lambda_penalty

    return dict(Q), w, w_c
