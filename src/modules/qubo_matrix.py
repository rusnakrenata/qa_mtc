from collections import defaultdict
from itertools import combinations
from normalize_congestion_weights import normalize_congestion_weights
from congestion_weights import congestion_weights

def qubo_matrix(n, t, w_df, vehicle_ids, lambda_strategy="normalized", fixed_lambda=1.0):
    """
    Constructs the QUBO dictionary for the traffic assignment problem.

    Parameters:
    - n: number of vehicles
    - t: number of route alternatives per vehicle
    - w: 3D list of congestion weights w[i][j][k]
    - lambda_strategy: either "normalized" or "max_weight"
    - fixed_lambda: used when lambda_strategy == "normalized"

    Returns:
    - QUBO matrix Q as a dictionary {(q1, q2): value}
    """
    Q = defaultdict(float)

    # --- Step 1: Determine λ ---
    if lambda_strategy == "normalized":
        lambda_penalty = fixed_lambda
        w = normalize_congestion_weights(w_df, n, t, vehicle_ids)
    else:        
        w, max_w = congestion_weights(w_df, n, t, vehicle_ids)
        lambda_penalty = max_w
        print(f"Using lambda_penalty = max_w = {lambda_penalty:.6f}")

    # --- Step 2: Congestion cost terms ---
    for k in range(t):
        for i, j in combinations(range(n), 2):
            q1 = i * t + k
            q2 = j * t + k
            Q[(q1, q2)] += w[i][j][k]

    # --- Step 3: One-hot assignment constraint terms ---
    for i in range(n):
        for k in range(t):
            q = i * t + k
            Q[(q, q)] += lambda_penalty * (1 - 2)  # x^2 - 2x → +1 -2

        for k1, k2 in combinations(range(t), 2):
            q1 = i * t + k1
            q2 = i * t + k2
            Q[(q1, q2)] += 2 * lambda_penalty

    return dict(Q)
