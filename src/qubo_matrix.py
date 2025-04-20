from collections import defaultdict
from itertools import combinations


def generate_qubo_matrix(n, t, w, lambda_par):
    """
    Construct the QUBO matrix for the car-to-trase assignment problem.

    Parameters:
        n: Number of cars
        t: Number of trases
        w: 3D list of weights w[i][j][k] representing the congestion cost if cars i and j take trase k
        lambda_par: Penalty coefficient for assignment constraint

    Returns:
        Q: QUBO matrix as a defaultdict(int)
    """
    Q = defaultdict(int)

    # Step 1: Congestion penalty (encourage cars to avoid same congested trase)
    for k in range(t):
        for i, j in combinations(range(n), 2):
            q1 = i * t + k
            q2 = j * t + k
            Q[(q1, q2)] += w[i][j][k]

    # Step 2: Assignment constraint (each car must take exactly one trase)
    for i in range(n):
        # Linear terms from expanding (sum_k x_i^k - 1)^2
        for k in range(t):
            q = i * t + k
            Q[(q, q)] += lambda_par * (1 - 2)  # x^2 - 2x → +1 -2 in the expansion

        # Quadratic terms from x_i^k1 * x_i^k2
        for k1, k2 in combinations(range(t), 2):
            q1 = i * t + k1
            q2 = i * t + k2
            Q[(q1, q2)] += 2 * lambda_par

    return Q
