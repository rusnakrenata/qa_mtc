from collections import defaultdict
from itertools import combinations

def generate_qubo_matrix(G, lambda_par):
    """
    Construct the QUBO matrix for the graph partitioning problem.

    Parameters:
        G: NetworkX graph
        lambda_par: Penalty coefficient for balancing

    Returns:
        Q: QUBO matrix as a dictionary
    """
    Q = defaultdict(int)

    # Encourage different labels for connected nodes (cut edges)
    for i, j in list(G.edges()):
        Q[(i, i)] += 1
        Q[(j, j)] += 1
        Q[(i, j)] += -2

    # Add balancing terms to penalize unbalanced partitions
    for i in G.nodes:
        Q[(i, i)] += lambda_par * (1 - len(G.nodes))

    for i, j in combinations(G.nodes, 2):
        Q[(i, j)] += 2 * lambda_par

    return Q
