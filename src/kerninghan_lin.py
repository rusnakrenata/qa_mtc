import random


def balanced_kernighan_lin(G, max_iterations=10):
    """
    Balanced Kernighan–Lin algorithm for graph partitioning, returning inter-community edges.

    Parameters:
        G: A NetworkX graph
        max_iterations: Maximum number of iterations for refinement

    Returns:
        A, B: Two balanced partitions of the graph
        inter_edges: Number of inter-community edges
    """
    # Step 1: Create an equal-sized initial partition
    nodes = list(G.nodes())
    random.shuffle(nodes)  # Shuffle to avoid bias
    A = set(nodes[:len(nodes) // 2])
    B = set(nodes[len(nodes) // 2:])

    # Step 2: Iterative refinement
    for _ in range(max_iterations):
        max_gain = float('-inf')
        best_pair = None

        # Step 3: Find the best pair of nodes to swap
        for u in A:
            for v in B:
                gain = calculate_gain(G, u, v, A, B)
                if gain > max_gain:
                    max_gain = gain
                    best_pair = (u, v)

        # Step 4: Swap nodes if it improves the cut
        if best_pair and max_gain > 0:
            u, v = best_pair
            A.remove(u)
            B.add(u)
            B.remove(v)
            A.add(v)
        else:
            # Stop if no improvement
            break

    # Step 5: Count inter-community edges
    inter_edges = sum(1 for u, v in list(G.edges()) if (u in A and v in B) or (u in B and v in A))

    return inter_edges

def calculate_gain(G, u, v, A, B):
    """
    Calculate the gain of swapping nodes u and v.

    Parameters:
        G: A NetworkX graph
        u, v: Nodes to evaluate for swapping
        A, B: Current partitions

    Returns:
        Gain value for the swap
    """
    # Gain for node u
    gain_u = sum(1 for neighbor in G[u] if neighbor in B) - sum(1 for neighbor in G[u] if neighbor in A)
    # Gain for node v
    gain_v = sum(1 for neighbor in G[v] if neighbor in A) - sum(1 for neighbor in G[v] if neighbor in B)
    return gain_u + gain_v

