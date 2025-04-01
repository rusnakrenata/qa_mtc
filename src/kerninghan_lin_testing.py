import time
from kerninghan_lin import balanced_kernighan_lin

def run_kerninghan_lin(G):
    """
    Run Kernighan–Lin partitioning on a graph.

    Parameters:
        G: NetworkX graph

    Returns:
        dict: Kernighan–Lin info
    """
    try:
        start = time.time()
        inter_edges = balanced_kernighan_lin(G)
        duration = time.time() - start
    except Exception:
        inter_edges = 0
        duration = 0.0

    return {
        "kl_inter_edges": inter_edges,
        "duration_keringham_lin": duration
    }
