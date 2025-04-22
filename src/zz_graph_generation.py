import networkx as nx


def max_degree(G):
    return max((G.degree[node] for node in G.nodes), default=0)

def generate_graph(nr_of_nodes, edge_probability):
    """
    Generate a graph and compute structural properties.

    Parameters:
        nr_of_nodes: Number of nodes
        edge_probability: Branching factor for random graph
        G_default: Optional predefined graph

    Returns:
        dict: Contains the graph (G) and its calculated properties
    """
    G = nx.gnp_random_graph(nr_of_nodes, edge_probability)

    lambda_est = (1 + min(nr_of_nodes / 2 - 1, max_degree(G))) / 2

    
    return {
        "graph": G,
        "nr_of_nodes": nr_of_nodes,
        "edge_probability": edge_probability,
        "edges": list(G.edges()) if nr_of_nodes < 100 else None,
        "lambda_est": lambda_est,
        "density": nx.density(G),
        "clustering": nx.average_clustering(G),
        "nr_of_edges": nx.number_of_edges(G)
      }
