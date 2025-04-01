import math
import time
import numpy as np
from dimod import SimulatedAnnealingSampler, BinaryQuadraticModel
from dwave.system import EmbeddingComposite, DWaveSampler, LeapHybridSampler
from qubo_matrix import generate_qubo_matrix

def qa_testing(G, lambda_par, comp_type='hybrid', num_reads=10):
    """
    Run QUBO formulation of the graph partitioning problem using a specified sampler.

    Parameters:
        G: NetworkX graph
        lambda_par: Penalty coefficient
        comp_type: 'test', 'hybrid', or 'qpu'
        num_reads: Number of reads for classical or QPU runs

    Returns:
        qa_inter_edges: Number of edges crossing between partitions
        best_partition: List of 0/1 indicating node assignments
        duration_qa: Runtime of hybrid sampling
    """
    duration_qa = 0.0

    # Construct the QUBO matrix
    Q = generate_qubo_matrix(G, lambda_par)

    if comp_type == 'test':
        # Classical simulated annealing
        sa_sampler = SimulatedAnnealingSampler()
        response = sa_sampler.sample_qubo(Q, num_reads=num_reads)

    elif comp_type == 'hybrid':
        # Use hybrid solver via Leap
        bqm = BinaryQuadraticModel.from_qubo(Q)
        start_time = time.time()
        hybrid_sampler = LeapHybridSampler(profile=profile)
        response = hybrid_sampler.sample(bqm)
        duration_qa = time.time() - start_time
                

    elif comp_type == 'qpu':
        # Use QPU solver
        qpu_sampler = EmbeddingComposite(DWaveSampler())
        response = qpu_sampler.sample(Q, num_reads=num_reads)

    # Evaluate the best valid partition found
    best_partition = [1e6, []]
    for sample in response.record:
        partition = list(map(int, sample[0]))
        # Ensure balanced split (±1 node)
        if np.sum(partition) in [math.floor(len(G.nodes) / 2), math.ceil(len(G.nodes) / 2)]:
            qa_inter_edges = sum(partition[u] + partition[v] - 2 * partition[u] * partition[v] for u, v in G.edges())
            if qa_inter_edges < best_partition[0]:
                best_partition = [qa_inter_edges, partition]
        else:
            # Penalize imbalance
            best_partition[0] += abs(np.sum(partition) - math.floor(len(G.nodes) / 2))

    del response
    return best_partition[0], list(best_partition[1]), duration_qa
