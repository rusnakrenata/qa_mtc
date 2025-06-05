import time, datetime
import numpy as np
import json
import gzip
import os
from collections import defaultdict
from dimod import SimulatedAnnealingSampler, BinaryQuadraticModel
from dwave.system import EmbeddingComposite, DWaveSampler, LeapHybridSampler
from qubo_matrix import qubo_matrix
from models import *

def qa_testing(n, t, weights_df, vehicle_ids, run_config_id, iteration_id, session, lambda_strategy="normalized", fixed_lambda=1.0,
               comp_type='hybrid', num_reads=10):
    """
    Run QUBO formulation for the car-to-trase assignment using a specified quantum/classical sampler.

    Parameters:
        n: Number of vehicles
        t: Number of routes per vehicle
        weights_df: DataFrame of congestion weights
        vehicle_ids: List of filtered vehicle IDs
        lambda_strategy: "normalized" or "max_weight"
        fixed_lambda: Used if lambda_strategy is not "normalized"
        comp_type: 'test', 'hybrid', or 'qpu'
        num_reads: Number of reads for classical or QPU runs

    Returns:
        dict: {'assignment_valid': bool, 'assignment': List[int], 'energy': float, 'duration': float}
    """
    duration_qa = 0.0

    # Generate QUBO
    Q = qubo_matrix(n, t, weights_df, vehicle_ids, lambda_strategy, fixed_lambda)

    # Create BinaryQuadraticModel
    bqm = BinaryQuadraticModel.from_qubo(Q)

    # Run sampler
    if comp_type == 'test':
        sampler = SimulatedAnnealingSampler()
        response = sampler.sample(bqm, num_reads=num_reads)

    elif comp_type == 'hybrid':
        start_time = time.time()
        sampler = LeapHybridSampler()
        response = sampler.sample(bqm)
        duration_qa = time.time() - start_time

    elif comp_type == 'qpu':
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample(bqm, num_reads=num_reads)

    # Evaluate best sample
    best_sample = response.first.sample
    energy = response.first.energy

    # Extract assignment
    x = [best_sample[i * t + k] for i in range(n) for k in range(t)]
    assignment = np.argmax(np.array(x).reshape((n, t)), axis=1)

    # Evaluate constraint satisfaction
    assignment_valid = all(
        sum(best_sample[i * t + k] for k in range(t)) == 1 for i in range(n)
    )

    


    # Store the matrix to filesystem
    def save_qubo(Q, filepath):
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(Q, f)

    def load_qubo(filepath):
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            return json.load(f)
        
    filename = f"run_{run_config_id}_iter_{iteration_id}.json.gz"
    filepath = os.path.join("qubo_matrices", filename)  # optional folder

    # Ensure the directory exists   
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
    save_qubo(Q, filepath)


    # Store the data in DB
    result_record = QAResult(
        run_configs_id=run_config_id,
        iteration_id=iteration_id,
        lambda_strategy=lambda_strategy,
        lambda_value=fixed_lambda,
        comp_type=comp_type,
        num_reads=num_reads,
        n_vehicles=n,
        k_alternattives=t,
        weights=weights_df.to_dict(orient='records'),
        vehicle_ids=vehicle_ids,
        assignment_valid=int(assignment_valid),
        assignment=assignment.tolist(),
        energy=energy,
        duration=duration_qa,
        created_at=datetime.utcnow()
    )

    # Store in DB
    session.add(result_record)
    session.commit()


    return {
        'lambda_strategy': lambda_strategy,
        'lambda' : fixed_lambda,
        'comp_type': comp_type, 
        'num_reads': num_reads,
        'n_vehicles': n, #filtered vehicles
        'k_alternattives': t, 
        'weights': weights_df, 
        'vehicle_ids': vehicle_ids, #filtered vehicles
        'assignment_valid': assignment_valid,
        'assignment': assignment,
        'energy': energy,
        'duration': duration_qa
    }
