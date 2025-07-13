import time
import datetime
import numpy as np
import json
import gzip
import os
from pathlib import Path
from collections import defaultdict
from dimod import SimulatedAnnealingSampler, BinaryQuadraticModel
from dwave.system import EmbeddingComposite, DWaveSampler, LeapHybridSampler
from dwave.cloud import Client
from models import QAResult
import logging
from typing import Any, Dict


logger = logging.getLogger(__name__)

def get_api_token() -> str:
    """
    Retrieve the API token securely from environment variable or fallback.
    """
    return os.environ.get('QA_API_TOKEN', 'notoken')


def authenticate_with_token(token: str) -> bool:
    try:
        sampler = DWaveSampler(token=token)
        logger.info(f"Connected to D-Wave. Solver: {sampler.solver.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to authenticate with D-Wave: {e}")
        return False



def qa_testing(
    Q: dict,
    run_configs_id: int,
    iteration_id: int,
    session: Any,
    n: int,
    t: int,
    vehicle_ids=None,
    lambda_strategy=None,
    lambda_value=None,
    comp_type: str = 'hybrid',
    num_reads: int = 10,
    vehicle_routes_df=None,
    dwave_constraints_check=True
) -> Dict[str, Any]:
    """
    Run QUBO formulation for the car-to-trase assignment using a specified quantum/classical sampler.
    Requires API token for authentication (set QA_API_TOKEN env variable or fallback to default).

    Args:
        Q: QUBO matrix as a dictionary {(q1, q2): value}
        run_config_id: Run configuration ID
        iteration_id: Iteration number
        session: SQLAlchemy session
        n: Number of vehicles
        t: Number of routes per vehicle
        weights: Weights for the QUBO
        vehicle_ids: IDs of vehicles
        lambda_strategy: Lambda strategy for the QUBO
        lambda_value: Lambda value for the QUBO
        comp_type: 'test', 'hybrid', or 'qpu'
        num_reads: Number of reads for classical or QPU runs
        vehicle_routes_df: DataFrame of vehicle routes (for padding info)
        method: Route selection method (for padding info)

    Returns:
        dict: Results including assignment validity, assignment, energy, and duration.
    """
    # --- Authentication ---
    api_token = get_api_token()
    #print("api_token: ", api_token)
    if authenticate_with_token(api_token):
        logger.info("Authentication successful. QA profile loaded.")
    else:
        logger.error("Authentication failed. Invalid API token.")
        raise PermissionError("Invalid API token for QA profile.")

    # --- QUBO to BQM ---
    bqm = BinaryQuadraticModel.from_qubo(Q)

    # --- Run sampler ---
    start_time = time.perf_counter()
    logger.info("Starting QA testing with comp_type: %s", comp_type)
    if comp_type == 'sa':
        sampler = SimulatedAnnealingSampler()
        response = sampler.sample(bqm, num_reads=num_reads)
        total_annealing_time_s = time.perf_counter() - start_time  # No direct timing info; use measured wall-clock

    elif comp_type == 'hybrid':
        sampler = LeapHybridSampler()
        response = sampler.sample(bqm)
        total_annealing_time_s = response.info.get('run_time', 0) / 1_000_000  # µs to s

    elif comp_type == 'qpu':
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample(bqm, num_reads=num_reads)
        annealing_time_us = response.info['timing']['annealing_time']  # per read (µs)
        total_annealing_time_s = (annealing_time_us * num_reads) / 1_000_000  # µs to s

    else:
        logger.error(f"Unknown comp_type: {comp_type}")
        raise ValueError(f"Unknown comp_type: {comp_type}")
    duration_qa = time.perf_counter() - start_time
   

    # --- Process results ---
    record = response.first
    best_sample, energy = record[:2]
    sample_values = list(best_sample.values())

    # Check validity (placeholder: always True)
    assignment = [int(x) for x in sample_values]

    # Robust assignment validity check (handles padding)
    if vehicle_routes_df is None or vehicle_ids is None:
        raise ValueError("vehicle_routes_df and vehicle_ids must not be None for assignment validity check.")

    invalid_assignment_vehicles = []
    for i, vehicle_id in enumerate(vehicle_ids):
        assignment_slice = assignment[i * t : (i + 1) * t]
        if assignment_slice.count(1) != 1:
            invalid_assignment_vehicles.append(vehicle_id)
    assignment_valid = len(invalid_assignment_vehicles) == 0
    invalid_assignment_vehicles_str = ",".join(str(v) for v in invalid_assignment_vehicles)

    # --- Save QUBO matrix to file ---
    def save_qubo(Q, filepath):
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in Q.items()}, f)

    filename = f"run_{run_configs_id}_iter_{iteration_id}.json.gz"
    qubo_dir = Path("qubo_matrices")
    filepath = qubo_dir / filename
    qubo_dir.mkdir(parents=True, exist_ok=True)
    save_qubo(Q, filepath)

    # --- Store results in DB ---
    result_record = QAResult(
        run_configs_id=run_configs_id,
        iteration_id=iteration_id,
        lambda_strategy=lambda_strategy,
        lambda_value=lambda_value,
        comp_type=comp_type,
        num_reads=num_reads,
        n_vehicles=n,
        k_alternatives=t,
        vehicle_ids=vehicle_ids,
        assignment_valid=int(assignment_valid),
        assignment=assignment,
        energy=energy,
        duration=total_annealing_time_s,
        qubo_path=str(filepath),
        invalid_assignment_vehicles=invalid_assignment_vehicles_str,
        dwave_constraints_check=dwave_constraints_check,
        created_at=datetime.datetime.utcnow()
    )
    session.add(result_record)
    session.commit()

    logger.info(f"QA testing complete: assignment_valid={assignment_valid}, energy={energy}, duration={duration_qa:.2f}s")

    return {
        'comp_type': comp_type,
        'num_reads': num_reads,
        'n_vehicles': n,
        'k_alternatives': t,
        'assignment_valid': assignment_valid,
        'assignment': assignment,
        'energy': energy,
        'qubo_path': str(filepath),
        'duration': total_annealing_time_s,
        'lambda_strategy': lambda_strategy,
        'lambda_value': lambda_value,
        'vehicle_ids': vehicle_ids
    }
