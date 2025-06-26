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
    return os.environ.get('QA_API_TOKEN', '123456456afsdgaeh')


def authenticate_with_token(token: str) -> bool:
    try:
        with Client(token=token) as client:
            # Log useful profile info
            logger.info(f"Connected as: {client.profile}")
            return True
    except Exception as e:
        logger.error(f"Failed to authenticate with D-Wave: {e}")
        return False



def qa_testing(
    Q: dict,
    run_config_id: int,
    iteration_id: int,
    session: Any,
    n: int,
    t: int,
    weights=None,
    vehicle_ids=None,
    lambda_strategy=None,
    lambda_value=None,
    comp_type: str = 'hybrid',
    num_reads: int = 10
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

    Returns:
        dict: Results including assignment validity, assignment, energy, and duration.
    """
    # --- Authentication ---
    api_token = get_api_token()
    if authenticate_with_token(api_token):
        logger.info("Authentication successful. QA profile loaded.")
    else:
        logger.error("Authentication failed. Invalid API token.")
        raise PermissionError("Invalid API token for QA profile.")

    # --- QUBO to BQM ---
    bqm = BinaryQuadraticModel.from_qubo(Q)

    # --- Run sampler ---
    start_time = time.perf_counter()
    if comp_type == 'sa':
        sampler = SimulatedAnnealingSampler()
        response = sampler.sample(bqm, num_reads=num_reads, seed=42)
    elif comp_type == 'hybrid':
        sampler = LeapHybridSampler()
        response = sampler.sample(bqm)
    elif comp_type == 'qpu':
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample(bqm, num_reads=num_reads)
    else:
        logger.error(f"Unknown comp_type: {comp_type}")
        raise ValueError(f"Unknown comp_type: {comp_type}")
    duration_qa = time.perf_counter() - start_time

    # --- Process results ---
    best_sample, energy = response.first

    # Check validity (placeholder: always True)
    assignment_valid = all(
        sum(best_sample.values()[i * t + k] for k in range(t)) == 1
        for i in range(n)
    )
    assignment = list(best_sample.values())

    # --- Save QUBO matrix to file ---
    def save_qubo(Q, filepath):
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(Q, f)

    filename = f"run_{run_config_id}_iter_{iteration_id}.json.gz"
    qubo_dir = Path("qubo_matrices")
    filepath = qubo_dir / filename
    qubo_dir.mkdir(parents=True, exist_ok=True)
    save_qubo(Q, filepath)

    # --- Store results in DB ---
    result_record = QAResult(
        run_configs_id=run_config_id,
        iteration_id=iteration_id,
        lambda_strategy=lambda_strategy,
        lambda_value=lambda_value,
        comp_type=comp_type,
        num_reads=num_reads,
        n_vehicles=n,
        k_alternatives=t,
        weights=weights,
        vehicle_ids=vehicle_ids,
        assignment_valid=int(assignment_valid),
        assignment=assignment,
        energy=energy,
        duration=duration_qa,
        qubo_path=str(filepath),
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
        'duration': duration_qa,
        'lambda_strategy': lambda_strategy,
        'lambda_value': lambda_value,
        'weights': weights,
        'vehicle_ids': vehicle_ids
    }
