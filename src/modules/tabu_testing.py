import time
import datetime
import logging
from typing import Any, Dict
from dimod import BinaryQuadraticModel
from dwave.samplers import TabuSampler
from models import TabuResult  # your new table

logger = logging.getLogger(__name__)

def tabu_testing(
    Q: dict,
    run_configs_id: int,
    iteration_id: int,
    session: Any,
    n: int,
    t: int,
    vehicle_ids=None,
    num_reads: int = 1,
    vehicle_routes_df=None,
    cluster_id: int = None
) -> Dict[str, Any]:
    """
    Run QUBO using D-Wave's Tabu Search Sampler and store result in TabuResult table.

    Args:
        Q: QUBO matrix as a dictionary {(q1, q2): value}
        run_configs_id: Run configuration ID
        iteration_id: Iteration number
        session: SQLAlchemy session
        n: Number of vehicles
        t: Number of routes per vehicle
        vehicle_ids: List of vehicle IDs
        num_reads: Number of Tabu reads (iterations)
        vehicle_routes_df: DataFrame of vehicle routes
        cluster_id: Optional cluster ID

    Returns:
        dict: Result data
    """
    start_time = time.perf_counter()
    logger.info("Starting Tabu Search QUBO solving")

    bqm = BinaryQuadraticModel.from_qubo(Q)
    sampler = TabuSampler()
    response = sampler.sample(bqm, num_reads=num_reads)
    duration = time.perf_counter() - start_time

    best_sample, energy = response.first.sample, response.first.energy
    assignment = [int(x) for x in best_sample.values()]

    if vehicle_routes_df is None or vehicle_ids is None:
        raise ValueError("vehicle_routes_df and vehicle_ids must not be None for assignment validity check.")

    # Validate assignment
    invalid_assignment_vehicles = []
    for i, vehicle_id in enumerate(vehicle_ids):
        assignment_slice = assignment[i * t: (i + 1) * t]
        if assignment_slice.count(1) != 1:
            invalid_assignment_vehicles.append(vehicle_id)

    assignment_valid = len(invalid_assignment_vehicles) == 0
    invalid_assignment_vehicles_str = ",".join(str(v) for v in invalid_assignment_vehicles)

    # Store result
    result_record = TabuResult(
        run_configs_id=run_configs_id,
        iteration_id=iteration_id,
        num_reads=num_reads,
        n_vehicles=n,
        k_alternatives=t,
        vehicle_ids=vehicle_ids,
        assignment_valid=assignment_valid,
        assignment=assignment,
        energy=energy,
        duration=duration,
        invalid_assignment_vehicles=invalid_assignment_vehicles_str,
        cluster_id=cluster_id,
        created_at=datetime.datetime.utcnow()
    )
    session.add(result_record)
    session.commit()

    logger.info(f"Tabu Search result stored: assignment_valid={assignment_valid}, energy={energy}, duration={duration:.2f}s")

    return {
        'comp_type': 'tabu',
        'num_reads': num_reads,
        'n_vehicles': n,
        'k_alternatives': t,
        'assignment_valid': assignment_valid,
        'assignment': assignment,
        'energy': energy,
        'duration': duration,
        'vehicle_ids': vehicle_ids
    }
