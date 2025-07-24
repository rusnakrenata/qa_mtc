import os
import time
from pathlib import Path
import gurobipy as gp
from gurobipy import Model, GRB, quicksum
from models import GurobiResult
import pandas as pd

# Set license path relative to project directory
project_root = Path(__file__).resolve().parents[2]
license_path = project_root / "gurobi.lic"
os.environ["GRB_LICENSE_FILE"] = str(license_path)


def gurobi_testing(Q: dict, n:int, t: int, run_configs_id, iteration_id, session, time_limit_seconds: int = 300, cluster_id: int = 0) -> tuple:
    """
    Solves a QUBO problem using Gurobi.
    Args:
        Q: dict of {(i,j): value}
        time_limit_seconds: Max time in seconds to let Gurobi run
    Returns:
        Dict of variable values and objective value
    """
    time_limit_seconds = max(time_limit_seconds, 60)  # Ensure at least 1 minute
    start_time = time.perf_counter()
    model = Model("QUBO_Traffic")
    model.setParam("TimeLimit", time_limit_seconds)
    model.setParam("OutputFlag", 0)  # show output, set 0 to suppress

    # Create binary variables
    variables = {}
    for i, j in Q:
        if i not in variables:
            variables[i] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}")
        if j not in variables:
            variables[j] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}")

    model.update()

    # Set objective: sum Q[i,j] * x_i * x_j
    obj = quicksum(Q[i, j] * variables[i] * variables[j] for i, j in Q)
    model.setObjective(obj, GRB.MINIMIZE)

    # Solve
    model.optimize()
    duration = time.perf_counter() - start_time

    # Parse result
    result = {v.VarName: int(v.X) for v in model.getVars()}
    objective_value = model.ObjVal if model.SolCount > 0 else None   
    best_bound = model.ObjBound if model.SolCount > 0 else None
    gap = model.MIPGap if model.SolCount > 0 else None
    

    # Convert result dict to list of tuples for storage (or just store as dict/JSON)
    assignment = list(result.items())

    gurobi_result = GurobiResult(
        run_configs_id=run_configs_id,
        iteration_id=iteration_id,
        assignment=assignment,
        objective_value=objective_value,
        duration=duration,
        best_bound=best_bound,
        gap=gap,
        cluster_id=cluster_id
        # congestion_score will be filled after post-processing
    )
    session.add(gurobi_result)
    session.commit()
    return result, objective_value


'''
from gurobipy import Model, GRB, quicksum
import time
import logging

logger = logging.getLogger(__name__)

def solve_qubo_with_gurobi(Q: dict, n: int, t: int, run_configs_id, iteration_id, session, time_limit_seconds: int = 300):
    """
    Solves a QUBO problem using Gurobi.
    Args:
        Q: dict of {(i,j): value}
        n: number of vehicles
        t: number of route alternatives per vehicle
        run_configs_id: identifier for the run config
        iteration_id: identifier for the iteration
        session: SQLAlchemy session to store results
        time_limit_seconds: Max time in seconds to let Gurobi run

    Returns:
        Tuple of (result_dict, objective_value)
    """
    start_time = time.perf_counter()
    model = Model("CQM_Traffic")
    model.setParam("TimeLimit", time_limit_seconds)
    model.setParam("OutputFlag", 0)

    # Define binary variables
    variables = {}
    for q in range(n * t):
        variables[q] = model.addVar(vtype=GRB.BINARY, name=f"x_{q}")

    model.update()

    # Define QUBO objective
    obj = quicksum(Q[i, j] * variables[i] * variables[j] for i, j in Q)
    model.setObjective(obj, GRB.MINIMIZE)

    # Add one-hot constraints for each vehicle
    for i in range(n):
        terms = [variables[i * t + k] for k in range(t)]
        model.addConstr(quicksum(terms) == 1, name=f"one_hot_vehicle_{i}")

    model.optimize()  

    duration = time.perf_counter() - start_time

    # Handle solution
    if model.SolCount == 0:
        logger.warning("No feasible solution found by Gurobi.")
        result = {}
        objective_value = None
    else:
        result = {v.VarName: int(v.X) for v in model.getVars()}
        objective_value = model.ObjVal

    assignment = list(result.items())

    gurobi_result = GurobiResult(
        run_configs_id=run_configs_id,
        iteration_id=iteration_id,
        assignment=assignment,
        objective_value=objective_value,
        duration=duration,
        congestion_score=None  # to be filled later
    )

    session.add(gurobi_result)
    session.commit()

    return result, objective_value
'''
