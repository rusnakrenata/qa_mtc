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

def solve_qubo_with_gurobi(Q: dict, run_configs_id, iteration_id, session, time_limit_seconds: int = 300):
    """
    Solves a QUBO problem using Gurobi.
    Args:
        Q: dict of {(i,j): value}
        time_limit_seconds: Max time in seconds to let Gurobi run
    Returns:
        Dict of variable values and objective value
    """

    start_time = time.perf_counter()
    model = Model("QUBO_Traffic")
    model.setParam("TimeLimit", time_limit_seconds)
    model.setParam("OutputFlag", 1)  # show output, set 0 to suppress

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
    

    # Convert result dict to list of tuples for storage (or just store as dict/JSON)
    assignment = list(result.items())

    gurobi_result = GurobiResult(
        run_configs_id=run_configs_id,
        iteration_id=iteration_id,
        assignment=assignment,
        objective_value=objective_value,
        duration=duration,
        # congestion_score will be filled after post-processing
    )
    session.add(gurobi_result)
    session.commit()
    return result, objective_value


