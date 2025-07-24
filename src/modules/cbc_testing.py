import time
from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD,
    LpBinary, LpStatusOptimal
)
from models import CbcResult
import logging

logger = logging.getLogger(__name__)


def cbc_testing(Q: dict, run_configs_id, iteration_id, session, time_limit_seconds: int = 300, cluster_id: int = 0):
    """
    Solves a QUBO problem using CBC by linearizing it into a MIP.

    Args:
        Q: QUBO matrix as a dict {(i, j): value}
        run_configs_id: Run config ID
        iteration_id: Iteration ID
        session: SQLAlchemy session
        time_limit_seconds: Max time allowed
        cluster_id: Optional clustering info

    Returns:
        (result_dict)
    """
    try:
        start_time = time.perf_counter()

        # Extract all variable indices
        all_vars = sorted(set(i for i, _ in Q) | set(j for _, j in Q))

        # Create problem and variables
        model = LpProblem("QUBO_Linearized", LpMinimize)
        x = {i: LpVariable(f"x_{i}", cat=LpBinary) for i in all_vars}

        # Linearization: create z_ij for i != j
        z = {}
        for (i, j), value in Q.items():
            if i != j and (i, j) not in z and (j, i) not in z:
                z_name = f"z_{i}_{j}"
                z[i, j] = LpVariable(z_name, cat=LpBinary)
                # Add constraints for z_ij = x_i * x_j
                model += z[i, j] <= x[i]
                model += z[i, j] <= x[j]
                model += z[i, j] >= x[i] + x[j] - 1

        # Objective: sum of Q[i,i]*x_i + Q[i,j]*z_ij
        linear_terms = [Q[i, i] * x[i] for (i, j) in Q if i == j]
        quad_terms = [Q[i, j] * z[min(i, j), max(i, j)] for (i, j) in Q if i != j and (min(i, j), max(i, j)) in z]

        model += lpSum(linear_terms + quad_terms)

        # Solve
        time_limit_seconds = min(time_limit_seconds, 60)
        solver = PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds, options=['-stop'])
        status = model.solve(solver)
        duration = time.perf_counter() - start_time

        if status != LpStatusOptimal:
            logger.warning("CBC did not find an optimal solution.")
            result = {}
            objective_value = None
        else:
            result = {v.name: int(v.varValue) for v in model.variables() if v.name.startswith("x_")}
            objective_value = model.objective.value()

        assignment = list(result.items())

        # Store in DB
        cbc_result = CbcResult(
            run_configs_id=run_configs_id,
            iteration_id=iteration_id,
            assignment=assignment,
            objective_value=objective_value,
            duration=duration,
            cluster_id=cluster_id
        )
        session.add(cbc_result)
        session.commit()

        return result

    except Exception as e:
        logger.error(f"Error in CBC linearized QUBO: {e}", exc_info=True)
        return {}
