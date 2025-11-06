from __future__ import annotations

from typing import Dict, Iterable, Tuple

import pyomo.environ as pyo


def solve_subproblem(P, C_out: Iterable[float], C_ret: Iterable[float], R_out: Iterable[float], R_ret: Iterable[float]) -> Tuple[dict, float]:
    """Build and solve the LP for both directions; return duals and objective.

    Returns (duals, objective_value), where duals is a dict with keys:
    alpha_OUT, alpha_RET, pi_OUT, pi_RET
    """
    m = pyo.ConcreteModel()
    T = range(P.T)

    C_out = list(C_out)
    C_ret = list(C_ret)
    R_out = list(R_out)
    R_ret = list(R_ret)

    # Variables: x[t, tau, d], u[t,d]
    m.x_OUT = pyo.Var(T, T, within=pyo.NonNegativeReals)
    m.x_RET = pyo.Var(T, T, within=pyo.NonNegativeReals)
    m.u_OUT = pyo.Var(T, within=pyo.NonNegativeReals)
    m.u_RET = pyo.Var(T, within=pyo.NonNegativeReals)

    def wait_cost(t: int, tau: int) -> float:
        return float(max(0, tau - t))

    m.obj = pyo.Objective(
        expr=sum(wait_cost(t, tau) * m.x_OUT[t, tau] for t in T for tau in T)
        + sum(wait_cost(t, tau) * m.x_RET[t, tau] for t in T for tau in T)
        + P.p * (sum(m.u_OUT[t] for t in T) + sum(m.u_RET[t] for t in T)),
        sense=pyo.minimize,
    )

    # Demand conservation (8.15) with feasible arcs only
    W = P.Wmax_slots

    def cons_dem_OUT(m, t):
        taus = [tau for tau in T if t <= tau <= min(P.T - 1, t + W)]
        return sum(m.x_OUT[t, tau] for tau in taus) + m.u_OUT[t] == R_out[t]

    m.D_out = pyo.Constraint(T, rule=cons_dem_OUT)

    def cons_dem_RET(m, t):
        taus = [tau for tau in T if t <= tau <= min(P.T - 1, t + W)]
        return sum(m.x_RET[t, tau] for tau in taus) + m.u_RET[t] == R_ret[t]

    m.D_ret = pyo.Constraint(T, rule=cons_dem_RET)

    # Per-departure capacity (8.16)
    def cap_out(m, tau):
        ts = [t for t in T if t <= tau <= min(P.T - 1, t + W)]
        return sum(m.x_OUT[t, tau] for t in ts) <= C_out[tau]

    m.Cap_out = pyo.Constraint(T, rule=cap_out)

    def cap_ret(m, tau):
        ts = [t for t in T if t <= tau <= min(P.T - 1, t + W)]
        return sum(m.x_RET[t, tau] for t in ts) <= C_ret[tau]

    m.Cap_ret = pyo.Constraint(T, rule=cap_ret)

    # Solve and extract duals α, π for both directions
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solver = pyo.SolverFactory(P.lp_solver)
    # Allow explicit path to solver executable (e.g., CPLEX binary)
    exec_path = getattr(P, "lp_solver_executable", None)
    if exec_path:
        try:
            solver.executable = exec_path  # type: ignore[attr-defined]
        except Exception:
            pass
    results = solver.solve(m, tee=False)

    alpha_OUT: Dict[int, float] = {t: float(m.dual.get(m.D_out[t], 0.0)) for t in T}
    alpha_RET: Dict[int, float] = {t: float(m.dual.get(m.D_ret[t], 0.0)) for t in T}
    pi_OUT: Dict[int, float] = {tau: float(m.dual.get(m.Cap_out[tau], 0.0)) for tau in T}
    pi_RET: Dict[int, float] = {tau: float(m.dual.get(m.Cap_ret[tau], 0.0)) for tau in T}

    return dict(alpha_OUT=alpha_OUT, alpha_RET=alpha_RET, pi_OUT=pi_OUT, pi_RET=pi_RET), float(pyo.value(m.obj))
