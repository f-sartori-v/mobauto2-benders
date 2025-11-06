from __future__ import annotations

from typing import Dict, Tuple

import pyomo.environ as pyo


def build_master(P) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()
    Q, T = range(P.Q), range(P.T)

    m.yOUT = pyo.Var(Q, T, within=pyo.Binary)
    m.yRET = pyo.Var(Q, T, within=pyo.Binary)
    m.c = pyo.Var(Q, T, within=pyo.Binary)  # charging
    m.s = pyo.Var(Q, T, within=pyo.Binary)  # location: Massy=1, Longvilliers=0
    m.inTrip = pyo.Var(Q, T, within=pyo.Binary)
    m.b = pyo.Var(Q, T, bounds=(0, P.Emax))  # SoC
    m.gchg = pyo.Var(Q, T, within=pyo.NonNegativeReals)
    m.theta = pyo.Var(within=pyo.NonNegativeReals)

    # Objective: min θ
    m.obj = pyo.Objective(expr=m.theta, sense=pyo.minimize)

    # C1 exclusivity and blocking (8.1)–(8.4)
    def exclusivity_rule(m, q, t):
        return m.yOUT[q, t] + m.yRET[q, t] + m.c[q, t] <= 1

    m.C1a = pyo.Constraint(Q, T, rule=exclusivity_rule)

    def intrip_rule(m, q, tau):
        return m.inTrip[q, tau] == sum(m.yOUT[q, t] + m.yRET[q, t] for t in T if t <= tau < t + P.trip_slots)

    m.C1b = pyo.Constraint(Q, T, rule=intrip_rule)

    def block_rule(m, q, t):
        return m.yOUT[q, t] + m.yRET[q, t] + m.c[q, t] <= 1 - m.inTrip[q, t]

    m.C1c = pyo.Constraint(Q, T, rule=block_rule)

    for t in range(P.T - P.trip_slots + 1, P.T):
        for q in Q:
            m.yOUT[q, t].fix(0)
            m.yRET[q, t].fix(0)  # (8.4)

    # C2 location flip and admissible actions (8.5)–(8.6)
    def loc_flip(m, q, t):
        if t + P.trip_slots <= P.T - 1:
            return m.s[q, t + P.trip_slots] == m.s[q, t] + m.yOUT[q, t] - m.yRET[q, t]
        return pyo.Constraint.Skip

    m.C2a = pyo.Constraint(Q, T, rule=loc_flip)

    def admissible(m, q, t):
        return pyo.inequality(0, m.yOUT[q, t], 1 - m.s[q, t])

    m.C2b = pyo.Constraint(Q, T, rule=admissible)

    m.C2c = pyo.Constraint(Q, T, rule=lambda m, q, t: m.yRET[q, t] <= m.s[q, t])
    m.C2d = pyo.Constraint(Q, T, rule=lambda m, q, t: m.c[q, t] <= 1 - m.s[q, t])

    # C3 start/end at Longvilliers (8.7)
    for q in Q:
        m.s[q, 0].fix(0)
        m.s[q, P.T - 1].fix(0)

    # C4 battery (8.8)–(8.10)
    for q in Q:
        m.b[q, 0].fix(P.binit[q])
        for t in range(P.T - 1):
            m.add_component(
                f"C4_bal_{q}_{t}",
                pyo.Constraint(expr=m.b[q, t + 1] == m.b[q, t] - P.L * (m.yOUT[q, t] + m.yRET[q, t]) + m.gchg[q, t]),
            )
            m.add_component(f"C4_chg1_{q}_{t}", pyo.Constraint(expr=m.gchg[q, t] <= P.delta_chg * m.c[q, t]))
            m.add_component(f"C4_chg2_{q}_{t}", pyo.Constraint(expr=m.gchg[q, t] <= P.Emax - m.b[q, t]))

    # C5 energy to start a trip (8.11)
    m.C5 = pyo.Constraint(Q, T, rule=lambda m, q, t: m.b[q, t] >= 2 * P.L * m.yOUT[q, t])

    return m


def add_benders_cut(m: pyo.ConcreteModel, coeff_const: float, coeff_yOUT: Dict[Tuple[int, int], float], coeff_yRET: Dict[Tuple[int, int], float]) -> None:
    """
    θ >= const + sum_{q,τ} [ coeff_yOUT[q,τ] * yOUT_{q,τ} + coeff_yRET[q,τ] * yRET_{q,τ} ]
    """
    expr = m.theta >= float(coeff_const) \
        + sum(float(coeff_yOUT[q, t]) * m.yOUT[q, t] for (q, t) in coeff_yOUT) \
        + sum(float(coeff_yRET[q, t]) * m.yRET[q, t] for (q, t) in coeff_yRET)
    cname = f"benders_{len(m.component_map(pyo.Constraint))}"
    setattr(m, cname, pyo.Constraint(expr=expr))


def solve_mip(m: pyo.ConcreteModel, solver_name: str = "glpk", tee: bool = False, executable: str | None = None, options: dict | None = None) -> None:
    solver = pyo.SolverFactory(solver_name)
    if executable:
        try:
            solver.executable = executable  # type: ignore[attr-defined]
        except Exception:
            pass
    if options:
        for k, v in options.items():
            solver.options[k] = v
    solver.solve(m, tee=tee)
