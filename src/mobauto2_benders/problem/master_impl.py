from __future__ import annotations

from typing import Any, Optional

import pyomo.environ as pyo

from ..benders.master import MasterProblem
from ..benders.types import Candidate, Cut, SolveResult, SolveStatus


class ProblemMaster(MasterProblem):
    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.m: pyo.ConcreteModel | None = None
        self._cut_idx = 0
        self._lb: Optional[float] = None

    def _p(self, key: str, default: Any | None = None) -> Any:
        if self.params is None:
            return default
        return self.params.get(key, default)

    def initialize(self) -> None:
        Q = int(self._p("Q"))
        T = int(self._p("T"))
        trip_slots = int(self._p("trip_slots"))
        Emax = float(self._p("Emax"))
        L = float(self._p("L"))
        delta_chg = float(self._p("delta_chg"))
        binit = self._p("binit") or [0.0] * Q

        m = pyo.ConcreteModel()
        m.Q = range(Q)
        m.T = range(T)

        m.yOUT = pyo.Var(m.Q, m.T, within=pyo.Binary)
        m.yRET = pyo.Var(m.Q, m.T, within=pyo.Binary)
        m.c = pyo.Var(m.Q, m.T, within=pyo.Binary)
        m.s = pyo.Var(m.Q, m.T, within=pyo.Binary)
        m.inTrip = pyo.Var(m.Q, m.T, within=pyo.Binary)
        m.b = pyo.Var(m.Q, m.T, bounds=(0, Emax))
        m.gchg = pyo.Var(m.Q, m.T, within=pyo.NonNegativeReals)
        # Theta models recourse cost; keep nonnegative to avoid initial unboundedness
        m.theta = pyo.Var(within=pyo.NonNegativeReals)

        m.obj = pyo.Objective(expr=m.theta, sense=pyo.minimize)

        def exclusivity_rule(m, q, t):
            return m.yOUT[q, t] + m.yRET[q, t] + m.c[q, t] <= 1

        m.C1a = pyo.Constraint(m.Q, m.T, rule=exclusivity_rule)

        def intrip_rule(m, q, tau):
            return m.inTrip[q, tau] == sum(
                m.yOUT[q, t] + m.yRET[q, t] for t in m.T if t <= tau < t + trip_slots
            )

        m.C1b = pyo.Constraint(m.Q, m.T, rule=intrip_rule)

        def block_rule(m, q, t):
            return m.yOUT[q, t] + m.yRET[q, t] + m.c[q, t] <= 1 - m.inTrip[q, t]

        m.C1c = pyo.Constraint(m.Q, m.T, rule=block_rule)

        for t in range(T - trip_slots + 1, T):
            for q in m.Q:
                m.yOUT[q, t].fix(0)
                m.yRET[q, t].fix(0)

        def loc_flip(m, q, t):
            if t + trip_slots <= T - 1:
                return m.s[q, t + trip_slots] == m.s[q, t] + m.yOUT[q, t] - m.yRET[q, t]
            return pyo.Constraint.Skip

        m.C2a = pyo.Constraint(m.Q, m.T, rule=loc_flip)

        def admissible(m, q, t):
            return m.yOUT[q, t] + m.s[q, t] <= 1

        m.C2b = pyo.Constraint(m.Q, m.T, rule=admissible)
        m.C2c = pyo.Constraint(m.Q, m.T, rule=lambda m, q, t: m.yRET[q, t] <= m.s[q, t])
        m.C2d = pyo.Constraint(m.Q, m.T, rule=lambda m, q, t: m.c[q, t] <= 1 - m.s[q, t])

        for q in m.Q:
            m.s[q, 0].fix(0)
            m.s[q, T - 1].fix(0)

        for q in m.Q:
            m.b[q, 0].fix(float(binit[q]))
            for t in range(T - 1):
                m.add_component(
                    f"C4_bal_{q}_{t}",
                    pyo.Constraint(expr=m.b[q, t + 1] == m.b[q, t] - L * (m.yOUT[q, t] + m.yRET[q, t]) + m.gchg[q, t]),
                )
                m.add_component(
                    f"C4_chg1_{q}_{t}", pyo.Constraint(expr=m.gchg[q, t] <= delta_chg * m.c[q, t])
                )
                m.add_component(
                    f"C4_chg2_{q}_{t}", pyo.Constraint(expr=m.gchg[q, t] <= Emax - m.b[q, t])
                )

        m.C5 = pyo.Constraint(m.Q, m.T, rule=lambda m, q, t: m.b[q, t] >= 2 * L * m.yOUT[q, t])

        self.m = m

    def _solver(self) -> pyo.SolverFactory:
        name = str(self._p("solver", "cplex_direct"))
        solver = pyo.SolverFactory(name)
        opts = self._p("solver_options", {}) or {}
        for k, v in opts.items():
            solver.options[k] = v
        return solver

    def _collect_candidate(self) -> Candidate:
        assert self.m is not None
        m = self.m
        cand: Candidate = {}
        for q in m.Q:
            for t in m.T:
                cand[f"yOUT[{q},{t}]"] = pyo.value(m.yOUT[q, t])
                cand[f"yRET[{q},{t}]"] = pyo.value(m.yRET[q, t])
        cand["theta"] = pyo.value(m.theta)
        return cand

    def solve(self) -> SolveResult:
        assert self.m is not None, "Call initialize() before solve()"
        m = self.m
        res = self._solver().solve(m, tee=False)

        term = getattr(res.solver, "termination_condition", None)
        st = getattr(res.solver, "status", None)
        if term in (pyo.TerminationCondition.optimal,):
            status = SolveStatus.OPTIMAL
        elif term in (pyo.TerminationCondition.feasible, pyo.TerminationCondition.maxTimeLimit):
            status = SolveStatus.FEASIBLE
        elif term in (
            pyo.TerminationCondition.infeasible,
            pyo.TerminationCondition.infeasibleOrUnbounded,
        ):
            status = SolveStatus.INFEASIBLE
        else:
            status = SolveStatus.UNKNOWN

        objective = float(pyo.value(m.theta))
        self._lb = objective
        candidate = self._collect_candidate()
        return SolveResult(status=status, objective=objective, candidate=candidate, lower_bound=self._lb)

    def add_cut(self, cut: Cut) -> None:
        assert self.m is not None
        m = self.m
        const = float(cut.metadata.get("const", 0.0)) if hasattr(cut, "metadata") else 0.0
        coeff_yOUT = cut.metadata.get("coeff_yOUT") if hasattr(cut, "metadata") else None
        coeff_yRET = cut.metadata.get("coeff_yRET") if hasattr(cut, "metadata") else None

        expr = m.theta >= const

        if isinstance(coeff_yOUT, dict):
            expr = expr + sum(float(coeff_yOUT[q, t]) * m.yOUT[q, t] for (q, t) in coeff_yOUT)
        if isinstance(coeff_yRET, dict):
            expr = expr + sum(float(coeff_yRET[q, t]) * m.yRET[q, t] for (q, t) in coeff_yRET)

        if (not isinstance(coeff_yOUT, dict)) and (not isinstance(coeff_yRET, dict)) and cut.coeffs:
            for name, coef in cut.coeffs.items():
                if isinstance(name, str) and name.startswith("yOUT["):
                    parts = name[name.find("[") + 1 : name.find("]")].split(",")
                    q, t = int(parts[0]), int(parts[1])
                    expr = expr + float(coef) * m.yOUT[q, t]
                elif isinstance(name, str) and name.startswith("yRET["):
                    parts = name[name.find("[") + 1 : name.find("]")].split(",")
                    q, t = int(parts[0]), int(parts[1])
                    expr = expr + float(coef) * m.yRET[q, t]

        cname = f"benders_{self._cut_idx}"
        self._cut_idx += 1
        setattr(m, cname, pyo.Constraint(expr=expr))

    def best_lower_bound(self) -> Optional[float]:
        return self._lb
