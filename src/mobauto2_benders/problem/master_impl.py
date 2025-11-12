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
        # Time discretization: prefer minutes + slot_resolution + trip_duration_minutes
        import math
        slot_res = int(self._p("slot_resolution", 1))
        T_minutes = self._p("T_minutes")
        trip_dur_min = self._p("trip_duration_minutes", self._p("trip_duration"))
        if T_minutes is not None:
            T = int(int(T_minutes) // max(1, slot_res))
        else:
            T = int(self._p("T"))
        if trip_dur_min is not None:
            trip_slots = int(math.ceil(float(trip_dur_min) / max(1, slot_res)))
        else:
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

        # Add a tiny start cost epsilon to help branching and discourage gratuitous starts
        eps_start = float(self._p("start_cost_epsilon", 0.0))
        if eps_start and eps_start > 0:
            m.obj = pyo.Objective(
                expr=m.theta
                + eps_start
                * sum(m.yOUT[q, t] + m.yRET[q, t] for q in m.Q for t in m.T),
                sense=pyo.minimize,
            )
        else:
            m.obj = pyo.Objective(expr=m.theta, sense=pyo.minimize)

        def exclusivity_rule(m, q, t):
            return m.yOUT[q, t] + m.yRET[q, t] + m.c[q, t] <= 1

        m.C1a = pyo.Constraint(m.Q, m.T, rule=exclusivity_rule)

        # inTrip implications (lighter than equality linking):
        # Lower bounds: inTrip[q,t] >= yOUT[q,u], inTrip[q,t] >= yRET[q,u] for u in (t-trip_slots+1..t-1)
        # Upper bound:  inTrip[q,t] <= sum_{u in (t-trip_slots+1..t-1)} (yOUT[q,u]+yRET[q,u])
        for q in m.Q:
            for t in m.T:
                lo = max(0, t - trip_slots + 1)
                hi = t - 1
                if lo <= hi:
                    # Upper bound
                    m.add_component(
                        f"C1b_intrip_ub_{q}_{t}",
                        pyo.Constraint(
                            expr=m.inTrip[q, t]
                            <= sum(m.yOUT[q, u] + m.yRET[q, u] for u in range(lo, hi + 1))
                        ),
                    )
                    # Lower bounds
                    for u in range(lo, hi + 1):
                        m.add_component(
                            f"C1b_intrip_lb_out_{q}_{t}_{u}",
                            pyo.Constraint(expr=m.inTrip[q, t] >= m.yOUT[q, u]),
                        )
                        m.add_component(
                            f"C1b_intrip_lb_ret_{q}_{t}_{u}",
                            pyo.Constraint(expr=m.inTrip[q, t] >= m.yRET[q, u]),
                        )
                else:
                    # No prior starts to account for
                    m.add_component(
                        f"C1b_intrip_zero_{q}_{t}", pyo.Constraint(expr=m.inTrip[q, t] == 0)
                    )

        # Block actions when in trip (keeps starts and charging off while busy)
        def block_rule(m, q, t):
            return m.yOUT[q, t] + m.yRET[q, t] + m.c[q, t] <= 1 - m.inTrip[q, t]

        m.C1c = pyo.Constraint(m.Q, m.T, rule=block_rule)

        # No two starts inside any trip-duration window (avoid overlapping trips)
        for q in m.Q:
            for w in range(T):
                w_end = min(T - 1, w + trip_slots - 1)
                m.add_component(
                    f"C1d_window_atmost1_{q}_{w}",
                    pyo.Constraint(
                        expr=sum(m.yOUT[q, u] + m.yRET[q, u] for u in range(w, w_end + 1)) <= 1
                    ),
                )

        # Disallow starting trips that cannot finish within the horizon
        # Allowed starts must satisfy t + trip_slots <= T - 1 -> t <= T - trip_slots - 1
        # Therefore, fix starts at t in [T - trip_slots, T-1] to 0
        for t in range(T - trip_slots, T):
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
            # Ensure no trip is ongoing at the end of horizon
            m.inTrip[q, T - 1].fix(0)

        # FIFO symmetry-breaking across vehicles: cumulative starts for k
        # cannot exceed those for k-1 over any time prefix
        if Q >= 2:
            for k in range(1, Q):
                for t in range(T):
                    m.add_component(
                        f"C3_fifo_{k}_{t}",
                        pyo.Constraint(
                            expr=
                            sum(m.yOUT[k, tau] + m.yRET[k, tau] for tau in range(0, t + 1))
                            <= sum(m.yOUT[k - 1, tau] + m.yRET[k - 1, tau] for tau in range(0, t + 1))
                        ),
                    )

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

        # Avoid uninitialized gchg at the last time period (not used in constraints)
        for q in m.Q:
            m.gchg[q, T - 1].fix(0)

        # Prepare a constraint list to store Benders cuts incrementally
        m.BendersCuts = pyo.ConstraintList()

        self.m = m

        # Create and retain a persistent solver; set instance once
        # Prefer cplex_persistent for incremental cut addition
        solver_name = str(self._p("solver", "cplex_persistent"))
        self._solver = pyo.SolverFactory(solver_name)
        try:
            # For persistent solvers, set the instance now
            if hasattr(self._solver, "set_instance"):
                self._solver.set_instance(self.m, symbolic_solver_labels=True)
        except Exception:
            # Fallback: keep non-persistent behavior if not supported
            pass

    def _get_solver(self) -> pyo.SolverFactory:
        assert self.m is not None
        # Initialize solver if needed (covers non-persistent or failed init)
        if getattr(self, "_solver", None) is None:
            solver_name = str(self._p("solver", "cplex_persistent"))
            self._solver = pyo.SolverFactory(solver_name)
        # Apply options each time (cheap)
        opts = self._p("solver_options", {}) or {}
        for k, v in opts.items():
            try:
                self._solver.options[k] = v
            except Exception:
                pass
        return self._solver

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
        solver = self._get_solver()
        # Prefer calling with model (works for direct solvers). If the solver is
        # persistent and does not accept a model arg, fall back gracefully.
        try:
            res = solver.solve(m, tee=False)
        except (TypeError, IndexError):
            res = solver.solve(tee=False)

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

    # Pretty-print the current master solution (if solved)
    def format_solution(self) -> str:
        assert self.m is not None
        m = self.m
        Q = list(m.Q)
        T = list(m.T)
        lines: list[str] = []
        lines.append(f"Q={len(Q)} T={len(T)}")
        # Theta
        try:
            lines.append(f"theta = {pyo.value(m.theta):.6g}")
        except Exception:
            lines.append("theta = (unavailable)")
        # Binary schedules
        def row(var, q):
            vals = []
            for t in T:
                v = var[q, t].value
                if v is None:
                    vals.append("-")
                else:
                    vals.append(str(int(round(float(v)))))
            return " ".join(vals)

        # Compact per-shuttle timeline with labels: OUT, INT, RET, CHR, IDL
        def lbl(q: int, t: int) -> str:
            yout = m.yOUT[q, t].value or 0.0
            yret = m.yRET[q, t].value or 0.0
            intr = m.inTrip[q, t].value or 0.0
            chg = m.c[q, t].value or 0.0
            if yout >= 0.5:
                return "OUT"
            if yret >= 0.5:
                return "RET"
            if intr >= 0.5:
                return "INT"
            if chg >= 0.5:
                return "CHR"
            return "IDL"

        lines.append("Timeline (per shuttle):")
        header = "       " + " ".join(f"{t:>3d}" for t in T)
        lines.append(header)
        for q in Q:
            seq = " ".join(f"{lbl(q, t):>3s}" for t in T)
            lines.append(f"  q={q}: {seq}")

        # Also show battery levels over time
        def rowf(var, q):
            vals = []
            for t in T:
                v = var[q, t].value
                if v is None:
                    vals.append("  -")
                else:
                    vals.append(f"{float(v):>3.0f}")
            return " ".join(vals)

        lines.append("Battery (per shuttle):")
        lines.append(header)
        for q in Q:
            lines.append(f"  q={q}: {rowf(m.b, q)}")

        return "\n".join(lines)

    def add_cut(self, cut: Cut) -> None:
        assert self.m is not None
        m = self.m
        const = float(cut.metadata.get("const", 0.0)) if hasattr(cut, "metadata") else 0.0
        coeff_yOUT = cut.metadata.get("coeff_yOUT") if hasattr(cut, "metadata") else None
        coeff_yRET = cut.metadata.get("coeff_yRET") if hasattr(cut, "metadata") else None

        # Build RHS of the cut as a linear expression, then form inequality
        rhs = const

        if isinstance(coeff_yOUT, dict):
            rhs = rhs + sum(float(coeff_yOUT[q, t]) * m.yOUT[q, t] for (q, t) in coeff_yOUT)
        if isinstance(coeff_yRET, dict):
            rhs = rhs + sum(float(coeff_yRET[q, t]) * m.yRET[q, t] for (q, t) in coeff_yRET)

        if (not isinstance(coeff_yOUT, dict)) and (not isinstance(coeff_yRET, dict)) and cut.coeffs:
            for name, coef in cut.coeffs.items():
                if isinstance(name, str) and name.startswith("yOUT["):
                    parts = name[name.find("[") + 1 : name.find("]")].split(",")
                    q, t = int(parts[0]), int(parts[1])
                    rhs = rhs + float(coef) * m.yOUT[q, t]
                elif isinstance(name, str) and name.startswith("yRET["):
                    parts = name[name.find("[") + 1 : name.find("]")].split(",")
                    q, t = int(parts[0]), int(parts[1])
                    rhs = rhs + float(coef) * m.yRET[q, t]

        expr = m.theta >= rhs

        # Add to a ConstraintList for uniqueness and persistence
        idx = m.BendersCuts.add(expr)

        # If using a persistent solver, inform it about the new constraint
        try:
            solver = self._get_solver()
            if hasattr(solver, "add_constraint"):
                solver.add_constraint(m.BendersCuts[idx])
        except Exception:
            pass

        # Simple logging: constant and nonzeros
        nnz = 0
        if isinstance(coeff_yOUT, dict):
            nnz += len([1 for v in coeff_yOUT.values() if abs(float(v)) > 1e-12])
        if isinstance(coeff_yRET, dict):
            nnz += len([1 for v in coeff_yRET.values() if abs(float(v)) > 1e-12])
        # For cuts built from generic name/coef pairs
        if (not isinstance(coeff_yOUT, dict)) and (not isinstance(coeff_yRET, dict)) and cut.coeffs:
            nnz += len([1 for (name, coef) in cut.coeffs.items() if abs(float(coef)) > 1e-12])
        try:
            print(f"[BENDERS] Added cut #{self._cut_idx}: const={const:.6g}, nnz={nnz}")
        except Exception:
            pass
        self._cut_idx += 1

    def best_lower_bound(self) -> Optional[float]:
        return self._lb
