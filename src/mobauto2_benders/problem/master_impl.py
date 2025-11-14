from __future__ import annotations

from typing import Any, Optional

import pyomo.environ as pyo

from ..benders.master import MasterProblem
from ..benders.subproblem import Subproblem
from ..benders.types import SubproblemResult

try:  # Optional import for CPLEX lazy callbacks
    import cplex  # type: ignore
    from cplex.callbacks import LazyConstraintCallback  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cplex = None
    LazyConstraintCallback = object  # type: ignore
from ..benders.types import Candidate, Cut, SolveResult, SolveStatus


class ProblemMaster(MasterProblem):
    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.m: pyo.ConcreteModel | None = None
        self._cut_idx = 0
        self._lb: Optional[float] = None
        # Fingerprints of cuts to avoid duplicates
        self._cut_fps: set[tuple] = set()
        self._lazy_installed: bool = False

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
        # Normalize initial battery vector to length Q
        _binit_raw = self._p("binit")
        if _binit_raw is None:
            binit = [0.0] * Q
        elif isinstance(_binit_raw, (int, float)):
            binit = [float(_binit_raw)] * Q
        else:
            try:
                binit = [float(x) for x in list(_binit_raw)]  # type: ignore[arg-type]
            except Exception:
                binit = [0.0] * Q
            if len(binit) < Q:
                fill = binit[-1] if binit else 0.0
                binit = binit + [fill] * (Q - len(binit))
            elif len(binit) > Q:
                binit = binit[:Q]

        m = pyo.ConcreteModel()
        m.Q = range(Q)
        m.T = range(T)

        m.yOUT = pyo.Var(m.Q, m.T, within=pyo.Binary)
        m.yRET = pyo.Var(m.Q, m.T, within=pyo.Binary)
        # Aggregated starts per time (to keep cuts sparse): Yout[t], Yret[t]
        m.Yout = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.Yret = pyo.Var(m.T, within=pyo.NonNegativeReals)
        # Time-bucket total starts z[t] = sum_q (yOUT+ yRET) -- used in cuts to reduce nnz
        m.Z = pyo.Var(m.T, within=pyo.NonNegativeReals)
        # Keep auxiliaries in relaxed domains except s which encodes location (discrete)
        m.c = pyo.Var(m.Q, m.T, bounds=(0.0, 1.0))
        m.s = pyo.Var(m.Q, m.T, within=pyo.Binary)
        m.inTrip = pyo.Var(m.Q, m.T, bounds=(0.0, 1.0))
        m.b = pyo.Var(m.Q, m.T, bounds=(0, Emax))
        m.gchg = pyo.Var(m.Q, m.T, within=pyo.NonNegativeReals)
        # Theta models recourse cost; keep nonnegative to avoid initial unboundedness
        m.theta = pyo.Var(within=pyo.NonNegativeReals)

        # Objective composition: theta + optional small start penalty + optional concurrency penalty
        eps_start = float(self._p("start_cost_epsilon", 0.0) or 0.0)
        conc_pen = float(self._p("concurrency_penalty", 0.0) or 0.0)

        # Optional concurrency penalty uses auxiliaries eOut[t], eRet[t] capturing excess starts beyond 1 per slot
        if conc_pen > 0.0:
            m.eOut = pyo.Var(m.T, within=pyo.NonNegativeReals)
            m.eRet = pyo.Var(m.T, within=pyo.NonNegativeReals)
            # eOut[t] >= Yout[t] - 1; eRet[t] >= Yret[t] - 1
            m.C_ex_out = pyo.Constraint(m.T, rule=lambda m, t: m.eOut[t] >= m.Yout[t] - 1)
            m.C_ex_ret = pyo.Constraint(m.T, rule=lambda m, t: m.eRet[t] >= m.Yret[t] - 1)

        obj_expr = m.theta
        if eps_start > 0.0:
            obj_expr = obj_expr + eps_start * sum(m.yOUT[q, t] + m.yRET[q, t] for q in m.Q for t in m.T)
        if conc_pen > 0.0:
            obj_expr = obj_expr + conc_pen * (sum(m.eOut[t] for t in m.T) + sum(m.eRet[t] for t in m.T))
        m.obj = pyo.Objective(expr=obj_expr, sense=pyo.minimize)

        def exclusivity_rule(m, q, t):
            return m.yOUT[q, t] + m.yRET[q, t] + m.c[q, t] <= 1

        m.C1a = pyo.Constraint(m.Q, m.T, rule=exclusivity_rule)

        # Define aggregation equalities for Yout/Yret
        m.Cagg_out = pyo.Constraint(m.T, rule=lambda m, t: m.Yout[t] == sum(m.yOUT[q, t] for q in m.Q))
        m.Cagg_ret = pyo.Constraint(m.T, rule=lambda m, t: m.Yret[t] == sum(m.yRET[q, t] for q in m.Q))
        # Time-bucket link: Z[t] = sum_q (yOUT+yRET)
        m.Cagg_z = pyo.Constraint(m.T, rule=lambda m, t: m.Z[t] == sum(m.yOUT[q, t] + m.yRET[q, t] for q in m.Q))

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

        # Location dynamics via arrival equality:
        def loc_flip(m, q, t):
            if t + trip_slots <= T - 1:
                return m.s[q, t + trip_slots] == m.s[q, t] + m.yOUT[q, t] - m.yRET[q, t]
            return pyo.Constraint.Skip

        m.C2a = pyo.Constraint(m.Q, m.T, rule=loc_flip)

        # Until the first possible arrival, the vehicle must remain at Longvilliers (s=0)
        for q in m.Q:
            for t in range(min(trip_slots, T)):
                m.s[q, t].fix(0)

        def admissible(m, q, t):
            return m.yOUT[q, t] + m.s[q, t] <= 1

        m.C2b = pyo.Constraint(m.Q, m.T, rule=admissible)
        m.C2c = pyo.Constraint(m.Q, m.T, rule=lambda m, q, t: m.yRET[q, t] <= m.s[q, t])
        m.C2d = pyo.Constraint(m.Q, m.T, rule=lambda m, q, t: m.c[q, t] <= 1 - m.s[q, t])

        # Note: We no longer enforce "first non-idle must be OUT".
        # Charging at Longvilliers (s=0) is allowed even before the first OUT.

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
                # Enforce that when charging is selected (c[q,t] = 1),
                # the battery increases by the nominal charge increment in that slot,
                # subject to the existing upper bound C4_chg2 (Emax - b).
                m.add_component(
                    f"C4_chg1_{q}_{t}", pyo.Constraint(expr=m.gchg[q, t] <= delta_chg * m.c[q, t])
                )
                m.add_component(
                    f"C4_chg1_lb_{q}_{t}", pyo.Constraint(expr=m.gchg[q, t] >= delta_chg * m.c[q, t])
                )
                m.add_component(
                    f"C4_chg2_{q}_{t}", pyo.Constraint(expr=m.gchg[q, t] <= Emax - m.b[q, t])
                )

        m.C5 = pyo.Constraint(m.Q, m.T, rule=lambda m, q, t: m.b[q, t] >= 2 * L * m.yOUT[q, t])

        # Avoid uninitialized gchg at the last time period (not used in constraints)
        for q in m.Q:
            m.gchg[q, T - 1].fix(0)
            # No effect from charging at the last slot (no next state), keep consistent
            m.c[q, T - 1].fix(0)

        # Container block to store explicit Benders cuts incrementally
        m.BendersCuts = pyo.Block(concrete=True)

        self.m = m

        # Create and retain a solver; if persistent CPLEX, set instance once
        solver_name = str(self._p("solver", "cplex_persistent"))
        self._solver = pyo.SolverFactory(solver_name)
        self._is_persistent = solver_name.lower() == "cplex_persistent"
        if self._is_persistent:
            # If using lazy cuts, we need symbolic labels to resolve indices by name
            use_lazy = bool(self._p("use_lazy_cuts", False))
            self._solver.set_instance(self.m, symbolic_solver_labels=use_lazy)
            # Set solver options once here
            opts = self._p("solver_options", {}) or {}
            for k, v in opts.items():
                self._solver.options[k] = v

    def _get_solver(self) -> pyo.SolverFactory:
        assert self.m is not None
        return self._solver

    # Optional: install a CPLEX lazy-constraint callback that generates Benders cuts
    # on-the-fly from the provided Subproblem implementation. Requires persistent CPLEX.
    def install_lazy_callback(self, subproblem: Subproblem) -> None:
        assert self.m is not None
        if str(self._p("solver", "")).lower() != "cplex_persistent":
            try:
                print("[BENDERS] Lazy cuts require solver=cplex_persistent; skipping callback install.")
            except Exception:
                pass
            return
        solver = self._get_solver()
        # Access underlying CPLEX model
        cpx = getattr(solver, "_solver_model", None)
        if (cplex is None) or (cpx is None):
            try:
                print("[BENDERS] CPLEX Python API not available; skipping lazy cuts.")
            except Exception:
                pass
            return

        m = self.m
        T = list(m.T)
        Q = list(m.Q)

        # Prepare variable names and indices in CPLEX
        theta_name = "theta"
        yout_names = [f"yOUT[{q},{t}]" for q in Q for t in T]
        yret_names = [f"yRET[{q},{t}]" for q in Q for t in T]
        Yout_names = [f"Yout[{t}]" for t in T]
        Yret_names = [f"Yret[{t}]" for t in T]

        try:
            idx_theta = cpx.variables.get_indices(theta_name)
            idx_yout = cpx.variables.get_indices(yout_names)
            idx_yret = cpx.variables.get_indices(yret_names)
            idx_Yout = cpx.variables.get_indices(Yout_names)
            idx_Yret = cpx.variables.get_indices(Yret_names)
        except Exception:
            # Ensure symbolic labels were used
            try:
                print("[BENDERS] Could not resolve variable indices by name; ensure symbolic_solver_labels=True.")
            except Exception:
                pass
            return

        # Build mappings from (q,t) and t to indices for fast lookup
        yout_index: dict[tuple[int, int], int] = {}
        yret_index: dict[tuple[int, int], int] = {}
        for i, q in enumerate(Q):
            for j, t in enumerate(T):
                pos = i * len(T) + j
                yout_index[(q, t)] = idx_yout[pos]
                yret_index[(q, t)] = idx_yret[pos]
        Yout_index: dict[int, int] = {t: idx_Yout[i] for i, t in enumerate(T)}
        Yret_index: dict[int, int] = {t: idx_Yret[i] for i, t in enumerate(T)}

        # Allow overriding the LP solver used by the subproblem inside the callback
        lp_override = str(self._p("lazy_cb_lp_solver", "")).strip() or None

        # Register callback
        class _BendersLazyCB(LazyConstraintCallback):  # type: ignore
            def __call__(self):  # noqa: D401
                # Build candidate from current integer solution
                cand: dict[str, float] = {}
                # yOUT
                for q in Q:
                    for t in T:
                        idx = yout_index[(q, t)]
                        try:
                            val = float(self.get_values(idx))
                        except Exception:
                            val = 0.0
                        cand[f"yOUT[{q},{t}]"] = val
                # yRET
                for q in Q:
                    for t in T:
                        idx = yret_index[(q, t)]
                        try:
                            val = float(self.get_values(idx))
                        except Exception:
                            val = 0.0
                        cand[f"yRET[{q},{t}]"] = val

                # Evaluate subproblem to get a cut at this candidate
                # Optionally override LP solver inside callback to avoid nested CPLEX
                old_lp = None
                try:
                    if lp_override is not None:
                        old_lp = subproblem.params.get("lp_solver")
                        subproblem.params["lp_solver"] = lp_override
                except Exception:
                    pass
                try:
                    sres: SubproblemResult = subproblem.evaluate(cand)
                except Exception:
                    # If SP fails, skip adding lazy cut
                    return
                finally:
                    try:
                        if lp_override is not None and old_lp is not None:
                            subproblem.params["lp_solver"] = old_lp
                    except Exception:
                        pass
                cuts = []
                if sres.cut is not None:
                    cuts.append(sres.cut)
                if getattr(sres, "cuts", None):
                    cuts.extend(sres.cuts)
                if not cuts:
                    return

                # We only add the first violated cut (can be extended to add multiple)
                cut = cuts[0]
                const = float(cut.metadata.get("const", 0.0)) if hasattr(cut, "metadata") else 0.0
                coeff_yOUT = cut.metadata.get("coeff_yOUT") if hasattr(cut, "metadata") else None
                coeff_yRET = cut.metadata.get("coeff_yRET") if hasattr(cut, "metadata") else None

                # Aggregate per time coefficients (raw dm) for Yout/Yret
                coeff_out_t: dict[int, float] = {}
                coeff_ret_t: dict[int, float] = {}
                if isinstance(coeff_yOUT, dict):
                    for (q, t), v in coeff_yOUT.items():
                        if t not in coeff_out_t:
                            coeff_out_t[t] = float(v)
                if isinstance(coeff_yRET, dict):
                    for (q, t), v in coeff_yRET.items():
                        if t not in coeff_ret_t:
                            coeff_ret_t[t] = float(v)

                # Re-anchor constant to pass through incumbent in Y-space
                # ub_est = const_dm + sum(dm*y_curr) using raw dm from metadata
                ub_est = float(const)
                if isinstance(coeff_yOUT, dict):
                    for (q, t), v in coeff_yOUT.items():
                        ub_est += float(v) * float(self.get_values(yout_index[(q, t)]))
                if isinstance(coeff_yRET, dict):
                    for (q, t), v in coeff_yRET.items():
                        ub_est += float(v) * float(self.get_values(yret_index[(q, t)]))
                # Keep const as provided (already passes through incumbent)
                const = float(const)

                # Build LHS: theta - sum_t (dm_out[t]*Yout[t] + dm_ret[t]*Yret[t]) >= const
                inds: list[int] = [idx_theta]
                vals: list[float] = [1.0]
                for t, a in coeff_out_t.items():
                    if t in Yout_index:
                        inds.append(Yout_index[t])
                        vals.append(-float(a))
                for t, a in coeff_ret_t.items():
                    if t in Yret_index:
                        inds.append(Yret_index[t])
                        vals.append(-float(a))

                # Check violation at current solution using Option A: viol = RHS - LHS
                lhs_val = 0.0
                try:
                    lhs_val += float(self.get_values(idx_theta))
                except Exception:
                    pass
                for idx, coef in zip(inds[1:], vals[1:]):
                    try:
                        lhs_val += coef * float(self.get_values(idx))
                    except Exception:
                        pass
                rhs_val = float(const)
                abs_tol = 1e-6
                rel_tol = 1e-6
                viol = rhs_val - lhs_val
                thr = abs_tol + rel_tol * abs(rhs_val)
                if not (viol > thr):
                    return

                # Add lazy constraint
                try:
                    spair = cplex.SparsePair(ind=inds, val=vals)
                    self.add(spair, "G", const)
                except Exception:
                    # Alternate signature fallback
                    try:
                        self.add(inds, vals, "G", const)
                    except Exception:
                        pass

        cb = cpx.register_callback(_BendersLazyCB)  # noqa: F841
        self._lazy_installed = True
        print("[BENDERS] Installed CPLEX lazy constraint callback for Benders cuts.")

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
        tee_flag = bool(self._p("solver_tee", self._p("mp_solve_tee", False)))
        if getattr(self, "_is_persistent", False):
            # Persistent: do not pass model
            res = solver.solve(tee=tee_flag)
        else:
            res = solver.solve(m, tee=tee_flag)

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

    def _add_cut(self, cut: Cut, force: bool = False) -> bool:
        assert self.m is not None
        m = self.m
        const = float(cut.metadata.get("const", 0.0)) if hasattr(cut, "metadata") else 0.0
        coeff_yOUT = cut.metadata.get("coeff_yOUT") if hasattr(cut, "metadata") else None
        coeff_yRET = cut.metadata.get("coeff_yRET") if hasattr(cut, "metadata") else None

        # Build RHS: theta >= const + sum(beta_out*yOUT) + sum(beta_ret*yRET)
        rhs = const

        # Optionally aggregate coefficients by time to use Yout/Yret and reduce density
        aggregate = bool(self._p("aggregate_cuts_by_tau", True))
        coeff_tol = float(self._p("cut_coeff_threshold", 0.0) or 0.0)
        # Aggregate raw dm slopes per time for OUT/RET to use Yout/Yret
        # We keep raw dm (can be negative), using one coefficient per time bucket.
        agg_out: dict[int | tuple[int, int], float] = {}
        agg_ret: dict[int | tuple[int, int], float] = {}
        raw_pos_dm = 0
        if isinstance(coeff_yOUT, dict):
            if aggregate:
                used = set()
                for (q, t), v in coeff_yOUT.items():
                    vraw = float(v)
                    if vraw > coeff_tol:
                        raw_pos_dm += 1
                    if t in used or abs(vraw) <= coeff_tol:
                        continue
                    used.add(t)
                    agg_out[t] = vraw
            else:
                for (q, t), v in coeff_yOUT.items():
                    vraw = float(v)
                    if vraw > coeff_tol:
                        raw_pos_dm += 1
                    if abs(vraw) <= coeff_tol:
                        continue
                    agg_out[(q, t)] = vraw
        if isinstance(coeff_yRET, dict):
            if aggregate:
                used = set()
                for (q, t), v in coeff_yRET.items():
                    vraw = float(v)
                    if vraw > coeff_tol:
                        raw_pos_dm += 1
                    if t in used or abs(vraw) <= coeff_tol:
                        continue
                    used.add(t)
                    agg_ret[t] = vraw
            else:
                for (q, t), v in coeff_yRET.items():
                    vraw = float(v)
                    if vraw > coeff_tol:
                        raw_pos_dm += 1
                    if abs(vraw) <= coeff_tol:
                        continue
                    agg_ret[(q, t)] = vraw

        # Re-anchor the constant so that the cut passes through the incumbent (y = current MP solution)
        # Using raw dm and Yout/Yret:
        # RHS = const_dm + sum_t dm_out[t]*Yout[t] + sum_t dm_ret[t]*Yret[t]
        # This passes through incumbent since const_dm came from SP with dm and sums.
        ub_est = float(const)
        # Sum over raw dm maps using current y values
        if isinstance(coeff_yOUT, dict):
            for (q, t), v in coeff_yOUT.items():
                yv = float(m.yOUT[q, t].value or 0.0)
                ub_est += float(v) * yv
        if isinstance(coeff_yRET, dict):
            for (q, t), v in coeff_yRET.items():
                yv = float(m.yRET[q, t].value or 0.0)
                ub_est += float(v) * yv
        # With Yout/Yret form, const stays as provided; ub_est is only used for diagnostics.
        # Keep const unchanged to preserve pass-through at incumbent.

        # Assemble RHS with aggregated Yout/Yret and raw dm
        if aggregate:
            for t, v in agg_out.items():
                rhs = rhs + float(v) * m.Yout[int(t)]
            for t, v in agg_ret.items():
                rhs = rhs + float(v) * m.Yret[int(t)]
        else:
            for key, v in agg_out.items():
                q, t = key  # type: ignore[misc]
                rhs = rhs + float(v) * m.yOUT[int(q), int(t)]
            for key, v in agg_ret.items():
                q, t = key  # type: ignore[misc]
                rhs = rhs + float(v) * m.yRET[int(q), int(t)]

        if (not isinstance(coeff_yOUT, dict)) and (not isinstance(coeff_yRET, dict)) and cut.coeffs:
            for name, coef in cut.coeffs.items():
                v2 = float(coef)
                if abs(v2) <= coeff_tol:
                    continue
                if isinstance(name, str) and name.startswith("yOUT["):
                    parts = name[name.find("[") + 1 : name.find("]")].split(",")
                    q, t = int(parts[0]), int(parts[1])
                    rhs = rhs - v2 * m.yOUT[q, t]
                elif isinstance(name, str) and name.startswith("yRET["):
                    parts = name[name.find("[") + 1 : name.find("]")].split(",")
                    q, t = int(parts[0]), int(parts[1])
                    rhs = rhs - v2 * m.yRET[q, t]

        # Scale both sides to normalize coefficient magnitudes (theta and RHS)
        max_beta = 0.0
        if agg_out:
            max_beta = max(max_beta, max(abs(float(v)) for v in agg_out.values()))
        if agg_ret:
            max_beta = max(max_beta, max(abs(float(v)) for v in agg_ret.values()))
        max_term = max(float(abs(const)), float(max_beta), 1.0)
        scale = 1.0 / max_term
        lhs = scale * m.theta
        rhs_scaled = scale * rhs

        # Evaluate violation using Option A: viol = RHS - theta
        add_abs_tol = float(self._p("cut_add_tolerance", 1e-6))
        add_rel_tol = float(self._p("cut_add_rel_tolerance", 1e-6))
        lhs_val = float(pyo.value(lhs))
        rhs_val = float(pyo.value(rhs_scaled))
        violation = rhs_val - lhs_val
        threshold = add_abs_tol + add_rel_tol * abs(rhs_val)
        if not force and not (violation > threshold):
            print(
                f"[BENDERS] Eval cut: lhs={lhs_val:.6g} rhs={rhs_val:.6g} viol={violation:.3g} thr={threshold:.3g} added=False"
            )
            return False

        # Fingerprint to avoid adding duplicates
        def _round6(x: float) -> float:
            return float(f"{x:.6g}")
        fp_out = []
        fp_ret = []
        # Fingerprint on Yout/Yret aggregated slopes (bypass if force=True)
        if aggregate:
            fp_out = sorted((int(t), _round6(float(v))) for t, v in agg_out.items())
            fp_ret = sorted((int(t), _round6(float(v))) for t, v in agg_ret.items())
        else:
            fp_out = sorted(((int(q), int(t), _round6(float(v))) for (q, t), v in agg_out.items()))  # type: ignore[misc]
            fp_ret = sorted(((int(q), int(t), _round6(float(v))) for (q, t), v in agg_ret.items()))  # type: ignore[misc]
        fp = ("opt", _round6(const), tuple(fp_out), tuple(fp_ret))
        if not force:
            if fp in self._cut_fps:
                print("[BENDERS] Skipped duplicate cut")
                return False
            self._cut_fps.add(fp)

        # Create explicit constraint and register with persistent solver if used
        cname = f"benders_cut_{self._cut_idx}"
        con = pyo.Constraint(expr=(lhs >= rhs_scaled))
        setattr(m.BendersCuts, cname, con)
        if getattr(self, "_is_persistent", False):
            self._solver.add_constraint(con)
        # Maintain pool for optional pruning
        if not hasattr(self, "_cut_cons"):
            self._cut_cons = []
            self._cut_names = []
        self._cut_cons.append(con)
        self._cut_names.append(cname)

        # Cut pool management: cap number of active cuts
        max_cuts = int(self._p("max_cuts_active", 0) or 0)
        if max_cuts > 0 and len(self._cut_cons) > max_cuts:
            # Remove the oldest cuts until within limit
            to_remove = len(self._cut_cons) - max_cuts
            for _ in range(to_remove):
                con_old = self._cut_cons.pop(0)
                name_old = self._cut_names.pop(0)
                if getattr(self, "_is_persistent", False):
                    self._solver.remove_constraint(con_old)
                # Remove from Pyomo block
                try:
                    delattr(m.BendersCuts, name_old)
                except Exception:
                    pass

        # Simple logging: constant and nonzeros
        nnz = (len(agg_out) if aggregate else len(agg_out)) + (len(agg_ret) if aggregate else len(agg_ret))
        # Log slope range
        all_betas = list(agg_out.values()) + list(agg_ret.values())
        if all_betas:
            rng = (min(all_betas), max(all_betas))
            print(
                f"[BENDERS] Added cut #{self._cut_idx}: const={const:.6g}, nnz={nnz}, slope_range=[{rng[0]:.3g},{rng[1]:.3g}], raw_pos_dm={raw_pos_dm}, scale={scale:.3g}"
            )
        else:
            print(f"[BENDERS] Added cut #{self._cut_idx}: const={const:.6g}, nnz={nnz}, raw_pos_dm={raw_pos_dm}, scale={scale:.3g}")
        # Sanity log with LHS and RHS values (scaled)
        print(
            f"[BENDERS] Eval cut: lhs={lhs_val:.6g} rhs={rhs_val:.6g} viol={violation:.3g} thr={threshold:.3g} added=True"
        )
        self._cut_idx += 1
        # Track last cut info for cross-iteration checks
        self._last_cut_const = const
        self._last_cut_nnz = nnz
        return True

    def add_cut(self, cut: Cut) -> None:
        # Normal filtered path
        self._add_cut(cut, force=False)

    def add_cut_force(self, cut: Cut) -> bool:
        # Force-accept path for the first violated cut of an iteration.
        # Returns True if a cut was added.
        return self._add_cut(cut, force=True)

    def best_lower_bound(self) -> Optional[float]:
        return self._lb

    # Introspection helpers
    def cuts_count(self) -> int:
        return int(self._cut_idx)

    def last_cut_info(self) -> tuple[float | None, int | None]:
        return getattr(self, "_last_cut_const", None), getattr(self, "_last_cut_nnz", None)

    def lazy_installed(self) -> bool:
        return bool(self._lazy_installed)
