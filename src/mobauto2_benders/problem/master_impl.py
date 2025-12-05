from __future__ import annotations

from typing import Any, Optional
from pathlib import Path

import pyomo.environ as pyo

from ..benders.master import MasterProblem
from ..benders.subproblem import Subproblem
from ..benders.types import SubproblemResult
from ..benders.solver import add_benders_cut  # shared cut filtering

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
        # Optional: warm start values for yOUT/yRET at next solve
        # Keys: ("yOUT"|"yRET", q:int, t:int) -> float(0/1)
        self._warm_start: dict[tuple[str, int, int], float] | None = None

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

        # Vehicle location encoding: 0 = Longvilliers (depot), 1 = Massy
        # All shuttles start at Longvilliers and must end at Longvilliers.

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
        # State and action variables
        # c[q,t] models a continuous charging intensity in [0,1] (fraction of slot/power level)
        m.c = pyo.Var(m.Q, m.T, bounds=(0.0, 1.0))
        # Discrete occupancy at locations: 0=Longvilliers (atL=1), 1=Massy (atM=1)
        m.atL = pyo.Var(m.Q, m.T, within=pyo.Binary)
        m.atM = pyo.Var(m.Q, m.T, within=pyo.Binary)
        # Traveling indicator per slot; make it binary to encode exclusivity windows exactly
        m.inTrip = pyo.Var(m.Q, m.T, within=pyo.Binary)
        m.b = pyo.Var(m.Q, m.T, bounds=(0, Emax))
        m.gchg = pyo.Var(m.Q, m.T, within=pyo.NonNegativeReals)
        # Theta models recourse cost; keep nonnegative to avoid initial unboundedness
        # Optionally: per-scenario thetas and/or split by direction.
        use_theta_per_scen = bool(self._p("theta_per_scenario", False))
        S = int(self._p("num_scenarios", 0) or 0)
        if use_theta_per_scen and S <= 0:
            use_theta_per_scen = False
        # If using per-scenario thetas, keep single theta per scenario for simplicity; otherwise allow dir split
        disagg_dir = False if use_theta_per_scen else bool(self._p("disaggregate_theta_by_direction", self._p("theta_split_by_direction", True)))
        if use_theta_per_scen:
            m.Scenarios = range(S)
            m.theta_s = pyo.Var(m.Scenarios, within=pyo.NonNegativeReals)
        elif disagg_dir:
            m.theta_out = pyo.Var(within=pyo.NonNegativeReals)
            m.theta_ret = pyo.Var(within=pyo.NonNegativeReals)
        else:
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

        # Build objective: combine theta terms depending on config
        if use_theta_per_scen:
            wts = list(self._p("scenario_weights", []) or [])
            if not wts or len(wts) != S:
                wts = [1.0 / float(max(1, S)) for _ in range(max(1, S))]
            obj_expr = sum(float(wts[s]) * m.theta_s[s] for s in range(S))
        else:
            obj_expr = (m.theta_out + m.theta_ret) if disagg_dir else m.theta
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

        # inTrip equality: 1 during travel slots strictly after a start until arrival
        # For a start at time u, travel occupies slots t in {u+1, ..., u+trip_slots-1}
        for q in m.Q:
            for t in m.T:
                lo = max(0, t - trip_slots + 1)
                hi = t - 1
                if lo <= hi:
                    m.add_component(
                        f"C1b_intrip_eq_{q}_{t}",
                        pyo.Constraint(expr=m.inTrip[q, t] == sum(m.yOUT[q, u] + m.yRET[q, u] for u in range(lo, hi + 1))),
                    )
                else:
                    m.add_component(
                        f"C1b_intrip_zero_{q}_{t}", pyo.Constraint(expr=m.inTrip[q, t] == 0)
                    )

        # Block actions when in trip (keeps starts and charging off while busy)
        m.C1c = pyo.Constraint(m.Q, m.T, rule=lambda m, q, t: m.yOUT[q, t] + m.yRET[q, t] + m.c[q, t] <= 1 - m.inTrip[q, t])

        # Occupancy conservation: atL + atM + inTrip == 1 each slot
        m.Cocc = pyo.Constraint(m.Q, m.T, rule=lambda m, q, t: m.atL[q, t] + m.atM[q, t] + m.inTrip[q, t] == 1)

        # Disallow starting trips that cannot finish within the horizon
        # Allowed starts must satisfy t + trip_slots <= T - 1 -> t <= T - trip_slots - 1
        # Therefore, fix starts at t in [T - trip_slots, T-1] to 0
        for t in range(T - trip_slots, T):
            for q in m.Q:
                m.yOUT[q, t].fix(0)
                m.yRET[q, t].fix(0)

        # Enforce "returnability" before horizon end: an OUT must leave enough time for a RET to return
        # A paired OUT+RET requires 2*trip_slots to return to Longvilliers by T-1, so forbid OUT at t >= T - 2*trip_slots
        t_cut = max(0, T - 2 * trip_slots)
        for t in range(t_cut, T):
            for q in m.Q:
                m.yOUT[q, t].fix(0)

        # Occupancy recursions: leave Longvilliers when starting OUT; arrive to Longvilliers after RET duration
        for q in m.Q:
            for t in range(1, T):
                # Arrivals from RET into Longvilliers at t from starts at (t - trip_slots)
                arr_ret = m.yRET[q, t - trip_slots] if (t - trip_slots) >= 0 else 0
                m.add_component(
                    f"C2a_locL_{q}_{t}",
                    pyo.Constraint(expr=m.atL[q, t] == m.atL[q, t - 1] - m.yOUT[q, t - 1] + arr_ret),
                )
                # Arrivals from OUT into Massy at t from starts at (t - trip_slots)
                arr_out = m.yOUT[q, t - trip_slots] if (t - trip_slots) >= 0 else 0
                m.add_component(
                    f"C2a_locM_{q}_{t}",
                    pyo.Constraint(expr=m.atM[q, t] == m.atM[q, t - 1] - m.yRET[q, t - 1] + arr_out),
                )

        # Gating by occupancy
        m.C2b = pyo.Constraint(m.Q, m.T, rule=lambda m, q, t: m.yOUT[q, t] <= m.atL[q, t])
        m.C2c = pyo.Constraint(m.Q, m.T, rule=lambda m, q, t: m.yRET[q, t] <= m.atM[q, t])
        m.C2d = pyo.Constraint(m.Q, m.T, rule=lambda m, q, t: m.c[q, t] <= m.atL[q, t])

        # Note: We no longer enforce "first non-idle must be OUT".
        # Charging at Longvilliers (s=0) is allowed even before the first OUT.

        # Never recharge right after an idle slot at Longvilliers.
        # Interpretation: if the previous slot at Longvilliers had neither a departure nor charging
        # (i.e., it was an idle/wait slot), then charging in the current slot is not allowed.
        # Linear form: c[q,t] <= yOUT[q,t-1] + c[q,t-1] + (1 - atL[q,t-1])
        # - If prev was idle at L: yOUT=0, c=0, atL=1 -> RHS=0, so c[q,t]=0
        # - If prev not at L (travel/Massy): atL=0 -> RHS>=1, no restriction beyond other gates
        # - If prev had charge or departure: RHS>=something positive, allowing continued charge if feasible
        for q in m.Q:
            for t in range(1, T):
                m.add_component(
                    f"C_no_recharge_after_idle_{q}_{t}",
                    pyo.Constraint(expr=m.c[q, t] <= m.yOUT[q, t - 1] + m.c[q, t - 1] + 1 - m.atL[q, t - 1]),
                )

        for q in m.Q:
            # Start at Longvilliers and end at Longvilliers (no ongoing trip at the end)
            m.atL[q, 0].fix(1)
            m.atM[q, 0].fix(0)
            m.atL[q, T - 1].fix(1)
            m.atM[q, T - 1].fix(0)
            m.inTrip[q, T - 1].fix(0)

        # Optional FIFO symmetry-breaking across vehicles (can restrict starts unintentionally)
        if bool(self._p("use_fifo_symmetry", False)) and Q >= 2:
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
                # Charging linkage (continuous): enforce gchg[q,t] = delta_chg * c[q,t],
                # but respect remaining capacity: gchg[q,t] <= Emax - b[q,t].
                # With c in [0,1], the model can throttle charging fractionally.
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
            # Allow charging label at the last slot if desired (battery won't change as gchg[T-1]=0)

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

    # Allow external code (CLI) to provide a warm-start schedule at the new resolution
    # Starts should be a dict with keys ("yOUT"|"yRET", q, t) and values in {0,1}
    def set_warm_start(self, starts: dict[tuple[str, int, int], float] | None) -> None:
        self._warm_start = dict(starts) if starts else None

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
                # Build slopes map over y-variable indices (β for RHS representation)
                slopes = {}
                for t, a in coeff_out_t.items():
                    if t in Yout_index:
                        slopes[Yout_index[t]] = float(a)
                for t, a in coeff_ret_t.items():
                    if t in Yret_index:
                        slopes[Yret_index[t]] = float(a)
                # Use shared filter (logs and deduplicates)
                if not add_benders_cut(iteration=0, const=float(const), slopes=slopes, lhs_value=lhs_val, rhs_value=rhs_val, cut_type="optimality"):
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
        # Report total theta for compatibility
        try:
            if hasattr(m, "theta"):
                cand["theta"] = pyo.value(m.theta)
            elif hasattr(m, "theta_s"):
                # Sum across scenarios for a single comparable number
                try:
                    cand["theta"] = sum(float(pyo.value(m.theta_s[s])) for s in getattr(m, "Scenarios", []))
                except Exception:
                    cand["theta"] = 0.0
            else:
                cand["theta"] = float(pyo.value(m.theta_out)) + float(pyo.value(m.theta_ret))
        except Exception:
            cand["theta"] = 0.0
        return cand

    def solve(self) -> SolveResult:
        assert self.m is not None, "Call initialize() before solve()"
        m = self.m
        solver = self._get_solver()
        tee_flag = bool(self._p("solver_tee", self._p("mp_solve_tee", False)))
        # Apply warm start if provided: set initial y values and aggregated Yout/Yret
        use_ws = False
        if self._warm_start:
            try:
                # Set binary starts only where provided; others left unset (partial MIP start)
                yout_sums: dict[int, float] = {}
                yret_sums: dict[int, float] = {}
                # Keep per-(q,t) copies for derived warm-start values
                _yout_qt: dict[tuple[int, int], float] = {}
                _yret_qt: dict[tuple[int, int], float] = {}
                for (typ, q, t), v in self._warm_start.items():
                    vv = float(v)
                    if typ == "yOUT" and (q in m.Q) and (t in m.T):
                        try:
                            m.yOUT[q, t].value = vv
                            yout_sums[t] = yout_sums.get(t, 0.0) + vv
                            _yout_qt[(int(q), int(t))] = vv
                        except Exception:
                            pass
                    elif typ == "yRET" and (q in m.Q) and (t in m.T):
                        try:
                            m.yRET[q, t].value = vv
                            yret_sums[t] = yret_sums.get(t, 0.0) + vv
                            _yret_qt[(int(q), int(t))] = vv
                        except Exception:
                            pass
                # Initialize aggregations for completeness (not required, but helpful for starts)
                for t in m.T:
                    try:
                        if hasattr(m, "Yout"):
                            m.Yout[t].value = float(yout_sums.get(int(t), 0.0))
                        if hasattr(m, "Yret"):
                            m.Yret[t].value = float(yret_sums.get(int(t), 0.0))
                        if hasattr(m, "Z"):
                            m.Z[t].value = float(yout_sums.get(int(t), 0.0) + yret_sums.get(int(t), 0.0))
                        # Excess variables if present
                        if hasattr(m, "eOut"):
                            m.eOut[t].value = max(0.0, float(yout_sums.get(int(t), 0.0)) - 1.0)
                        if hasattr(m, "eRet"):
                            m.eRet[t].value = max(0.0, float(yret_sums.get(int(t), 0.0)) - 1.0)
                    except Exception:
                        pass
                # Derive occupancy and inTrip consistent with y if available; this can help MIP start acceptance
                try:
                    import math as _math
                    slot_res = int(self._p("slot_resolution", 1))
                    trip_dur_min = self._p("trip_duration_minutes", self._p("trip_duration"))
                    if trip_dur_min is not None:
                        trip_slots = int(_math.ceil(float(trip_dur_min) / max(1, slot_res)))
                    else:
                        trip_slots = int(self._p("trip_slots", 0))
                except Exception:
                    trip_slots = 0
                if trip_slots > 0:
                    for q in m.Q:
                        # Initial occupancy
                        try:
                            m.atL[q, 0].value = 1.0
                            m.atM[q, 0].value = 0.0
                            m.inTrip[q, 0].value = 0.0
                        except Exception:
                            pass
                        # inTrip[q,t] per definition: sum of starts in previous (trip_slots-1) slots
                        for t in m.T:
                            t = int(t)
                            if t == 0:
                                continue
                            lo = max(0, t - trip_slots + 1)
                            hi = t - 1
                            s = 0.0
                            for u in range(lo, hi + 1):
                                s += float(_yout_qt.get((int(q), int(u)), 0.0))
                                s += float(_yret_qt.get((int(q), int(u)), 0.0))
                            try:
                                m.inTrip[q, t].value = s
                            except Exception:
                                pass
                        # Occupancy recursions
                        for t in m.T:
                            t = int(t)
                            if t == 0:
                                continue
                            arr_ret = float(_yret_qt.get((int(q), t - trip_slots), 0.0)) if (t - trip_slots) >= 0 else 0.0
                            arr_out = float(_yout_qt.get((int(q), t - trip_slots), 0.0)) if (t - trip_slots) >= 0 else 0.0
                            try:
                                prevL = float(m.atL[q, t - 1].value or 0.0)
                                prevM = float(m.atM[q, t - 1].value or 0.0)
                            except Exception:
                                prevL = prevM = 0.0
                            try:
                                m.atL[q, t].value = prevL - float(_yout_qt.get((int(q), t - 1), 0.0)) + arr_ret
                                m.atM[q, t].value = prevM - float(_yret_qt.get((int(q), t - 1), 0.0)) + arr_out
                            except Exception:
                                pass
                use_ws = True
            except Exception:
                use_ws = False
            finally:
                # Clear after applying to avoid reusing stale starts in later solves
                self._warm_start = None
        if getattr(self, "_is_persistent", False):
            # Persistent: do not pass model; allow warmstart
            # If we built a binary warm start, also push an explicit CPLEX MIP start
            if use_ws:
                try:
                    cpx = getattr(solver, "_solver_model", None)
                    if (cpx is not None) and (cplex is not None):
                        inds: list[int] = []
                        vals: list[float] = []
                        # yOUT / yRET indices by name
                        for (q, t), vv in list(_yout_qt.items()) + []:
                            try:
                                idx = cpx.variables.get_indices(f"yOUT[{int(q)},{int(t)}]")
                                inds.append(idx)
                                vals.append(float(vv))
                            except Exception:
                                pass
                        for (q, t), vv in list(_yret_qt.items()) + []:
                            try:
                                idx = cpx.variables.get_indices(f"yRET[{int(q)},{int(t)}]")
                                inds.append(idx)
                                vals.append(float(vv))
                            except Exception:
                                pass
                        if inds:
                            try:
                                spair = cplex.SparsePair(ind=inds, val=vals)
                                cpx.MIP_starts.add(spair, cpx.MIP_starts.effort_level.auto, "warmstart")
                            except Exception:
                                pass
                except Exception:
                    pass
            res = solver.solve(tee=tee_flag, warmstart=use_ws)
        else:
            res = solver.solve(m, tee=tee_flag, warmstart=use_ws)

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

        # Use full master objective as lower bound on total cost (first-stage + recourse proxy)
        objective = float(pyo.value(m.obj))
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
            if hasattr(m, "theta"):
                lines.append(f"theta = {pyo.value(m.theta):.6g}")
            elif hasattr(m, "theta_out") and hasattr(m, "theta_ret"):
                tot = float(pyo.value(m.theta_out)) + float(pyo.value(m.theta_ret))
                lines.append(f"theta = {tot:.6g}")
                lines.append(f"  - theta_out = {float(pyo.value(m.theta_out)):.6g}")
                lines.append(f"  - theta_ret = {float(pyo.value(m.theta_ret)):.6g}")
            elif hasattr(m, "theta_s"):
                try:
                    S = len(getattr(m, "Scenarios", []))
                except Exception:
                    S = 0
                vals = []
                for s in range(S):
                    try:
                        vals.append(float(pyo.value(m.theta_s[s])))
                    except Exception:
                        vals.append(0.0)
                try:
                    wts = list(self._p("scenario_weights", []) or [])
                    if not wts or len(wts) != S:
                        wts = [1.0 for _ in range(S)]
                except Exception:
                    wts = [1.0 for _ in range(S)]
                tot = sum(w * v for w, v in zip(wts, vals))
                lines.append(f"theta = {tot:.6g}")
                for s, v in enumerate(vals):
                    lines.append(f"  - theta_s[{s}] = {v:.6g}")
            else:
                lines.append("theta = (unavailable)")
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
        const_out_meta = float(cut.metadata.get("const_out", 0.0)) if hasattr(cut, "metadata") and ("const_out" in cut.metadata) else None
        const_ret_meta = float(cut.metadata.get("const_ret", 0.0)) if hasattr(cut, "metadata") and ("const_ret" in cut.metadata) else None
        coeff_yOUT = cut.metadata.get("coeff_yOUT") if hasattr(cut, "metadata") else None
        coeff_yRET = cut.metadata.get("coeff_yRET") if hasattr(cut, "metadata") else None
        # Optional: scenario index for per-scenario thetas
        try:
            scen_idx = int(cut.metadata.get("scenario_index")) if hasattr(cut, "metadata") and ("scenario_index" in cut.metadata) else None
        except Exception:
            scen_idx = None

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
        # RHS_raw(y) = const_dm + sum_t dm_out[t]*Yout[t] + sum_t dm_ret[t]*Yret[t]
        # This equals the SP objective at incumbent y. We'll compute ub_est for diagnostics
        # and then recompute the constant after any aggregation/thresholding so pass-through holds exactly.
        ub_est = float(const)
        # If we have per-direction constants, track those too
        ub_est_out = float(const_out_meta) if const_out_meta is not None else None
        ub_est_ret = float(const_ret_meta) if const_ret_meta is not None else None
        # Sum over raw dm maps using current y values
        if isinstance(coeff_yOUT, dict):
            for (q, t), v in coeff_yOUT.items():
                yv = float(m.yOUT[q, t].value or 0.0)
                ub_est += float(v) * yv
                if ub_est_out is not None:
                    ub_est_out += float(v) * yv
        if isinstance(coeff_yRET, dict):
            for (q, t), v in coeff_yRET.items():
                yv = float(m.yRET[q, t].value or 0.0)
                ub_est += float(v) * yv
                if ub_est_ret is not None:
                    ub_est_ret += float(v) * yv
        # Recompute constant after aggregation/thresholding to preserve pass-through at incumbent
        # Compute incumbent sums
        if aggregate:
            # Aggregate by time: use Yout[t], Yret[t]
            yout_val = {}
            yret_val = {}
            for t in set(list(agg_out.keys()) + list(agg_ret.keys())):  # type: ignore[arg-type]
                try:
                    yout_val[int(t)] = float(m.Yout[int(t)].value or 0.0)
                except Exception:
                    yout_val[int(t)] = 0.0
                try:
                    yret_val[int(t)] = float(m.Yret[int(t)].value or 0.0)
                except Exception:
                    yret_val[int(t)] = 0.0
            contrib = sum(float(v) * float(yout_val.get(int(t), 0.0)) for t, v in agg_out.items())
            contrib += sum(float(v) * float(yret_val.get(int(t), 0.0)) for t, v in agg_ret.items())
            contrib_out = sum(float(v) * float(yout_val.get(int(t), 0.0)) for t, v in agg_out.items())
            contrib_ret = sum(float(v) * float(yret_val.get(int(t), 0.0)) for t, v in agg_ret.items())
        else:
            # Per-(q,t) coefficients
            contrib = 0.0
            contrib_out = 0.0
            contrib_ret = 0.0
            for (q, t), v in agg_out.items():  # type: ignore[misc]
                try:
                    vv = float(m.yOUT[int(q), int(t)].value or 0.0)
                    contrib += float(v) * vv
                    contrib_out += float(v) * vv
                except Exception:
                    pass
            for (q, t), v in agg_ret.items():  # type: ignore[misc]
                try:
                    vv = float(m.yRET[int(q), int(t)].value or 0.0)
                    contrib += float(v) * vv
                    contrib_ret += float(v) * vv
                except Exception:
                    pass
        const_adj = float(ub_est) - float(contrib)
        const_adj_out = (float(ub_est_out) - float(contrib_out)) if ub_est_out is not None else None
        const_adj_ret = (float(ub_est_ret) - float(contrib_ret)) if ub_est_ret is not None else None

        # Assemble RHS with aggregated coefficients using adjusted constant
        rhs = const_adj
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

        # Temporarily disable scaling: enforce raw cut(s)
        # Support disaggregation by direction to strengthen the linearization
        scale = 1.0
        disagg_dir = hasattr(m, "theta_out") and hasattr(m, "theta_ret")
        added_any = True
        if disagg_dir and (const_adj_out is not None) and (const_adj_ret is not None):
            # Build separate RHS for OUT/RET
            if aggregate:
                rhs_out = float(const_adj_out) + sum(float(v) * m.Yout[int(t)] for t, v in agg_out.items())
                rhs_ret = float(const_adj_ret) + sum(float(v) * m.Yret[int(t)] for t, v in agg_ret.items())
            else:
                rhs_out = float(const_adj_out) + sum(float(v) * m.yOUT[int(q), int(t)] for (q, t), v in agg_out.items())  # type: ignore[misc]
                rhs_ret = float(const_adj_ret) + sum(float(v) * m.yRET[int(q), int(t)] for (q, t), v in agg_ret.items())  # type: ignore[misc]

            lhs_out = m.theta_out
            lhs_ret = m.theta_ret

            # Violation values and shared filter per direction
            lhs_out_val = float(pyo.value(lhs_out))
            rhs_out_val = float(pyo.value(rhs_out))
            lhs_ret_val = float(pyo.value(lhs_ret))
            rhs_ret_val = float(pyo.value(rhs_ret))
            if aggregate:
                slopes_out = {("Yout", int(t)): float(v) for t, v in agg_out.items()}
                slopes_ret = {("Yret", int(t)): float(v) for t, v in agg_ret.items()}
            else:
                slopes_out = {("yOUT", int(q), int(t)): float(v) for (q, t), v in agg_out.items()}  # type: ignore[misc]
                slopes_ret = {("yRET", int(q), int(t)): float(v) for (q, t), v in agg_ret.items()}  # type: ignore[misc]
            added_out = True
            added_ret = True
            if not force:
                added_out = add_benders_cut(
                    iteration=-1,
                    const=float(const_adj_out),
                    slopes=slopes_out,
                    lhs_value=lhs_out_val,
                    rhs_value=rhs_out_val,
                    cut_type="optimality:out",
                    signature_scope=("dir:out", scen_idx),
                )
                added_ret = add_benders_cut(
                    iteration=-1,
                    const=float(const_adj_ret),
                    slopes=slopes_ret,
                    lhs_value=lhs_ret_val,
                    rhs_value=rhs_ret_val,
                    cut_type="optimality:ret",
                    signature_scope=("dir:ret", scen_idx),
                )
                if not (added_out or added_ret):
                    return False
        else:
            # Choose theta variable: per-scenario if available, else single theta
            if hasattr(m, "theta_s") and (scen_idx is not None):
                lhs = m.theta_s[int(scen_idx)]
            else:
                lhs = m.theta
            rhs_scaled = rhs
            lhs_val = float(pyo.value(lhs))
            rhs_val = float(pyo.value(rhs_scaled))
            if aggregate:
                slopes_all = {("Yout", int(t)): float(v) for t, v in agg_out.items()}
                slopes_all.update({("Yret", int(t)): float(v) for t, v in agg_ret.items()})
            else:
                slopes_all = {("yOUT", int(q), int(t)): float(v) for (q, t), v in agg_out.items()}  # type: ignore[misc]
                slopes_all.update({("yRET", int(q), int(t)): float(v) for (q, t), v in agg_ret.items()})  # type: ignore[misc]
            if not force:
                ok = add_benders_cut(
                    iteration=-1,
                    const=float(const_adj),
                    slopes=slopes_all,
                    lhs_value=lhs_val,
                    rhs_value=rhs_val,
                    cut_type=str(cut.cut_type).lower() if hasattr(cut, "cut_type") else "optimality",
                    signature_scope=("scen", scen_idx),
                )
                if not ok:
                    return False

        # Duplicate check handled by add_benders_cut

        # Create explicit constraint(s) and register with persistent solver if used
        if hasattr(m, "theta_out") and hasattr(m, "theta_ret") and (const_adj_out is not None) and (const_adj_ret is not None):
            con_list = []
            name_list = []
            # OUT direction
            if force or 'added_out' in locals() and added_out:
                cname_out = f"benders_cut_out_{self._cut_idx}"
                con_out = pyo.Constraint(expr=(m.theta_out >= rhs_out))
                setattr(m.BendersCuts, cname_out, con_out)
                if getattr(self, "_is_persistent", False):
                    self._solver.add_constraint(con_out)
                con_list.append(con_out)
                name_list.append(cname_out)
            # RET direction
            if force or 'added_ret' in locals() and added_ret:
                cname_ret = f"benders_cut_ret_{self._cut_idx}"
                con_ret = pyo.Constraint(expr=(m.theta_ret >= rhs_ret))
                setattr(m.BendersCuts, cname_ret, con_ret)
                if getattr(self, "_is_persistent", False):
                    self._solver.add_constraint(con_ret)
                con_list.append(con_ret)
                name_list.append(cname_ret)
        else:
            cname = f"benders_cut_{self._cut_idx}"
            if hasattr(m, "theta_s") and (scen_idx is not None):
                lhs = m.theta_s[int(scen_idx)]
            else:
                lhs = m.theta
            rhs_scaled = rhs
            con = pyo.Constraint(expr=(lhs >= rhs_scaled))
            setattr(m.BendersCuts, cname, con)
            if getattr(self, "_is_persistent", False):
                self._solver.add_constraint(con)
            con_list = [con]
            name_list = [cname]
        # Maintain pool for optional pruning
        if not hasattr(self, "_cut_cons"):
            self._cut_cons = []
            self._cut_names = []
        for con_i, name_i in zip(con_list, name_list):
            self._cut_cons.append(con_i)
            self._cut_names.append(name_i)

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

        # Optional: export MP LP after adding a cut for inspection
        try:
            if bool(self._p("write_lp_after_cut", False)):
                out_dir = Path(self._p("lp_output_dir", "Report"))
                out_dir.mkdir(parents=True, exist_ok=True)
                lp_path = out_dir / f"master_after_cut_{self._cut_idx}.lp"
                # Prefer underlying CPLEX model if persistent, else use Pyomo writer
                wrote = False
                if getattr(self, "_is_persistent", False):
                    try:
                        cpx = getattr(self._solver, "_solver_model", None)
                        if cpx is not None:
                            cpx.write(str(lp_path))
                            wrote = True
                    except Exception:
                        wrote = False
                if not wrote:
                    try:
                        # Fall back to Pyomo's model writer
                        m.write(str(lp_path), io_options={"symbolic_solver_labels": True})
                        wrote = True
                    except Exception:
                        wrote = False
                if wrote:
                    print(f"[BENDERS] Wrote LP to {lp_path}")
                # Also write a symbolic LP via Pyomo for readability
                try:
                    sym_lp_path = out_dir / f"master_after_cut_{self._cut_idx}_sym.lp"
                    m.write(str(sym_lp_path), io_options={"symbolic_solver_labels": True})
                    print(f"[BENDERS] Wrote symbolic LP to {sym_lp_path}")
                except Exception:
                    pass
        except Exception:
            pass

        # Simple logging: constant and nonzeros
        nnz = (len(agg_out) if aggregate else len(agg_out)) + (len(agg_ret) if aggregate else len(agg_ret))
        # Log slope range
        all_betas = list(agg_out.values()) + list(agg_ret.values())
        if all_betas:
            rng = (min(all_betas), max(all_betas))
            if hasattr(m, "theta_out") and hasattr(m, "theta_ret") and (const_adj_out is not None) and (const_adj_ret is not None):
                print(
                    f"[BENDERS] Added cut #{self._cut_idx}: const_out={const_adj_out:.6g}, const_ret={const_adj_ret:.6g}, nnz={nnz}, slope_range=[{rng[0]:.3g},{rng[1]:.3g}], raw_pos_dm={raw_pos_dm}, scale={scale:.3g}"
                )
            else:
                print(
                    f"[BENDERS] Added cut #{self._cut_idx}: const={const_adj:.6g}, nnz={nnz}, slope_range=[{rng[0]:.3g},{rng[1]:.3g}], raw_pos_dm={raw_pos_dm}, scale={scale:.3g}"
                )
        else:
            if hasattr(m, "theta_out") and hasattr(m, "theta_ret") and (const_adj_out is not None) and (const_adj_ret is not None):
                print(
                    f"[BENDERS] Added cut #{self._cut_idx}: const_out={const_adj_out:.6g}, const_ret={const_adj_ret:.6g}, nnz={nnz}, raw_pos_dm={raw_pos_dm}, scale={scale:.3g}"
                )
            else:
                print(f"[BENDERS] Added cut #{self._cut_idx}: const={const_adj:.6g}, nnz={nnz}, raw_pos_dm={raw_pos_dm}, scale={scale:.3g}")
        # Sanity log with LHS and RHS values (scaled)
        try:
            if hasattr(m, "theta_out") and hasattr(m, "theta_ret") and (const_adj_out is not None) and (const_adj_ret is not None):
                print(
                    f"[BENDERS] Eval cut (dir): OUT lhs={lhs_out_val:.6g} rhs={rhs_out_val:.6g}; "
                    f"RET lhs={lhs_ret_val:.6g} rhs={rhs_ret_val:.6g}"
                )
            else:
                print(f"[BENDERS] Eval cut: lhs={lhs_val:.6g} rhs={rhs_val:.6g}")
        except Exception:
            pass
        self._cut_idx += 1
        # Track last cut info for cross-iteration checks
        self._last_cut_const = (const_adj_out + const_adj_ret) if (const_adj_out is not None and const_adj_ret is not None) else const_adj
        self._last_cut_nnz = nnz
        return True

    # Evaluate deterministic first-stage cost f(y) for a given candidate
    # Mirrors the master objective components excluding theta: start epsilon and concurrency penalty
    def first_stage_cost(self, candidate: Candidate) -> float:
        eps_start = float(self._p("start_cost_epsilon", 0.0) or 0.0)
        conc_pen = float(self._p("concurrency_penalty", 0.0) or 0.0)
        if eps_start == 0.0 and conc_pen == 0.0:
            return 0.0

        # Collect sums per time from candidate
        # Prefer model T if available; otherwise infer from candidate indices
        try:
            T = int(len(self.m.T)) if self.m is not None else None  # type: ignore[arg-type]
        except Exception:
            T = None
        if T is None:
            # Infer T from candidate keys
            tmax = -1
            for name in candidate.keys():
                if isinstance(name, str) and (name.startswith("yOUT[") or name.startswith("yRET[")):
                    inside = name[name.find("[") + 1 : name.find("]")]
                    try:
                        _, t_str = inside.split(",")
                        tmax = max(tmax, int(t_str.strip()))
                    except Exception:
                        pass
            T = tmax + 1 if tmax >= 0 else 0

        Yout = [0.0 for _ in range(T)]
        Yret = [0.0 for _ in range(T)]
        starts = 0.0
        for name, val in candidate.items():
            if not isinstance(name, str):
                continue
            try:
                if name.startswith("yOUT["):
                    inside = name[name.find("[") + 1 : name.find("]")]
                    _, tau_str = inside.split(",")
                    tau = int(tau_str.strip())
                    vv = float(val)
                    if 0 <= tau < T:
                        Yout[tau] += vv
                        starts += vv
                elif name.startswith("yRET["):
                    inside = name[name.find("[") + 1 : name.find("]")]
                    _, tau_str = inside.split(",")
                    tau = int(tau_str.strip())
                    vv = float(val)
                    if 0 <= tau < T:
                        Yret[tau] += vv
                        starts += vv
            except Exception:
                continue

        cost = 0.0
        if eps_start > 0.0:
            cost += eps_start * float(starts)
        if conc_pen > 0.0:
            cost += conc_pen * sum(max(0.0, y - 1.0) for y in Yout)
            cost += conc_pen * sum(max(0.0, y - 1.0) for y in Yret)
        return float(cost)

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
