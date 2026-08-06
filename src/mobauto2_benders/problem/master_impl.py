from __future__ import annotations

from typing import Any, Optional
from datetime import datetime
from pathlib import Path

import pyomo.environ as pyo

from ..benders.master import MasterProblem
from ..benders.solver import add_benders_cut  # shared cut filtering
from ..benders.types import Candidate, Cut, SolveResult, SolveStatus
from ..benders.cplex_log import parse_cplex_log_bounds
from ..tolerances import project_binary_value


class ProblemMaster(MasterProblem):
    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        self.m: pyo.ConcreteModel | None = None
        self._cut_idx = 0
        self._lb: Optional[float] = None
        # Fingerprints of cuts to avoid duplicates
        self._cut_fps: set[tuple] = set()
        # Per-run cut signatures to avoid duplicates (reset in initialize)
        self._cut_signatures: set[tuple] = set()
        # Per-run dominance map: slope signature -> best (largest) const
        self._cut_best_const: dict[tuple, float] = {}
        self._last_log_path: Optional[str] = None
        # Optional: warm start values for yOUT/yRET at next solve
        # Keys: ("yOUT"|"yRET", q:int, t:int) -> float(0/1)
        self._warm_start: dict[tuple[str, int, int], float] | None = None
        # Optional: MIP start values from previous iteration
        self._last_solution: dict[str, float] | None = None
        # Best-incumbent snapshot for final reporting
        self._report_candidate: Candidate | None = None
        self._report_realized_departure_min_map: dict[tuple[int, int], float] | None = None
        self._last_cut_attempt: dict[str, Any] | None = None

    def _p(self, key: str, default: Any | None = None) -> Any:
        if self.params is None:
            return default
        return self.params.get(key, default)

    def _is_report(self) -> bool:
        return str(self._p("log_level", "")).upper() == "REPORT"

    def _vprint(self, *args, **kwargs) -> None:
        if not self._is_report():
            print(*args, **kwargs)

    def _extract_solver_stats(self, res) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        try:
            stats["termination_condition"] = getattr(res.solver, "termination_condition", None)
            stats["status"] = getattr(res.solver, "status", None)
            stats["solver_time"] = getattr(res.solver, "time", None)
            stats["wall_time"] = getattr(res.solver, "wall_time", None)
            stats["iterations"] = getattr(res.solver, "iterations", None)
            stats["nodes"] = getattr(res.solver, "nodes", None)
            stats["gap"] = getattr(res.solver, "gap", None)
            stats["best_bound"] = getattr(res.solver, "best_bound", None)
            stats["incumbent"] = getattr(res.solver, "objective", None)
        except Exception:
            pass
        return stats

    def _parse_cplex_log_bounds(self, log_path: str | None) -> dict[str, Optional[float] | str]:
        return parse_cplex_log_bounds(log_path)

    def _extract_cplex_api_stats(self, solver) -> dict[str, Any]:
        """Extract MIP stats directly from CPLEX Python API when available."""
        stats: dict[str, Any] = {}
        try:
            cpx = getattr(solver, "_solver_model", None)
        except Exception:
            cpx = None
        if cpx is None:
            return stats
        try:
            if cpx.solution.is_primal_feasible():
                stats["incumbent"] = cpx.solution.get_objective_value()
        except Exception:
            pass
        try:
            if hasattr(cpx.solution, "MIP"):
                stats["best_bound"] = cpx.solution.MIP.get_best_objective()
                stats["gap"] = cpx.solution.MIP.get_mip_relative_gap()
        except Exception:
            pass
        try:
            if hasattr(cpx.solution, "progress"):
                stats["nodes"] = cpx.solution.progress.get_num_nodes_processed()
        except Exception:
            pass
        return stats

    def initialize(self) -> None:
        # Reset per-run cut signatures
        self._cut_signatures = set()
        self._cut_best_const = {}
        self._report_candidate = None
        self._report_realized_departure_min_map = None
        self._last_cut_attempt = None
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
        _binit_raw = self._p("initial_battery", self._p("binit"))
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
        # Normalize forced first actions to length Q
        _initial_actions_raw = self._p("initial_actions")
        if _initial_actions_raw is None:
            initial_actions = ["IDL"] * Q
        elif isinstance(_initial_actions_raw, str):
            initial_actions = [str(_initial_actions_raw).strip().upper()] * Q
        else:
            try:
                initial_actions = [str(x).strip().upper() for x in list(_initial_actions_raw)]  # type: ignore[arg-type]
            except Exception:
                initial_actions = ["IDL"] * Q
            if len(initial_actions) < Q:
                initial_actions = initial_actions + ["IDL"] * (Q - len(initial_actions))
            elif len(initial_actions) > Q:
                initial_actions = initial_actions[:Q]
        allowed_initial_actions = {"IDL", "CHR", "OUT", "RET"}
        invalid_initial_actions = sorted({action for action in initial_actions if action not in allowed_initial_actions})
        if invalid_initial_actions:
            raise ValueError(
                "initial_actions contains invalid value(s): "
                + ", ".join(invalid_initial_actions)
                + ". Allowed values are IDL, CHR, OUT, RET."
            )

        # Vehicle location encoding: 0 = Longvilliers (depot), 1 = Massy
        # Initial location is implied by the forced first action; all shuttles must end at Longvilliers.

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
            first_action = initial_actions[q]
            starts_at_massy = first_action == "RET"
            # The first forced action defines the shuttle's location at t=0.
            m.atL[q, 0].fix(0 if starts_at_massy else 1)
            m.atM[q, 0].fix(1 if starts_at_massy else 0)
            m.atL[q, T - 1].fix(1)
            m.atM[q, T - 1].fix(0)
            m.inTrip[q, T - 1].fix(0)
            # Slot 0 is the demand accumulation bucket and is never a departure slot.
            # Preserve the legacy initial location shorthand but keep starts fixed off at t=0.
            m.yOUT[q, 0].fix(0)
            m.yRET[q, 0].fix(0)
            if first_action == "CHR":
                m.c[q, 0].fix(1)
            else:
                m.c[q, 0].fix(0)

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

        # Symmetry breaking: order vehicles by cumulative departures (OUT+RET)
        if bool(self._p("symmetry_breaking", False)) and Q >= 2:
            for k in range(Q - 1):
                # Cumulative ordering by time prefix
                for t in range(T):
                    m.add_component(
                        f"C_sym_break_pref_{k}_{t}",
                        pyo.Constraint(
                            expr=
                            sum(m.yOUT[k, tau] + m.yRET[k, tau] for tau in range(0, t + 1))
                            >= sum(m.yOUT[k + 1, tau] + m.yRET[k + 1, tau] for tau in range(0, t + 1))
                        ),
                    )
                # Redundant total-ordering reinforcement
                m.add_component(
                    f"C_sym_break_tot_{k}",
                    pyo.Constraint(
                        expr=
                        sum(m.yOUT[k, t] + m.yRET[k, t] for t in m.T)
                        >= sum(m.yOUT[k + 1, t] + m.yRET[k + 1, t] for t in m.T)
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

        # Create and retain a standard (non-persistent) solver
        backend = str(self._p("solver_backend", "")).strip()
        if backend:
            solver_name = backend
        else:
            solver_name = str(self._p("solver", "cplex"))
        self._solver = pyo.SolverFactory(solver_name)
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

    def solve(self) -> SolveResult:
        assert self.m is not None, "Call initialize() before solve()"
        m = self.m
        solver = self._get_solver()
        tee_flag = bool(self._p("solver_tee", self._p("mp_solve_tee", False)))
        emit_reports = bool(self._p("emit_reports", True))
        # Per-iteration solver controls
        try:
            time_limit = self._p("solve_time_limit_s")
            if time_limit is not None:
                solver.options["timelimit"] = float(time_limit)
            mipgap = self._p("mipgap")
            if mipgap is not None:
                solver.options["mipgap"] = float(mipgap)
            cplex_opts = self._p("cplex_options", {}) or {}
            backend = str(self._p("solver_backend", "")).lower()
            # CPLEX file-based plugin accepts CPXPARAM_* keys; cplex_direct does not.
            if backend in ("cplex", ""):
                for k, v in cplex_opts.items():
                    solver.options[k] = v
            else:
                # Skip CPXPARAM_* for cplex_direct to avoid AttributeError.
                for k, v in cplex_opts.items():
                    if str(k).upper().startswith("CPXPARAM_"):
                        continue
                    solver.options[k] = v
        except Exception:
            pass
        # Apply warm start if provided: set initial y values only (x-only start)
        use_ws = False
        if self._warm_start:
            try:
                # Set binary starts only where provided; others left unset (partial MIP start)
                eps_bin = float(self._p("eps_bin", 1e-6))
                n_out = 0
                n_ret = 0
                # Clear existing y values to avoid stale carry-over for partial starts
                try:
                    for q in m.Q:
                        for t in m.T:
                            m.yOUT[q, t].value = None
                            m.yRET[q, t].value = None
                except Exception:
                    pass
                for (typ, q, t), v in self._warm_start.items():
                    try:
                        vv = float(v)
                    except Exception:
                        continue
                    pv = project_binary_value(vv, eps_bin)
                    if pv is None:
                        continue
                    if typ == "yOUT" and (q in m.Q) and (t in m.T):
                        try:
                            m.yOUT[q, t].value = pv
                            n_out += 1
                        except Exception:
                            pass
                    elif typ == "yRET" and (q in m.Q) and (t in m.T):
                        try:
                            m.yRET[q, t].value = pv
                            n_ret += 1
                        except Exception:
                            pass
                if n_out + n_ret > 0:
                    self._vprint(f"[MIPSTART] x-only start applied: n_out={n_out} n_ret={n_ret}")
                    use_ws = True
                else:
                    self._vprint("[MIPSTART] no x vars found; skipping warm start")
            except Exception:
                use_ws = False
            finally:
                # Clear after applying to avoid reusing stale starts in later solves
                self._warm_start = None
        # Use previous master solution as MIP start if enabled and no explicit warm start
        if (not use_ws) and bool(self._p("use_mip_start", False)) and self._last_solution:
            eps_bin = float(self._p("eps_bin", 1e-6))
            try:
                n_out = 0
                n_ret = 0
                for name, val in self._last_solution.items():
                    try:
                        vv = float(val)
                    except Exception:
                        continue
                    pv = project_binary_value(vv, eps_bin)
                    if pv is None:
                        continue
                    if name.startswith("yOUT["):
                        inside = name[name.find("[") + 1 : name.find("]")]
                        q_str, t_str = inside.split(",")
                        q = int(q_str.strip())
                        t = int(t_str.strip())
                        if (q in m.Q) and (t in m.T):
                            m.yOUT[q, t].value = pv
                            n_out += 1
                    elif name.startswith("yRET["):
                        inside = name[name.find("[") + 1 : name.find("]")]
                        q_str, t_str = inside.split(",")
                        q = int(q_str.strip())
                        t = int(t_str.strip())
                        if (q in m.Q) and (t in m.T):
                            m.yRET[q, t].value = pv
                            n_ret += 1
                if n_out + n_ret > 0:
                    self._vprint(f"[MIPSTART] x-only start applied: n_out={n_out} n_ret={n_ret}")
                    use_ws = True
                else:
                    self._vprint("[MIPSTART] no x vars found; skipping warm start")
            except Exception:
                use_ws = False
        # Diagnostics: write LP and enable solver logs
        log_path = None
        if emit_reports:
            try:
                out_dir = Path(self._p("lp_output_dir", "Report"))
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                try:
                    it = int(self._p("iteration", -1))
                except Exception:
                    it = -1
                cuts = int(self._cut_idx)
                if it >= 0:
                    lp_path = out_dir / f"master_iter_{it:03d}_cuts_{cuts:03d}_{ts}.lp"
                else:
                    lp_path = out_dir / f"master_iter_cuts_{cuts:03d}_{ts}.lp"
                m.write(str(lp_path), io_options={"symbolic_solver_labels": True})
                self._vprint(f"[MP] Wrote LP: {lp_path}")
            except Exception:
                lp_path = None

            try:
                out_dir = Path(self._p("lp_output_dir", "Report"))
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                try:
                    it = int(self._p("iteration", -1))
                except Exception:
                    it = -1
                cuts = int(self._cut_idx)
                if it >= 0:
                    log_path = out_dir / f"master_iter_{it:03d}_cuts_{cuts:03d}_{ts}.log"
                else:
                    log_path = out_dir / f"master_iter_cuts_{cuts:03d}_{ts}.log"
                backend = str(self._p("solver_backend", "")).lower()
                if backend in ("cplex", ""):
                    solver.options["logfile"] = str(log_path)
                else:
                    # cplex_direct: set internal log file attribute if available
                    try:
                        setattr(solver, "_log_file", str(log_path))
                    except Exception:
                        pass
            except Exception:
                log_path = None
        self._last_log_path = str(log_path) if log_path is not None else None

        res = solver.solve(
            m,
            tee=bool(tee_flag and emit_reports),
            warmstart=use_ws,
            load_solutions=False,
            keepfiles=bool(emit_reports),
            symbolic_solver_labels=True,
        )
        term = getattr(res.solver, "termination_condition", None)
        if term in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible, pyo.TerminationCondition.maxTimeLimit):
            try:
                m.solutions.load_from(res)
            except Exception:
                # Retry once with solution loading enabled
                res = solver.solve(
                    m,
                    tee=bool(tee_flag and emit_reports),
                    warmstart=use_ws,
                    load_solutions=True,
                    keepfiles=bool(emit_reports),
                    symbolic_solver_labels=True,
                )
        # Try to capture the actual solver log path (including temp log from direct interfaces)
        try:
            for attr in ("logfile", "log_file", "_log_file", "_logfile", "log_path", "_log_path", "log", "logfile_name", "log_filename"):
                val = getattr(res.solver, attr, None)
                if isinstance(val, str) and val:
                    self._last_log_path = val
                    break
        except Exception:
            pass
        term = getattr(res.solver, "termination_condition", None)
        # Fallback: if file-based cplex returns UNKNOWN, retry with cplex_direct
        try:
            backend = str(self._p("solver_backend", "")).lower()
        except Exception:
            backend = ""
        if term not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible, pyo.TerminationCondition.maxTimeLimit) and backend in ("cplex", ""):
            try:
                fallback = pyo.SolverFactory("cplex_direct")
                # Apply options compatible with cplex_direct
                try:
                    cplex_opts = self._p("cplex_options", {}) or {}
                    for k, v in cplex_opts.items():
                        if str(k).upper().startswith("CPXPARAM_"):
                            continue
                        fallback.options[k] = v
                except Exception:
                    pass
                res = fallback.solve(
                    m,
                    tee=True,
                    warmstart=use_ws,
                    load_solutions=True,
                    keepfiles=True,
                    symbolic_solver_labels=True,
                )
                term = getattr(res.solver, "termination_condition", None)
                self._vprint("[MP] Fallback to cplex_direct due to UNKNOWN termination.")
            except Exception:
                pass
        try:
            m.solutions.load_from(res)
        except Exception:
            pass
        st = getattr(res.solver, "status", None)

        stats = self._extract_solver_stats(res)
        sources: dict[str, str] = {}
        if stats.get("incumbent") is not None:
            sources["incumbent"] = "solver_results"
        if stats.get("best_bound") is not None:
            sources["best_bound"] = "solver_results"
        if stats.get("gap") is not None:
            sources["gap"] = "solver_results"

        api_stats = self._extract_cplex_api_stats(solver)
        for k in ("incumbent", "best_bound", "gap", "nodes"):
            if stats.get(k) is None and api_stats.get(k) is not None:
                stats[k] = api_stats[k]
                sources[k] = "cplex_api"

        parsed = self._parse_cplex_log_bounds(self._last_log_path)
        parsed_best_int = parsed.get("best_integer")
        parsed_best_bound = parsed.get("best_bound")
        parsed_gap = parsed.get("gap")
        parsed_source = parsed.get("source")

        if stats.get("incumbent") is None and parsed_best_int is not None:
            stats["incumbent"] = parsed_best_int
            sources["incumbent"] = f"cplex_log:{parsed_source}" if parsed_source else "cplex_log"
        if stats.get("best_bound") is None and parsed_best_bound is not None:
            stats["best_bound"] = parsed_best_bound
            sources["best_bound"] = f"cplex_log:{parsed_source}" if parsed_source else "cplex_log"
        if stats.get("gap") is None and parsed_gap is not None:
            stats["gap"] = parsed_gap
            sources["gap"] = f"cplex_log:{parsed_source}" if parsed_source else "cplex_log"

        if stats.get("gap") is None and stats.get("incumbent") is not None and stats.get("best_bound") is not None:
            try:
                inc = float(stats["incumbent"])
                bb = float(stats["best_bound"])
                gap_val = (inc - bb) / max(1.0, abs(inc))
                if gap_val < 0.0:
                    gap_val = 0.0
                stats["gap"] = gap_val
                sources["gap"] = "computed"
            except Exception:
                pass

        if stats.get("best_integer") is None:
            if stats.get("incumbent") is not None:
                stats["best_integer"] = stats.get("incumbent")
            elif parsed_best_int is not None:
                stats["best_integer"] = parsed_best_int

        stats["incumbent_source"] = sources.get("incumbent")
        stats["best_bound_source"] = sources.get("best_bound")
        stats["gap_source"] = sources.get("gap")

        if stats.get("best_bound") is None:
            if self._last_log_path is None:
                stats["best_bound_reason"] = "no_log_path"
            else:
                stats["best_bound_reason"] = "not_in_solver_or_log"

        self._vprint(
            "MP raw solver: status=%s term=%s incumbent=%s best_bound=%s"
            % (
                str(st),
                str(term),
                str(stats.get("incumbent")),
                str(stats.get("best_bound")),
            )
        )
        if parsed_best_int is not None or parsed_best_bound is not None:
            self._vprint(
                "MP parsed bounds: best_integer=%s best_bound=%s"
                % (
                    (f"{float(parsed_best_int):.6g}" if parsed_best_int is not None else "-"),
                    (f"{float(parsed_best_bound):.6g}" if parsed_best_bound is not None else "-"),
                )
            )
        try:
            sol_len = len(getattr(res, "solution", []))
            sol_keys = list(getattr(res, "solution", {}).keys()) if hasattr(res, "solution") else []
            self._vprint(f"MP solutions: count={sol_len} keys={sol_keys}")
        except Exception:
            pass
        if log_path is not None:
            self._vprint(f"[MP] Solver log: {log_path}")
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

        if status == SolveStatus.OPTIMAL:
            try:
                gap_val = stats.get("gap")
                if gap_val is not None and float(gap_val) > 1e-6:
                    status = SolveStatus.FEASIBLE
            except Exception:
                pass

        # Use full master objective as lower bound on total cost (first-stage + recourse proxy)
        val_obj = pyo.value(m.obj, exception=False)
        if val_obj is None:
            objective = None
            self._lb = None
        else:
            objective = float(val_obj)
            if stats.get("best_bound") is not None:
                try:
                    self._lb = float(stats.get("best_bound"))
                except Exception:
                    self._lb = objective
            else:
                self._lb = objective

        # Assert representative variable has a value
        try:
            rep_val = pyo.value(m.yOUT[0, 0], exception=False)
            if rep_val is None:
                raise RuntimeError(
                    "Master solve did not load variable values. Check solver log/LP for details."
                )
        except Exception as exc:
            raise RuntimeError(
                "Master solve did not load variable values. Check solver log/LP for details."
            ) from exc
        # If we loaded variable values, treat as having an incumbent
        if status in (SolveStatus.FEASIBLE, SolveStatus.OPTIMAL):
            pass
        else:
            # Keep UNKNOWN/INFEASIBLE as-is
            pass

        # Sanity: if we have any cuts, at least one y should be nonzero
        try:
            if self._cut_idx > 0:
                nonzero = False
                for q in m.Q:
                    for t in m.T:
                        if float(m.yOUT[q, t].value or 0.0) > 1e-6 or float(m.yRET[q, t].value or 0.0) > 1e-6:
                            nonzero = True
                            break
                    if nonzero:
                        break
                if not nonzero:
                    self._vprint("[WARN] Master has cuts but all y are zero.")
        except Exception:
            pass
        self._last_solve_stats = stats
        # Print per-shuttle total departures to confirm symmetry constraints are active
        try:
            totals = []
            for q in m.Q:
                tot_q = sum(float(m.yOUT[q, t].value or 0.0) + float(m.yRET[q, t].value or 0.0) for t in m.T)
                totals.append(tot_q)
            self._vprint("Shuttle totals (OUT+RET): " + ", ".join(f"q{idx}={tot:.0f}" for idx, tot in enumerate(totals)))
        except Exception:
            pass
        # Store solution values for next MIP start (binary vars)
        try:
            sol: dict[str, float] = {}
            for q in m.Q:
                for t in m.T:
                    sol[f"yOUT[{int(q)},{int(t)}]"] = float(m.yOUT[q, t].value or 0.0)
                    sol[f"yRET[{int(q)},{int(t)}]"] = float(m.yRET[q, t].value or 0.0)
                    sol[f"atL[{int(q)},{int(t)}]"] = float(m.atL[q, t].value or 0.0)
                    sol[f"atM[{int(q)},{int(t)}]"] = float(m.atM[q, t].value or 0.0)
                    sol[f"inTrip[{int(q)},{int(t)}]"] = float(m.inTrip[q, t].value or 0.0)
            self._last_solution = sol
        except Exception:
            self._last_solution = None
        try:
            stats = dict(self._last_solve_stats or {})
            nodes = stats.get("nodes")
            best_bound = stats.get("best_bound")
            incumbent = stats.get("incumbent")
            gap = stats.get("gap")
            if nodes is not None or best_bound is not None or incumbent is not None or gap is not None:
                self._vprint(
                    "MP stats (solver): nodes=%s best_bound=%s incumbent=%s gap=%s"
                    % (str(nodes), str(best_bound), str(incumbent), str(gap))
                )
        except Exception:
            pass
        candidate = self._collect_candidate()
        return SolveResult(status=status, objective=objective, candidate=candidate, lower_bound=self._lb)

    def last_solve_stats(self) -> dict[str, Any]:
        return dict(getattr(self, "_last_solve_stats", {}) or {})

    def last_solver_log_path(self) -> Optional[str]:
        return self._last_log_path

    def _collect_candidate(self) -> Candidate:
        assert self.m is not None
        m = self.m
        cand: Candidate = {}
        eps_bin = float(self._p("eps_bin", 1e-6))
        offenders: list[tuple[str, float]] = []
        for q in m.Q:
            for t in m.T:
                try:
                    v_out = float(m.yOUT[q, t].value or 0.0)
                except Exception:
                    v_out = 0.0
                try:
                    v_ret = float(m.yRET[q, t].value or 0.0)
                except Exception:
                    v_ret = 0.0
                pv_out = project_binary_value(v_out, eps_bin)
                pv_ret = project_binary_value(v_ret, eps_bin)
                if pv_out is None:
                    offenders.append((f"yOUT[{int(q)},{int(t)}]", v_out))
                else:
                    cand[f"yOUT[{int(q)},{int(t)}]"] = pv_out
                if pv_ret is None:
                    offenders.append((f"yRET[{int(q)},{int(t)}]", v_ret))
                else:
                    cand[f"yRET[{int(q)},{int(t)}]"] = pv_ret
                try:
                    cand[f"c[{int(q)},{int(t)}]"] = float(m.c[q, t].value or 0.0)
                except Exception:
                    pass
        if offenders:
            offenders.sort(key=lambda kv: abs(kv[1] - 0.5), reverse=True)
            top = offenders[:5]
            raise RuntimeError(
                "Non-binary master solution; refusing SP evaluation. Offenders: "
                + ", ".join(f"{k}={v:.6g}" for k, v in top)
            )
        # Include theta values (if available) for early-exit checks in subproblem
        try:
            if hasattr(m, "theta"):
                val = pyo.value(m.theta, exception=False)
                if val is not None:
                    cand["__theta"] = float(val)
        except Exception:
            pass
        # Directional thetas, when disaggregation is active. Without these the
        # candidate carries no theta at all under theta_out/theta_ret, and the end
        # of run report falls back to the LIVE model values -- i.e. the last master
        # solve rather than the incumbent being reported.
        try:
            if hasattr(m, "theta_out") and hasattr(m, "theta_ret"):
                v_out = pyo.value(m.theta_out, exception=False)
                v_ret = pyo.value(m.theta_ret, exception=False)
                if v_out is not None:
                    cand["__theta_out"] = float(v_out)
                if v_ret is not None:
                    cand["__theta_ret"] = float(v_ret)
        except Exception:
            pass
        try:
            if hasattr(m, "theta_s"):
                try:
                    S = len(getattr(m, "Scenarios", []))
                except Exception:
                    S = 0
                for s in range(int(S)):
                    try:
                        v = pyo.value(m.theta_s[s], exception=False)
                    except Exception:
                        v = None
                    if v is not None:
                        cand[f"__theta_s[{int(s)}]"] = float(v)
        except Exception:
            pass
        return cand

    def set_report_solution(
        self,
        candidate: Candidate | None,
        realized_departure_min_map: dict[tuple[int, int], float] | None = None,
    ) -> None:
        self._report_candidate = dict(candidate) if isinstance(candidate, dict) else None
        self._report_realized_departure_min_map = (
            dict(realized_departure_min_map) if isinstance(realized_departure_min_map, dict) else None
        )

    # Pretty-print the current master solution (if solved)
    def format_solution(self) -> str:
        assert self.m is not None
        m = self.m
        Q = list(m.Q)
        T = list(m.T)
        report_candidate = self._report_candidate if isinstance(self._report_candidate, dict) else None
        realized_map = (
            self._report_realized_departure_min_map
            if isinstance(self._report_realized_departure_min_map, dict)
            else None
        )
        lines: list[str] = []
        lines.append(f"Q={len(Q)} T={len(T)}")

        def _candidate_value(name: str, default: float = 0.0) -> float:
            if report_candidate is None:
                return default
            try:
                return float(report_candidate.get(name, default))
            except Exception:
                return default

        def _trip_slots() -> int:
            import math

            slot_res = int(self._p("slot_resolution", 1))
            trip_dur_min = self._p("trip_duration_minutes", self._p("trip_duration"))
            if trip_dur_min is not None:
                return int(math.ceil(float(trip_dur_min) / max(1, slot_res)))
            return int(self._p("trip_slots", 0))

        def _is_service_start(q: int, t: int) -> bool:
            if int(t) < 1:
                yout0 = _candidate_value(f"yOUT[{int(q)},{int(t)}]") if report_candidate is not None else float(m.yOUT[q, t].value or 0.0)
                yret0 = _candidate_value(f"yRET[{int(q)},{int(t)}]") if report_candidate is not None else float(m.yRET[q, t].value or 0.0)
                if yout0 >= 0.5 or yret0 >= 0.5:
                    raise AssertionError(
                        f"Invalid service start slot q={int(q)} t={int(t)}; slot 0 is not a departure slot"
                    )
                return False
            if report_candidate is not None:
                return (
                    _candidate_value(f"yOUT[{int(q)},{int(t)}]") >= 0.5
                    or _candidate_value(f"yRET[{int(q)},{int(t)}]") >= 0.5
                )
            try:
                return (
                    float(m.yOUT[q, t].value or 0.0) >= 0.5
                    or float(m.yRET[q, t].value or 0.0) >= 0.5
                )
            except Exception:
                return False

        def _is_charge(q: int, t: int) -> bool:
            if report_candidate is not None:
                return _candidate_value(f"c[{int(q)},{int(t)}]") >= 0.5
            try:
                return float(m.c[q, t].value or 0.0) >= 0.5
            except Exception:
                return False

        def _is_in_trip(q: int, t: int) -> bool:
            if report_candidate is None:
                try:
                    return float(m.inTrip[q, t].value or 0.0) >= 0.5
                except Exception:
                    return False
            trip_slots = max(1, _trip_slots())
            tau_min = max(int(T[0]), int(t) - trip_slots + 1)
            for tau in range(tau_min, int(t)):
                if _is_service_start(int(q), int(tau)):
                    return True
            return False

        def _format_hhmm(min_from_day_start: float | int | None) -> str:
            if min_from_day_start is None:
                return "--"
            total = int(round(float(min_from_day_start)))
            hh = total // 60
            mm = total % 60
            return f"{hh:02d}:{mm:02d}"

        # Theta
        try:
            if report_candidate is not None and "__theta" in report_candidate:
                lines.append(f"theta = {_candidate_value('__theta'):.6g}")
            elif report_candidate is not None and "__theta_out" in report_candidate and "__theta_ret" in report_candidate:
                t_out = _candidate_value("__theta_out")
                t_ret = _candidate_value("__theta_ret")
                lines.append(f"theta = {t_out + t_ret:.6g}")
                lines.append(f"  - theta_out = {t_out:.6g}")
                lines.append(f"  - theta_ret = {t_ret:.6g}")
            elif hasattr(m, "theta"):
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
            if report_candidate is not None:
                yout = _candidate_value(f"yOUT[{int(q)},{int(t)}]")
                yret = _candidate_value(f"yRET[{int(q)},{int(t)}]")
                intr = 1.0 if _is_in_trip(int(q), int(t)) else 0.0
                chg = _candidate_value(f"c[{int(q)},{int(t)}]")
            else:
                yout = m.yOUT[q, t].value or 0.0
                yret = m.yRET[q, t].value or 0.0
                intr = m.inTrip[q, t].value or 0.0
                chg = m.c[q, t].value or 0.0
            if float(yout) >= 0.5:
                return "OUT"
            if float(yret) >= 0.5:
                return "RET"
            if float(intr) >= 0.5:
                return "INT"
            if float(chg) >= 0.5:
                return "CHR"
            return "IDL"

        lines.append("Timeline (per shuttle):")
        colw = 5
        header = "       " + " ".join(f"{t:>{colw}d}" for t in T)
        lines.append(header)
        for q in Q:
            seq = " ".join(f"{lbl(q, t):>{colw}s}" for t in T)
            lines.append(f"  q={q}: {seq}")
            if realized_map:
                dep_seq = []
                for t in T:
                    if _is_service_start(int(q), int(t)):
                        dep_seq.append(_format_hhmm(realized_map.get((int(q), int(t)))))
                    else:
                        dep_seq.append("--")
                lines.append("  dep: " + " ".join(f"{v:>{colw}s}" for v in dep_seq))

        # Also show battery levels over time
        def rowf(var, q):
            vals = []
            for t in T:
                v = var[q, t].value
                if v is None:
                    vals.append(f"{'-':>{colw}s}")
                else:
                    vals.append(f"{float(v):>{colw}.0f}")
            return " ".join(vals)

        lines.append("Battery (per shuttle):")
        lines.append(header)
        for q in Q:
            lines.append(f"  q={q}: {rowf(m.b, q)}")

        return "\n".join(lines)

    def _add_cut(self, cut: Cut, force: bool = False) -> bool:
        assert self.m is not None
        m = self.m
        anti_trivial_min_total = None
        try:
            anti_trivial_min_total = int(cut.metadata.get("anti_trivial_min_total_starts")) if hasattr(cut, "metadata") and ("anti_trivial_min_total_starts" in cut.metadata) else None
        except Exception:
            anti_trivial_min_total = None
        if anti_trivial_min_total is not None:
            lhs = sum(m.yOUT[q, t] for q in m.Q for t in m.T) + sum(m.yRET[q, t] for q in m.Q for t in m.T)
            lhs_val = float(pyo.value(lhs, exception=False) or 0.0)
            rhs_val = float(anti_trivial_min_total)
            eps_cut = float(self._p("eps_cut", 1e-8))
            if lhs_val >= rhs_val - eps_cut:
                return False
            sig = ("anti_trivial_idle", int(anti_trivial_min_total))
            if (not force) and sig in self._cut_signatures:
                return False
            cname = f"benders_cut_anti_idle_{self._cut_idx}"
            con = pyo.Constraint(expr=(lhs >= rhs_val))
            setattr(m.BendersCuts, cname, con)
            self._cut_signatures.add(sig)
            self._cut_idx += 1
            self._vprint(f"[BENDERS] Added anti-trivial fallback cut #{self._cut_idx - 1}: total_starts>={anti_trivial_min_total}")
            return True
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

        iteration = int(self._p("iteration", -1) or -1)

        def _active_start_id(limit: int = 12) -> str:
            starts: list[str] = []
            try:
                for q in m.Q:
                    for t in m.T:
                        yout = float(m.yOUT[q, t].value or 0.0)
                        yret = float(m.yRET[q, t].value or 0.0)
                        if yout >= 0.5:
                            starts.append(f"OUT[{int(q)},{int(t)}]")
                        if yret >= 0.5:
                            starts.append(f"RET[{int(q)},{int(t)}]")
            except Exception:
                return "candidate:unavailable"
            if len(starts) > limit:
                return "|".join(starts[:limit]) + f"|...(+{len(starts) - limit})"
            return "|".join(starts) if starts else "candidate:idle"

        def _theta_snapshot() -> tuple[float | None, float | None, float | None]:
            try:
                theta_out_val = float(pyo.value(m.theta_out, exception=False)) if hasattr(m, "theta_out") else None
            except Exception:
                theta_out_val = None
            try:
                theta_ret_val = float(pyo.value(m.theta_ret, exception=False)) if hasattr(m, "theta_ret") else None
            except Exception:
                theta_ret_val = None
            try:
                if hasattr(m, "theta"):
                    theta_total_val = float(pyo.value(m.theta, exception=False))
                else:
                    theta_total_val = (float(theta_out_val or 0.0) + float(theta_ret_val or 0.0))
            except Exception:
                theta_total_val = None
            return theta_out_val, theta_ret_val, theta_total_val

        def _raw_cut_value(constant: float | None, coeffs: Any, prefix: str) -> float:
            total = float(constant or 0.0)
            if isinstance(coeffs, dict):
                for (q, t), v in coeffs.items():
                    try:
                        if prefix == "yOUT":
                            total += float(v) * float(m.yOUT[int(q), int(t)].value or 0.0)
                        else:
                            total += float(v) * float(m.yRET[int(q), int(t)].value or 0.0)
                    except Exception:
                        continue
            return total

        theta_out_cur, theta_ret_cur, theta_total_cur = _theta_snapshot()
        raw_rhs_out_total = _raw_cut_value(const_out_meta, coeff_yOUT, "yOUT") if const_out_meta is not None else None
        raw_rhs_ret_total = _raw_cut_value(const_ret_meta, coeff_yRET, "yRET") if const_ret_meta is not None else None
        raw_rhs_total = _raw_cut_value(const, coeff_yOUT, "yOUT")
        raw_rhs_total += sum(
            float(v) * float(m.yRET[int(q), int(t)].value or 0.0)
            for (q, t), v in (coeff_yRET.items() if isinstance(coeff_yRET, dict) else [])
        )
        attempt_diag: dict[str, Any] = {
            "iteration": iteration,
            "candidate_id": _active_start_id(),
            "cut_name": getattr(cut, "name", None),
            "cut_type": str(getattr(cut, "cut_type", "")),
            "theta_out": theta_out_cur,
            "theta_ret": theta_ret_cur,
            "theta_total": theta_total_cur,
            "recourse_out": (float(cut.metadata.get("recourse_out")) if hasattr(cut, "metadata") and ("recourse_out" in cut.metadata) else None),
            "recourse_ret": (float(cut.metadata.get("recourse_ret")) if hasattr(cut, "metadata") and ("recourse_ret" in cut.metadata) else None),
            "recourse_total": (float(cut.metadata.get("recourse_total")) if hasattr(cut, "metadata") and ("recourse_total" in cut.metadata) else None),
            "raw_rhs_out": raw_rhs_out_total,
            "raw_rhs_ret": raw_rhs_ret_total,
            "raw_rhs_total": raw_rhs_total,
            "aggregate_cut_logging_gap": (
                float(raw_rhs_total) - float(cut.metadata.get("recourse_total"))
                if hasattr(cut, "metadata") and ("recourse_total" in cut.metadata)
                else None
            ),
            "violation_tolerance_master": float(self._p("eps_cut", 1e-8)),
            "violation_tolerance_filter_rel": None,
            "skip_reason": None,
        }
        self._last_cut_attempt = attempt_diag

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

        def _assert_q_invariant(coeffs: dict, direction: str) -> None:
            """Collapsing (q,tau) -> tau is only valid if the coefficients agree across q.

            Aggregation replaces sum_q coeff[q,tau]*y[q,tau] with coeff[tau]*Y[tau],
            where Y[tau] == sum_q y[q,tau]. That substitution is an identity only when
            coeff is the same for every q. On the production path it is, by
            construction: the subproblem's duals are per-slot (summed over layers,
            never indexed by vehicle) and are broadcast to every q. This check exists
            so that a future per-vehicle formulation fails loudly here instead of
            silently emitting an invalid cut (D10 / spec §2.7).
            """
            if not isinstance(coeffs, dict):
                return
            first: dict[int, tuple[int, float]] = {}
            for (q, t), v in coeffs.items():
                t_i = int(t)
                val = float(v)
                if t_i not in first:
                    first[t_i] = (int(q), val)
                    continue
                q0, v0 = first[t_i]
                # Scale-relative tolerance: these are cut slopes, order 1e2-1e3.
                if abs(val - v0) > 1e-9 * max(1.0, abs(v0)):
                    raise RuntimeError(
                        f"Cut coefficient varies across q at fixed tau; aggregation "
                        f"invalid. direction={direction} tau={t_i} "
                        f"coeff[q={q0}]={v0!r} coeff[q={int(q)}]={val!r}. "
                        f"Check the cut-generation mode: aggregate_cuts_by_tau "
                        f"requires per-slot duals broadcast identically to every q."
                    )

        if aggregate:
            _assert_q_invariant(coeff_yOUT, "OUT")
            _assert_q_invariant(coeff_yRET, "RET")
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

        # Check that the re-anchoring only does what it is entitled to do.
        #
        # const_adj - const has exactly two legitimate sources:
        #   1. q-aggregation, which is an identity when the coefficients are
        #      q-invariant (enforced above) and so contributes nothing; and
        #   2. cut_coeff_threshold dropping small coefficients from the aggregated
        #      maps while ub_est still counts them -- a deliberate sparsification.
        # Anything beyond the dropped mass means the aggregated cut does not
        # reproduce the per-(q,tau) cut at the incumbent, which is a real defect.
        #
        # This used to be applied unconditionally as a silent correction, which would
        # paper over precisely the inconsistency the q-invariance check exists to
        # catch. Raise instead (spec §2.7).
        dropped = 0.0
        for _coeffs, _var in ((coeff_yOUT, "yOUT"), (coeff_yRET, "yRET")):
            if not isinstance(_coeffs, dict):
                continue
            for (q, t), v in _coeffs.items():
                if abs(float(v)) > coeff_tol:
                    continue
                try:
                    yv = float((m.yOUT if _var == "yOUT" else m.yRET)[int(q), int(t)].value or 0.0)
                except Exception:
                    yv = 0.0
                dropped += float(v) * yv
        _residual = abs((float(const_adj) - float(const)) - dropped)
        if _residual > max(1e-6, 1e-9 * max(1.0, abs(float(const)))):
            raise RuntimeError(
                f"Cut constant re-anchoring is not accounted for: const={float(const)!r} "
                f"reconstructed={float(const_adj)!r} "
                f"delta={float(const_adj) - float(const)!r} "
                f"explained_by_threshold={dropped!r} residual={_residual!r}. "
                f"The aggregated cut does not reproduce the per-(q,tau) cut at the "
                f"incumbent beyond what cut_coeff_threshold={coeff_tol!r} explains. "
                f"Do not mask this by re-anchoring -- find the cause."
            )

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
        attempt_diag["scale"] = float(scale)
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
            lhs_out_val = pyo.value(lhs_out, exception=False)
            rhs_out_val = pyo.value(rhs_out, exception=False)
            lhs_ret_val = pyo.value(lhs_ret, exception=False)
            rhs_ret_val = pyo.value(rhs_ret, exception=False)
            if lhs_out_val is None or rhs_out_val is None or lhs_ret_val is None or rhs_ret_val is None:
                return False
            lhs_out_val = float(lhs_out_val)
            rhs_out_val = float(rhs_out_val)
            lhs_ret_val = float(lhs_ret_val)
            rhs_ret_val = float(rhs_ret_val)
            attempt_diag.update({
                "transformed_rhs_out_before_scaling": float(rhs_out_val),
                "transformed_rhs_ret_before_scaling": float(rhs_ret_val),
                "transformed_rhs_total_before_scaling": float(rhs_out_val + rhs_ret_val),
                "transformed_rhs_out_after_scaling": float(rhs_out_val),
                "transformed_rhs_ret_after_scaling": float(rhs_ret_val),
                "transformed_rhs_total_after_scaling": float(rhs_out_val + rhs_ret_val),
                "lhs_out": float(lhs_out_val),
                "lhs_ret": float(lhs_ret_val),
                "lhs_total": float(lhs_out_val + lhs_ret_val),
                "const_out_adjusted": float(const_adj_out),
                "const_ret_adjusted": float(const_adj_ret),
            })
            # Tightness + violation checks
            eps_cut = float(self._p("eps_cut", 1e-8))
            attempt_diag["violation_tolerance_master"] = float(eps_cut)
            if ub_est_out is not None and abs(rhs_out_val - float(ub_est_out)) > eps_cut * max(1.0, abs(float(ub_est_out))):
                self._vprint("[CUT DEBUG] Tightness failed (OUT). ub=%.6g rhs=%.6g" % (float(ub_est_out), rhs_out_val))
                attempt_diag["skip_reason"] = "tightness_failed_out"
                return False
            if ub_est_ret is not None and abs(rhs_ret_val - float(ub_est_ret)) > eps_cut * max(1.0, abs(float(ub_est_ret))):
                self._vprint("[CUT DEBUG] Tightness failed (RET). ub=%.6g rhs=%.6g" % (float(ub_est_ret), rhs_ret_val))
                attempt_diag["skip_reason"] = "tightness_failed_ret"
                return False
            viol_out = rhs_out_val - lhs_out_val
            viol_ret = rhs_ret_val - lhs_ret_val
            attempt_diag["violation_out"] = float(viol_out)
            attempt_diag["violation_ret"] = float(viol_ret)
            if (viol_out <= eps_cut * max(1.0, abs(rhs_out_val))) and (viol_ret <= eps_cut * max(1.0, abs(rhs_ret_val))):
                self._vprint("[CUT DEBUG] Not violated (dir). viol_out=%.6g viol_ret=%.6g" % (viol_out, viol_ret))
                attempt_diag["skip_reason"] = "not_violated_directional"
                if attempt_diag.get("aggregate_cut_logging_gap") is not None and float(attempt_diag["aggregate_cut_logging_gap"]) < -float(eps_cut):
                    attempt_diag["mismatch_explanation"] = (
                        "Raw cut line at y is below recorded SP recourse, but skip logic compares transformed directional RHS "
                        "against current theta_out/theta_ret. A negative raw recourse gap does not imply the cut is violated in theta-space."
                    )
                return False
            if aggregate:
                slopes_out = {("Yout", int(t)): float(v) for t, v in agg_out.items()}
                slopes_ret = {("Yret", int(t)): float(v) for t, v in agg_ret.items()}
            else:
                slopes_out = {("yOUT", int(q), int(t)): float(v) for (q, t), v in agg_out.items()}  # type: ignore[misc]
                slopes_ret = {("yRET", int(q), int(t)): float(v) for (q, t), v in agg_ret.items()}  # type: ignore[misc]
            attempt_diag["slopes_out"] = dict(slopes_out)
            attempt_diag["slopes_ret"] = dict(slopes_ret)
            added_out = True
            added_ret = True
            if not force:
                filter_out_diag: dict[str, Any] = {}
                added_out = add_benders_cut(
                    iteration=iteration,
                    const=float(const_adj_out),
                    slopes=slopes_out,
                    lhs_value=lhs_out_val,
                    rhs_value=rhs_out_val,
                    cut_type="optimality:out",
                    signature_scope=("dir:out", scen_idx),
                    cuts_in_model=self._cut_idx,
                    signature_set=self._cut_signatures,
                    slope_const_map=self._cut_best_const,
                    diagnostics=filter_out_diag,
                )
                filter_ret_diag: dict[str, Any] = {}
                added_ret = add_benders_cut(
                    iteration=iteration,
                    const=float(const_adj_ret),
                    slopes=slopes_ret,
                    lhs_value=lhs_ret_val,
                    rhs_value=rhs_ret_val,
                    cut_type="optimality:ret",
                    signature_scope=("dir:ret", scen_idx),
                    cuts_in_model=self._cut_idx,
                    signature_set=self._cut_signatures,
                    slope_const_map=self._cut_best_const,
                    diagnostics=filter_ret_diag,
                )
                attempt_diag["filter_out"] = filter_out_diag
                attempt_diag["filter_ret"] = filter_ret_diag
                if filter_out_diag.get("violation_threshold") is not None:
                    attempt_diag["violation_tolerance_filter_rel"] = float(filter_out_diag["violation_threshold"])
                if not (added_out or added_ret):
                    attempt_diag["skip_reason"] = "filtered_out"
                    return False
        else:
            # Choose theta variable: per-scenario if available, else single theta
            if hasattr(m, "theta_s") and (scen_idx is not None):
                lhs = m.theta_s[int(scen_idx)]
            else:
                lhs = m.theta
            rhs_scaled = rhs
            lhs_val = pyo.value(lhs, exception=False)
            rhs_val = pyo.value(rhs_scaled, exception=False)
            if lhs_val is None or rhs_val is None:
                return False
            lhs_val = float(lhs_val)
            rhs_val = float(rhs_val)
            attempt_diag.update({
                "transformed_rhs_before_scaling": float(rhs_val),
                "transformed_rhs_after_scaling": float(rhs_val),
                "lhs_total": float(lhs_val),
                "const_adjusted": float(const_adj),
            })
            # Tightness + violation checks
            eps_cut = float(self._p("eps_cut", 1e-8))
            attempt_diag["violation_tolerance_master"] = float(eps_cut)
            if abs(rhs_val - float(ub_est)) > eps_cut * max(1.0, abs(float(ub_est))):
                self._vprint("[CUT DEBUG] Tightness failed. ub=%.6g rhs=%.6g" % (float(ub_est), rhs_val))
                attempt_diag["skip_reason"] = "tightness_failed"
                return False
            viol = rhs_val - lhs_val
            attempt_diag["violation_total"] = float(viol)
            if viol <= eps_cut * max(1.0, abs(rhs_val)):
                self._vprint("[CUT DEBUG] Not violated. viol=%.6g" % (viol,))
                attempt_diag["skip_reason"] = "not_violated"
                if attempt_diag.get("aggregate_cut_logging_gap") is not None and float(attempt_diag["aggregate_cut_logging_gap"]) < -float(eps_cut):
                    attempt_diag["mismatch_explanation"] = (
                        "Raw cut line at y is below recorded SP recourse, but skip logic compares transformed RHS against theta. "
                        "The candidate can still be nonviolated if theta already dominates the proposed cut."
                    )
                return False
            if aggregate:
                slopes_all = {("Yout", int(t)): float(v) for t, v in agg_out.items()}
                slopes_all.update({("Yret", int(t)): float(v) for t, v in agg_ret.items()})
            else:
                slopes_all = {("yOUT", int(q), int(t)): float(v) for (q, t), v in agg_out.items()}  # type: ignore[misc]
                slopes_all.update({("yRET", int(q), int(t)): float(v) for (q, t), v in agg_ret.items()})  # type: ignore[misc]
            if lhs_val is None or rhs_val is None:
                return False
            if not force:
                filter_diag: dict[str, Any] = {}
                ok = add_benders_cut(
                    iteration=iteration,
                    const=float(const_adj),
                    slopes=slopes_all,
                    lhs_value=lhs_val,
                    rhs_value=rhs_val,
                    cut_type=str(cut.cut_type).lower() if hasattr(cut, "cut_type") else "optimality",
                    signature_scope=("scen", scen_idx),
                    cuts_in_model=self._cut_idx,
                    signature_set=self._cut_signatures,
                    slope_const_map=self._cut_best_const,
                    diagnostics=filter_diag,
                )
                attempt_diag["filter_total"] = filter_diag
                if filter_diag.get("violation_threshold") is not None:
                    attempt_diag["violation_tolerance_filter_rel"] = float(filter_diag["violation_threshold"])
                if not ok:
                    attempt_diag["skip_reason"] = str(filter_diag.get("skip_reason", "filtered_out"))
                    return False

        # Duplicate check handled by add_benders_cut

        # Create explicit constraint(s)
        if hasattr(m, "theta_out") and hasattr(m, "theta_ret") and (const_adj_out is not None) and (const_adj_ret is not None):
            con_list = []
            name_list = []
            # OUT direction
            if force or 'added_out' in locals() and added_out:
                cname_out = f"benders_cut_out_{self._cut_idx}"
                con_out = pyo.Constraint(expr=(m.theta_out >= rhs_out))
                setattr(m.BendersCuts, cname_out, con_out)
                con_list.append(con_out)
                name_list.append(cname_out)
            # RET direction
            if force or 'added_ret' in locals() and added_ret:
                cname_ret = f"benders_cut_ret_{self._cut_idx}"
                con_ret = pyo.Constraint(expr=(m.theta_ret >= rhs_ret))
                setattr(m.BendersCuts, cname_ret, con_ret)
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
                # Remove from Pyomo block
                try:
                    delattr(m.BendersCuts, name_old)
                except Exception:
                    pass

        # Optional: export MP LP after adding a cut for inspection
        try:
            if bool(self._p("write_lp_after_cut", False)) and (not self._is_report()):
                out_dir = Path(self._p("lp_output_dir", "Report"))
                out_dir.mkdir(parents=True, exist_ok=True)
                lp_path = out_dir / f"master_after_cut_{self._cut_idx}.lp"
                wrote = False
                try:
                    # Fall back to Pyomo's model writer
                    m.write(str(lp_path), io_options={"symbolic_solver_labels": True})
                    wrote = True
                except Exception:
                    wrote = False
                if wrote:
                    self._vprint(f"[BENDERS] Wrote LP to {lp_path}")
                # Also write a symbolic LP via Pyomo for readability
                try:
                    sym_lp_path = out_dir / f"master_after_cut_{self._cut_idx}_sym.lp"
                    m.write(str(sym_lp_path), io_options={"symbolic_solver_labels": True})
                    self._vprint(f"[BENDERS] Wrote symbolic LP to {sym_lp_path}")
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
                self._vprint(
                    f"[BENDERS] Added cut #{self._cut_idx}: const_out={const_adj_out:.6g}, const_ret={const_adj_ret:.6g}, nnz={nnz}, slope_range=[{rng[0]:.3g},{rng[1]:.3g}], raw_pos_dm={raw_pos_dm}, scale={scale:.3g}"
                )
            else:
                self._vprint(
                    f"[BENDERS] Added cut #{self._cut_idx}: const={const_adj:.6g}, nnz={nnz}, slope_range=[{rng[0]:.3g},{rng[1]:.3g}], raw_pos_dm={raw_pos_dm}, scale={scale:.3g}"
                )
        else:
            if hasattr(m, "theta_out") and hasattr(m, "theta_ret") and (const_adj_out is not None) and (const_adj_ret is not None):
                self._vprint(
                    f"[BENDERS] Added cut #{self._cut_idx}: const_out={const_adj_out:.6g}, const_ret={const_adj_ret:.6g}, nnz={nnz}, raw_pos_dm={raw_pos_dm}, scale={scale:.3g}"
                )
            else:
                self._vprint(f"[BENDERS] Added cut #{self._cut_idx}: const={const_adj:.6g}, nnz={nnz}, raw_pos_dm={raw_pos_dm}, scale={scale:.3g}")
        # Sanity log with LHS and RHS values (scaled)
        try:
            if hasattr(m, "theta_out") and hasattr(m, "theta_ret") and (const_adj_out is not None) and (const_adj_ret is not None):
                self._vprint(
                    f"[BENDERS] Eval cut (dir): OUT lhs={lhs_out_val:.6g} rhs={rhs_out_val:.6g}; "
                    f"RET lhs={lhs_ret_val:.6g} rhs={rhs_ret_val:.6g}"
                )
            else:
                self._vprint(f"[BENDERS] Eval cut: lhs={lhs_val:.6g} rhs={rhs_val:.6g}")
        except Exception:
            pass
        self._cut_idx += 1
        attempt_diag["decision"] = "added"
        # Track last cut info for cross-iteration checks
        self._last_cut_const = (const_adj_out + const_adj_ret) if (const_adj_out is not None and const_adj_ret is not None) else const_adj
        self._last_cut_nnz = nnz
        try:
            ct = cut.cut_type
            self._last_cut_type = ct.value if hasattr(ct, "value") else str(ct)
        except Exception:
            self._last_cut_type = None
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

    def last_cut_meta(self) -> tuple[str | None, int | None]:
        return getattr(self, "_last_cut_type", None), getattr(self, "_last_cut_nnz", None)

    def last_cut_attempt(self) -> dict[str, Any] | None:
        return dict(self._last_cut_attempt) if isinstance(self._last_cut_attempt, dict) else None

