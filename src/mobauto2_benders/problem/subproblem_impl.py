from __future__ import annotations

from typing import Any, Dict, Tuple, Iterable, Optional, Mapping
from bisect import bisect_right
import time
from pathlib import Path
import json
import math
try:
    import yaml as _yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _yaml = None

import pyomo.environ as pyo
from dataclasses import dataclass

from ..benders.subproblem import Subproblem
from ..benders.types import Candidate, Cut, CutType, SubproblemResult


class ProblemSubproblem(Subproblem):
    """LP assignment/waiting subproblem generating optimality cuts.

    Builds a time-expanded assignment LP for both directions (OUT, RET) with
    waiting costs and unmet-demand penalties. Extracts dual multipliers to
    form the Benders optimality cut for a minimization master:

        theta >= const - S * sum_{q,tau} [pi_OUT[tau]*yOUT[q,tau] + pi_RET[tau]*yRET[q,tau]]

    where const = sum_t alpha_OUT[t]*R_out[t] + sum_t alpha_RET[t]*R_ret[t], and
    pi_* are duals on capacity constraints (>= 0). Increasing capacity (y) reduces
    the subproblem cost, hence the negative sign.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)

    def _is_report(self) -> bool:
        return str((self.params or {}).get("log_level", "")).upper() == "REPORT"

    def _vprint(self, *args, **kwargs) -> None:
        if not self._is_report():
            print(*args, **kwargs)

    def _parse_candidate_indices(self, candidate: Candidate) -> Tuple[set[int], set[int]]:
        qs: set[int] = set()
        ts: set[int] = set()
        for name in candidate.keys():
            if isinstance(name, str) and (name.startswith("yOUT[") or name.startswith("yRET[")):
                inside = name[name.find("[") + 1 : name.find("]")]
                q_str, t_str = inside.split(",")
                q = int(q_str.strip())
                t = int(t_str.strip())
                qs.add(q)
                ts.add(t)
        return qs, ts

    def evaluate(self, candidate: Candidate) -> SubproblemResult:
        params = self.params or {}
        debug_early_exit = bool(params.get("debug_early_exit", False))
        debug_timing = bool(params.get("debug_timing", False))
        debug_skip_cut_generation = bool(params.get("debug_skip_cut_generation", False))
        # Parameters (with safe defaults)
        S = float(params.get("S", 1.0))
        # Resolution in minutes per slot (copied from master params via config or set here)
        slot_res = int(params.get("slot_resolution", params.get("resolution", 1)))
        time_step_min = int(params.get("time_step_minutes", 1) or 1)
        # D9: the elastic/minute-level relaxation is deprecated from the default path.
        # Beyond selecting the subproblem model, this flag also gates mw_enabled,
        # use_dual_slopes and cut_lb_valid below -- with it True, cuts carry no valid
        # lower bound and the solver discards best_lb entirely.
        temporal_refinement = bool(params.get("enable_temporal_refinement", False))
        refined_cut_mode = str(
            params.get(
                "refined_cut_generation_mode",
                "refined_lp_relaxation" if temporal_refinement else "native",
            )
        ).strip().lower()
        # Allow Wmax to be specified in minutes
        if "Wmax_minutes" in params:
            Wmax = int(math.ceil(float(params.get("Wmax_minutes", 0)) / max(1, slot_res)))
        else:
            Wmax = int(params.get("Wmax_slots", params.get("Wmax", 0)))
        p_pen = float(params.get("p", 0.0))
        _eps_raw = params.get("eps_cut", None)
        eps_cut = float(_eps_raw) if _eps_raw is not None else 1e-6
        lp_solver = str(params.get("lp_solver", "cplex_direct"))
        # Optional: solver-specific options (e.g., CPLEX: {"lpmethod": 2, "threads": 0})
        solver_options = dict(params.get("solver_options", {}) or {})
        # Prefer packing demand into the first vehicle layer, then the next (LP tie-breaker)
        fill_eps = float(params.get("fill_first_epsilon", 1e-6) or 0.0)

        # Determine T and Q from candidate if not configured
        q_idx, t_idx = self._parse_candidate_indices(candidate)
        master_q_idx = sorted(q_idx) if q_idx else list(range(int(params.get("Q", 0) or 0)))
        T_cand = (max(t_idx) + 1) if t_idx else int(params.get("T", 0))
        T = int(params.get("T", T_cand))

        # Helpers to read demand from files or inline and aggregate into R vectors
        def _load_doc(path: Path) -> Any:
            if not path.exists():
                raise FileNotFoundError(f"Demand file not found: {path}")
            ext = path.suffix.lower()
            if ext == ".json":
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            if ext in {".yaml", ".yml"}:
                if _yaml is None:
                    raise RuntimeError("PyYAML is required to read YAML demand files. Install with 'pip install pyyaml'.")
                with path.open("r", encoding="utf-8") as f:
                    return _yaml.safe_load(f)
            # Fallback: try JSON
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)

        def _aggregate_requests(container: Any, Tlen: int) -> tuple[list[float], list[float]]:
            R_out = [0.0 for _ in range(Tlen)]
            R_ret = [0.0 for _ in range(Tlen)]
            if container is None:
                return R_out, R_ret
            # Direct arrays
            if isinstance(container, dict) and ("R_out" in container or "R_ret" in container):
                rout = list(container.get("R_out", [0.0] * Tlen))
                rret = list(container.get("R_ret", [0.0] * Tlen))
                if len(rout) != Tlen:
                    rout = (rout + [0.0] * Tlen)[:Tlen]
                if len(rret) != Tlen:
                    rret = (rret + [0.0] * Tlen)[:Tlen]
                return [float(x) for x in rout], [float(x) for x in rret]
            # Pull list from mapping under 'requests' or 'req_matrix'
            if isinstance(container, dict):
                container = container.get("requests") or container.get("req_matrix") or []
            # List of dicts [{dir,time}, ...]
            if isinstance(container, list) and container and isinstance(container[0], dict):
                import math as _math
                def _slot_idx_from_minutes(tmin: float) -> int:
                    # Map continuous minutes to slot index via floor: [0,res)->0, [res,2res)->1, ...
                    res = max(1, slot_res)
                    return max(0, int(_math.floor(float(tmin) / res)))
                for r in container:
                    d = r.get("dir")
                    try:
                        tmin = float(r.get("time", -1))
                    except Exception:
                        continue
                    if tmin < 0:
                        continue
                    # Floor-based slot mapping
                    t = _slot_idx_from_minutes(tmin)
                    if not (0 <= t < Tlen):
                        continue
                    if isinstance(d, str):
                        dd = d.upper()
                        if dd == "OUT":
                            R_out[t] += 1.0
                        elif dd == "RET":
                            R_ret[t] += 1.0
                    else:
                        if int(d) == 0:
                            R_out[t] += 1.0
                        else:
                            R_ret[t] += 1.0
                return R_out, R_ret
            # Matrix [[dir,time], ...]
            if isinstance(container, list):
                import math as _math
                def _slot_idx_from_minutes(tmin: float) -> int:
                    res = max(1, slot_res)
                    return max(0, int(_math.floor(float(tmin) / res)))
                for row in container:
                    if not isinstance(row, (list, tuple)) or len(row) < 2:
                        continue
                    d, tt = row[0], row[1]
                    try:
                        tmin = float(tt)
                    except Exception:
                        continue
                    if tmin < 0:
                        continue
                    t = _slot_idx_from_minutes(tmin)
                    if not (0 <= t < Tlen):
                        continue
                    if isinstance(d, str):
                        dd = d.upper()
                        if dd == "OUT":
                            R_out[t] += 1.0
                        elif dd == "RET":
                            R_ret[t] += 1.0
                    else:
                        if int(d) == 0:
                            R_out[t] += 1.0
                        else:
                            R_ret[t] += 1.0
                return R_out, R_ret
            return R_out, R_ret

        def _ok(lhs: float | None, rhs: float | None, eps: float) -> bool:
            if lhs is None or rhs is None:
                return False
            return float(lhs) >= float(rhs) - float(eps) * (1.0 + abs(float(rhs)))

        def _cand_theta(key: str) -> float | None:
            try:
                v = candidate.get(key)
            except Exception:
                v = None
            if v is None:
                return None

        def _dbg(msg: str) -> None:
            if debug_timing:
                self._vprint(msg)

        def _candidate_is_all_idle() -> bool:
            return all(float(v) <= 1e-9 for v in C_out) and all(float(v) <= 1e-9 for v in C_ret)
            try:
                return float(v)
            except Exception:
                return None

        def _load_demand_from_file(path_like: Any, Tlen: int) -> tuple[list[float], list[float]]:
            p = Path(str(path_like))
            doc = _load_doc(p)
            return _aggregate_requests(doc, Tlen)

        def _load_exact_arrivals_from_file(path_like: Any) -> tuple[list[float] | None, list[float] | None]:
            p = Path(str(path_like))
            doc = _load_doc(p)
            return _extract_exact_arrival_minutes(doc)

        # (legacy) aggregate_requests_to_R removed; using _aggregate_requests instead

        # Capacities induced by master decisions: C_*[tau] = S * sum_q y*_{q,tau}
        C_out = [0.0 for _ in range(T)]
        C_ret = [0.0 for _ in range(T)]
        # Vehicle counts per departure slot (per direction)
        K_out = [0 for _ in range(T)]
        K_ret = [0 for _ in range(T)]
        for name, val in candidate.items():
            if not isinstance(name, str):
                continue
            if name.startswith("yOUT["):
                inside = name[name.find("[") + 1 : name.find("]")]
                _, tau_str = inside.split(",")
                tau = int(tau_str.strip())
                if 0 <= tau < T:
                    C_out[tau] += float(val) * S
                    try:
                        if float(val) >= 0.5:
                            K_out[tau] += 1
                    except Exception:
                        pass
            elif name.startswith("yRET["):
                inside = name[name.find("[") + 1 : name.find("]")]
                _, tau_str = inside.split(",")
                tau = int(tau_str.strip())
                if 0 <= tau < T:
                    C_ret[tau] += float(val) * S
                    try:
                        if float(val) >= 0.5:
                            K_ret[tau] += 1
                    except Exception:
                        pass

        # Multi-scenario support
        scenarios: list[dict] = list(params.get("scenarios", []))
        # Allow specifying scenarios as file paths
        if not scenarios and isinstance(params.get("scenario_files"), list):
            scenarios = list(params.get("scenario_files"))
        # Normalize single-scenario lists to single-demand path
        single_scenario_override: tuple[list[float], list[float]] | None = None
        single_scenario_arrivals: tuple[list[float] | None, list[float] | None] | None = None
        if scenarios and len(scenarios) == 1:
            s0 = scenarios[0]
            if isinstance(s0, (str, Path)):
                R_out0, R_ret0 = _load_demand_from_file(s0, T)
                single_scenario_arrivals = _load_exact_arrivals_from_file(s0)
            elif isinstance(s0, dict) and ("requests" in s0 or "req_matrix" in s0 or "R_out" in s0 or "R_ret" in s0):
                R_out0, R_ret0 = _aggregate_requests(s0, T)
                single_scenario_arrivals = _extract_exact_arrival_minutes(s0)
            else:
                # Best effort
                R_out0 = list(getattr(s0, "R_out", [0.0] * T))
                R_ret0 = list(getattr(s0, "R_ret", [0.0] * T))
                single_scenario_arrivals = (None, None)
            R_out0 = (R_out0 + [0.0] * T)[:T]
            R_ret0 = (R_ret0 + [0.0] * T)[:T]
            single_scenario_override = (R_out0, R_ret0)
            scenarios = []
        # Multi-cut vs averaged cut control.
        # New flag: multi_cuts_by_scenario (True => return one cut per scenario)
        # Backward compat: if not provided, use legacy average_cuts_across_scenarios (True => single averaged cut)
        _mc = params.get("multi_cuts_by_scenario", None)
        if _mc is None:
            average_cuts: bool = bool(params.get("average_cuts_across_scenarios", False))
            multi_cuts: bool = not average_cuts
        else:
            multi_cuts = bool(_mc)
            average_cuts = not multi_cuts
        ub_aggregation: str = str(params.get("ub_aggregation", "mean"))
        weights: list[float] | None = params.get("scenario_weights")

        # Only evaluate finite differences for time slots that appear in candidate (fewer solves)
        active_taus = sorted(t_idx) if t_idx else list(range(T))

        # Optional Magnanti–Wong selection
        mw_enabled: bool = bool(params.get("use_magnanti_wong", False)) and (not temporal_refinement)
        core_point = params.get("mw_core_point") or {}
        Ybar_out = list(core_point.get("Yout", [])) if isinstance(core_point, dict) else []
        Ybar_ret = list(core_point.get("Yret", [])) if isinstance(core_point, dict) else []
        if len(Ybar_out) < T:
            Ybar_out = (Ybar_out + [0.0] * T)[:T]
        if len(Ybar_ret) < T:
            Ybar_ret = (Ybar_ret + [0.0] * T)[:T]
        # If the core point is still all zeros (common in early iters), seed it to a small positive
        # profile so MW has direction to select non-trivial duals.
        try:
            if sum(Ybar_out) + sum(Ybar_ret) == 0.0 and T > 0:
                Ybar_out = [1.0 for _ in range(T)]
                Ybar_ret = [1.0 for _ in range(T)]
        except Exception:
            pass

        def _pick_vehicle_for_tau(tau: int) -> int | None:
            q_candidates = master_q_idx if master_q_idx else list(range(int(params.get("Q", 0) or 0)))
            if not q_candidates:
                return None
            for q in q_candidates:
                used = False
                for tt in range(T):
                    if _cand_float(candidate, f"yOUT[{q},{tt}]", 0.0) >= 0.5 or _cand_float(candidate, f"yRET[{q},{tt}]", 0.0) >= 0.5:
                        used = True
                        break
                if not used:
                    return int(q)
            for q in q_candidates:
                if _cand_float(candidate, f"yOUT[{q},{tau}]", 0.0) < 0.5 and _cand_float(candidate, f"yRET[{q},{tau}]", 0.0) < 0.5:
                    return int(q)
            return int(q_candidates[0])

        def solve_mw_dual(
            T_: int,
            Wmax_slots: int,
            p_penalty: float,
            S_cap: float,
            K_out_use: list[int],
            K_ret_use: list[int],
            C_out_vec: list[float],
            C_ret_vec: list[float],
            R_out_vec: list[float],
            R_ret_vec: list[float],
            Ybar_out_vec: list[float],
            Ybar_ret_vec: list[float],
            ub_base: float,
            lp: str,
            lp_opts: dict | None = None,
        ) -> tuple[dict[int, float], dict[int, float]] | None:
            """Solve the dual LP on the optimal face to select a Pareto-optimal dual.

            Returns dm_out[t], dm_ret[t] (slopes w.r.t. sum_y_out[t], sum_y_ret[t]).
            """
            md = pyo.ConcreteModel()
            md.name = "mw_dual"
            Tset = range(T_)

            # Dual variables
            md.a_OUT = pyo.Var(Tset)
            md.a_RET = pyo.Var(Tset)
            md.pi_OUT = pyo.Var([(tau, k) for tau in Tset for k in range(int(K_out_use[tau]) if tau < len(K_out_use) else 0)], within=pyo.NonNegativeReals)
            md.pi_RET = pyo.Var([(tau, k) for tau in Tset for k in range(int(K_ret_use[tau]) if tau < len(K_ret_use) else 0)], within=pyo.NonNegativeReals)

            # Dual feasibility: for every primal x_OUT[t, tau, k]
            def df_out_rule(m, t, tau, k):
                # active arc iff (t+1) <= tau <= min(T-1, t+W)
                if not ((t + 1) <= tau <= min(T_ - 1, t + Wmax_slots)):
                    return pyo.Constraint.Skip
                return m.a_OUT[t] + m.pi_OUT[tau, k] >= float(max(0, tau - t)) + max(0.0, float(params.get("fill_first_epsilon", 0.0))) * float(k)
            md.DF_OUT = pyo.Constraint([(t, tau, k) for t in Tset for tau in Tset for k in range(int(K_out_use[tau]) if tau < len(K_out_use) else 0)],
                                       rule=lambda m, t, tau, k: df_out_rule(m, t, tau, k))
            # Dual feasibility for RET
            def df_ret_rule(m, t, tau, k):
                if not ((t + 1) <= tau <= min(T_ - 1, t + Wmax_slots)):
                    return pyo.Constraint.Skip
                return m.a_RET[t] + m.pi_RET[tau, k] >= float(max(0, tau - t)) + max(0.0, float(params.get("fill_first_epsilon", 0.0))) * float(k)
            md.DF_RET = pyo.Constraint([(t, tau, k) for t in Tset for tau in Tset for k in range(int(K_ret_use[tau]) if tau < len(K_ret_use) else 0)],
                                       rule=lambda m, t, tau, k: df_ret_rule(m, t, tau, k))

            # u constraints (nonnegativity vars): a_[t] >= p
            md.A_OUT_CAP = pyo.Constraint(Tset, rule=lambda m, t: m.a_OUT[t] >= float(p_penalty))
            md.A_RET_CAP = pyo.Constraint(Tset, rule=lambda m, t: m.a_RET[t] >= float(p_penalty))

            # Optimality face equality: dual objective equals primal UB at incumbent
            cap_out_rhs = [min(float(S_cap), float(C_out_vec[tau])) for tau in Tset]
            cap_ret_rhs = [min(float(S_cap), float(C_ret_vec[tau])) for tau in Tset]
            def dual_obj_expr(m):
                term_dem = sum(float(R_out_vec[t]) * m.a_OUT[t] for t in Tset) + sum(float(R_ret_vec[t]) * m.a_RET[t] for t in Tset)
                term_cap = sum(cap_out_rhs[tau] * m.pi_OUT[tau, k] for tau in Tset for k in range(int(K_out_use[tau]) if tau < len(K_out_use) else 0)) \
                           + sum(cap_ret_rhs[tau] * m.pi_RET[tau, k] for tau in Tset for k in range(int(K_ret_use[tau]) if tau < len(K_ret_use) else 0))
                return term_dem + term_cap
            md.OptFace = pyo.Constraint(expr=(dual_obj_expr(md) == float(ub_base)))

            # MW objective: maximize dm·Ybar where dm[tau] = S * sum_k pi[tau,k]
            md.obj = pyo.Objective(
                expr= float(S_cap) * (
                    sum(float(Ybar_out_vec[tau]) * sum(md.pi_OUT[tau, k] for k in range(int(K_out_use[tau]) if tau < len(K_out_use) else 0)) for tau in Tset)
                    + sum(float(Ybar_ret_vec[tau]) * sum(md.pi_RET[tau, k] for k in range(int(K_ret_use[tau]) if tau < len(K_ret_use) else 0)) for tau in Tset)
                ),
                sense=pyo.maximize,
            )

            solver = pyo.SolverFactory(lp)
            try:
                for k, v in (lp_opts or {}).items():
                    solver.options[k] = v
            except Exception:
                pass
            res = solver.solve(md, tee=False, load_solutions=False)
            term = getattr(res.solver, "termination_condition", None)
            if term not in (pyo.TerminationCondition.optimal,):
                return None
            try:
                md.solutions.load_from(res)
            except Exception:
                pass

            dm_out = {}
            dm_ret = {}
            for tau in range(T_):
                dm_out[tau] = float(S_cap) * sum(float(pyo.value(md.pi_OUT[tau, k])) for k in range(int(K_out_use[tau]) if tau < len(K_out_use) else 0))
                dm_ret[tau] = float(S_cap) * sum(float(pyo.value(md.pi_RET[tau, k])) for k in range(int(K_ret_use[tau]) if tau < len(K_ret_use) else 0))
            return dm_out, dm_ret

        # Finite-difference coefficient builder: for each tau, solve with +S capacity
        def coeffs_by_fdiff(
            ub_base: float,
            C_out_base: list[float],
            C_ret_base: list[float],
            K_out_base: list[int],
            K_ret_base: list[int],
            R_out_vec: list[float],
            R_ret_vec: list[float],
        ) -> tuple[Dict[tuple[int, int], float], Dict[tuple[int, int], float], Dict[int, float], Dict[int, float]]:
            coeff_y_out: Dict[tuple[int, int], float] = {}
            coeff_y_ret: Dict[tuple[int, int], float] = {}
            # Marginal effects by time (per one vehicle start at tau)
            dm_out: Dict[int, float] = {}
            dm_ret: Dict[int, float] = {}

            for tau in active_taus:
                q_probe = _pick_vehicle_for_tau(tau)
                if temporal_refinement and q_probe is not None:
                    cand_out = dict(candidate)
                    cand_out[f"yOUT[{q_probe},{tau}]"] = 1.0
                    _, ub_plus = solve_refined_subproblem(
                        SPParams(
                            T=T, Wmax_slots=Wmax, p=p_pen, lp_solver=lp_solver, S=S,
                            K_out=K_out_base, K_ret=K_ret_base, fill_eps=fill_eps,
                            solver_options=solver_options, eps_cut=eps_cut,
                            slot_resolution=slot_res, time_step_minutes=time_step_min,
                            T_minutes=params.get("T_minutes"), trip_duration_minutes=params.get("trip_duration_minutes"),
                            trip_slots=params.get("trip_slots"), Q=int(params.get("Q", len(q_idx) if q_idx else 0) or 0),
                            binit=params.get("binit"), initial_actions=params.get("initial_actions"),
                            Emax=params.get("Emax"), L=params.get("L"), delta_chg=params.get("delta_chg"),
                            eps_feas=float(params.get("eps_feas", 1e-7)),
                            solve_time_limit_s=params.get("solve_time_limit_s"),
                        ),
                        R_out_vec,
                        R_ret_vec,
                        cand_out,
                    )
                    dm_out[tau] = float(ub_plus - ub_base) if math.isfinite(float(ub_plus)) else 0.0

                    cand_ret = dict(candidate)
                    cand_ret[f"yRET[{q_probe},{tau}]"] = 1.0
                    _, ub_plus_r = solve_refined_subproblem(
                        SPParams(
                            T=T, Wmax_slots=Wmax, p=p_pen, lp_solver=lp_solver, S=S,
                            K_out=K_out_base, K_ret=K_ret_base, fill_eps=fill_eps,
                            solver_options=solver_options, eps_cut=eps_cut,
                            slot_resolution=slot_res, time_step_minutes=time_step_min,
                            T_minutes=params.get("T_minutes"), trip_duration_minutes=params.get("trip_duration_minutes"),
                            trip_slots=params.get("trip_slots"), Q=int(params.get("Q", len(q_idx) if q_idx else 0) or 0),
                            binit=params.get("binit"), initial_actions=params.get("initial_actions"),
                            Emax=params.get("Emax"), L=params.get("L"), delta_chg=params.get("delta_chg"),
                            eps_feas=float(params.get("eps_feas", 1e-7)),
                            solve_time_limit_s=params.get("solve_time_limit_s"),
                        ),
                        R_out_vec,
                        R_ret_vec,
                        cand_ret,
                    )
                    dm_ret[tau] = float(ub_plus_r - ub_base) if math.isfinite(float(ub_plus_r)) else 0.0
                else:
                    # OUT marginal in the legacy nominal-capacity LP.
                    C_out_tau = C_out_base.copy()
                    C_out_tau[tau] = C_out_tau[tau] + S
                    K_out_tau = K_out_base.copy()
                    K_out_tau[tau] = K_out_tau[tau] + 1
                    _, ub_plus = solve_subproblem(
                        SPParams(
                            T=T, Wmax_slots=Wmax, p=p_pen, lp_solver=lp_solver, S=S,
                            K_out=K_out_tau, K_ret=K_ret_base, fill_eps=fill_eps,
                            solver_options=solver_options, eps_cut=eps_cut,
                            slot_resolution=slot_res, time_step_minutes=time_step_min,
                            T_minutes=params.get("T_minutes"), trip_duration_minutes=params.get("trip_duration_minutes"),
                            trip_slots=params.get("trip_slots"), Q=int(params.get("Q", len(q_idx) if q_idx else 0) or 0),
                            binit=params.get("binit"), initial_actions=params.get("initial_actions"),
                            Emax=params.get("Emax"), L=params.get("L"), delta_chg=params.get("delta_chg"),
                            eps_feas=float(params.get("eps_feas", 1e-7)),
                            solve_time_limit_s=params.get("solve_time_limit_s"),
                        ),
                        C_out_tau,
                        C_ret_base,
                        R_out_vec,
                        R_ret_vec,
                    )
                    dm_out[tau] = float(ub_plus - ub_base)

                    # RET marginal in the legacy nominal-capacity LP.
                    C_ret_tau = C_ret_base.copy()
                    C_ret_tau[tau] = C_ret_tau[tau] + S
                    K_ret_tau = K_ret_base.copy()
                    K_ret_tau[tau] = K_ret_tau[tau] + 1
                    _, ub_plus_r = solve_subproblem(
                        SPParams(
                            T=T, Wmax_slots=Wmax, p=p_pen, lp_solver=lp_solver, S=S,
                            K_out=K_out_base, K_ret=K_ret_tau, fill_eps=fill_eps,
                            solver_options=solver_options, eps_cut=eps_cut,
                            slot_resolution=slot_res, time_step_minutes=time_step_min,
                            T_minutes=params.get("T_minutes"), trip_duration_minutes=params.get("trip_duration_minutes"),
                            trip_slots=params.get("trip_slots"), Q=int(params.get("Q", len(q_idx) if q_idx else 0) or 0),
                            binit=params.get("binit"), initial_actions=params.get("initial_actions"),
                            Emax=params.get("Emax"), L=params.get("L"), delta_chg=params.get("delta_chg"),
                            eps_feas=float(params.get("eps_feas", 1e-7)),
                            solve_time_limit_s=params.get("solve_time_limit_s"),
                        ),
                        C_out_base,
                        C_ret_tau,
                        R_out_vec,
                        R_ret_vec,
                    )
                    dm_ret[tau] = float(ub_plus_r - ub_base)

            # Expand to the full master variable grid, not just sparse incumbent keys.
            for q in master_q_idx:
                for tau, v in dm_out.items():
                    if 0 <= tau < T and abs(float(v)) > 0.0:
                        coeff_y_out[(int(q), int(tau))] = float(v)
                for tau, v in dm_ret.items():
                    if 0 <= tau < T and abs(float(v)) > 0.0:
                        coeff_y_ret[(int(q), int(tau))] = float(v)

            return coeff_y_out, coeff_y_ret, dm_out, dm_ret

        def _expand_time_slopes(
            dm_out: Dict[int, float],
            dm_ret: Dict[int, float],
        ) -> tuple[Dict[tuple[int, int], float], Dict[tuple[int, int], float]]:
            coeff_y_out: Dict[tuple[int, int], float] = {}
            coeff_y_ret: Dict[tuple[int, int], float] = {}
            for q in master_q_idx:
                for tau, v in dm_out.items():
                    if 0 <= tau < T and abs(float(v)) > 0.0:
                        coeff_y_out[(int(q), int(tau))] = float(v)
                for tau, v in dm_ret.items():
                    if 0 <= tau < T and abs(float(v)) > 0.0:
                        coeff_y_ret[(int(q), int(tau))] = float(v)
            return coeff_y_out, coeff_y_ret

        def _slot_exposure(R_vec: list[float]) -> list[float]:
            exposure = [0.0 for _ in range(T)]
            for tau in range(T):
                total = 0.0
                for t in range(T):
                    if (t + 1) <= tau <= min(T - 1, t + Wmax):
                        total += float(R_vec[t])
                exposure[tau] = total
            return exposure

        def _top_probe_slots(R_vec: list[float], max_probes: int) -> list[int]:
            exposure = _slot_exposure(R_vec)
            ranked = sorted(
                range(T),
                key=lambda tau: (float(exposure[tau]), float(R_vec[tau]) if tau < len(R_vec) else 0.0, -int(tau)),
                reverse=True,
            )
            chosen = [int(tau) for tau in ranked if exposure[tau] > 1e-9]
            if not chosen:
                chosen = [int(tau) for tau in ranked if (tau < len(R_vec) and float(R_vec[tau]) > 1e-9)]
            return chosen[: max(0, int(max_probes))]

        def _restricted_temporal_fdiff(
            ub_base: float,
            R_out_vec: list[float],
            R_ret_vec: list[float],
            top_k_out: int,
            top_k_ret: int,
        ) -> tuple[Dict[tuple[int, int], float], Dict[tuple[int, int], float], Dict[int, float], Dict[int, float], dict]:
            coeff_y_out: Dict[tuple[int, int], float] = {}
            coeff_y_ret: Dict[tuple[int, int], float] = {}
            dm_out: Dict[int, float] = {}
            dm_ret: Dict[int, float] = {}
            probe_slots_out = _top_probe_slots(R_out_vec, top_k_out)
            probe_slots_ret = _top_probe_slots(R_ret_vec, top_k_ret)
            probe_time = 0.0
            for tau in probe_slots_out:
                q_probe = _pick_vehicle_for_tau(tau)
                if q_probe is None:
                    continue
                cand_out = dict(candidate)
                cand_out[f"yOUT[{q_probe},{tau}]"] = 1.0
                t0 = time.perf_counter()
                _, ub_plus = solve_refined_subproblem(sp_params, R_out_vec, R_ret_vec, cand_out)
                t1 = time.perf_counter()
                probe_time += float(t1 - t0)
                dm_out[int(tau)] = float(ub_plus - ub_base) if math.isfinite(float(ub_plus)) else 0.0
            for tau in probe_slots_ret:
                q_probe = _pick_vehicle_for_tau(tau)
                if q_probe is None:
                    continue
                cand_ret = dict(candidate)
                cand_ret[f"yRET[{q_probe},{tau}]"] = 1.0
                t0 = time.perf_counter()
                _, ub_plus = solve_refined_subproblem(sp_params, R_out_vec, R_ret_vec, cand_ret)
                t1 = time.perf_counter()
                probe_time += float(t1 - t0)
                dm_ret[int(tau)] = float(ub_plus - ub_base) if math.isfinite(float(ub_plus)) else 0.0
            coeff_y_out, coeff_y_ret = _expand_time_slopes(dm_out, dm_ret)
            diag = {
                "mode": "restricted_finite_difference",
                "probe_slots_out": list(probe_slots_out),
                "probe_slots_ret": list(probe_slots_ret),
                "probe_time_s": float(probe_time),
            }
            return coeff_y_out, coeff_y_ret, dm_out, dm_ret, diag

        def _is_degenerate_cut(
            ub_val: float,
            coeff_y_out: Dict[tuple[int, int], float],
            coeff_y_ret: Dict[tuple[int, int], float],
            zero_tol: float,
        ) -> bool:
            if float(ub_val) <= zero_tol:
                return False
            nnz = sum(1 for v in coeff_y_out.values() if abs(float(v)) > zero_tol) + sum(1 for v in coeff_y_ret.values() if abs(float(v)) > zero_tol)
            if nnz == 0:
                return True
            max_abs = 0.0
            for v in coeff_y_out.values():
                max_abs = max(max_abs, abs(float(v)))
            for v in coeff_y_ret.values():
                max_abs = max(max_abs, abs(float(v)))
            return max_abs <= zero_tol

        def _build_anti_trivial_cut(reason: str) -> Cut:
            return Cut(
                name="anti_trivial_idle",
                cut_type=CutType.FEASIBILITY,
                metadata={
                    "anti_trivial_min_total_starts": 1,
                    "fallback_reason": reason,
                    "cut_family": "anti_trivial_idle",
                },
            )

        def _proxy_cut_from_nominal_lp(
            sp_params: "SPParams",
            R_out_vec: list[float],
            R_ret_vec: list[float],
            ub_target: float,
        ) -> tuple[Dict[tuple[int, int], float], Dict[tuple[int, int], float], Dict[int, float], Dict[int, float], dict]:
            proxy_t0 = time.perf_counter()
            proxy_duals, proxy_ub = solve_subproblem(sp_params, C_out, C_ret, R_out_vec, R_ret_vec, candidate)
            proxy_t1 = time.perf_counter()
            pi_out = dict(proxy_duals.get("pi_OUT", {}))
            pi_ret = dict(proxy_duals.get("pi_RET", {}))
            dm_out = {int(t): float(S) * float(pi_out.get(int(t), 0.0)) for t in range(T)}
            dm_ret = {int(t): float(S) * float(pi_ret.get(int(t), 0.0)) for t in range(T)}
            coeff_y_out, coeff_y_ret = _expand_time_slopes(dm_out, dm_ret)
            proxy_diag = {
                "mode": "nominal_lp_proxy",
                "proxy_recourse_objective": float(proxy_ub),
                "proxy_cut_solve_s": float(proxy_t1 - proxy_t0),
            }
            if isinstance(proxy_duals, dict):
                for key in (
                    "timing_build_s",
                    "timing_solve_s",
                    "timing_extract_s",
                    "timing_postprocess_s",
                    "timing_lp_export_s",
                    "model_num_variables",
                    "model_num_binary_variables",
                    "model_num_constraints",
                ):
                    if key in proxy_duals:
                        proxy_diag[f"proxy_{key}"] = proxy_duals[key]
            return coeff_y_out, coeff_y_ret, dm_out, dm_ret, proxy_diag

        # If scenarios provided, iterate; else use single-demand params
        if scenarios:
            if weights and len(weights) != len(scenarios):
                raise ValueError("scenario_weights must match number of scenarios")
            if not weights:
                weights = [1.0 / len(scenarios)] * len(scenarios)

            cuts: list[Cut] = []
            ub_vals: list[float] = []
            consts: list[float] = []
            consts_out: list[float] = []
            consts_ret: list[float] = []
            coeffs_out_list: list[Dict[tuple[int, int], float]] = []
            coeffs_ret_list: list[Dict[tuple[int, int], float]] = []
            scenario_diags: list[dict] = []
            scenario_records: list[dict] = []
            agg = {
                "objective_value": [],
                "waiting_cost_slots": [],
                "fill_eps_cost": [],
                "penalty_cost": [],
                "penalty_pax": [],
                "served_total": [],
                "total_demand": [],
            }

            for idx_s, s in enumerate(scenarios):
                if isinstance(s, (str, Path)):
                    R_out, R_ret = _load_demand_from_file(s, T)
                    arrival_minutes_out, arrival_minutes_ret = _load_exact_arrivals_from_file(s)
                    scen_label = str(s)
                elif isinstance(s, dict) and ("requests" in s or "req_matrix" in s or "R_out" in s or "R_ret" in s):
                    R_out, R_ret = _aggregate_requests(s, T)
                    arrival_minutes_out, arrival_minutes_ret = _extract_exact_arrival_minutes(s)
                    scen_label = str(s.get("name") or s.get("label") or "scenario")
                else:
                    # Best effort
                    R_out = list(getattr(s, "R_out", [0.0] * T))
                    R_ret = list(getattr(s, "R_ret", [0.0] * T))
                    arrival_minutes_out, arrival_minutes_ret = (None, None)
                    scen_label = str(getattr(s, "name", "scenario"))
                R_out = (R_out + [0.0] * T)[:T]
                R_ret = (R_ret + [0.0] * T)[:T]

                # If using dual slopes, force at least one layer per time to create capacity constraints
                use_dual = bool(params.get("use_dual_slopes", False)) and (not temporal_refinement)
                K_out_lp = [max(1, int(K_out[t])) for t in range(T)] if use_dual else K_out
                K_ret_lp = [max(1, int(K_ret[t])) for t in range(T)] if use_dual else K_ret
                sp_params = SPParams(
                    T=T, Wmax_slots=Wmax, p=p_pen, lp_solver=lp_solver, S=S,
                    K_out=K_out_lp, K_ret=K_ret_lp, fill_eps=fill_eps,
                    solver_options=solver_options, eps_cut=eps_cut,
                    slot_resolution=slot_res, time_step_minutes=time_step_min,
                    T_minutes=params.get("T_minutes"), trip_duration_minutes=params.get("trip_duration_minutes"),
                    trip_slots=params.get("trip_slots"), Q=int(params.get("Q", len(q_idx) if q_idx else 0) or 0),
                    binit=params.get("binit"), initial_actions=params.get("initial_actions"),
                    Emax=params.get("Emax"), L=params.get("L"), delta_chg=params.get("delta_chg"),
                    arrival_minutes_out=arrival_minutes_out, arrival_minutes_ret=arrival_minutes_ret,
                    eps_feas=float(params.get("eps_feas", 1e-7)),
                    debug_timing=debug_timing,
                    debug_solver_tee=bool(params.get("debug_solver_tee", False)),
                    debug_export_lp_iteration=params.get("debug_export_lp_iteration"),
                    debug_current_iteration=int(params.get("debug_current_iteration", -1) or -1),
                    debug_report_dir=params.get("debug_report_dir", params.get("report_dir", "Report")),
                    debug_force_nominal_departures=bool(params.get("debug_force_nominal_departures", False)),
                    debug_scenario_label=str(scen_label),
                    solve_time_limit_s=params.get("solve_time_limit_s"),
                )
                t_solve0 = time.perf_counter()
                if temporal_refinement:
                    duals, ub_val = solve_refined_subproblem(sp_params, R_out, R_ret, candidate)
                else:
                    duals, ub_val = solve_subproblem(sp_params, C_out, C_ret, R_out, R_ret, candidate)
                t_solve1 = time.perf_counter()
                sp_solve_time = t_solve1 - t_solve0
                _dbg(
                    "[SP TIMING] iter=%s scenario=%s solve_total=%.3fs build=%.3fs solve=%.3fs extract=%.3fs post=%.3fs cut_mode=%s"
                    % (
                        str(params.get("debug_current_iteration", "-")),
                        scen_label,
                        float(sp_solve_time),
                        float(duals.get("timing_build_s", 0.0) or 0.0),
                        float(duals.get("timing_solve_s", 0.0) or 0.0),
                        float(duals.get("timing_extract_s", 0.0) or 0.0),
                        float(duals.get("timing_postprocess_s", 0.0) or 0.0),
                        refined_cut_mode,
                    )
                )
                if not bool(duals.get("is_feasible", True)):
                    scenario_diags.append({
                        "label": scen_label,
                        "T": T,
                        "R_out": [float(R_out[t]) for t in range(T)],
                        "R_ret": [float(R_ret[t]) for t in range(T)],
                        "infeasible": True,
                        "infeasibility_reason": duals.get("infeasibility_reason"),
                        "first_violation": duals.get("first_violation"),
                        "timing_sp_solve_s": sp_solve_time,
                        "timing_cutgen_s": 0.0,
                    })
                    return SubproblemResult(
                        is_feasible=False,
                        upper_bound=None,
                        diagnostics={
                            "T": T,
                            "scenarios": scenario_diags,
                            "scenario_weights": list(weights) if weights is not None else None,
                            "infeasible": True,
                            "infeasibility_reason": duals.get("infeasibility_reason"),
                            "first_violation": duals.get("first_violation"),
                            "slot_resolution": int(params.get("slot_resolution", 1)),
                            "timing_sp_solve_s": sp_solve_time,
                            "timing_cutgen_s": 0.0,
                        },
                    )
                ub_vals.append(ub_val)

                scenario_records.append({
                    "idx": int(idx_s),
                    "label": scen_label,
                    "R_out": R_out,
                    "R_ret": R_ret,
                    "duals": duals,
                    "ub_val": float(ub_val),
                    "sp_solve_time": float(sp_solve_time),
                })
                agg["objective_value"].append(float(duals.get("objective_value", ub_val)))
                agg["waiting_cost_slots"].append(float(duals.get("waiting_cost_slots", 0.0)))
                agg["fill_eps_cost"].append(float(duals.get("fill_eps_cost", 0.0)))
                agg["penalty_cost"].append(float(duals.get("penalty_cost", 0.0)))
                agg["penalty_pax"].append(float(duals.get("penalty_pax", 0.0)))
                agg["served_total"].append(float(duals.get("served_total", 0.0)))
                agg["total_demand"].append(float(duals.get("total_demand", 0.0)))
                agg.setdefault("timing_sp_solve_s", []).append(sp_solve_time)

            # Aggregate UB
            if ub_aggregation == "mean":
                ub_val_agg = sum(w * u for w, u in zip(weights, ub_vals))
                agg_obj = sum(w * u for w, u in zip(weights, agg["objective_value"]))
                agg_wait = sum(w * u for w, u in zip(weights, agg["waiting_cost_slots"]))
                agg_fill = sum(w * u for w, u in zip(weights, agg["fill_eps_cost"]))
                agg_pen = sum(w * u for w, u in zip(weights, agg["penalty_cost"]))
                agg_pen_pax = sum(w * u for w, u in zip(weights, agg["penalty_pax"]))
                agg_served = sum(w * u for w, u in zip(weights, agg["served_total"]))
                agg_total = sum(w * u for w, u in zip(weights, agg["total_demand"]))
            elif ub_aggregation == "sum":
                ub_val_agg = sum(ub_vals)
                agg_obj = sum(agg["objective_value"])
                agg_wait = sum(agg["waiting_cost_slots"])
                agg_fill = sum(agg["fill_eps_cost"])
                agg_pen = sum(agg["penalty_cost"])
                agg_pen_pax = sum(agg["penalty_pax"])
                agg_served = sum(agg["served_total"])
                agg_total = sum(agg["total_demand"])
            elif ub_aggregation == "max":
                ub_val_agg = max(ub_vals)
                idx_max = int(ub_vals.index(ub_val_agg))
                agg_obj = agg["objective_value"][idx_max]
                agg_wait = agg["waiting_cost_slots"][idx_max]
                agg_fill = agg["fill_eps_cost"][idx_max]
                agg_pen = agg["penalty_cost"][idx_max]
                agg_pen_pax = agg["penalty_pax"][idx_max]
                agg_served = agg["served_total"][idx_max]
                agg_total = agg["total_demand"][idx_max]
            else:
                raise ValueError("ub_aggregation must be one of 'mean', 'sum', 'max'")

            if debug_skip_cut_generation:
                for rec in scenario_records:
                    duals = rec["duals"]
                    R_out = rec["R_out"]
                    R_ret = rec["R_ret"]
                    scenario_diags.append({
                        "label": rec["label"],
                        "T": T,
                        "R_out": [float(R_out[t]) for t in range(T)],
                        "R_ret": [float(R_ret[t]) for t in range(T)],
                        "pax_out_by_tau": list(duals.get("served_out_by_tau", [0.0] * T)),
                        "pax_ret_by_tau": list(duals.get("served_ret_by_tau", [0.0] * T)),
                        "pax_out_by_tau_k": list(duals.get("served_out_by_tau_k", [[] for _ in range(T)])),
                        "pax_ret_by_tau_k": list(duals.get("served_ret_by_tau_k", [[] for _ in range(T)])),
                        "objective_value": float(duals.get("objective_value", rec["ub_val"])),
                        "waiting_cost_slots": float(duals.get("waiting_cost_slots", 0.0)),
                        "fill_eps_cost": float(duals.get("fill_eps_cost", 0.0)),
                        "penalty_cost": float(duals.get("penalty_cost", 0.0)),
                        "penalty_pax": float(duals.get("penalty_pax", 0.0)),
                        "served_total": float(duals.get("served_total", 0.0)),
                        "total_demand": float(duals.get("total_demand", 0.0)),
                        "realized_departures": list(duals.get("realized_departures", [])),
                        "realized_departure_min_map": dict(duals.get("realized_departure_min_map", {})),
                        "refined_departure_diagnostics": list(duals.get("refined_departure_diagnostics", [])),
                        "refined_departure_diagnostics_focus": list(duals.get("refined_departure_diagnostics_focus", [])),
                        "effective_pre_service": list(duals.get("effective_pre_service", [])),
                        "battery_trajectory": duals.get("battery_trajectory", {}),
                        "timing_sp_solve_s": rec["sp_solve_time"],
                        "timing_cutgen_s": 0.0,
                        "cut_generation_skipped": True,
                    })
                return SubproblemResult(
                    is_feasible=True,
                    cuts=[],
                    upper_bound=ub_val_agg,
                    diagnostics={
                        "T": T,
                        "scenarios": scenario_diags,
                        "scenario_weights": list(weights) if weights is not None else None,
                        "objective_value": agg_obj,
                        "waiting_cost_slots": agg_wait,
                        "fill_eps_cost": agg_fill,
                        "penalty_cost": agg_pen,
                        "penalty_pax": agg_pen_pax,
                        "served_total": agg_served,
                        "total_demand": agg_total,
                        "slot_resolution": int(params.get("slot_resolution", 1)),
                        "timing_sp_solve_s": sum(float(x) for x in agg.get("timing_sp_solve_s", []) or []),
                        "timing_cutgen_s": 0.0,
                        "cut_generation_mode": "skipped_debug",
                    },
                )

            # Early-exit for scalar theta in multi-scenario runs (skip all cut generation)
            theta_val = _cand_theta("__theta")
            has_theta_s = any(isinstance(k, str) and k.startswith("__theta_s[") for k in candidate.keys())
            if debug_early_exit:
                try:
                    rhs = float(ub_val_agg) - float(eps_cut) * (1.0 + abs(float(ub_val_agg)))
                except Exception:
                    rhs = None
                try:
                    theta_keys = [k for k in candidate.keys() if isinstance(k, str) and k.startswith("__theta")][:20]
                except Exception:
                    theta_keys = []
                early_exit_ok = (theta_val is not None) and (not has_theta_s) and _ok(theta_val, float(ub_val_agg), eps_cut)
                msg = (
                    "[EARLY-EXIT] scenarios=%s theta_val=%s has_theta_s=%s ub_val_agg=%s eps_cut=%s rhs=%s early_exit_ok=%s theta_keys=%s"
                    % (
                        str(len(scenarios)),
                        (f"{float(theta_val):.6g}" if theta_val is not None else "-"),
                        str(bool(has_theta_s)),
                        (f"{float(ub_val_agg):.6g}" if ub_val_agg is not None else "-"),
                        (f"{float(eps_cut):.6g}" if eps_cut is not None else "-"),
                        (f"{float(rhs):.6g}" if rhs is not None else "-"),
                        str(bool(early_exit_ok)),
                        str(theta_keys),
                    )
                )
                try:
                    import logging as _logging
                    _log = _logging.getLogger(__name__)
                    _log.setLevel(_logging.INFO)
                    if not _log.handlers:
                        _logging.basicConfig(level=_logging.INFO)
                    _log.info(msg)
                except Exception:
                    self._vprint(msg)
            if (theta_val is not None) and (not has_theta_s) and _ok(theta_val, float(ub_val_agg), eps_cut):
                for rec in scenario_records:
                    duals = rec["duals"]
                    R_out = rec["R_out"]
                    R_ret = rec["R_ret"]
                    scenario_diags.append({
                        "label": rec["label"],
                        "T": T,
                        "R_out": [float(R_out[t]) for t in range(T)],
                        "R_ret": [float(R_ret[t]) for t in range(T)],
                        "pax_out_by_tau": list(duals.get("served_out_by_tau", [0.0] * T)),
                        "pax_ret_by_tau": list(duals.get("served_ret_by_tau", [0.0] * T)),
                        "pax_out_by_tau_k": list(duals.get("served_out_by_tau_k", [[] for _ in range(T)])),
                        "pax_ret_by_tau_k": list(duals.get("served_ret_by_tau_k", [[] for _ in range(T)])),
                        "objective_value": float(duals.get("objective_value", rec["ub_val"])),
                        "waiting_cost_slots": float(duals.get("waiting_cost_slots", 0.0)),
                        "fill_eps_cost": float(duals.get("fill_eps_cost", 0.0)),
                        "penalty_cost": float(duals.get("penalty_cost", 0.0)),
                        "penalty_pax": float(duals.get("penalty_pax", 0.0)),
                        "served_total": float(duals.get("served_total", 0.0)),
                        "total_demand": float(duals.get("total_demand", 0.0)),
                        "timing_sp_solve_s": rec["sp_solve_time"],
                        "timing_cutgen_s": 0.0,
                    })
                return SubproblemResult(
                    is_feasible=True,
                    cuts=[],
                    upper_bound=ub_val_agg,
                    diagnostics={
                        "T": T,
                        "scenarios": scenario_diags,
                        "scenario_weights": list(weights) if weights is not None else None,
                        "objective_value": agg_obj,
                        "waiting_cost_slots": agg_wait,
                        "fill_eps_cost": agg_fill,
                        "penalty_cost": agg_pen,
                        "penalty_pax": agg_pen_pax,
                        "served_total": agg_served,
                        "total_demand": agg_total,
                        "slot_resolution": int(params.get("slot_resolution", 1)),
                        "timing_sp_solve_s": sum(float(x) for x in agg.get("timing_sp_solve_s", []) or []),
                        "timing_cutgen_s": 0.0,
                    },
                )

            # Phase 2: generate cuts (unless early-exit above)
            use_dual = bool(params.get("use_dual_slopes", False)) and (not temporal_refinement)
            K_out_lp = [max(1, int(K_out[t])) for t in range(T)] if use_dual else K_out
            K_ret_lp = [max(1, int(K_ret[t])) for t in range(T)] if use_dual else K_ret
            for rec in scenario_records:
                idx_s = int(rec["idx"])
                scen_label = rec["label"]
                R_out = rec["R_out"]
                R_ret = rec["R_ret"]
                duals = rec["duals"]
                ub_val = float(rec["ub_val"])
                sp_solve_time = float(rec["sp_solve_time"])

                # Early-exit per scenario if per-scenario theta is already consistent
                # Skip any cut generation work for this scenario.
                theta_s = _cand_theta(f"__theta_s[{int(idx_s)}]")
                if (theta_s is not None) and _ok(theta_s, float(ub_val), eps_cut):
                    cutgen_time = 0.0
                    # Collect per-scenario diagnostics for reporting
                    scenario_diags.append({
                        "label": scen_label,
                        "T": T,
                        "R_out": [float(R_out[t]) for t in range(T)],
                        "R_ret": [float(R_ret[t]) for t in range(T)],
                        "pax_out_by_tau": list(duals.get("served_out_by_tau", [0.0] * T)),
                        "pax_ret_by_tau": list(duals.get("served_ret_by_tau", [0.0] * T)),
                        "pax_out_by_tau_k": list(duals.get("served_out_by_tau_k", [[] for _ in range(T)])),
                        "pax_ret_by_tau_k": list(duals.get("served_ret_by_tau_k", [[] for _ in range(T)])),
                        "objective_value": float(duals.get("objective_value", ub_val)),
                        "waiting_cost_slots": float(duals.get("waiting_cost_slots", 0.0)),
                        "fill_eps_cost": float(duals.get("fill_eps_cost", 0.0)),
                        "penalty_cost": float(duals.get("penalty_cost", 0.0)),
                        "penalty_pax": float(duals.get("penalty_pax", 0.0)),
                        "served_total": float(duals.get("served_total", 0.0)),
                        "total_demand": float(duals.get("total_demand", 0.0)),
                        "timing_sp_solve_s": sp_solve_time,
                        "timing_cutgen_s": cutgen_time,
                    })
                    agg.setdefault("timing_cutgen_s", []).append(cutgen_time)
                    continue

                # Build marginal slopes either from duals (fast) or finite differences (fallback)
                t_cut0 = time.perf_counter()
                cut_mode_used = refined_cut_mode if temporal_refinement else ("dual" if use_dual else "finite_difference")
                proxy_diag: dict[str, Any] = {}
                cut_lb_valid = not temporal_refinement
                if temporal_refinement and refined_cut_mode == "refined_lp_relaxation":
                    cut_lp, _ = solve_refined_lp_relaxation_cut(sp_params, R_out, R_ret, candidate)
                    c_out_map = dict(cut_lp.get("coeff_yOUT", {}))
                    c_ret_map = dict(cut_lp.get("coeff_yRET", {}))
                    dm_out = {int(t): sum(float(c_out_map.get((int(q), int(t)), 0.0)) for q in master_q_idx) for t in range(T)}
                    dm_ret = {int(t): sum(float(c_ret_map.get((int(q), int(t)), 0.0)) for q in master_q_idx) for t in range(T)}
                    cut_lb_valid = True
                    proxy_diag = cut_lp
                elif temporal_refinement and refined_cut_mode in {"nominal_lp_proxy", "proxy_dual"}:
                    c_out_map, c_ret_map, dm_out, dm_ret, proxy_diag = _proxy_cut_from_nominal_lp(sp_params, R_out, R_ret, ub_val)
                    cut_lb_valid = False
                elif mw_enabled:
                    # MW-selected dual slopes on optimal face
                    # Ensure at least one capacity layer per tau for dual π variables
                    K_out_mw = [max(1, int(K_out_lp[t])) for t in range(T)]
                    K_ret_mw = [max(1, int(K_ret_lp[t])) for t in range(T)]
                    dm_pair = solve_mw_dual(
                        T, Wmax, p_pen, S,
                        K_out_mw, K_ret_mw,
                        C_out, C_ret,
                        R_out, R_ret,
                        Ybar_out, Ybar_ret,
                        ub_val,
                        lp_solver,
                        solver_options,
                    )
                    if dm_pair is None:
                        # Fallback to finite differences to guarantee nonzero slopes
                        c_out_fd, c_ret_fd, dm_out, dm_ret = coeffs_by_fdiff(ub_val, C_out, C_ret, K_out, K_ret, R_out, R_ret)
                    else:
                        dm_out, dm_ret = dm_pair
                    # Expand to per-(q,t)
                    c_out_map: Dict[tuple[int, int], float] = {}
                    c_ret_map: Dict[tuple[int, int], float] = {}
                    for name in candidate.keys():
                        if not isinstance(name, str):
                            continue
                        if name.startswith("yOUT["):
                            inside = name[name.find("[") + 1 : name.find("]")]
                            q_str, tau_str = inside.split(",")
                            q = int(q_str.strip()); tau = int(tau_str.strip())
                            if 0 <= tau < T:
                                c_out_map[(q, tau)] = dm_out.get(tau, 0.0)
                        elif name.startswith("yRET["):
                            inside = name[name.find("[") + 1 : name.find("]")]
                            q_str, tau_str = inside.split(",")
                            q = int(q_str.strip()); tau = int(tau_str.strip())
                            if 0 <= tau < T:
                                c_ret_map[(q, tau)] = dm_ret.get(tau, 0.0)
                elif use_dual:
                    pi_out = dict(duals.get("pi_OUT", {}))
                    pi_ret = dict(duals.get("pi_RET", {}))
                    # Duals on capacity (<=) constraints in Pyomo have negative sign for binding constraints
                    # Build supporting hyperplane slopes consistent with finite differences: dm should be ≤ 0
                    dm_out = {int(t): float(S) * float(pi_out.get(int(t), 0.0)) for t in range(T)}
                    dm_ret = {int(t): float(S) * float(pi_ret.get(int(t), 0.0)) for t in range(T)}
                    # Expand to per-(q,t)
                    c_out_map: Dict[tuple[int, int], float] = {}
                    c_ret_map: Dict[tuple[int, int], float] = {}
                    for name in candidate.keys():
                        if not isinstance(name, str):
                            continue
                        if name.startswith("yOUT["):
                            inside = name[name.find("[") + 1 : name.find("]")]
                            q_str, tau_str = inside.split(",")
                            q = int(q_str.strip()); tau = int(tau_str.strip())
                            if 0 <= tau < T:
                                c_out_map[(q, tau)] = dm_out.get(tau, 0.0)
                        elif name.startswith("yRET["):
                            inside = name[name.find("[") + 1 : name.find("]")]
                            q_str, tau_str = inside.split(",")
                            q = int(q_str.strip()); tau = int(tau_str.strip())
                            if 0 <= tau < T:
                                c_ret_map[(q, tau)] = dm_ret.get(tau, 0.0)
                else:
                    # Finite-difference coefficients and constant per scenario
                    c_out_map, c_ret_map, dm_out, dm_ret = coeffs_by_fdiff(ub_val, C_out, C_ret, K_out, K_ret, R_out, R_ret)
                    cut_lb_valid = False
                # Proxy cut first; if it degenerates, escalate to a sparse restricted
                # finite-difference fallback on promising slots rather than probing all slots.
                deg_tol = float(params.get("degenerate_cut_zero_tol", 1e-9) or 1e-9)
                fallback_diag: dict[str, Any] = {}
                if temporal_refinement and (not cut_lb_valid) and _is_degenerate_cut(float(ub_val), c_out_map, c_ret_map, deg_tol):
                    top_k_out = int(params.get("degenerate_cut_probe_top_k_out", params.get("degenerate_cut_probe_top_k", 6)))
                    top_k_ret = int(params.get("degenerate_cut_probe_top_k_ret", params.get("degenerate_cut_probe_top_k", 6)))
                    c_out_map, c_ret_map, dm_out, dm_ret, fallback_diag = _restricted_temporal_fdiff(ub_val, R_out, R_ret, top_k_out, top_k_ret)
                    if _is_degenerate_cut(float(ub_val), c_out_map, c_ret_map, deg_tol):
                        if _candidate_is_all_idle():
                            cuts.append(_build_anti_trivial_cut("degenerate_proxy_after_restricted_fdiff"))
                            scenario_diags.append({
                                "label": scen_label,
                                "T": T,
                                "R_out": [float(R_out[t]) for t in range(T)],
                                "R_ret": [float(R_ret[t]) for t in range(T)],
                                "objective_value": float(duals.get("objective_value", ub_val)),
                                "timing_sp_solve_s": sp_solve_time,
                                "timing_cutgen_s": time.perf_counter() - t_cut0,
                                "cut_generation_mode": "anti_trivial_idle_fallback",
                                "cut_generation_proxy": proxy_diag,
                                "cut_generation_fallback": fallback_diag,
                            })
                            agg.setdefault("timing_cutgen_s", []).append(time.perf_counter() - t_cut0)
                            continue
                    cut_mode_used = "restricted_finite_difference_fallback"
                t_cut1 = time.perf_counter()
                cutgen_time = t_cut1 - t_cut0
                _dbg(
                    "[SP TIMING] iter=%s scenario=%s cutgen=%.3fs mode=%s proxy_obj=%s"
                    % (
                        str(params.get("debug_current_iteration", "-")),
                        scen_label,
                        float(cutgen_time),
                        cut_mode_used,
                        (
                            f"{float(proxy_diag.get('proxy_recourse_objective')):.6g}"
                            if proxy_diag.get("proxy_recourse_objective") is not None
                            else "-"
                        ),
                    )
                )

                sum_y_out = [float(C_out[tau]) / S if S != 0 else 0.0 for tau in range(T)]
                sum_y_ret = [float(C_ret[tau]) / S if S != 0 else 0.0 for tau in range(T)]
                if temporal_refinement and refined_cut_mode == "refined_lp_relaxation":
                    const = float(proxy_diag.get("const", 0.0))
                else:
                    const = float(ub_val)
                    const -= sum(dm_out.get(tau, 0.0) * sum_y_out[tau] for tau in range(T))
                    const -= sum(dm_ret.get(tau, 0.0) * sum_y_ret[tau] for tau in range(T))
                consts.append(const)
                coeffs_out_list.append(c_out_map)
                coeffs_ret_list.append(c_ret_map)
                # Per-direction constants if available from SP diagnostics
                if temporal_refinement and refined_cut_mode == "refined_lp_relaxation":
                    const_out = float(proxy_diag.get("const_out", const))
                    const_ret = float(proxy_diag.get("const_ret", 0.0))
                else:
                    try:
                        ub_out = float(duals.get("ub_out", const))
                    except Exception:
                        ub_out = float(const)
                    try:
                        ub_ret = float(duals.get("ub_ret", 0.0))
                    except Exception:
                        ub_ret = 0.0
                    const_out = float(ub_out) - sum(dm_out.get(tau, 0.0) * sum_y_out[tau] for tau in range(T))
                    const_ret = float(ub_ret) - sum(dm_ret.get(tau, 0.0) * sum_y_ret[tau] for tau in range(T))
                consts_out.append(const_out)
                consts_ret.append(const_ret)
                # Evaluate line at incumbent for diagnostics
                theta_lb_s = float(const) + sum(float(v) * _cand_float(candidate, f"yOUT[{int(q)},{int(tau)}]", 0.0) for (q, tau), v in c_out_map.items()) \
                    + sum(float(v) * _cand_float(candidate, f"yRET[{int(q)},{int(tau)}]", 0.0) for (q, tau), v in c_ret_map.items())
                target_val = float(proxy_diag.get("objective_value", ub_val)) if temporal_refinement and refined_cut_mode == "refined_lp_relaxation" else float(ub_val)
                if abs(float(target_val) - float(theta_lb_s)) > eps_cut * max(1.0, abs(float(target_val))):
                    raise RuntimeError("Cut tightness failed at incumbent; aborting cut generation.")
                cuts.append(Cut(
                    name=f"opt_cut_s_{int(idx_s)}",
                    cut_type=CutType.OPTIMALITY,
                    metadata={
                        "const": const,
                        "const_out": const_out,
                        "const_ret": const_ret,
                        "coeff_yOUT": c_out_map,
                        "theta_lb": float(theta_lb_s),
                        "coeff_yRET": c_ret_map,
                        "recourse_total": float(target_val),
                        "recourse_out": float(duals.get("ub_out", 0.0)),
                        "recourse_ret": float(duals.get("ub_ret", 0.0)),
                        "scenario_index": int(idx_s),
                        "cut_valid_lower_bound": bool(cut_lb_valid),
                    },
                ))
                # Collect per-scenario diagnostics for reporting
                scenario_diags.append({
                    "label": scen_label,
                    "T": T,
                    "R_out": [float(R_out[t]) for t in range(T)],
                    "R_ret": [float(R_ret[t]) for t in range(T)],
                    "pax_out_by_tau": list(duals.get("served_out_by_tau", [0.0] * T)),
                    "pax_ret_by_tau": list(duals.get("served_ret_by_tau", [0.0] * T)),
                    "pax_out_by_tau_k": list(duals.get("served_out_by_tau_k", [[] for _ in range(T)])),
                    "pax_ret_by_tau_k": list(duals.get("served_ret_by_tau_k", [[] for _ in range(T)])),
                    "objective_value": float(duals.get("objective_value", ub_val)),
                    "waiting_cost_slots": float(duals.get("waiting_cost_slots", 0.0)),
                    "fill_eps_cost": float(duals.get("fill_eps_cost", 0.0)),
                    "penalty_cost": float(duals.get("penalty_cost", 0.0)),
                    "penalty_pax": float(duals.get("penalty_pax", 0.0)),
                    "served_total": float(duals.get("served_total", 0.0)),
                    "total_demand": float(duals.get("total_demand", 0.0)),
                    "realized_departures": list(duals.get("realized_departures", [])),
                    "realized_departure_min_map": dict(duals.get("realized_departure_min_map", {})),
                    "refined_departure_diagnostics": list(duals.get("refined_departure_diagnostics", [])),
                    "refined_departure_diagnostics_focus": list(duals.get("refined_departure_diagnostics_focus", [])),
                    "effective_pre_service": list(duals.get("effective_pre_service", [])),
                    "battery_trajectory": duals.get("battery_trajectory", {}),
                    "timing_sp_solve_s": sp_solve_time,
                    "timing_cutgen_s": cutgen_time,
                    "cut_generation_mode": cut_mode_used,
                    "cut_generation_proxy": proxy_diag,
                    "cut_generation_fallback": fallback_diag,
                    "cut_valid_lower_bound": bool(cut_lb_valid),
                })
                agg.setdefault("timing_cutgen_s", []).append(cutgen_time)

            # Aggregate UB
            if ub_aggregation == "mean":
                ub_val_agg = sum(w * u for w, u in zip(weights, ub_vals))
                agg_obj = sum(w * u for w, u in zip(weights, agg["objective_value"]))
                agg_wait = sum(w * u for w, u in zip(weights, agg["waiting_cost_slots"]))
                agg_fill = sum(w * u for w, u in zip(weights, agg["fill_eps_cost"]))
                agg_pen = sum(w * u for w, u in zip(weights, agg["penalty_cost"]))
                agg_pen_pax = sum(w * u for w, u in zip(weights, agg["penalty_pax"]))
                agg_served = sum(w * u for w, u in zip(weights, agg["served_total"]))
                agg_total = sum(w * u for w, u in zip(weights, agg["total_demand"]))
                agg_sp_time = sum(w * u for w, u in zip(weights, agg["timing_sp_solve_s"]))
                agg_cut_time = sum(w * u for w, u in zip(weights, agg["timing_cutgen_s"]))
            elif ub_aggregation == "sum":
                ub_val_agg = sum(ub_vals)
                agg_obj = sum(agg["objective_value"])
                agg_wait = sum(agg["waiting_cost_slots"])
                agg_fill = sum(agg["fill_eps_cost"])
                agg_pen = sum(agg["penalty_cost"])
                agg_pen_pax = sum(agg["penalty_pax"])
                agg_served = sum(agg["served_total"])
                agg_total = sum(agg["total_demand"])
                agg_sp_time = sum(agg["timing_sp_solve_s"])
                agg_cut_time = sum(agg["timing_cutgen_s"])
            elif ub_aggregation == "max":
                ub_val_agg = max(ub_vals)
                idx_max = int(ub_vals.index(ub_val_agg))
                agg_obj = agg["objective_value"][idx_max]
                agg_wait = agg["waiting_cost_slots"][idx_max]
                agg_fill = agg["fill_eps_cost"][idx_max]
                agg_pen = agg["penalty_cost"][idx_max]
                agg_pen_pax = agg["penalty_pax"][idx_max]
                agg_served = agg["served_total"][idx_max]
                agg_total = agg["total_demand"][idx_max]
                agg_sp_time = agg["timing_sp_solve_s"][idx_max]
                agg_cut_time = agg["timing_cutgen_s"][idx_max]
            else:
                raise ValueError("ub_aggregation must be one of 'mean', 'sum', 'max'")
            agg_cut_lb_valid = all(bool(sd.get("cut_valid_lower_bound", False)) for sd in scenario_diags) if scenario_diags else False

            if not multi_cuts:
                # Weighted average of constants and coefficients
                const_avg = sum(w * c for w, c in zip(weights, consts))
                const_out_avg = sum(w * c for w, c in zip(weights, consts_out)) if consts_out else const_avg
                const_ret_avg = sum(w * c for w, c in zip(weights, consts_ret)) if consts_ret else 0.0
                keys_out = set().union(*[set(d.keys()) for d in coeffs_out_list])
                keys_ret = set().union(*[set(d.keys()) for d in coeffs_ret_list])
                avg_out: Dict[tuple[int, int], float] = {}
                avg_ret: Dict[tuple[int, int], float] = {}
                for k in keys_out:
                    avg_out[k] = sum(weights[i] * coeffs_out_list[i].get(k, 0.0) for i in range(len(coeffs_out_list)))
                for k in keys_ret:
                    avg_ret[k] = sum(weights[i] * coeffs_ret_list[i].get(k, 0.0) for i in range(len(coeffs_ret_list)))
                cut = Cut(
                    name="opt_cut_avg",
                    cut_type=CutType.OPTIMALITY,
                    metadata={
                        "const": const_avg,
                        "const_out": const_out_avg,
                        "const_ret": const_ret_avg,
                        "coeff_yOUT": avg_out,
                        "coeff_yRET": avg_ret,
                        "recourse_total": float(ub_val_agg),
                        "recourse_out": float(sum(weights[i] * scenario_records[i]["duals"].get("ub_out", 0.0) for i in range(len(scenario_records)))),
                        "recourse_ret": float(sum(weights[i] * scenario_records[i]["duals"].get("ub_ret", 0.0) for i in range(len(scenario_records)))),
                        "cut_valid_lower_bound": bool(agg_cut_lb_valid),
                    },
                )
                return SubproblemResult(
                    is_feasible=True,
                    cut=cut,
                    upper_bound=ub_val_agg,
                    diagnostics={
                        "T": T,
                        "scenarios": scenario_diags,
                        "scenario_weights": list(weights) if weights is not None else None,
                        "objective_value": agg_obj,
                        "waiting_cost_slots": agg_wait,
                        "fill_eps_cost": agg_fill,
                        "penalty_cost": agg_pen,
                        "penalty_pax": agg_pen_pax,
                        "served_total": agg_served,
                        "total_demand": agg_total,
                        "slot_resolution": int(params.get("slot_resolution", 1)),
                        "timing_sp_solve_s": agg_sp_time,
                        "timing_cutgen_s": agg_cut_time,
                        "cut_valid_lower_bound": bool(agg_cut_lb_valid),
                    },
                )
            else:
                return SubproblemResult(
                    is_feasible=True,
                    cuts=cuts,
                    upper_bound=ub_val_agg,
                    diagnostics={
                        "T": T,
                        "scenarios": scenario_diags,
                        "scenario_weights": list(weights) if weights is not None else None,
                        "objective_value": agg_obj,
                        "waiting_cost_slots": agg_wait,
                        "fill_eps_cost": agg_fill,
                        "penalty_cost": agg_pen,
                        "penalty_pax": agg_pen_pax,
                        "served_total": agg_served,
                        "total_demand": agg_total,
                        "slot_resolution": int(params.get("slot_resolution", 1)),
                        "timing_sp_solve_s": agg_sp_time,
                        "timing_cutgen_s": agg_cut_time,
                        "cut_valid_lower_bound": bool(agg_cut_lb_valid),
                    },
                )
        else:
            # Single-demand case from params (prefer external file if given)
            if single_scenario_override is not None:
                R_out, R_ret = single_scenario_override
                arrival_minutes_out, arrival_minutes_ret = single_scenario_arrivals or (None, None)
            elif params.get("demand_file"):
                R_out, R_ret = _load_demand_from_file(params.get("demand_file"), T)
                arrival_minutes_out, arrival_minutes_ret = _load_exact_arrivals_from_file(params.get("demand_file"))
            elif ("requests" in params) or ("req_matrix" in params) or ("R_out" in params) or ("R_ret" in params):
                R_out, R_ret = _aggregate_requests(params, T)
                arrival_minutes_out, arrival_minutes_ret = _extract_exact_arrival_minutes(params)
            else:
                R_out = [0.0] * T
                R_ret = [0.0] * T
                arrival_minutes_out, arrival_minutes_ret = (None, None)
            if len(R_out) != T:
                R_out = (R_out + [0.0] * T)[:T]
            if len(R_ret) != T:
                R_ret = (R_ret + [0.0] * T)[:T]

            # If using dual slopes, ensure at least one layer to create capacity constraints
            use_dual = bool(params.get("use_dual_slopes", False)) and (not temporal_refinement)
            K_out_lp = [max(1, int(K_out[t])) for t in range(T)] if use_dual else K_out
            K_ret_lp = [max(1, int(K_ret[t])) for t in range(T)] if use_dual else K_ret
            sp_params = SPParams(
                T=T, Wmax_slots=Wmax, p=p_pen, lp_solver=lp_solver, S=S,
                K_out=K_out_lp, K_ret=K_ret_lp, fill_eps=fill_eps,
                solver_options=solver_options, eps_cut=eps_cut,
                slot_resolution=slot_res, time_step_minutes=time_step_min,
                T_minutes=params.get("T_minutes"), trip_duration_minutes=params.get("trip_duration_minutes"),
                trip_slots=params.get("trip_slots"), Q=int(params.get("Q", len(q_idx) if q_idx else 0) or 0),
                binit=params.get("binit"), initial_actions=params.get("initial_actions"),
                Emax=params.get("Emax"), L=params.get("L"), delta_chg=params.get("delta_chg"),
                arrival_minutes_out=arrival_minutes_out, arrival_minutes_ret=arrival_minutes_ret,
                eps_feas=float(params.get("eps_feas", 1e-7)),
                debug_timing=debug_timing,
                debug_solver_tee=bool(params.get("debug_solver_tee", False)),
                debug_export_lp_iteration=params.get("debug_export_lp_iteration"),
                debug_current_iteration=int(params.get("debug_current_iteration", -1) or -1),
                debug_report_dir=params.get("debug_report_dir", params.get("report_dir", "Report")),
                debug_force_nominal_departures=bool(params.get("debug_force_nominal_departures", False)),
                debug_scenario_label="single",
                solve_time_limit_s=params.get("solve_time_limit_s"),
            )
            t_solve0 = time.perf_counter()
            if temporal_refinement:
                duals, ub_val = solve_refined_subproblem(sp_params, R_out, R_ret, candidate)
            else:
                duals, ub_val = solve_subproblem(sp_params, C_out, C_ret, R_out, R_ret, candidate)
            t_solve1 = time.perf_counter()
            sp_solve_time = t_solve1 - t_solve0
            _dbg(
                "[SP TIMING] iter=%s scenario=single solve_total=%.3fs build=%.3fs solve=%.3fs extract=%.3fs post=%.3fs cut_mode=%s"
                % (
                    str(params.get("debug_current_iteration", "-")),
                    float(sp_solve_time),
                    float(duals.get("timing_build_s", 0.0) or 0.0),
                    float(duals.get("timing_solve_s", 0.0) or 0.0),
                    float(duals.get("timing_extract_s", 0.0) or 0.0),
                    float(duals.get("timing_postprocess_s", 0.0) or 0.0),
                    refined_cut_mode,
                )
            )
            if not bool(duals.get("is_feasible", True)):
                diagnostics = {
                    "T": T,
                    "R_out": [float(R_out[t]) for t in range(T)],
                    "R_ret": [float(R_ret[t]) for t in range(T)],
                    "infeasible": True,
                    "infeasibility_reason": duals.get("infeasibility_reason"),
                    "first_violation": duals.get("first_violation"),
                    "slot_resolution": int(params.get("slot_resolution", 1)),
                    "timing_sp_solve_s": sp_solve_time,
                    "timing_cutgen_s": 0.0,
                }
                return SubproblemResult(is_feasible=False, upper_bound=None, diagnostics=diagnostics)

            if debug_skip_cut_generation:
                diagnostics = {
                    "T": T,
                    "R_out": [float(R_out[t]) for t in range(T)],
                    "R_ret": [float(R_ret[t]) for t in range(T)],
                    "pax_out_by_tau": list(duals.get("served_out_by_tau", [0.0] * T)),
                    "pax_ret_by_tau": list(duals.get("served_ret_by_tau", [0.0] * T)),
                    "pax_out_by_tau_k": list(duals.get("served_out_by_tau_k", [[] for _ in range(T)])),
                    "pax_ret_by_tau_k": list(duals.get("served_ret_by_tau_k", [[] for _ in range(T)])),
                    "objective_value": float(duals.get("objective_value", ub_val)),
                    "waiting_cost_slots": float(duals.get("waiting_cost_slots", 0.0)),
                    "fill_eps_cost": float(duals.get("fill_eps_cost", 0.0)),
                    "penalty_cost": float(duals.get("penalty_cost", 0.0)),
                    "penalty_pax": float(duals.get("penalty_pax", 0.0)),
                    "served_total": float(duals.get("served_total", 0.0)),
                    "total_demand": float(duals.get("total_demand", 0.0)),
                    "realized_departures": list(duals.get("realized_departures", [])),
                    "realized_departure_min_map": dict(duals.get("realized_departure_min_map", {})),
                    "refined_departure_diagnostics": list(duals.get("refined_departure_diagnostics", [])),
                    "refined_departure_diagnostics_focus": list(duals.get("refined_departure_diagnostics_focus", [])),
                    "effective_pre_service": list(duals.get("effective_pre_service", [])),
                    "battery_trajectory": duals.get("battery_trajectory", {}),
                    "slot_resolution": int(params.get("slot_resolution", 1)),
                    "timing_sp_solve_s": sp_solve_time,
                    "timing_cutgen_s": 0.0,
                    "cut_generation_mode": "skipped_debug",
                }
                return SubproblemResult(is_feasible=True, upper_bound=ub_val, diagnostics=diagnostics)

            # Early-exit if scalar theta is already consistent
            theta_val = _cand_theta("__theta")
            if (theta_val is not None) and _ok(theta_val, float(ub_val), eps_cut):
                diagnostics = {
                    "T": T,
                    "R_out": [float(R_out[t]) for t in range(T)],
                    "R_ret": [float(R_ret[t]) for t in range(T)],
                    "pax_out_by_tau": list(duals.get("served_out_by_tau", [0.0] * T)),
                    "pax_ret_by_tau": list(duals.get("served_ret_by_tau", [0.0] * T)),
                    "pax_out_by_tau_k": list(duals.get("served_out_by_tau_k", [[] for _ in range(T)])),
                    "pax_ret_by_tau_k": list(duals.get("served_ret_by_tau_k", [[] for _ in range(T)])),
                    "objective_value": float(duals.get("objective_value", ub_val)),
                    "waiting_cost_slots": float(duals.get("waiting_cost_slots", 0.0)),
                    "fill_eps_cost": float(duals.get("fill_eps_cost", 0.0)),
                    "penalty_cost": float(duals.get("penalty_cost", 0.0)),
                    "penalty_pax": float(duals.get("penalty_pax", 0.0)),
                    "served_total": float(duals.get("served_total", 0.0)),
                    "total_demand": float(duals.get("total_demand", 0.0)),
                    "realized_departures": list(duals.get("realized_departures", [])),
                    "realized_departure_min_map": dict(duals.get("realized_departure_min_map", {})),
                    "refined_departure_diagnostics": list(duals.get("refined_departure_diagnostics", [])),
                    "refined_departure_diagnostics_focus": list(duals.get("refined_departure_diagnostics_focus", [])),
                    "effective_pre_service": list(duals.get("effective_pre_service", [])),
                    "battery_trajectory": duals.get("battery_trajectory", {}),
                    "slot_resolution": int(params.get("slot_resolution", 1)),
                    "timing_sp_solve_s": sp_solve_time,
                    "timing_cutgen_s": 0.0,
                }
                return SubproblemResult(is_feasible=True, upper_bound=ub_val, diagnostics=diagnostics)

            # Build coefficients via MW, duals (fast) or finite differences (fallback)
            t_cut0 = time.perf_counter()
            cut_mode_used = refined_cut_mode if temporal_refinement else ("dual" if use_dual else "finite_difference")
            proxy_diag: dict[str, Any] = {}
            cut_lb_valid = not temporal_refinement
            if temporal_refinement and refined_cut_mode == "refined_lp_relaxation":
                cut_lp, _ = solve_refined_lp_relaxation_cut(sp_params, R_out, R_ret, candidate)
                c_out_map = dict(cut_lp.get("coeff_yOUT", {}))
                c_ret_map = dict(cut_lp.get("coeff_yRET", {}))
                dm_out = {int(t): sum(float(c_out_map.get((int(q), int(t)), 0.0)) for q in master_q_idx) for t in range(T)}
                dm_ret = {int(t): sum(float(c_ret_map.get((int(q), int(t)), 0.0)) for q in master_q_idx) for t in range(T)}
                proxy_diag = cut_lp
                cut_lb_valid = True
            elif temporal_refinement and refined_cut_mode in {"nominal_lp_proxy", "proxy_dual"}:
                c_out_map, c_ret_map, dm_out, dm_ret, proxy_diag = _proxy_cut_from_nominal_lp(sp_params, R_out, R_ret, ub_val)
                cut_lb_valid = False
            elif mw_enabled:
                dm_pair = solve_mw_dual(
                    T, Wmax, p_pen, S,
                    # Ensure at least one capacity layer per tau for dual π variables
                    [max(1, int(K_out_lp[t])) for t in range(T)],
                    [max(1, int(K_ret_lp[t])) for t in range(T)],
                    C_out, C_ret,
                    R_out, R_ret,
                    Ybar_out, Ybar_ret,
                    ub_val,
                    lp_solver,
                    solver_options,
                )
                if dm_pair is None:
                    # Fallback to finite differences to guarantee nonzero slopes
                    c_out_fd, c_ret_fd, dm_out, dm_ret = coeffs_by_fdiff(ub_val, C_out, C_ret, K_out, K_ret, R_out, R_ret)
                else:
                    dm_out, dm_ret = dm_pair
                # Expand to per-(q,t)
                c_out_map: Dict[tuple[int, int], float] = {}
                c_ret_map: Dict[tuple[int, int], float] = {}
                for name in candidate.keys():
                    if not isinstance(name, str):
                        continue
                    if name.startswith("yOUT["):
                        inside = name[name.find("[") + 1 : name.find("]")]
                        q_str, tau_str = inside.split(",")
                        q = int(q_str.strip()); tau = int(tau_str.strip())
                        if 0 <= tau < T:
                            c_out_map[(q, tau)] = dm_out.get(tau, 0.0)
                    elif name.startswith("yRET["):
                        inside = name[name.find("[") + 1 : name.find("]")]
                        q_str, tau_str = inside.split(",")
                        q = int(q_str.strip()); tau = int(tau_str.strip())
                        if 0 <= tau < T:
                            c_ret_map[(q, tau)] = dm_ret.get(tau, 0.0)
            elif use_dual:
                # Read duals π on capacity layers and aggregate by time tau
                pi_out = dict(duals.get("pi_OUT", {}))
                pi_ret = dict(duals.get("pi_RET", {}))
                # Slopes dm[t] = S * π[t] (typically <= 0 in minimization; more capacity reduces cost)
                dm_out = {int(t): float(S) * float(pi_out.get(int(t), 0.0)) for t in range(T)}
                dm_ret = {int(t): float(S) * float(pi_ret.get(int(t), 0.0)) for t in range(T)}
                # Expand to per-(q,t)
                c_out_map: Dict[tuple[int, int], float] = {}
                c_ret_map: Dict[tuple[int, int], float] = {}
                for name in candidate.keys():
                    if not isinstance(name, str):
                        continue
                    if name.startswith("yOUT["):
                        inside = name[name.find("[") + 1 : name.find("]")]
                        q_str, tau_str = inside.split(",")
                        q = int(q_str.strip()); tau = int(tau_str.strip())
                        if 0 <= tau < T:
                            c_out_map[(q, tau)] = dm_out.get(tau, 0.0)
                    elif name.startswith("yRET["):
                        inside = name[name.find("[") + 1 : name.find("]")]
                        q_str, tau_str = inside.split(",")
                        q = int(q_str.strip()); tau = int(tau_str.strip())
                        if 0 <= tau < T:
                            c_ret_map[(q, tau)] = dm_ret.get(tau, 0.0)
            else:
                # Finite differences fallback
                c_out_map, c_ret_map, dm_out, dm_ret = coeffs_by_fdiff(ub_val, C_out, C_ret, K_out, K_ret, R_out, R_ret)
                cut_lb_valid = False
            # Proxy cut first; if it degenerates, escalate to a sparse restricted
            # finite-difference fallback on promising slots rather than probing all slots.
            deg_tol = float(params.get("degenerate_cut_zero_tol", 1e-9) or 1e-9)
            fallback_diag: dict[str, Any] = {}
            if temporal_refinement and (not cut_lb_valid) and _is_degenerate_cut(float(ub_val), c_out_map, c_ret_map, deg_tol):
                top_k_out = int(params.get("degenerate_cut_probe_top_k_out", params.get("degenerate_cut_probe_top_k", 6)))
                top_k_ret = int(params.get("degenerate_cut_probe_top_k_ret", params.get("degenerate_cut_probe_top_k", 6)))
                c_out_map, c_ret_map, dm_out, dm_ret, fallback_diag = _restricted_temporal_fdiff(ub_val, R_out, R_ret, top_k_out, top_k_ret)
                if _is_degenerate_cut(float(ub_val), c_out_map, c_ret_map, deg_tol):
                    if _candidate_is_all_idle():
                        diagnostics = {
                            "T": T,
                            "R_out": [float(R_out[t]) for t in range(T)],
                            "R_ret": [float(R_ret[t]) for t in range(T)],
                            "objective_value": float(duals.get("objective_value", ub_val)),
                            "waiting_cost_slots": float(duals.get("waiting_cost_slots", 0.0)),
                            "fill_eps_cost": float(duals.get("fill_eps_cost", 0.0)),
                            "penalty_cost": float(duals.get("penalty_cost", 0.0)),
                            "penalty_pax": float(duals.get("penalty_pax", 0.0)),
                            "served_total": float(duals.get("served_total", 0.0)),
                            "total_demand": float(duals.get("total_demand", 0.0)),
                            "slot_resolution": int(params.get("slot_resolution", 1)),
                            "timing_sp_solve_s": sp_solve_time,
                            "timing_cutgen_s": time.perf_counter() - t_cut0,
                            "cut_generation_mode": "anti_trivial_idle_fallback",
                            "cut_generation_proxy": proxy_diag,
                            "cut_generation_fallback": fallback_diag,
                        }
                        return SubproblemResult(
                            is_feasible=True,
                            cut=_build_anti_trivial_cut("degenerate_proxy_after_restricted_fdiff"),
                            upper_bound=ub_val,
                            diagnostics=diagnostics,
                        )
                cut_mode_used = "restricted_finite_difference_fallback"

            t_cut1 = time.perf_counter()
            cutgen_time = t_cut1 - t_cut0
            _dbg(
                "[SP TIMING] iter=%s scenario=single cutgen=%.3fs mode=%s proxy_obj=%s"
                % (
                    str(params.get("debug_current_iteration", "-")),
                    float(cutgen_time),
                    cut_mode_used,
                    (
                        f"{float(proxy_diag.get('proxy_recourse_objective')):.6g}"
                        if proxy_diag.get("proxy_recourse_objective") is not None
                        else "-"
                    ),
                )
            )

            # Number of vehicles per departure time from current candidate
            sum_y_out = [float(C_out[tau]) / S if S != 0 else 0.0 for tau in range(T)]
            sum_y_ret = [float(C_ret[tau]) / S if S != 0 else 0.0 for tau in range(T)]

            # Intercept (const) so that the cut passes through current incumbent: const = Q(y) - dm·Y
            if temporal_refinement and refined_cut_mode == "refined_lp_relaxation":
                const = float(proxy_diag.get("const", 0.0))
            else:
                const = float(ub_val)
                const -= sum(dm_out.get(t, 0.0) * sum_y_out[t] for t in range(T))
                const -= sum(dm_ret.get(t, 0.0) * sum_y_ret[t] for t in range(T))

            # Directional intercepts if available from decomposition diagnostics
            if temporal_refinement and refined_cut_mode == "refined_lp_relaxation":
                const_out = float(proxy_diag.get("const_out", const))
                const_ret = float(proxy_diag.get("const_ret", 0.0))
            else:
                try:
                    ub_out = float(duals.get("ub_out", const))
                except Exception:
                    ub_out = float(const)
                try:
                    ub_ret = float(duals.get("ub_ret", 0.0))
                except Exception:
                    ub_ret = 0.0
                const_out = float(ub_out) - sum(dm_out.get(t, 0.0) * sum_y_out[t] for t in range(T))
                const_ret = float(ub_ret) - sum(dm_ret.get(t, 0.0) * sum_y_ret[t] for t in range(T))

            # Optional: evaluate the line at the incumbent (theta_lb) to verify tightness (≈ ub_val)
            theta_lb = float(const) + sum(float(v) * _cand_float(candidate, f"yOUT[{int(q)},{int(tau)}]", 0.0) for (q, tau), v in c_out_map.items()) \
                + sum(float(v) * _cand_float(candidate, f"yRET[{int(q)},{int(tau)}]", 0.0) for (q, tau), v in c_ret_map.items())
            target_val = float(proxy_diag.get("objective_value", ub_val)) if temporal_refinement and refined_cut_mode == "refined_lp_relaxation" else float(ub_val)
            if abs(float(target_val) - float(theta_lb)) > eps_cut * max(1.0, abs(float(target_val))):
                raise RuntimeError("Cut tightness failed at incumbent; aborting cut generation.")

            # Emit cut metadata
            cut = Cut(
                name="opt_cut",
                cut_type=CutType.OPTIMALITY,
                metadata={
                    "const": float(const),
                    "const_out": float(const_out),
                    "const_ret": float(const_ret),
                    "coeff_yOUT": c_out_map,
                    "coeff_yRET": c_ret_map,
                    "recourse_total": float(target_val),
                    "recourse_out": float(duals.get("ub_out", 0.0)),
                    "recourse_ret": float(duals.get("ub_ret", 0.0)),
                    # diagnostics
                    "theta_lb": float(theta_lb),
                    "cut_valid_lower_bound": bool(cut_lb_valid),
                },
            )
            # Diagnostics: demand per slot split by direction and served pax per departure slot (OUT/RET)
            diagnostics = {
                "T": T,
                "R_out": [float(R_out[t]) for t in range(T)],
                "R_ret": [float(R_ret[t]) for t in range(T)],
                "pax_out_by_tau": list(duals.get("served_out_by_tau", [0.0] * T)),
                "pax_ret_by_tau": list(duals.get("served_ret_by_tau", [0.0] * T)),
                "pax_out_by_tau_k": list(duals.get("served_out_by_tau_k", [[] for _ in range(T)])),
                "pax_ret_by_tau_k": list(duals.get("served_ret_by_tau_k", [[] for _ in range(T)])),
                "objective_value": float(duals.get("objective_value", ub_val)),
                "waiting_cost_slots": float(duals.get("waiting_cost_slots", 0.0)),
                "fill_eps_cost": float(duals.get("fill_eps_cost", 0.0)),
                "penalty_cost": float(duals.get("penalty_cost", 0.0)),
                "penalty_pax": float(duals.get("penalty_pax", 0.0)),
                "served_total": float(duals.get("served_total", 0.0)),
                "total_demand": float(duals.get("total_demand", 0.0)),
                "realized_departures": list(duals.get("realized_departures", [])),
                "realized_departure_min_map": dict(duals.get("realized_departure_min_map", {})),
                "refined_departure_diagnostics": list(duals.get("refined_departure_diagnostics", [])),
                "refined_departure_diagnostics_focus": list(duals.get("refined_departure_diagnostics_focus", [])),
                "effective_pre_service": list(duals.get("effective_pre_service", [])),
                "battery_trajectory": duals.get("battery_trajectory", {}),
                "slot_resolution": int(params.get("slot_resolution", 1)),
                "timing_sp_solve_s": sp_solve_time,
                "timing_cutgen_s": cutgen_time,
                "cut_generation_mode": cut_mode_used,
                "cut_generation_proxy": proxy_diag,
                "cut_generation_fallback": fallback_diag,
                "cut_valid_lower_bound": bool(cut_lb_valid),
            }
            return SubproblemResult(is_feasible=True, cut=cut, upper_bound=ub_val, diagnostics=diagnostics)

@dataclass
class RefinedServiceEvent:
    sid: int
    q: int
    tau: int
    direction: str
    nominal_min: int
    window_lb_min: int
    window_ub_min: int
    prev_sid: Optional[int]
    layer_index: int


def _cand_float(candidate: Candidate | None, name: str, default: float = 0.0) -> float:
    if candidate is None:
        return default
    try:
        return float(candidate.get(name, default))
    except Exception:
        return default


def _normalize_binit(P: "SPParams") -> list[float]:
    raw = P.binit
    if raw is None:
        return [0.0] * max(0, int(P.Q))
    if isinstance(raw, (int, float)):
        return [float(raw)] * max(0, int(P.Q))
    vals = [float(x) for x in list(raw)]
    if len(vals) < int(P.Q):
        fill = vals[-1] if vals else 0.0
        vals.extend([fill] * (int(P.Q) - len(vals)))
    return vals[: int(P.Q)]


def _report_dir_from_debug(path_like: Any) -> Path:
    try:
        p = Path(str(path_like)) if path_like else Path("Report")
    except Exception:
        p = Path("Report")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _model_size_stats(model: pyo.ConcreteModel) -> dict[str, int]:
    num_vars = 0
    num_bin = 0
    for var in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        num_vars += 1
        try:
            if var.is_binary():
                num_bin += 1
        except Exception:
            pass
    num_cons = sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True, descend_into=True))
    return {
        "model_num_variables": int(num_vars),
        "model_num_binary_variables": int(num_bin),
        "model_num_constraints": int(num_cons),
    }


def _apply_solver_time_limit(solver: Any, seconds: float | None) -> None:
    if seconds is None:
        return
    try:
        limit = float(seconds)
    except Exception:
        return
    if not math.isfinite(limit) or limit <= 0.0:
        return
    try:
        solver.options["timelimit"] = limit
    except Exception:
        pass


def _has_loaded_solution(model: pyo.ConcreteModel) -> bool:
    for var in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        try:
            if var.value is not None:
                return True
        except Exception:
            continue
    return False


def _maybe_export_lp(model: pyo.ConcreteModel, P: "SPParams") -> tuple[Optional[str], float]:
    export_iter = getattr(P, "debug_export_lp_iteration", None)
    current_iter = int(getattr(P, "debug_current_iteration", -1) or -1)
    if export_iter is None or current_iter < 0 or int(export_iter) != current_iter:
        return None, 0.0
    out_dir = _report_dir_from_debug(getattr(P, "debug_report_dir", None))
    label = str(getattr(P, "debug_scenario_label", "single") or "single")
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    lp_path = out_dir / f"{model.name}_iter_{current_iter}_{safe_label}.lp"
    t0 = time.perf_counter()
    model.write(str(lp_path), io_options={"symbolic_solver_labels": True})
    t1 = time.perf_counter()
    return str(lp_path), float(t1 - t0)


def _charge_profile(candidate: Candidate | None, P: "SPParams") -> dict[tuple[int, int], float]:
    prof: dict[tuple[int, int], float] = {}
    for q in range(int(P.Q)):
        for t in range(int(P.T)):
            v = _cand_float(candidate, f"c[{q},{t}]", 0.0)
            if abs(v) > 1e-9:
                prof[(q, t)] = max(0.0, min(1.0, v))
    return prof


def _service_slot_window_lb_min(tau: int, slot_res: int) -> int:
    tau = int(tau)
    if tau < 1:
        raise AssertionError(f"Invalid service slot index tau={tau}; service starts must satisfy tau >= 1")
    return (tau - 1) * int(slot_res) + 1


def _service_slot_window_ub_min(tau: int, slot_res: int) -> int:
    tau = int(tau)
    if tau < 1:
        raise AssertionError(f"Invalid service slot index tau={tau}; service starts must satisfy tau >= 1")
    return tau * int(slot_res)


def _slot_nominal_departure_min(tau: int, slot_res: int) -> int:
    return _service_slot_window_ub_min(int(tau), int(slot_res))


def _demand_release_min(t: int, slot_res: int) -> int:
    return int(t) * int(slot_res) + 1


def _wait_slots_from_minute(t: int, dep_min: int, slot_res: int) -> float:
    release_min = _demand_release_min(int(t), int(slot_res))
    return float(max(0, int(dep_min) - release_min)) / float(max(1, int(slot_res)))


def _slot_index_from_minute(arrival_min: float, slot_res: int, T: int) -> int | None:
    try:
        minute = float(arrival_min)
    except Exception:
        return None
    if minute < 0.0:
        return None
    if T <= 0:
        return None
    slot = int(math.floor(minute / float(max(1, int(slot_res)))))
    return max(0, min(int(T) - 1, slot))


def _extract_exact_arrival_minutes(container: Any) -> tuple[list[float] | None, list[float] | None]:
    if container is None:
        return None, None
    if isinstance(container, dict):
        if "requests" in container:
            container = container.get("requests") or []
        elif "req_matrix" in container:
            container = container.get("req_matrix") or []
        else:
            return None, None
    if not isinstance(container, list):
        return None, None

    out_minutes: list[float] = []
    ret_minutes: list[float] = []
    if container and isinstance(container[0], dict):
        for req in container:
            try:
                minute = float(req.get("time", -1))
            except Exception:
                continue
            if minute < 0.0:
                continue
            direction = req.get("dir")
            if isinstance(direction, str):
                dd = direction.upper()
                if dd == "OUT":
                    out_minutes.append(minute)
                elif dd == "RET":
                    ret_minutes.append(minute)
            else:
                try:
                    if int(direction) == 0:
                        out_minutes.append(minute)
                    else:
                        ret_minutes.append(minute)
                except Exception:
                    continue
        return out_minutes, ret_minutes
    for row in container:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        direction, raw_minute = row[0], row[1]
        try:
            minute = float(raw_minute)
        except Exception:
            continue
        if minute < 0.0:
            continue
        if isinstance(direction, str):
            dd = direction.upper()
            if dd == "OUT":
                out_minutes.append(minute)
            elif dd == "RET":
                ret_minutes.append(minute)
        else:
            try:
                if int(direction) == 0:
                    out_minutes.append(minute)
                else:
                    ret_minutes.append(minute)
            except Exception:
                continue
    return out_minutes, ret_minutes


def _eligible_arrivals_by_slot_and_cutoff(
    R: list[float],
    arrival_minutes: list[float] | None,
    T: int,
    slot_res: int,
    cutoffs_by_slot: Mapping[int, Iterable[int]],
) -> dict[tuple[int, int], float]:
    eligible: dict[tuple[int, int], float] = {}
    if arrival_minutes is None:
        for t, cutoffs in cutoffs_by_slot.items():
            full_available_min = (int(t) + 1) * int(slot_res)
            demand = float(R[t]) if 0 <= int(t) < len(R) else 0.0
            for cutoff in cutoffs:
                eligible[(int(t), int(cutoff))] = demand if int(cutoff) >= full_available_min else 0.0
        return eligible

    arrivals_by_slot: list[list[float]] = [[] for _ in range(max(0, int(T)))]
    for minute in arrival_minutes:
        slot = _slot_index_from_minute(minute, slot_res, T)
        if slot is None:
            continue
        arrivals_by_slot[slot].append(float(minute))
    for arrs in arrivals_by_slot:
        arrs.sort()

    for t, cutoffs in cutoffs_by_slot.items():
        arrs = arrivals_by_slot[int(t)] if 0 <= int(t) < len(arrivals_by_slot) else []
        for cutoff in cutoffs:
            eligible[(int(t), int(cutoff))] = float(bisect_right(arrs, float(cutoff)))
    return eligible


def _direction_eligible_arrivals_by_minute(
    R: list[float],
    arrival_minutes: list[float] | None,
    T: int,
    slot_res: int,
    minute: float,
) -> float:
    cutoff = float(minute)
    if arrival_minutes is not None:
        clean = sorted(float(v) for v in arrival_minutes if v is not None)
        return float(bisect_right(clean, cutoff))
    total = 0.0
    for t in range(min(int(T), len(R))):
        if cutoff >= float((t + 1) * int(slot_res)):
            total += float(R[t])
    return total


def _validate_realized_departure_minutes(
    events: list[RefinedServiceEvent],
    dep_by_sid: Mapping[int, float],
    slot_res: int,
) -> None:
    for event in events:
        if int(event.tau) < 1:
            raise AssertionError(
                f"Invalid service start slot q={event.q} tau={event.tau}; service starts must satisfy tau >= 1"
            )
        try:
            dep = float(dep_by_sid[event.sid])
        except Exception as exc:
            raise AssertionError(f"Missing realized departure for event sid={event.sid}") from exc
        lb = _service_slot_window_lb_min(int(event.tau), int(slot_res))
        ub = _service_slot_window_ub_min(int(event.tau), int(slot_res))
        if dep < float(lb) - 1.0e-9 or dep > float(ub) + 1.0e-9:
            raise AssertionError(
                "Realized departure outside slot window: "
                f"q={event.q} tau={event.tau} dep={dep:.6g} expected in [{lb}, {ub}]"
            )


def _build_service_events(candidate: Candidate | None, P: "SPParams") -> list[RefinedServiceEvent]:
    slot_res = max(1, int(P.slot_resolution))
    events: list[RefinedServiceEvent] = []
    per_tau_dir_count: dict[tuple[str, int], int] = {}
    for q in range(int(P.Q)):
        prev_sid: Optional[int] = None
        for tau in range(int(P.T)):
            direction = ""
            if _cand_float(candidate, f"yOUT[{q},{tau}]", 0.0) >= 0.5:
                direction = "OUT"
            elif _cand_float(candidate, f"yRET[{q},{tau}]", 0.0) >= 0.5:
                direction = "RET"
            if not direction:
                continue
            if int(tau) < 1:
                raise AssertionError(
                    f"Invalid service start slot q={q} tau={tau}; slot 0 is a demand bucket, not a departure slot"
                )
            nominal = _slot_nominal_departure_min(int(tau), slot_res)
            key = (direction, int(tau))
            layer_index = per_tau_dir_count.get(key, 0)
            per_tau_dir_count[key] = layer_index + 1
            events.append(
                RefinedServiceEvent(
                    sid=len(events),
                    q=int(q),
                    tau=int(tau),
                    direction=direction,
                    nominal_min=nominal,
                    window_lb_min=_service_slot_window_lb_min(int(tau), slot_res),
                    window_ub_min=_service_slot_window_ub_min(int(tau), slot_res),
                    prev_sid=prev_sid,
                    layer_index=layer_index,
                )
            )
            prev_sid = events[-1].sid
    return events


def _charge_minutes_between(
    q: int,
    start_min: float,
    end_min: float,
    charge_prof: dict[tuple[int, int], float],
    slot_res: int,
    T: int,
) -> float:
    if end_min <= start_min:
        return 0.0
    total = 0.0
    for tau in range(T):
        frac = float(charge_prof.get((q, tau), 0.0))
        if frac <= 0.0:
            continue
        lo = float(tau * slot_res)
        hi = float((tau + 1) * slot_res)
        overlap = max(0.0, min(end_min, hi) - max(start_min, lo))
        if overlap > 0.0:
            total += frac * overlap
    return total


def _first_violation(
    events: list[RefinedServiceEvent],
    charge_prof: dict[tuple[int, int], float],
    P: "SPParams",
) -> dict[str, Any] | None:
    slot_res = max(1, int(P.slot_resolution))
    trip_minutes = int(P.trip_duration_minutes or (int(P.trip_slots or 0) * slot_res))
    charge_rate = float(P.delta_chg or 0.0) / float(slot_res)
    emax = float(P.Emax) if P.Emax is not None else float("inf")
    binit = _normalize_binit(P)
    by_vehicle: dict[int, list[RefinedServiceEvent]] = {}
    for event in events:
        by_vehicle.setdefault(event.q, []).append(event)
    for q, seq in sorted(by_vehicle.items()):
        batt = float(binit[q] if q < len(binit) else 0.0)
        prev_dep = 0.0
        prev_event: Optional[RefinedServiceEvent] = None
        for event in seq:
            dep = float(event.window_ub_min)
            if prev_event is not None:
                req = prev_dep + trip_minutes
                if dep + float(P.eps_feas) < req:
                    return {
                        "type": "timing",
                        "vehicle_id": q,
                        "predecessor_slot": prev_event.tau,
                        "current_slot": event.tau,
                        "required_min_time": req,
                        "available_realized_time": dep,
                    }
            charge_min = _charge_minutes_between(q, prev_dep + trip_minutes if prev_event is not None else 0.0, dep, charge_prof, slot_res, int(P.T))
            batt = min(emax, batt + charge_rate * charge_min)
            if batt + float(P.eps_feas) < float(P.L or 0.0):
                return {
                    "type": "battery",
                    "vehicle_id": q,
                    "predecessor_slot": prev_event.tau if prev_event is not None else None,
                    "current_slot": event.tau,
                    "required_min_charge": float(P.L or 0.0),
                    "available_realized_charge": batt,
                }
            batt -= float(P.L or 0.0)
            prev_dep = dep
            prev_event = event
    return None


def solve_refined_lp_relaxation_cut(
    P: "SPParams",
    R_out: Iterable[float],
    R_ret: Iterable[float],
    candidate: Candidate | None = None,
):
    """Solve a valid LP lower-bounding recourse model parameterized by master y.

    This is not the refined MILP recourse itself. It is a relaxation that keeps
    the time-within-slot activation structure and passenger assignment dependence
    on the master schedule, while relaxing temporal refinement choices to
    continuous convex combinations. Because it is an LP with y only on RHS
    activation constraints, its duals define globally valid lower-bounding cuts.
    """
    t_build0 = time.perf_counter()
    m = pyo.ConcreteModel()
    m.name = "subproblem_refined_lp_relax"
    R_out = [float(v) for v in R_out]
    R_ret = [float(v) for v in R_ret]
    slot_res = max(1, int(P.slot_resolution))
    step = max(1, int(P.time_step_minutes))
    W_minutes = int(P.Wmax_slots) * slot_res

    Tset = range(int(P.T))
    service_slots = range(1, int(P.T))
    Qset = range(int(P.Q))
    out_events = [(int(q), int(tau)) for q in Qset for tau in service_slots]
    ret_events = [(int(q), int(tau)) for q in Qset for tau in service_slots]

    for q in Qset:
        if _cand_float(candidate, f"yOUT[{int(q)},0]", 0.0) >= 0.5 or _cand_float(candidate, f"yRET[{int(q)},0]", 0.0) >= 0.5:
            raise AssertionError(
                f"Invalid service start slot q={int(q)} tau=0; slot 0 is a demand bucket, not a departure slot"
            )

    out_choices = []
    ret_choices = []
    for q, tau in out_events:
        nominal = _slot_nominal_departure_min(int(tau), slot_res)
        lb = _service_slot_window_lb_min(int(tau), slot_res)
        for h in range(lb, nominal + 1, step):
            out_choices.append((q, tau, int(h)))
    for q, tau in ret_events:
        nominal = _slot_nominal_departure_min(int(tau), slot_res)
        lb = _service_slot_window_lb_min(int(tau), slot_res)
        for h in range(lb, nominal + 1, step):
            ret_choices.append((q, tau, int(h)))

    out_choice_map: dict[tuple[int, int], list[int]] = {}
    ret_choice_map: dict[tuple[int, int], list[int]] = {}
    for q, tau, h in out_choices:
        out_choice_map.setdefault((q, tau), []).append(h)
    for q, tau, h in ret_choices:
        ret_choice_map.setdefault((q, tau), []).append(h)

    m.OutChoices = pyo.Set(initialize=out_choices, dimen=3, ordered=False)
    m.RetChoices = pyo.Set(initialize=ret_choices, dimen=3, ordered=False)
    m.z_OUT = pyo.Var(m.OutChoices, bounds=(0.0, 1.0))
    m.z_RET = pyo.Var(m.RetChoices, bounds=(0.0, 1.0))

    y_out_rhs = {(int(q), int(tau)): _cand_float(candidate, f"yOUT[{int(q)},{int(tau)}]", 0.0) for q in Qset for tau in service_slots}
    y_ret_rhs = {(int(q), int(tau)): _cand_float(candidate, f"yRET[{int(q)},{int(tau)}]", 0.0) for q in Qset for tau in service_slots}

    def _act_out(mm, q, tau):
        return sum(mm.z_OUT[q, tau, h] for h in out_choice_map[(int(q), int(tau))]) == float(y_out_rhs[(int(q), int(tau))])
    m.Act_OUT = pyo.Constraint(out_events, rule=_act_out)

    def _act_ret(mm, q, tau):
        return sum(mm.z_RET[q, tau, h] for h in ret_choice_map[(int(q), int(tau))]) == float(y_ret_rhs[(int(q), int(tau))])
    m.Act_RET = pyo.Constraint(ret_events, rule=_act_ret)

    out_arcs = []
    for t, dem in enumerate(R_out):
        if dem <= 0.0:
            continue
        release = _demand_release_min(int(t), slot_res)
        for q, tau, h in out_choices:
            if release <= h <= release + W_minutes:
                out_arcs.append((int(t), int(q), int(tau), int(h)))
    ret_arcs = []
    for t, dem in enumerate(R_ret):
        if dem <= 0.0:
            continue
        release = _demand_release_min(int(t), slot_res)
        for q, tau, h in ret_choices:
            if release <= h <= release + W_minutes:
                ret_arcs.append((int(t), int(q), int(tau), int(h)))

    out_cutoffs_by_t: dict[int, set[int]] = {}
    ret_cutoffs_by_t: dict[int, set[int]] = {}
    for t, _q, _tau, h in out_arcs:
        out_cutoffs_by_t.setdefault(int(t), set()).add(int(h))
    for t, _q, _tau, h in ret_arcs:
        ret_cutoffs_by_t.setdefault(int(t), set()).add(int(h))
    eligible_out_prefix = _eligible_arrivals_by_slot_and_cutoff(
        R_out,
        getattr(P, "arrival_minutes_out", None),
        int(P.T),
        slot_res,
        out_cutoffs_by_t,
    )
    eligible_ret_prefix = _eligible_arrivals_by_slot_and_cutoff(
        R_ret,
        getattr(P, "arrival_minutes_ret", None),
        int(P.T),
        slot_res,
        ret_cutoffs_by_t,
    )

    m.OutArcs = pyo.Set(initialize=out_arcs, dimen=4, ordered=False)
    m.RetArcs = pyo.Set(initialize=ret_arcs, dimen=4, ordered=False)
    m.x_OUT = pyo.Var(m.OutArcs, within=pyo.NonNegativeReals)
    m.x_RET = pyo.Var(m.RetArcs, within=pyo.NonNegativeReals)
    m.u_OUT = pyo.Var(Tset, within=pyo.NonNegativeReals)
    m.u_RET = pyo.Var(Tset, within=pyo.NonNegativeReals)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    out_arc_index: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    ret_arc_index: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for arc in out_arcs:
        out_arc_index.setdefault((int(arc[1]), int(arc[2])), []).append(arc)
    for arc in ret_arcs:
        ret_arc_index.setdefault((int(arc[1]), int(arc[2])), []).append(arc)

    def _wait_slots(t: int, h: int) -> float:
        return _wait_slots_from_minute(int(t), int(h), slot_res)

    m.obj = pyo.Objective(
        expr=
        sum(_wait_slots(t, h) * m.x_OUT[t, q, tau, h] for (t, q, tau, h) in m.OutArcs)
        + sum(_wait_slots(t, h) * m.x_RET[t, q, tau, h] for (t, q, tau, h) in m.RetArcs)
        + float(P.p) * (sum(m.u_OUT[t] for t in Tset) + sum(m.u_RET[t] for t in Tset)),
        sense=pyo.minimize,
    )

    def _dem_out(mm, t):
        rel = [(q, tau, h) for (tt, q, tau, h) in mm.OutArcs if tt == t]
        return sum(mm.x_OUT[t, q, tau, h] for (q, tau, h) in rel) + mm.u_OUT[t] == float(R_out[t])
    m.D_out_relax = pyo.Constraint(Tset, rule=_dem_out)

    def _dem_ret(mm, t):
        rel = [(q, tau, h) for (tt, q, tau, h) in mm.RetArcs if tt == t]
        return sum(mm.x_RET[t, q, tau, h] for (q, tau, h) in rel) + mm.u_RET[t] == float(R_ret[t])
    m.D_ret_relax = pyo.Constraint(Tset, rule=_dem_ret)

    def _gate_out(mm, t, q, tau, h):
        return mm.x_OUT[t, q, tau, h] <= float(R_out[t]) * mm.z_OUT[q, tau, h]
    m.Gate_OUT = pyo.Constraint(m.OutArcs, rule=_gate_out)

    def _gate_ret(mm, t, q, tau, h):
        return mm.x_RET[t, q, tau, h] <= float(R_ret[t]) * mm.z_RET[q, tau, h]
    m.Gate_RET = pyo.Constraint(m.RetArcs, rule=_gate_ret)

    def _cap_out(mm, q, tau):
        rel = out_arc_index.get((int(q), int(tau)), [])
        if not rel:
            return pyo.Constraint.Skip
        return sum(mm.x_OUT[t, q, tau, h] for (t, q, tau, h) in rel) <= float(P.S) * sum(mm.z_OUT[q, tau, h] for h in out_choice_map[(int(q), int(tau))])
    m.Cap_OUT = pyo.Constraint(out_events, rule=_cap_out)

    def _cap_ret(mm, q, tau):
        rel = ret_arc_index.get((int(q), int(tau)), [])
        if not rel:
            return pyo.Constraint.Skip
        return sum(mm.x_RET[t, q, tau, h] for (t, q, tau, h) in rel) <= float(P.S) * sum(mm.z_RET[q, tau, h] for h in ret_choice_map[(int(q), int(tau))])
    m.Cap_RET = pyo.Constraint(ret_events, rule=_cap_ret)

    for t, cutoffs in out_cutoffs_by_t.items():
        for cutoff in sorted(cutoffs):
            rel = [(q, tau, h) for (tt, q, tau, h) in out_arcs if int(tt) == int(t) and int(h) <= int(cutoff)]
            if not rel:
                continue
            eligible = float(eligible_out_prefix.get((int(t), int(cutoff)), 0.0))
            m.add_component(
                f"EligPrefixOut_{int(t)}_{int(cutoff)}",
                pyo.Constraint(expr=sum(m.x_OUT[t, q, tau, h] for (q, tau, h) in rel) <= eligible),
            )
    for t, cutoffs in ret_cutoffs_by_t.items():
        for cutoff in sorted(cutoffs):
            rel = [(q, tau, h) for (tt, q, tau, h) in ret_arcs if int(tt) == int(t) and int(h) <= int(cutoff)]
            if not rel:
                continue
            eligible = float(eligible_ret_prefix.get((int(t), int(cutoff)), 0.0))
            m.add_component(
                f"EligPrefixRet_{int(t)}_{int(cutoff)}",
                pyo.Constraint(expr=sum(m.x_RET[t, q, tau, h] for (q, tau, h) in rel) <= eligible),
            )

    build_stats = _model_size_stats(m)
    lp_path = None
    lp_export_time = 0.0
    try:
        lp_path, lp_export_time = _maybe_export_lp(m, P)
    except Exception:
        lp_path, lp_export_time = (None, 0.0)
    t_build1 = time.perf_counter()

    solver = pyo.SolverFactory(P.lp_solver)
    try:
        if getattr(P, "solver_options", None):
            for k, v in (P.solver_options or {}).items():
                solver.options[k] = v
    except Exception:
        pass
    _apply_solver_time_limit(solver, getattr(P, "solve_time_limit_s", None))
    t_solve0 = time.perf_counter()
    res = solver.solve(m, tee=bool(getattr(P, "debug_solver_tee", False)), load_solutions=False)
    t_solve1 = time.perf_counter()
    term = getattr(res.solver, "termination_condition", None)
    if term not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible):
        raise RuntimeError(f"Refined LP-relaxation solve failed: termination_condition={term}")
    try:
        m.solutions.load_from(res)
    except Exception:
        pass
    t_extract0 = time.perf_counter()

    coeff_y_out = {(int(q), int(tau)): 0.0 for q in Qset for tau in Tset}
    coeff_y_ret = {(int(q), int(tau)): 0.0 for q in Qset for tau in Tset}
    for (q, tau) in out_events:
        coeff_y_out[(int(q), int(tau))] = float(m.dual.get(m.Act_OUT[q, tau], 0.0))
    for (q, tau) in ret_events:
        coeff_y_ret[(int(q), int(tau))] = float(m.dual.get(m.Act_RET[q, tau], 0.0))

    y_out_sum = sum(float(coeff_y_out[(int(q), int(tau))]) * float(_cand_float(candidate, f"yOUT[{int(q)},{int(tau)}]", 0.0)) for q in Qset for tau in Tset)
    y_ret_sum = sum(float(coeff_y_ret[(int(q), int(tau))]) * float(_cand_float(candidate, f"yRET[{int(q)},{int(tau)}]", 0.0)) for q in Qset for tau in Tset)
    obj_val = float(pyo.value(m.obj))
    const = float(obj_val) - float(y_out_sum) - float(y_ret_sum)

    out_cost = sum(_wait_slots(t, h) * float(m.x_OUT[t, q, tau, h].value or 0.0) for (t, q, tau, h) in out_arcs)
    out_cost += float(P.p) * sum(float(m.u_OUT[t].value or 0.0) for t in Tset)
    ret_cost = sum(_wait_slots(t, h) * float(m.x_RET[t, q, tau, h].value or 0.0) for (t, q, tau, h) in ret_arcs)
    ret_cost += float(P.p) * sum(float(m.u_RET[t].value or 0.0) for t in Tset)
    const_out = float(out_cost) - float(y_out_sum)
    const_ret = float(ret_cost) - float(y_ret_sum)

    served_out_by_tau = [0.0 for _ in Tset]
    served_ret_by_tau = [0.0 for _ in Tset]
    for (t, q, tau, h) in out_arcs:
        served_out_by_tau[int(tau)] += float(m.x_OUT[t, q, tau, h].value or 0.0)
    for (t, q, tau, h) in ret_arcs:
        served_ret_by_tau[int(tau)] += float(m.x_RET[t, q, tau, h].value or 0.0)
    penalty_pax = float(sum(float(m.u_OUT[t].value or 0.0) for t in Tset) + sum(float(m.u_RET[t].value or 0.0) for t in Tset))
    wait_cost_slots = float(out_cost + ret_cost - float(P.p) * penalty_pax)
    t_extract1 = time.perf_counter()

    return (
        {
            "objective_value": obj_val,
            "const": float(const),
            "const_out": float(const_out),
            "const_ret": float(const_ret),
            "coeff_yOUT": coeff_y_out,
            "coeff_yRET": coeff_y_ret,
            "served_out_by_tau": served_out_by_tau,
            "served_ret_by_tau": served_ret_by_tau,
            "served_out_by_tau_k": [[] for _ in Tset],
            "served_ret_by_tau_k": [[] for _ in Tset],
            "waiting_cost_slots": wait_cost_slots,
            "fill_eps_cost": 0.0,
            "penalty_cost": float(P.p) * penalty_pax,
            "penalty_pax": penalty_pax,
            "served_total": float(sum(served_out_by_tau) + sum(served_ret_by_tau)),
            "total_demand": float(sum(R_out) + sum(R_ret)),
            "timing_build_s": float(t_build1 - t_build0),
            "timing_solve_s": float(t_solve1 - t_solve0),
            "timing_extract_s": float(t_extract1 - t_extract0),
            "timing_postprocess_s": 0.0,
            "timing_lp_export_s": float(lp_export_time),
            "exported_lp_path": lp_path,
            "cut_valid_lower_bound": True,
            "cut_generation_mode": "refined_lp_relaxation",
            **build_stats,
        },
        obj_val,
    )


def solve_refined_subproblem(
    P: "SPParams",
    R_out: Iterable[float],
    R_ret: Iterable[float],
    candidate: Candidate | None = None,
):
    t_build0 = time.perf_counter()
    m = pyo.ConcreteModel()
    m.name = "subproblem_refined"
    R_out = [float(v) for v in R_out]
    R_ret = [float(v) for v in R_ret]
    slot_res = max(1, int(P.slot_resolution))
    step = max(1, int(P.time_step_minutes))
    W_minutes = int(P.Wmax_slots) * slot_res
    trip_minutes = int(P.trip_duration_minutes or (int(P.trip_slots or 0) * slot_res))
    charge_rate = float(P.delta_chg or 0.0) / float(slot_res)
    emax = float(P.Emax) if P.Emax is not None else float("inf")
    events = _build_service_events(candidate, P)
    charge_prof = _charge_profile(candidate, P)
    binit = _normalize_binit(P)

    if not events:
        penalty_pax = float(sum(R_out) + sum(R_ret))
        return (
            {
                "is_feasible": True,
                "alpha_OUT": {},
                "alpha_RET": {},
                "pi_OUT": {},
                "pi_RET": {},
                "served_out_by_tau": [0.0 for _ in range(int(P.T))],
                "served_ret_by_tau": [0.0 for _ in range(int(P.T))],
                "served_out_by_tau_k": [[] for _ in range(int(P.T))],
                "served_ret_by_tau_k": [[] for _ in range(int(P.T))],
                "realized_departures": [],
                "realized_departure_min_map": {},
                "refined_departure_diagnostics": [],
                "refined_departure_diagnostics_focus": [],
                "effective_pre_service": [],
                "battery_trajectory": {},
                "objective_value": float(P.p) * penalty_pax,
                "waiting_cost_slots": 0.0,
                "fill_eps_cost": 0.0,
                "penalty_cost": float(P.p) * penalty_pax,
                "penalty_pax": penalty_pax,
                "served_total": 0.0,
                "total_demand": penalty_pax,
                "ub_out": float(P.p) * float(sum(R_out)),
                "ub_ret": float(P.p) * float(sum(R_ret)),
            },
            float(P.p) * penalty_pax,
        )

    event_ids = [e.sid for e in events]
    event_by_id = {e.sid: e for e in events}
    time_choices = []
    force_nominal = bool(getattr(P, "debug_force_nominal_departures", False))
    for e in events:
        if force_nominal:
            time_choices.append((e.sid, int(e.nominal_min)))
        else:
            for h in range(e.window_lb_min, e.window_ub_min + 1, step):
                time_choices.append((e.sid, h))
    choice_map: dict[int, list[int]] = {}
    for sid, h in time_choices:
        choice_map.setdefault(sid, []).append(h)
    for sid in event_ids:
        if not choice_map.get(sid):
            event = event_by_id[sid]
            raise RuntimeError(
                "No feasible realized departure choices for service slot "
                f"q={event.q} tau={event.tau} window=[{event.window_lb_min},{event.window_ub_min}]"
            )

    m.EventTimes = pyo.Set(initialize=time_choices, dimen=2, ordered=False)
    m.z = pyo.Var(m.EventTimes, within=pyo.Binary)
    m.b_after = pyo.Var(event_ids, within=pyo.NonNegativeReals)
    m.Dep = pyo.Expression(event_ids, rule=lambda mm, sid: sum(h * mm.z[sid, h] for h in choice_map[sid]))
    m.OneHot = pyo.Constraint(event_ids, rule=lambda mm, sid: sum(mm.z[sid, h] for h in choice_map[sid]) == 1)

    if trip_minutes > 0:
        def _precedence_rule(mm, sid):
            event = event_by_id[sid]
            if event.prev_sid is None:
                return pyo.Constraint.Skip
            return mm.Dep[sid] >= mm.Dep[event.prev_sid] + trip_minutes
        m.Precedence = pyo.Constraint(event_ids, rule=_precedence_rule)

    battery_cons: list[tuple[int, int, Optional[int], Optional[int], float]] = []
    for event in events:
        if event.prev_sid is None:
            base_batt = float(binit[event.q] if event.q < len(binit) else 0.0)
            for h in choice_map[event.sid]:
                gain = charge_rate * _charge_minutes_between(event.q, 0.0, float(h), charge_prof, slot_res, int(P.T))
                rhs = base_batt + gain - float(P.L or 0.0)
                cname = f"BattInitHi_{event.sid}_{h}"
                m.add_component(cname, pyo.Constraint(expr=m.b_after[event.sid] <= rhs + 1.0e6 * (1 - m.z[event.sid, h])))
                cname = f"BattInitLo_{event.sid}_{h}"
                m.add_component(cname, pyo.Constraint(expr=m.b_after[event.sid] >= rhs - 1.0e6 * (1 - m.z[event.sid, h])))
                cname = f"BattInitFeas_{event.sid}_{h}"
                m.add_component(cname, pyo.Constraint(expr=base_batt + gain + 1.0e6 * (1 - m.z[event.sid, h]) >= float(P.L or 0.0)))
        else:
            prev_sid = int(event.prev_sid)
            for hp in choice_map[prev_sid]:
                for h in choice_map[event.sid]:
                    gain = charge_rate * _charge_minutes_between(
                        event.q,
                        float(hp + trip_minutes),
                        float(h),
                        charge_prof,
                        slot_res,
                        int(P.T),
                    )
                    rhs = gain - float(P.L or 0.0)
                    cname = f"BattHi_{prev_sid}_{hp}_{event.sid}_{h}"
                    m.add_component(cname, pyo.Constraint(expr=m.b_after[event.sid] <= m.b_after[prev_sid] + rhs + 1.0e6 * (2 - m.z[prev_sid, hp] - m.z[event.sid, h])))
                    cname = f"BattLo_{prev_sid}_{hp}_{event.sid}_{h}"
                    m.add_component(cname, pyo.Constraint(expr=m.b_after[event.sid] >= m.b_after[prev_sid] + rhs - 1.0e6 * (2 - m.z[prev_sid, hp] - m.z[event.sid, h])))
                    cname = f"BattFeas_{prev_sid}_{hp}_{event.sid}_{h}"
                    m.add_component(cname, pyo.Constraint(expr=m.b_after[prev_sid] + gain + 1.0e6 * (2 - m.z[prev_sid, hp] - m.z[event.sid, h]) >= float(P.L or 0.0)))

    out_events = [e for e in events if e.direction == "OUT"]
    ret_events = [e for e in events if e.direction == "RET"]
    out_by_sid = {e.sid: e for e in out_events}
    ret_by_sid = {e.sid: e for e in ret_events}

    out_arcs = []
    for t, dem in enumerate(R_out):
        if dem <= 0.0:
            continue
        release = _demand_release_min(int(t), slot_res)
        for e in out_events:
            for h in choice_map[e.sid]:
                if release <= h <= release + W_minutes:
                    out_arcs.append((t, e.sid, h))
    ret_arcs = []
    for t, dem in enumerate(R_ret):
        if dem <= 0.0:
            continue
        release = _demand_release_min(int(t), slot_res)
        for e in ret_events:
            for h in choice_map[e.sid]:
                if release <= h <= release + W_minutes:
                    ret_arcs.append((t, e.sid, h))

    out_cutoffs_by_t: dict[int, set[int]] = {}
    ret_cutoffs_by_t: dict[int, set[int]] = {}
    for t, _sid, h in out_arcs:
        out_cutoffs_by_t.setdefault(int(t), set()).add(int(h))
    for t, _sid, h in ret_arcs:
        ret_cutoffs_by_t.setdefault(int(t), set()).add(int(h))
    eligible_out_prefix = _eligible_arrivals_by_slot_and_cutoff(
        R_out,
        getattr(P, "arrival_minutes_out", None),
        int(P.T),
        slot_res,
        out_cutoffs_by_t,
    )
    eligible_ret_prefix = _eligible_arrivals_by_slot_and_cutoff(
        R_ret,
        getattr(P, "arrival_minutes_ret", None),
        int(P.T),
        slot_res,
        ret_cutoffs_by_t,
    )

    m.OutArcs = pyo.Set(initialize=out_arcs, dimen=3, ordered=False)
    m.RetArcs = pyo.Set(initialize=ret_arcs, dimen=3, ordered=False)
    m.x_OUT = pyo.Var(m.OutArcs, within=pyo.NonNegativeReals)
    m.x_RET = pyo.Var(m.RetArcs, within=pyo.NonNegativeReals)
    m.u_OUT = pyo.Var(range(len(R_out)), within=pyo.NonNegativeReals)
    m.u_RET = pyo.Var(range(len(R_ret)), within=pyo.NonNegativeReals)

    def _wait_slots(t: int, h: int) -> float:
        return _wait_slots_from_minute(int(t), int(h), slot_res)

    m.obj = pyo.Objective(
        expr=
        sum((_wait_slots(t, h) + float(P.fill_eps or 0.0) * float(out_by_sid[sid].layer_index)) * m.x_OUT[t, sid, h] for (t, sid, h) in m.OutArcs)
        + sum((_wait_slots(t, h) + float(P.fill_eps or 0.0) * float(ret_by_sid[sid].layer_index)) * m.x_RET[t, sid, h] for (t, sid, h) in m.RetArcs)
        + float(P.p) * (sum(m.u_OUT[t] for t in range(len(R_out))) + sum(m.u_RET[t] for t in range(len(R_ret)))),
        sense=pyo.minimize,
    )

    def _dem_out(mm, t):
        rel = [(sid, h) for (tt, sid, h) in mm.OutArcs if tt == t]
        return sum(mm.x_OUT[t, sid, h] for (sid, h) in rel) + mm.u_OUT[t] == float(R_out[t])
    m.D_out_refined = pyo.Constraint(range(len(R_out)), rule=_dem_out)

    def _dem_ret(mm, t):
        rel = [(sid, h) for (tt, sid, h) in mm.RetArcs if tt == t]
        return sum(mm.x_RET[t, sid, h] for (sid, h) in rel) + mm.u_RET[t] == float(R_ret[t])
    m.D_ret_refined = pyo.Constraint(range(len(R_ret)), rule=_dem_ret)

    for (t, sid, h) in out_arcs:
        m.add_component(f"GateOut_{t}_{sid}_{h}", pyo.Constraint(expr=m.x_OUT[t, sid, h] <= float(R_out[t]) * m.z[sid, h]))
    for (t, sid, h) in ret_arcs:
        m.add_component(f"GateRet_{t}_{sid}_{h}", pyo.Constraint(expr=m.x_RET[t, sid, h] <= float(R_ret[t]) * m.z[sid, h]))

    def _cap_out(mm, sid):
        rel = [(t, h) for (t, sid2, h) in mm.OutArcs if sid2 == sid]
        if not rel:
            return pyo.Constraint.Skip
        return sum(mm.x_OUT[t, sid, h] for (t, h) in rel) <= float(P.S)
    m.Cap_out_refined = pyo.Constraint([e.sid for e in out_events], rule=_cap_out)

    def _cap_ret(mm, sid):
        rel = [(t, h) for (t, sid2, h) in mm.RetArcs if sid2 == sid]
        if not rel:
            return pyo.Constraint.Skip
        return sum(mm.x_RET[t, sid, h] for (t, h) in rel) <= float(P.S)
    m.Cap_ret_refined = pyo.Constraint([e.sid for e in ret_events], rule=_cap_ret)

    for t, cutoffs in out_cutoffs_by_t.items():
        for cutoff in sorted(cutoffs):
            rel = [(sid, h) for (tt, sid, h) in out_arcs if int(tt) == int(t) and int(h) <= int(cutoff)]
            if not rel:
                continue
            eligible = float(eligible_out_prefix.get((int(t), int(cutoff)), 0.0))
            m.add_component(
                f"EligPrefixOut_{int(t)}_{int(cutoff)}",
                pyo.Constraint(expr=sum(m.x_OUT[t, sid, h] for (sid, h) in rel) <= eligible),
            )
    for t, cutoffs in ret_cutoffs_by_t.items():
        for cutoff in sorted(cutoffs):
            rel = [(sid, h) for (tt, sid, h) in ret_arcs if int(tt) == int(t) and int(h) <= int(cutoff)]
            if not rel:
                continue
            eligible = float(eligible_ret_prefix.get((int(t), int(cutoff)), 0.0))
            m.add_component(
                f"EligPrefixRet_{int(t)}_{int(cutoff)}",
                pyo.Constraint(expr=sum(m.x_RET[t, sid, h] for (sid, h) in rel) <= eligible),
            )

    build_stats = _model_size_stats(m)
    lp_path = None
    lp_export_time = 0.0
    try:
        lp_path, lp_export_time = _maybe_export_lp(m, P)
    except Exception:
        lp_path, lp_export_time = (None, 0.0)
    t_build1 = time.perf_counter()

    solver = pyo.SolverFactory(P.lp_solver)
    try:
        if P.solver_options:
            for k, v in P.solver_options.items():
                solver.options[k] = v
    except Exception:
        pass
    _apply_solver_time_limit(solver, getattr(P, "solve_time_limit_s", None))
    if getattr(P, "debug_timing", False):
        print(
            "[SP DEBUG] iter=%s scenario=%s refined build=%.3fs vars=%s bin=%s cons=%s lp=%s"
            % (
                str(getattr(P, "debug_current_iteration", -1)),
                str(getattr(P, "debug_scenario_label", "single")),
                float(t_build1 - t_build0),
                build_stats["model_num_variables"],
                build_stats["model_num_binary_variables"],
                build_stats["model_num_constraints"],
                lp_path or "-",
            )
        )
    t_solve0 = time.perf_counter()
    res = solver.solve(m, tee=bool(getattr(P, "debug_solver_tee", False)), load_solutions=False)
    t_solve1 = time.perf_counter()
    term = getattr(res.solver, "termination_condition", None)
    infeasible_terms = {
        pyo.TerminationCondition.infeasible,
        pyo.TerminationCondition.infeasibleOrUnbounded,
    }
    if term in infeasible_terms:
        first = _first_violation(events, charge_prof, P)
        return (
            {
                "is_feasible": False,
                "infeasible": True,
                "infeasibility_reason": "Refined timing/battery schedule is infeasible.",
                "first_violation": first,
                "refined_departure_diagnostics": [],
                "refined_departure_diagnostics_focus": [],
                "timing_build_s": float(t_build1 - t_build0),
                "timing_solve_s": float(t_solve1 - t_solve0),
                "timing_extract_s": 0.0,
                "timing_postprocess_s": 0.0,
                "timing_lp_export_s": float(lp_export_time),
                **build_stats,
            },
            float("inf"),
        )
    load_ok = False
    try:
        m.solutions.load_from(res)
        load_ok = _has_loaded_solution(m)
    except Exception:
        load_ok = False
    time_limited_with_incumbent = term == pyo.TerminationCondition.maxTimeLimit and load_ok
    if term not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.feasible) and not time_limited_with_incumbent:
        raise RuntimeError(f"Refined subproblem solve failed: termination_condition={term}")
    t_extract0 = time.perf_counter()

    served_out_by_tau = [0.0 for _ in range(int(P.T))]
    served_ret_by_tau = [0.0 for _ in range(int(P.T))]
    served_out_by_tau_k = [[] for _ in range(int(P.T))]
    served_ret_by_tau_k = [[] for _ in range(int(P.T))]
    out_layer_sizes: dict[int, int] = {}
    ret_layer_sizes: dict[int, int] = {}
    for e in out_events:
        out_layer_sizes[e.tau] = max(out_layer_sizes.get(e.tau, 0), e.layer_index + 1)
    for e in ret_events:
        ret_layer_sizes[e.tau] = max(ret_layer_sizes.get(e.tau, 0), e.layer_index + 1)
    for tau, size in out_layer_sizes.items():
        served_out_by_tau_k[tau] = [0.0 for _ in range(size)]
    for tau, size in ret_layer_sizes.items():
        served_ret_by_tau_k[tau] = [0.0 for _ in range(size)]

    realized_departures: list[dict[str, Any]] = []
    realized_departure_min_map: dict[tuple[int, int], float] = {}
    effective_pre_service: list[dict[str, Any]] = []
    battery_trajectory: dict[str, list[dict[str, Any]]] = {}
    refined_departure_diagnostics: list[dict[str, Any]] = []
    wait_cost_slots = 0.0
    fill_eps_cost = 0.0
    ub_out = 0.0
    ub_ret = 0.0
    dep_by_sid: dict[int, float] = {}
    x_out_vals: dict[tuple[int, int, int], float] = {}
    x_ret_vals: dict[tuple[int, int, int], float] = {}
    out_arcs_by_sid: dict[int, list[tuple[int, int, int]]] = {}
    ret_arcs_by_sid: dict[int, list[tuple[int, int, int]]] = {}
    for sid, h in time_choices:
        z_val = float(m.z[sid, h].value or 0.0)
        if z_val >= 0.5:
            dep_by_sid[sid] = float(h)
    for sid in event_ids:
        dep_by_sid.setdefault(sid, float(choice_map[sid][0]))
    _validate_realized_departure_minutes(events, dep_by_sid, slot_res)
    for arc in out_arcs:
        val = float(m.x_OUT[arc].value or 0.0)
        x_out_vals[arc] = val
        out_arcs_by_sid.setdefault(int(arc[1]), []).append(arc)
    for arc in ret_arcs:
        val = float(m.x_RET[arc].value or 0.0)
        x_ret_vals[arc] = val
        ret_arcs_by_sid.setdefault(int(arc[1]), []).append(arc)

    for e in out_events:
        served = sum(x_out_vals[arc] for arc in out_arcs_by_sid.get(e.sid, []))
        served_out_by_tau[e.tau] += served
        served_out_by_tau_k[e.tau][e.layer_index] = served
    for e in ret_events:
        served = sum(x_ret_vals[arc] for arc in ret_arcs_by_sid.get(e.sid, []))
        served_ret_by_tau[e.tau] += served
        served_ret_by_tau_k[e.tau][e.layer_index] = served

    for e in out_events:
        dep = float(dep_by_sid[e.sid])
        boarded = float(sum(x_out_vals[arc] for arc in out_arcs_by_sid.get(e.sid, [])))
        eligible = _direction_eligible_arrivals_by_minute(
            R_out,
            getattr(P, "arrival_minutes_out", None),
            int(P.T),
            slot_res,
            dep,
        )
        refined_departure_diagnostics.append({
            "vehicle_id": int(e.q),
            "direction": "OUT",
            "slot": int(e.tau),
            "departure_min": dep,
            "boarded": boarded,
            "eligible_arrivals": eligible,
            "violation": bool(boarded > eligible + 1.0e-9),
        })
    for e in ret_events:
        dep = float(dep_by_sid[e.sid])
        boarded = float(sum(x_ret_vals[arc] for arc in ret_arcs_by_sid.get(e.sid, [])))
        eligible = _direction_eligible_arrivals_by_minute(
            R_ret,
            getattr(P, "arrival_minutes_ret", None),
            int(P.T),
            slot_res,
            dep,
        )
        refined_departure_diagnostics.append({
            "vehicle_id": int(e.q),
            "direction": "RET",
            "slot": int(e.tau),
            "departure_min": dep,
            "boarded": boarded,
            "eligible_arrivals": eligible,
            "violation": bool(boarded > eligible + 1.0e-9),
        })
    refined_departure_diagnostics.sort(
        key=lambda rec: (str(rec.get("direction", "")), float(rec.get("departure_min", 0.0)), int(rec.get("vehicle_id", 0)), int(rec.get("slot", 0)))
    )
    focus_counts = {"OUT": 0, "RET": 0}
    refined_departure_diagnostics_focus: list[dict[str, Any]] = []
    for rec in refined_departure_diagnostics:
        direction = str(rec.get("direction", ""))
        if focus_counts.get(direction, 0) >= 2:
            continue
        refined_departure_diagnostics_focus.append(dict(rec))
        focus_counts[direction] = focus_counts.get(direction, 0) + 1

    for (t, sid, h) in out_arcs:
        val = x_out_vals[(t, sid, h)]
        if val <= 0.0:
            continue
        cost = _wait_slots(t, h)
        wait_cost_slots += cost * val
        fill_eps_cost += float(P.fill_eps or 0.0) * float(out_by_sid[sid].layer_index) * val
        ub_out += (cost + float(P.fill_eps or 0.0) * float(out_by_sid[sid].layer_index)) * val
    for (t, sid, h) in ret_arcs:
        val = x_ret_vals[(t, sid, h)]
        if val <= 0.0:
            continue
        cost = _wait_slots(t, h)
        wait_cost_slots += cost * val
        fill_eps_cost += float(P.fill_eps or 0.0) * float(ret_by_sid[sid].layer_index) * val
        ub_ret += (cost + float(P.fill_eps or 0.0) * float(ret_by_sid[sid].layer_index)) * val

    u_out_vals = [float(m.u_OUT[t].value or 0.0) for t in range(len(R_out))]
    u_ret_vals = [float(m.u_RET[t].value or 0.0) for t in range(len(R_ret))]
    penalty_pax = float(sum(u_out_vals) + sum(u_ret_vals))
    penalty_cost = float(P.p) * penalty_pax
    total_demand = float(sum(R_out) + sum(R_ret))
    served_total = float(sum(served_out_by_tau) + sum(served_ret_by_tau))
    obj_val = float(pyo.value(m.obj))
    t_extract1 = time.perf_counter()

    t_post0 = time.perf_counter()
    for q in range(int(P.Q)):
        seq = [e for e in events if e.q == q]
        seq.sort(key=lambda e: e.sid)
        batt = float(binit[q] if q < len(binit) else 0.0)
        prev_dep = 0.0
        prev_event: Optional[RefinedServiceEvent] = None
        battery_trajectory[str(q)] = []
        for e in seq:
            dep = dep_by_sid[e.sid]
            charge_min = _charge_minutes_between(q, prev_dep + trip_minutes if prev_event is not None else 0.0, dep, charge_prof, slot_res, int(P.T))
            idle_min = max(0.0, dep - (prev_dep + trip_minutes if prev_event is not None else 0.0) - charge_min)
            batt = min(emax, batt + charge_rate * charge_min) - float(P.L or 0.0)
            battery_trajectory[str(q)].append({
                "slot": e.tau,
                "direction": e.direction,
                "departure_min": dep,
                "battery_after": batt,
            })
            realized_departures.append({
                "vehicle_id": e.q,
                "slot": e.tau,
                "direction": e.direction,
                "nominal_departure_min": e.nominal_min,
                "realized_departure_min": dep,
            })
            realized_departure_min_map[(int(e.q), int(e.tau))] = float(dep)
            effective_pre_service.append({
                "vehicle_id": e.q,
                "slot": e.tau,
                "direction": e.direction,
                "charge_minutes": charge_min,
                "idle_minutes": idle_min,
                "battery_after": batt,
                "predecessor_slot": prev_event.tau if prev_event is not None else None,
            })
            prev_dep = dep
            prev_event = e
    t_post1 = time.perf_counter()

    return (
        {
            "is_feasible": True,
            "alpha_OUT": {},
            "alpha_RET": {},
            "pi_OUT": {},
            "pi_RET": {},
            "served_out_by_tau": served_out_by_tau,
            "served_ret_by_tau": served_ret_by_tau,
            "served_out_by_tau_k": served_out_by_tau_k,
            "served_ret_by_tau_k": served_ret_by_tau_k,
            "realized_departures": realized_departures,
            "realized_departure_min_map": realized_departure_min_map,
            "refined_departure_diagnostics": refined_departure_diagnostics,
            "refined_departure_diagnostics_focus": refined_departure_diagnostics_focus,
            "effective_pre_service": effective_pre_service,
            "battery_trajectory": battery_trajectory,
            "ub_out": ub_out + float(P.p) * float(sum(u_out_vals)),
            "ub_ret": ub_ret + float(P.p) * float(sum(u_ret_vals)),
            "objective_value": obj_val,
            "waiting_cost_slots": wait_cost_slots,
            "fill_eps_cost": fill_eps_cost,
            "penalty_cost": penalty_cost,
            "penalty_pax": penalty_pax,
            "served_total": served_total,
            "total_demand": total_demand,
            "timing_build_s": float(t_build1 - t_build0),
            "timing_solve_s": float(t_solve1 - t_solve0),
            "timing_extract_s": float(t_extract1 - t_extract0),
            "timing_postprocess_s": float(t_post1 - t_post0),
            "timing_lp_export_s": float(lp_export_time),
            "exported_lp_path": lp_path,
            "time_limited_incumbent": bool(time_limited_with_incumbent),
            **build_stats,
        },
        obj_val,
    )


@dataclass
class SPParams:
    T: int
    Wmax_slots: int
    p: float
    lp_solver: str
    S: float
    K_out: list[int]
    K_ret: list[int]
    fill_eps: float = 0.0
    solver_options: dict | None = None
    eps_cut: float = 1e-8
    slot_resolution: int = 1
    time_step_minutes: int = 1
    T_minutes: int | None = None
    trip_duration_minutes: int | None = None
    trip_slots: int | None = None
    Q: int = 0
    binit: Any = None
    initial_actions: Any = None
    Emax: float | None = None
    L: float | None = None
    delta_chg: float | None = None
    arrival_minutes_out: list[float] | None = None
    arrival_minutes_ret: list[float] | None = None
    eps_feas: float = 1e-7
    debug_timing: bool = False
    debug_solver_tee: bool = False
    debug_export_lp_iteration: int | None = None
    debug_current_iteration: int = -1
    debug_report_dir: str | None = None
    debug_force_nominal_departures: bool = False
    debug_scenario_label: str | None = None
    solve_time_limit_s: float | None = None


def solve_subproblem(
    P: SPParams,
    C_out: Iterable[float],
    C_ret: Iterable[float],
    R_out: Iterable[float],
    R_ret: Iterable[float],
    candidate: Candidate | None = None,
):
    """Replicates user's subproblem sketch and returns duals and objective.

    Returns (duals: dict[str, dict[int, float]], objective_value: float)
    """
    t_build0 = time.perf_counter()
    m = pyo.ConcreteModel()
    m.name = "subproblem"
    Tset = range(P.T)

    C_out = list(C_out)
    C_ret = list(C_ret)
    R_out = list(R_out)
    R_ret = list(R_ret)

    W = P.Wmax_slots

    # Define valid arcs with causality and max-wait: (t + 1) <= tau <= min(T-1, t+W)
    # Interpretation: demand aggregated in slot t (arrivals during [t*res, (t+1)*res))
    # can be served by the next slot's departure at the earliest (tau = t+1).
    # Same-slot service (tau = t) is disallowed to avoid serving passengers after the slot's departure.
    Arcs_list = [(t, tau) for t in Tset for tau in Tset if (t + 1) <= tau <= min(P.T - 1, t + W)]
    m.Arcs = pyo.Set(initialize=Arcs_list, dimen=2, ordered=False)

    # Layered arcs per departure time based on number of vehicles at tau
    OutLayers = [(tau, k) for tau in Tset for k in range(int(P.K_out[tau]) if tau < len(P.K_out) else 0)]
    RetLayers = [(tau, k) for tau in Tset for k in range(int(P.K_ret[tau]) if tau < len(P.K_ret) else 0)]
    m.OutLayers = pyo.Set(initialize=OutLayers, dimen=2, ordered=False)
    m.RetLayers = pyo.Set(initialize=RetLayers, dimen=2, ordered=False)

    ArcsOut = [(t, tau, k) for (t, tau) in Arcs_list for (tau2, k) in OutLayers if tau2 == tau]
    ArcsRet = [(t, tau, k) for (t, tau) in Arcs_list for (tau2, k) in RetLayers if tau2 == tau]
    m.ArcsOut = pyo.Set(initialize=ArcsOut, dimen=3, ordered=False)
    m.ArcsRet = pyo.Set(initialize=ArcsRet, dimen=3, ordered=False)

    # Variables defined on layered arcs only
    m.x_OUT = pyo.Var(m.ArcsOut, within=pyo.NonNegativeReals)
    m.x_RET = pyo.Var(m.ArcsRet, within=pyo.NonNegativeReals)
    m.u_OUT = pyo.Var(Tset, within=pyo.NonNegativeReals)
    m.u_RET = pyo.Var(Tset, within=pyo.NonNegativeReals)

    def wait_cost(t: int, tau: int) -> float:
        return float(max(0, tau - t))

    # Objective sums over layered arcs; small per-layer epsilon encourages packing into lower k first
    def layer_cost(t: int, tau: int, k: int) -> float:
        return float(max(0, tau - t)) + max(0.0, float(P.fill_eps)) * float(k)

    m.obj = pyo.Objective(
        expr=
            sum(layer_cost(t, tau, k) * m.x_OUT[t, tau, k] for (t, tau, k) in m.ArcsOut)
            + sum(layer_cost(t, tau, k) * m.x_RET[t, tau, k] for (t, tau, k) in m.ArcsRet)
            + P.p * (sum(m.u_OUT[t] for t in Tset) + sum(m.u_RET[t] for t in Tset)),
        sense=pyo.minimize,
    )

    def cons_dem_OUT(m, t):
        taus = [tau for tau in Tset if t <= tau <= min(P.T - 1, t + W)]
        return sum(m.x_OUT[t, tau, k] for tau in taus for k in range(int(P.K_out[tau]) if tau < len(P.K_out) else 0) if (t, tau, k) in m.ArcsOut) + m.u_OUT[t] == R_out[t]

    m.D_out = pyo.Constraint(Tset, rule=cons_dem_OUT)

    def cons_dem_RET(m, t):
        taus = [tau for tau in Tset if t <= tau <= min(P.T - 1, t + W)]
        return sum(m.x_RET[t, tau, k] for tau in taus for k in range(int(P.K_ret[tau]) if tau < len(P.K_ret) else 0) if (t, tau, k) in m.ArcsRet) + m.u_RET[t] == R_ret[t]

    m.D_ret = pyo.Constraint(Tset, rule=cons_dem_RET)

    # No same-slot or last-slot caps in the original model

    # Per-layer capacities: each vehicle layer is one shuttle => up to S seats
    def cap_out_layer(m, tau, k):
        ts = [t for t in Tset if t <= tau <= min(P.T - 1, t + W)]
        # If no valid arcs exist for this layer, skip the constraint
        if not any((t, tau, k) in m.ArcsOut for t in ts):
            return pyo.Constraint.Skip
        return sum(m.x_OUT[t, tau, k] for t in ts if (t, tau, k) in m.ArcsOut) <= min(float(P.S), float(C_out[tau]))

    m.Cap_out = pyo.Constraint(m.OutLayers, rule=cap_out_layer)

    def cap_ret_layer(m, tau, k):
        ts = [t for t in Tset if t <= tau <= min(P.T - 1, t + W)]
        if not any((t, tau, k) in m.ArcsRet for t in ts):
            return pyo.Constraint.Skip
        return sum(m.x_RET[t, tau, k] for t in ts if (t, tau, k) in m.ArcsRet) <= min(float(P.S), float(C_ret[tau]))

    m.Cap_ret = pyo.Constraint(m.RetLayers, rule=cap_ret_layer)

    # Dual suffix required to read duals
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    build_stats = _model_size_stats(m)
    lp_path = None
    lp_export_time = 0.0
    try:
        lp_path, lp_export_time = _maybe_export_lp(m, P)
    except Exception:
        lp_path, lp_export_time = (None, 0.0)
    t_build1 = time.perf_counter()

    solver = pyo.SolverFactory(P.lp_solver)
    # Allow tuning via options, e.g., for CPLEX: {"lpmethod": 2, "threads": 0, "parallel": 1}
    try:
        if getattr(P, "solver_options", None):
            for k, v in (P.solver_options or {}).items():
                solver.options[k] = v
    except Exception:
        pass
    _apply_solver_time_limit(solver, getattr(P, "solve_time_limit_s", None))
    if getattr(P, "debug_timing", False):
        print(
            "[SP DEBUG] iter=%s scenario=%s nominal build=%.3fs vars=%s bin=%s cons=%s lp=%s"
            % (
                str(getattr(P, "debug_current_iteration", -1)),
                str(getattr(P, "debug_scenario_label", "single")),
                float(t_build1 - t_build0),
                build_stats["model_num_variables"],
                build_stats["model_num_binary_variables"],
                build_stats["model_num_constraints"],
                lp_path or "-",
            )
        )
    t_solve0 = time.perf_counter()
    res = solver.solve(m, tee=bool(getattr(P, "debug_solver_tee", False)), load_solutions=False)
    term = getattr(res.solver, "termination_condition", None)
    if term not in (pyo.TerminationCondition.optimal,):
        # Retry with presolve off if possible
        try:
            solver.options["preind"] = 0
            solver.options["presolve"] = 0
            solver.options["reduce"] = 0
        except Exception:
            pass
        res = solver.solve(m, tee=bool(getattr(P, "debug_solver_tee", False)), load_solutions=False)
        term = getattr(res.solver, "termination_condition", None)
        if term not in (pyo.TerminationCondition.optimal,):
            try:
                out_dir = _report_dir_from_debug(getattr(P, "debug_report_dir", None))
                fail_lp_path = out_dir / "subproblem_failed.lp"
                m.write(str(fail_lp_path), io_options={"symbolic_solver_labels": True})
                print(f"[SP] Wrote LP: {fail_lp_path}")
                if candidate is not None:
                    cand_path = out_dir / "subproblem_failed_candidate.txt"
                    with cand_path.open("w", encoding="utf-8") as f:
                        for k, v in sorted(candidate.items()):
                            f.write(f"{k}={v}\n")
                    print(f"[SP] Wrote candidate: {cand_path}")
            except Exception:
                pass
            raise RuntimeError(f"Subproblem solve ambiguous: termination_condition={term}")
    t_solve1 = time.perf_counter()
    # Load solution only after optimal termination
    try:
        m.solutions.load_from(res)
    except Exception:
        pass
    t_extract0 = time.perf_counter()

    alpha_OUT = {t: float(m.dual.get(m.D_out[t], 0.0)) for t in Tset}
    alpha_RET = {t: float(m.dual.get(m.D_ret[t], 0.0)) for t in Tset}
    pi_OUT = {}
    for tau in Tset:
        total = 0.0
        kmax = int(P.K_out[tau]) if tau < len(P.K_out) else 0
        for k in range(kmax):
            if (tau, k) in m.Cap_out:
                total += float(m.dual.get(m.Cap_out[tau, k], 0.0))
        pi_OUT[tau] = total
    pi_RET = {}
    for tau in Tset:
        total = 0.0
        kmax = int(P.K_ret[tau]) if tau < len(P.K_ret) else 0
        for k in range(kmax):
            if (tau, k) in m.Cap_ret:
                total += float(m.dual.get(m.Cap_ret[tau, k], 0.0))
        pi_RET[tau] = total

    # Gather simple primal summaries
    served_out_by_tau = [0.0 for _ in Tset]
    served_ret_by_tau = [0.0 for _ in Tset]
    # Also collect per-layer (per shuttle) served counts at each departure slot
    served_out_by_tau_k = [[] for _ in Tset]
    served_ret_by_tau_k = [[] for _ in Tset]
    for tau in Tset:
        # Aggregate across demand time t for each layer k
        kmax_out = int(P.K_out[tau]) if tau < len(P.K_out) else 0
        kmax_ret = int(P.K_ret[tau]) if tau < len(P.K_ret) else 0
        # Initialize per-layer arrays
        if kmax_out > 0:
            served_out_by_tau_k[tau] = [0.0 for _ in range(kmax_out)]
        if kmax_ret > 0:
            served_ret_by_tau_k[tau] = [0.0 for _ in range(kmax_ret)]
        # Sum flows
        total_out_tau = 0.0
        total_ret_tau = 0.0
        for k in range(kmax_out):
            val_k = sum(float(pyo.value(m.x_OUT[t, tau, k])) for t in Tset if (t, tau, k) in m.ArcsOut)
            served_out_by_tau_k[tau][k] = val_k
            total_out_tau += val_k
        for k in range(kmax_ret):
            val_k = sum(float(pyo.value(m.x_RET[t, tau, k])) for t in Tset if (t, tau, k) in m.ArcsRet)
            served_ret_by_tau_k[tau][k] = val_k
            total_ret_tau += val_k
        served_out_by_tau[tau] = total_out_tau
        served_ret_by_tau[tau] = total_ret_tau

    # Component costs (per direction)
    try:
        out_cost_val = sum(layer_cost(t, tau, k) * float(pyo.value(m.x_OUT[t, tau, k])) for (t, tau, k) in m.ArcsOut)
        out_cost_val += float(P.p) * sum(float(pyo.value(m.u_OUT[t])) for t in Tset)
    except Exception:
        out_cost_val = 0.0
    try:
        ret_cost_val = sum(layer_cost(t, tau, k) * float(pyo.value(m.x_RET[t, tau, k])) for (t, tau, k) in m.ArcsRet)
        ret_cost_val += float(P.p) * sum(float(pyo.value(m.u_RET[t])) for t in Tset)
    except Exception:
        ret_cost_val = 0.0

    # Component costs
    wait_cost_slots = 0.0
    fill_eps_cost = 0.0
    neg_contribs: list[tuple[float, int, int, int, float]] = []
    for (t, tau, k) in m.ArcsOut:
        val = float(pyo.value(m.x_OUT[t, tau, k]) or 0.0)
        if val == 0.0:
            continue
        w = float(max(0, tau - t))
        wait_cost_slots += w * val
        fill_eps_cost += float(max(0.0, float(P.fill_eps))) * float(k) * val
        contrib = w * val
        if contrib < -1e-9:
            neg_contribs.append((contrib, int(t), int(tau), int(k), val))
    for (t, tau, k) in m.ArcsRet:
        val = float(pyo.value(m.x_RET[t, tau, k]) or 0.0)
        if val == 0.0:
            continue
        w = float(max(0, tau - t))
        wait_cost_slots += w * val
        fill_eps_cost += float(max(0.0, float(P.fill_eps))) * float(k) * val
        contrib = w * val
        if contrib < -1e-9:
            neg_contribs.append((contrib, int(t), int(tau), int(k), val))

    penalty_pax = float(sum(float(pyo.value(m.u_OUT[t])) for t in Tset) + sum(float(pyo.value(m.u_RET[t])) for t in Tset))
    penalty_cost = float(P.p) * penalty_pax

    obj_val = float(pyo.value(m.obj))
    # Strong duality check
    try:
        cap_out_rhs = [min(float(P.S), float(C_out[tau])) for tau in Tset]
        cap_ret_rhs = [min(float(P.S), float(C_ret[tau])) for tau in Tset]
        dual_obj = sum(float(R_out[t]) * alpha_OUT[t] for t in Tset) + sum(float(R_ret[t]) * alpha_RET[t] for t in Tset)
        dual_obj += sum(cap_out_rhs[tau] * pi_OUT[tau] for tau in Tset)
        dual_obj += sum(cap_ret_rhs[tau] * pi_RET[tau] for tau in Tset)
        eps_cut = float(getattr(P, "eps_cut", 1e-8)) if hasattr(P, "eps_cut") else 1e-8
        if abs(float(obj_val) - float(dual_obj)) > eps_cut * max(1.0, abs(float(obj_val))):
            raise RuntimeError(
                f"Strong duality check failed: primal={obj_val} dual={dual_obj}"
            )
    except Exception as exc:
        try:
            out_dir = _report_dir_from_debug(getattr(P, "debug_report_dir", None))
            fail_lp_path = out_dir / "subproblem_duality_failed.lp"
            m.write(str(fail_lp_path), io_options={"symbolic_solver_labels": True})
            print(f"[SP] Wrote LP: {fail_lp_path}")
        except Exception:
            pass
        raise
    sum_components = wait_cost_slots + fill_eps_cost + penalty_cost
    if abs(sum_components - obj_val) > 1e-5:
        print(
            "[SP DIAG] Objective mismatch: obj=%.6g wait=%.6g fill_eps=%.6g penalty=%.6g sum=%.6g"
            % (obj_val, wait_cost_slots, fill_eps_cost, penalty_cost, sum_components)
        )
        assert abs(sum_components - obj_val) <= 1e-5
    if wait_cost_slots < -1e-9:
        neg_contribs.sort(key=lambda x: x[0])
        print("[SP DIAG] Negative waiting cost detected. Top negative contributions:")
        for c, t, tau, k, val in neg_contribs[:10]:
            print(f"  contrib={c:.6g} t={t} tau={tau} k={k} x={val:.6g}")
        assert wait_cost_slots >= -1e-9
    if penalty_cost < -1e-9:
        print(f"[SP DIAG] Negative penalty cost detected: {penalty_cost:.6g}")
        assert penalty_cost >= -1e-9
    total_demand = float(sum(R_out) + sum(R_ret))
    served_total = float(sum(served_out_by_tau) + sum(served_ret_by_tau))
    t_extract1 = time.perf_counter()
    t_post0 = time.perf_counter()
    # Consistency check: served + unmet == total demand (within tolerance)
    if abs((served_total + penalty_pax) - total_demand) > 1e-5:
        print("[SP DIAG] Demand mismatch: total=%.6g served=%.6g unmet=%.6g" % (total_demand, served_total, penalty_pax))
        # Do not assert here; keep diagnostic only
    t_post1 = time.perf_counter()

    return (
        {
            "alpha_OUT": alpha_OUT,
            "alpha_RET": alpha_RET,
            "pi_OUT": pi_OUT,
            "pi_RET": pi_RET,
            # diagnostics
            "served_out_by_tau": served_out_by_tau,
            "served_ret_by_tau": served_ret_by_tau,
            # per-layer diagnostics (per departure layer k at each tau)
            "served_out_by_tau_k": served_out_by_tau_k,
            "served_ret_by_tau_k": served_ret_by_tau_k,
            # components
            "ub_out": float(out_cost_val),
            "ub_ret": float(ret_cost_val),
            "objective_value": obj_val,
            "waiting_cost_slots": wait_cost_slots,
            "fill_eps_cost": fill_eps_cost,
            "penalty_cost": penalty_cost,
            "penalty_pax": penalty_pax,
            "served_total": served_total,
            "total_demand": total_demand,
            "timing_build_s": float(t_build1 - t_build0),
            "timing_solve_s": float(t_solve1 - t_solve0),
            "timing_extract_s": float(t_extract1 - t_extract0),
            "timing_postprocess_s": float(t_post1 - t_post0),
            "timing_lp_export_s": float(lp_export_time),
            "exported_lp_path": lp_path,
            **build_stats,
        },
        obj_val,
    )
