from __future__ import annotations

from typing import Any, Dict, Tuple, Iterable
from pathlib import Path
import json
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
        # Parameters (with safe defaults)
        S = float(params.get("S", 1.0))
        # Resolution in minutes per slot (copied from master params via config or set here)
        slot_res = int(params.get("slot_resolution", params.get("resolution", 1)))
        # Allow Wmax to be specified in minutes
        import math
        if "Wmax_minutes" in params:
            Wmax = int(math.ceil(float(params.get("Wmax_minutes", 0)) / max(1, slot_res)))
        else:
            Wmax = int(params.get("Wmax_slots", params.get("Wmax", 0)))
        p_pen = float(params.get("p", 0.0))
        lp_solver = str(params.get("lp_solver", "cplex_direct"))
        # Prefer packing demand into the first vehicle layer, then the next (LP tie-breaker)
        fill_eps = float(params.get("fill_first_epsilon", 1e-6) or 0.0)

        # Determine T and Q from candidate if not configured
        q_idx, t_idx = self._parse_candidate_indices(candidate)
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

        def _load_demand_from_file(path_like: Any, Tlen: int) -> tuple[list[float], list[float]]:
            p = Path(str(path_like))
            doc = _load_doc(p)
            return _aggregate_requests(doc, Tlen)

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
        average_cuts: bool = bool(params.get("average_cuts_across_scenarios", False))
        ub_aggregation: str = str(params.get("ub_aggregation", "mean"))
        weights: list[float] | None = params.get("scenario_weights")

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

            for tau in range(T):
                # OUT marginal
                C_out_tau = C_out_base.copy()
                C_out_tau[tau] = C_out_tau[tau] + S
                K_out_tau = K_out_base.copy()
                K_out_tau[tau] = K_out_tau[tau] + 1
                _, ub_plus = solve_subproblem(
                    SPParams(T=T, Wmax_slots=Wmax, p=p_pen, lp_solver=lp_solver, S=S, K_out=K_out_tau, K_ret=K_ret_base, fill_eps=fill_eps),
                    C_out_tau,
                    C_ret_base,
                    R_out_vec,
                    R_ret_vec,
                )
                dm = float(ub_plus - ub_base)
                dm_out[tau] = dm

                # RET marginal
                C_ret_tau = C_ret_base.copy()
                C_ret_tau[tau] = C_ret_tau[tau] + S
                K_ret_tau = K_ret_base.copy()
                K_ret_tau[tau] = K_ret_tau[tau] + 1
                _, ub_plus_r = solve_subproblem(
                    SPParams(T=T, Wmax_slots=Wmax, p=p_pen, lp_solver=lp_solver, S=S, K_out=K_out_base, K_ret=K_ret_tau, fill_eps=fill_eps),
                    C_out_base,
                    C_ret_tau,
                    R_out_vec,
                    R_ret_vec,
                )
                dm_r = float(ub_plus_r - ub_base)
                dm_ret[tau] = dm_r

            # Expand to per-(q,tau) using candidate indices
            for name in candidate.keys():
                if not isinstance(name, str):
                    continue
                if name.startswith("yOUT["):
                    inside = name[name.find("[") + 1 : name.find("]")]
                    q_str, tau_str = inside.split(",")
                    q = int(q_str.strip())
                    tau = int(tau_str.strip())
                    if 0 <= tau < T:
                        coeff_y_out[(q, tau)] = dm_out.get(tau, 0.0)
                elif name.startswith("yRET["):
                    inside = name[name.find("[") + 1 : name.find("]")]
                    q_str, tau_str = inside.split(",")
                    q = int(q_str.strip())
                    tau = int(tau_str.strip())
                    if 0 <= tau < T:
                        coeff_y_ret[(q, tau)] = dm_ret.get(tau, 0.0)

            return coeff_y_out, coeff_y_ret, dm_out, dm_ret

        # If scenarios provided, iterate; else use single-demand params
        if scenarios:
            if weights and len(weights) != len(scenarios):
                raise ValueError("scenario_weights must match number of scenarios")
            if not weights:
                weights = [1.0 / len(scenarios)] * len(scenarios)

            cuts: list[Cut] = []
            ub_vals: list[float] = []
            consts: list[float] = []
            coeffs_out_list: list[Dict[tuple[int, int], float]] = []
            coeffs_ret_list: list[Dict[tuple[int, int], float]] = []

            for s in scenarios:
                if isinstance(s, (str, Path)):
                    R_out, R_ret = _load_demand_from_file(s, T)
                elif isinstance(s, dict) and ("requests" in s or "req_matrix" in s or "R_out" in s or "R_ret" in s):
                    R_out, R_ret = _aggregate_requests(s, T)
                else:
                    # Best effort
                    R_out = list(getattr(s, "R_out", [0.0] * T))
                    R_ret = list(getattr(s, "R_ret", [0.0] * T))
                R_out = (R_out + [0.0] * T)[:T]
                R_ret = (R_ret + [0.0] * T)[:T]

                sp_params = SPParams(T=T, Wmax_slots=Wmax, p=p_pen, lp_solver=lp_solver, S=S, K_out=K_out, K_ret=K_ret, fill_eps=fill_eps)
                duals, ub_val = solve_subproblem(sp_params, C_out, C_ret, R_out, R_ret)
                ub_vals.append(ub_val)

                # Finite-difference coefficients and constant per scenario
                c_out_map, c_ret_map, dm_out, dm_ret = coeffs_by_fdiff(ub_val, C_out, C_ret, K_out, K_ret, R_out, R_ret)
                sum_y_out = [float(C_out[tau]) / S if S != 0 else 0.0 for tau in range(T)]
                sum_y_ret = [float(C_ret[tau]) / S if S != 0 else 0.0 for tau in range(T)]
                const = float(ub_val)
                const -= sum(dm_out[tau] * sum_y_out[tau] for tau in range(T))
                const -= sum(dm_ret[tau] * sum_y_ret[tau] for tau in range(T))
                consts.append(const)
                coeffs_out_list.append(c_out_map)
                coeffs_ret_list.append(c_ret_map)
                cuts.append(Cut(name="opt_cut_s", cut_type=CutType.OPTIMALITY, metadata={"const": const, "coeff_yOUT": c_out_map, "coeff_yRET": c_ret_map}))

            # Aggregate UB
            if ub_aggregation == "mean":
                ub_val_agg = sum(w * u for w, u in zip(weights, ub_vals))
            elif ub_aggregation == "sum":
                ub_val_agg = sum(ub_vals)
            elif ub_aggregation == "max":
                ub_val_agg = max(ub_vals)
            else:
                raise ValueError("ub_aggregation must be one of 'mean', 'sum', 'max'")

            if average_cuts:
                # Weighted average of constants and coefficients
                const_avg = sum(w * c for w, c in zip(weights, consts))
                keys_out = set().union(*[set(d.keys()) for d in coeffs_out_list])
                keys_ret = set().union(*[set(d.keys()) for d in coeffs_ret_list])
                avg_out: Dict[tuple[int, int], float] = {}
                avg_ret: Dict[tuple[int, int], float] = {}
                for k in keys_out:
                    avg_out[k] = sum(weights[i] * coeffs_out_list[i].get(k, 0.0) for i in range(len(coeffs_out_list)))
                for k in keys_ret:
                    avg_ret[k] = sum(weights[i] * coeffs_ret_list[i].get(k, 0.0) for i in range(len(coeffs_ret_list)))
                cut = Cut(name="opt_cut_avg", cut_type=CutType.OPTIMALITY, metadata={"const": const_avg, "coeff_yOUT": avg_out, "coeff_yRET": avg_ret})
                return SubproblemResult(is_feasible=True, cut=cut, upper_bound=ub_val_agg)
            else:
                return SubproblemResult(is_feasible=True, cuts=cuts, upper_bound=ub_val_agg)
        else:
            # Single-demand case from params (prefer external file if given)
            if params.get("demand_file"):
                R_out, R_ret = _load_demand_from_file(params.get("demand_file"), T)
            elif ("requests" in params) or ("req_matrix" in params) or ("R_out" in params) or ("R_ret" in params):
                R_out, R_ret = _aggregate_requests(params, T)
            else:
                R_out = [0.0] * T
                R_ret = [0.0] * T
            if len(R_out) != T:
                R_out = (R_out + [0.0] * T)[:T]
            if len(R_ret) != T:
                R_ret = (R_ret + [0.0] * T)[:T]

            sp_params = SPParams(T=T, Wmax_slots=Wmax, p=p_pen, lp_solver=lp_solver, S=S, K_out=K_out, K_ret=K_ret, fill_eps=fill_eps)
            duals, ub_val = solve_subproblem(sp_params, C_out, C_ret, R_out, R_ret)

            # Build coefficients via finite differences around current capacity
            c_out_map, c_ret_map, dm_out, dm_ret = coeffs_by_fdiff(ub_val, C_out, C_ret, K_out, K_ret, R_out, R_ret)

            # Compute sum_y per tau (number of vehicles starting at tau) from current capacity
            sum_y_out = [float(C_out[tau]) / S if S != 0 else 0.0 for tau in range(T)]
            sum_y_ret = [float(C_ret[tau]) / S if S != 0 else 0.0 for tau in range(T)]

            # Constant so that cut passes through (y, Q(y))
            const = float(ub_val)
            const -= sum(dm_out[tau] * sum_y_out[tau] for tau in range(T))
            const -= sum(dm_ret[tau] * sum_y_ret[tau] for tau in range(T))

            cut = Cut(name="opt_cut", cut_type=CutType.OPTIMALITY, metadata={"const": const, "coeff_yOUT": c_out_map, "coeff_yRET": c_ret_map})
            # Diagnostics: demand per slot split by direction and served pax per departure slot (OUT/RET)
            diagnostics = {
                "T": T,
                "R_out": [float(R_out[t]) for t in range(T)],
                "R_ret": [float(R_ret[t]) for t in range(T)],
                "pax_out_by_tau": list(duals.get("served_out_by_tau", [0.0] * T)),
                "pax_ret_by_tau": list(duals.get("served_ret_by_tau", [0.0] * T)),
            }
            return SubproblemResult(is_feasible=True, cut=cut, upper_bound=ub_val, diagnostics=diagnostics)


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


def solve_subproblem(P: SPParams, C_out: Iterable[float], C_ret: Iterable[float], R_out: Iterable[float], R_ret: Iterable[float]):
    """Replicates user's subproblem sketch and returns duals and objective.

    Returns (duals: dict[str, dict[int, float]], objective_value: float)
    """
    m = pyo.ConcreteModel()
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

    solver = pyo.SolverFactory(P.lp_solver)
    solver.solve(m, tee=False)

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
    for tau in Tset:
        served_out_by_tau[tau] = sum(float(pyo.value(m.x_OUT[t, tau, k])) for t in Tset for k in range(int(P.K_out[tau]) if tau < len(P.K_out) else 0) if (t, tau, k) in m.ArcsOut)
        served_ret_by_tau[tau] = sum(float(pyo.value(m.x_RET[t, tau, k])) for t in Tset for k in range(int(P.K_ret[tau]) if tau < len(P.K_ret) else 0) if (t, tau, k) in m.ArcsRet)

    obj_val = float(pyo.value(m.obj))
    return (
        {
            "alpha_OUT": alpha_OUT,
            "alpha_RET": alpha_RET,
            "pi_OUT": pi_OUT,
            "pi_RET": pi_RET,
            # diagnostics
            "served_out_by_tau": served_out_by_tau,
            "served_ret_by_tau": served_ret_by_tau,
        },
        obj_val,
    )
