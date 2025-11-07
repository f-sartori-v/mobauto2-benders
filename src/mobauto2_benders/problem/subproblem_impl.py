from __future__ import annotations

from typing import Any, Dict, Tuple, Iterable

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
        Wmax = int(params.get("Wmax_slots", params.get("Wmax", 0)))
        p_pen = float(params.get("p", 0.0))
        lp_solver = str(params.get("lp_solver", "cplex_direct"))

        # Determine T and Q from candidate if not configured
        q_idx, t_idx = self._parse_candidate_indices(candidate)
        T_cand = (max(t_idx) + 1) if t_idx else int(params.get("T", 0))
        T = int(params.get("T", T_cand))

        # Capacities induced by master decisions: C_*[tau] = S * sum_q y*_{q,tau}
        C_out = [0.0 for _ in range(T)]
        C_ret = [0.0 for _ in range(T)]
        for name, val in candidate.items():
            if not isinstance(name, str):
                continue
            if name.startswith("yOUT["):
                inside = name[name.find("[") + 1 : name.find("]")]
                _, tau_str = inside.split(",")
                tau = int(tau_str.strip())
                if 0 <= tau < T:
                    C_out[tau] += float(val) * S
            elif name.startswith("yRET["):
                inside = name[name.find("[") + 1 : name.find("]")]
                _, tau_str = inside.split(",")
                tau = int(tau_str.strip())
                if 0 <= tau < T:
                    C_ret[tau] += float(val) * S

        # Multi-scenario support
        scenarios: list[dict] = list(params.get("scenarios", []))
        average_cuts: bool = bool(params.get("average_cuts_across_scenarios", False))
        ub_aggregation: str = str(params.get("ub_aggregation", "mean"))
        weights: list[float] | None = params.get("scenario_weights")

        # Finite-difference coefficient builder: for each tau, solve with +S capacity
        def coeffs_by_fdiff(
            ub_base: float,
            C_out_base: list[float],
            C_ret_base: list[float],
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
                _, ub_plus = solve_subproblem(SPParams(T=T, Wmax_slots=Wmax, p=p_pen, lp_solver=lp_solver), C_out_tau, C_ret_base, R_out_vec, R_ret_vec)
                dm = float(ub_plus - ub_base)
                dm_out[tau] = dm

                # RET marginal
                C_ret_tau = C_ret_base.copy()
                C_ret_tau[tau] = C_ret_tau[tau] + S
                _, ub_plus_r = solve_subproblem(SPParams(T=T, Wmax_slots=Wmax, p=p_pen, lp_solver=lp_solver), C_out_base, C_ret_tau, R_out_vec, R_ret_vec)
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
                R_out = list(s.get("R_out", [0.0] * T))
                R_ret = list(s.get("R_ret", [0.0] * T))
                R_out = (R_out + [0.0] * T)[:T]
                R_ret = (R_ret + [0.0] * T)[:T]

                sp_params = SPParams(T=T, Wmax_slots=Wmax, p=p_pen, lp_solver=lp_solver)
                duals, ub_val = solve_subproblem(sp_params, C_out, C_ret, R_out, R_ret)
                ub_vals.append(ub_val)

                # Finite-difference coefficients and constant per scenario
                c_out_map, c_ret_map, dm_out, dm_ret = coeffs_by_fdiff(ub_val, C_out, C_ret, R_out, R_ret)
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
            # Single-demand case from params
            R_out = list(params.get("R_out", [0.0] * T))
            R_ret = list(params.get("R_ret", [0.0] * T))
            if len(R_out) != T:
                R_out = (R_out + [0.0] * T)[:T]
            if len(R_ret) != T:
                R_ret = (R_ret + [0.0] * T)[:T]

            sp_params = SPParams(T=T, Wmax_slots=Wmax, p=p_pen, lp_solver=lp_solver)
            duals, ub_val = solve_subproblem(sp_params, C_out, C_ret, R_out, R_ret)

            # Build coefficients via finite differences around current capacity
            c_out_map, c_ret_map, dm_out, dm_ret = coeffs_by_fdiff(ub_val, C_out, C_ret, R_out, R_ret)

            # Compute sum_y per tau (number of vehicles starting at tau) from current capacity
            sum_y_out = [float(C_out[tau]) / S if S != 0 else 0.0 for tau in range(T)]
            sum_y_ret = [float(C_ret[tau]) / S if S != 0 else 0.0 for tau in range(T)]

            # Constant so that cut passes through (y, Q(y))
            const = float(ub_val)
            const -= sum(dm_out[tau] * sum_y_out[tau] for tau in range(T))
            const -= sum(dm_ret[tau] * sum_y_ret[tau] for tau in range(T))

            cut = Cut(name="opt_cut", cut_type=CutType.OPTIMALITY, metadata={"const": const, "coeff_yOUT": c_out_map, "coeff_yRET": c_ret_map})
            return SubproblemResult(is_feasible=True, cut=cut, upper_bound=ub_val)


@dataclass
class SPParams:
    T: int
    Wmax_slots: int
    p: float
    lp_solver: str


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

    # Define only valid arcs (t <= tau <= min(T-1, t+W)) to avoid unused variables
    Arcs_list = [(t, tau) for t in Tset for tau in Tset if t <= tau <= min(P.T - 1, t + W)]
    m.Arcs = pyo.Set(initialize=Arcs_list, dimen=2, ordered=False)

    # Variables defined on valid arcs only
    m.x_OUT = pyo.Var(m.Arcs, within=pyo.NonNegativeReals)
    m.x_RET = pyo.Var(m.Arcs, within=pyo.NonNegativeReals)
    m.u_OUT = pyo.Var(Tset, within=pyo.NonNegativeReals)
    m.u_RET = pyo.Var(Tset, within=pyo.NonNegativeReals)

    def wait_cost(t: int, tau: int) -> float:
        return float(max(0, tau - t))

    # Objective sums over valid arcs only
    m.obj = pyo.Objective(
        expr=
            sum(wait_cost(t, tau) * m.x_OUT[t, tau] for (t, tau) in m.Arcs)
            + sum(wait_cost(t, tau) * m.x_RET[t, tau] for (t, tau) in m.Arcs)
            + P.p * (sum(m.u_OUT[t] for t in Tset) + sum(m.u_RET[t] for t in Tset)),
        sense=pyo.minimize,
    )

    def cons_dem_OUT(m, t):
        taus = [tau for tau in Tset if t <= tau <= min(P.T - 1, t + W)]
        return sum(m.x_OUT[t, tau] for tau in taus) + m.u_OUT[t] == R_out[t]

    m.D_out = pyo.Constraint(Tset, rule=cons_dem_OUT)

    def cons_dem_RET(m, t):
        taus = [tau for tau in Tset if t <= tau <= min(P.T - 1, t + W)]
        return sum(m.x_RET[t, tau] for tau in taus) + m.u_RET[t] == R_ret[t]

    m.D_ret = pyo.Constraint(Tset, rule=cons_dem_RET)

    def cap_out(m, tau):
        ts = [t for t in Tset if t <= tau <= min(P.T - 1, t + W)]
        return sum(m.x_OUT[t, tau] for t in ts) <= C_out[tau]

    m.Cap_out = pyo.Constraint(Tset, rule=cap_out)

    def cap_ret(m, tau):
        ts = [t for t in Tset if t <= tau <= min(P.T - 1, t + W)]
        return sum(m.x_RET[t, tau] for t in ts) <= C_ret[tau]

    m.Cap_ret = pyo.Constraint(Tset, rule=cap_ret)

    # Dual suffix required to read duals
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    solver = pyo.SolverFactory(P.lp_solver)
    solver.solve(m, tee=False)

    alpha_OUT = {t: float(m.dual.get(m.D_out[t], 0.0)) for t in Tset}
    alpha_RET = {t: float(m.dual.get(m.D_ret[t], 0.0)) for t in Tset}
    pi_OUT = {tau: float(m.dual.get(m.Cap_out[tau], 0.0)) for tau in Tset}
    pi_RET = {tau: float(m.dual.get(m.Cap_ret[tau], 0.0)) for tau in Tset}

    obj_val = float(pyo.value(m.obj))
    return (
        {"alpha_OUT": alpha_OUT, "alpha_RET": alpha_RET, "pi_OUT": pi_OUT, "pi_RET": pi_RET},
        obj_val,
    )
