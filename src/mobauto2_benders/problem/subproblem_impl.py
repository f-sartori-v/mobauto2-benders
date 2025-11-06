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
    form the Benders optimality cut:

        theta >= const + S * sum_{q,tau} [pi_OUT[tau]*yOUT[q,tau] + pi_RET[tau]*yRET[q,tau]]

    where const = sum_t alpha_OUT[t]*R_out[t] + sum_t alpha_RET[t]*R_ret[t].
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
        lp_solver = str(params.get("lp_solver", "glpk"))

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

        # Helper to build coeff maps for a given pi_*
        def coeffs_from_pi(pi_OUT: Dict[int, float], pi_RET: Dict[int, float]) -> tuple[Dict[tuple[int, int], float], Dict[tuple[int, int], float]]:
            c_out: Dict[tuple[int, int], float] = {}
            c_ret: Dict[tuple[int, int], float] = {}
            for name in candidate.keys():
                if not isinstance(name, str):
                    continue
                if name.startswith("yOUT["):
                    inside = name[name.find("[") + 1 : name.find("]")]
                    q_str, tau_str = inside.split(",")
                    q = int(q_str.strip())
                    tau = int(tau_str.strip())
                    if 0 <= tau < T:
                        c_out[(q, tau)] = S * pi_OUT.get(tau, 0.0)
                elif name.startswith("yRET["):
                    inside = name[name.find("[") + 1 : name.find("]")]
                    q_str, tau_str = inside.split(",")
                    q = int(q_str.strip())
                    tau = int(tau_str.strip())
                    if 0 <= tau < T:
                        c_ret[(q, tau)] = S * pi_RET.get(tau, 0.0)
            return c_out, c_ret

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
                alpha_OUT, alpha_RET, pi_OUT, pi_RET = (
                    duals["alpha_OUT"],
                    duals["alpha_RET"],
                    duals["pi_OUT"],
                    duals["pi_RET"],
                )
                const = sum(alpha_OUT[t] * float(R_out[t]) for t in range(T)) + sum(alpha_RET[t] * float(R_ret[t]) for t in range(T))
                consts.append(const)
                c_out, c_ret = coeffs_from_pi(pi_OUT, pi_RET)
                coeffs_out_list.append(c_out)
                coeffs_ret_list.append(c_ret)
                cuts.append(Cut(name="opt_cut_s", cut_type=CutType.OPTIMALITY, metadata={"const": const, "coeff_yOUT": c_out, "coeff_yRET": c_ret}))

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
            alpha_OUT, alpha_RET, pi_OUT, pi_RET = (
                duals["alpha_OUT"],
                duals["alpha_RET"],
                duals["pi_OUT"],
                duals["pi_RET"],
            )
            const = sum(alpha_OUT[t] * float(R_out[t]) for t in range(T)) + sum(alpha_RET[t] * float(R_ret[t]) for t in range(T))
            c_out, c_ret = coeffs_from_pi(pi_OUT, pi_RET)
            cut = Cut(name="opt_cut", cut_type=CutType.OPTIMALITY, metadata={"const": const, "coeff_yOUT": c_out, "coeff_yRET": c_ret})
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

    # Variables
    m.x_OUT = pyo.Var(Tset, Tset, within=pyo.NonNegativeReals)
    m.x_RET = pyo.Var(Tset, Tset, within=pyo.NonNegativeReals)
    m.u_OUT = pyo.Var(Tset, within=pyo.NonNegativeReals)
    m.u_RET = pyo.Var(Tset, within=pyo.NonNegativeReals)

    def wait_cost(t: int, tau: int) -> float:
        return float(max(0, tau - t))

    m.obj = pyo.Objective(
        expr=sum(wait_cost(t, tau) * m.x_OUT[t, tau] for t in Tset for tau in Tset)
        + sum(wait_cost(t, tau) * m.x_RET[t, tau] for t in Tset for tau in Tset)
        + P.p * (sum(m.u_OUT[t] for t in Tset) + sum(m.u_RET[t] for t in Tset)),
        sense=pyo.minimize,
    )

    W = P.Wmax_slots

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
