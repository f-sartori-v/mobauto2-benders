from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pyomo.environ as pyo

from .model import MobautoMilpModel
from .cplex_log import parse_cplex_log_bounds
from .solver import RunResult

try:
    import yaml as _yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _yaml = None


@dataclass(slots=True)
class ScenarioData:
    label: str
    R_out: list[float]
    R_ret: list[float]
    weight: float



def wmax_minutes_to_slots(wmax_minutes: float, slot_resolution: int) -> int:
    """Largest whole number of slots whose implied wait never exceeds `wmax_minutes`.

    FLOOR, not ceil. `ceil` grants MORE waiting than the config asked for: at 30-minute
    slots, `Wmax_minutes: 45` becomes ceil(1.5) = 2 slots, and a passenger may then be
    made to wait a full 60 minutes against a stated cap of 45. The cap is a service
    promise, so rounding it up is the one direction that must not happen silently.

    Measured consequence of the old behaviour, on the shipped settings: none. At
    Wmax_minutes 60 and slot_resolution 30 the quotient is exactly 2, so ceil and floor
    agree and no result in this repository changes. The fix matters for any Wmax that is
    not a whole multiple of the slot width, and for the finer grids the multi-resolution
    work introduces.

    NOTE what this does and does not guarantee. It bounds the wait measured from the
    START of the arrival slot to the START of the departure slot -- which is exactly what
    `minute_pricer`'s default (`start`, D76) prices, since the departure's own committed
    instant IS its slot's start (D7/D8's t+1 rule). A passenger arriving mid-slot waits
    less than this bound, never more, under that convention. The risk described in the
    original note here -- a departure ASSUMED to leave mid-slot or later (the `midpoint`/
    `end` counterfactuals) adding up to half a slot and pushing the true wait past the cap
    (at 30-minute slots, up to 75 minutes against a stated 60) -- only arises if one of
    those counterfactuals is deliberately selected; it is not a property of the default
    path. `minute_pricer` still enforces the real cap in minutes whenever it prices a
    schedule, regardless of which convention is asked for.
    """
    import math as _math

    slot_resolution = int(slot_resolution)
    if slot_resolution <= 0:
        raise ValueError(
            f"slot_resolution must be positive to convert Wmax_minutes, "
            f"got {slot_resolution!r}"
        )
    slots = int(_math.floor(float(wmax_minutes) / float(slot_resolution)))
    if slots < 1:
        raise ValueError(
            f"Wmax_minutes={wmax_minutes!r} is shorter than one slot "
            f"({slot_resolution} min), so no departure can serve any demand inside the "
            "cap and every passenger would be counted as unserved. Raise Wmax_minutes "
            "to at least the slot width, or refine slot_resolution. (The previous "
            "ceil() hid this by silently rounding the cap up to one slot.)"
        )
    return slots

class MonolithSolver:
    def __init__(
        self,
        master_cls: type[MobautoMilpModel],
        cfg,
        master_params: dict[str, Any],
        subproblem_params: dict[str, Any],
    ) -> None:
        self.cfg = cfg
        self.master_params = dict(master_params)
        self.subproblem_params = dict(subproblem_params)
        self.master = master_cls(self.master_params)
        self._scenarios: list[ScenarioData] = []
        self._last_stats: dict[str, Any] = {}
        self._last_diagnostics: dict[str, Any] = {}
        self._slot_resolution = int(self.subproblem_params.get("slot_resolution", self.master_params.get("slot_resolution", 1)))
        if "Wmax_minutes" in self.subproblem_params:
            self._Wmax = wmax_minutes_to_slots(
                float(self.subproblem_params.get("Wmax_minutes", 0.0)),
                self._slot_resolution,
            )
        else:
            self._Wmax = int(self.subproblem_params.get("Wmax_slots", self.subproblem_params.get("Wmax", 0)))

    def run(self) -> RunResult:
        started_at = datetime.now()
        t0 = time.perf_counter()
        self.master.initialize()
        assert self.master.m is not None
        model = self.master.m
        self._scenarios = self._load_scenarios(len(list(model.T)))
        self._attach_recourse_model(model, self._scenarios)

        solve_result = self.master.solve()
        stats = self.master.last_solve_stats()
        self._last_stats = dict(stats)
        self._last_stats["elapsed_wall_s"] = time.perf_counter() - t0
        try:
            log_info = parse_cplex_log_bounds(self.master.last_solver_log_path())
            if log_info.get("first_incumbent_time_s") is not None:
                self._last_stats["first_incumbent_time_s"] = log_info.get("first_incumbent_time_s")
        except Exception:
            pass
        objective = solve_result.objective
        best_bound = solve_result.lower_bound
        incumbent = stats.get("incumbent")
        if objective is None and incumbent is not None:
            try:
                objective = float(incumbent)
            except Exception:
                objective = None
        diagnostics: dict[str, Any] = {}
        if self._has_service_solution(model):
            diagnostics = self._collect_diagnostics(model, self._scenarios)
            self._last_diagnostics = dict(diagnostics)
        else:
            self._last_diagnostics = {
                "warning": (
                    "Detailed service diagnostics unavailable because the solver did not load "
                    "the service variables for an incumbent solution."
                )
            }
        finished_at = datetime.now()
        elapsed_wall_s = self._last_stats.get("elapsed_wall_s")

        return RunResult(
            status=solve_result.status,
            iterations=1,
            best_lower_bound=best_bound,
            best_upper_bound=objective,
            pax_served=diagnostics.get("pax_served"),
            pax_total=diagnostics.get("pax_total"),
            subproblem_obj=diagnostics.get("subproblem_obj"),
            sp_wait_cost_slots=diagnostics.get("sp_wait_cost_slots"),
            sp_fill_eps_cost=diagnostics.get("sp_fill_eps_cost"),
            sp_penalty_cost=diagnostics.get("sp_penalty_cost"),
            sp_penalty_pax=diagnostics.get("sp_penalty_pax"),
            sp_total_demand=diagnostics.get("sp_total_demand"),
            sp_slot_resolution=self._slot_resolution,
            solve_started_at=started_at,
            solve_finished_at=finished_at,
            elapsed_wall_s=float(elapsed_wall_s) if elapsed_wall_s is not None else None,
        )

    def format_solution(self) -> str:
        return self.master.format_solution()

    def has_incumbent_solution(self) -> bool:
        try:
            m = self.master.m
            if m is None:
                return False
            Q = list(m.Q)
            T = list(m.T)
            if not Q or not T:
                return False
            probe_t = T[1] if len(T) > 1 else T[0]
            return pyo.value(m.b[Q[0], probe_t], exception=False) is not None
        except Exception:
            return False

    def format_report(self) -> str:
        if not self._last_diagnostics:
            return ""
        diag = self._last_diagnostics
        lines: list[str] = []
        warning = diag.get("warning")
        if warning:
            return str(warning)

        def _fmt_header(T: int) -> str:
            return "         " + " ".join(f"{t:>3d}" for t in range(T))

        def _fmt_row(vals: list[float], T: int) -> str:
            seq = (list(vals) + [0.0] * T)[:T]
            return " ".join(f"{float(v):>3.0f}" for v in seq)

        def _emit_matrix(title: str, mat: list[list[float]], T: int) -> None:
            lines.append(title)
            lines.append(_fmt_header(T))
            for q, row in enumerate(mat):
                lines.append(f"  q={q}:   {_fmt_row(list(row), T)}")

        scenarios = diag.get("scenarios") or []
        served_out = diag.get("pax_served_out")
        total_out = diag.get("pax_total_out")
        served_ret = diag.get("pax_served_ret")
        total_ret = diag.get("pax_total_ret")
        if served_out is not None and total_out is not None and served_ret is not None and total_ret is not None:
            lines.append("Pax carried by direction:")
            lines.append(f"  OUT: {float(served_out):.0f}/{float(total_out):.0f}")
            lines.append(f"  RET: {float(served_ret):.0f}/{float(total_ret):.0f}")
        for idx, sdiag in enumerate(scenarios):
            T = int(sdiag.get("T", 0))
            label = str(sdiag.get("label", f"scenario_{idx}"))
            multi = len(scenarios) > 1
            if multi:
                lines.append(f"\nScenario {idx + 1}: {label}")
            lines.append("\nDemand per slot (OUT/RET):")
            lines.append(_fmt_header(T))
            lines.append(f"    OUT: {_fmt_row(list(sdiag.get('R_out') or []), T)}")
            lines.append(f"    RET: {_fmt_row(list(sdiag.get('R_ret') or []), T)}")
            lines.append("")
            _emit_matrix("Pax per shuttle and slot (TOTAL):", list(sdiag.get("pax_total_by_q_tau") or []), T)

        try:
            total_time = self._last_stats.get("elapsed_wall_s", self._last_stats.get("wall_time", self._last_stats.get("solver_time")))
            nodes = self._last_stats.get("nodes")
            gap = self._last_stats.get("gap")
            best_bound = self._last_stats.get("best_bound")
            incumbent = self._last_stats.get("incumbent")
            first_inc_time = self._last_stats.get("first_incumbent_time_s")
            if total_time is not None or nodes is not None or gap is not None:
                lines.append("\n=== Timing Summary ===")
                if total_time is not None:
                    lines.append(f"Total solve time: {float(total_time):.3f} seconds")
                if first_inc_time is not None:
                    lines.append(f"Time to first incumbent: {float(first_inc_time):.3f} seconds")
                stat_parts: list[str] = []
                if nodes is not None:
                    stat_parts.append(f"nodes={int(nodes)}")
                if gap is not None:
                    stat_parts.append(f"gap={100.0 * float(gap):.3f}%")
                if stat_parts:
                    lines.append("Solver stats: " + " ".join(stat_parts))
                bound_parts: list[str] = []
                if best_bound is not None:
                    bound_parts.append(f"best_bound={float(best_bound):.6g}")
                if incumbent is not None:
                    bound_parts.append(f"incumbent={float(incumbent):.6g}")
                if bound_parts:
                    lines.append("Bounds: " + " ".join(bound_parts))
        except Exception:
            pass

        return "\n".join(lines).strip()

    def _has_service_solution(self, m: pyo.ConcreteModel) -> bool:
        try:
            scen_ids = list(getattr(m, "MonoScenarios", []))
            times = list(getattr(m, "T", []))
            if not scen_ids or not times:
                return False
            sample = pyo.value(m.u_OUT[scen_ids[0], times[0]], exception=False)
            return sample is not None
        except Exception:
            return False

    def _load_doc(self, path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Demand file not found: {path}")
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        if path.suffix.lower() in {".yaml", ".yml"}:
            if _yaml is None:
                raise RuntimeError("PyYAML is required to read YAML demand files.")
            return _yaml.safe_load(path.read_text(encoding="utf-8"))
        return json.loads(path.read_text(encoding="utf-8"))

    def _slot_idx_from_minutes(self, value: float) -> int:
        return max(0, int(math.floor(float(value) / max(1, self._slot_resolution))))

    def _aggregate_requests(self, container: Any, T: int) -> tuple[list[float], list[float]]:
        R_out = [0.0] * T
        R_ret = [0.0] * T
        if container is None:
            return R_out, R_ret
        if isinstance(container, dict) and ("R_out" in container or "R_ret" in container):
            rout = list(container.get("R_out", [0.0] * T))
            rret = list(container.get("R_ret", [0.0] * T))
            return ([float(x) for x in (rout + [0.0] * T)[:T]], [float(x) for x in (rret + [0.0] * T)[:T]])
        if isinstance(container, dict):
            container = container.get("requests") or container.get("req_matrix") or []
        if isinstance(container, list) and container and isinstance(container[0], dict):
            for req in container:
                direction = req.get("dir")
                try:
                    t = self._slot_idx_from_minutes(float(req.get("time", -1)))
                except Exception:
                    continue
                if not (0 <= t < T):
                    continue
                if isinstance(direction, str):
                    dd = direction.upper()
                    if dd == "OUT":
                        R_out[t] += 1.0
                    elif dd == "RET":
                        R_ret[t] += 1.0
                else:
                    if int(direction) == 0:
                        R_out[t] += 1.0
                    else:
                        R_ret[t] += 1.0
            return R_out, R_ret
        if isinstance(container, list):
            for row in container:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                direction, raw_time = row[0], row[1]
                try:
                    t = self._slot_idx_from_minutes(float(raw_time))
                except Exception:
                    continue
                if not (0 <= t < T):
                    continue
                if isinstance(direction, str):
                    dd = direction.upper()
                    if dd == "OUT":
                        R_out[t] += 1.0
                    elif dd == "RET":
                        R_ret[t] += 1.0
                else:
                    if int(direction) == 0:
                        R_out[t] += 1.0
                    else:
                        R_ret[t] += 1.0
        return R_out, R_ret

    def _pad_series(self, values: Any, T: int) -> list[float]:
        seq = [float(x) for x in list(values or [])]
        return (seq + [0.0] * T)[:T]

    def _load_single_scenario(self, payload: Any, T: int, label: str) -> ScenarioData:
        if isinstance(payload, (str, Path)):
            doc = self._load_doc(Path(str(payload)))
        else:
            doc = payload
        R_out, R_ret = self._aggregate_requests(doc, T)
        return ScenarioData(label=label, R_out=R_out, R_ret=R_ret, weight=1.0)

    def _load_scenarios(self, T: int) -> list[ScenarioData]:
        params = self.subproblem_params
        scenarios_raw = params.get("scenarios")
        scenario_files = params.get("scenario_files")
        weights = params.get("scenario_weights")

        scenarios: list[ScenarioData] = []
        if isinstance(scenarios_raw, list) and scenarios_raw:
            for idx, payload in enumerate(scenarios_raw):
                scenarios.append(self._load_single_scenario(payload, T, f"scenario_{idx}"))
        elif isinstance(scenario_files, list) and scenario_files:
            for idx, payload in enumerate(scenario_files):
                scenarios.append(self._load_single_scenario(payload, T, Path(str(payload)).stem))
        else:
            demand_file = params.get("demand_file")
            if demand_file is not None:
                scenarios.append(self._load_single_scenario(demand_file, T, Path(str(demand_file)).stem))
            else:
                scenarios.append(
                    ScenarioData(
                        label="base",
                        R_out=self._pad_series(params.get("R_out"), T),
                        R_ret=self._pad_series(params.get("R_ret"), T),
                        weight=1.0,
                    )
                )

        if not scenarios:
            raise ValueError("No demand data found for monolithic solve.")

        if not isinstance(weights, list) or len(weights) != len(scenarios):
            weights = [1.0 / float(len(scenarios)) for _ in scenarios]
        else:
            total_w = sum(float(w) for w in weights)
            if total_w <= 0.0:
                weights = [1.0 / float(len(scenarios)) for _ in scenarios]
            else:
                weights = [float(w) / float(total_w) for w in weights]

        for scen, weight in zip(scenarios, weights):
            scen.weight = float(weight)
            scen.R_out = (list(scen.R_out) + [0.0] * T)[:T]
            scen.R_ret = (list(scen.R_ret) + [0.0] * T)[:T]
        return scenarios

    def _attach_recourse_model(self, m: pyo.ConcreteModel, scenarios: list[ScenarioData]) -> None:
        if bool(self.subproblem_params.get("aggregate_service_by_start", True)):
            self._attach_aggregate_recourse_model(m, scenarios)
            return

        Q = list(m.Q)
        T = list(m.T)
        seat_capacity = float(self.subproblem_params.get("S", 0.0))
        penalty = float(self.subproblem_params.get("p", 0.0))
        fill_eps = float(self.subproblem_params.get("fill_first_epsilon", 0.0) or 0.0)

        scen_ids = list(range(len(scenarios)))
        arc_out: list[tuple[int, int, int, int]] = []
        arc_ret: list[tuple[int, int, int, int]] = []
        demand_out: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        demand_ret: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        cap_out: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = {}
        cap_ret: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = {}

        for s in scen_ids:
            for q in Q:
                for tau in T:
                    cap_out[s, q, tau] = []
                    cap_ret[s, q, tau] = []
            for t in T:
                demand_out[s, t] = []
                demand_ret[s, t] = []
                tau_max = min(T[-1], t + self._Wmax) if T else -1
                for tau in range(t + 1, tau_max + 1):
                    for q in Q:
                        idx = (s, q, t, tau)
                        arc_out.append(idx)
                        arc_ret.append(idx)
                        demand_out[s, t].append(idx)
                        demand_ret[s, t].append(idx)
                        cap_out[s, q, tau].append(idx)
                        cap_ret[s, q, tau].append(idx)

        m.MonoScenarios = pyo.Set(initialize=scen_ids, ordered=True)
        m.MonoArcsOut = pyo.Set(initialize=arc_out, dimen=4, ordered=False)
        m.MonoArcsRet = pyo.Set(initialize=arc_ret, dimen=4, ordered=False)
        m.x_OUT = pyo.Var(m.MonoArcsOut, within=pyo.NonNegativeReals)
        m.x_RET = pyo.Var(m.MonoArcsRet, within=pyo.NonNegativeReals)
        m.u_OUT = pyo.Var(m.MonoScenarios, m.T, within=pyo.NonNegativeReals)
        m.u_RET = pyo.Var(m.MonoScenarios, m.T, within=pyo.NonNegativeReals)

        m.MonoDemandOut = pyo.Constraint(
            m.MonoScenarios,
            m.T,
            rule=lambda model, s, t: sum(model.x_OUT[idx] for idx in demand_out[s, t]) + model.u_OUT[s, t] == float(scenarios[int(s)].R_out[int(t)]),
        )
        m.MonoDemandRet = pyo.Constraint(
            m.MonoScenarios,
            m.T,
            rule=lambda model, s, t: sum(model.x_RET[idx] for idx in demand_ret[s, t]) + model.u_RET[s, t] == float(scenarios[int(s)].R_ret[int(t)]),
        )
        m.MonoCapOut = pyo.Constraint(
            m.MonoScenarios,
            m.Q,
            m.T,
            rule=lambda model, s, q, tau: sum(model.x_OUT[idx] for idx in cap_out[s, q, tau]) <= seat_capacity * model.yOUT[q, tau],
        )
        m.MonoCapRet = pyo.Constraint(
            m.MonoScenarios,
            m.Q,
            m.T,
            rule=lambda model, s, q, tau: sum(model.x_RET[idx] for idx in cap_ret[s, q, tau]) <= seat_capacity * model.yRET[q, tau],
        )

        def _out_cost_expr(s: int):
            return (
                sum((float(tau - t) + fill_eps * float(q)) * m.x_OUT[s, q, t, tau] for (ss, q, t, tau) in arc_out if ss == s)
                + penalty * sum(m.u_OUT[s, t] for t in T)
            )

        def _ret_cost_expr(s: int):
            return (
                sum((float(tau - t) + fill_eps * float(q)) * m.x_RET[s, q, t, tau] for (ss, q, t, tau) in arc_ret if ss == s)
                + penalty * sum(m.u_RET[s, t] for t in T)
            )

        m.MonoScenarioCostOut = pyo.Expression(m.MonoScenarios, rule=lambda model, s: _out_cost_expr(int(s)))
        m.MonoScenarioCostRet = pyo.Expression(m.MonoScenarios, rule=lambda model, s: _ret_cost_expr(int(s)))
        m.MonoScenarioCost = pyo.Expression(
            m.MonoScenarios,
            rule=lambda model, s: model.MonoScenarioCostOut[s] + model.MonoScenarioCostRet[s],
        )

        weights = [float(s.weight) for s in scenarios]
        if hasattr(m, "theta_s"):
            m.MonoThetaLink = pyo.Constraint(
                m.MonoScenarios,
                rule=lambda model, s: model.theta_s[s] == model.MonoScenarioCost[s],
            )
        elif hasattr(m, "theta_out") and hasattr(m, "theta_ret"):
            m.MonoThetaOutLink = pyo.Constraint(
                expr=m.theta_out == sum(weights[s] * m.MonoScenarioCostOut[s] for s in scen_ids)
            )
            m.MonoThetaRetLink = pyo.Constraint(
                expr=m.theta_ret == sum(weights[s] * m.MonoScenarioCostRet[s] for s in scen_ids)
            )
        elif hasattr(m, "theta"):
            m.MonoThetaLink = pyo.Constraint(
                expr=m.theta == sum(weights[s] * m.MonoScenarioCost[s] for s in scen_ids)
            )
        else:
            raise RuntimeError("Master model does not expose theta variables for monolithic linkage.")

    def _attach_aggregate_recourse_model(self, m: pyo.ConcreteModel, scenarios: list[ScenarioData]) -> None:
        T = list(m.T)
        seat_capacity = float(self.subproblem_params.get("S", 0.0))
        penalty = float(self.subproblem_params.get("p", 0.0))

        scen_ids = list(range(len(scenarios)))
        arc_out: list[tuple[int, int, int]] = []
        arc_ret: list[tuple[int, int, int]] = []
        demand_out: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        demand_ret: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        cap_out: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        cap_ret: dict[tuple[int, int], list[tuple[int, int, int]]] = {}

        for s in scen_ids:
            for tau in T:
                cap_out[s, tau] = []
                cap_ret[s, tau] = []
            for t in T:
                demand_out[s, t] = []
                demand_ret[s, t] = []
                tau_max = min(T[-1], t + self._Wmax) if T else -1
                for tau in range(t + 1, tau_max + 1):
                    idx = (s, t, tau)
                    arc_out.append(idx)
                    arc_ret.append(idx)
                    demand_out[s, t].append(idx)
                    demand_ret[s, t].append(idx)
                    cap_out[s, tau].append(idx)
                    cap_ret[s, tau].append(idx)

        m.MonoScenarios = pyo.Set(initialize=scen_ids, ordered=True)
        m.MonoArcsOut = pyo.Set(initialize=arc_out, dimen=3, ordered=False)
        m.MonoArcsRet = pyo.Set(initialize=arc_ret, dimen=3, ordered=False)
        m.x_OUT = pyo.Var(m.MonoArcsOut, within=pyo.NonNegativeReals)
        m.x_RET = pyo.Var(m.MonoArcsRet, within=pyo.NonNegativeReals)
        m.u_OUT = pyo.Var(m.MonoScenarios, m.T, within=pyo.NonNegativeReals)
        m.u_RET = pyo.Var(m.MonoScenarios, m.T, within=pyo.NonNegativeReals)
        m.aggregate_service_by_start = True

        m.MonoDemandOut = pyo.Constraint(
            m.MonoScenarios,
            m.T,
            rule=lambda model, s, t: sum(model.x_OUT[idx] for idx in demand_out[s, t]) + model.u_OUT[s, t] == float(scenarios[int(s)].R_out[int(t)]),
        )
        m.MonoDemandRet = pyo.Constraint(
            m.MonoScenarios,
            m.T,
            rule=lambda model, s, t: sum(model.x_RET[idx] for idx in demand_ret[s, t]) + model.u_RET[s, t] == float(scenarios[int(s)].R_ret[int(t)]),
        )
        m.MonoCapOut = pyo.Constraint(
            m.MonoScenarios,
            m.T,
            rule=lambda model, s, tau: sum(model.x_OUT[idx] for idx in cap_out[s, tau]) <= seat_capacity * model.Yout[tau],
        )
        m.MonoCapRet = pyo.Constraint(
            m.MonoScenarios,
            m.T,
            rule=lambda model, s, tau: sum(model.x_RET[idx] for idx in cap_ret[s, tau]) <= seat_capacity * model.Yret[tau],
        )

        def _out_cost_expr(s: int):
            return (
                sum(float(tau - t) * m.x_OUT[s, t, tau] for (ss, t, tau) in arc_out if ss == s)
                + penalty * sum(m.u_OUT[s, t] for t in T)
            )

        def _ret_cost_expr(s: int):
            return (
                sum(float(tau - t) * m.x_RET[s, t, tau] for (ss, t, tau) in arc_ret if ss == s)
                + penalty * sum(m.u_RET[s, t] for t in T)
            )

        m.MonoScenarioCostOut = pyo.Expression(m.MonoScenarios, rule=lambda model, s: _out_cost_expr(int(s)))
        m.MonoScenarioCostRet = pyo.Expression(m.MonoScenarios, rule=lambda model, s: _ret_cost_expr(int(s)))
        m.MonoScenarioCost = pyo.Expression(
            m.MonoScenarios,
            rule=lambda model, s: model.MonoScenarioCostOut[s] + model.MonoScenarioCostRet[s],
        )

        weights = [float(s.weight) for s in scenarios]
        if hasattr(m, "theta_s"):
            m.MonoThetaLink = pyo.Constraint(
                m.MonoScenarios,
                rule=lambda model, s: model.theta_s[s] == model.MonoScenarioCost[s],
            )
        elif hasattr(m, "theta_out") and hasattr(m, "theta_ret"):
            m.MonoThetaOutLink = pyo.Constraint(
                expr=m.theta_out == sum(weights[s] * m.MonoScenarioCostOut[s] for s in scen_ids)
            )
            m.MonoThetaRetLink = pyo.Constraint(
                expr=m.theta_ret == sum(weights[s] * m.MonoScenarioCostRet[s] for s in scen_ids)
            )
        elif hasattr(m, "theta"):
            m.MonoThetaLink = pyo.Constraint(
                expr=m.theta == sum(weights[s] * m.MonoScenarioCost[s] for s in scen_ids)
            )
        else:
            raise RuntimeError("Master model does not expose theta variables for monolithic linkage.")

    def _collect_diagnostics(self, m: pyo.ConcreteModel, scenarios: list[ScenarioData]) -> dict[str, Any]:
        if bool(getattr(m, "aggregate_service_by_start", False)):
            return self._collect_aggregate_diagnostics(m, scenarios)

        Q = list(m.Q)
        T = list(m.T)
        fill_eps = float(self.subproblem_params.get("fill_first_epsilon", 0.0) or 0.0)
        penalty = float(self.subproblem_params.get("p", 0.0))
        wait_total = 0.0
        fill_total = 0.0
        penalty_cost_total = 0.0
        penalty_pax_total = 0.0
        demand_total = 0.0
        served_total = 0.0
        served_out_total = 0.0
        served_ret_total = 0.0
        demand_out_total = 0.0
        demand_ret_total = 0.0
        subproblem_obj = 0.0
        scenario_diags: list[dict[str, Any]] = []

        for s, scen in enumerate(scenarios):
            scen_wait = 0.0
            scen_fill = 0.0
            scen_pen_pax = 0.0
            scen_served = 0.0
            pax_out_by_q_tau = [[0.0 for _ in T] for _ in Q]
            pax_ret_by_q_tau = [[0.0 for _ in T] for _ in Q]
            for t in T:
                scen_pen_pax += float(pyo.value(m.u_OUT[s, t]) or 0.0)
                scen_pen_pax += float(pyo.value(m.u_RET[s, t]) or 0.0)
                for tau in range(t + 1, min(T[-1], t + self._Wmax) + 1):
                    for q in Q:
                        out_val = float(pyo.value(m.x_OUT[s, q, t, tau]) or 0.0) if (s, q, t, tau) in m.MonoArcsOut else 0.0
                        ret_val = float(pyo.value(m.x_RET[s, q, t, tau]) or 0.0) if (s, q, t, tau) in m.MonoArcsRet else 0.0
                        pax_out_by_q_tau[q][tau] += out_val
                        pax_ret_by_q_tau[q][tau] += ret_val
                        scen_wait += float(tau - t) * (out_val + ret_val)
                        scen_fill += fill_eps * float(q) * (out_val + ret_val)
                        scen_served += out_val + ret_val
            scen_penalty_cost = penalty * scen_pen_pax
            scen_served_out = float(sum(sum(row) for row in pax_out_by_q_tau))
            scen_served_ret = float(sum(sum(row) for row in pax_ret_by_q_tau))
            scen_demand_out = float(sum(scen.R_out))
            scen_demand_ret = float(sum(scen.R_ret))
            scen_demand = float(sum(scen.R_out) + sum(scen.R_ret))
            scen_total = scen_wait + scen_fill + scen_penalty_cost
            subproblem_obj += float(scen.weight) * scen_total
            wait_total += float(scen.weight) * scen_wait
            fill_total += float(scen.weight) * scen_fill
            penalty_cost_total += float(scen.weight) * scen_penalty_cost
            penalty_pax_total += float(scen.weight) * scen_pen_pax
            demand_total += float(scen.weight) * scen_demand
            served_total += float(scen.weight) * scen_served
            served_out_total += float(scen.weight) * scen_served_out
            served_ret_total += float(scen.weight) * scen_served_ret
            demand_out_total += float(scen.weight) * scen_demand_out
            demand_ret_total += float(scen.weight) * scen_demand_ret
            scenario_diags.append(
                {
                    "label": scen.label,
                    "T": len(T),
                    "R_out": list(scen.R_out),
                    "R_ret": list(scen.R_ret),
                    "pax_out_by_q_tau": pax_out_by_q_tau,
                    "pax_ret_by_q_tau": pax_ret_by_q_tau,
                    "pax_total_by_q_tau": [
                        [float(pax_out_by_q_tau[q][tau]) + float(pax_ret_by_q_tau[q][tau]) for tau in range(len(T))]
                        for q in range(len(Q))
                    ],
                }
            )

        return {
            "pax_served": served_total,
            "pax_total": demand_total,
            "pax_served_out": served_out_total,
            "pax_served_ret": served_ret_total,
            "pax_total_out": demand_out_total,
            "pax_total_ret": demand_ret_total,
            "subproblem_obj": subproblem_obj,
            "sp_wait_cost_slots": wait_total,
            "sp_fill_eps_cost": fill_total,
            "sp_penalty_cost": penalty_cost_total,
            "sp_penalty_pax": penalty_pax_total,
            "sp_total_demand": demand_total,
            "scenarios": scenario_diags,
        }

    def _collect_aggregate_diagnostics(self, m: pyo.ConcreteModel, scenarios: list[ScenarioData]) -> dict[str, Any]:
        Q = list(m.Q)
        T = list(m.T)
        seat_capacity = float(self.subproblem_params.get("S", 0.0))
        penalty = float(self.subproblem_params.get("p", 0.0))
        wait_total = 0.0
        penalty_cost_total = 0.0
        penalty_pax_total = 0.0
        demand_total = 0.0
        served_total = 0.0
        served_out_total = 0.0
        served_ret_total = 0.0
        demand_out_total = 0.0
        demand_ret_total = 0.0
        subproblem_obj = 0.0
        scenario_diags: list[dict[str, Any]] = []

        def _active_shuttles(var, tau: int) -> list[int]:
            active: list[int] = []
            for q in Q:
                if float(pyo.value(var[q, tau], exception=False) or 0.0) > 0.5:
                    active.append(int(q))
            return active

        def _distribute(mat: list[list[float]], active: list[int], tau: int, amount: float) -> None:
            remaining = float(amount)
            for q in active:
                if remaining <= 1.0e-9:
                    break
                load = min(seat_capacity, remaining)
                mat[q][tau] += load
                remaining -= load

        for s, scen in enumerate(scenarios):
            scen_wait = 0.0
            scen_pen_pax = 0.0
            scen_served = 0.0
            pax_out_by_q_tau = [[0.0 for _ in T] for _ in Q]
            pax_ret_by_q_tau = [[0.0 for _ in T] for _ in Q]
            active_out = {int(tau): _active_shuttles(m.yOUT, int(tau)) for tau in T}
            active_ret = {int(tau): _active_shuttles(m.yRET, int(tau)) for tau in T}

            for t in T:
                scen_pen_pax += float(pyo.value(m.u_OUT[s, t]) or 0.0)
                scen_pen_pax += float(pyo.value(m.u_RET[s, t]) or 0.0)
                for tau in range(t + 1, min(T[-1], t + self._Wmax) + 1):
                    out_val = float(pyo.value(m.x_OUT[s, t, tau]) or 0.0) if (s, t, tau) in m.MonoArcsOut else 0.0
                    ret_val = float(pyo.value(m.x_RET[s, t, tau]) or 0.0) if (s, t, tau) in m.MonoArcsRet else 0.0
                    _distribute(pax_out_by_q_tau, active_out.get(int(tau), []), int(tau), out_val)
                    _distribute(pax_ret_by_q_tau, active_ret.get(int(tau), []), int(tau), ret_val)
                    scen_wait += float(tau - t) * (out_val + ret_val)
                    scen_served += out_val + ret_val

            scen_penalty_cost = penalty * scen_pen_pax
            scen_served_out = float(sum(sum(row) for row in pax_out_by_q_tau))
            scen_served_ret = float(sum(sum(row) for row in pax_ret_by_q_tau))
            scen_demand_out = float(sum(scen.R_out))
            scen_demand_ret = float(sum(scen.R_ret))
            scen_demand = float(sum(scen.R_out) + sum(scen.R_ret))
            scen_total = scen_wait + scen_penalty_cost
            subproblem_obj += float(scen.weight) * scen_total
            wait_total += float(scen.weight) * scen_wait
            penalty_cost_total += float(scen.weight) * scen_penalty_cost
            penalty_pax_total += float(scen.weight) * scen_pen_pax
            demand_total += float(scen.weight) * scen_demand
            served_total += float(scen.weight) * scen_served
            served_out_total += float(scen.weight) * scen_served_out
            served_ret_total += float(scen.weight) * scen_served_ret
            demand_out_total += float(scen.weight) * scen_demand_out
            demand_ret_total += float(scen.weight) * scen_demand_ret
            scenario_diags.append(
                {
                    "label": scen.label,
                    "T": len(T),
                    "R_out": list(scen.R_out),
                    "R_ret": list(scen.R_ret),
                    "pax_out_by_q_tau": pax_out_by_q_tau,
                    "pax_ret_by_q_tau": pax_ret_by_q_tau,
                    "pax_total_by_q_tau": [
                        [float(pax_out_by_q_tau[q][tau]) + float(pax_ret_by_q_tau[q][tau]) for tau in range(len(T))]
                        for q in range(len(Q))
                    ],
                }
            )

        return {
            "pax_served": served_total,
            "pax_total": demand_total,
            "pax_served_out": served_out_total,
            "pax_served_ret": served_ret_total,
            "pax_total_out": demand_out_total,
            "pax_total_ret": demand_ret_total,
            "subproblem_obj": subproblem_obj,
            "sp_wait_cost_slots": wait_total,
            "sp_fill_eps_cost": 0.0,
            "sp_penalty_cost": penalty_cost_total,
            "sp_penalty_pax": penalty_pax_total,
            "sp_total_demand": demand_total,
            "scenarios": scenario_diags,
        }
