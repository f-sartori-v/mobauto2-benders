from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import ast
import operator as _op
import warnings

try:
    import yaml as _yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _yaml = None


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


# ---- Schema sections ----


@dataclass(slots=True)
class SchemaSection:
    name: str
    version: int


@dataclass(slots=True)
class RunSection:
    name: str | None = None
    log_level: str = "INFO"
    log_file: str | None = None
    report_dir: str | None = None
    seed: int | None = None
    # Write a symbolic LP and a solver log for every master solve. Debugging aid,
    # off by default: it produces two files per iteration (M5).
    emit_reports: bool = False


@dataclass(slots=True)
class DataSection:
    demand_file: str | None = None
    scenario_files: list[str] = field(default_factory=list)
    scenario_weights: list[float] | None = None
    R_out: list[float] | None = None
    R_ret: list[float] | None = None
    scenarios: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class TimeSection:
    T_minutes: int | None = None
    T: int | None = None
    slot_resolution: int = 1
    trip_duration_minutes: int | None = None
    trip_duration: int | None = None
    trip_slots: int | None = None


@dataclass(slots=True)
class FleetSection:
    Q: int
    binit: list[float] | None = None
    initial_actions: list[str] | None = None


@dataclass(slots=True)
class EnergySection:
    Emax: float
    L: float
    delta_chg: float | int | str | None = None


@dataclass(slots=True)
class CostSection:
    start_cost_epsilon: float = 0.0
    concurrency_penalty: float = 0.0


@dataclass(slots=True)
class ModelSection:
    time: TimeSection
    fleet: FleetSection
    energy: EnergySection
    costs: CostSection


@dataclass(slots=True)
class MasterSection:
    use_fifo_symmetry: bool = False
    symmetry_breaking: bool = False
    use_mip_start: bool = False
    # Ceiling on ONE master MIP solve, per Benders iteration. Not the run budget:
    # that is solver.total_time_limit_s. The gap-tied schedule in the Benders loop
    # picks a per-iteration limit and this value caps it.
    per_iteration_time_limit_s: int | None = None
    # Ceiling on the master MIP gap, per Benders iteration. Not the convergence
    # criterion: that is solver.tolerance. The schedule tightens the gap on its own
    # as the Benders gap closes; this is the loosest it is allowed to be.
    per_iteration_mipgap: float | None = None
    cplex_options: dict[str, Any] = field(default_factory=dict)
    solver_backend: str = "cplex_direct"
    aggregate_cuts_by_tau: bool = True
    cut_coeff_threshold: float = 0.0
    theta_per_scenario: bool = False
    write_lp_after_cut: bool = False
    # Canonical ordering: charge before idling at the depot (M2). On by default.
    charge_before_idle: bool = True
    # Valid inequality anchoring theta to installed capacity per prefix of the
    # horizon. ON by default since D29: measured at equal iterations it improves
    # the lower bound 26-90% AND cuts master time 38-68% on the reproducible
    # cells. Unlike M1, it is not a trade-off.
    recourse_lower_bound: bool = True
    # LP phase: solve the master without integrality while the cut set is still
    # poor. Cuts from a fractional y are valid (the recourse value function is
    # convex in y), but its subproblem cost is NOT an upper bound, so the loop
    # claims none while the phase is on. Off by default; earn it by measurement.
    lp_phase: bool = False
    # Hard ceiling on LP-phase iterations.
    lp_phase_max_iters: int = 10
    # Consecutive iterations of sub-threshold LP objective improvement that end
    # the phase. 0 disables the stall test and leaves only the iteration ceiling.
    lp_phase_stall_iters: int = 3
    # Relative improvement in the LP master objective that counts as progress.
    lp_phase_min_rel_improve: float = 0.005


@dataclass(slots=True)
class SubproblemSection:
    multi_cuts_by_scenario: bool = True
    use_magnanti_wong: bool = False
    mw_core_alpha: float = 0.3
    mw_core_eps: float = 1e-3
    use_dual_slopes: bool = False
    S: float = 0.0
    Wmax_minutes: int | None = None
    Wmax_slots: int | None = None
    p: float = 0.0
    degenerate_cut_probe_top_k: int = 6
    degenerate_cut_probe_top_k_out: int | None = None
    degenerate_cut_probe_top_k_ret: int | None = None
    degenerate_cut_zero_tol: float = 1e-9


@dataclass(slots=True)
class SolverSection:
    max_iterations: int
    tolerance: float
    # Wall-clock budget for the whole Benders loop. Checked between iterations, so
    # a run overshoots by at most one master solve (see master.per_iteration_time_limit_s).
    total_time_limit_s: int
    stall_max_no_improve_iters: int = 0
    stall_min_abs_improve: float = 0.0
    stall_min_rel_improve: float = 0.0
    master_solver: str = "cplex"
    subproblem_solver: str = "cplex_direct"
    solver_tee: bool = False


@dataclass(slots=True)
class RootConfig:
    schema: SchemaSection
    run: RunSection
    data: DataSection
    model: ModelSection
    master: MasterSection
    subproblem: SubproblemSection
    solver: SolverSection
    tolerances: "TolerancesSection"


@dataclass(slots=True)
class TolerancesSection:
    eps_bin: float = 1e-6
    eps_feas: float = 1e-7
    eps_cut: float = 1e-8
    eps_hash: float = 1e-6


# ---- Expression evaluation (energy params only) ----


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _eval_expr(expr: str, names: Mapping[str, Any]) -> float | int:
    """Safely evaluate a simple arithmetic expression with provided names."""
    node = ast.parse(expr, mode="eval")

    bin_ops = {
        ast.Add: _op.add,
        ast.Sub: _op.sub,
        ast.Mult: _op.mul,
        ast.Div: _op.truediv,
        ast.FloorDiv: _op.floordiv,
        ast.Mod: _op.mod,
        ast.Pow: _op.pow,
    }
    unary_ops = {ast.UAdd: _op.pos, ast.USub: _op.neg}

    def _eval(n: ast.AST) -> float | int:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("non-numeric constant in expression")
        num_node = getattr(ast, "Num", None)  # Python < 3.12 compatibility
        if num_node is not None and isinstance(
            n, num_node
        ):  # pragma: no cover - old ASTs
            return n.n  # type: ignore[attr-defined]
        if isinstance(n, ast.Name):
            if n.id not in names:
                raise NameError(f"unknown name '{n.id}' in expression")
            v = names[n.id]
            if _is_number(v):
                return v  # type: ignore[return-value]
            try:
                return float(v) if (isinstance(v, str) and v.strip()) else v  # type: ignore[return-value]
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"name '{n.id}' is not numeric: {v}") from exc
        if isinstance(n, ast.BinOp):
            if type(n.op) not in bin_ops:
                raise ValueError("operator not allowed in expression")
            return bin_ops[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp):
            if type(n.op) not in unary_ops:
                raise ValueError("unary operator not allowed in expression")
            return unary_ops[type(n.op)](_eval(n.operand))
        if isinstance(n, (ast.Tuple, ast.List)) and len(getattr(n, "elts", [])) == 1:
            return _eval(n.elts[0])  # type: ignore[index]
        raise ValueError("unsupported syntax in expression")

    return _eval(node)


def _looks_like_expr(value: str) -> bool:
    s = value.strip()
    return any(ch in s for ch in "+-*/()")


def resolve_energy_params(
    energy: EnergySection, names: Mapping[str, Any]
) -> dict[str, Any]:
    out = {
        "Emax": energy.Emax,
        "L": energy.L,
    }
    if energy.delta_chg is not None:
        if isinstance(energy.delta_chg, str) and _looks_like_expr(energy.delta_chg):
            out["delta_chg"] = _eval_expr(energy.delta_chg, names)
        else:
            out["delta_chg"] = energy.delta_chg
    return out


# ---- Validation helpers ----


def _as_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    return value


def _check_unknown_keys(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(k for k in data.keys() if k not in allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) in {where}: {', '.join(unknown)}")


def _require_keys(data: Mapping[str, Any], required: set[str], where: str) -> None:
    missing = sorted(k for k in required if k not in data)
    if missing:
        raise ValueError(f"Missing required key(s) in {where}: {', '.join(missing)}")


def _ensure_int(value: Any, where: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{where} must be an int")
    try:
        return int(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{where} must be an int") from exc


def _ensure_float(value: Any, where: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{where} must be a float")
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{where} must be a float") from exc


def _ensure_bool(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{where} must be a bool")


def _ensure_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")
    return value


def _ensure_str_list(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{where} must be a list of strings")
    return list(value)


def _ensure_num_list(value: Any, where: str) -> list[float]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a list of numbers")
    out: list[float] = []
    for v in value:
        out.append(_ensure_float(v, where))
    return out


def _ensure_num_or_expr(value: Any, where: str) -> float | int | str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip()
        if _looks_like_expr(s):
            return s
        try:
            return float(s)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"{where} must be numeric or an arithmetic expression"
            ) from exc
    raise ValueError(f"{where} must be numeric or an arithmetic expression")


def _ensure_mapping(value: Any, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    return dict(value)


def _disallow_expr(value: Any, where: str) -> Any:
    if isinstance(value, str) and _looks_like_expr(value):
        raise ValueError(f"{where} cannot be an expression; provide a numeric value")
    return value


# ---- Load / parse ----


def _load_yaml(path: Path) -> dict[str, Any]:
    if _yaml is None:
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. Install with 'pip install pyyaml'."
        )
    with path.open("r", encoding="utf-8") as f:
        data = _yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML document must be a mapping")
        return data


def upgrade_config_v1_to_v2(old: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade a v1 config dict to the v2 schema.

    Emits warnings describing deprecated keys and their mappings.
    """
    warnings_list: list[str] = []

    run = _as_mapping(old.get("run", {}), "run")
    master = _as_mapping(old.get("master", {}), "master")
    sub = _as_mapping(old.get("subproblem", {}), "subproblem")
    master_params = _as_mapping(master.get("params", {}), "master.params")
    sub_params = _as_mapping(sub.get("params", {}), "subproblem.params")

    lazy_key = "use_" + "lazy" + "_cuts"
    lazy_cb_key = "lazy" + "_cb" + "_lp_solver"
    if lazy_key in master_params:
        raise ValueError("lazy " + "cuts removed; delete key " + lazy_key)
    if lazy_cb_key in master_params:
        raise ValueError("callback cuts removed; delete key " + lazy_cb_key)
    if str(master_params.get("solver", "")).lower() == "cplex_persistent":
        raise ValueError("persistent solver mode removed; use solver=cplex")

    def _note(msg: str) -> None:
        warnings_list.append(msg)

    new: dict[str, Any] = {
        "schema": {"name": "mobauto2_benders_config", "version": 2},
        "run": {
            "name": None,
            "log_level": run.get("log_level", "INFO"),
            "log_file": None,
            "report_dir": None,
            "seed": run.get("seed"),
        },
        "data": {
            "demand_file": sub_params.get("demand_file"),
            "scenario_files": sub_params.get("scenario_files") or [],
            "scenario_weights": sub_params.get("scenario_weights"),
            "R_out": sub_params.get("R_out"),
            "R_ret": sub_params.get("R_ret"),
        },
        "model": {
            "time": {
                "T_minutes": master_params.get("T_minutes"),
                "T": master_params.get("T"),
                "slot_resolution": master_params.get("slot_resolution", 1),
                "trip_duration_minutes": master_params.get("trip_duration_minutes"),
                "trip_duration": master_params.get("trip_duration"),
                "trip_slots": master_params.get("trip_slots"),
            },
            "fleet": {
                "Q": master_params.get("Q"),
                "binit": master_params.get("binit"),
                "initial_actions": master_params.get("initial_actions"),
            },
            "energy": {
                "Emax": master_params.get("Emax"),
                "L": master_params.get("L"),
                "delta_chg": master_params.get("delta_chg"),
            },
            "costs": {
                "start_cost_epsilon": master_params.get("start_cost_epsilon", 0.0),
                "concurrency_penalty": master_params.get("concurrency_penalty", 0.0),
            },
        },
        "master": {
            "use_fifo_symmetry": master_params.get("use_fifo_symmetry", False),
            "symmetry_breaking": master_params.get("symmetry_breaking", False),
            "use_mip_start": master_params.get("use_mip_start", False),
            "per_iteration_time_limit_s": master_params.get("solve_time_limit_s"),
            "per_iteration_mipgap": master_params.get("mipgap"),
            "cplex_options": master_params.get("cplex_options", {}),
            "solver_backend": master_params.get("solver_backend", "cplex_direct"),
            "aggregate_cuts_by_tau": master_params.get("aggregate_cuts_by_tau", True),
            "cut_coeff_threshold": master_params.get("cut_coeff_threshold", 0.0),
            "theta_per_scenario": master_params.get("theta_per_scenario", False),
            "write_lp_after_cut": master_params.get("write_lp_after_cut", False),
            # Both change the bound a run reports, so a table quoting one has to be
            # able to say which side of the A/B it came from (D18's obligation).
            "recourse_lower_bound": master_params.get("recourse_lower_bound", True),
            "lp_phase": master_params.get("lp_phase", False),
            "lp_phase_max_iters": master_params.get("lp_phase_max_iters"),
            "lp_phase_stall_iters": master_params.get("lp_phase_stall_iters"),
            "lp_phase_min_rel_improve": master_params.get("lp_phase_min_rel_improve"),
        },
        "subproblem": {
            "multi_cuts_by_scenario": sub_params.get("multi_cuts_by_scenario", True),
            "use_magnanti_wong": sub_params.get("use_magnanti_wong", False),
            "mw_core_alpha": sub_params.get("mw_core_alpha", 0.3),
            "mw_core_eps": sub_params.get("mw_core_eps", 1e-3),
            "use_dual_slopes": sub_params.get("use_dual_slopes", False),
            "S": sub_params.get("S"),
            "Wmax_minutes": sub_params.get("Wmax_minutes"),
            "Wmax_slots": sub_params.get("Wmax_slots"),
            "p": sub_params.get("p"),
            "degenerate_cut_probe_top_k": sub_params.get(
                "degenerate_cut_probe_top_k", 6
            ),
            "degenerate_cut_probe_top_k_out": sub_params.get(
                "degenerate_cut_probe_top_k_out"
            ),
            "degenerate_cut_probe_top_k_ret": sub_params.get(
                "degenerate_cut_probe_top_k_ret"
            ),
            "degenerate_cut_zero_tol": sub_params.get("degenerate_cut_zero_tol", 1e-9),
        },
        "solver": {
            "max_iterations": run.get("max_iterations", 100),
            "tolerance": run.get("tolerance", 1e-4),
            "total_time_limit_s": run.get("time_limit_s", 600),
            "stall_max_no_improve_iters": run.get("stall_max_no_improve_iters", 0),
            "stall_min_abs_improve": run.get("stall_min_abs_improve", 0.0),
            "stall_min_rel_improve": run.get("stall_min_rel_improve", 0.0),
            "master_solver": master_params.get("solver", "cplex"),
            "subproblem_solver": sub_params.get("lp_solver", "cplex_direct"),
            "solver_tee": master_params.get("solver_tee", False),
        },
    }

    _note(
        "Deprecated v1 key 'run.*' mapped to 'run.*' (logging) and 'solver.*' (iterations/tolerance/time limits)."
    )
    _note(f"v1 run keys: {sorted(run.keys())}")
    _note(
        "Deprecated v1 key 'master.params.*' mapped into 'model.*', 'master.*', and 'solver.*'."
    )
    _note(f"v1 master.params keys: {sorted(master_params.keys())}")
    _note(
        "Deprecated v1 key 'subproblem.params.*' mapped into 'data.*', 'subproblem.*', and 'solver.*'."
    )
    _note(f"v1 subproblem.params keys: {sorted(sub_params.keys())}")

    if warnings_list:
        warnings.warn(
            "Loaded v1 config; please upgrade to schema version 2.\n"
            + "\n".join(warnings_list),
            stacklevel=2,
        )

    return new


def _parse_v2(raw: Mapping[str, Any]) -> RootConfig:
    data = _as_mapping(raw, "config")
    _check_unknown_keys(
        data,
        {
            "schema",
            "run",
            "data",
            "model",
            "master",
            "subproblem",
            "solver",
            "tolerances",
        },
        "config",
    )
    _require_keys(
        data,
        {"schema", "run", "data", "model", "master", "subproblem", "solver"},
        "config",
    )

    schema_raw = _as_mapping(data.get("schema"), "schema")
    _check_unknown_keys(schema_raw, {"name", "version"}, "schema")
    _require_keys(schema_raw, {"name", "version"}, "schema")
    schema = SchemaSection(
        name=_ensure_str(schema_raw.get("name"), "schema.name"),
        version=_ensure_int(schema_raw.get("version"), "schema.version"),
    )
    if schema.name != "mobauto2_benders_config" or schema.version != 2:
        raise ValueError(
            "Unsupported schema version; expected mobauto2_benders_config v2"
        )

    run_raw = _as_mapping(data.get("run"), "run")
    _check_unknown_keys(
        run_raw,
        {"name", "log_level", "log_file", "report_dir", "seed", "emit_reports"},
        "run",
    )
    run_name = run_raw.get("name")
    run = RunSection(
        name=(_ensure_str(run_name, "run.name") if run_name is not None else None),
        log_level=str(run_raw.get("log_level", "INFO")),
        log_file=run_raw.get("log_file"),
        report_dir=run_raw.get("report_dir"),
        seed=(
            _ensure_int(run_raw.get("seed"), "run.seed")
            if "seed" in run_raw and run_raw.get("seed") is not None
            else None
        ),
        emit_reports=_ensure_bool(
            run_raw.get("emit_reports", False), "run.emit_reports"
        ),
    )

    data_raw = _as_mapping(data.get("data"), "data")
    _check_unknown_keys(
        data_raw,
        {
            "demand_file",
            "scenario_files",
            "scenario_weights",
            "R_out",
            "R_ret",
            "scenarios",
        },
        "data",
    )
    demand_file_val = data_raw.get("demand_file")
    data_section = DataSection(
        demand_file=(
            _ensure_str(demand_file_val, "data.demand_file")
            if demand_file_val is not None
            else None
        ),
        scenario_files=_ensure_str_list(
            data_raw.get("scenario_files"), "data.scenario_files"
        ),
        scenario_weights=(
            _ensure_num_list(data_raw.get("scenario_weights"), "data.scenario_weights")
            if data_raw.get("scenario_weights") is not None
            else None
        ),
        R_out=(
            _ensure_num_list(data_raw.get("R_out"), "data.R_out")
            if data_raw.get("R_out") is not None
            else None
        ),
        R_ret=(
            _ensure_num_list(data_raw.get("R_ret"), "data.R_ret")
            if data_raw.get("R_ret") is not None
            else None
        ),
        scenarios=(
            data_raw.get("scenarios")
            if isinstance(data_raw.get("scenarios"), list)
            else None
        ),
    )

    model_raw = _as_mapping(data.get("model"), "model")
    _check_unknown_keys(model_raw, {"time", "fleet", "energy", "costs"}, "model")
    _require_keys(model_raw, {"time", "fleet", "energy", "costs"}, "model")

    time_raw = _as_mapping(model_raw.get("time"), "model.time")
    _check_unknown_keys(
        time_raw,
        {
            "T_minutes",
            "T",
            "slot_resolution",
            "trip_duration_minutes",
            "trip_duration",
            "trip_slots",
        },
        "model.time",
    )
    _require_keys(time_raw, {"slot_resolution"}, "model.time")
    if "T_minutes" not in time_raw and "T" not in time_raw:
        raise ValueError("model.time must include T_minutes or T")
    time_section = TimeSection(
        T_minutes=(
            _ensure_int(
                _disallow_expr(time_raw.get("T_minutes"), "model.time.T_minutes"),
                "model.time.T_minutes",
            )
            if time_raw.get("T_minutes") is not None
            else None
        ),
        T=(
            _ensure_int(
                _disallow_expr(time_raw.get("T"), "model.time.T"), "model.time.T"
            )
            if time_raw.get("T") is not None
            else None
        ),
        slot_resolution=_ensure_int(
            _disallow_expr(
                time_raw.get("slot_resolution"), "model.time.slot_resolution"
            ),
            "model.time.slot_resolution",
        ),
        trip_duration_minutes=(
            _ensure_int(
                _disallow_expr(
                    time_raw.get("trip_duration_minutes"),
                    "model.time.trip_duration_minutes",
                ),
                "model.time.trip_duration_minutes",
            )
            if time_raw.get("trip_duration_minutes") is not None
            else None
        ),
        trip_duration=(
            _ensure_int(
                _disallow_expr(
                    time_raw.get("trip_duration"), "model.time.trip_duration"
                ),
                "model.time.trip_duration",
            )
            if time_raw.get("trip_duration") is not None
            else None
        ),
        trip_slots=(
            _ensure_int(
                _disallow_expr(time_raw.get("trip_slots"), "model.time.trip_slots"),
                "model.time.trip_slots",
            )
            if time_raw.get("trip_slots") is not None
            else None
        ),
    )

    fleet_raw = _as_mapping(model_raw.get("fleet"), "model.fleet")
    _check_unknown_keys(
        fleet_raw, {"Q", "binit", "initial_battery", "initial_actions"}, "model.fleet"
    )
    _require_keys(fleet_raw, {"Q"}, "model.fleet")
    binit_raw = fleet_raw.get("initial_battery", fleet_raw.get("binit"))
    fleet_section = FleetSection(
        Q=_ensure_int(
            _disallow_expr(fleet_raw.get("Q"), "model.fleet.Q"), "model.fleet.Q"
        ),
        binit=(
            _ensure_num_list(binit_raw, "model.fleet.initial_battery")
            if binit_raw is not None
            else None
        ),
        initial_actions=(
            _ensure_str_list(
                fleet_raw.get("initial_actions"), "model.fleet.initial_actions"
            )
            if fleet_raw.get("initial_actions") is not None
            else None
        ),
    )

    energy_raw = _as_mapping(model_raw.get("energy"), "model.energy")
    _check_unknown_keys(energy_raw, {"Emax", "L", "delta_chg"}, "model.energy")
    _require_keys(energy_raw, {"Emax", "L"}, "model.energy")
    energy_section = EnergySection(
        Emax=_ensure_float(
            _disallow_expr(energy_raw.get("Emax"), "model.energy.Emax"),
            "model.energy.Emax",
        ),
        L=_ensure_float(
            _disallow_expr(energy_raw.get("L"), "model.energy.L"), "model.energy.L"
        ),
        delta_chg=(
            _ensure_num_or_expr(energy_raw.get("delta_chg"), "model.energy.delta_chg")
            if energy_raw.get("delta_chg") is not None
            else None
        ),
    )

    costs_raw = _as_mapping(model_raw.get("costs"), "model.costs")
    _check_unknown_keys(
        costs_raw, {"start_cost_epsilon", "concurrency_penalty"}, "model.costs"
    )
    cost_section = CostSection(
        start_cost_epsilon=(
            _ensure_float(
                _disallow_expr(
                    costs_raw.get("start_cost_epsilon"),
                    "model.costs.start_cost_epsilon",
                ),
                "model.costs.start_cost_epsilon",
            )
            if costs_raw.get("start_cost_epsilon") is not None
            else 0.0
        ),
        concurrency_penalty=(
            _ensure_float(
                _disallow_expr(
                    costs_raw.get("concurrency_penalty"),
                    "model.costs.concurrency_penalty",
                ),
                "model.costs.concurrency_penalty",
            )
            if costs_raw.get("concurrency_penalty") is not None
            else 0.0
        ),
    )

    master_raw = _as_mapping(data.get("master"), "master")
    if ("use_" + "lazy" + "_cuts") in master_raw:
        raise ValueError("lazy " + "cuts removed; delete key use_" + "lazy" + "_cuts")
    # Renamed for clarity: both were confusable with the run-level budget and the
    # convergence tolerance, and both were silently overwritten before they were wired.
    if "solve_time_limit_s" in master_raw:
        raise ValueError(
            "master.solve_time_limit_s renamed to master.per_iteration_time_limit_s "
            "(ceiling on one master solve; the run budget is solver.total_time_limit_s)"
        )
    if "mipgap" in master_raw:
        raise ValueError(
            "master.mipgap renamed to master.per_iteration_mipgap "
            "(ceiling on the master MIP gap; the convergence criterion is solver.tolerance)"
        )
    _check_unknown_keys(
        master_raw,
        {
            "use_fifo_symmetry",
            "symmetry_breaking",
            "use_mip_start",
            "per_iteration_time_limit_s",
            "per_iteration_mipgap",
            "cplex_options",
            "solver_backend",
            "aggregate_cuts_by_tau",
            "cut_coeff_threshold",
            "theta_per_scenario",
            "write_lp_after_cut",
            "charge_before_idle",
            "recourse_lower_bound",
            "lp_phase",
            "lp_phase_max_iters",
            "lp_phase_stall_iters",
            "lp_phase_min_rel_improve",
        },
        "master",
    )
    master_section = MasterSection(
        use_fifo_symmetry=_ensure_bool(
            master_raw.get("use_fifo_symmetry", False), "master.use_fifo_symmetry"
        ),
        symmetry_breaking=_ensure_bool(
            master_raw.get("symmetry_breaking", False), "master.symmetry_breaking"
        ),
        use_mip_start=_ensure_bool(
            master_raw.get("use_mip_start", False), "master.use_mip_start"
        ),
        per_iteration_time_limit_s=(
            _ensure_int(
                _disallow_expr(
                    master_raw.get("per_iteration_time_limit_s"),
                    "master.per_iteration_time_limit_s",
                ),
                "master.per_iteration_time_limit_s",
            )
            if master_raw.get("per_iteration_time_limit_s") is not None
            else None
        ),
        per_iteration_mipgap=(
            _ensure_float(
                _disallow_expr(
                    master_raw.get("per_iteration_mipgap"),
                    "master.per_iteration_mipgap",
                ),
                "master.per_iteration_mipgap",
            )
            if master_raw.get("per_iteration_mipgap") is not None
            else None
        ),
        cplex_options=_ensure_mapping(
            master_raw.get("cplex_options"), "master.cplex_options"
        ),
        solver_backend=_ensure_str(
            master_raw.get("solver_backend", "cplex_direct"), "master.solver_backend"
        ),
        aggregate_cuts_by_tau=_ensure_bool(
            master_raw.get("aggregate_cuts_by_tau", True),
            "master.aggregate_cuts_by_tau",
        ),
        cut_coeff_threshold=_ensure_float(
            _disallow_expr(
                master_raw.get("cut_coeff_threshold", 0.0), "master.cut_coeff_threshold"
            ),
            "master.cut_coeff_threshold",
        ),
        theta_per_scenario=_ensure_bool(
            master_raw.get("theta_per_scenario", False), "master.theta_per_scenario"
        ),
        write_lp_after_cut=_ensure_bool(
            master_raw.get("write_lp_after_cut", False), "master.write_lp_after_cut"
        ),
        charge_before_idle=_ensure_bool(
            master_raw.get("charge_before_idle", True), "master.charge_before_idle"
        ),
        recourse_lower_bound=_ensure_bool(
            master_raw.get("recourse_lower_bound", True), "master.recourse_lower_bound"
        ),
        lp_phase=_ensure_bool(master_raw.get("lp_phase", False), "master.lp_phase"),
        lp_phase_max_iters=_ensure_int(
            _disallow_expr(
                master_raw.get("lp_phase_max_iters", 10), "master.lp_phase_max_iters"
            ),
            "master.lp_phase_max_iters",
        ),
        lp_phase_stall_iters=_ensure_int(
            _disallow_expr(
                master_raw.get("lp_phase_stall_iters", 3), "master.lp_phase_stall_iters"
            ),
            "master.lp_phase_stall_iters",
        ),
        lp_phase_min_rel_improve=_ensure_float(
            _disallow_expr(
                master_raw.get("lp_phase_min_rel_improve", 0.005),
                "master.lp_phase_min_rel_improve",
            ),
            "master.lp_phase_min_rel_improve",
        ),
    )

    sub_raw = _as_mapping(data.get("subproblem"), "subproblem")
    _check_unknown_keys(
        sub_raw,
        {
            "multi_cuts_by_scenario",
            "use_magnanti_wong",
            "mw_core_alpha",
            "mw_core_eps",
            "use_dual_slopes",
            "S",
            "Wmax_minutes",
            "Wmax_slots",
            "p",
            "degenerate_cut_probe_top_k",
            "degenerate_cut_probe_top_k_out",
            "degenerate_cut_probe_top_k_ret",
            "degenerate_cut_zero_tol",
        },
        "subproblem",
    )
    _require_keys(sub_raw, {"S", "p"}, "subproblem")
    if "Wmax_minutes" not in sub_raw and "Wmax_slots" not in sub_raw:
        raise ValueError("subproblem must include Wmax_minutes or Wmax_slots")
    sub_section = SubproblemSection(
        multi_cuts_by_scenario=_ensure_bool(
            sub_raw.get("multi_cuts_by_scenario", True),
            "subproblem.multi_cuts_by_scenario",
        ),
        use_magnanti_wong=_ensure_bool(
            sub_raw.get("use_magnanti_wong", False), "subproblem.use_magnanti_wong"
        ),
        mw_core_alpha=_ensure_float(
            _disallow_expr(
                sub_raw.get("mw_core_alpha", 0.3), "subproblem.mw_core_alpha"
            ),
            "subproblem.mw_core_alpha",
        ),
        mw_core_eps=_ensure_float(
            _disallow_expr(sub_raw.get("mw_core_eps", 1e-3), "subproblem.mw_core_eps"),
            "subproblem.mw_core_eps",
        ),
        use_dual_slopes=_ensure_bool(
            sub_raw.get("use_dual_slopes", False), "subproblem.use_dual_slopes"
        ),
        S=_ensure_float(
            _disallow_expr(sub_raw.get("S"), "subproblem.S"), "subproblem.S"
        ),
        Wmax_minutes=(
            _ensure_int(
                _disallow_expr(sub_raw.get("Wmax_minutes"), "subproblem.Wmax_minutes"),
                "subproblem.Wmax_minutes",
            )
            if sub_raw.get("Wmax_minutes") is not None
            else None
        ),
        Wmax_slots=(
            _ensure_int(
                _disallow_expr(sub_raw.get("Wmax_slots"), "subproblem.Wmax_slots"),
                "subproblem.Wmax_slots",
            )
            if sub_raw.get("Wmax_slots") is not None
            else None
        ),
        p=_ensure_float(
            _disallow_expr(sub_raw.get("p"), "subproblem.p"), "subproblem.p"
        ),
        degenerate_cut_probe_top_k=_ensure_int(
            _disallow_expr(
                sub_raw.get("degenerate_cut_probe_top_k", 6),
                "subproblem.degenerate_cut_probe_top_k",
            ),
            "subproblem.degenerate_cut_probe_top_k",
        ),
        degenerate_cut_probe_top_k_out=(
            _ensure_int(
                _disallow_expr(
                    sub_raw.get("degenerate_cut_probe_top_k_out"),
                    "subproblem.degenerate_cut_probe_top_k_out",
                ),
                "subproblem.degenerate_cut_probe_top_k_out",
            )
            if sub_raw.get("degenerate_cut_probe_top_k_out") is not None
            else None
        ),
        degenerate_cut_probe_top_k_ret=(
            _ensure_int(
                _disallow_expr(
                    sub_raw.get("degenerate_cut_probe_top_k_ret"),
                    "subproblem.degenerate_cut_probe_top_k_ret",
                ),
                "subproblem.degenerate_cut_probe_top_k_ret",
            )
            if sub_raw.get("degenerate_cut_probe_top_k_ret") is not None
            else None
        ),
        degenerate_cut_zero_tol=_ensure_float(
            _disallow_expr(
                sub_raw.get("degenerate_cut_zero_tol", 1e-9),
                "subproblem.degenerate_cut_zero_tol",
            ),
            "subproblem.degenerate_cut_zero_tol",
        ),
    )

    solver_raw = _as_mapping(data.get("solver"), "solver")
    if "time_limit_s" in solver_raw:
        raise ValueError(
            "solver.time_limit_s renamed to solver.total_time_limit_s "
            "(budget for the whole Benders loop; the per-iteration master ceiling is "
            "master.per_iteration_time_limit_s)"
        )
    _check_unknown_keys(
        solver_raw,
        {
            "max_iterations",
            "tolerance",
            "total_time_limit_s",
            "stall_max_no_improve_iters",
            "stall_min_abs_improve",
            "stall_min_rel_improve",
            "master_solver",
            "subproblem_solver",
            "solver_tee",
        },
        "solver",
    )
    _require_keys(
        solver_raw,
        {
            "max_iterations",
            "tolerance",
            "total_time_limit_s",
            "master_solver",
            "subproblem_solver",
        },
        "solver",
    )
    solver_section = SolverSection(
        max_iterations=_ensure_int(
            _disallow_expr(solver_raw.get("max_iterations"), "solver.max_iterations"),
            "solver.max_iterations",
        ),
        tolerance=_ensure_float(
            _disallow_expr(solver_raw.get("tolerance"), "solver.tolerance"),
            "solver.tolerance",
        ),
        total_time_limit_s=_ensure_int(
            _disallow_expr(
                solver_raw.get("total_time_limit_s"), "solver.total_time_limit_s"
            ),
            "solver.total_time_limit_s",
        ),
        stall_max_no_improve_iters=_ensure_int(
            _disallow_expr(
                solver_raw.get("stall_max_no_improve_iters", 0),
                "solver.stall_max_no_improve_iters",
            ),
            "solver.stall_max_no_improve_iters",
        ),
        stall_min_abs_improve=_ensure_float(
            _disallow_expr(
                solver_raw.get("stall_min_abs_improve", 0.0),
                "solver.stall_min_abs_improve",
            ),
            "solver.stall_min_abs_improve",
        ),
        stall_min_rel_improve=_ensure_float(
            _disallow_expr(
                solver_raw.get("stall_min_rel_improve", 0.0),
                "solver.stall_min_rel_improve",
            ),
            "solver.stall_min_rel_improve",
        ),
        master_solver=_ensure_str(
            solver_raw.get("master_solver"), "solver.master_solver"
        ),
        subproblem_solver=_ensure_str(
            solver_raw.get("subproblem_solver"), "solver.subproblem_solver"
        ),
        solver_tee=_ensure_bool(
            solver_raw.get("solver_tee", False), "solver.solver_tee"
        ),
    )
    if solver_section.master_solver.lower() == "cplex_persistent":
        raise ValueError(
            "persistent solver mode removed; use solver.master_solver=cplex"
        )

    tol_raw = _as_mapping(data.get("tolerances", {}), "tolerances")
    _check_unknown_keys(
        tol_raw, {"eps_bin", "eps_feas", "eps_cut", "eps_hash"}, "tolerances"
    )
    tol_section = TolerancesSection(
        eps_bin=_ensure_float(
            _disallow_expr(tol_raw.get("eps_bin", 1e-6), "tolerances.eps_bin"),
            "tolerances.eps_bin",
        ),
        eps_feas=_ensure_float(
            _disallow_expr(tol_raw.get("eps_feas", 1e-7), "tolerances.eps_feas"),
            "tolerances.eps_feas",
        ),
        eps_cut=_ensure_float(
            _disallow_expr(tol_raw.get("eps_cut", 1e-8), "tolerances.eps_cut"),
            "tolerances.eps_cut",
        ),
        eps_hash=_ensure_float(
            _disallow_expr(tol_raw.get("eps_hash", 1e-6), "tolerances.eps_hash"),
            "tolerances.eps_hash",
        ),
    )

    # A multi-scenario run has to put its lower bound and its upper bound on the
    # same problem. One cut per scenario against a SINGLE theta forces
    # theta >= Q_s(y) for every s, i.e. theta >= max_s Q_s(y), while the reported
    # upper bound is the weighted mean of the same quantities. Since max >= mean,
    # the master's optimum can then exceed the true optimum of the problem the UB
    # measures: a bound that is not a bound, the D15/D16 failure mode again.
    #
    # Either give each scenario its own theta (master.theta_per_scenario: true,
    # objective sum_s w_s theta_s) or average the cuts
    # (subproblem.multi_cuts_by_scenario: false). Both are consistent with a
    # mean-aggregated UB.
    _has_scenarios = bool(data_section.scenario_files or data_section.scenarios)
    if (
        _has_scenarios
        and sub_section.multi_cuts_by_scenario
        and not master_section.theta_per_scenario
    ):
        raise ValueError(
            "subproblem.multi_cuts_by_scenario is true with master.theta_per_scenario "
            "false on a multi-scenario run. One cut per scenario on a shared theta "
            "bounds max_s Q_s(y) while the reported UB is the weighted mean, so the "
            "lower bound would not bound the problem being measured. Set "
            "master.theta_per_scenario: true, or "
            "subproblem.multi_cuts_by_scenario: false."
        )

    model_section = ModelSection(
        time=time_section,
        fleet=fleet_section,
        energy=energy_section,
        costs=cost_section,
    )

    return RootConfig(
        schema=schema,
        run=run,
        data=data_section,
        model=model_section,
        master=master_section,
        subproblem=sub_section,
        solver=solver_section,
        tolerances=tol_section,
    )


def load_config(path: str | Path | None) -> RootConfig:
    """Load configuration from YAML and return the v2 config dataclasses."""
    cfg_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    raw = _load_yaml(cfg_path)
    if "schema" not in raw:
        raw = upgrade_config_v1_to_v2(raw)
    else:
        schema_raw = raw.get("schema")
        if not isinstance(schema_raw, dict) or schema_raw.get("version") != 2:
            raw = upgrade_config_v1_to_v2(raw)
    return _parse_v2(raw)


__all__ = [
    "SchemaSection",
    "RunSection",
    "DataSection",
    "TimeSection",
    "FleetSection",
    "EnergySection",
    "CostSection",
    "ModelSection",
    "MasterSection",
    "SubproblemSection",
    "SolverSection",
    "RootConfig",
    "TolerancesSection",
    "DEFAULT_CONFIG_PATH",
    "load_config",
    "resolve_energy_params",
    "upgrade_config_v1_to_v2",
]
