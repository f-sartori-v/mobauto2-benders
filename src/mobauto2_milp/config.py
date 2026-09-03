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


@dataclass(slots=True)
class DataSection:
    demand_file: str | None = None
    demand_files: list[str] = field(default_factory=list)
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
    # B2 (audit 1.8). How many vehicles may draw from the depot's chargers in one
    # slot. `None` means Q -- every vehicle can charge at once, which is the
    # assumption every archived result was produced under, and at which the row is
    # implied by c in [0,1] so those results reproduce exactly. State a number to
    # make the site's real charger count part of the instance rather than an
    # accident of the formulation.
    K_chg: int | None = None
    # Whether a charger is held for a whole slot (True) or is preemptible within it
    # (False, the divisible default). These are different physical claims and give
    # different schedules; the manifest records which one a run used.
    charger_occupancy_binary: bool = False


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
class MilpSection:
    use_fifo_symmetry: bool = False
    symmetry_breaking: bool = False
    use_mip_start: bool = False
    solve_time_limit_s: int | None = None
    mipgap: float | None = None
    cplex_options: dict[str, Any] = field(default_factory=dict)
    solver_backend: str = "cplex_direct"


@dataclass(slots=True)
class ServiceSection:
    S: float = 0.0
    Wmax_minutes: int | None = None
    Wmax_slots: int | None = None
    # Always slot units here; p_minutes is the resolution-independent input form and
    # is converted once at load. Mirrors mobauto2_benders.config.SubproblemSection so
    # the benchmark and the thing it benchmarks cannot express different policies
    # (D50).
    p: float = 0.0
    p_minutes: float | None = None
    fill_first_epsilon: float = 0.0


@dataclass(slots=True)
class SolverSection:
    solver_tee: bool = False


@dataclass(slots=True)
class RootConfig:
    schema: SchemaSection
    run: RunSection
    data: DataSection
    model: ModelSection
    milp: MilpSection
    service: ServiceSection
    solver: SolverSection


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
        if num_node is not None and isinstance(n, num_node):  # pragma: no cover - old ASTs
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


def resolve_energy_params(energy: EnergySection, names: Mapping[str, Any]) -> dict[str, Any]:
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


def _ensure_str_or_str_list(value: Any, where: str) -> str | list[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ValueError(f"{where} must be a string or a list of strings")


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
            raise ValueError(f"{where} must be numeric or an arithmetic expression") from exc
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
    """Upgrade a v1 config dict to the current schema.

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
        "schema": {"name": "mobauto2_milp_config", "version": 3},
        "run": {
            "name": None,
            "log_level": run.get("log_level", "INFO"),
            "log_file": None,
            "report_dir": None,
            "seed": run.get("seed"),
        },
        "data": {
            "demand_file": sub_params.get("demand_file"),
            "demand_files": sub_params.get("demand_files") or [],
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
        "milp": {
            "use_fifo_symmetry": master_params.get("use_fifo_symmetry", False),
            "symmetry_breaking": master_params.get("symmetry_breaking", False),
            "use_mip_start": master_params.get("use_mip_start", False),
            "solve_time_limit_s": master_params.get("solve_time_limit_s"),
            "mipgap": master_params.get("mipgap"),
            "cplex_options": master_params.get("cplex_options", {}),
            "solver_backend": master_params.get("solver_backend", "cplex_direct"),
        },
        "service": {
            "S": sub_params.get("S"),
            "Wmax_minutes": sub_params.get("Wmax_minutes"),
            "Wmax_slots": sub_params.get("Wmax_slots"),
            "p": sub_params.get("p"),
            "fill_first_epsilon": sub_params.get("fill_first_epsilon", 0.0),
        },
        "solver": {"solver_tee": master_params.get("solver_tee", False)},
    }

    _note("Deprecated v1 key 'run.*' mapped to 'run.*' (logging only).")
    _note(f"v1 run keys: {sorted(run.keys())}")
    _note("Deprecated v1 key 'master.params.*' mapped into 'model.*', 'milp.*', and 'solver.*'.")
    _note(f"v1 master.params keys: {sorted(master_params.keys())}")
    _note("Deprecated v1 key 'subproblem.params.*' mapped into 'data.*', 'service.*', and 'solver.*'.")
    _note(f"v1 subproblem.params keys: {sorted(sub_params.keys())}")

    if warnings_list:
        warnings.warn(
            "Loaded v1 config; please upgrade to schema version 2.\n" + "\n".join(warnings_list),
            stacklevel=2,
        )

    return new


def _parse_v3(raw: Mapping[str, Any]) -> RootConfig:
    data = _as_mapping(raw, "config")
    _check_unknown_keys(
        data,
        {"schema", "run", "data", "model", "milp", "service", "master", "subproblem", "solver", "tolerances"},
        "config",
    )
    _require_keys(data, {"schema", "run", "data", "model", "solver"}, "config")

    schema_raw = _as_mapping(data.get("schema"), "schema")
    _check_unknown_keys(schema_raw, {"name", "version"}, "schema")
    _require_keys(schema_raw, {"name", "version"}, "schema")
    schema = SchemaSection(
        name=_ensure_str(schema_raw.get("name"), "schema.name"),
        version=_ensure_int(schema_raw.get("version"), "schema.version"),
    )
    if schema.name != "mobauto2_milp_config" or schema.version not in (2, 3):
        raise ValueError("Unsupported schema version; expected mobauto2_milp_config v2 or v3")

    run_raw = _as_mapping(data.get("run"), "run")
    _check_unknown_keys(run_raw, {"name", "log_level", "log_file", "report_dir", "seed"}, "run")
    run_name = run_raw.get("name")
    run = RunSection(
        name=(_ensure_str(run_name, "run.name") if run_name is not None else None),
        log_level=str(run_raw.get("log_level", "INFO")),
        log_file=run_raw.get("log_file"),
        report_dir=run_raw.get("report_dir"),
        seed=(_ensure_int(run_raw.get("seed"), "run.seed") if "seed" in run_raw and run_raw.get("seed") is not None else None),
    )

    data_raw = _as_mapping(data.get("data"), "data")
    _check_unknown_keys(
        data_raw,
        {"demand_file", "demand_files", "scenario_files", "scenario_weights", "R_out", "R_ret", "scenarios"},
        "data",
    )
    demand_file_val = data_raw.get("demand_file")
    demand_file_parsed: str | list[str] | None = None
    if demand_file_val is not None:
        demand_file_parsed = _ensure_str_or_str_list(demand_file_val, "data.demand_file")
    scenario_files_val = _ensure_str_list(data_raw.get("scenario_files"), "data.scenario_files")
    if isinstance(demand_file_parsed, list):
        if scenario_files_val:
            raise ValueError(
                "data.demand_file cannot be a list when data.scenario_files is also provided; "
                "use only data.scenario_files for multiple scenarios"
            )
        scenario_files_val = list(demand_file_parsed)
        demand_file_parsed = None
    data_section = DataSection(
        demand_file=(str(demand_file_parsed) if isinstance(demand_file_parsed, str) else None),
        demand_files=_ensure_str_list(data_raw.get("demand_files"), "data.demand_files"),
        scenario_files=scenario_files_val,
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
        scenarios=(data_raw.get("scenarios") if isinstance(data_raw.get("scenarios"), list) else None),
    )

    model_raw = _as_mapping(data.get("model"), "model")
    _check_unknown_keys(model_raw, {"time", "fleet", "energy", "costs"}, "model")
    _require_keys(model_raw, {"time", "fleet", "energy", "costs"}, "model")

    time_raw = _as_mapping(model_raw.get("time"), "model.time")
    _check_unknown_keys(
        time_raw,
        {"T_minutes", "T", "slot_resolution", "trip_duration_minutes", "trip_duration", "trip_slots"},
        "model.time",
    )
    _require_keys(time_raw, {"slot_resolution"}, "model.time")
    if "T_minutes" not in time_raw and "T" not in time_raw:
        raise ValueError("model.time must include T_minutes or T")
    time_section = TimeSection(
        T_minutes=(
            _ensure_int(_disallow_expr(time_raw.get("T_minutes"), "model.time.T_minutes"), "model.time.T_minutes")
            if time_raw.get("T_minutes") is not None
            else None
        ),
        T=(
            _ensure_int(_disallow_expr(time_raw.get("T"), "model.time.T"), "model.time.T")
            if time_raw.get("T") is not None
            else None
        ),
        slot_resolution=_ensure_int(
            _disallow_expr(time_raw.get("slot_resolution"), "model.time.slot_resolution"),
            "model.time.slot_resolution",
        ),
        trip_duration_minutes=(
            _ensure_int(
                _disallow_expr(time_raw.get("trip_duration_minutes"), "model.time.trip_duration_minutes"),
                "model.time.trip_duration_minutes",
            )
            if time_raw.get("trip_duration_minutes") is not None
            else None
        ),
        trip_duration=(
            _ensure_int(
                _disallow_expr(time_raw.get("trip_duration"), "model.time.trip_duration"),
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
    _check_unknown_keys(fleet_raw, {"Q", "binit", "initial_battery", "initial_actions"}, "model.fleet")
    _require_keys(fleet_raw, {"Q"}, "model.fleet")
    binit_raw = fleet_raw.get("initial_battery", fleet_raw.get("binit"))
    fleet_section = FleetSection(
        Q=_ensure_int(_disallow_expr(fleet_raw.get("Q"), "model.fleet.Q"), "model.fleet.Q"),
        binit=(
            _ensure_num_list(binit_raw, "model.fleet.initial_battery")
            if binit_raw is not None
            else None
        ),
        initial_actions=(
            _ensure_str_list(fleet_raw.get("initial_actions"), "model.fleet.initial_actions")
            if fleet_raw.get("initial_actions") is not None
            else None
        ),
    )

    energy_raw = _as_mapping(model_raw.get("energy"), "model.energy")
    _check_unknown_keys(
        energy_raw,
        {"Emax", "L", "delta_chg", "K_chg", "charger_occupancy_binary"},
        "model.energy",
    )
    _require_keys(energy_raw, {"Emax", "L"}, "model.energy")
    energy_section = EnergySection(
        Emax=_ensure_float(_disallow_expr(energy_raw.get("Emax"), "model.energy.Emax"), "model.energy.Emax"),
        L=_ensure_float(_disallow_expr(energy_raw.get("L"), "model.energy.L"), "model.energy.L"),
        delta_chg=(
            _ensure_num_or_expr(energy_raw.get("delta_chg"), "model.energy.delta_chg")
            if energy_raw.get("delta_chg") is not None
            else None
        ),
        K_chg=(
            int(
                _ensure_float(
                    _disallow_expr(energy_raw.get("K_chg"), "model.energy.K_chg"),
                    "model.energy.K_chg",
                )
            )
            if energy_raw.get("K_chg") is not None
            else None
        ),
        charger_occupancy_binary=bool(
            energy_raw.get("charger_occupancy_binary", False)
        ),
    )

    costs_raw = _as_mapping(model_raw.get("costs"), "model.costs")
    _check_unknown_keys(costs_raw, {"start_cost_epsilon", "concurrency_penalty"}, "model.costs")
    cost_section = CostSection(
        start_cost_epsilon=(
            _ensure_float(
                _disallow_expr(costs_raw.get("start_cost_epsilon"), "model.costs.start_cost_epsilon"),
                "model.costs.start_cost_epsilon",
            )
            if costs_raw.get("start_cost_epsilon") is not None
            else 0.0
        ),
        concurrency_penalty=(
            _ensure_float(
                _disallow_expr(costs_raw.get("concurrency_penalty"), "model.costs.concurrency_penalty"),
                "model.costs.concurrency_penalty",
            )
            if costs_raw.get("concurrency_penalty") is not None
            else 0.0
        ),
    )

    milp_key = "milp" if "milp" in data else "master"
    service_key = "service" if "service" in data else "subproblem"
    if milp_key not in data:
        raise ValueError("config must include milp")
    if service_key not in data:
        raise ValueError("config must include service")

    milp_raw = _as_mapping(data.get(milp_key), milp_key)
    if ("use_" + "lazy" + "_cuts") in milp_raw:
        raise ValueError("lazy " + "cuts removed; delete key use_" + "lazy" + "_cuts")
    _check_unknown_keys(
        milp_raw,
        {"use_fifo_symmetry", "symmetry_breaking", "use_mip_start", "solve_time_limit_s", "mipgap", "cplex_options", "solver_backend"},
        milp_key,
    )
    milp_section = MilpSection(
        use_fifo_symmetry=_ensure_bool(milp_raw.get("use_fifo_symmetry", False), f"{milp_key}.use_fifo_symmetry"),
        symmetry_breaking=_ensure_bool(milp_raw.get("symmetry_breaking", False), f"{milp_key}.symmetry_breaking"),
        use_mip_start=_ensure_bool(milp_raw.get("use_mip_start", False), f"{milp_key}.use_mip_start"),
        solve_time_limit_s=(
            _ensure_int(
                _disallow_expr(milp_raw.get("solve_time_limit_s"), f"{milp_key}.solve_time_limit_s"),
                f"{milp_key}.solve_time_limit_s",
            )
            if milp_raw.get("solve_time_limit_s") is not None
            else None
        ),
        mipgap=(
            _ensure_float(
                _disallow_expr(milp_raw.get("mipgap"), f"{milp_key}.mipgap"),
                f"{milp_key}.mipgap",
            )
            if milp_raw.get("mipgap") is not None
            else None
        ),
        cplex_options=_ensure_mapping(milp_raw.get("cplex_options"), f"{milp_key}.cplex_options"),
        solver_backend=_ensure_str(milp_raw.get("solver_backend", "cplex_direct"), f"{milp_key}.solver_backend"),
    )

    service_raw = _as_mapping(data.get(service_key), service_key)
    _check_unknown_keys(
        service_raw,
        {"S", "Wmax_minutes", "Wmax_slots", "p", "p_minutes", "fill_first_epsilon"},
        service_key,
    )
    _require_keys(service_raw, {"S"}, service_key)
    if "Wmax_minutes" not in service_raw and "Wmax_slots" not in service_raw:
        raise ValueError(f"{service_key} must include Wmax_minutes or Wmax_slots")

    # Penalty units (D50), same rule as the Benders side: exactly one form, and both
    # present is an error rather than a precedence order.
    _has_p = "p" in service_raw
    _has_p_minutes = "p_minutes" in service_raw
    if _has_p and _has_p_minutes:
        raise ValueError(
            f"{service_key} sets both p and p_minutes; they state the same policy in "
            "different units and only one may be given. p is in slot units and moves "
            "with model.time.slot_resolution; p_minutes does not. Prefer p_minutes."
        )
    if not _has_p and not _has_p_minutes:
        raise ValueError(f"{service_key} must include p or p_minutes")
    if _has_p_minutes:
        _p_minutes_val = _ensure_float(
            _disallow_expr(service_raw.get("p_minutes"), f"{service_key}.p_minutes"),
            f"{service_key}.p_minutes",
        )
        _slot_res = int(time_section.slot_resolution)
        if _slot_res <= 0:
            raise ValueError(
                f"model.time.slot_resolution must be positive to convert p_minutes, "
                f"got {_slot_res!r}"
            )
        # No rounding: p is a cost coefficient, not an index bound like Wmax.
        _p_slots_val = float(_p_minutes_val) / float(_slot_res)
    else:
        _p_minutes_val = None
        _p_slots_val = _ensure_float(
            _disallow_expr(service_raw.get("p"), f"{service_key}.p"), f"{service_key}.p"
        )
    service_section = ServiceSection(
        S=_ensure_float(_disallow_expr(service_raw.get("S"), f"{service_key}.S"), f"{service_key}.S"),
        Wmax_minutes=(
            _ensure_int(
                _disallow_expr(service_raw.get("Wmax_minutes"), f"{service_key}.Wmax_minutes"),
                f"{service_key}.Wmax_minutes",
            )
            if service_raw.get("Wmax_minutes") is not None
            else None
        ),
        Wmax_slots=(
            _ensure_int(
                _disallow_expr(service_raw.get("Wmax_slots"), f"{service_key}.Wmax_slots"),
                f"{service_key}.Wmax_slots",
            )
            if service_raw.get("Wmax_slots") is not None
            else None
        ),
        p=_p_slots_val,
        p_minutes=_p_minutes_val,
        fill_first_epsilon=_ensure_float(
            _disallow_expr(service_raw.get("fill_first_epsilon", 0.0), f"{service_key}.fill_first_epsilon"),
            f"{service_key}.fill_first_epsilon",
        ),
    )

    solver_raw = _as_mapping(data.get("solver"), "solver")
    _check_unknown_keys(solver_raw, {"solver_tee"}, "solver")
    solver_section = SolverSection(
        solver_tee=_ensure_bool(solver_raw.get("solver_tee", False), "solver.solver_tee"),
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
        milp=milp_section,
        service=service_section,
        solver=solver_section,
    )


def load_config(path: str | Path | None) -> RootConfig:
    """Load configuration from YAML and return the current config dataclasses."""
    cfg_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    raw = _load_yaml(cfg_path)
    if "schema" not in raw:
        raw = upgrade_config_v1_to_v2(raw)
    return _parse_v3(raw)


__all__ = [
    "SchemaSection",
    "RunSection",
    "DataSection",
    "TimeSection",
    "FleetSection",
    "EnergySection",
    "CostSection",
    "ModelSection",
    "MilpSection",
    "ServiceSection",
    "SolverSection",
    "RootConfig",
    "DEFAULT_CONFIG_PATH",
    "load_config",
    "resolve_energy_params",
    "upgrade_config_v1_to_v2",
]
