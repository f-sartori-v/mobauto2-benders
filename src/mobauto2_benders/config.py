from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import ast
import operator as _op

try:
    import yaml as _yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _yaml = None


@dataclass(slots=True)
class RunConfig:
    max_iterations: int = 100
    tolerance: float = 1e-4
    time_limit_s: int = 600
    log_level: str = "INFO"
    seed: int = 42
    # Optional stall-stopping controls
    stall_max_no_improve_iters: int = 0
    stall_min_abs_improve: float = 0.0
    stall_min_rel_improve: float = 0.0
    # Print iteration summary every N iters (1 = every iter)
    print_every: int = 10


@dataclass(slots=True)
class ComponentConfig:
    impl: str = "to_fill"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BendersConfig:
    run: RunConfig = field(default_factory=RunConfig)
    master: ComponentConfig = field(default_factory=ComponentConfig)
    subproblem: ComponentConfig = field(default_factory=ComponentConfig)


def _as_dict(m: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(m) if m else {}


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _eval_expr(expr: str, names: Mapping[str, Any]) -> float | int:
    """Safely evaluate a simple arithmetic expression with provided names.

    Allowed:
      - literals: ints and floats
      - names: variables present in 'names' (must be numeric)
      - operators: +, -, *, /, //, %, **
      - parentheses and unary +/-

    Disallowed: function calls, attribute access, subscripting, comprehensions, etc.
    """

    # Parse expression into AST
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
        if isinstance(n, ast.Num):  # pragma: no cover - for older Python ASTs
            return n.n  # type: ignore[attr-defined]
        if isinstance(n, ast.Name):
            if n.id not in names:
                raise NameError(f"unknown name '{n.id}' in expression")
            v = names[n.id]
            if _is_number(v):
                return v  # type: ignore[return-value]
            # attempt to coerce numeric-looking strings
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
            # Support one-element tuple/list wrapping for safety (rare)
            return _eval(n.elts[0])  # type: ignore[index]
        # Everything else is unsafe/not supported
        raise ValueError("unsupported syntax in expression")

    return _eval(node)


def _resolve_param_expressions(params: dict[str, Any]) -> dict[str, Any]:
    """Resolve arithmetic string expressions within a parameter dict.

    Only attempts evaluation for string values that look like math expressions
    (contain one of '+-*/()'). Leaves other values untouched. Variables can
    reference other keys in the same dict.
    """
    if not params:
        return params
    names = dict(params)
    out: dict[str, Any] = dict(params)
    for k, v in params.items():
        if isinstance(v, str):
            s = v.strip()
            if any(ch in s for ch in "+-*/()"):
                try:
                    out[k] = _eval_expr(s, names)
                except Exception:
                    # Leave as-is; downstream code may surface a clearer error
                    out[k] = v
    return out


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


def load_config(path: str | Path | None) -> BendersConfig:
    """Load configuration from a YAML file or return defaults.

    The schema is minimal and forgiving; unknown keys are ignored. Only YAML is supported.
    """
    if path is None:
        return BendersConfig()
    p = Path(path)
    if not p.exists():
        # Return defaults but allow the CLI to keep going
        return BendersConfig()

    if p.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Unsupported config format '{p.suffix}'. Please provide a YAML file.")
    raw = _load_yaml(p)
    run = _as_dict(raw.get("run"))
    master = _as_dict(raw.get("master"))
    sub = _as_dict(raw.get("subproblem"))

    run_cfg = RunConfig(
        max_iterations=int(run.get("max_iterations", 100)),
        tolerance=float(run.get("tolerance", 1e-4)),
        time_limit_s=int(run.get("time_limit_s", 600)),
        log_level=str(run.get("log_level", "INFO")),
        seed=int(run.get("seed", 42)),
        stall_max_no_improve_iters=int(run.get("stall_max_no_improve_iters", 0) or 0),
        stall_min_abs_improve=float(run.get("stall_min_abs_improve", 0.0) or 0.0),
        stall_min_rel_improve=float(run.get("stall_min_rel_improve", 0.0) or 0.0),
        print_every=int(run.get("print_every", 10) or 10),
    )

    master_params = _as_dict(master.get("params"))
    # Evaluate arithmetic expressions within master params (e.g., "30 * (30 / slot_resolution)")
    master_params = _resolve_param_expressions(master_params)
    master_cfg = ComponentConfig(
        impl=str(master.get("impl", "to_fill")),
        params=master_params,
    )
    sub_params = _as_dict(sub.get("params"))
    # Propagate time discretization keys from master to subproblem if missing
    for key in ("slot_resolution", "resolution", "T_minutes", "trip_duration", "trip_duration_minutes"):
        if key in master_cfg.params and key not in sub_params:
            sub_params[key] = master_cfg.params[key]
    # Evaluate arithmetic expressions within subproblem params as well (after propagation)
    sub_params = _resolve_param_expressions(sub_params)
    sub_cfg = ComponentConfig(
        impl=str(sub.get("impl", "to_fill")),
        params=sub_params,
    )
    return BendersConfig(run=run_cfg, master=master_cfg, subproblem=sub_cfg)


__all__ = ["RunConfig", "ComponentConfig", "BendersConfig", "load_config"]
