from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


try:  # Python 3.11+
    import tomllib as _toml
except Exception:  # pragma: no cover - fallback if older Python
    import tomli as _toml  # type: ignore

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


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return _toml.load(f)


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
    """Load configuration from a TOML file or return defaults.

    The schema is minimal and forgiving; unknown keys are ignored.
    """
    if path is None:
        return BendersConfig()
    p = Path(path)
    if not p.exists():
        # Return defaults but allow the CLI to keep going
        return BendersConfig()

    if p.suffix.lower() in {".yaml", ".yml"}:
        raw = _load_yaml(p)
    else:
        raw = _load_toml(p)
    run = _as_dict(raw.get("run"))
    master = _as_dict(raw.get("master"))
    sub = _as_dict(raw.get("subproblem"))

    run_cfg = RunConfig(
        max_iterations=int(run.get("max_iterations", 100)),
        tolerance=float(run.get("tolerance", 1e-4)),
        time_limit_s=int(run.get("time_limit_s", 600)),
        log_level=str(run.get("log_level", "INFO")),
        seed=int(run.get("seed", 42)),
    )

    master_cfg = ComponentConfig(
        impl=str(master.get("impl", "to_fill")),
        params=_as_dict(master.get("params")),
    )
    sub_cfg = ComponentConfig(
        impl=str(sub.get("impl", "to_fill")),
        params=_as_dict(sub.get("params")),
    )
    return BendersConfig(run=run_cfg, master=master_cfg, subproblem=sub_cfg)


__all__ = ["RunConfig", "ComponentConfig", "BendersConfig", "load_config"]
