from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .config import load_config, _resolve_param_expressions as _eval_param_exprs
from .logging_config import setup_logging
from .benders.solver import BendersSolver, BendersRunResult


def _import_problem_impl():
    """Import default problem-specific implementations.

    Expects classes `ProblemMaster` and `ProblemSubproblem` in
    `mobauto2_benders.problem.master_impl` and `.subproblem_impl`.
    """
    from .problem.master_impl import ProblemMaster  # type: ignore
    from .problem.subproblem_impl import ProblemSubproblem  # type: ignore
    return ProblemMaster, ProblemSubproblem


def _default_config_path() -> Path:
    """Best-effort discovery of the default YAML config.

    Tries these, in order:
    1) CWD `configs/default.yaml`
    2) Repo root relative to this file
    Falls back to `configs/default.yaml` in CWD regardless.
    """
    # 1) CWD
    cwd_path = Path("configs/default.yaml")
    if cwd_path.exists():
        return cwd_path
    # 2) Repo root relative to this file (src/ -> repo root)
    here = Path(__file__).resolve()
    repo_path = here.parents[2] / "configs" / "default.yaml"
    if repo_path.exists():
        return repo_path
    return cwd_path


def run(config_path: str | Path | None = None) -> BendersRunResult:
    """Run the Benders solver reading all options from YAML.

    Parameters are taken from `configs/default.yaml` by default.
    No CLI overrides are applied here; YAML controls features such as
    Magnanti–Wong, multi-cuts, solver names, time discretization, etc.
    """
    cfg_path = Path(config_path) if config_path is not None else _default_config_path()
    cfg = load_config(cfg_path)
    setup_logging(cfg.run.log_level)

    ProblemMaster, ProblemSubproblem = _import_problem_impl()

    # Start from YAML params; allow expression-evaluated values
    mp = _eval_param_exprs(dict(cfg.master.params or {}))
    sp = _eval_param_exprs(dict(cfg.subproblem.params or {}))

    # Propagate slot_resolution from master to subproblem if not explicitly set
    if "slot_resolution" not in sp and "slot_resolution" in mp:
        sp["slot_resolution"] = mp["slot_resolution"]
        sp = _eval_param_exprs(dict(sp))  # re-evaluate any dependent expressions

    # Single-resolution run driven entirely by YAML
    master = ProblemMaster(mp)
    sub = ProblemSubproblem(sp)
    solver = BendersSolver(master, sub, cfg)
    return solver.run()


__all__ = ["run"]

