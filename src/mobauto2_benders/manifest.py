"""Run manifest: tie every reported number back to what produced it.

BENDERS_SPEC_v3 §0 non-negotiables 4 and 7 require that every run be
reproducible -- git commit, config hash, seed, solver version, and the swept
parameters (W_max, p) -- and that every reported number state which subproblem
mode produced it.

This is not bookkeeping for its own sake. Two defects this codebase carried for
an unknown length of time were invisible precisely because nothing recorded
provenance: Magnanti-Wong never ran, so every cut was a finite difference and no
lower bound was valid; and prefix-ordering symmetry breaking meant the master
was not a relaxation, so its bound could exceed the true optimum. Neither left a
trace in any output. Deciding retrospectively which results were affected was
guesswork.

The manifest therefore records, alongside the usual provenance:
  - cut_generation_mode: which generator actually produced the cuts
  - cut_valid_lower_bound: whether those cuts support a lower bound at all
  - concurrency_penalty: active in the objective, absent from the published
    formulation, so any table quoting a result must state it
  - clock_truncated_master_solves / bit_reproducible: whether the run's numbers
    can be reproduced at all, or are one draw from a machine-load dependent
    distribution. Same failure shape as the two defects above -- a run that
    stopped on the clock looks exactly like one that converged.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit(repo_root: Path) -> dict[str, Any]:
    """Current commit and whether the tree was dirty when the run started."""
    out: dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    try:
        out["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        out["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, text=True,
            stderr=subprocess.DEVNULL,
        )
        out["dirty"] = bool(status.strip())
    except (subprocess.CalledProcessError, OSError, FileNotFoundError):
        # Not a git checkout, or git unavailable. Leave the fields None rather
        # than inventing a value -- an absent provenance must be visible.
        pass
    return out


def _config_hash(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {"path": None, "sha256": None}
    try:
        raw = Path(config_path).read_bytes()
        return {
            "path": str(config_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    except OSError:
        return {"path": str(config_path), "sha256": None}


def _solver_version(solver_name: str) -> str | None:
    try:
        import pyomo.environ as pyo
        s = pyo.SolverFactory(solver_name)
        v = getattr(s, "version", None)
        if callable(v):
            ver = v()
            if ver:
                return ".".join(str(x) for x in ver)
    except Exception:
        pass
    return None


def build_manifest(
    cfg: Any,
    config_path: Path | None,
    result: Any,
    repo_root: Path,
    diagnostics: dict | None = None,
) -> dict[str, Any]:
    diag = diagnostics or {}
    lb = getattr(result, "best_lower_bound", None)
    ub = getattr(result, "best_upper_bound", None)
    gap = None
    if lb is not None and ub is not None:
        gap = abs(ub - lb) / max(1.0, abs(ub))
    _truncated = getattr(result, "clock_truncated_master_solves", None)

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": {
            **_git_commit(repo_root),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "config": {
            **_config_hash(config_path),
            "run_name": getattr(cfg.run, "name", None),
            "seed": getattr(cfg.run, "seed", None),
        },
        "solver": {
            "master": cfg.solver.master_solver,
            "subproblem": cfg.solver.subproblem_solver,
            "master_version": _solver_version(cfg.solver.master_solver),
            "max_iterations": cfg.solver.max_iterations,
            "tolerance": cfg.solver.tolerance,
            "total_time_limit_s": cfg.solver.total_time_limit_s,
            "per_iteration_time_limit_s": cfg.master.per_iteration_time_limit_s,
            "per_iteration_mipgap": cfg.master.per_iteration_mipgap,
        },
        # Whether this run's numbers can be reproduced at all. A master solve that
        # stopped on the clock rather than on the gap explored a machine-load
        # dependent number of nodes, and every later iteration inherits the
        # difference: measured, a binding limit moved the LB 8% between two runs of
        # one config, while a non-binding one reproduced to the last digit.
        # Without this field the manifest cannot say whether an archived result was
        # a measurement or one draw from a distribution.
        "reproducibility": {
            "clock_truncated_master_solves": _truncated,
            "bit_reproducible": (None if _truncated is None else _truncated == 0),
            "cplex_options": dict(cfg.master.cplex_options or {}),
        },
        # Swept per D2/D3: every table must state the pair it was produced with.
        "swept_parameters": {
            "Wmax_minutes": cfg.subproblem.Wmax_minutes,
            "Wmax_slots": cfg.subproblem.Wmax_slots,
            "p": cfg.subproblem.p,
        },
        "objective_terms": {
            "start_cost_epsilon": cfg.model.costs.start_cost_epsilon,
            # Active in the master objective and NOT in the published
            # formulation, so it must appear on any table quoting this run.
            "concurrency_penalty": cfg.model.costs.concurrency_penalty,
        },
        "model": {
            "Q": cfg.model.fleet.Q,
            "T_minutes": cfg.model.time.T_minutes,
            "slot_resolution": cfg.model.time.slot_resolution,
            "trip_duration_minutes": cfg.model.time.trip_duration_minutes,
            "S": cfg.subproblem.S,
            "Emax": cfg.model.energy.Emax,
            "L": cfg.model.energy.L,
            "symmetry_breaking": bool(cfg.master.use_fifo_symmetry or cfg.master.symmetry_breaking),
            "charge_before_idle": cfg.master.charge_before_idle,
            "aggregate_cuts_by_tau": cfg.master.aggregate_cuts_by_tau,
            "theta_per_scenario": cfg.master.theta_per_scenario,
        },
        "cut_generation": {
            "use_magnanti_wong": cfg.subproblem.use_magnanti_wong,
            "use_dual_slopes": cfg.subproblem.use_dual_slopes,
            # What actually produced the cuts, and whether they support a bound.
            "mode_used": diag.get("cut_generation_mode"),
            "valid_lower_bound": diag.get("cut_valid_lower_bound"),
        },
        "data": {
            "demand_file": cfg.data.demand_file,
            "scenario_files": list(cfg.data.scenario_files or []),
            "scenario_weights": cfg.data.scenario_weights,
        },
        "result": {
            "status": str(getattr(result, "status", None)),
            "iterations": getattr(result, "iterations", None),
            "best_lower_bound": lb,
            "best_upper_bound": ub,
            "relative_gap": gap,
            "pax_served": getattr(result, "pax_served", None),
            "pax_total": getattr(result, "pax_total", None),
        },
    }


def write_manifest(manifest: dict[str, Any], out_dir: Path, run_name: str | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"manifest_{run_name or 'run'}_{stamp}.json"
    path = out_dir / name
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


__all__ = ["build_manifest", "write_manifest"]
