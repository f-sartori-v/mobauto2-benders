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
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        out["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
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


def _rho_per_hour(cfg) -> float | None:
    """Charging rate in energy per HOUR, which is the unit the contract states.

    `delta_chg` is per SLOT and therefore moves with `slot_resolution`; two runs at
    different resolutions with the same physical charger would carry different
    `delta_chg` and, if that were hashed, different manifest ids for no physical
    reason. Converting once here keeps the id about the site rather than the grid.
    """
    try:
        from .config import resolve_energy_params

        energy = resolve_energy_params(
            cfg.model.energy,
            {"slot_resolution": cfg.model.time.slot_resolution},
        )
        delta_chg = energy.get("delta_chg")
        if delta_chg is None:
            return None
        return float(delta_chg) * (60.0 / float(cfg.model.time.slot_resolution))
    except Exception:
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

    # The short id the shared contract with Agent CP-LBBD is written in. Computed from
    # the contract's own field list, in the contract's own order, by one function both
    # agents call -- an id computed twice from two orderings is two ids.
    from .results_emitter import (
        demand_checksum,
        manifest_fields_from_config,
        manifest_id,
    )

    _sources = list(cfg.data.scenario_files or [])
    if not _sources and cfg.data.demand_file:
        _sources = [cfg.data.demand_file]
    _mf = manifest_fields_from_config(
        cfg,
        {
            "demand_checksum": demand_checksum(_sources) if _sources else None,
            "solver_version": _solver_version(cfg.solver.master_solver),
            "git_revision": _git_commit(repo_root).get("commit"),
            "rho": _rho_per_hour(cfg),
        },
    )
    _id = manifest_id(_mf)

    return {
        # B9/contract. The one field a reported table must carry. Two rows sharing it
        # were produced under the same H, delta, Q, S, Emax, b0, c_trip, rho,
        # tau_trip, Wmax, p_min, epsilon, kappa, K_chg, o, same_slot_eligibility,
        # demand, scenario set and weights, objective mode, solver, threads, seed,
        # budgets and revision. Two rows that do not share it are not comparable, and
        # results_emitter.Table refuses to render a table that mixes them.
        "manifest_id": _id,
        "manifest_fields": _mf,
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
            # B10 (audit item 2.3). CENSORED, NOT DISCARDED. The previous protocol
            # rejected any run containing a clock-truncated master solve, which threw
            # away a 228-iteration run that had reached an 11.8% interval -- the
            # observation was destroyed to protect a reproducibility claim that could
            # simply have been labelled. `censored` is that label. A censored run is
            # kept, reported with its truncation count, and summarised across
            # repetitions by median trajectory and performance profile rather than by
            # a single wall time (results_emitter.median_trajectory /
            # performance_profile).
            "censored": (None if _truncated is None else bool(_truncated)),
            "cplex_options": dict(cfg.master.cplex_options or {}),
        },
        # Swept per D2/D3: every table must state the pair it was produced with.
        "swept_parameters": {
            "Wmax_minutes": cfg.subproblem.Wmax_minutes,
            "Wmax_slots": cfg.subproblem.Wmax_slots,
            "p": cfg.subproblem.p,
            # Slot units. p_minutes is the resolution-independent form it came
            # from, or null when p was stated directly (D50).
            "p_minutes": cfg.subproblem.p_minutes,
            "recourse_resolution": cfg.subproblem.recourse_resolution,
            "departure_policy": cfg.subproblem.departure_policy,
            # B6. Which eligibility convention priced this run. Two runs that differ
            # only here are two different operational claims, not two estimates of one
            # number, and the delta=1 comparison was reported without it.
            "same_slot_eligibility": cfg.subproblem.same_slot_eligibility,
            # F2: the departure-offset grid o. Named in the shared manifest contract.
            "placement_offsets": (
                list(cfg.subproblem.placement_offsets)
                if cfg.subproblem.placement_offsets
                else None
            ),
        },
        "objective_terms": {
            # B7.3. "weighted_sum" or "lexicographic". Named in the shared manifest
            # contract because the two are not two solvers of one problem -- they are
            # two problems, and a table may not mix them.
            "objective_mode": cfg.model.costs.objective_mode,
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
            # B2. None means "as many chargers as vehicles" -- resolved to Q inside
            # the master, and recorded here as given so a table cannot silently mix a
            # constrained run with an unconstrained one.
            "K_chg": cfg.model.energy.K_chg,
            "charger_occupancy_binary": bool(
                cfg.model.energy.charger_occupancy_binary
            ),
            "symmetry_breaking": bool(
                cfg.master.use_fifo_symmetry or cfg.master.symmetry_breaking
            ),
            "charge_before_idle": cfg.master.charge_before_idle,
            "aggregate_cuts_by_tau": cfg.master.aggregate_cuts_by_tau,
            "theta_per_scenario": cfg.master.theta_per_scenario,
            # B1. "aggregated" or "disaggregated" -- resolved by config.py, never
            # inferred here, so the manifest and the engines cannot disagree about
            # which architecture ran.
            "cut_architecture": cfg.subproblem.cut_architecture,
        },
        "cut_generation": {
            "use_magnanti_wong": cfg.subproblem.use_magnanti_wong,
            # B14. Whether core-point certification was attempted, and under which
            # method. Relative interiority is never established; see
            # subproblem_impl.CUT_MODE_DISPLAY_NAME for what may be claimed.
            "mw_core_point_certification": (
                cfg.subproblem.mw_core_point_certification
            ),
            "mw_core_point_certification_result": diag.get(
                "mw_core_point_certification"
            ),
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
        # Comparison A (forward-plan A4a, report meth-protocol-engines). Previously
        # this existed only as text printed once at the end of a run, from four
        # independently-maintained print blocks -- exactly the kind of number a
        # manifest exists to make mechanical instead of eyeballed (README section
        # 6.5). total_master_time_s + total_sp_solve_time_s + total_cutgen_time_s +
        # model_management_overhead_s sums to total_wall_time_s by construction.
        "runtime": {
            "time_to_first_feasible_s": getattr(
                result, "time_to_first_feasible_s", None
            ),
            "total_wall_time_s": getattr(result, "total_wall_time_s", None),
            "total_master_time_s": getattr(result, "total_master_time_s", None),
            "total_sp_solve_time_s": getattr(result, "total_sp_solve_time_s", None),
            "total_cutgen_time_s": getattr(result, "total_cutgen_time_s", None),
            "total_cutadd_time_s": getattr(result, "total_cutadd_time_s", None),
            "model_management_overhead_s": getattr(
                result, "model_management_overhead_s", None
            ),
        },
    }


def write_manifest(
    manifest: dict[str, Any], out_dir: Path, run_name: str | None
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"manifest_{run_name or 'run'}_{stamp}.json"
    path = out_dir / name
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


__all__ = ["build_manifest", "write_manifest"]
