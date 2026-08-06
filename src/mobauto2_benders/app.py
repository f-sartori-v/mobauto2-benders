from __future__ import annotations

from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, load_config, resolve_energy_params
from .logging_config import setup_logging
from .benders.solver import BendersSolver, BendersRunResult


def import_problem_impl():
    """Import default problem-specific implementations.

    Expects classes `ProblemMaster` and `ProblemSubproblem` in
    `mobauto2_benders.problem.master_impl` and `.subproblem_impl`.
    """
    try:
        from .problem.master_impl import ProblemMaster  # type: ignore
        from .problem.subproblem_impl import ProblemSubproblem  # type: ignore
        return ProblemMaster, ProblemSubproblem
    except Exception as exc:  # noqa: BLE001 - provide friendly message
        raise SystemExit(
            "Problem-specific implementations not found.\n"
            "Create classes `ProblemMaster` and `ProblemSubproblem` under:\n"
            "  src/mobauto2_benders/problem/master_impl.py\n"
            "  src/mobauto2_benders/problem/subproblem_impl.py\n"
            "Each should extend the abstract base classes in\n"
            "  src/mobauto2_benders/benders/master.py and subproblem.py\n"
            f"\nOriginal import error: {exc}"
        )


def _apply_run_overrides(cfg, overrides: dict | None) -> None:
    if not overrides:
        return
    run_overrides = overrides.get("run") if isinstance(overrides, dict) else None
    if not isinstance(run_overrides, dict):
        return
    for key, value in run_overrides.items():
        if hasattr(cfg.run, key):
            try:
                setattr(cfg.run, key, value)
            except Exception:
                pass


def _set_if_not_none(target: dict, key: str, value) -> None:
    if value is not None:
        target[key] = value


def _energy_params_for_resolution(cfg, slot_resolution: int) -> dict[str, float | int]:
    names = {
        "slot_resolution": slot_resolution,
        "T_minutes": cfg.model.time.T_minutes,
        "T": cfg.model.time.T,
        "trip_duration_minutes": cfg.model.time.trip_duration_minutes,
        "trip_duration": cfg.model.time.trip_duration,
        "trip_slots": cfg.model.time.trip_slots,
    }
    return resolve_energy_params(cfg.model.energy, names)


def _prepare_params(cfg, overrides: dict | None) -> tuple[dict, dict]:
    mp: dict[str, float | int | str | list | bool] = {}
    sp: dict[str, float | int | str | list | bool] = {}

    time = cfg.model.time
    fleet = cfg.model.fleet
    costs = cfg.model.costs

    _set_if_not_none(mp, "T_minutes", time.T_minutes)
    _set_if_not_none(mp, "T", time.T)
    mp["slot_resolution"] = int(time.slot_resolution)
    _set_if_not_none(mp, "trip_duration_minutes", time.trip_duration_minutes)
    _set_if_not_none(mp, "trip_duration", time.trip_duration)
    _set_if_not_none(mp, "trip_slots", time.trip_slots)

    mp["Q"] = int(fleet.Q)
    _set_if_not_none(mp, "binit", fleet.binit)
    _set_if_not_none(mp, "initial_actions", fleet.initial_actions)

    mp.update(_energy_params_for_resolution(cfg, int(time.slot_resolution)))

    _set_if_not_none(mp, "start_cost_epsilon", costs.start_cost_epsilon)
    _set_if_not_none(mp, "concurrency_penalty", costs.concurrency_penalty)

    mp["use_fifo_symmetry"] = bool(cfg.master.use_fifo_symmetry)
    mp["symmetry_breaking"] = bool(cfg.master.symmetry_breaking)
    mp["eps_bin"] = float(cfg.tolerances.eps_bin)
    mp["eps_cut"] = float(cfg.tolerances.eps_cut)
    mp["use_mip_start"] = bool(cfg.master.use_mip_start)
    # These are the starting values for the master's per-iteration controls. The
    # Benders loop's gap-tied schedule overwrites both every iteration, bounded by
    # the same two config values (see BendersSolver: mp_gap_max, mp_tl_cap).
    if cfg.master.per_iteration_time_limit_s is not None:
        mp["solve_time_limit_s"] = int(cfg.master.per_iteration_time_limit_s)
    if cfg.master.per_iteration_mipgap is not None:
        mp["mipgap"] = float(cfg.master.per_iteration_mipgap)
    if cfg.master.cplex_options:
        mp["cplex_options"] = dict(cfg.master.cplex_options)
    if cfg.master.solver_backend:
        mp["solver_backend"] = str(cfg.master.solver_backend)
    mp["aggregate_cuts_by_tau"] = bool(cfg.master.aggregate_cuts_by_tau)
    mp["cut_coeff_threshold"] = float(cfg.master.cut_coeff_threshold)
    mp["theta_per_scenario"] = bool(cfg.master.theta_per_scenario)
    mp["write_lp_after_cut"] = bool(cfg.master.write_lp_after_cut)
    mp["charge_before_idle"] = bool(cfg.master.charge_before_idle)

    mp["solver"] = cfg.solver.master_solver
    mp["solver_tee"] = bool(cfg.solver.solver_tee)
    mp["log_level"] = str(cfg.run.log_level)
    # Off by default (M5). Writing a symbolic LP plus a solver log on every master
    # solve is a debugging aid, not something a normal run should produce -- a
    # 10-iteration run left 20 files behind. Previously this was derived from
    # log_level != "REPORT", i.e. on unless you set a log level that reads as if it
    # would enable reports rather than disable them.
    mp["emit_reports"] = bool(cfg.run.emit_reports)

    sp["lp_solver"] = cfg.solver.subproblem_solver
    sp["multi_cuts_by_scenario"] = bool(cfg.subproblem.multi_cuts_by_scenario)
    sp["use_magnanti_wong"] = bool(cfg.subproblem.use_magnanti_wong)
    sp["mw_core_alpha"] = float(cfg.subproblem.mw_core_alpha)
    sp["mw_core_eps"] = float(getattr(cfg.subproblem, "mw_core_eps", 1e-3))
    sp["use_dual_slopes"] = bool(cfg.subproblem.use_dual_slopes)
    sp["S"] = cfg.subproblem.S
    _set_if_not_none(sp, "Wmax_minutes", cfg.subproblem.Wmax_minutes)
    _set_if_not_none(sp, "Wmax_slots", cfg.subproblem.Wmax_slots)
    sp["p"] = cfg.subproblem.p
    sp["fill_first_epsilon"] = float(cfg.subproblem.fill_first_epsilon)
    sp["degenerate_cut_probe_top_k"] = int(cfg.subproblem.degenerate_cut_probe_top_k)
    _set_if_not_none(sp, "degenerate_cut_probe_top_k_out", cfg.subproblem.degenerate_cut_probe_top_k_out)
    _set_if_not_none(sp, "degenerate_cut_probe_top_k_ret", cfg.subproblem.degenerate_cut_probe_top_k_ret)
    sp["degenerate_cut_zero_tol"] = float(cfg.subproblem.degenerate_cut_zero_tol)
    # tolerances
    sp["eps_cut"] = float(cfg.tolerances.eps_cut)

    _set_if_not_none(sp, "demand_file", cfg.data.demand_file)
    _set_if_not_none(sp, "scenario_files", cfg.data.scenario_files if cfg.data.scenario_files else None)
    _set_if_not_none(sp, "scenario_weights", cfg.data.scenario_weights)
    _set_if_not_none(sp, "R_out", cfg.data.R_out)
    _set_if_not_none(sp, "R_ret", cfg.data.R_ret)
    _set_if_not_none(sp, "scenarios", cfg.data.scenarios)

    sp["slot_resolution"] = int(time.slot_resolution)
    sp["T_minutes"] = int(time.T_minutes) if time.T_minutes is not None else None
    _set_if_not_none(sp, "T", time.T)
    _set_if_not_none(sp, "trip_duration_minutes", time.trip_duration_minutes)
    _set_if_not_none(sp, "trip_duration", time.trip_duration)
    _set_if_not_none(sp, "trip_slots", time.trip_slots)
    sp["Q"] = int(fleet.Q)
    _set_if_not_none(sp, "binit", fleet.binit)
    _set_if_not_none(sp, "initial_actions", fleet.initial_actions)
    sp.update(_energy_params_for_resolution(cfg, int(time.slot_resolution)))
    sp["eps_feas"] = float(cfg.tolerances.eps_feas)
    sp["log_level"] = str(cfg.run.log_level)

    if overrides:
        mp.update((overrides.get("master_params") or {}) if isinstance(overrides, dict) else {})
        sp.update((overrides.get("subproblem_params") or {}) if isinstance(overrides, dict) else {})

    # Propagate slot_resolution from master to subproblem if not explicitly set
    if "slot_resolution" not in sp and "slot_resolution" in mp:
        sp["slot_resolution"] = mp["slot_resolution"]

    # If multi-cuts by scenario is enabled and scenarios present, propagate scenario count/weights to master
    try:
        multi_cuts = bool(sp.get("multi_cuts_by_scenario", False))
    except Exception:
        multi_cuts = False
    scen_list = []
    try:
        if isinstance(sp.get("scenarios"), list) and sp.get("scenarios"):
            scen_list = list(sp.get("scenarios"))
        elif isinstance(sp.get("scenario_files"), list) and sp.get("scenario_files"):
            scen_list = list(sp.get("scenario_files"))
    except Exception:
        scen_list = []
    if multi_cuts and scen_list:
        S = len(scen_list)
        mp.setdefault("theta_per_scenario", True)
        mp["num_scenarios"] = S
        # Pass weights if provided; else default uniform weights summing to 1
        wts = sp.get("scenario_weights")
        if not isinstance(wts, list) or len(wts) != S:
            wts = [1.0 / float(S) for _ in range(S)]
        mp["scenario_weights"] = wts

    return mp, sp


def _build_solver(cfg, mp: dict, sp: dict):
    ProblemMaster, ProblemSubproblem = import_problem_impl()
    master = ProblemMaster(mp)
    sub = ProblemSubproblem(sp)
    solver = BendersSolver(master, sub, cfg)
    return solver, master, sub


def _print_cfg(cfg, mp: dict, sp: dict) -> None:
    print("Run configuration:")
    print(
        f"  solver: iterations={cfg.solver.max_iterations} tol={cfg.solver.tolerance} "
        f"total_time_limit_s={cfg.solver.total_time_limit_s} seed={cfg.run.seed}"
    )
    T_minutes = mp.get("T_minutes")
    slot_res = mp.get("slot_resolution", 1)
    trip_dur_min = mp.get("trip_duration_minutes", mp.get("trip_duration"))
    if T_minutes is not None:
        try:
            T_slots = int(int(T_minutes) // int(slot_res or 1))
        except Exception:
            T_slots = mp.get("T", "-")
    else:
        T_slots = mp.get("T", "-")
    trip_slots = mp.get("trip_slots")
    print(
        "  master: solver=%s Q=%s T_minutes=%s slot_res=%s (slots=%s) trip_dur_min=%s Emax=%s L=%s eps=%s conc_pen=%s delta_chg=%s"
        % (
            mp.get("solver", "-"),
            mp.get("Q", "-"),
            T_minutes if T_minutes is not None else mp.get("T", "-"),
            slot_res,
            T_slots,
            trip_dur_min if trip_dur_min is not None else trip_slots,
            mp.get("Emax", "-"),
            mp.get("L", "-"),
            mp.get("start_cost_epsilon", "-"),
            mp.get("concurrency_penalty", "-"),
            mp.get("delta_chg", "-"),
        )
    )
    try:
        _T = int(T_slots) if isinstance(T_slots, int) else int(mp.get("T"))
        import math

        if trip_dur_min is not None:
            _res = int(slot_res or 1)
            _ts = int(math.ceil(float(trip_dur_min) / max(1, _res)))
        else:
            _ts = int(mp.get("trip_slots"))
        if _ts >= _T:
            print(
                "  NOTE: trip duration (in slots) >= horizon; starts limited to t=0 and may prevent serving demand."
            )
    except Exception:
        pass
    print(
        "  subproblem: solver=%s S=%s Wmax=%s p=%s fill_eps=%s (slot_res=%s)"
        % (
            sp.get("lp_solver", "-"),
            sp.get("S", "-"),
            sp.get("Wmax_minutes", sp.get("Wmax_slots", sp.get("Wmax", "-"))),
            sp.get("p", "-"),
            sp.get("fill_first_epsilon", "-"),
            sp.get("slot_resolution", mp.get("slot_resolution", "-")),
        )
    )
    if "demand_file" in sp:
        print(f"  demand_file: {sp.get('demand_file')}")
    if "scenario_files" in sp:
        print(f"  scenario_files: {sp.get('scenario_files')}")
    if "R_out" in sp:
        print(f"  R_out: {sp.get('R_out')} (inline)")
    if "R_ret" in sp:
        print(f"  R_ret: {sp.get('R_ret')} (inline)")


def _maybe_print_summary(result: BendersRunResult, sp: dict) -> None:
    try:
        if result.pax_served is not None and result.pax_total is not None:
            print(f"Pax served: {result.pax_served:.0f}/{result.pax_total:.0f}")
        # Use subproblem diagnostics for consistent decomposition
        if result.subproblem_obj is not None:
            wait_slots = float(result.sp_wait_cost_slots or 0.0)
            fill_eps = float(result.sp_fill_eps_cost or 0.0)
            pen_cost = float(result.sp_penalty_cost or 0.0)
            pen_pax = float(result.sp_penalty_pax or 0.0)
            total_dem = float(result.sp_total_demand or 0.0)
            slot_res = int(result.sp_slot_resolution or 1)
            sum_components = wait_slots + fill_eps + pen_cost
            if abs(sum_components - float(result.subproblem_obj)) > 1e-5:
                print(
                    "[DIAG] Subproblem objective mismatch: obj=%.6g wait=%.6g fill_eps=%.6g penalty=%.6g sum=%.6g"
                    % (float(result.subproblem_obj), wait_slots, fill_eps, pen_cost, sum_components)
                )
            if wait_slots < -1e-9:
                print(f"[DIAG] Negative waiting cost detected: {wait_slots:.6g}")
            if pen_cost < -1e-9:
                print(f"[DIAG] Negative penalty cost detected: {pen_cost:.6g}")
            wait_per_pax_min = None
            if result.pax_served and result.pax_served > 0:
                wait_per_pax_min = (wait_slots * float(slot_res)) / float(result.pax_served)
            print(
                "Subproblem (last): obj=%.6g wait_slots=%.6g fill_eps=%.6g penalty_cost=%.6g penalty_pax=%.6g total_demand=%.6g"
                % (float(result.subproblem_obj), wait_slots, fill_eps, pen_cost, pen_pax, total_dem)
            )
            if wait_per_pax_min is not None:
                print(f"Avg wait (min): {wait_per_pax_min:.6g}")
        if result.best_upper_bound is not None:
            print(f"UB_total (best): {float(result.best_upper_bound):.6g}")
    except Exception:
        pass
    print(
        f"\nResult: status={result.status} iterations={result.iterations} "
        f"best_lb={result.best_lower_bound} best_ub={result.best_upper_bound}"
    )
    # A run whose master solves stopped on the clock is not bit-reproducible, and
    # must not be quoted as if it were. Measured: a binding per-iteration limit moved
    # the LB 8% between two runs of one config; a non-binding one reproduced exactly.
    truncated = getattr(result, "clock_truncated_master_solves", None)
    if truncated:
        print(
            f"NOT REPRODUCIBLE: {truncated} master solve(s) stopped on the clock, not "
            "the gap. Bounds from this run vary with machine load. For a comparable "
            "number, raise master.per_iteration_time_limit_s and solver.total_time_limit_s "
            "until the master always terminates on the gap (configs/baseline_d9.yaml)."
        )


def _map_candidate_to_warm_start(
    cand: dict[str, float],
    res_old: int,
    res_new: int,
    mp: dict,
) -> dict[tuple[str, int, int], float]:
    # Compute T_new and trip_slots at new resolution to avoid proposing invalid starts
    import math

    T_minutes = mp.get("T_minutes")
    if T_minutes is not None:
        T_new = int(int(T_minutes) // max(1, int(res_new)))
    else:
        T_new = int(mp.get("T", 0))
    trip_min = mp.get("trip_duration_minutes", mp.get("trip_duration"))
    if trip_min is not None:
        trip_slots_new = int(math.ceil(float(trip_min) / max(1, int(res_new))))
    else:
        trip_slots_new = int(mp.get("trip_slots", 0))

    def _map_t(t_old: int) -> int:
        # Map by minutes, rounding to nearest slot at new resolution
        minutes = int(t_old) * int(res_old)
        return int(round(minutes / float(max(1, int(res_new)))))

    starts: dict[tuple[str, int, int], float] = {}
    for k, v in (cand or {}).items():
        if not isinstance(k, str) or float(v or 0.0) < 0.5:
            continue
        if k.startswith("yOUT[") or k.startswith("yRET["):
            inside = k[k.find("[") + 1 : k.find("]")]
            q_str, t_str = inside.split(",")
            try:
                q = int(q_str.strip())
                t_old = int(t_str.strip())
            except Exception:
                continue
            t_new = _map_t(t_old)
            if not (0 <= t_new < T_new):
                continue
            # Respect last-start feasibility windows at new resolution
            if t_new > (T_new - trip_slots_new - 1):
                continue
            typ = "yOUT" if k.startswith("yOUT[") else "yRET"
            starts[(typ, q, t_new)] = 1.0
    return starts


def _run_single(
    cfg,
    mp: dict,
    sp: dict,
    emit_cli_output: bool,
    warm_start: dict | None = None,
    emit_summary: bool = True,
):
    solver, master, _sub = _build_solver(cfg, mp, sp)
    if warm_start:
        try:
            master.set_warm_start(warm_start)
        except Exception:
            pass
    if emit_cli_output:
        _print_cfg(cfg, mp, sp)
    result = solver.run()
    if emit_cli_output and emit_summary:
        _maybe_print_summary(result, sp)
    return result, master


def _emit_manifest(cfg, config_path, result, master, emit_cli_output: bool) -> None:
    """Write the run manifest (spec §0.4). Never fails a run.

    A manifest that cannot be written must not take a completed solve with it,
    but the failure has to be visible -- silently skipping provenance is how the
    invalid-lower-bound problem became hard to scope after the fact.
    """
    from .manifest import build_manifest, write_manifest

    try:
        diag = {
            "cut_generation_mode": getattr(result, "cut_generation_mode", None),
            "cut_valid_lower_bound": getattr(result, "cut_valid_lower_bound", None),
        }
        repo_root = Path(__file__).resolve().parents[2]
        out_dir = Path(cfg.run.report_dir) if cfg.run.report_dir else (repo_root / "manifests")
        manifest = build_manifest(cfg, Path(config_path) if config_path else None, result, repo_root, diag)
        path = write_manifest(manifest, out_dir, cfg.run.name)
        if emit_cli_output:
            print(f"\nManifest: {path}")
    except Exception as exc:  # noqa: BLE001 - provenance must not break a solve
        print(f"\n[WARN] could not write run manifest: {exc!r}")


def run(config_path: str | Path | None = None, overrides: dict | None = None) -> BendersRunResult:
    """Run the Benders solver with a single canonical execution path.

    Parameters are taken from configs/default.yaml by default.
    """
    cfg = load_config(config_path)
    _apply_run_overrides(cfg, overrides)
    setup_logging(cfg.run.log_level)
    report_mode = str(cfg.run.log_level).upper() == "REPORT"

    mp_base, sp_base = _prepare_params(cfg, overrides)
    emit_cli_output = bool(overrides.get("emit_cli_output")) if overrides else False

    multi_res = overrides.get("multi_res") if isinstance(overrides, dict) else None
    if multi_res:
        seq = [int(x) for x in multi_res if str(x).strip()]
        if not seq:
            raise ValueError("No valid resolutions provided for multi-res run.")
        prev_cand: dict[str, float] | None = None
        prev_res: int | None = None
        last_result: BendersRunResult | None = None
        for i, res in enumerate(seq, start=1):
            mp = dict(mp_base)
            sp = dict(sp_base)
            try:
                prev_slot_res = int(mp.get("slot_resolution", res))
            except Exception:
                prev_slot_res = int(res)
            mp["slot_resolution"] = int(res)
            sp["slot_resolution"] = int(res)
            mp.update(_energy_params_for_resolution(cfg, int(res)))
            if cfg.model.energy.delta_chg is not None and not isinstance(cfg.model.energy.delta_chg, str):
                try:
                    if "delta_chg" in mp:
                        mp["delta_chg"] = float(mp["delta_chg"]) * (
                            float(res) / max(1.0, float(prev_slot_res))
                        )
                except Exception:
                    pass
            if emit_cli_output:
                if not report_mode:
                    print(f"\n=== Multi-res stage {i}/{len(seq)}: slot_resolution={res} ===")
            warm_start = None
            if prev_cand is not None and prev_res is not None:
                warm_start = _map_candidate_to_warm_start(prev_cand, prev_res, int(res), mp)
                if emit_cli_output and warm_start:
                    if not report_mode:
                        print(f"Applied warm start with {len(warm_start)} start(s).")
            result, master = _run_single(
                cfg,
                mp,
                sp,
                emit_cli_output,
                warm_start=warm_start,
                emit_summary=False,
            )
            last_result = result
            if emit_cli_output:
                if not report_mode:
                    print(
                        f"Stage {i} result: status={result.status} iters={result.iterations} "
                        f"LB={result.best_lower_bound} UB={result.best_upper_bound}"
                    )
            try:
                prev_cand = getattr(master, "_collect_candidate")()
                prev_res = int(res)
            except Exception:
                prev_cand = None
                prev_res = None
        if last_result is None:
            raise ValueError("Multi-res run produced no results.")
        return last_result

    result, _master = _run_single(cfg, mp_base, sp_base, emit_cli_output)
    _emit_manifest(cfg, config_path, result, _master, emit_cli_output)
    return result


__all__ = ["DEFAULT_CONFIG_PATH", "import_problem_impl", "run"]
