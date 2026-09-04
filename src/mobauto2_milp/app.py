from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
from pathlib import Path
import subprocess
import sys

from .config import DEFAULT_CONFIG_PATH, load_config, resolve_energy_params
from .logging_config import setup_logging
from .monolith import MonolithSolver
from .solver import RunResult


_QUIET_RUN_LEVELS = {"REPORT", "FINAL"}


def _run_level(cfg) -> str:
    return str(cfg.run.log_level).upper()


def import_problem_impl():
    """Import the integrated MILP model implementation."""
    try:
        from .model import MobautoMilpModel  # type: ignore
        return MobautoMilpModel
    except Exception as exc:  # noqa: BLE001 - provide friendly message
        raise SystemExit(
            "Integrated MILP model implementation not found.\n"
            "Expected class `MobautoMilpModel` in:\n"
            "  src/mobauto2_milp/model.py\n"
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

    # B2. Same default resolution as the Benders master: None means Q, decided inside
    # the model so the two engines cannot drift apart on it.
    _set_if_not_none(mp, "K_chg", cfg.model.energy.K_chg)
    mp["charger_occupancy_binary"] = bool(cfg.model.energy.charger_occupancy_binary)

    _set_if_not_none(mp, "start_cost_epsilon", costs.start_cost_epsilon)
    _set_if_not_none(mp, "concurrency_penalty", costs.concurrency_penalty)
    # B7.2. Read by MonolithSolver.run, which dispatches to the lexicographic
    # hierarchy or to the single weighted-sum solve.
    mp["objective_mode"] = str(costs.objective_mode)

    mp["use_fifo_symmetry"] = bool(cfg.milp.use_fifo_symmetry)
    mp["symmetry_breaking"] = bool(cfg.milp.symmetry_breaking)
    mp["eps_bin"] = 1.0e-5
    mp["eps_cut"] = 1.0e-8
    mp["use_mip_start"] = bool(cfg.milp.use_mip_start)
    if cfg.milp.solve_time_limit_s is not None:
        mp["solve_time_limit_s"] = int(cfg.milp.solve_time_limit_s)
    if cfg.milp.mipgap is not None:
        mp["mipgap"] = float(cfg.milp.mipgap)
    if cfg.milp.cplex_options:
        mp["cplex_options"] = dict(cfg.milp.cplex_options)
    if cfg.milp.solver_backend:
        mp["solver_backend"] = str(cfg.milp.solver_backend)
    mp["aggregate_cuts_by_tau"] = True
    mp["cut_coeff_threshold"] = 0.0
    mp["theta_per_scenario"] = False
    mp["write_lp_after_cut"] = False

    mp["solver"] = str(cfg.milp.solver_backend)
    mp["solver_tee"] = bool(cfg.solver.solver_tee)
    mp["log_level"] = str(cfg.run.log_level)
    mp["emit_reports"] = _run_level(cfg) not in _QUIET_RUN_LEVELS

    sp["lp_solver"] = str(cfg.milp.solver_backend)
    sp["multi_cuts_by_scenario"] = False
    sp["use_magnanti_wong"] = False
    sp["mw_core_alpha"] = 0.3
    sp["mw_core_eps"] = 1e-3
    sp["use_dual_slopes"] = False
    sp["S"] = cfg.service.S
    _set_if_not_none(sp, "Wmax_minutes", cfg.service.Wmax_minutes)
    _set_if_not_none(sp, "Wmax_slots", cfg.service.Wmax_slots)
    sp["p"] = cfg.service.p
    sp["fill_first_epsilon"] = float(cfg.service.fill_first_epsilon)
    sp["unused_capacity_penalty"] = 0.0
    sp["aggregate_service_by_start"] = True
    sp["eps_cut"] = 1.0e-8

    _set_if_not_none(sp, "demand_file", cfg.data.demand_file)
    _set_if_not_none(sp, "scenario_files", cfg.data.scenario_files if cfg.data.scenario_files else None)
    _set_if_not_none(sp, "scenario_weights", cfg.data.scenario_weights)
    _set_if_not_none(sp, "R_out", cfg.data.R_out)
    _set_if_not_none(sp, "R_ret", cfg.data.R_ret)
    _set_if_not_none(sp, "scenarios", cfg.data.scenarios)

    sp["slot_resolution"] = int(time.slot_resolution)
    sp["log_level"] = str(cfg.run.log_level)

    if overrides:
        mp.update((overrides.get("master_params") or {}) if isinstance(overrides, dict) else {})
        sp.update((overrides.get("subproblem_params") or {}) if isinstance(overrides, dict) else {})

    # Propagate slot_resolution from master to subproblem if not explicitly set
    if "slot_resolution" not in sp and "slot_resolution" in mp:
        sp["slot_resolution"] = mp["slot_resolution"]

    # Propagate scenario count/weights to the integrated model whenever scenarios are present.
    scen_list = []
    try:
        if isinstance(sp.get("scenarios"), list) and sp.get("scenarios"):
            scen_list = list(sp.get("scenarios"))
        elif isinstance(sp.get("scenario_files"), list) and sp.get("scenario_files"):
            scen_list = list(sp.get("scenario_files"))
    except Exception:
        scen_list = []
    if scen_list:
        S = len(scen_list)
        mp["num_scenarios"] = S
        wts = sp.get("scenario_weights")
        if not isinstance(wts, list) or len(wts) != S:
            wts = [1.0 / float(S) for _ in range(S)]
        mp["scenario_weights"] = wts

    return mp, sp


def _print_cfg(cfg, mp: dict, sp: dict) -> None:
    print("Run configuration:")
    print(
        f"  solver: backend={mp.get('solver', '-')} time_limit_s={mp.get('solve_time_limit_s', 3600)} "
        f"mipgap={mp.get('mipgap', '-')} seed={cfg.run.seed}"
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
        "  milp: solver=%s Q=%s T_minutes=%s slot_res=%s (slots=%s) trip_dur_min=%s Emax=%s L=%s eps=%s conc_pen=%s delta_chg=%s"
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
        "  service: solver=%s S=%s Wmax=%s p=%s fill_eps=%s (slot_res=%s)"
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


def _maybe_print_summary(result: RunResult, sp: dict) -> None:
    try:
        if result.solve_started_at is not None and result.solve_finished_at is not None:
            started = result.solve_started_at.strftime("%Y-%m-%d %H:%M:%S")
            finished = result.solve_finished_at.strftime("%Y-%m-%d %H:%M:%S")
            elapsed = (
                f"{float(result.elapsed_wall_s):.3f}s"
                if result.elapsed_wall_s is not None
                else f"{(result.solve_finished_at - result.solve_started_at).total_seconds():.3f}s"
            )
            print(f"Solver time: start={started} end={finished} elapsed={elapsed}")
        if result.pax_served is not None and result.pax_total is not None:
            print(f"Pax served: {result.pax_served:.0f}/{result.pax_total:.0f}")
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
                "Service block: obj=%.6g wait_slots=%.6g fill_eps=%.6g penalty_cost=%.6g penalty_pax=%.6g total_demand=%.6g"
                % (float(result.subproblem_obj), wait_slots, fill_eps, pen_cost, pen_pax, total_dem)
            )
            if wait_per_pax_min is not None:
                print(f"Avg wait (min): {wait_per_pax_min:.6g}")
        if result.best_upper_bound is not None:
            print(f"UB_total (best): {float(result.best_upper_bound):.6g}")
    except Exception:
        pass
    print(f"\nResult: status={result.status} iterations={result.iterations} best_lb={result.best_lower_bound} best_ub={result.best_upper_bound}")


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
    emit_detailed_report: bool = True,
):
    model_cls = import_problem_impl()
    solver = MonolithSolver(model_cls, cfg, mp, sp)
    if warm_start:
        try:
            solver.master.set_warm_start(warm_start)
        except Exception:
            pass
    if emit_cli_output:
        _print_cfg(cfg, mp, sp)
    result = solver.run()
    if emit_cli_output and emit_summary:
        _maybe_print_summary(result, sp)
        if solver.has_incumbent_solution():
            try:
                print("\nBest Master Solution:")
                print(solver.format_solution())
            except Exception:
                pass
        else:
            print("\nNo incumbent solution loaded; detailed schedule is unavailable.")
        if emit_detailed_report:
            try:
                report = solver.format_report()
                if report:
                    print("")
                    print(report)
            except Exception:
                pass
    return result, solver.master


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _path_from_project_root(project_root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root / p


def get_demand_file_list(cfg) -> list[str]:
    if getattr(cfg.data, "demand_files", None):
        return [str(p) for p in cfg.data.demand_files]
    if getattr(cfg.data, "demand_file", None):
        return [str(cfg.data.demand_file)]
    raise ValueError("Config must contain either data.demand_file or data.demand_files.")


def _plot_solver_output(project_root: Path, demand_file: str, solver_output_path: Path, scenario_dir: Path) -> None:
    plot_script = project_root / "aux_py" / "plot_figs.py"
    if not plot_script.exists():
        print(f"WARNING: plot_figs.py not found at {plot_script}")
        return
    subprocess.run(
        [
            sys.executable,
            str(plot_script),
            "--demand-file",
            str(_path_from_project_root(project_root, demand_file)),
            "--solver-output",
            str(solver_output_path),
            "--out-dir",
            str(scenario_dir),
        ],
        cwd=project_root,
        check=True,
    )


def _run_batch_demands(cfg, config_path: str | Path | None, overrides: dict | None) -> RunResult:
    project_root = _project_root()
    last_result: RunResult | None = None

    for demand_file in get_demand_file_list(cfg):
        scenario_name = Path(demand_file).stem
        scenario_dir = project_root / "outputs" / "demand_variations" / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        solver_output_path = scenario_dir / "solver-output"

        print(f"\n=== Running scenario: {scenario_name} ===", flush=True)
        print(f"Demand file: {demand_file}", flush=True)
        print(f"Output folder: {scenario_dir}", flush=True)

        run_cfg = copy.deepcopy(cfg)
        run_cfg.data.demand_file = demand_file
        run_cfg.data.demand_files = []

        mp, sp = _prepare_params(run_cfg, overrides)
        emit_cli_output = True
        emit_detailed_report = _run_level(run_cfg) != "FINAL"

        with solver_output_path.open("w", encoding="utf-8") as f:
            with redirect_stdout(f), redirect_stderr(f):
                result, _master = _run_single(
                    run_cfg,
                    mp,
                    sp,
                    emit_cli_output,
                    emit_detailed_report=emit_detailed_report,
                )
        last_result = result
        _plot_solver_output(project_root, demand_file, solver_output_path, scenario_dir)

    if last_result is None:
        raise ValueError("No demand files were run.")
    return last_result

def run(config_path: str | Path | None = None, overrides: dict | None = None) -> RunResult:
    """Run the monolithic MILP solver with a single canonical execution path.

    Parameters are taken from configs/default.yaml by default.
    """
    cfg = load_config(config_path)
    _apply_run_overrides(cfg, overrides)
    setup_logging(cfg.run.log_level)
    log_level = _run_level(cfg)
    report_mode = log_level in _QUIET_RUN_LEVELS
    emit_detailed_report = log_level != "FINAL"

    if cfg.data.demand_files:
        return _run_batch_demands(cfg, config_path, overrides)

    mp_base, sp_base = _prepare_params(cfg, overrides)
    emit_cli_output = bool(overrides.get("emit_cli_output")) if overrides else False

    multi_res = overrides.get("multi_res") if isinstance(overrides, dict) else None
    if multi_res:
        seq = [int(x) for x in multi_res if str(x).strip()]
        if not seq:
            raise ValueError("No valid resolutions provided for multi-res run.")
        prev_cand: dict[str, float] | None = None
        prev_res: int | None = None
        last_result: RunResult | None = None
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
                emit_detailed_report=emit_detailed_report,
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

    result, _master = _run_single(
        cfg,
        mp_base,
        sp_base,
        emit_cli_output,
        emit_detailed_report=emit_detailed_report,
    )
    return result


__all__ = ["DEFAULT_CONFIG_PATH", "import_problem_impl", "run"]
