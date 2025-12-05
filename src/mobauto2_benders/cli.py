import argparse
import sys
from pathlib import Path

# Allow running as a standalone script (python path/to/cli.py)
if __package__ in (None, ""):
    THIS_FILE = Path(__file__).resolve()
    SRC_ROOT = THIS_FILE.parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from mobauto2_benders.config import load_config  # type: ignore
    from mobauto2_benders.config import _resolve_param_expressions as _eval_param_exprs  # type: ignore
    from mobauto2_benders.logging_config import setup_logging  # type: ignore
    from mobauto2_benders.benders.solver import BendersSolver  # type: ignore
else:
    from .config import load_config
    from .config import _resolve_param_expressions as _eval_param_exprs
    from .logging_config import setup_logging
    from .benders.solver import BendersSolver


def _import_problem_impl():
    """Try to import default problem-specific implementations.

    Expects classes `ProblemMaster` and `ProblemSubproblem` in
    `mobauto2_benders.problem.master_impl` and `.subproblem_impl`.
    """
    try:
        if __package__ in (None, ""):
            # Executed as a script: import via absolute package path
            from mobauto2_benders.problem.master_impl import ProblemMaster  # type: ignore
            from mobauto2_benders.problem.subproblem_impl import ProblemSubproblem  # type: ignore
        else:
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mobauto2-benders",
        description="Benders decomposition runner for MobAuto2",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config. Default: configs/default.yaml",
    )
    sub = p.add_subparsers(dest="cmd")
    sub.required = False

    run_p = sub.add_parser("run", help="Run the Benders solver loop")
    run_p.add_argument(
        "--multi-res",
        dest="multi_res",
        type=str,
        default=None,
        help="Comma-separated slot resolutions to run coarse-to-fine, e.g. '30,15,5,1'",
    )
    run_p.add_argument(
        "--mw",
        dest="mw",
        action="store_true",
        help="Enable Magnanti–Wong (Pareto-optimal) cut selection",
    )
    run_p.add_argument(
        "--mw-alpha",
        dest="mw_alpha",
        type=float,
        default=None,
        help="Core-point mixing factor alpha in (0,1]; default from config or 0.3",
    )
    sub.add_parser("validate", help="Validate config and problem stubs")
    sub.add_parser("info", help="Show current configuration")
    return p


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    setup_logging(cfg.run.log_level)

    ProblemMaster, ProblemSubproblem = _import_problem_impl()
    mp = dict(cfg.master.params or {})
    sp = dict(cfg.subproblem.params or {})
    # Propagate slot_resolution from master to subproblem if not explicitly set
    if "slot_resolution" not in sp and "slot_resolution" in mp:
        sp["slot_resolution"] = mp["slot_resolution"]
    # Enable MW via CLI switch if requested
    if getattr(args, "mw", False):
        sp["use_magnanti_wong"] = True
    if getattr(args, "mw_alpha", None) is not None:
        sp["mw_core_alpha"] = float(args.mw_alpha)

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

    def _print_cfg(mp_local: dict, sp_local: dict) -> None:
        print("Run configuration:")
        print(
            f"  run: iterations={cfg.run.max_iterations} tol={cfg.run.tolerance} "
            f"time_limit_s={cfg.run.time_limit_s} seed={cfg.run.seed}"
        )
        T_minutes = mp_local.get("T_minutes")
        slot_res = mp_local.get("slot_resolution", 1)
        trip_dur_min = mp_local.get("trip_duration_minutes", mp_local.get("trip_duration"))
        if T_minutes is not None:
            try:
                T_slots = int(int(T_minutes) // int(slot_res or 1))
            except Exception:
                T_slots = mp_local.get("T", "-")
        else:
            T_slots = mp_local.get("T", "-")
        trip_slots = mp_local.get("trip_slots")
        print(
            "  master: solver=%s Q=%s T_minutes=%s slot_res=%s (slots=%s) trip_dur_min=%s Emax=%s L=%s eps=%s conc_pen=%s delta_chg=%s" % (
                mp_local.get("solver", "-"), mp_local.get("Q", "-"), T_minutes if T_minutes is not None else mp_local.get("T", "-"), slot_res, T_slots, trip_dur_min if trip_dur_min is not None else trip_slots, mp_local.get("Emax", "-"), mp_local.get("L", "-"), mp_local.get("start_cost_epsilon", "-"), mp_local.get("concurrency_penalty", "-"), mp_local.get("delta_chg", "-"),
            )
        )
        try:
            _T = int(T_slots) if isinstance(T_slots, int) else int(mp_local.get("T"))
            import math
            if trip_dur_min is not None:
                _res = int(slot_res or 1)
                _ts = int(math.ceil(float(trip_dur_min) / max(1, _res)))
            else:
                _ts = int(mp_local.get("trip_slots"))
            if _ts >= _T:
                print("  NOTE: trip duration (in slots) >= horizon; starts limited to t=0 and may prevent serving demand.")
        except Exception:
            pass
        print(
            "  subproblem: solver=%s S=%s Wmax=%s p=%s fill_eps=%s (slot_res=%s)" % (
                sp_local.get("lp_solver", "-"), sp_local.get("S", "-"), sp_local.get("Wmax_minutes", sp_local.get("Wmax_slots", sp_local.get("Wmax", "-"))), sp_local.get("p", "-"), sp_local.get("fill_first_epsilon", "-"), sp_local.get("slot_resolution", mp_local.get("slot_resolution", "-")),
            )
        )
        if "demand_file" in sp_local:
            print(f"  demand_file: {sp_local.get('demand_file')}")
        if "scenario_files" in sp_local:
            print(f"  scenario_files: {sp_local.get('scenario_files')}")
        if "R_out" in sp_local:
            print(f"  R_out: {sp_local.get('R_out')} (inline)")
        if "R_ret" in sp_local:
            print(f"  R_ret: {sp_local.get('R_ret')} (inline)")

    def _map_candidate_to_warm_start(cand: dict[str, float], res_old: int, res_new: int, mp_local: dict) -> dict[tuple[str, int, int], float]:
        # Compute T_new and trip_slots at new resolution to avoid proposing invalid starts
        import math
        T_minutes = mp_local.get("T_minutes")
        if T_minutes is not None:
            T_new = int(int(T_minutes) // max(1, int(res_new)))
        else:
            T_new = int(mp_local.get("T", 0))
        trip_min = mp_local.get("trip_duration_minutes", mp_local.get("trip_duration"))
        if trip_min is not None:
            trip_slots_new = int(math.ceil(float(trip_min) / max(1, int(res_new))))
        else:
            trip_slots_new = int(mp_local.get("trip_slots", 0))
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

    # Multi-resolution run (coarse -> fine)
    if getattr(args, "multi_res", None):
        seq = [int(x.strip()) for x in str(args.multi_res).split(",") if x.strip()]
        if not seq:
            print("No valid resolutions given to --multi-res; exiting.")
            return 2
        prev_cand: dict[str, float] | None = None
        prev_res: int | None = None
        for i, res in enumerate(seq, start=1):
            # Capture previous resolution before overriding
            try:
                prev_slot_res = int(mp.get("slot_resolution", res))
            except Exception:
                prev_slot_res = int(res)
            mp["slot_resolution"] = int(res)
            # Ensure subproblem sees the same resolution
            sp["slot_resolution"] = int(res)
            # Re-evaluate arithmetic expressions that depend on slot_resolution (e.g., delta_chg)
            mp = _eval_param_exprs(dict(mp))
            sp = _eval_param_exprs(dict(sp))
            # If delta_chg was already numeric (from initial config evaluation), scale it with resolution
            try:
                if "delta_chg" in mp:
                    mp["delta_chg"] = float(mp["delta_chg"]) * (float(res) / max(1.0, float(prev_slot_res)))
            except Exception:
                pass
            master = ProblemMaster(mp)
            sub = ProblemSubproblem(sp)
            solver = BendersSolver(master, sub, cfg)
            print(f"\n=== Multi-res stage {i}/{len(seq)}: slot_resolution={res} ===")
            _print_cfg(mp, sp)
            if prev_cand is not None and prev_res is not None:
                starts = _map_candidate_to_warm_start(prev_cand, prev_res, int(res), mp)
                if starts:
                    try:
                        master.set_warm_start(starts)
                        print(f"Applied warm start with {len(starts)} start(s).")
                    except Exception:
                        pass
            result = solver.run()
            print(
                f"Stage {i} result: status={result.status} iters={result.iterations} "
                f"LB={result.best_lower_bound} UB={result.best_upper_bound}"
            )
            # Capture candidate from the solved master for next stage warm start
            try:
                prev_cand = getattr(master, "_collect_candidate")()
                prev_res = int(res)
            except Exception:
                prev_cand = None
                prev_res = None
        return 0

    # Single-resolution run (default)
    master = ProblemMaster(mp)
    sub = ProblemSubproblem(sp)
    solver = BendersSolver(master, sub, cfg)
    _print_cfg(mp, sp)
    result = solver.run()
    print(
        f"\nResult: status={result.status} iterations={result.iterations} "
        f"best_lb={result.best_lower_bound} best_ub={result.best_upper_bound}"
    )
    return 0


def cmd_validate(args) -> int:
    cfg = load_config(args.config)
    setup_logging(cfg.run.log_level)
    try:
        _import_problem_impl()
        print("Config OK. Problem stubs found.")
        return 0
    except SystemExit as e:  # from _import_problem_impl
        print("Config OK. Problem stubs missing:")
        print(e)
        return 1


def cmd_info(args) -> int:
    cfg = load_config(args.config)
    print(cfg)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd in (None, "run"):
        return cmd_run(args)
    if args.cmd == "validate":
        return cmd_validate(args)
    if args.cmd == "info":
        return cmd_info(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
