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
    from mobauto2_benders.logging_config import setup_logging  # type: ignore
    from mobauto2_benders.benders.solver import BendersSolver  # type: ignore
else:
    from .config import load_config
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

    sub.add_parser("run", help="Run the Benders solver loop")
    sub.add_parser("validate", help="Validate config and problem stubs")
    sub.add_parser("info", help="Show current configuration")
    return p


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    setup_logging(cfg.run.log_level)

    ProblemMaster, ProblemSubproblem = _import_problem_impl()
    master = ProblemMaster(cfg.master.params)
    sub = ProblemSubproblem(cfg.subproblem.params)

    solver = BendersSolver(master, sub, cfg)
    # Friendly header with initial parameters
    print("Run configuration:")
    print(
        f"  run: iterations={cfg.run.max_iterations} tol={cfg.run.tolerance} "
        f"time_limit_s={cfg.run.time_limit_s} seed={cfg.run.seed}"
    )
    mp = cfg.master.params or {}
    sp = cfg.subproblem.params or {}
    # Derive slots from minutes + resolution if provided
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
        "  master: solver=%s Q=%s T_minutes=%s slot_res=%s (slots=%s) trip_dur_min=%s Emax=%s L=%s delta_chg=%s" % (
            mp.get("solver", "-"), mp.get("Q", "-"), T_minutes if T_minutes is not None else mp.get("T", "-"), slot_res, T_slots, trip_dur_min if trip_dur_min is not None else trip_slots, mp.get("Emax", "-"), mp.get("L", "-"), mp.get("delta_chg", "-"),
        )
    )
    # Inform when trip duration exceeds horizon after discretization
    try:
        _T = int(T_slots) if isinstance(T_slots, int) else int(mp.get("T"))
        import math
        if trip_dur_min is not None:
            _res = int(slot_res or 1)
            _ts = int(math.ceil(float(trip_dur_min) / max(1, _res)))
        else:
            _ts = int(mp.get("trip_slots"))
        if _ts >= _T:
            print("  NOTE: trip duration (in slots) >= horizon; starts limited to t=0 and may prevent serving demand.")
    except Exception:
        pass
    print(
        "  subproblem: solver=%s S=%s Wmax=%s p=%s (slot_res=%s)" % (
            sp.get("lp_solver", "-"), sp.get("S", "-"), sp.get("Wmax_minutes", sp.get("Wmax_slots", sp.get("Wmax", "-"))), sp.get("p", "-"), sp.get("slot_resolution", mp.get("slot_resolution", "-")),
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
    result = solver.run()
    # Final summary
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
