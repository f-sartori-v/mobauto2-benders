import argparse
import sys
from pathlib import Path

# Allow running as a standalone script (python path/to/cli.py)
if __package__ in (None, ""):
    THIS_FILE = Path(__file__).resolve()
    SRC_ROOT = THIS_FILE.parents[1]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from mobauto2_benders.app import DEFAULT_CONFIG_PATH, import_problem_impl, run as app_run  # type: ignore
    from mobauto2_benders.config import load_config  # type: ignore
else:
    from .app import DEFAULT_CONFIG_PATH, import_problem_impl, run as app_run
    from .config import load_config


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mobauto2-benders",
        description="Benders decomposition runner for MobAuto2",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
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
        help=(
            "Enable Magnanti-Wong-inspired dual selection. NOT certified "
            "Pareto-optimal: that requires the core point in the relative interior "
            "of conv(Y), which nothing here establishes (B14)."
        ),
    )
    run_p.add_argument(
        "--mw-alpha",
        dest="mw_alpha",
        type=float,
        default=None,
        help="Core-point mixing factor alpha in (0,1]; default from config or 0.3",
    )
    run_p.add_argument(
        "--solver",
        dest="solver",
        type=str,
        default=None,
        help=(
            "Override the solver backend for master, subproblem and the seeding LP "
            "phase, e.g. 'appsi_highs'. Every shipped config names a CPLEX plugin; "
            "this is what runs them without a licence. Timings from a non-CPLEX "
            "backend are NOT comparable with anything in docs/."
        ),
    )
    sub.add_parser("validate", help="Validate config and problem stubs")
    sub.add_parser("info", help="Show current configuration")
    return p


def _parse_multi_res(value: str | None) -> list[int] | None:
    if value is None:
        return None
    seq = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    return seq


def cmd_run(args) -> int:
    overrides: dict = {"emit_cli_output": True}

    if getattr(args, "mw", False):
        overrides.setdefault("subproblem_params", {})["use_magnanti_wong"] = True
    if getattr(args, "mw_alpha", None) is not None:
        overrides.setdefault("subproblem_params", {})["mw_core_alpha"] = float(
            args.mw_alpha
        )
    if getattr(args, "solver", None):
        overrides["solver_backend"] = str(args.solver)
    if getattr(args, "multi_res", None):
        seq = _parse_multi_res(args.multi_res)
        if not seq:
            print("No valid resolutions given to --multi-res; exiting.")
            return 2
        overrides["multi_res"] = seq

    app_run(args.config, overrides)
    return 0


def cmd_validate(args) -> int:
    cfg_path = args.config or DEFAULT_CONFIG_PATH
    cfg = load_config(cfg_path)
    try:
        import_problem_impl()
        print("Config OK. Problem stubs found.")
        _print_config_summary(cfg)
        return 0
    except SystemExit as e:  # from import_problem_impl
        print("Config OK. Problem stubs missing:")
        print(e)
        _print_config_summary(cfg)
        return 1


def cmd_info(args) -> int:
    cfg_path = args.config or DEFAULT_CONFIG_PATH
    cfg = load_config(cfg_path)
    _print_config_summary(cfg)
    return 0


def _print_config_summary(cfg) -> None:
    print(f"Schema: {cfg.schema.name} v{cfg.schema.version}")
    demand = cfg.data.demand_file or (
        "inline R_out/R_ret" if cfg.data.R_out or cfg.data.R_ret else "none"
    )
    print(f"Data: demand={demand}")
    if cfg.data.scenario_files:
        print(f"Data: scenario_files={len(cfg.data.scenario_files)}")
    if cfg.data.scenario_weights:
        print(f"Data: scenario_weights={cfg.data.scenario_weights}")
    print(
        "Solver: max_iterations=%s tolerance=%s time_limit_s=%s"
        % (
            cfg.solver.max_iterations,
            cfg.solver.tolerance,
            cfg.solver.total_time_limit_s,
        )
    )


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
