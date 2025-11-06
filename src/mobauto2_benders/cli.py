import argparse
import sys
from pathlib import Path

from .config import load_config
from .logging_config import setup_logging
from .benders.solver import BendersSolver


def _import_problem_impl():
    """Try to import default problem-specific implementations.

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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mobauto2-benders",
        description="Benders decomposition runner for MobAuto2",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
        help="Path to TOML config (default: configs/default.toml)",
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
    result = solver.run()

    # Print concise summary
    print(
        f"status={result.status} iterations={result.iterations} "
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

