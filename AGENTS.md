# Repository Guidelines

## Project Structure & Module Organization
Core package code lives under `src/mobauto2_benders/`. Use `benders/` for the generic decomposition loop, `problem/` for MobAuto2-specific model implementations, and top-level modules such as `app.py`, `cli.py`, and `config.py` for orchestration, CLI entry points, and YAML loading. Default runtime settings live in `configs/default.yaml`. Design notes are in `docs/`, reference material in `Report/`, helper scripts in `scripts/`, and tests in `tests/`.

## Build, Test, and Development Commands
Create or activate a Python 3.10 environment first; `p310.cmd` is the local helper for that workflow.

- `pip install -e .` installs the package in editable mode.
- `mobauto2-benders run` runs the default configuration.
- `python -m mobauto2_benders validate` checks config loading and problem stubs.
- `python -m mobauto2_benders info` prints the active configuration summary.
- `python -m pytest tests` runs the test suite.
- `python scripts/diagnostics_smoke.py` is useful for quick diagnostics when changing solver or logging behavior.

## Coding Style & Naming Conventions
Follow standard Python conventions: 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and explicit type hints where practical. Keep new code aligned with the existing style: small focused functions, dataclasses for structured results, and `pathlib.Path` for filesystem paths. There is no formatter configured in `pyproject.toml`, so keep formatting PEP 8 compliant and avoid introducing tool-specific style drift.

## Testing Guidelines
Tests currently live in `tests/` and follow `test_*.py` naming. Existing coverage uses `unittest`, but the suite is run cleanly through `pytest`. Add targeted unit tests for parser, config, and solver-logic changes; avoid tests that require a commercial solver unless guarded or clearly documented.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects such as `fix versioning.` and `update: tolerances margin implementation.` Keep commits focused and scoped to one change. For pull requests, include the problem being solved, the key files touched, config or solver assumptions, and the commands you ran to validate the change. Include sample CLI output when behavior or reporting changes.

## Configuration & Solver Notes
Default execution assumes Pyomo plus YAML support and usually a CPLEX-backed solver. If you change solver-related defaults, update `README.md`, `configs/default.yaml`, and any affected diagnostics together.
