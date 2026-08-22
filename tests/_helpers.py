"""Shared helpers. Keeps the src/ layout importable without installing."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONFIGS = REPO_ROOT / "configs"

# The tracked reference config. NOT configs/default.yaml, which is untracked and
# is the live experiment file: it is edited by runs, by sweeps and by the parallel
# external audit working in this folder, so a test reading it asserts against
# whatever someone is setting up right now. That already happened -- six D29 tests
# failed the moment the live file went multi-scenario, and the tests were not
# wrong. Reading the tracked copy also means the suite runs on a fresh clone,
# where configs/default.yaml does not exist at all.
DEFAULT_CONFIG = "default.example.yaml"


def load_cfg(name: str):
    from mobauto2_benders.config import load_config

    return load_config(CONFIGS / name)


def master_params(name: str = "baseline_d9.yaml") -> dict:
    """Master params exactly as app.py would assemble them."""
    from mobauto2_benders.app import _prepare_params

    mp, _sp = _prepare_params(load_cfg(name), {})
    mp = dict(mp)
    # _prepare_params leaves T implicit when the config gives T_minutes; the
    # solver derives it. Do the same here so the model can be built standalone.
    if mp.get("T") is None:
        t_min = int(mp.get("T_minutes") or 0)
        res = int(mp.get("slot_resolution") or 1)
        mp["T"] = max(1, t_min // max(1, res))
    return mp


def build_master(params: dict):
    from mobauto2_benders.problem.master_impl import ProblemMaster

    pm = ProblemMaster(params)
    pm.initialize()
    return pm


def constraint_names(model) -> set[str]:
    import pyomo.environ as pyo

    return {c.name for c in model.component_objects(pyo.Constraint)}


# --------------------------------------------------------------------------
# Backend selection for the solver-gated tests.
#
# Every gate in tests/ named `cplex`/`cplex_direct` literally, so a checkout
# without a CPLEX licence skipped 63 tests and printed a green suite that had
# checked NONE of the end-to-end soundness invariants -- E1-E4, cut
# underestimation, the Phase-5 equality against the monolith. That is the same
# shape as the defect `_require_solvers` was written to end: "not run" reading
# as "passed".
#
# Nothing in those invariants is CPLEX-specific. They are properties of the
# FORMULATION, and any backend that solves an LP to optimality and returns duals
# can check them. HiGHS does, and installs as a pip wheel with no licence.
#
# CPLEX stays first in the preference order, so a licensed checkout keeps
# measuring exactly what it measured before and no archived number moves.
#
# What this does NOT cover: `cplex_persistent`. Branch-and-cut needs a lazy
# constraint callback, HiGHS has no callback interface, and a substitute that
# silently ran the tree without the callback would assert the D44 contract
# against a solver that never registered it. tests/test_branch_and_cut.py
# therefore keeps its own CPLEX-only gate on purpose.
# --------------------------------------------------------------------------

_SOLVER_PREFERENCE = ("cplex_direct", "appsi_highs")

_ENV_PIN = "MOBAUTO2_TEST_SOLVER"


def _solver_is_available(name: str) -> bool:
    import pyomo.environ as pyo

    try:
        return bool(pyo.SolverFactory(name).available(exception_flag=False))
    except Exception:
        # Ask the question directly: an unavailable solver is one state, and a
        # solver that raised while being asked is another. Neither is "the test
        # passed", but only the first is a legitimate skip.
        return False


def solver_backend():
    """The backend the solver-gated tests run on, or None if none is available.

    `MOBAUTO2_TEST_SOLVER` pins one explicitly. A pin that is not installed
    RAISES rather than skipping: someone who named a solver wants that solver,
    and quietly running a different one -- or quietly running nothing -- is how
    a result gets attributed to the wrong instrument.
    """
    import os

    pinned = os.environ.get(_ENV_PIN)
    if pinned:
        if not _solver_is_available(pinned):
            raise RuntimeError(
                f"{_ENV_PIN}={pinned!r} is not available to Pyomo. Install it or "
                f"unset {_ENV_PIN} to fall back to {' then '.join(_SOLVER_PREFERENCE)}."
            )
        return pinned
    for name in _SOLVER_PREFERENCE:
        if _solver_is_available(name):
            return name
    return None


def require_solver_backend() -> str:
    """Return the backend, or skip naming what went unchecked."""
    import unittest

    backend = solver_backend()
    if backend is None:
        raise unittest.SkipTest(
            "not run: no LP/MIP backend available (tried "
            f"{', '.join(_SOLVER_PREFERENCE)}). These are end-to-end soundness "
            "invariants; a green suite without them has not checked any of them. "
            "`pip install highspy` is enough to run them."
        )
    return backend


def repoint_solvers(cfg, backend: str | None = None) -> str:
    """Point a loaded config's three solver fields at `backend`.

    The fixtures pin `cplex`/`cplex_direct` by name, and the master's own
    `solver_backend` is the one the seeding LP phase runs on. Repointing all
    three together is what keeps a run internally consistent -- a master on one
    backend and a subproblem on another is a configuration nobody measured.
    """
    backend = backend or require_solver_backend()
    cfg.solver.master_solver = backend
    cfg.solver.subproblem_solver = backend
    cfg.master.solver_backend = backend
    if backend != "cplex_direct":
        # CPXPARAM_* keys mean nothing to another backend, and master_impl maps
        # them by name. Dropping them is honest; passing them on is not.
        cfg.master.cplex_options = {}
    return backend


def fixture_for_backend(fixture_path, backend: str | None = None):
    """A copy of `fixture_path` whose solver keys name the resolved backend.

    `app.run` takes a config PATH and its override dict reaches `cfg.run` only, so
    a test that drives the whole loop cannot repoint the solver in memory the way
    `repoint_solvers` does. Rewriting the file is the honest alternative: the
    fixture stays the single source of every other parameter, and only the three
    solver names and the CPLEX-specific option block move.

    Returns the original path unchanged when the backend is already what the
    fixture names, so a licensed checkout drives the exact tracked file and no
    archived number can drift on a copy.
    """
    import tempfile

    import yaml

    backend = backend or require_solver_backend()
    raw = yaml.safe_load(Path(fixture_path).read_text(encoding="utf-8"))
    _named = {
        raw.get("solver", {}).get("master_solver"),
        raw.get("solver", {}).get("subproblem_solver"),
        raw.get("master", {}).get("solver_backend"),
        raw.get("milp", {}).get("solver_backend"),
    } - {None}
    if _named == {backend}:
        return Path(fixture_path)

    # Two config families live in this repo and they name the solver differently:
    # the Benders configs under solver/master, the monolith's under `milp`. Rewrite
    # whichever keys the file actually has, and do not invent the others -- adding a
    # `milp:` block to a Benders config would fail its own unknown-key check.
    # Two config families live in this repo and they name the solver differently:
    # the Benders configs under `solver`/`master`, the monolith's under `milp`.
    # Rewrite only keys the file ALREADY has. Both loaders reject unknown keys, so
    # adding `solver.master_solver` to a monolith config -- whose `solver` block
    # holds only `solver_tee` -- fails the load outright.
    for section, key in (
        ("solver", "master_solver"),
        ("solver", "subproblem_solver"),
        ("master", "solver_backend"),
        ("milp", "solver_backend"),
    ):
        if isinstance(raw.get(section), dict) and key in raw[section]:
            raw[section][key] = backend
    if backend != "cplex_direct":
        for section in ("master", "milp"):
            if isinstance(raw.get(section), dict):
                raw[section].pop("cplex_options", None)

    tmp = Path(tempfile.mkdtemp(prefix="mobauto2_fixture_")) / Path(fixture_path).name
    tmp.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return tmp
