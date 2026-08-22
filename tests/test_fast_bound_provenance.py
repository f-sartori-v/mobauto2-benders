"""The portable dual-bound source, and the sense check that keeps it valid.

`res.solver.best_bound` is populated by the CPLEX plugins and by nothing else. On
any other backend the master solved to proven optimality and reported no bound at
all: measured on `appsi_highs` against tests/fixtures/soundness.yaml, every
iteration came back `term=optimal status=ok lb_valid=valid` with
`mp_best_bound=-`, the run ended `best_lower_bound=None`, and the two
bound-validity invariants in test_solver_soundness -- LB <= a known feasible
objective, and LB <= UB -- skipped THEMSELVES for want of a bound to check.

`_bounds_from_problem_section` reads Pyomo's generic `res.problem` section
instead. The dangerous half is the sense: on a minimisation the dual bound is
`lower_bound`, and reading that same field on a maximisation would return the
PRIMAL side and claim a lower bound at or above the optimum. That is the one error
a lower bound must never make, and it is the shape of both C4 and D30. These tests
exist so the sense check cannot be dropped by someone simplifying the function.

No solver needed: the results object is constructed directly.

Run just these:
    python -m unittest tests.test_fast_bound_provenance -v
"""

from __future__ import annotations

import unittest

import _helpers  # noqa: F401  (puts src/ on sys.path)

import pyomo.environ as pyo
from pyomo.opt.results.results_ import SolverResults

from mobauto2_benders.problem.master_impl import _bounds_from_problem_section


def _results(lower=None, upper=None, sense=pyo.minimize):
    res = SolverResults()
    if lower is not None:
        res.problem.lower_bound = lower
    if upper is not None:
        res.problem.upper_bound = upper
    if sense is not None:
        res.problem.sense = sense
    return res


class TestTheSenseDecidesWhichFieldIsTheDualBound(unittest.TestCase):
    def test_minimisation_takes_the_lower_bound(self):
        got = _bounds_from_problem_section(_results(2.5, 4.0, pyo.minimize))
        self.assertEqual(got["best_bound"], 2.5)
        self.assertEqual(got["incumbent"], 4.0)

    def test_maximisation_takes_the_upper_bound(self):
        """The half that would be an invalid bound if the sense were assumed."""
        got = _bounds_from_problem_section(_results(2.5, 4.0, pyo.maximize))
        self.assertEqual(got["best_bound"], 4.0)
        self.assertEqual(got["incumbent"], 2.5)

    def test_an_unknown_sense_returns_nothing(self):
        """Refusing beats guessing: a bound read off the wrong field is worse than
        no bound, because the run reports it as if it were one."""
        self.assertEqual(_bounds_from_problem_section(_results(2.5, 4.0, None)), {})


class TestNonNumbersNeverBecomeBounds(unittest.TestCase):
    def test_an_unset_section_yields_no_bounds(self):
        """Pyomo leaves `UndefinedData` in place rather than None, so a truthiness
        or `is None` test would let the sentinel through as a bound."""
        self.assertEqual(_bounds_from_problem_section(_results(None, None)), {})

    def test_infinities_are_not_bounds(self):
        """An unbounded section reports +/-inf. Carried forward it is not a bound
        anyone can act on, and it poisons the gap computed from it downstream."""
        got = _bounds_from_problem_section(
            _results(float("-inf"), float("inf"), pyo.minimize)
        )
        self.assertEqual(got, {})

    def test_a_results_object_without_a_problem_section_is_handled(self):
        class _Bare:
            pass

        self.assertEqual(_bounds_from_problem_section(_Bare()), {})


class TestItIsLastInTheProvenanceChain(unittest.TestCase):
    """CPLEX sources must still win, so a licensed run resolves exactly as before
    and no archived number moves."""

    def test_the_source_label_names_this_path(self):
        import inspect

        from mobauto2_benders.problem import master_impl

        src = inspect.getsource(master_impl)
        self.assertIn('sources[_k] = "problem_section"', src)
        # It must come after the cplex_api and cplex_log fallbacks in the file.
        self.assertLess(src.index('sources[k] = "cplex_api"'), src.index('"problem_section"'))
        self.assertLess(src.index("parsed_best_bound"), src.index('"problem_section"'))


class TestTheBackendResolverDoesNotSubstituteSilently(unittest.TestCase):
    """Same principle one level up: a result must name the instrument that made it.

    `_helpers.solver_backend` is what lets the soundness invariants run without a
    CPLEX licence. The risk it introduces is attribution -- someone who pinned a
    solver getting a different one, or getting nothing, and reading the result as
    if the pin held.
    """

    def test_a_pin_that_is_not_installed_raises_rather_than_skipping(self):
        import os

        prev = os.environ.get("MOBAUTO2_TEST_SOLVER")
        os.environ["MOBAUTO2_TEST_SOLVER"] = "a_solver_that_does_not_exist"
        try:
            with self.assertRaises(RuntimeError) as ctx:
                _helpers.solver_backend()
            self.assertIn("a_solver_that_does_not_exist", str(ctx.exception))
        finally:
            if prev is None:
                os.environ.pop("MOBAUTO2_TEST_SOLVER", None)
            else:
                os.environ["MOBAUTO2_TEST_SOLVER"] = prev

    def test_cplex_is_preferred_when_it_is_present(self):
        """A licensed checkout must keep measuring what it measured before."""
        self.assertEqual(_helpers._SOLVER_PREFERENCE[0], "cplex_direct")

    def test_rewriting_a_config_only_touches_keys_it_already_has(self):
        """Both loaders reject unknown keys, and the two config families name the
        solver differently. Adding `solver.master_solver` to a monolith config
        fails its load outright, which is how this was found."""
        import yaml

        rewritten = _helpers.fixture_for_backend(
            _helpers.CONFIGS / "milp" / "phase5_tiny_monolith.yaml"
        )
        raw = yaml.safe_load(rewritten.read_text(encoding="utf-8"))
        self.assertNotIn("master_solver", raw.get("solver", {}))
        self.assertNotIn("subproblem_solver", raw.get("solver", {}))
        self.assertEqual(
            raw["milp"]["solver_backend"], _helpers.require_solver_backend()
        )


if __name__ == "__main__":
    unittest.main()
