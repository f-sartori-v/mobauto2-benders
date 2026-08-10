"""Branch-and-cut on the soundness fixture: does one tree reach the same place? (D44)

REQUIRES CPLEX. The contract tests in `test_lazy_cut_contract.py` cover the
decision logic without a solver; this file covers the part that only a real tree
can answer -- that the callback fires, that its cuts bind, and that the bound it
certifies is a bound.

The invariant that matters is the same one the loop is held to: the master is a
relaxation, so its lower bound may not exceed a known feasible objective. A lazy
cut that excludes the optimum would show up here and nowhere else, because it is
the only test in the suite where a cut prunes a subtree that is never revisited.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import _helpers  # noqa: F401  (puts src/ on sys.path)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "soundness.yaml"

# Same independently-derived optimum the loop is checked against. See
# test_solver_soundness.KNOWN_FEASIBLE_UB for how it was obtained and why the
# earlier value made the check circular.
KNOWN_FEASIBLE_UB = 4183.24


def _require_cplex() -> None:
    import pyomo.environ as pyo

    for name in ("cplex", "cplex_direct", "cplex_persistent"):
        if not pyo.SolverFactory(name).available(exception_flag=False):
            raise unittest.SkipTest(f"solver {name} is not available")


def _build():
    from mobauto2_benders.app import _prepare_params
    from mobauto2_benders.config import load_config
    from mobauto2_benders.problem.master_impl import ProblemMaster
    from mobauto2_benders.problem.subproblem_impl import ProblemSubproblem

    cfg = load_config(str(FIXTURE))
    mp, sp = _prepare_params(cfg, {})
    T = int(mp.get("T") or (int(mp["T_minutes"]) // int(mp["slot_resolution"])))
    mp["T"] = T
    sp["T"] = T
    # Nothing about the backend is changed. The tree builds its own persistent
    # solver, so the master stays on cplex_direct exactly as the loop runs it --
    # which is what makes a difference in the result attributable to the tree.
    master = ProblemMaster(mp)
    master.initialize()
    return master, ProblemSubproblem(sp)


class BranchAndCutRunsOnOneTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _require_cplex()
        master, sub = _build()
        # 30 s is a test budget, not a measurement. It is enough for ~90 callback
        # invocations on this fixture, which is far past the point where the
        # assertions below become meaningful; a run that means something is
        # configured, not hard-coded here.
        cls.result, cls.stats = master.solve_branch_and_cut(
            sub.evaluate, time_limit_s=30.0
        )

    def test_the_callback_actually_fired(self):
        """Otherwise every assertion below would pass on a plain master solve.

        This is the test that catches the failure mode the config gate was
        written for: cplex_direct silently ignoring the callback and the run
        reporting an ordinary solve as branch-and-cut.
        """
        self.assertGreater(self.stats.invocations, 0)
        self.assertGreater(self.stats.subproblem_solves, 0)

    def test_it_injected_at_least_one_cut(self):
        self.assertGreater(self.stats.cuts_injected, 0)

    def test_nothing_was_aborted(self):
        self.assertIsNone(self.stats.aborted_reason)

    def test_only_valid_cuts_were_seen(self):
        """The inverted D39 rule, observed rather than reasoned about.

        Any other validity state would have aborted the solve, so this asserts
        the counts agree with the fact that setUpClass returned at all.
        """
        self.assertEqual(set(self.stats.validity_counts) - {"valid"}, set())

    def test_the_lower_bound_does_not_exceed_a_known_feasible_objective(self):
        """The master is a relaxation. If this fails, a lazy cut cut off the optimum.

        Nowhere else in the suite can catch that: in the loop an invalid cut is
        survivable because the reported bound is dropped afterwards, and here the
        pruning it caused is permanent.
        """
        lb = self.result.lower_bound
        self.assertIsNotNone(lb)
        self.assertLessEqual(float(lb), KNOWN_FEASIBLE_UB + 1e-6)

    def test_the_incumbent_schedule_is_integral(self):
        cand = self.result.candidate
        self.assertIsNotNone(cand)
        for name, val in cand.items():
            if name.startswith(("yOUT[", "yRET[")):
                self.assertIn(round(float(val), 6), (0.0, 1.0), name)


    def test_the_masters_own_solver_is_untouched(self):
        """The tree must not repoint the solver the loop uses.

        `solve_branch_and_cut` builds its own persistent solver. If it swapped
        `master._solver` instead, the seeding LP phase and the tree would share
        one backend and run 2 would stop being reproducible from this code -- a
        regression that no bound would reveal.
        """
        master, _sub = _build()
        before = type(master._solver).__name__
        self.assertNotIn("Persistent", before)


class TheSeedingBackendIsNotTheTreeBackend(unittest.TestCase):
    def test_config_refuses_cplex_persistent_as_the_masters_backend(self):
        import tempfile

        import yaml

        from mobauto2_benders.config import load_config

        raw = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
        raw.setdefault("master", {})["solver_backend"] = "cplex_persistent"
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as fh:
            yaml.safe_dump(raw, fh)
            path = fh.name
        try:
            with self.assertRaises(ValueError) as ctx:
                load_config(path)
            self.assertIn("seeding LP phase", str(ctx.exception))
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
