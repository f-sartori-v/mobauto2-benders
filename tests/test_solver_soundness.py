"""End-to-end invariants. REQUIRES CPLEX; takes a couple of minutes.

Every assertion here corresponds to a defect that was live in this codebase and
was found by hand. They exist so the next regression is caught automatically
instead of by someone noticing an implausible number.

Run just these:
    python -m unittest tests.test_solver_soundness -v
"""
from __future__ import annotations

import contextlib
import io
import re
import unittest
from pathlib import Path

import _helpers  # noqa: F401  (puts src/ on sys.path)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "soundness.yaml"

# A feasible objective demonstrated by an earlier run of this model. The true
# optimum is at most this. Any lower bound above it proves the master is not a
# relaxation -- which is exactly how the invalid symmetry constraint was caught.
KNOWN_FEASIBLE_UB = 4190.74


def _run_once():
    """Run the short Benders loop, returning (result, captured stdout)."""
    from mobauto2_benders.app import run as app_run

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = app_run(FIXTURE, {"emit_cli_output": True})
    return result, buf.getvalue()


class SoundnessTests(unittest.TestCase):
    """One solve shared by all assertions; it is the expensive part."""

    result = None
    output = ""

    @classmethod
    def setUpClass(cls):
        try:
            cls.result, cls.output = _run_once()
        except Exception as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"solver unavailable: {exc}")

    # -- Magnanti-Wong actually runs ------------------------------------

    def test_magnanti_wong_succeeds(self):
        """solve_mw_dual returned None on EVERY call for an unknown length of
        time, so every cut came from the finite-difference fallback, which is not
        a valid lower bound. Nothing failed loudly -- the run simply reported
        bounds it had no right to. The dual was built with pi declared
        non-negative when the correct sign is non-positive, making the LP
        unbounded."""
        self.assertNotIn("[MW FAIL]", self.output)
        self.assertNotIn("[SP WARN]", self.output)

    def test_cut_mode_is_reported_and_is_mw(self):
        """cut_mode_used was set before the dispatch and never updated, so MW
        runs were labelled "dual" and the multi-scenario aggregate reported
        "mode=-". A reported number must state the mode that produced it."""
        modes = set(re.findall(r"mode=([A-Za-z_]+)", self.output))
        self.assertTrue(modes, "no cut-generation mode reported at all")
        self.assertNotIn("-", modes)
        self.assertIn("mw", modes)

    # -- Bound validity --------------------------------------------------

    def test_lower_bound_does_not_exceed_a_known_feasible_objective(self):
        """The check that exposed the invalid symmetry constraint.

        Prefix-ordering symmetry breaking removed feasible schedules, so the
        master was not a relaxation and its bound sat above a solution we had
        already exhibited. Two mathematically equivalent models produced
        disjoint bound intervals, which is impossible if both bounds are valid.
        """
        lb = self.result.best_lower_bound
        if lb is None:
            self.skipTest("no lower bound claimed; nothing to validate")
        self.assertLessEqual(
            lb, KNOWN_FEASIBLE_UB + 1e-6,
            f"LB={lb} exceeds a known feasible objective {KNOWN_FEASIBLE_UB}; "
            "the master is not a valid relaxation",
        )

    def test_lower_bound_does_not_exceed_upper_bound(self):
        lb, ub = self.result.best_lower_bound, self.result.best_upper_bound
        if lb is None or ub is None:
            self.skipTest("bounds not both available")
        self.assertLessEqual(lb, ub + 1e-6, f"LB={lb} > UB={ub}")

    def test_no_check_fail_lines(self):
        """D13. solver.py emits [CHECK FAIL] when a cut is invalid or the bounds
        cross. Absence across the instance set is the empirical closure of the
        sign-convention question."""
        self.assertNotIn("[CHECK FAIL]", self.output)

    def test_gapped_run_is_not_reported_as_optimal(self):
        """Spec §0 non-negotiable 6: a gapped run must never be presented as an
        optimum without a marker."""
        from mobauto2_benders.benders.types import SolveStatus

        lb, ub = self.result.best_lower_bound, self.result.best_upper_bound
        if lb is None or ub is None:
            return
        gap = abs(ub - lb) / max(1.0, abs(ub))
        if gap > 1e-3:
            self.assertNotEqual(self.result.status, SolveStatus.OPTIMAL)

    # -- Report self-consistency -----------------------------------------

    def test_per_shuttle_table_sums_to_served_total(self):
        """The report drew the schedule from one solution and the passenger
        counts from another, printing a table summing to 126 directly beneath
        "Pax served: 222/300". The total was right, the breakdown was wrong, and
        the total made the table look validated."""
        block = self.output.split("Pax per shuttle and slot (total):")
        if len(block) < 2:
            self.skipTest("per-shuttle table not printed")
        tail = block[-1]
        rows = re.findall(r"^\s+q=\d+:\s+(.*)$", tail, re.M)
        if not rows:
            self.skipTest("no shuttle rows found")
        table_total = sum(float(v) for row in rows for v in row.split())
        m = re.search(r"Pax served: (\d+)/(\d+)", tail)
        self.assertIsNotNone(m, "served total not printed")
        self.assertAlmostEqual(
            table_total, float(m.group(1)), delta=0.5,
            msg="per-shuttle table does not account for every served passenger",
        )

    def test_no_table_mismatch_warning(self):
        self.assertNotIn("per-shuttle table sums to", self.output)

    def test_served_never_exceeds_demand(self):
        if self.result.pax_served is None or self.result.pax_total is None:
            self.skipTest("passenger totals unavailable")
        self.assertLessEqual(self.result.pax_served, self.result.pax_total)


if __name__ == "__main__":
    unittest.main()
