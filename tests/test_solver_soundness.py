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
import logging
import re
import unittest
from pathlib import Path

import _helpers  # noqa: F401  (puts src/ on sys.path)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "soundness.yaml"

# A feasible objective demonstrated by an earlier run of this model. The true
# optimum is at most this. Any lower bound above it proves the master is not a
# relaxation -- which is exactly how the invalid symmetry constraint was caught.
KNOWN_FEASIBLE_UB = 4190.74


_CACHE: tuple | None = None


def _run_once():
    """Run the short Benders loop once, returning (result, captured stdout).

    Memoised: every assertion in this module examines the same solve, so running
    it per TestCase class would multiply the only slow part of the suite.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    from mobauto2_benders.app import run as app_run

    # The solver logs progress at INFO to stderr. The assertions read stdout, so
    # silencing it only removes noise from the test report.
    prev = logging.getLogger("mobauto2_benders").level
    logging.getLogger("mobauto2_benders").setLevel(logging.WARNING)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = app_run(FIXTURE, {"emit_cli_output": True})
        _CACHE = (result, buf.getvalue())
    finally:
        logging.getLogger("mobauto2_benders").setLevel(prev)
    return _CACHE


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



class ManifestTests(unittest.TestCase):
    """Spec §0.4. The manifest must actually capture provenance.

    Its two most important fields are which generator produced the cuts and
    whether they support a lower bound -- the pair whose absence made the
    invalid-bound problem impossible to scope retrospectively. A manifest that
    records None for them looks complete and tells you nothing, so assert they
    are populated rather than merely present.
    """

    @classmethod
    def setUpClass(cls):
        try:
            cls.result, cls.output = _run_once()
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"solver unavailable: {exc}")

    def test_result_carries_cut_provenance(self):
        self.assertIsNotNone(self.result.cut_generation_mode)
        self.assertIsNotNone(self.result.cut_valid_lower_bound)

    def test_manifest_records_cut_provenance(self):
        from pathlib import Path as _P
        from mobauto2_benders.config import load_config
        from mobauto2_benders.manifest import build_manifest

        cfg = load_config(FIXTURE)
        m = build_manifest(
            cfg, _P(FIXTURE), self.result, _P(__file__).resolve().parents[1],
            {
                "cut_generation_mode": self.result.cut_generation_mode,
                "cut_valid_lower_bound": self.result.cut_valid_lower_bound,
            },
        )
        self.assertIsNotNone(m["cut_generation"]["mode_used"])
        self.assertIsNotNone(m["cut_generation"]["valid_lower_bound"])

    def test_manifest_records_swept_and_objective_parameters(self):
        """D2/D3 sweep W_max and p; concurrency_penalty is active but absent
        from the published formulation. All must be traceable from a result."""
        from pathlib import Path as _P
        from mobauto2_benders.config import load_config
        from mobauto2_benders.manifest import build_manifest

        cfg = load_config(FIXTURE)
        m = build_manifest(cfg, _P(FIXTURE), self.result, _P(__file__).resolve().parents[1], {})
        self.assertIsNotNone(m["swept_parameters"]["p"])
        self.assertIsNotNone(m["swept_parameters"]["Wmax_minutes"])
        self.assertIsNotNone(m["objective_terms"]["concurrency_penalty"])
        self.assertIsNotNone(m["config"]["sha256"])

    def test_manifest_records_whether_the_run_is_reproducible(self):
        """A run that stopped on the clock looks exactly like one that converged.

        Same failure shape as the two defects the manifest was built for: without
        a recorded flag, deciding retrospectively which archived numbers were
        measurements and which were single draws is guesswork. Measured, the
        difference is not small -- a binding per-iteration limit moved the LB 8%
        between two runs of one config.
        """
        from pathlib import Path as _P
        from mobauto2_benders.config import load_config
        from mobauto2_benders.manifest import build_manifest

        cfg = load_config(FIXTURE)
        m = build_manifest(cfg, _P(FIXTURE), self.result, _P(__file__).resolve().parents[1], {})
        rep = m["reproducibility"]
        self.assertIsNotNone(rep["clock_truncated_master_solves"])
        self.assertEqual(rep["clock_truncated_master_solves"], 0)
        self.assertTrue(rep["bit_reproducible"])
        # The settings that decide it must travel with the verdict.
        self.assertIsNotNone(m["solver"]["per_iteration_time_limit_s"])
        self.assertIsNotNone(m["solver"]["total_time_limit_s"])

    def test_manifest_marks_a_clock_truncated_run_as_not_reproducible(self):
        """The flag must follow the run, not the config."""
        from pathlib import Path as _P
        from copy import copy
        from mobauto2_benders.config import load_config
        from mobauto2_benders.manifest import build_manifest

        truncated = copy(self.result)
        truncated.clock_truncated_master_solves = 3

        cfg = load_config(FIXTURE)
        m = build_manifest(cfg, _P(FIXTURE), truncated, _P(__file__).resolve().parents[1], {})
        self.assertEqual(m["reproducibility"]["clock_truncated_master_solves"], 3)
        self.assertFalse(m["reproducibility"]["bit_reproducible"])

if __name__ == "__main__":
    unittest.main()


class TestDemandOutsideHorizonIsReported(unittest.TestCase):
    """Requests past the horizon were dropped with no warning and no count.

    `_aggregate_requests` filtered with `if not (0 <= t < Tlen): continue`, so a
    660-minute horizon fed an 830-minute instance counted 224 of 284 requests and
    then printed "Pax served: 173/224". The denominator had already lost 21% of
    demand, which understates unmet demand -- the metric weighted by p, and the
    headline number of the whole study.

    Relevant to D6: the horizon is to extend from 10h to 24h, at which point the
    trap moves rather than disappearing.
    """

    def _evaluate_with_requests(self, requests, T_minutes=60, slot_resolution=30):
        from mobauto2_benders.problem.subproblem_impl import ProblemSubproblem

        T = T_minutes // slot_resolution
        sp = ProblemSubproblem({
            "T": T,
            "T_minutes": T_minutes,
            "slot_resolution": slot_resolution,
            "trip_duration_minutes": slot_resolution,
            "Q": 1,
            "S": 15.0,
            "Emax": 150.0,
            "L": 30.0,
            "delta_chg": 35.0,
            "Wmax_minutes": slot_resolution,
            "p": 50.0,
            "lp_solver": "cplex_direct",
            "scenarios": [{"requests": requests}],
            "use_magnanti_wong": False,
            "eps_cut": 1e-8,
        })
        candidate = {f"yOUT[0,{t}]": 0.0 for t in range(T)}
        candidate.update({f"yRET[0,{t}]": 0.0 for t in range(T)})
        with self.assertLogs("mobauto2_benders.problem.subproblem_impl", level="WARNING") as cm:
            sp.evaluate(candidate)
        return "\n".join(cm.output)

    def test_requests_past_the_horizon_are_counted_and_warned(self):
        requests = [
            {"dir": "OUT", "time": 10},    # inside
            {"dir": "OUT", "time": 700},   # past a 60-minute horizon
            {"dir": "RET", "time": 900},   # past
        ]
        log = self._evaluate_with_requests(requests)
        self.assertIn("[DEMAND]", log)
        self.assertIn("after_horizon=2", log)
        self.assertIn("900", log, "the warning should name the latest request time")


class TestClockTruncationIsReported(unittest.TestCase):
    """A time-limited master solve makes the whole run machine-dependent.

    configs/baseline_d9.yaml has always listed this as a determinism requirement:
    the per-iteration limit must be "high enough that the master always terminates
    on mipgap, never on wall clock". Measured, the rule is right -- one config run
    twice with a NON-binding limit reproduced to the last digit (LB
    2422.5195186024557 both times), and the same config with a binding 15s limit
    gave 2333.29, 2153.79 and 2175.87 across three runs.

    The requirement was violated for the repo's whole history (CPXPARAM_Threads was
    dropped, D24) and again by D22's remaining-budget clamp. A run that breaks it
    must say so instead of looking reproducible.
    """

    def test_converged_reference_run_is_not_clock_truncated(self):
        result, _ = _run_once()
        truncated = getattr(result, "clock_truncated_master_solves", None)
        self.assertIsNotNone(
            truncated, "the run must report whether it was clock-truncated at all"
        )
        self.assertEqual(
            truncated,
            0,
            "the soundness fixture must terminate on the gap, or every bound it "
            "asserts is a sample rather than a measurement",
        )
