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

# The optimum of this instance, from an independent monolithic MILP of the same
# model, cross-checked by evaluating its schedule with THIS codebase's subproblem
# (4183.00 recourse + 0.24 start cost). Any lower bound above it proves the master
# is not a relaxation, or that the cuts are not valid lower bounds.
#
# This was 4190.74 for most of the project's life -- a number taken from a Benders
# run of this same code, which made the check circular: it could not detect a
# defect that inflated the bound by less than 7.5. It did not detect the layered
# subproblem (D30), whose cuts pushed the reported LB to 4492 on a longer run.
KNOWN_FEASIBLE_UB = 4183.24


_CACHE: tuple | None = None
_FAILURE: BaseException | None = None

# The solvers the fixture names. Kept next to FIXTURE so the two cannot drift.
_REQUIRED_SOLVERS = ("cplex", "cplex_direct")


def _require_solvers(*names: str) -> None:
    """Skip only when a solver is genuinely absent.

    This exists because the three entry points below used to wrap their work in
    `except Exception: raise SkipTest("solver unavailable")`. That turned EVERY
    failure into a skip with a label blaming the environment -- a config error, a
    RuntimeError from the D39 validity guards, an AssertionError, a bug in this
    file. The class whose tests each correspond to a defect that was once live
    would switch itself off and report the machine's fault.

    Asking the question directly separates the two states: "the solver is not
    installed" is an environment fact worth skipping on, and everything else is a
    failure worth seeing. `SolverFactory` returns an UnknownSolver for a name it
    does not recognise rather than raising, so no exception handling is needed
    here either.
    """
    import pyomo.environ as pyo

    missing = [n for n in names if not pyo.SolverFactory(n).available(exception_flag=False)]
    if missing:
        raise unittest.SkipTest(
            "not run: solver(s) unavailable: "
            + ", ".join(missing)
            + ". These are end-to-end soundness invariants; a green suite without "
            "them has not checked any of them."
        )


def _run_once():
    """Run the short Benders loop once, returning (result, captured stdout).

    Memoised: every assertion in this module examines the same solve, so running
    it per TestCase class would multiply the only slow part of the suite.

    A failure is memoised too, and re-raised unchanged. Three classes call this;
    without it, a solve that raises would be retried three times at a couple of
    minutes each. Re-raising is not swallowing -- the caller still sees it.
    """
    global _CACHE, _FAILURE
    if _FAILURE is not None:
        raise _FAILURE
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
    except BaseException as exc:
        _FAILURE = exc
        raise
    finally:
        logging.getLogger("mobauto2_benders").setLevel(prev)
    return _CACHE


class SoundnessTests(unittest.TestCase):
    """One solve shared by all assertions; it is the expensive part."""

    result = None
    output = ""

    @classmethod
    def setUpClass(cls):
        _require_solvers(*_REQUIRED_SOLVERS)
        cls.result, cls.output = _run_once()

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
            lb,
            KNOWN_FEASIBLE_UB + 1e-6,
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
            table_total,
            float(m.group(1)),
            delta=0.5,
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
        _require_solvers(*_REQUIRED_SOLVERS)
        cls.result, cls.output = _run_once()

    def test_result_carries_cut_provenance(self):
        self.assertIsNotNone(self.result.cut_generation_mode)
        self.assertIsNotNone(self.result.cut_valid_lower_bound)

    def test_manifest_records_cut_provenance(self):
        from pathlib import Path as _P
        from mobauto2_benders.config import load_config
        from mobauto2_benders.manifest import build_manifest

        cfg = load_config(FIXTURE)
        m = build_manifest(
            cfg,
            _P(FIXTURE),
            self.result,
            _P(__file__).resolve().parents[1],
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
        m = build_manifest(
            cfg, _P(FIXTURE), self.result, _P(__file__).resolve().parents[1], {}
        )
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
        m = build_manifest(
            cfg, _P(FIXTURE), self.result, _P(__file__).resolve().parents[1], {}
        )
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
        m = build_manifest(
            cfg, _P(FIXTURE), truncated, _P(__file__).resolve().parents[1], {}
        )
        self.assertEqual(m["reproducibility"]["clock_truncated_master_solves"], 3)
        self.assertFalse(m["reproducibility"]["bit_reproducible"])


class RecourseMatchesTheMonolith(unittest.TestCase):
    """The subproblem must price a schedule exactly as an independent MILP does.

    This is the only check in the suite that compares against something outside
    this codebase. Every other guard validates Benders against Benders, which is
    how a layered subproblem whose constraint set moved with y survived: its
    recourse was right, so nothing downstream looked wrong, while its duals were
    not subgradients and the cuts it produced excluded the optimum (D30).

    The schedule below is the MILP's optimum for setups/base.yaml at Q=2, T=22.
    Priced by this subproblem it must come to 4183.00, and with the master's own
    start-cost term to 4183.24 -- which is also KNOWN_FEASIBLE_UB above.
    """

    MILP_OPTIMUM = 4183.24
    SCHEDULE = {
        0: "IDL OUT RET OUT RET CHR OUT RET CHR CHR CHR OUT RET CHR CHR OUT IDL RET OUT RET IDL IDL",
        1: "IDL IDL OUT RET OUT RET CHR OUT IDL RET CHR CHR CHR OUT RET CHR CHR OUT RET OUT RET IDL",
    }

    def test_subproblem_prices_the_milp_optimum_exactly(self):
        from mobauto2_benders.config import load_config
        from mobauto2_benders.app import _prepare_params
        from mobauto2_benders.problem.subproblem_impl import ProblemSubproblem

        cfg = load_config(str(FIXTURE))
        mp, sp = _prepare_params(cfg, {})
        T = int(mp.get("T") or (int(mp["T_minutes"]) // int(mp["slot_resolution"])))
        sp["T"] = T

        acts = {q: a.split() for q, a in self.SCHEDULE.items()}
        self.assertEqual(
            len(acts[0]), T, "fixture horizon no longer matches the schedule"
        )

        candidate, trips = {}, 0
        for q, a in acts.items():
            for t, x in enumerate(a[:T]):
                candidate[f"yOUT[{q},{t}]"] = 1.0 if x == "OUT" else 0.0
                candidate[f"yRET[{q},{t}]"] = 1.0 if x == "RET" else 0.0
                trips += 1 if x in ("OUT", "RET") else 0

        # This is the external oracle (D30): the MILP's own schedule, priced by
        # THIS codebase's subproblem, must reproduce the MILP objective. Wrapping
        # it in a broad catch meant a subproblem that had stopped working was
        # reported as a missing solver.
        _require_solvers(*_REQUIRED_SOLVERS)
        res = ProblemSubproblem(sp).evaluate(candidate)

        total = (
            float(res.upper_bound) + float(mp.get("start_cost_epsilon", 0.0)) * trips
        )
        self.assertAlmostEqual(
            total,
            self.MILP_OPTIMUM,
            places=2,
            msg="the subproblem no longer prices the monolith's optimum; the two "
            "models have diverged and every bound this code reports is suspect",
        )


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

    @classmethod
    def setUpClass(cls):
        # Builds and solves an LP through `lp_solver: cplex_direct` below. Without
        # this the class raised instead of skipping on a machine without CPLEX --
        # the inverse of the defect `_require_solvers` was written to end.
        _require_solvers("cplex_direct")

    def _evaluate_with_requests(self, requests, T_minutes=60, slot_resolution=30):
        from mobauto2_benders.problem.subproblem_impl import ProblemSubproblem

        T = T_minutes // slot_resolution
        sp = ProblemSubproblem(
            {
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
            }
        )
        candidate = {f"yOUT[0,{t}]": 0.0 for t in range(T)}
        candidate.update({f"yRET[0,{t}]": 0.0 for t in range(T)})
        with self.assertLogs(
            "mobauto2_benders.problem.subproblem_impl", level="WARNING"
        ) as cm:
            sp.evaluate(candidate)
        return "\n".join(cm.output)

    def test_requests_past_the_horizon_are_counted_and_warned(self):
        requests = [
            {"dir": "OUT", "time": 10},  # inside
            {"dir": "OUT", "time": 700},  # past a 60-minute horizon
            {"dir": "RET", "time": 900},  # past
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

    @classmethod
    def setUpClass(cls):
        # Calls _run_once(), i.e. a full solve. Same reason as the class above.
        _require_solvers(*_REQUIRED_SOLVERS)

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


class MagnantiWongSelectsANonDominatedCut(unittest.TestCase):
    """MW must pick a dual that DOMINATES, not merely a dual that exists.

    Until now nothing checked this. `test_magnanti_wong_succeeds` asserts MW ran
    instead of falling back, and `test_cut_mode_is_reported_and_is_mw` asserts the
    run labelled its cuts `mw`. Both are provenance. Neither says the cut is any
    good, and D30 is the proof the two are independent: MW, the plain dual and
    finite differences all produced invalid cuts identically, so a green MW label
    has already coexisted with cuts that excluded the optimum.

    What MW claims: the subproblem LP is degenerate, so many duals are optimal.
    Every one of them yields a cut tight at the incumbent; they differ away from
    it. MW maximises the cut's value at a core point Ybar over the optimal face,
    which is what makes the cut Pareto-optimal.

    That gives an invariant which holds BY CONSTRUCTION, so a failure is proof of a
    defect rather than an unlucky instance:

        cut_MW(Ybar) >= cut_dual(Ybar)   for every Ybar

    because the plain dual is just another point on the face MW maximises over.
    This is the check that would have caught the bug the generator's own comment
    records -- an earlier version maximised `sum(dm*Ybar)` and dropped the `-y_inc`
    term, which is not constant over the optimal face, so it selected the wrong
    dual.

    Both configurations set `use_dual_slopes: True`. That flag is not only a
    generator switch: it also floors K at 1, which changes the LP that gets built.
    Toggling only `use_magnanti_wong` keeps the subproblem identical, so both cuts
    are read off the same optimal face -- asserted, not assumed, by comparing the
    recourse value across the pair.
    """

    CORE_POINTS = {
        "uniform": (0.5, 0.3),
        "all_ones": (1.0, 1.0),
        "all_zeros": (0.0, 0.0),
        "out_only": (1.0, 0.0),
        "ret_only": (0.0, 1.0),
    }

    @classmethod
    def setUpClass(cls):
        _require_solvers("cplex_direct")

        from mobauto2_benders.config import load_config
        from mobauto2_benders.app import _prepare_params
        from mobauto2_benders.problem.subproblem_impl import ProblemSubproblem

        cfg = load_config(str(FIXTURE))
        mp, sp = _prepare_params(cfg, {})
        T = int(mp.get("T") or (int(mp["T_minutes"]) // int(mp["slot_resolution"])))
        sp["T"] = T
        Q = int(mp.get("Q", 2))
        cls.T = T

        # A deliberately non-optimal schedule. The point is a degenerate LP with
        # several optimal duals, not a good candidate.
        cand = {}
        for q in range(Q):
            for t in range(T):
                cand[f"yOUT[{q},{t}]"] = 1.0 if (t % 4 == q % 4) else 0.0
                cand[f"yRET[{q},{t}]"] = 1.0 if (t % 4 == (q + 2) % 4) else 0.0
        cls.candidate = cand
        cls.agg_out = [sum(cand[f"yOUT[{q},{t}]"] for q in range(Q)) for t in range(T)]
        cls.agg_ret = [sum(cand[f"yRET[{q},{t}]"] for q in range(Q)) for t in range(T)]

        def evaluate(use_mw, core):
            params = dict(sp)
            params["use_magnanti_wong"] = use_mw
            params["use_dual_slopes"] = True
            params["mw_core_point"] = {"Yout": list(core[0]), "Yret": list(core[1])}
            res = ProblemSubproblem(params).evaluate(dict(cand))
            cut = (res.cuts or [res.cut])[0]
            return cut.metadata, res

        flat = ([0.5] * T, [0.3] * T)
        cls.dual_md, cls.dual_res = evaluate(False, flat)
        cls.mw = {}
        for name, (o, r) in cls.CORE_POINTS.items():
            core = ([o] * T, [r] * T)
            cls.mw[name] = (evaluate(True, core), core)

        # Neighbouring schedules: flip one slot. The true recourse at each is what
        # the cut has to stay under. Priced once here; the assertions are cheap.
        cls.neighbours = []
        for q in range(Q):
            for t in range(0, T, 5):
                for direction in ("yOUT", "yRET"):
                    y = dict(cand)
                    key = f"{direction}[{q},{t}]"
                    y[key] = 1.0 - y[key]
                    if y[f"yOUT[{q},{t}]"] + y[f"yRET[{q},{t}]"] > 1.0:
                        continue  # would break exclusivity; not a schedule
                    priced = ProblemSubproblem(
                        {**sp, "use_magnanti_wong": True, "use_dual_slopes": True}
                    ).evaluate(dict(y))
                    cls.neighbours.append((key, y, float(priced.upper_bound)))
        assert cls.neighbours, "no admissible single-slot perturbation was built"

    def _value_at(self, md, out_vec, ret_vec):
        """Cut value at an aggregate profile; coeff[(q,tau)] is equal across q."""
        co, cr = md["coeff_yOUT"], md["coeff_yRET"]
        return (
            float(md["const"])
            + sum(co.get((0, t), 0.0) * out_vec[t] for t in range(self.T))
            + sum(cr.get((0, t), 0.0) * ret_vec[t] for t in range(self.T))
        )

    def _cut_value(self, md, cand):
        """Cut value at a full per-(q,t) schedule."""
        total = float(md["const"])
        for (q, t), c in md["coeff_yOUT"].items():
            total += c * cand.get(f"yOUT[{q},{t}]", 0.0)
        for (q, t), c in md["coeff_yRET"].items():
            total += c * cand.get(f"yRET[{q},{t}]", 0.0)
        return total

    def _worst_slack(self, generator):
        """Smallest `recourse(y) - cut(y)` over the neighbours, and where."""
        md = self.dual_md if generator == "dual" else self.mw["uniform"][0][0]
        worst, where = float("inf"), None
        for key, y, recourse in self.neighbours:
            slack = recourse - self._cut_value(md, y)
            if slack < worst:
                worst, where = slack, key
        return worst, where

    def test_both_generators_see_the_same_subproblem(self):
        """If the recourse differs, the two cuts came off different LPs and no
        comparison between them means anything. `use_dual_slopes` is held fixed
        precisely because it silently changes the model."""
        (mw_md, mw_res), _ = self.mw["uniform"]
        self.assertAlmostEqual(
            float(mw_res.upper_bound), float(self.dual_res.upper_bound), places=6
        )

    def test_both_cuts_are_tight_at_the_candidate(self):
        """A Benders optimality cut must reproduce the recourse at the point it was
        generated at. If MW's is not tight, the OptFace constraint pinning the dual
        objective to the primal optimum is mis-set, and MW is selecting from the
        wrong set."""
        ub = float(self.dual_res.upper_bound)
        for label, md in (("dual", self.dual_md), ("mw", self.mw["uniform"][0][0])):
            with self.subTest(generator=label):
                self.assertAlmostEqual(
                    self._value_at(md, self.agg_out, self.agg_ret),
                    ub,
                    places=4,
                    msg=f"{label} cut is not tight at the candidate it was built at",
                )

    def test_mw_dominates_the_plain_dual_at_every_core_point(self):
        """The invariant. MW maximises the cut's value at Ybar over the optimal
        face and the plain dual is a point on that face, so MW can never be worse.
        A negative margin means the MW objective or a sign convention is wrong; it
        is not an instance-specific outcome."""
        for name, ((mw_md, _), core) in self.mw.items():
            with self.subTest(core_point=name):
                mw_val = self._value_at(mw_md, core[0], core[1])
                dual_val = self._value_at(self.dual_md, core[0], core[1])
                tol = 1e-6 * max(1.0, abs(dual_val))
                self.assertGreaterEqual(
                    mw_val,
                    dual_val - tol,
                    msg=(
                        f"MW cut is dominated at core point {name}: "
                        f"mw={mw_val!r} < dual={dual_val!r}. MW selects the "
                        "maximiser over the optimal face, so this cannot happen "
                        "unless the selection is wrong."
                    ),
                )

    def test_the_cut_underestimates_the_recourse_at_neighbouring_schedules(self):
        """Strong duality, checked where it is observable.

        `const` is computed as `ub - sum(dm * y_inc)`, so asserting the cut is tight
        at its own candidate re-derives an identity the code just imposed. It catches
        an index or aggregation slip and nothing else. It is NOT a duality check,
        despite reading like one.

        The property that does depend on the dual being right is the defining one:
        a Benders optimality cut must UNDERESTIMATE the recourse everywhere. The
        `OptFace` row pins the selected dual to `dual_obj >= ub - tol`; the other
        side, `dual_obj <= ub`, is weak duality, and it only holds if the dual
        feasible region is the true dual of the primal. A region that is too large
        breaks it -- which is exactly how the stale `min(S, C[tau])` in the dual
        objective showed up, as `primal=5105.0 dual=5825.0`.

        Neighbouring schedules are the sharp place to look. Far away the cut has
        thousands of units of slack and would absorb a badly wrong slope; one slot
        away it is nearly tight. Measured here the minimum slack over the
        perturbations is 0.000000 at `yOUT[0,0]` -- exactly tight, so a wrong slope
        has nothing to hide in. `test_the_underestimation_check_is_sharp` asserts
        that property directly, because a version of this test with slack everywhere
        would pass while checking very little.
        """
        for label in ("mw", "dual"):
            worst, where = self._worst_slack(label)
            with self.subTest(generator=label):
                self.assertGreaterEqual(
                    worst,
                    -1e-6 * max(1.0, abs(worst)),
                    msg=(
                        f"the {label} cut OVERESTIMATES the recourse at {where} by "
                        f"{-worst!r}. It is not a valid lower bound, so no bound "
                        "built from it is one either."
                    ),
                )

    def test_the_underestimation_check_is_sharp(self):
        """The test above is only meaningful if some perturbation is nearly tight.

        With slack everywhere it would pass for a dual that is wrong by less than
        the slack. Assert that at least one neighbour leaves the cut within a unit
        of the true recourse.
        """
        for label in ("mw", "dual"):
            worst, _ = self._worst_slack(label)
            with self.subTest(generator=label):
                self.assertLess(
                    worst,
                    1.0,
                    msg=(
                        f"no perturbation brought the {label} cut close to the "
                        f"recourse (min slack {worst!r}); the validity assertion "
                        "has too much room to be evidence."
                    ),
                )

    def test_mw_is_strictly_better_on_at_least_one_core_point(self):
        """Guards the test above against passing vacuously.

        If MW silently degraded to returning whatever dual the solver handed back,
        every margin would be exactly zero and the dominance assertion would still
        pass while checking nothing. Measured on this fixture, by core point:
        all_zeros 0, ret_only 0, all_ones ~2e-4, uniform ~21, out_only ~30. Two
        exact zeros are expected -- at those core points the MW objective does not
        separate the face -- so the guard is that a strict win exists somewhere,
        not at every point.
        """
        margins = [
            self._value_at(mw_md, core[0], core[1])
            - self._value_at(self.dual_md, core[0], core[1])
            for (mw_md, _), core in self.mw.values()
        ]
        self.assertGreater(
            max(margins),
            1e-3,
            msg=(
                "MW never beat the plain dual at any core point. Either MW is not "
                f"selecting at all, or the LP is not degenerate here. margins={margins}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
