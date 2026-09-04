"""Exactness conditions E1 and E2 (DESIGN_DD_v1, D48). REQUIRES AN LP BACKEND.

E1 -- the recourse depends on `y` only through the signature `Y_d[tau] = sum_q y_d[q,tau]`.
E2 -- `y` enters the subproblem in the right-hand side only.

Both are true today. E2 is D30 restated as a standing condition rather than a historical
fix: when the subproblem's constraint SET moved with `y`, no cut generator on top of it
could be valid, and the defect went six months unseen while every run reported healthy
provenance.

WHAT B4 CHANGED, AND WHY (audit item 1.5). This file used to assert, under E1, that two
schedules sharing a signature return the same DUAL VECTOR. That requirement is wrong and
has been deleted. A degenerate LP has several optimal dual solutions; each is dual
feasible, so each yields a VALID Benders cut, and which one comes back is a fact about
the solver's pivoting rather than about the model. Asserting equality there tested the
solver and would have sent someone hunting a formulation bug on a correct
implementation. Dual agreement is now LOGGED as a diagnostic -- it is genuinely useful,
because a run whose cut depends on pivoting is not bit-reproducible -- and never
asserted.

What replaced it is what E1 actually requires: two schedules with the same signature
build a byte-identical LP. Identical recourse matrix fingerprint, identical variable and
constraint sets, identical objective coefficients, identical non-capacity right-hand
sides, and identical optimal VALUE. See `mobauto2_benders/recourse_fingerprint.py`.

TWO CONDITIONS THAT ARE NOT HERE ANY MORE, AND WHY. E3 (separability by vehicle) and E4
(fleet homogeneity) were listed as "conditions for exactness". Neither is required for
Benders validity, and calling them that misstated what a failure of either would mean:

  * E3 is a COMPUTATIONAL property. It licenses the stage-3 Dantzig-Wolfe
    reformulation. If it fails, that reformulation is unavailable; the cuts stay
    valid and the decomposition stays exact. Renamed M1 -- master equivalence with
    the monolith -- and tested in tests/test_fast_signature.py.
  * E4 is the validity condition for the vehicle-ordering symmetry constraint ALONE.
    Without a homogeneous fleet that one inequality can cut off the optimum; nothing
    else in the model cares. Renamed M2 -- symmetry validity -- and tested in
    tests/test_fast_model.py, which also refuses the constraint at load.

Run just these:
    python -m unittest tests.test_signature_exactness -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

import _helpers  # noqa: F401  (puts src/ on sys.path)
from _helpers import require_solver_backend

from mobauto2_benders.signature import candidate_signature

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "soundness.yaml"


def _setup():
    from mobauto2_benders.config import load_config
    from mobauto2_benders.app import _prepare_params

    cfg = load_config(str(FIXTURE))
    _helpers.repoint_solvers(cfg)
    mp, sp = _prepare_params(cfg, {})
    T = int(mp.get("T") or (int(mp["T_minutes"]) // int(mp["slot_resolution"])))
    sp["T"] = T
    Q = int(mp.get("Q", 2))
    return sp, T, Q


def _evaluate(sp_params: dict, cand: dict):
    from mobauto2_benders.problem.subproblem_impl import ProblemSubproblem

    params = dict(sp_params)
    params["use_magnanti_wong"] = True
    params["use_dual_slopes"] = True
    # A fixed core point, so Magnanti-Wong's selection is a deterministic function of
    # the LP. Without pinning it, two identical LPs could still be compared against
    # different directions and E1 would be testing nothing.
    T = int(params["T"])
    params["mw_core_point"] = {"Yout": [0.5] * T, "Yret": [0.3] * T}
    res = ProblemSubproblem(params).evaluate(dict(cand))
    cut = (res.cuts or [res.cut])[0]
    return res, cut.metadata


def _build_model(sp_params: dict, cand: dict):
    """Build (do not solve) the slot recourse LP for one candidate.

    Reads the candidate exactly as `ProblemSubproblem.evaluate` does -- through the
    signature and nothing else -- so what is fingerprinted is the LP the production
    path would have solved.
    """
    from mobauto2_benders.recourse_fingerprint import slot_recourse_model_for
    from mobauto2_benders.problem.subproblem_impl import (
        aggregate_requests,
        load_demand_doc,
        wmax_minutes_to_slots,
    )

    params = dict(sp_params)
    T = int(params["T"])
    slot_res = int(params.get("slot_resolution", 1))
    if "Wmax_slots" not in params or params.get("Wmax_slots") is None:
        params["Wmax_slots"] = wmax_minutes_to_slots(
            float(params["Wmax_minutes"]), slot_res
        )
    R_out, R_ret = aggregate_requests(
        load_demand_doc(Path(str(params["demand_file"]))), T, slot_res
    )
    Y_out, Y_ret = candidate_signature(cand, T)
    return slot_recourse_model_for(params, Y_out, Y_ret, R_out, R_ret)


def _slopes_by_tau(meta: dict, key: str) -> dict[int, float]:
    """Collapse the cut's (q,tau) coefficients to tau.

    Valid because the production path broadcasts one per-slot dual to every q --
    the same identity `_assert_q_invariant` enforces in the master.
    """
    out: dict[int, float] = {}
    for (_q, tau), v in dict(meta.get(key) or {}).items():
        out[int(tau)] = float(v)
    return out


class TestE1RecourseDependsOnlyOnTheSignature(unittest.TestCase):
    """Same signature, different per-vehicle assignment => identical LP."""

    @classmethod
    def setUpClass(cls):
        require_solver_backend()
        cls.sp, cls.T, cls.Q = _setup()
        if cls.Q < 2:
            raise unittest.SkipTest("E1 needs at least two vehicles to permute")

        T, Q = cls.T, cls.Q

        # Two schedules with the SAME signature and a genuinely different assignment.
        # `a` sends vehicle 0 on the even departure slots and vehicle 1 on the odd
        # ones; `b` swaps them slot by slot. This is deliberately NOT a global
        # relabelling of the fleet -- a global permutation would give an identical LP
        # for a reason weaker than the one under test.
        cls.cand_a, cls.cand_b = {}, {}
        for q in range(Q):
            for t in range(T):
                cls.cand_a[f"yOUT[{q},{t}]"] = 0.0
                cls.cand_a[f"yRET[{q},{t}]"] = 0.0
                cls.cand_b[f"yOUT[{q},{t}]"] = 0.0
                cls.cand_b[f"yRET[{q},{t}]"] = 0.0
        for t in range(1, T - 1):
            first, second = (0, 1) if (t % 2 == 0) else (1, 0)
            if t % 3 == 0:
                cls.cand_a[f"yOUT[{first},{t}]"] = 1.0
                cls.cand_b[f"yOUT[{second},{t}]"] = 1.0
            elif t % 3 == 1:
                cls.cand_a[f"yRET[{first},{t}]"] = 1.0
                cls.cand_b[f"yRET[{second},{t}]"] = 1.0

        cls.res_a, cls.meta_a = _evaluate(cls.sp, cls.cand_a)
        cls.res_b, cls.meta_b = _evaluate(cls.sp, cls.cand_b)

    def _model_for(self, cand: dict):
        """The recourse LP this candidate produces, built but never solved."""
        return _build_model(self.sp, cand)

    def test_the_two_candidates_really_do_differ(self):
        """Guard against a test that compares a schedule with itself."""
        differing = [k for k in self.cand_a if self.cand_a[k] != self.cand_b[k]]
        self.assertTrue(
            differing,
            "the two candidates are identical, so E1 is being asserted vacuously",
        )
        self.assertGreaterEqual(len(differing), 4)

    def test_the_two_candidates_share_a_signature(self):
        self.assertEqual(
            candidate_signature(self.cand_a, self.T),
            candidate_signature(self.cand_b, self.T),
        )

    def test_recourse_values_agree_to_the_cent(self):
        self.assertIsNotNone(self.res_a.upper_bound)
        self.assertAlmostEqual(
            float(self.res_a.upper_bound), float(self.res_b.upper_bound), places=6
        )

    def test_fibre_value_invariance(self):
        """B4. The fibre test, asserting what E1 actually requires.

        Two schedules with the same signature must build the SAME LP: same
        fingerprint, same variables, same rows, same objective coefficients, same
        non-capacity right-hand sides -- and therefore the same optimal value. The
        capacity right-hand side is deliberately excluded from the fingerprint and
        checked separately below: it is the one channel `y` is allowed to use, and
        excluding it is what makes the fingerprint an assertion rather than a
        tautology.

        This replaces the old dual-equality assertion. See the module docstring.
        """
        from mobauto2_benders.recourse_fingerprint import (
            capacity_rhs,
            recourse_fingerprint,
            structural_description,
        )

        m_a = self._model_for(self.cand_a)
        m_b = self._model_for(self.cand_b)

        desc_a = structural_description(m_a)
        desc_b = structural_description(m_b)

        self.assertEqual(
            desc_a["variables"],
            desc_b["variables"],
            "the two members of the fibre built different variable sets",
        )
        self.assertEqual(
            sorted(desc_a["rows"]),
            sorted(desc_b["rows"]),
            "the two members of the fibre built different constraint sets -- `y` is "
            "entering the constraint MATRIX, which is exactly the D30 failure",
        )
        self.assertEqual(
            desc_a["objective"],
            desc_b["objective"],
            "the objective coefficients moved with the schedule",
        )
        for name in desc_a["rows"]:
            with self.subTest(row=name):
                self.assertEqual(
                    desc_a["rows"][name],
                    desc_b["rows"][name],
                    f"row {name} differs between two schedules with one signature",
                )
        self.assertEqual(
            recourse_fingerprint(m_a),
            recourse_fingerprint(m_b),
            "recourse matrix fingerprints differ across the fibre",
        )
        # The half that IS allowed to move -- and here must not, because these two
        # schedules share a signature and the capacity vector is S times it.
        self.assertEqual(capacity_rhs(m_a), capacity_rhs(m_b))
        # And the value, which is the observable the cut is built from.
        self.assertAlmostEqual(
            float(self.res_a.upper_bound), float(self.res_b.upper_bound), places=6
        )

    def test_dual_agreement_is_reported_but_not_required(self):
        """B4. The demoted assertion, kept as a diagnostic.

        Degenerate LPs have several optimal duals and every one gives a valid cut, so
        a difference here is NOT a defect. It is worth knowing about: a run whose cut
        depends on which optimal dual the solver landed on is not bit-reproducible,
        and the manifest's `bit_reproducible` field is the place that matters. So the
        comparison still runs and still prints; it just cannot fail the suite.
        """
        for key in ("coeff_yOUT", "coeff_yRET"):
            a = _slopes_by_tau(self.meta_a, key)
            b = _slopes_by_tau(self.meta_b, key)
            differing = sorted(
                tau
                for tau in set(a) | set(b)
                if abs(a.get(tau, 0.0) - b.get(tau, 0.0)) > 1e-9
            )
            if differing:
                print(
                    f"[E1 DIAGNOSTIC] {key}: two members of the fibre returned "
                    f"different optimal duals at tau={differing}. Both cuts are "
                    "valid; the run is not bit-reproducible in its cut coefficients."
                )
        # Asserting nothing on purpose. The assertion this replaced was wrong.
        self.assertTrue(True)

    def test_the_cut_constants_agree(self):
        self.assertAlmostEqual(
            float(self.meta_a["const"]), float(self.meta_b["const"]), places=6
        )

    def test_the_slopes_are_not_all_zero(self):
        """An all-zero slope vector would make the dual comparison vacuous."""
        a = _slopes_by_tau(self.meta_a, "coeff_yOUT")
        self.assertTrue(
            any(abs(v) > 1e-9 for v in a.values()),
            "every OUT slope is zero, so E1's dual comparison proves nothing",
        )


class TestE2YEntersTheRightHandSideOnly(unittest.TestCase):
    """The D30 condition, as a standing check rather than a fixed bug."""

    @classmethod
    def setUpClass(cls):
        require_solver_backend()
        cls.sp, cls.T, cls.Q = _setup()
        T, Q = cls.T, cls.Q

        def sched(period: int) -> dict:
            c = {}
            for q in range(Q):
                for t in range(T):
                    c[f"yOUT[{q},{t}]"] = (
                        1.0 if (period and 1 <= t < T - 1 and t % period == 0) else 0.0
                    )
                    c[f"yRET[{q},{t}]"] = (
                        1.0 if (period and 1 <= t < T - 1 and t % period == 1) else 0.0
                    )
            return c

        # An ascending chain of signatures: every slot served by `sched(2)` is also
        # served by `sched(1)`, and `sched(0)` is the all-idle schedule.
        cls.levels = [sched(0), sched(4), sched(2)]
        cls.results = [_evaluate(cls.sp, c) for c in cls.levels]

    def test_the_chain_is_actually_ascending(self):
        sigs = [candidate_signature(c, self.T) for c in self.levels]
        for lo, hi in zip(sigs, sigs[1:]):
            self.assertTrue(
                all(a <= b for a, b in zip(lo[0], hi[0]))
                and all(a <= b for a, b in zip(lo[1], hi[1])),
                "the test schedules are not componentwise ordered, so monotonicity "
                "is not being tested",
            )
            self.assertTrue(
                any(a < b for a, b in zip(lo[0] + lo[1], hi[0] + hi[1])),
                "two consecutive levels are equal; the chain is not strict",
            )

    def test_recourse_is_non_increasing_in_the_signature(self):
        """P3. More capacity can only lower the cost -- if `y` is in the RHS alone.

        This is a direct probe of the D30 failure. When the constraint set moved with
        `y`, the dual of one instance was not a subgradient of the recourse across `y`
        and the cut slopes came out roughly K times too steep. A monotonicity sweep
        over real solves sees that without needing to inspect the model.
        """
        ubs = [float(r.upper_bound) for r, _ in self.results]
        for lo, hi in zip(ubs, ubs[1:]):
            self.assertLessEqual(
                hi, lo + 1e-6, f"recourse rose with capacity: {lo} -> {hi}"
            )
        self.assertLess(
            ubs[-1], ubs[0], "adding capacity changed nothing; the sweep is vacuous"
        )

    def test_the_dual_row_set_does_not_move_with_y(self):
        """Which capacity rows exist must be a function of (T, Wmax), not of `y`.

        A row that appears or vanishes with the schedule is exactly `y` entering the
        constraint matrix.
        """
        keysets = [set(_slopes_by_tau(m, "coeff_yOUT")) for _, m in self.results]
        for a, b in zip(keysets, keysets[1:]):
            self.assertEqual(a, b, "the OUT slope index set changed with the schedule")

    def test_a_cut_underestimates_the_recourse_lower_down_the_chain(self):
        """The direction DESIGN_DD_v1 stage 2 will rely on.

        D43 checks underestimation at single-slot neighbours. The down-set cut needs
        it in the decreasing direction specifically, so it is checked here before
        anything is built on it.
        """
        _, meta_hi = self.results[-1]
        const = float(meta_hi["const"])
        slopes_out = _slopes_by_tau(meta_hi, "coeff_yOUT")
        slopes_ret = _slopes_by_tau(meta_hi, "coeff_yRET")

        for level_idx in range(len(self.levels) - 1):
            cand = self.levels[level_idx]
            Y_out, Y_ret = candidate_signature(cand, self.T)
            cut_val = (
                const
                + sum(v * Y_out[tau] for tau, v in slopes_out.items())
                + sum(v * Y_ret[tau] for tau, v in slopes_ret.items())
            )
            true_recourse = float(self.results[level_idx][0].upper_bound)
            with self.subTest(level=level_idx):
                self.assertLessEqual(
                    cut_val,
                    true_recourse + 1e-6,
                    f"the cut OVERESTIMATES the recourse at a lower signature "
                    f"(cut={cut_val:.6f} > true={true_recourse:.6f}). A cut that does "
                    "this excludes feasible schedules and can cut off the optimum.",
                )


if __name__ == "__main__":
    unittest.main()
