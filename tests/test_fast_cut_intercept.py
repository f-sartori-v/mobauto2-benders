"""S1 and S2: the cut intercept is derived from the duals, and the fallback is valid.

No solver. Every assertion here is about the arithmetic that turns a dual solution
into a cut, which is exactly the arithmetic that had no detector before:

  S1  The Magnanti-Wong fallback used to switch to finite differences, whose slopes
      carry no lower-bound guarantee, and set cut_valid_lower_bound = False -- which
      makes solver.py drop best_lb for the WHOLE run. It now falls back to the plain
      capacity duals, which are valid (handout 77), already computed, and were being
      discarded.

  S2  The intercept used to be imposed as `Q(y) - sum(dm * y_inc)`, so the tightness
      property the formal formulation checks in 20.1 could not fail: the code wrote
      the identity it then verified. It is now derived as `sum_t alpha[t] * R[t]` and
      checked against the imposed form, which makes 20.3 (strong duality) live.
"""

from __future__ import annotations

import unittest

import _helpers  # noqa: F401  -- puts src/ on sys.path

from mobauto2_benders.problem.subproblem_impl import (
    CUT_MODE_VALID_LOWER_BOUND,
    MWDual,
    cut_mode_carries_lower_bound,
    derive_cut_intercepts,
    expand_slopes_to_candidate,
    slopes_from_capacity_duals,
)


SEATS = 15.0
T = 4


def _consistent_dual() -> tuple[MWDual, dict]:
    """A dual solution and the primal data it is consistent with.

    Built so strong duality holds exactly by construction: pick alpha and pi, then
    define the directional objectives as the dual objective they imply. That is the
    right direction for a fixture -- inventing an objective and hoping the duals
    match it would test nothing but the fixture author's arithmetic.
    """
    R_out = [10.0, 0.0, 6.0, 0.0]
    R_ret = [0.0, 4.0, 0.0, 8.0]
    # pi <= 0 for a '<=' row in a minimisation.
    pi_out = {0: 0.0, 1: -2.0, 2: 0.0, 3: -1.0}
    pi_ret = {0: 0.0, 1: 0.0, 2: -3.0, 3: 0.0}
    alpha_out = {0: 5.0, 1: 0.0, 2: 4.0, 3: 0.0}
    alpha_ret = {0: 0.0, 1: 2.0, 2: 0.0, 3: 7.0}
    # Incumbent capacity, in vehicles: y_inc = C / S.
    y_inc_out = [1.0, 2.0, 0.0, 1.0]
    y_inc_ret = [0.0, 1.0, 2.0, 1.0]

    mw = MWDual(
        dm_out={t: SEATS * pi_out[t] for t in range(T)},
        dm_ret={t: SEATS * pi_ret[t] for t in range(T)},
        alpha_out=alpha_out,
        alpha_ret=alpha_ret,
    )
    # Q_d = sum_t alpha_d R_d + sum_tau C_d[tau] pi_d[tau], with C = S * y_inc.
    ub_out = sum(alpha_out[t] * R_out[t] for t in range(T)) + sum(
        SEATS * y_inc_out[t] * pi_out[t] for t in range(T)
    )
    ub_ret = sum(alpha_ret[t] * R_ret[t] for t in range(T)) + sum(
        SEATS * y_inc_ret[t] * pi_ret[t] for t in range(T)
    )
    data = dict(
        R_out=R_out,
        R_ret=R_ret,
        y_inc_out=y_inc_out,
        y_inc_ret=y_inc_ret,
        ub_out=ub_out,
        ub_ret=ub_ret,
        T=T,
        eps_dual=1e-9,
    )
    return mw, data


class InterceptComesFromAlpha(unittest.TestCase):
    def test_it_equals_sum_alpha_times_demand(self):
        mw, data = _consistent_dual()
        a_out, a_ret, diag = derive_cut_intercepts(mw, **data)
        self.assertAlmostEqual(
            a_out, sum(mw.alpha_out[t] * data["R_out"][t] for t in range(T)), places=9
        )
        self.assertAlmostEqual(
            a_ret, sum(mw.alpha_ret[t] * data["R_ret"][t] for t in range(T)), places=9
        )
        # The diagnostics must carry both forms so a run can be audited after the
        # fact, not only at the moment the check passes.
        self.assertAlmostEqual(diag["intercept_out_from_alpha"], a_out, places=9)
        self.assertAlmostEqual(diag["intercept_out_imposed"], a_out, places=9)
        self.assertAlmostEqual(diag["intercept_gap_out"], 0.0, places=9)

    def test_the_cut_is_tight_at_the_incumbent(self):
        """Formulation 20.1, but now as a consequence rather than an identity.

        cut(y_inc) = a_d + sum_tau dm_d[tau] * y_inc_d[tau] must equal Q_d(y_inc).
        Nothing in derive_cut_intercepts forces this -- a_d comes from alpha and the
        slopes come from pi, so the equality holds only if the dual really is the
        dual. That is the whole point of the change.
        """
        mw, data = _consistent_dual()
        a_out, a_ret, _ = derive_cut_intercepts(mw, **data)
        cut_out = a_out + sum(
            mw.dm_out[t] * data["y_inc_out"][t] for t in range(T)
        )
        cut_ret = a_ret + sum(
            mw.dm_ret[t] * data["y_inc_ret"][t] for t in range(T)
        )
        self.assertAlmostEqual(cut_out, data["ub_out"], places=9)
        self.assertAlmostEqual(cut_ret, data["ub_ret"], places=9)

    def test_a_broken_dual_raises_rather_than_producing_a_cut(self):
        """The detector S2 exists to install. Perturb alpha; strong duality must fail.

        Before S2 this perturbation was undetectable: the intercept was computed from
        ub_out, so alpha could be anything at all -- including the wrong sign or a
        stale scenario's values -- and the cut would still pass the tightness check.
        """
        mw, data = _consistent_dual()
        broken = MWDual(
            dm_out=mw.dm_out,
            dm_ret=mw.dm_ret,
            alpha_out={**mw.alpha_out, 0: mw.alpha_out[0] + 1.0},
            alpha_ret=mw.alpha_ret,
        )
        with self.assertRaises(RuntimeError) as ctx:
            derive_cut_intercepts(broken, **data)
        self.assertIn("strong duality", str(ctx.exception).lower())

    def test_the_check_is_sharp(self):
        """It must not fire on noise, or it will be loosened until it fires on nothing.

        Companion to test_the_underestimation_check_is_sharp: a guard that cries wolf
        gets disabled, so the tolerance has to admit a perturbation just inside it.
        """
        mw, data = _consistent_dual()
        data = dict(data, eps_dual=1e-6)
        # R_out[0] = 10, so a 1e-8 nudge to alpha moves the intercept by 1e-7 < eps.
        nudged = MWDual(
            dm_out=mw.dm_out,
            dm_ret=mw.dm_ret,
            alpha_out={**mw.alpha_out, 0: mw.alpha_out[0] + 1e-8},
            alpha_ret=mw.alpha_ret,
        )
        _a_out, _a_ret, diag = derive_cut_intercepts(nudged, **data)
        # Inside the band, and the band is what admitted it.
        self.assertLess(abs(diag["intercept_gap_out"]), data["eps_dual"])
        self.assertAlmostEqual(abs(diag["intercept_gap_out"]), 1e-7, places=12)


class PlainDualFallbackIsValid(unittest.TestCase):
    def test_slopes_are_seats_times_pi_and_alpha_is_carried(self):
        duals = {
            "pi_OUT": {0: -1.0, 1: 0.0, 2: -2.0, 3: 0.0},
            "pi_RET": {0: 0.0, 1: -3.0, 2: 0.0, 3: 0.0},
            "alpha_OUT": {0: 4.0, 1: 4.0, 2: 1.0, 3: 0.0},
            "alpha_RET": {0: 0.0, 1: 6.0, 2: 0.0, 3: 2.0},
        }
        mw = slopes_from_capacity_duals(duals, SEATS, T)
        self.assertEqual(mw.dm_out[0], SEATS * -1.0)
        self.assertEqual(mw.dm_out[2], SEATS * -2.0)
        self.assertEqual(mw.dm_ret[1], SEATS * -3.0)
        self.assertEqual(mw.alpha_out[0], 4.0)
        self.assertEqual(mw.alpha_ret[3], 2.0)

    def test_every_slope_is_non_positive(self):
        """Formulation 20.2. A positive slope means more seats cost more."""
        duals = {
            "pi_OUT": {t: -float(t) for t in range(T)},
            "pi_RET": {t: 0.0 for t in range(T)},
            "alpha_OUT": {t: 1.0 for t in range(T)},
            "alpha_RET": {t: 1.0 for t in range(T)},
        }
        mw = slopes_from_capacity_duals(duals, SEATS, T)
        for t in range(T):
            self.assertLessEqual(mw.dm_out[t], 0.0)
            self.assertLessEqual(mw.dm_ret[t], 0.0)

    def test_a_missing_dual_reads_as_zero_not_as_an_error(self):
        """A tau with no capacity row has slope exactly 0 -- arithmetic, not a guess."""
        mw = slopes_from_capacity_duals({"pi_OUT": {1: -2.0}}, SEATS, T)
        self.assertEqual(mw.dm_out[0], 0.0)
        self.assertEqual(mw.dm_out[1], SEATS * -2.0)
        self.assertEqual(mw.dm_ret[3], 0.0)


class ModeValidityIsOneTable(unittest.TestCase):
    def test_the_mw_fallback_now_carries_a_lower_bound(self):
        """S1, stated as the contract that changed.

        The old fallback was `mw_fdiff_fallback` and it was NOT a valid lower bound,
        so a single MW failure voided best_lb for the run.
        """
        self.assertTrue(cut_mode_carries_lower_bound("mw_dual_fallback"))
        self.assertNotIn("mw_fdiff_fallback", CUT_MODE_VALID_LOWER_BOUND)

    def test_finite_difference_still_does_not(self):
        """Handout 75/76. A heuristic value bounds Q(x) from above, so it is not a cut."""
        self.assertFalse(cut_mode_carries_lower_bound("finite_difference"))

    def test_mw_and_plain_dual_both_do(self):
        self.assertTrue(cut_mode_carries_lower_bound("mw"))
        self.assertTrue(cut_mode_carries_lower_bound("dual"))

    def test_an_unrecorded_mode_raises_rather_than_defaulting(self):
        """Neither True nor False is a safe answer to "nobody decided"."""
        with self.assertRaises(RuntimeError) as ctx:
            cut_mode_carries_lower_bound("some_new_generator")
        self.assertIn("lower-bound guarantee", str(ctx.exception))

    def test_every_mode_the_dispatch_can_emit_is_in_the_table(self):
        """Guards against a branch label drifting away from the table.

        Read out of the source rather than restated, so adding a branch without a
        table entry fails here instead of at the first run that takes the branch.
        """
        import re
        from pathlib import Path

        src = (
            Path(_helpers.REPO_ROOT)
            / "src"
            / "mobauto2_benders"
            / "problem"
            / "subproblem_impl.py"
        ).read_text(encoding="utf-8")
        # Every quoted literal on a `cut_mode_used = ...` line, so the ternary
        # `"dual" if use_dual else "finite_difference"` contributes both of its labels
        # rather than only the first.
        emitted: set[str] = set()
        for line in src.splitlines():
            if re.search(r"\bcut_mode_used\s*=", line):
                emitted |= set(re.findall(r'"([a-z_]+)"', line))
        self.assertTrue(emitted, "no cut_mode_used assignments found -- regex stale")
        # Exact, not one-way: an untabled label would ship a mode with no recorded
        # guarantee, and a tabled label nothing emits is a mode that was removed
        # without removing its entry, which is how a stale claim survives in docs.
        self.assertEqual(
            emitted,
            set(CUT_MODE_VALID_LOWER_BOUND),
            "the dispatch labels and the validity table have drifted apart",
        )


class SlopesBroadcastIdenticallyAcrossVehicles(unittest.TestCase):
    def test_every_vehicle_at_a_slot_gets_the_same_coefficient(self):
        """E1/D48: the recourse sees only Y[tau] = sum_q y[q,tau].

        This identity is what makes the master's aggregate_cuts_by_tau a collapse
        rather than an approximation; _assert_q_invariant enforces the other side.
        """
        candidate = {
            f"yOUT[{q},{t}]": 0.0 for q in range(3) for t in range(T)
        } | {f"yRET[{q},{t}]": 0.0 for q in range(3) for t in range(T)}
        dm_out = {0: -15.0, 1: 0.0, 2: -30.0, 3: 0.0}
        dm_ret = {0: 0.0, 1: -45.0, 2: 0.0, 3: 0.0}
        c_out, c_ret = expand_slopes_to_candidate(candidate, dm_out, dm_ret, T)
        for t in range(T):
            vals = {c_out[(q, t)] for q in range(3)}
            self.assertEqual(len(vals), 1, f"OUT slope varies across q at tau={t}")
            self.assertEqual(vals.pop(), dm_out[t])
            vals = {c_ret[(q, t)] for q in range(3)}
            self.assertEqual(len(vals), 1, f"RET slope varies across q at tau={t}")
            self.assertEqual(vals.pop(), dm_ret[t])

    def test_slots_outside_the_horizon_are_dropped(self):
        candidate = {"yOUT[0,0]": 0.0, "yOUT[0,99]": 0.0}
        c_out, _ = expand_slopes_to_candidate(candidate, {0: -1.0}, {}, T)
        self.assertIn((0, 0), c_out)
        self.assertNotIn((0, 99), c_out)

    def test_non_departure_keys_are_ignored(self):
        candidate = {"yOUT[0,0]": 0.0, "__theta": 12.0, "b[0,3]": 150.0}
        c_out, c_ret = expand_slopes_to_candidate(candidate, {0: -1.0}, {}, T)
        self.assertEqual(set(c_out), {(0, 0)})
        self.assertEqual(c_ret, {})


if __name__ == "__main__":
    unittest.main()
