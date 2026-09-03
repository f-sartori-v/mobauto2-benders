"""B2 (handout item): Magnanti-Wong's Pareto-optimal dual, ported to the minute
recourse. REQUIRES AN LP BACKEND; a couple of seconds.

Mirrors the slot path's own cut-soundness invariants (S2/S3, D63/D65,
tests/test_fast_cut_intercept.py) on the minute recourse instead: the intercept
is derived from the demand duals and reconciled against the imposed form, not
imposed outright (S2); and the resulting cut never overestimates the true
recourse, checked at points other than its own anchor, not merely tight there.
"""

from __future__ import annotations

import unittest

import _helpers  # noqa: F401  (puts src/ on sys.path)
from _helpers import require_solver_backend
from mobauto2_benders.minute_pricer import solve_minute_recourse, solve_mw_dual_minute
from mobauto2_benders.problem.subproblem_impl import MWDual, derive_cut_intercepts

T = 8
DELTA = 30
WMAX_MINUTES = 60
P_SLOTS = 56.0 / 30.0
SEATS = 15.0
REQUESTS = {"OUT": [10, 35, 50, 90, 150, 200], "RET": [40, 70, 100, 160]}

# Two candidate schedules on the same instance -- the incumbent MW is anchored
# at, and a second one to check the cut against a point it was NOT built for.
INCUMBENT_TAUS = {"OUT": [1, 2, 4, 6], "RET": [2, 3, 5, 7]}
OTHER_TAUS = {"OUT": [2, 4, 5, 7], "RET": [1, 3, 6, 7]}


def _caps(taus_by_dir: dict[str, list[int]], seats: float = SEATS) -> tuple[list[float], list[float]]:
    c_out = [0.0] * T
    c_ret = [0.0] * T
    for t in taus_by_dir["OUT"]:
        c_out[t] += seats
    for t in taus_by_dir["RET"]:
        c_ret[t] += seats
    return c_out, c_ret


def _y_inc(taus_by_dir: dict[str, list[int]]) -> tuple[list[float], list[float]]:
    c_out, c_ret = _caps(taus_by_dir, seats=1.0)
    return c_out, c_ret


class MagnantiWongSucceedsOnTheMinuteRecourse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = require_solver_backend()

    def test_returns_a_solution_not_none(self):
        """The port's first job: it must actually run, not silently decline every
        time the way the pre-fix slot path did (C3's failure mode)."""
        c_out, c_ret = _caps(INCUMBENT_TAUS)
        duals, ub = solve_minute_recourse(
            T, DELTA, WMAX_MINUTES, P_SLOTS, c_out, c_ret, REQUESTS,
            policy="start", lp_solver=self.backend,
        )
        # A core point distinct from the incumbent, so MW has a real direction to
        # select in (an all-zero or incumbent-identical core point degenerates the
        # selection, as the slot path's own [MW CORE] seeding guards against).
        ybar_out = [0.0, 0.6, 1.0, 0.0, 0.5, 0.0, 0.9, 0.0]
        ybar_ret = [0.0, 0.0, 0.7, 1.0, 0.0, 0.4, 0.0, 1.0]
        mw = solve_mw_dual_minute(
            T, DELTA, WMAX_MINUTES, P_SLOTS, SEATS, c_out, c_ret, REQUESTS,
            ybar_out, ybar_ret, ub, policy="start", lp_solver=self.backend,
        )
        self.assertIsNotNone(mw, "solve_mw_dual_minute returned None -- MW declined")
        # Sign convention (D65): pi <= 0 for a '<=' capacity row in a minimisation,
        # so dm = seats*pi <= 0 -- a cut whose slopes are all <= 0.
        for t in range(T):
            self.assertLessEqual(mw.dm_out[t], 1e-9)
            self.assertLessEqual(mw.dm_ret[t], 1e-9)

    def test_intercept_is_derived_and_reconciles_not_imposed(self):
        """S2, on the minute recourse: a_d = sum_m alpha_d[m] * pool_d[m] must equal
        the imposed form ub_d - sum dm_d[tau]*y_inc_d[tau] to derive_cut_intercepts'
        own tolerance -- i.e. the port satisfies the SAME strong-duality check the
        slot path is held to, unmodified."""
        c_out, c_ret = _caps(INCUMBENT_TAUS)
        y_inc_out, y_inc_ret = _y_inc(INCUMBENT_TAUS)
        duals, ub = solve_minute_recourse(
            T, DELTA, WMAX_MINUTES, P_SLOTS, c_out, c_ret, REQUESTS,
            policy="start", lp_solver=self.backend,
        )
        ybar_out = [0.0, 0.6, 1.0, 0.0, 0.5, 0.0, 0.9, 0.0]
        ybar_ret = [0.0, 0.0, 0.7, 1.0, 0.0, 0.4, 0.0, 1.0]
        mw = solve_mw_dual_minute(
            T, DELTA, WMAX_MINUTES, P_SLOTS, SEATS, c_out, c_ret, REQUESTS,
            ybar_out, ybar_ret, ub, policy="start", lp_solver=self.backend,
        )
        self.assertIsNotNone(mw)
        mw_dual = MWDual(
            dm_out=mw.dm_out, dm_ret=mw.dm_ret,
            intercept_out=mw.intercept_out, intercept_ret=mw.intercept_ret,
            alpha_out=mw.alpha_out, alpha_ret=mw.alpha_ret,
        )
        ub_out = float(duals["ub_out"])
        ub_ret = float(duals["ub_ret"])
        # Must not raise (derive_cut_intercepts raises RuntimeError on a strong-
        # duality mismatch beyond eps_dual -- unchanged, generic code, run here on
        # a minute-recourse dual for the first time).
        a_out, a_ret, diag = derive_cut_intercepts(
            mw_dual, y_inc_out, y_inc_ret, ub_out, ub_ret, T, eps_dual=1e-4
        )
        self.assertAlmostEqual(diag["intercept_gap_out"], 0.0, places=3)
        self.assertAlmostEqual(diag["intercept_gap_ret"], 0.0, places=3)

    def test_the_cut_never_overestimates_at_a_point_other_than_its_own_anchor(self):
        """The property that matters (D30's exact failure mode is an overestimate).
        Tight-at-the-anchor is necessary but not sufficient -- a cut can be exactly
        right at y_inc and still overestimate everywhere else if the slopes are
        wrong. Check it at a genuinely different schedule instead."""
        c_out_inc, c_ret_inc = _caps(INCUMBENT_TAUS)
        y_inc_out, y_inc_ret = _y_inc(INCUMBENT_TAUS)
        duals_inc, ub_inc = solve_minute_recourse(
            T, DELTA, WMAX_MINUTES, P_SLOTS, c_out_inc, c_ret_inc, REQUESTS,
            policy="start", lp_solver=self.backend,
        )
        ybar_out = [0.0, 0.6, 1.0, 0.0, 0.5, 0.0, 0.9, 0.0]
        ybar_ret = [0.0, 0.0, 0.7, 1.0, 0.0, 0.4, 0.0, 1.0]
        mw = solve_mw_dual_minute(
            T, DELTA, WMAX_MINUTES, P_SLOTS, SEATS, c_out_inc, c_ret_inc, REQUESTS,
            ybar_out, ybar_ret, ub_inc, policy="start", lp_solver=self.backend,
        )
        self.assertIsNotNone(mw)
        a_out = mw.intercept_out
        a_ret = mw.intercept_ret

        # Evaluate the cut, and the true recourse, at OTHER_TAUS -- a schedule MW
        # was never anchored at.
        y_other_out, y_other_ret = _y_inc(OTHER_TAUS)
        cut_out = a_out + sum(mw.dm_out[t] * y_other_out[t] for t in range(T))
        cut_ret = a_ret + sum(mw.dm_ret[t] * y_other_ret[t] for t in range(T))

        c_out_other, c_ret_other = _caps(OTHER_TAUS)
        _duals_other, ub_other = solve_minute_recourse(
            T, DELTA, WMAX_MINUTES, P_SLOTS, c_out_other, c_ret_other, REQUESTS,
            policy="start", lp_solver=self.backend,
        )
        # A single combined theta bound in this project's convention is
        # cut_out + cut_ret <= Q_out(other) + Q_ret(other) = ub_other (both terms
        # share the objective scale, slot-equivalent units).
        self.assertLessEqual(cut_out + cut_ret, ub_other + 1e-6)

    def test_a_pi_with_no_dual_feasibility_row_is_fixed_to_zero_not_left_unbounded(self):
        """The one real structural difference from the slot path (see
        solve_mw_dual_minute's own docstring): whether pi[tau] appears in any row
        is data-dependent here, not just tau=0. Slot 0 in this instance (policy
        start, offset 0) can only be reached by an arrival at minute 0 -- there is
        none in REQUESTS -- so it is exactly the orphaned case. It must not blow
        up the LP (unbounded) or silently vanish (KeyError); it must come back
        0.0."""
        c_out, c_ret = _caps(INCUMBENT_TAUS)
        # Give slot 0 real capacity, so its objective coefficient (seats*Ybar - C)
        # is negative if Ybar[0] < 1 -- the unboundedness direction this guards.
        c_out[0] = SEATS
        ybar_out = [0.0, 0.6, 1.0, 0.0, 0.5, 0.0, 0.9, 0.0]
        ybar_ret = [0.0, 0.0, 0.7, 1.0, 0.0, 0.4, 0.0, 1.0]
        _duals, ub = solve_minute_recourse(
            T, DELTA, WMAX_MINUTES, P_SLOTS, c_out, c_ret, REQUESTS,
            policy="start", lp_solver=self.backend,
        )
        mw = solve_mw_dual_minute(
            T, DELTA, WMAX_MINUTES, P_SLOTS, SEATS, c_out, c_ret, REQUESTS,
            ybar_out, ybar_ret, ub, policy="start", lp_solver=self.backend,
        )
        self.assertIsNotNone(mw, "the LP must solve, not go unbounded")
        self.assertEqual(mw.dm_out[0], 0.0)


if __name__ == "__main__":
    unittest.main()
