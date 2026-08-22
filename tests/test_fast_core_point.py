"""S3: the Magnanti-Wong core point lies inside the projected master region.

No solver. Magnanti-Wong requires a point in the relative interior of
``proj_Y(conv(Z))`` (formal formulation 16.2). Cut VALIDITY does not depend on this --
that comes from dual feasibility -- but the Pareto-optimality claim does, and the claim
is the only reason to run MW rather than the plain dual.

The core point used to be an EMA clamped to the box ``[eps, Q-eps]`` per slot, which is
outside the region on two counts: positive ``Ybar`` on slots the master FIXES to zero,
and no trip-window constraint at all, so a point could assert that the whole fleet
departs in every slot.
"""

from __future__ import annotations

import unittest

import _helpers  # noqa: F401  -- puts src/ on sys.path

from mobauto2_benders.signature import (
    core_point_violations,
    departures_are_possible,
    project_core_point,
)


class WhichSlotsCanCarryADeparture(unittest.TestCase):
    def test_it_matches_the_masters_fixings(self):
        """T=6, trip=1: OUT fixed for t >= T-2*trip = 4, RET for t >= T-trip = 5."""
        ok_out, ok_ret = departures_are_possible(6, 1)
        self.assertEqual(ok_out, [True, True, True, True, False, False])
        self.assertEqual(ok_ret, [True, True, True, True, True, False])

    def test_a_longer_trip_fixes_more(self):
        ok_out, ok_ret = departures_are_possible(6, 2)
        self.assertEqual(ok_out, [True, True, False, False, False, False])
        self.assertEqual(ok_ret, [True, True, True, True, False, False])

    def test_a_horizon_too_short_for_any_round_trip_fixes_every_out(self):
        ok_out, _ = departures_are_possible(3, 2)
        self.assertFalse(any(ok_out))


class TheProjectionEnforcesTheRegion(unittest.TestCase):
    Q = 2
    T = 8
    TRIP = 2
    EPS = 1e-3

    def _project(self, Y_out, Y_ret):
        return project_core_point(Y_out, Y_ret, self.Q, self.TRIP, self.EPS)

    def test_a_box_point_is_brought_inside(self):
        """The old clamp's output: Q-eps everywhere, both directions."""
        hi = float(self.Q) - self.EPS
        out, ret = self._project([hi] * self.T, [hi] * self.T)
        self.assertEqual(
            core_point_violations(out, ret, self.Q, self.TRIP),
            [],
            "the projection left the box point outside the region",
        )

    def test_fixed_slots_come_out_exactly_zero(self):
        """Not floored to eps. Flooring them would reintroduce the defect: MW would be
        asked which dual best values capacity in a slot that cannot carry a departure."""
        out, ret = self._project([1.0] * self.T, [1.0] * self.T)
        ok_out, ok_ret = departures_are_possible(self.T, self.TRIP)
        for t in range(self.T):
            if not ok_out[t]:
                self.assertEqual(out[t], 0.0, f"Yout[{t}] should be exactly 0")
            if not ok_ret[t]:
                self.assertEqual(ret[t], 0.0, f"Yret[{t}] should be exactly 0")

    def test_free_slots_stay_strictly_positive(self):
        """Otherwise the point is on the boundary and MW's selection degrades."""
        out, ret = self._project([0.0] * self.T, [0.0] * self.T)
        ok_out, ok_ret = departures_are_possible(self.T, self.TRIP)
        for t in range(self.T):
            if ok_out[t]:
                self.assertGreater(out[t], 0.0, f"Yout[{t}] sits on the boundary")
            if ok_ret[t]:
                self.assertGreater(ret[t], 0.0, f"Yret[{t}] sits on the boundary")

    def test_the_trip_window_holds(self):
        """A vehicle starting at u cannot start again before u+trip_slots, so any window
        of trip_slots consecutive slots carries at most Q starts fleet-wide."""
        out, ret = self._project([2.0] * self.T, [2.0] * self.T)
        for t0 in range(self.T):
            total = sum(
                out[t] + ret[t] for t in range(t0, min(self.T, t0 + self.TRIP))
            )
            self.assertLessEqual(
                total, float(self.Q) + 1e-9, f"window starting at {t0} starts {total}"
            )

    def test_it_is_idempotent(self):
        """Projecting a projected point must change nothing.

        The solver asserts this at runtime -- it re-checks after projecting and raises --
        so a non-idempotent projection would abort every run rather than degrade quietly.
        """
        once = self._project([2.0] * self.T, [1.3] * self.T)
        twice = project_core_point(*once, self.Q, self.TRIP, self.EPS)
        for a, b in zip(once[0], twice[0]):
            self.assertAlmostEqual(a, b, places=12)
        for a, b in zip(once[1], twice[1]):
            self.assertAlmostEqual(a, b, places=12)

    def test_scaling_preserves_the_shape_of_a_profile(self):
        """Windows are scaled proportionally, so a peak stays a peak. A projection that
        flattened the profile would erase the direction MW is meant to select along."""
        Y = [0.1, 3.0, 0.1, 3.0, 0.1, 0.1, 0.1, 0.1]
        out, _ = self._project(Y, [0.0] * self.T)
        self.assertGreater(out[1], out[0])
        self.assertGreater(out[3], out[2])

    def test_negative_input_is_clipped_not_propagated(self):
        out, ret = self._project([-5.0] * self.T, [-5.0] * self.T)
        self.assertEqual(core_point_violations(out, ret, self.Q, self.TRIP), [])
        self.assertTrue(all(v >= 0.0 for v in out + ret))

    def test_mismatched_halves_raise(self):
        with self.assertRaises(ValueError):
            project_core_point([1.0, 1.0], [1.0], self.Q, self.TRIP, self.EPS)


class TheViolationReportNamesTheOffender(unittest.TestCase):
    def test_it_flags_a_fixed_slot(self):
        bad = core_point_violations([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], Q=2, trip_slots=1)
        self.assertTrue(any("fixes to 0" in b for b in bad))

    def test_it_flags_an_overfull_window(self):
        bad = core_point_violations([3.0, 0.0], [0.0, 0.0], Q=2, trip_slots=1)
        self.assertTrue(any("above Q=2" in b for b in bad))

    def test_a_projected_point_reports_nothing(self):
        out, ret = project_core_point([9.0] * 8, [9.0] * 8, 3, 2, 1e-3)
        self.assertEqual(core_point_violations(out, ret, 3, 2), [])

    def test_the_report_is_not_vacuous(self):
        """It must be capable of returning something -- a checker that always passes is
        the same as no checker."""
        self.assertNotEqual(
            core_point_violations([5.0], [5.0], Q=1, trip_slots=1),
            [],
        )


if __name__ == "__main__":
    unittest.main()
