"""The single-vehicle decision diagram (DESIGN_DD_v1 stage 1, D48). No solver needed.

The load-bearing assertion here is `test_dp_matches_exhaustive_enumeration`. The DP
carries one number per `(time, state, trips)` -- the maximum reachable battery -- on the
argument that more battery is weakly better for every continuation. That dominance rule
is what makes the diagram small, and it is exactly the kind of claim this project has
had refuted four times by measurement (D30, D34, D37, D40). So it is checked against a
brute-force search that does no merging at all.

Run just these:
    python -m unittest tests.test_fast_vehicle_dd -v
"""

from __future__ import annotations

import unittest

import _helpers  # noqa: F401  (puts src/ on sys.path)

from mobauto2_benders.problem.vehicle_dd import (
    LONGVILLIERS,
    MASSY,
    VehicleParams,
    departure_eligible,
    energy_bound,
    eligible_slot_count,
    max_trips_in_window,
    travel_time_bound,
    window_trip_caps,
)


# The Fase 1 test point: T_minutes 660 / slot_resolution 15 -> T=44,
# trip_duration_minutes 30 -> trip_slots 2, delta_chg = 70/(60/15) = 17.5.
TEST_POINT = VehicleParams(T=44, trip_slots=2, Emax=150.0, L=30.0, delta_chg=17.5)

# The soundness fixture: slot_resolution 30 -> T=22, trip_slots 1, delta_chg = 35.
SOUNDNESS = VehicleParams(T=22, trip_slots=1, Emax=150.0, L=30.0, delta_chg=35.0)

# Charging cannot keep up with consumption here, so the battery is what limits trips
# rather than the trip spacing. Shared by the enumeration check and the non-vacuity
# check so the two cannot drift apart.
ENERGY_STARVED = VehicleParams(T=8, trip_slots=1, Emax=90.0, L=40.0, delta_chg=5.0)


def _brute_force_max_trips(P: VehicleParams, t1: int, t2: int) -> int:
    """Exhaustive search over action sequences. No dominance, no merging.

    Deliberately written as a plain recursion so that it shares no structure with the
    DP beyond the model semantics. If the two agree, the DP's max-battery dominance is
    doing what it claims.
    """

    def rec(t: int, loc: int, rem: int, b: float) -> int:
        if t > t2:
            return 0
        if rem > 0:
            return rec(t + 1, loc, rem - 1, b)
        best = 0
        if loc == LONGVILLIERS:
            if departure_eligible(P, t, LONGVILLIERS) and b >= 2.0 * P.L:
                best = max(best, 1 + rec(t + 1, MASSY, P.trip_slots - 1, b - P.L))
            # charge
            best = max(
                best, rec(t + 1, LONGVILLIERS, 0, min(P.Emax, b + P.delta_chg))
            )
            # idle -- kept explicitly here even though charging dominates, so the
            # brute force does not assume the very thing under test
            best = max(best, rec(t + 1, LONGVILLIERS, 0, b))
        else:
            if departure_eligible(P, t, MASSY) and b >= P.L:
                best = max(
                    best, 1 + rec(t + 1, LONGVILLIERS, P.trip_slots - 1, b - P.L)
                )
            best = max(best, rec(t + 1, MASSY, 0, b))
        return best

    if t2 < t1:
        return 0
    return max(
        rec(t1, LONGVILLIERS, 0, float(P.Emax)),
        rec(t1, MASSY, 0, float(P.Emax)),
    )


class TestDominanceRule(unittest.TestCase):
    def test_dp_matches_exhaustive_enumeration(self):
        """The max-battery dominance is exact for this relaxed diagram.

        Small instances only -- the brute force is exponential, which is the point.
        """
        # Kept at T<=8: the brute force is exponential by design, so these are sized
        # to stay in the fast suite rather than to look impressive.
        cases = [
            VehicleParams(T=8, trip_slots=2, Emax=150.0, L=30.0, delta_chg=17.5),
            VehicleParams(T=8, trip_slots=1, Emax=150.0, L=30.0, delta_chg=35.0),
            # Energy-starved: charging cannot keep up, so battery is what binds and
            # the dominance rule is actually load-bearing rather than vacuous.
            ENERGY_STARVED,
            # No charger at all: the vehicle has a hard trip budget.
            VehicleParams(T=8, trip_slots=2, Emax=150.0, L=30.0, delta_chg=0.0),
        ]
        for P in cases:
            for t1 in range(P.T):
                for t2 in range(t1, P.T):
                    with self.subTest(P=P, t1=t1, t2=t2):
                        self.assertEqual(
                            max_trips_in_window(P, t1, t2),
                            _brute_force_max_trips(P, t1, t2),
                        )

    def test_the_enumeration_check_is_not_vacuous(self):
        """At least one case must have battery, not spacing, as the binding limit.

        Without this, every subTest above could be passing because the travel-time
        spacing alone determines the answer, and the dominance rule -- the only thing
        the DP does that the brute force does not -- would be untested.
        """
        P = ENERGY_STARVED
        binding = [
            (t1, t2)
            for t1 in range(P.T)
            for t2 in range(t1, P.T)
            if energy_bound(P, t1, t2) < travel_time_bound(P, t1, t2)
        ]
        self.assertTrue(
            binding,
            "no window in the energy-starved case is energy-limited; the dominance "
            "rule is not being exercised",
        )


class TestIndependentBounds(unittest.TestCase):
    def test_dp_never_exceeds_either_analytic_bound(self):
        """A DP above either bound is a bug in the diagram, not a property of data."""
        for P in (TEST_POINT, SOUNDNESS):
            for t1 in range(P.T):
                for t2 in range(t1, P.T):
                    with self.subTest(P=P, t1=t1, t2=t2):
                        k = max_trips_in_window(P, t1, t2)
                        self.assertLessEqual(k, travel_time_bound(P, t1, t2))
                        self.assertLessEqual(k, energy_bound(P, t1, t2))

    def test_bounds_disagree_somewhere(self):
        """The two bounds must not be the same function in disguise.

        If travel time always dominated energy (or vice versa) the cross-check in
        `window_trip_caps` would be one check reported as two.
        """
        P = TEST_POINT
        tt_tighter = any(
            travel_time_bound(P, t1, t2) < energy_bound(P, t1, t2)
            for t1 in range(P.T)
            for t2 in range(t1, P.T)
        )
        en_tighter = any(
            energy_bound(P, t1, t2) < travel_time_bound(P, t1, t2)
            for t1 in range(P.T)
            for t2 in range(t1, P.T)
        )
        self.assertTrue(tt_tighter, "travel-time bound is never the tighter one")
        self.assertTrue(en_tighter, "energy bound is never the tighter one")


class TestValidityDirection(unittest.TestCase):
    """A cap that is too large is weak; a cap that is too small is a wrong answer."""

    def test_max_trips_is_monotone_in_the_window(self):
        """Widening a window cannot reduce the number of departures it admits."""
        P = TEST_POINT
        for t1 in range(0, P.T, 3):
            prev = -1
            for t2 in range(t1, P.T):
                k = max_trips_in_window(P, t1, t2)
                self.assertGreaterEqual(k, prev, f"window [{t1},{t2}] shrank the cap")
                prev = k

    def test_a_full_horizon_window_admits_at_least_one_round_trip(self):
        """Sanity floor: the cap must not be so tight it forbids operating at all."""
        for P in (TEST_POINT, SOUNDNESS):
            self.assertGreaterEqual(max_trips_in_window(P, 0, P.T - 1), 2)

    def test_windows_of_forbidden_slots_admit_nothing(self):
        """The horizon-end fixings must show up in the diagram."""
        P = TEST_POINT
        # Every departure is fixed to 0 from T - trip_slots onward.
        self.assertEqual(max_trips_in_window(P, P.T - P.trip_slots, P.T - 1), 0)
        # Slot 0 is the demand bucket, never a departure slot.
        self.assertEqual(max_trips_in_window(P, 0, 0), 0)

    def test_out_is_forbidden_before_ret_is(self):
        """C5's returnability fixing bites earlier than the arrival fixing."""
        P = TEST_POINT
        t = P.T - 2 * P.trip_slots
        self.assertFalse(departure_eligible(P, t, LONGVILLIERS))
        self.assertTrue(departure_eligible(P, t, MASSY))


class TestWindowCaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.caps_q3 = window_trip_caps(TEST_POINT, Q=3)
        cls.caps_q5 = window_trip_caps(TEST_POINT, Q=5)

    def test_caps_are_strictly_below_the_trivial_bound(self):
        caps = self.caps_q3
        self.assertTrue(caps, "no non-trivial window cap at the Fase 1 test point")
        for c in caps:
            trivial = eligible_slot_count(TEST_POINT, c.t1, c.t2)
            self.assertLess(
                c.max_trips_per_vehicle,
                trivial,
                f"cap on [{c.t1},{c.t2}] is implied by Y_OUT+Y_RET <= Q",
            )

    def test_rhs_scales_with_the_fleet(self):
        """The property D33's recourse anchor lacked: it must not go slack as Q grows."""
        c3 = {(c.t1, c.t2): c for c in self.caps_q3}
        c5 = {(c.t1, c.t2): c for c in self.caps_q5}
        self.assertEqual(set(c3), set(c5))
        for key, cap3 in c3.items():
            self.assertEqual(c5[key].rhs, cap3.rhs // 3 * 5)

    def test_q_must_be_positive(self):
        with self.assertRaises(ValueError):
            window_trip_caps(TEST_POINT, Q=0)


class TestParameterGuards(unittest.TestCase):
    """Fail closed on units, rather than returning a cap of 0 that silently
    forbids every departure."""

    def test_emax_below_two_L_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            VehicleParams(T=10, trip_slots=1, Emax=50.0, L=30.0, delta_chg=10.0)
        self.assertIn("units", str(ctx.exception))

    def test_degenerate_parameters_are_refused(self):
        for kwargs in (
            dict(T=0, trip_slots=1, Emax=150.0, L=30.0, delta_chg=10.0),
            dict(T=10, trip_slots=0, Emax=150.0, L=30.0, delta_chg=10.0),
            dict(T=10, trip_slots=1, Emax=150.0, L=0.0, delta_chg=10.0),
            dict(T=10, trip_slots=1, Emax=150.0, L=30.0, delta_chg=-1.0),
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    VehicleParams(**kwargs)


if __name__ == "__main__":
    unittest.main()
