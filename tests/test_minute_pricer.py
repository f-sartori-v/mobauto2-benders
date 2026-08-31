"""The minute-level pricer. REQUIRES AN LP BACKEND; a couple of seconds.

This module is a measuring instrument -- the number it produces is the evidence for or
against the whole multi-resolution premise -- so its arithmetic is pinned against cases
small enough to check by hand.

Run just these:
    python -m unittest tests.test_minute_pricer -v
"""

from __future__ import annotations

import unittest

import _helpers  # noqa: F401  (puts src/ on sys.path)
from _helpers import require_solver_backend
from mobauto2_benders import minute_pricer as _minute_pricer
from mobauto2_benders.minute_pricer import (
    departure_minutes,
    load_request_minutes,
    slot_objective_in_minutes,
)

P_MIN = 1500.0


# The pricer's two solving entry points default to `cplex_direct` in their own
# signatures, which is right for production and wrong for a test: it would pin the
# assertions to a backend the checkout may not have, which is how these tests came
# to be skipped wholesale. Route them through the resolved backend instead. Every
# call below passes `lp_solver` by keyword or not at all, so setdefault cannot
# shadow an explicit choice.
def price_direction_at_minutes(*args, **kwargs):
    kwargs.setdefault("lp_solver", require_solver_backend())
    return _minute_pricer.price_direction_at_minutes(*args, **kwargs)


def price_schedule_at_minutes(*args, **kwargs):
    kwargs.setdefault("lp_solver", require_solver_backend())
    return _minute_pricer.price_schedule_at_minutes(*args, **kwargs)


class TestPricingArithmetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_solver_backend()

    def test_everyone_boards_the_single_departure(self):
        """Arrivals at 0, 10, 20; departure at 30; seats to spare.
        Waits are 30 + 20 + 10 = 60 passenger-minutes."""
        w, u, s = price_direction_at_minutes([0, 10, 20], [30.0], 15, 60, P_MIN)
        self.assertAlmostEqual(w, 60.0, places=6)
        self.assertEqual(u, 0.0)
        self.assertEqual(s, 3.0)

    def test_capacity_forces_the_cheapest_two(self):
        """Two seats for three passengers. Leaving one costs 1500, so it serves two --
        and it keeps the two who wait least: 20 + 10 = 30, plus 1500."""
        w, u, s = price_direction_at_minutes([0, 10, 20], [30.0], 2, 60, P_MIN)
        self.assertAlmostEqual(w, 30.0, places=6)
        self.assertEqual(u, 1.0)
        self.assertEqual(s, 2.0)

    def test_the_wait_cap_is_enforced_in_real_minutes(self):
        """A departure 90 minutes after arrival is out of reach at Wmax 60, so the
        passenger is unserved however much capacity there is. This is the check the
        slot model cannot make -- its window is counted in slots."""
        w, u, s = price_direction_at_minutes([0], [90.0], 15, 60, P_MIN)
        self.assertEqual(u, 1.0)
        self.assertEqual(s, 0.0)
        self.assertAlmostEqual(w, 0.0, places=6)

    def test_boarding_at_the_arrival_minute_is_free_and_allowed(self):
        w, u, s = price_direction_at_minutes([30], [30.0], 15, 60, P_MIN)
        self.assertAlmostEqual(w, 0.0, places=6)
        self.assertEqual(u, 0.0)

    def test_no_departures_means_everyone_is_unserved(self):
        w, u, s = price_direction_at_minutes([0, 5], [], 15, 60, P_MIN)
        self.assertEqual(u, 2.0)
        self.assertEqual(s, 0.0)


class TestPlacementPolicy(unittest.TestCase):
    def test_start_and_midpoint_differ_by_half_a_slot(self):
        self.assertEqual(departure_minutes([0, 2], 30, "start"), [0.0, 60.0])
        self.assertEqual(departure_minutes([0, 2], 30, "midpoint"), [15.0, 75.0])

    def test_midpoint_can_breach_a_cap_that_start_respects(self):
        """The slot abstraction's own artifact, in one assertion: two slots is 60
        minutes under `start` and 75 under `midpoint`, against a stated cap of 60.
        This is why the pricer enforces the cap in minutes."""
        require_solver_backend()
        for policy, expect_served in (("start", 1.0), ("midpoint", 0.0)):
            with self.subTest(policy=policy):
                deps = departure_minutes([2], 30, policy)
                _w, u, s = price_direction_at_minutes([0], deps, 15, 60, P_MIN)
                self.assertEqual(s, expect_served)
                self.assertEqual(u, 1.0 - expect_served)


class TestUnitConversion(unittest.TestCase):
    def test_slot_objective_converts_to_passenger_minutes(self):
        """baseline_d9: 283 slots of waiting and 78 unserved at p=50, slot 30.
        283*30 + 30*50*78 = 8490 + 117000 = 125490."""
        self.assertAlmostEqual(
            slot_objective_in_minutes(283, 78, 30, 50), 125490.0, places=6
        )


class TestDemandLoading(unittest.TestCase):
    def test_arrival_minutes_are_kept_unaggregated(self):
        """The whole point: the setups carry per-request minutes and the slot model
        discards them at load."""
        reqs = load_request_minutes(_helpers.REPO_ROOT / "setups" / "base.yaml")
        self.assertEqual(len(reqs["OUT"]), 150)
        self.assertEqual(len(reqs["RET"]), 150)
        self.assertGreater(
            len(set(reqs["OUT"])), 22, "minutes look aggregated to slots, not kept raw"
        )


if __name__ == "__main__":
    unittest.main()


class TestDecomposedMinuteRecourse(unittest.TestCase):
    """`solve_minute_recourse` is what the Benders loop calls in minute mode. It is
    cross-checked against `price_schedule_at_minutes`, which is an independent
    implementation of the same quantity -- one builds a slot-indexed capacity LP for the
    decomposition, the other a per-departure LP for reporting. Agreement between two
    implementations written for different purposes is worth more than either alone.
    """

    @classmethod
    def setUpClass(cls):
        require_solver_backend()

    def _case(self):
        T, delta, S, wmax, p_slots = 6, 30, 15.0, 60.0, 50.0
        # Departures in slots 2 and 4, one vehicle each -> capacity S at those slots.
        C_out = [0.0] * T
        C_out[2] = S
        C_out[4] = S
        C_ret = [0.0] * T
        reqs = {"OUT": [5, 35, 40, 100, 101], "RET": []}
        return T, delta, S, wmax, p_slots, C_out, C_ret, reqs

    def test_it_agrees_with_the_reporting_pricer(self):
        from mobauto2_benders.minute_pricer import solve_minute_recourse

        T, delta, S, wmax, p_slots, C_out, C_ret, reqs = self._case()
        duals, obj_slot_units = solve_minute_recourse(
            T, delta, wmax, p_slots, C_out, C_ret, reqs, policy="midpoint",
            lp_solver=require_solver_backend(),
        )
        priced = price_schedule_at_minutes(
            {"OUT": [2, 4], "RET": []}, reqs, delta, S, wmax, p_slots * delta,
            policy="midpoint",
        )
        # solve_minute_recourse works in slot-equivalent units; the pricer in minutes.
        self.assertAlmostEqual(obj_slot_units * delta, priced.total_cost, places=6)
        self.assertAlmostEqual(
            duals["penalty_pax"], priced.unserved_passengers, places=6
        )

    def test_capacity_duals_are_indexed_by_departure_slot_and_non_positive(self):
        """The property that lets the existing cut machinery work untouched: one dual
        per slot, and non-positive because more capacity cannot raise the cost."""
        from mobauto2_benders.minute_pricer import solve_minute_recourse

        T, delta, S, wmax, p_slots, C_out, C_ret, reqs = self._case()
        duals, _ = solve_minute_recourse(
            T, delta, wmax, p_slots, C_out, C_ret, reqs, policy="midpoint",
            lp_solver=require_solver_backend(),
        )
        self.assertEqual(set(duals["pi_OUT"]), set(range(T)))
        self.assertEqual(set(duals["pi_RET"]), set(range(T)))
        for t, v in duals["pi_OUT"].items():
            self.assertLessEqual(v, 1e-9, f"pi_OUT[{t}] is positive: {v}")

    def test_the_recourse_is_non_increasing_in_capacity(self):
        """P3 at minute resolution. If this fails the cut slopes have the wrong sign
        and every bound built on them is void (D30)."""
        from mobauto2_benders.minute_pricer import solve_minute_recourse

        T, delta, S, wmax, p_slots, C_out, C_ret, reqs = self._case()
        _, base = solve_minute_recourse(
            T, delta, wmax, p_slots, C_out, C_ret, reqs, policy="midpoint",
            lp_solver=require_solver_backend(),
        )
        richer = list(C_out)
        richer[2] += S
        _, more = solve_minute_recourse(
            T, delta, wmax, p_slots, richer, C_ret, reqs, policy="midpoint",
            lp_solver=require_solver_backend(),
        )
        self.assertLessEqual(more, base + 1e-9)

    def test_placement_offsets_none_matches_todays_single_offset_model(self):
        """F2's whole safety argument in one test: the default must be numerically
        IDENTICAL to a singleton grid holding exactly the policy's own offset, not
        merely 'close'. If these ever drift apart, F2 has silently changed the
        model everyone else's numbers were produced under."""
        from mobauto2_benders.minute_pricer import placement_offset, solve_minute_recourse

        T, delta, S, wmax, p_slots, C_out, C_ret, reqs = self._case()
        duals_default, obj_default = solve_minute_recourse(
            T, delta, wmax, p_slots, C_out, C_ret, reqs, policy="midpoint",
            lp_solver=require_solver_backend(),
        )
        duals_explicit, obj_explicit = solve_minute_recourse(
            T, delta, wmax, p_slots, C_out, C_ret, reqs, policy="midpoint",
            lp_solver=require_solver_backend(),
            placement_offsets=[placement_offset("midpoint", delta)],
        )
        self.assertAlmostEqual(obj_default, obj_explicit, places=9)
        self.assertEqual(duals_default["pi_OUT"], duals_explicit["pi_OUT"])
        self.assertEqual(duals_default["pi_RET"], duals_explicit["pi_RET"])
        self.assertEqual(duals_explicit["placement_offsets"], [15.0])

    def test_placement_freedom_is_a_relaxation_never_worse_than_either_offset_alone(self):
        """F2's central claim: a richer offset grid can only lower or match the
        recourse, never raise it, because every single-offset arc set is a subset
        of the multi-offset one. Q_relaxed <= Q_true in the direction that matters
        for a lower bound."""
        from mobauto2_benders.minute_pricer import solve_minute_recourse

        T, delta, S, wmax, p_slots, C_out, C_ret, reqs = self._case()
        _, obj_start = solve_minute_recourse(
            T, delta, wmax, p_slots, C_out, C_ret, reqs,
            lp_solver=require_solver_backend(), placement_offsets=[0.0],
        )
        _, obj_end = solve_minute_recourse(
            T, delta, wmax, p_slots, C_out, C_ret, reqs,
            lp_solver=require_solver_backend(), placement_offsets=[delta],
        )
        _, obj_both = solve_minute_recourse(
            T, delta, wmax, p_slots, C_out, C_ret, reqs,
            lp_solver=require_solver_backend(), placement_offsets=[0.0, delta],
        )
        self.assertLessEqual(obj_both, min(obj_start, obj_end) + 1e-9)

    def test_capacity_row_stays_one_per_slot_under_a_multi_offset_grid(self):
        """The interface condition F2 depends on: however many offsets are in the
        grid, capacity is still indexed by slot alone, so the cut the master
        receives is the same object it always was."""
        from mobauto2_benders.minute_pricer import solve_minute_recourse

        T, delta, S, wmax, p_slots, C_out, C_ret, reqs = self._case()
        duals, _ = solve_minute_recourse(
            T, delta, wmax, p_slots, C_out, C_ret, reqs,
            lp_solver=require_solver_backend(), placement_offsets=[0.0, 10.0, 20.0, 30.0],
        )
        self.assertEqual(set(duals["pi_OUT"]), set(range(T)))
        self.assertEqual(set(duals["pi_RET"]), set(range(T)))
        for t, v in duals["pi_OUT"].items():
            self.assertLessEqual(v, 1e-9, f"pi_OUT[{t}] is positive: {v}")

    def test_placement_offsets_outside_the_slot_are_refused(self):
        from mobauto2_benders.minute_pricer import solve_minute_recourse

        T, delta, S, wmax, p_slots, C_out, C_ret, reqs = self._case()
        with self.assertRaises(ValueError):
            solve_minute_recourse(
                T, delta, wmax, p_slots, C_out, C_ret, reqs,
                lp_solver=require_solver_backend(), placement_offsets=[-5.0],
            )
        with self.assertRaises(ValueError):
            solve_minute_recourse(
                T, delta, wmax, p_slots, C_out, C_ret, reqs,
                lp_solver=require_solver_backend(), placement_offsets=[delta + 1.0],
            )

    def test_reaches_solve_subproblem_through_sp_params(self):
        """End-to-end through the same dispatch the Benders loop uses, not just the
        library function directly."""
        from mobauto2_benders.problem.subproblem_impl import SPParams, solve_subproblem

        T, delta, S, wmax, p_slots, C_out, C_ret, reqs = self._case()
        P = SPParams(
            T=T, Wmax_slots=2, p=p_slots, lp_solver=require_solver_backend(), S=S,
            K_out=[0] * T, K_ret=[0] * T, slot_resolution=delta,
            recourse_resolution="minute", Wmax_minutes=wmax, request_minutes=reqs,
            placement_offsets=[0.0, delta],
        )
        duals, obj = solve_subproblem(P, C_out, C_ret, [0.0] * T, [0.0] * T)
        self.assertEqual(duals["placement_offsets"], [0.0, delta])

    def test_minute_mode_refuses_to_run_without_arrival_minutes(self):
        """Falling back to the slot model here would report a multi-resolution run
        that never happened."""
        from mobauto2_benders.problem.subproblem_impl import SPParams, solve_subproblem

        P = SPParams(
            T=6, Wmax_slots=2, p=50.0, lp_solver=require_solver_backend(), S=15.0,
            K_out=[0] * 6, K_ret=[0] * 6, slot_resolution=30,
            recourse_resolution="minute", Wmax_minutes=60.0, request_minutes=None,
        )
        with self.assertRaises(ValueError) as ctx:
            solve_subproblem(P, [0.0] * 6, [0.0] * 6, [0.0] * 6, [0.0] * 6)
        self.assertIn("request_minutes", str(ctx.exception))
