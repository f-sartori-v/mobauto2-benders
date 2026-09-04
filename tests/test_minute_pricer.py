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


def price_schedule_given_departure_minutes(*args, **kwargs):
    kwargs.setdefault("lp_solver", require_solver_backend())
    return _minute_pricer.price_schedule_given_departure_minutes(*args, **kwargs)


class TestPricingArithmetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_solver_backend()

    def test_everyone_boards_the_single_departure(self):
        """Arrivals at 0, 10, 20; departure at 30; seats to spare.
        Waits are 30 + 20 + 10 = 60 passenger-minutes."""
        w, u, s, _fs, _rows = price_direction_at_minutes([0, 10, 20], [30.0], 15, 60, P_MIN)
        self.assertAlmostEqual(w, 60.0, places=6)
        self.assertEqual(u, 0.0)
        self.assertEqual(s, 3.0)

    def test_capacity_forces_the_cheapest_two(self):
        """Two seats for three passengers. Leaving one costs 1500, so it serves two --
        and it keeps the two who wait least: 20 + 10 = 30, plus 1500."""
        w, u, s, _fs, _rows = price_direction_at_minutes([0, 10, 20], [30.0], 2, 60, P_MIN)
        self.assertAlmostEqual(w, 30.0, places=6)
        self.assertEqual(u, 1.0)
        self.assertEqual(s, 2.0)

    def test_the_wait_cap_is_enforced_in_real_minutes(self):
        """A departure 90 minutes after arrival is out of reach at Wmax 60, so the
        passenger is unserved however much capacity there is. This is the check the
        slot model cannot make -- its window is counted in slots."""
        w, u, s, _fs, _rows = price_direction_at_minutes([0], [90.0], 15, 60, P_MIN)
        self.assertEqual(u, 1.0)
        self.assertEqual(s, 0.0)
        self.assertAlmostEqual(w, 0.0, places=6)

    def test_same_slot_flag(self):
        """B6. `forbid` and `allow` give different arc sets and different objectives.

        The instance is the minimal one that separates them: a single arrival at
        minute 30 and a single departure at minute 30. Under `allow` the passenger
        steps onto the waiting vehicle for zero wait; under `forbid` that arc does not
        exist, there is nowhere else to go, and the passenger is unserved at cost
        `p_minutes`.

        This test fails on the pre-B6 code in the only way it can: the convention was
        hard-coded to `allow` there, so `forbid` was not expressible and the two arms
        of the comparison were the same run.
        """
        w_allow, u_allow, s_allow, _fs, rows_allow = price_direction_at_minutes(
            [30], [30.0], 15, 60, P_MIN, same_slot_eligibility="allow"
        )
        self.assertAlmostEqual(w_allow, 0.0, places=6)
        self.assertEqual(u_allow, 0.0)
        self.assertEqual(s_allow, 1.0)

        w_forbid, u_forbid, s_forbid, _fs2, rows_forbid = price_direction_at_minutes(
            [30], [30.0], 15, 60, P_MIN, same_slot_eligibility="forbid"
        )
        self.assertEqual(u_forbid, 1.0)
        self.assertEqual(s_forbid, 0.0)
        self.assertAlmostEqual(w_forbid, 0.0, places=6)

        # Different arc sets: one boarding row against one rejection row.
        self.assertEqual([r.served for r in rows_allow], [True])
        self.assertEqual([r.served for r in rows_forbid], [False])
        # Different objectives.
        self.assertNotAlmostEqual(
            w_allow + P_MIN * u_allow, w_forbid + P_MIN * u_forbid, places=6
        )

    def test_forbid_is_the_default(self):
        """The default must be the master's own tau >= t+1 rule, not the opposite.

        Under `departure_policy="start"`, `dep - m >= 1` is exactly `tau >= t + 1` at
        any resolution, so a minute run and a slot run share an arc set by default and
        differ only in how finely they price the wait. That is what makes them
        comparable, and its absence is why the reported delta=1 figures compared two
        conventions at once.
        """
        default = price_direction_at_minutes([30], [30.0], 15, 60, P_MIN)
        forbid = price_direction_at_minutes(
            [30], [30.0], 15, 60, P_MIN, same_slot_eligibility="forbid"
        )
        self.assertEqual(default[:4], forbid[:4])

    def test_an_unknown_eligibility_is_refused(self):
        with self.assertRaises(ValueError):
            price_direction_at_minutes(
                [30], [30.0], 15, 60, P_MIN, same_slot_eligibility="sometimes"
            )

    def test_no_departures_means_everyone_is_unserved(self):
        w, u, s, _fs, _rows = price_direction_at_minutes([0, 5], [], 15, 60, P_MIN)
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
                _w, u, s, _fs, _rows = price_direction_at_minutes([0], deps, 15, 60, P_MIN)
                self.assertEqual(s, expect_served)
                self.assertEqual(u, 1.0 - expect_served)


class TestGivenDepartureMinutes(unittest.TestCase):
    """A4c: the validator accepts departures already expressed in minutes -- the
    piece Comparison C needs, because the continuous-time CP model does not place
    departures on any slot grid at all."""

    @classmethod
    def setUpClass(cls):
        require_solver_backend()

    def test_same_arithmetic_as_the_slot_path_on_a_slot_aligned_schedule(self):
        """A departure at slot 1 under `midpoint`, delta=30, lands at minute 15 --
        exactly `test_everyone_boards_the_single_departure`'s scenario restated in
        minutes. Both paths must agree to the cent: this function is that test's
        LP with the slot-to-minute conversion done by the caller instead of inside
        it, not a different calculation."""
        w, u, s, _fs, _rows = price_direction_at_minutes([0, 10, 20], [15.0], 15, 60, P_MIN)
        result = price_schedule_given_departure_minutes(
            {"OUT": [15.0], "RET": []},
            {"OUT": [0, 10, 20], "RET": []},
            15,
            60,
            P_MIN,
        )
        self.assertAlmostEqual(result.waiting_minutes, w, places=6)
        self.assertAlmostEqual(result.unserved_passengers, u, places=6)
        self.assertAlmostEqual(result.served_passengers, s, places=6)

    def test_agrees_with_the_slot_path_when_fed_the_same_converted_minutes(self):
        """Build a schedule the ordinary way (slots + a placement policy), convert
        it once with the module's own `departure_minutes()`, and confirm this
        function reproduces `price_schedule_at_minutes`'s result exactly. This is
        the regression guard: existing slot-based results must not move."""
        departures_slots = {"OUT": [0, 2, 4], "RET": [1, 3]}
        requests = {"OUT": [5, 20, 65, 95, 130], "RET": [10, 50, 80]}
        via_slots = price_schedule_at_minutes(
            departures_slots, requests, 30, 15, 60, P_MIN, policy="midpoint"
        )
        given_minutes = {
            d: departure_minutes(taus, 30, "midpoint")
            for d, taus in departures_slots.items()
        }
        via_minutes = price_schedule_given_departure_minutes(
            given_minutes, requests, 15, 60, P_MIN
        )
        self.assertAlmostEqual(via_minutes.total_cost, via_slots.total_cost, places=6)
        self.assertAlmostEqual(
            via_minutes.waiting_minutes, via_slots.waiting_minutes, places=6
        )
        self.assertAlmostEqual(
            via_minutes.unserved_passengers, via_slots.unserved_passengers, places=6
        )

    def test_accepts_departures_no_slot_grid_would_ever_produce(self):
        """The actual capability gained. 37.5 and 82.25 are not `tau*30 + offset`
        for any integer tau and any of the three placement policies -- a schedule
        only a continuous-time model like the CP engine could produce. The old
        slot-only path had no way to even express this input."""
        result = price_schedule_given_departure_minutes(
            {"OUT": [37.5, 82.25], "RET": []},
            {"OUT": [10, 40, 70], "RET": []},
            15,
            60,
            P_MIN,
        )
        self.assertEqual(result.unserved_passengers, 0.0)
        self.assertEqual(result.served_passengers, 3.0)
        # 10 -> 37.5 (27.5), 40 -> 82.25 wins over 37.5 is infeasible (82.25-40=42.25
        # <= 60, 37.5-40 < 0 infeasible); solver picks the cheapest feasible pairing.
        self.assertGreater(result.waiting_minutes, 0.0)

    def test_departures_used_counts_both_directions(self):
        result = price_schedule_given_departure_minutes(
            {"OUT": [10.0, 20.0], "RET": [15.0]},
            {"OUT": [], "RET": []},
            15,
            60,
            P_MIN,
        )
        self.assertEqual(result.departures_used, 3)


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
        """D76: the valid range is symmetric, [-delta, delta] -- anticipation as well as
        the (non-default, counterfactual) positive direction -- so the refused values
        have to sit outside THAT wider range, not merely be negative."""
        from mobauto2_benders.minute_pricer import solve_minute_recourse

        T, delta, S, wmax, p_slots, C_out, C_ret, reqs = self._case()
        with self.assertRaises(ValueError):
            solve_minute_recourse(
                T, delta, wmax, p_slots, C_out, C_ret, reqs,
                lp_solver=require_solver_backend(), placement_offsets=[-delta - 1.0],
            )
        with self.assertRaises(ValueError):
            solve_minute_recourse(
                T, delta, wmax, p_slots, C_out, C_ret, reqs,
                lp_solver=require_solver_backend(), placement_offsets=[delta + 1.0],
            )

    def test_placement_offsets_none_now_defaults_to_start(self):
        """D76: the single-offset default tracks `policy`, which now defaults to
        `start` -- so with neither passed explicitly, the recourse must match the
        anticipate-only default in `_offset_grid` collapsed to its ceiling, offset 0."""
        from mobauto2_benders.minute_pricer import solve_minute_recourse

        T, delta, S, wmax, p_slots, C_out, C_ret, reqs = self._case()
        duals, _obj = solve_minute_recourse(
            T, delta, wmax, p_slots, C_out, C_ret, reqs,
            lp_solver=require_solver_backend(),
        )
        self.assertEqual(duals["placement_offsets"], [0.0])

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


class TestOptimalPlacement(unittest.TestCase):
    """`price_schedule_optimal_placement` chooses the departure INSTANT instead of
    assuming it. It is the newest component here and the one most likely to be wrong,
    so it is pinned three independent ways: against the fixed-policy pricer when the
    choice is removed, against brute-force enumeration when the instance is small
    enough to enumerate, and against the F2 relaxation, which must lower-bound it.

    D76: `tau*delta` (offset 0, `start`) is the master's own committed instant, a
    CEILING, not the low end of a free window -- the genuine degree of freedom is
    anticipation only. The sandwich

        Q_relaxed  <=  Q_optimal  <=  Q_fixed[start]

    holds for the DEFAULT (anticipate-only) grid, because 0 is still its rightmost/
    ceiling element. The wider claim against ALL THREE fixed policies still holds too,
    but only for a grid built to contain all three offsets explicitly -- `start` is no
    longer alone in being guaranteed dominated by an unqualified call.
    """

    @classmethod
    def setUpClass(cls):
        require_solver_backend()

    def _price_opt(self, *args, **kwargs):
        kwargs.setdefault("mip_solver", require_solver_backend())
        return _minute_pricer.price_schedule_optimal_placement(*args, **kwargs)

    def _case(self):
        # delta=4 keeps the default offset grid at five candidates {0,1,2,3,4}, small
        # enough for the enumeration test below to be exhaustive rather than sampled.
        delta, S, wmax, p_min = 4, 3.0, 10.0, 100.0
        sched = {"OUT": [1, 3], "RET": []}
        reqs = {"OUT": [2, 3, 5, 6, 11, 13], "RET": []}
        return delta, S, wmax, p_min, sched, reqs

    def test_a_single_offset_reproduces_that_fixed_policy_exactly(self):
        """With one candidate instant there is no choice left to make, so the MIP must
        return precisely what the fixed-policy LP returns. This is the identity that
        keeps the two code paths from silently diverging."""
        from mobauto2_benders.minute_pricer import placement_offset

        delta, S, wmax, p_min, sched, reqs = self._case()
        for policy in ("start", "midpoint", "end"):
            with self.subTest(policy=policy):
                fixed = price_schedule_at_minutes(
                    sched, reqs, delta, S, wmax, p_min, policy=policy
                )
                opt = self._price_opt(
                    sched, reqs, delta, S, wmax, p_min,
                    offsets=[placement_offset(policy, float(delta))],
                )
                self.assertAlmostEqual(opt.total_cost, fixed.total_cost, places=9)
                self.assertAlmostEqual(
                    opt.unserved_passengers, fixed.unserved_passengers, places=9
                )

    def test_default_optimal_placement_is_never_worse_than_start(self):
        """The upper side of the sandwich for the DEFAULT (anticipate-only) grid.
        `start` (offset 0) is the one fixed policy still guaranteed to be a point in
        that grid, so it is the one none of it can beat."""
        delta, S, wmax, p_min, sched, reqs = self._case()
        opt = self._price_opt(sched, reqs, delta, S, wmax, p_min)
        fixed = price_schedule_at_minutes(
            sched, reqs, delta, S, wmax, p_min, policy="start"
        )
        self.assertLessEqual(opt.total_cost, fixed.total_cost + 1e-9)

    def test_a_grid_spanning_all_three_dominates_every_fixed_policy(self):
        """The general MIP-correctness property this module claims -- optimal
        placement over a set of candidate instants can never lose to any single one of
        them -- checked independently of what the production default happens to be, by
        handing the MIP an explicit grid that contains all three fixed policies."""
        from mobauto2_benders.minute_pricer import placement_offset

        delta, S, wmax, p_min, sched, reqs = self._case()
        grid = sorted({placement_offset(p, float(delta)) for p in ("start", "midpoint", "end")})
        opt = self._price_opt(sched, reqs, delta, S, wmax, p_min, offsets=grid)
        for policy in ("start", "midpoint", "end"):
            fixed = price_schedule_at_minutes(
                sched, reqs, delta, S, wmax, p_min, policy=policy
            )
            self.assertLessEqual(opt.total_cost, fixed.total_cost + 1e-9, policy)

    def test_the_f2_relaxation_lower_bounds_optimal_placement(self):
        """The lower side of the sandwich, and the reason F2 is the only one of the
        three that may generate a cut for the optimal-placement model: it is the only
        one guaranteed to sit at or below the truth."""
        from mobauto2_benders.minute_pricer import solve_minute_recourse

        T, delta, S, wmax, p_slots = 6, 4, 3.0, 10.0, 25.0
        p_min = p_slots * delta
        grid = [0.0, 2.0, 4.0]
        sched = {"OUT": [1, 3], "RET": []}
        reqs = {"OUT": [2, 3, 5, 6, 11, 13], "RET": []}
        C_out = [0.0] * T
        C_out[1] = S
        C_out[3] = S

        _duals, obj_slot_units = solve_minute_recourse(
            T, delta, wmax, p_slots, C_out, [0.0] * T, reqs,
            policy="midpoint", lp_solver=require_solver_backend(),
            placement_offsets=grid,
        )
        relaxed_cost = obj_slot_units * delta
        opt = self._price_opt(sched, reqs, delta, S, wmax, p_min, offsets=grid)
        self.assertLessEqual(relaxed_cost, opt.total_cost + 1e-9)

    def test_it_matches_brute_force_enumeration(self):
        """The MIP is checked against the definition of what it claims to compute:
        enumerate every assignment of instants to departures, price each one with the
        independent fixed-minutes pricer, and take the minimum. The grid is passed
        explicitly to both sides so this is a check of the MIP against the definition,
        not an accidental check against whatever the default grid happens to be.

        `same_slot_eligibility` is passed explicitly for the same reason the grid is.
        The optimal-placement MIP chooses the departure instant, so it prices under
        "allow" (see price_direction_optimal_placement); the fixed-minutes pricer
        defaults to "forbid", which is the recourse convention. Comparing them without
        saying so would be comparing two conventions and calling the difference a MIP
        bug -- which is B6's defect in miniature."""
        import itertools

        delta, S, wmax, p_min, sched, reqs = self._case()
        slots = sched["OUT"]
        grid = [float(k) for k in range(delta + 1)]

        best = float("inf")
        for combo in itertools.product(grid, repeat=len(slots)):
            dep_minutes = [
                float(t) * float(delta) + off for t, off in zip(slots, combo)
            ]
            w, u, _s, _fs, _rows = price_direction_at_minutes(
                reqs["OUT"],
                dep_minutes,
                S,
                wmax,
                p_min,
                same_slot_eligibility="allow",
            )
            best = min(best, w + p_min * u)

        opt = self._price_opt(sched, reqs, delta, S, wmax, p_min, offsets=grid)
        self.assertAlmostEqual(opt.total_cost, best, places=6)

    def test_a_hand_checkable_case(self):
        """One departure in slot 1 (ceiling 10 at delta=10 -- D76: tau*delta, not
        tau*delta+delta, is the committed instant), two passengers at minutes 2 and 8,
        seats enough for both, Wmax generous. Anticipating below 8 strands the
        8-arrival; anything above 8 (up to the ceiling, 10) just adds needless wait to
        both. The cheapest legal instant is 8 itself: (8-2) + (8-8) = 6 passenger-
        minutes -- the same hand-checked total as before D76, now reached by
        anticipating down to the later passenger instead of delaying up to it."""
        delta, S, wmax, p_min = 10, 5.0, 60.0, 500.0
        sched = {"OUT": [1], "RET": []}
        reqs = {"OUT": [2, 8], "RET": []}
        opt = self._price_opt(sched, reqs, delta, S, wmax, p_min)
        self.assertAlmostEqual(opt.total_cost, 6.0, places=6)
        self.assertAlmostEqual(opt.unserved_passengers, 0.0, places=9)

    def test_offsets_outside_the_slot_are_refused(self):
        delta, S, wmax, p_min, sched, reqs = self._case()
        with self.assertRaises(ValueError):
            self._price_opt(sched, reqs, delta, S, wmax, p_min, offsets=[0.0, delta + 1.0])
        with self.assertRaises(ValueError):
            self._price_opt(sched, reqs, delta, S, wmax, p_min, offsets=[-delta - 1.0, 0.0])

    def test_default_grid_anticipates_a_departure_into_its_predecessors_window(self):
        """The concrete scenario D76 was written to fix: demand assigned to a
        departure all arrived at the very start of the PRECEDING slot, and is served
        today at the slot boundary. The optimal placement, searched only over the
        default (anticipate-only) grid, must pull the departure all the way back to
        meet them, not merely to somewhere inside its own slot."""
        delta, S, wmax, p_min = 30, 5.0, 60.0, 500.0
        sched = {"OUT": [1], "RET": []}  # slot 1 -> ceiling (committed instant) = 30
        reqs = {"OUT": [0, 0, 0], "RET": []}  # all arrive at the previous slot's start
        opt = self._price_opt(sched, reqs, delta, S, wmax, p_min)
        # No waiting at all is achievable only by anticipating all the way to minute 0 --
        # the previous slot's own start -- which is outside [0, delta] (the old,
        # incorrect default) but inside [-delta, 0] (the corrected one).
        self.assertAlmostEqual(opt.waiting_minutes, 0.0, places=6)
        self.assertAlmostEqual(opt.unserved_passengers, 0.0, places=9)


class TestOptimalPlacementWithChain(unittest.TestCase):
    """`price_schedule_optimal_placement_with_chain` is the version a vehicle can
    actually fly: departures still choose their instant, but a vehicle's trips must
    follow one another and no trip may eat into a slot its vehicle spends charging.

    It closes the sandwich at four terms rather than three:

        Q_relaxed <= Q_optimal(free) <= Q_optimal(chain) <= Q_fixed[start]

    The right-hand end holds because `start` -- every departure at its slot boundary --
    is always chain-feasible whenever the master's own slot schedule was: a trip that
    fills exactly the slots allocated to it encroaches on nothing and delays nothing.
    That makes `start` the witness that the chain-constrained problem is feasible at
    all, and the reason an infeasible result here would indict the master, not this
    model.
    """

    @classmethod
    def setUpClass(cls):
        require_solver_backend()

    def _chain(self, *args, **kwargs):
        kwargs.setdefault("mip_solver", require_solver_backend())
        return _minute_pricer.price_schedule_optimal_placement_with_chain(*args, **kwargs)

    def _free(self, *args, **kwargs):
        kwargs.setdefault("mip_solver", require_solver_backend())
        return _minute_pricer.price_schedule_optimal_placement(*args, **kwargs)

    def test_one_trip_per_vehicle_and_no_charging_matches_free_shifting(self):
        """With nothing to chain to and nothing to encroach on, the two constraints
        are vacuous and the chain model must reduce to the free one exactly."""
        delta, S, wmax, p_min, dur = 4, 3.0, 10.0, 100.0, 4.0
        reqs = {"OUT": [2, 3, 5, 6, 11, 13], "RET": []}
        # Two vehicles, one trip each -> no precedence pair exists.
        veh = {0: [(1, "OUT")], 1: [(3, "OUT")]}
        chain, _minutes = self._chain(
            veh, {}, reqs, delta, S, wmax, p_min, dur
        )
        free = self._free({"OUT": [1, 3], "RET": []}, reqs, delta, S, wmax, p_min)
        self.assertAlmostEqual(chain.total_cost, free.total_cost, places=6)

    def test_precedence_is_respected_by_the_chosen_minutes(self):
        """The constraint is checked where it is easiest to check honestly: on the
        departure instants the model actually returns.

        D76: requests are shifted one `delta` earlier than the pre-D76 version of this
        test used, to sit inside each trip's ANTICIPATION window (slot tau's legal band
        is now `[(tau-1)*delta, tau*delta]`, not `[tau*delta, (tau+1)*delta]`) -- so the
        scenario still exercises real, servable demand under the corrected default grid
        instead of demand that arrived during the departure's own slot."""
        delta, S, wmax, p_min, dur = 10, 5.0, 60.0, 200.0, 10.0
        reqs = {"OUT": [1, 4, 9], "RET": [15, 18, 23]}
        veh = {0: [(1, "OUT"), (2, "RET")]}
        _res, minutes = self._chain(veh, {}, reqs, delta, S, wmax, p_min, dur)
        d_out = minutes[(0, 1, "OUT")]
        d_ret = minutes[(0, 2, "RET")]
        self.assertGreaterEqual(d_ret + 1e-9, d_out + dur)

    def test_a_charging_slot_pins_the_trip_before_it_to_its_slot_boundary(self):
        """D76: the only freedom left is anticipation, so it is now a charging slot
        immediately PRECEDING a trip that fills its own slot exactly which forbids any
        shift at all -- any anticipation would spill into that preceding slot. The
        departure is pinned to offset 0, which is what `start` already assumes."""
        delta, S, wmax, p_min, dur = 10, 5.0, 60.0, 200.0, 10.0
        reqs = {"OUT": [2, 5, 9], "RET": []}
        veh = {0: [(1, "OUT")]}
        pinned, minutes = self._chain(
            veh, {0: [0]}, reqs, delta, S, wmax, p_min, dur
        )
        self.assertAlmostEqual(minutes[(0, 1, "OUT")], 10.0, places=9)
        at_start = price_schedule_at_minutes(
            {"OUT": [1], "RET": []}, reqs, delta, S, wmax, p_min, policy="start"
        )
        self.assertAlmostEqual(pinned.total_cost, at_start.total_cost, places=6)

        # And without the charging slot the same instance is free to anticipate and do
        # better (best legal instant is 9, the latest of the three arrivals).
        free_to_shift, _m = self._chain(veh, {}, reqs, delta, S, wmax, p_min, dur)
        self.assertLess(free_to_shift.total_cost, pinned.total_cost)

    def test_the_four_term_sandwich_holds(self):
        """Every ordering the module claims, asserted end to end on one instance.

        D76: the grid is anticipate-only (`[-delta, 0]`) and is passed EXPLICITLY to
        `solve_minute_recourse`, `self._free` and `self._chain` alike, so the sandwich
        is checked on a shared, known offset set rather than relying on all three
        functions' defaults to agree by construction (they do, but that identity has
        its own dedicated test elsewhere). Requests are shifted one `delta` earlier
        than the pre-D76 version of this test, to sit inside each trip's legal
        anticipation band instead of its own (same-slot) collection window."""
        from mobauto2_benders.minute_pricer import solve_minute_recourse

        T, delta, S, wmax, p_slots, dur = 8, 10, 5.0, 60.0, 20.0, 10.0
        p_min = p_slots * delta
        reqs = {"OUT": [1, 4, 9, 12, 17], "RET": [25, 31, 34]}
        veh = {0: [(1, "OUT"), (3, "RET")], 1: [(2, "OUT")]}
        sched = {"OUT": [1, 2], "RET": [3]}

        C_out = [0.0] * T
        C_out[1] = S
        C_out[2] = S
        C_ret = [0.0] * T
        C_ret[3] = S
        grid = [float(k) for k in range(-delta, 1)]
        _d, obj_slots = solve_minute_recourse(
            T, delta, wmax, p_slots, C_out, C_ret, reqs, policy="start",
            lp_solver=require_solver_backend(), placement_offsets=grid,
        )
        relaxed = obj_slots * delta
        free = self._free(sched, reqs, delta, S, wmax, p_min, offsets=grid)
        chain, _m = self._chain(veh, {}, reqs, delta, S, wmax, p_min, dur, offsets=grid)
        at_start = price_schedule_at_minutes(
            sched, reqs, delta, S, wmax, p_min, policy="start"
        )

        self.assertLessEqual(relaxed, free.total_cost + 1e-6)
        self.assertLessEqual(free.total_cost, chain.total_cost + 1e-6)
        self.assertLessEqual(chain.total_cost, at_start.total_cost + 1e-6)
