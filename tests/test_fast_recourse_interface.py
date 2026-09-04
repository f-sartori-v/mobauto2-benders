"""B5. The recourse INTERFACE: slot-indexed capacity, minute-indexed demand. No solver.

The audit's item 1.6. Equation (e2) as drafted defined a minute-indexed capacity
`C_d[m]` reached through an incidence matrix. The code never built that -- but nothing
asserted the difference, and the two readings are not cosmetic variants of each other:

  * MINUTE-indexed capacity would yield one dual per minute. The master's cut is
    written `theta >= const + sum_tau S * pi_d[tau] * Y_d[tau]`, one term per
    departure SLOT. A per-minute dual has no home in it, so substituting the minute
    recourse for the slot recourse would require changing the cut space, the master's
    rows, and every aggregation and validity check downstream.
  * SLOT-indexed capacity, which is what the code builds, yields exactly `T` duals per
    direction whatever the demand resolution. That is the reason the minute recourse
    can stand in for the slot recourse and the master notices nothing.

So this file pins the interface, not the arithmetic: how many capacity rows exist, what
they are indexed by, and that the demand side is free to be as fine as the data are.

It also carries E2's companion check -- that the structural fingerprint DOES move when
the demand or the penalty moves. A fingerprint that never changes is not evidence of
invariance; it is evidence of a broken fingerprint.

Run just these:
    python -m unittest tests.test_fast_recourse_interface -v
"""

from __future__ import annotations

import unittest

import _helpers  # noqa: F401  (puts src/ on sys.path)

from mobauto2_benders.recourse_fingerprint import (
    capacity_rhs,
    capacity_rows_are_slot_indexed,
    recourse_fingerprint,
    slot_recourse_model_for,
)


T = 8
S_SEATS = 15.0
WMAX_SLOTS = 2
P_SLOTS = 1.867
SLOT_RES = 30

# Arrivals spread across minutes INSIDE slots, not on slot boundaries. If the capacity
# rows were minute-indexed, this is the demand that would give them away: 12 distinct
# arrival minutes against 8 departure slots.
ARRIVAL_MINUTES = {
    "OUT": [5, 17, 33, 41, 62, 77, 95, 101, 128, 133, 150, 161],
    "RET": [12, 25, 48, 55, 70, 88, 99, 110, 121, 140, 155, 170],
}


def _sp(p_slots: float = P_SLOTS) -> dict:
    return {
        "T": T,
        "S": S_SEATS,
        "Wmax_slots": WMAX_SLOTS,
        "p": p_slots,
        "slot_resolution": SLOT_RES,
        "lp_solver": "cplex_direct",
    }


def _slot_demand(scale: float = 1.0) -> tuple[list[float], list[float]]:
    r_out = [scale * v for v in [0, 12, 30, 25, 18, 22, 9, 0]]
    r_ret = [scale * v for v in [0, 8, 16, 28, 31, 14, 11, 0]]
    return r_out, r_ret


class TestSlotRecourseCapacityRowsAreSlotIndexed(unittest.TestCase):
    def setUp(self):
        r_out, r_ret = _slot_demand()
        self.Y_out = [0, 1, 1, 0, 2, 1, 0, 0]
        self.Y_ret = [0, 0, 1, 1, 1, 0, 1, 0]
        self.model = slot_recourse_model_for(
            _sp(), self.Y_out, self.Y_ret, r_out, r_ret
        )

    def test_capacity_rows_are_slot_indexed(self):
        ok, why = capacity_rows_are_slot_indexed(self.model, T)
        self.assertTrue(ok, why)

    def test_there_is_at_most_one_capacity_row_per_direction_per_slot(self):
        for name in ("Cap_out", "Cap_ret"):
            con = getattr(self.model, name)
            self.assertLessEqual(
                sum(1 for _ in con),
                T,
                f"{name} has more rows than there are departure slots, so its dual is "
                "not the per-slot pi the master's cut is written in",
            )

    def test_the_capacity_right_hand_side_is_exactly_S_times_the_signature(self):
        """The one channel `y` may use, and it must carry exactly `S * Y_d[tau]`.

        Anything else -- a rounding, a `max(1, ...)`, a per-vehicle layer count --
        would mean the cut's slope `S*pi` is not the recourse's subgradient.
        """
        rhs = capacity_rhs(self.model)
        for tau in range(T):
            for name, sig in (("Cap_out", self.Y_out), ("Cap_ret", self.Y_ret)):
                key = f"{name}[{tau}]"
                if key not in rhs:
                    # Slots with no incident arc carry no row at all, which is the
                    # correct treatment of a capacity that binds nothing.
                    continue
                with self.subTest(row=key):
                    self.assertAlmostEqual(rhs[key], S_SEATS * sig[tau], places=9)


class TestFingerprintMovesWithTheDataThatShouldMoveIt(unittest.TestCase):
    """E2's companion. An invariance check is only worth what its converse is worth."""

    def setUp(self):
        self.r_out, self.r_ret = _slot_demand()
        self.Y_out = [0, 1, 1, 0, 2, 1, 0, 0]
        self.Y_ret = [0, 0, 1, 1, 1, 0, 1, 0]
        self.base = slot_recourse_model_for(
            _sp(), self.Y_out, self.Y_ret, self.r_out, self.r_ret
        )

    def test_the_schedule_alone_does_not_change_the_fingerprint(self):
        other = slot_recourse_model_for(
            _sp(), [0, 2, 0, 1, 1, 1, 0, 0], [0, 1, 1, 0, 1, 1, 0, 0],
            self.r_out, self.r_ret,
        )
        self.assertEqual(
            recourse_fingerprint(self.base),
            recourse_fingerprint(other),
            "the fingerprint moved with the SCHEDULE, which is `y` entering the "
            "constraint matrix -- the D30 failure",
        )
        self.assertNotEqual(
            capacity_rhs(self.base),
            capacity_rhs(other),
            "two different schedules produced the same capacity vector, so this "
            "comparison proves nothing",
        )

    def test_perturbing_demand_changes_the_fingerprint(self):
        r_out, r_ret = _slot_demand(scale=1.5)
        other = slot_recourse_model_for(
            _sp(), self.Y_out, self.Y_ret, r_out, r_ret
        )
        self.assertNotEqual(
            recourse_fingerprint(self.base),
            recourse_fingerprint(other),
            "the demand rows' right-hand side changed and the fingerprint did not, "
            "so the fingerprint is not reading the non-capacity right-hand sides",
        )

    def test_perturbing_the_penalty_changes_the_fingerprint(self):
        other = slot_recourse_model_for(
            _sp(p_slots=P_SLOTS * 3.0),
            self.Y_out,
            self.Y_ret,
            self.r_out,
            self.r_ret,
        )
        self.assertNotEqual(
            recourse_fingerprint(self.base),
            recourse_fingerprint(other),
            "the objective coefficients changed and the fingerprint did not, so the "
            "fingerprint is not reading the objective",
        )


class TestMinuteRecourseKeepsSlotIndexedCapacity(unittest.TestCase):
    """The substitution B5 exists to license, checked on the real builder."""

    def setUp(self):
        import pyomo.environ as pyo

        from mobauto2_benders.minute_pricer import attach_minute_recourse

        m = pyo.ConcreteModel()
        m.T = range(T)
        m.Yout = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.Yret = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.theta = pyo.Var(within=pyo.NonNegativeReals)
        attach_minute_recourse(
            m,
            requests=ARRIVAL_MINUTES,
            slot_resolution=SLOT_RES,
            seats=S_SEATS,
            wmax_minutes=60.0,
            p_minutes=56.0,
        )
        self.model = m

    def test_capacity_rows_stay_slot_indexed(self):
        ok, why = capacity_rows_are_slot_indexed(self.model, T)
        self.assertTrue(ok, why)

    def test_demand_rows_are_minute_indexed_and_far_more_numerous(self):
        """The asymmetry IS the interface.

        If both sides were slot-indexed this would be the slot model with extra steps;
        if both were minute-indexed the dual would be per-minute and the cut would not
        fit the master. One row per arrival minute, one row per departure slot.
        """
        n_dem_out = sum(1 for _ in self.model.MinDemandOut)
        n_cap_out = sum(1 for _ in self.model.MinCapOut)
        self.assertEqual(n_dem_out, len(set(ARRIVAL_MINUTES["OUT"])))
        self.assertLessEqual(n_cap_out, T)
        self.assertGreater(
            n_dem_out,
            n_cap_out,
            "the demand side is no finer than the capacity side, so this instance "
            "cannot tell a minute-indexed capacity path from a slot-indexed one",
        )

    def test_no_capacity_row_is_indexed_by_a_minute(self):
        """A minute-indexed capacity row would name a minute in its index.

        Checked by name rather than by count alone, so a demand file that happened to
        have exactly T distinct arrival minutes could not make the count test pass
        vacuously.
        """
        minutes = {int(m) for v in ARRIVAL_MINUTES.values() for m in v} - set(range(T))
        for name in ("MinCapOut", "MinCapRet"):
            con = getattr(self.model, name)
            for idx in con:
                # index is (scenario, tau)
                tau = int(idx[-1])
                with self.subTest(row=f"{name}[{idx}]"):
                    self.assertLess(tau, T)
                    self.assertNotIn(tau, minutes)


if __name__ == "__main__":
    unittest.main()
