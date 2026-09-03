"""The continuous-time schedule exchange format (B4, handout item, closes report
Comparison C's missing half). REQUIRES AN LP BACKEND; a couple of seconds.

D72 already made `price_schedule_given_departure_minutes` accept departures at
arbitrary minutes -- the piece a continuous-time model (the CP/LBBD engine) needs. What
was missing was a documented format for such a schedule to arrive in, and a loader. This
module tests that loader: a schedule written in the exchange format, loaded and priced
through the minute path, must reproduce a cost computed independently through the
existing slot path -- the same schedule, the same units, two different routes to the
LP `price_schedule_given_departure_minutes`/`price_schedule_at_minutes` both call.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import _helpers  # noqa: F401  (puts src/ on sys.path)
from _helpers import require_solver_backend
from mobauto2_benders.continuous_schedule import (
    ChargingEvent,
    load_continuous_schedule,
    parse_continuous_schedule,
    price_continuous_schedule,
    to_departures_minutes,
)
from mobauto2_benders.minute_pricer import departure_minutes, price_schedule_at_minutes

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "continuous_schedule_roundtrip.yaml"

# The same instance the fixture file's own header derives its minutes from.
DEPARTURES_SLOTS = {"OUT": [1, 2, 4, 6], "RET": [2, 3, 5, 7]}
REQUESTS = {"OUT": [10, 35, 50, 90, 150, 200], "RET": [40, 70, 100, 160]}
SEATS = 15
WMAX_MINUTES = 60
P_MINUTES = 56.0


class TestRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = require_solver_backend()

    def test_reproduces_a_cost_computed_independently_through_the_slot_path(self):
        """The round trip: load the file, price it via the minute path, and check
        against a cost computed independently -- built from the same slots the
        fixture's own comment says it was converted from, through the slot path
        (price_schedule_at_minutes), not through any code this module shares with
        the loader. Two different routes to the same answer is the point (R8)."""
        known = price_schedule_at_minutes(
            DEPARTURES_SLOTS, REQUESTS, 30, SEATS, WMAX_MINUTES, P_MINUTES,
            policy="start", lp_solver=self.backend,
        )

        schedule = load_continuous_schedule(FIXTURE)
        priced = price_continuous_schedule(
            schedule, REQUESTS, WMAX_MINUTES, P_MINUTES, lp_solver=self.backend
        )

        self.assertAlmostEqual(priced.total_cost, known.total_cost, places=6)
        self.assertAlmostEqual(priced.waiting_minutes, known.waiting_minutes, places=6)
        self.assertAlmostEqual(
            priced.unserved_passengers, known.unserved_passengers, places=6
        )
        self.assertAlmostEqual(
            priced.served_passengers, known.served_passengers, places=6
        )

    def test_flattening_discards_vehicle_identity_but_keeps_every_departure(self):
        """Two vehicles in the file; the flattened pool must hold every departure
        from both, order aside -- a dropped departure would silently understate
        capacity and nothing downstream would notice."""
        schedule = load_continuous_schedule(FIXTURE)
        flat = to_departures_minutes(schedule)
        self.assertEqual(sorted(flat["OUT"]), [30.0, 60.0, 120.0, 180.0])
        self.assertEqual(sorted(flat["RET"]), [60.0, 90.0, 150.0, 210.0])

    def test_charging_plan_is_carried_through_but_not_priced(self):
        """A continuous-time schedule is not fully specified without its charging
        plan (a vehicle's departures must be feasible against its own charge
        state); the loader keeps it even though pricing does not consume it."""
        schedule = load_continuous_schedule(FIXTURE)
        v0 = next(v for v in schedule.vehicles if v.id == "0")
        self.assertEqual(v0.charging_plan, (ChargingEvent(165.0, 195.0),))

    def test_minutes_match_the_slot_conversion_this_fixture_documents(self):
        """The fixture's own comment claims it is {1,2,4,6}/{2,3,5,7} at delta=30,
        policy=start. Check that claim directly rather than trusting the comment."""
        schedule = load_continuous_schedule(FIXTURE)
        flat = to_departures_minutes(schedule)
        expected = {
            d: sorted(departure_minutes(taus, 30, "start"))
            for d, taus in DEPARTURES_SLOTS.items()
        }
        self.assertEqual(sorted(flat["OUT"]), expected["OUT"])
        self.assertEqual(sorted(flat["RET"]), expected["RET"])


class TestValidation(unittest.TestCase):
    """Fail-closed: malformed input raises, it is never coerced or silently dropped."""

    def _base(self) -> dict:
        return {
            "schema": "mobauto2_continuous_schedule",
            "version": 1,
            "horizon_minutes": 660,
            "seats": 15,
            "vehicles": [
                {
                    "id": 0,
                    "departures": {"OUT": [30.0], "RET": []},
                    "charging_plan": [],
                }
            ],
        }

    def test_wrong_schema_name_is_rejected(self):
        raw = self._base()
        raw["schema"] = "something_else"
        with self.assertRaises(ValueError):
            parse_continuous_schedule(raw)

    def test_wrong_version_is_rejected(self):
        raw = self._base()
        raw["version"] = 2
        with self.assertRaises(ValueError):
            parse_continuous_schedule(raw)

    def test_departure_outside_horizon_is_rejected(self):
        raw = self._base()
        raw["vehicles"][0]["departures"]["OUT"] = [700.0]
        with self.assertRaises(ValueError):
            parse_continuous_schedule(raw)

    def test_duplicate_vehicle_id_is_rejected(self):
        raw = self._base()
        raw["vehicles"].append(dict(raw["vehicles"][0]))
        with self.assertRaises(ValueError):
            parse_continuous_schedule(raw)

    def test_unknown_direction_is_rejected(self):
        raw = self._base()
        raw["vehicles"][0]["departures"]["DIAGONAL"] = [10.0]
        with self.assertRaises(ValueError):
            parse_continuous_schedule(raw)

    def test_charging_event_with_end_before_start_is_rejected(self):
        raw = self._base()
        raw["vehicles"][0]["charging_plan"] = [
            {"start_minute": 100.0, "end_minute": 50.0}
        ]
        with self.assertRaises(ValueError):
            parse_continuous_schedule(raw)

    def test_empty_vehicle_list_is_rejected(self):
        raw = self._base()
        raw["vehicles"] = []
        with self.assertRaises(ValueError):
            parse_continuous_schedule(raw)


if __name__ == "__main__":
    unittest.main()
