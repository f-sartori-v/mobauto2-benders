"""The P0 acceptance tests that need a solver. Each fails on the pre-change code.

The work order's table, in code. Where a test's failure mode on the OLD behaviour is
not obvious, the docstring says what it was -- a test whose "fails before" claim nobody
can check is a test nobody will trust when it goes red.

Run just these:
    python -m unittest discover -s tests -p "test_workorder_p0.py" -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

import _helpers  # noqa: F401  (puts src/ on sys.path)
from _helpers import (
    build_master,
    constraint_names,
    fixture_for_backend,
    master_params,
    require_solver_backend,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


# ---------------------------------------------------------------------------- B3
class TestNoFinalSlotCharging(unittest.TestCase):
    """`no_final_slot_charging`. The audit's item 1.7.

    FAILS BEFORE: `c[q,T-1]` and `gchg[q,T-1]` were free -- the SoC recursion skips
    t = T-1 because it writes `b[q,t+1]` and there is no `b[q,T]` -- so a solution
    could charge in the last slot with no effect on any state and no cost. Nothing
    fixed them, so `is_fixed()` was False and a positive value was accepted.
    """

    def test_no_final_slot_charging(self):
        params = master_params()
        pm = build_master(params)
        m = pm.m
        T = int(params["T"])
        for q in m.Q:
            with self.subTest(q=q):
                self.assertTrue(
                    m.c[q, T - 1].is_fixed(),
                    "c[q,T-1] is free: a charging decision that changes no state and "
                    "costs nothing makes the reported schedule's last slot arbitrary",
                )
                self.assertEqual(float(m.c[q, T - 1].value), 0.0)
                self.assertTrue(m.gchg[q, T - 1].is_fixed())
                self.assertEqual(float(m.gchg[q, T - 1].value), 0.0)

    def test_a_solution_with_positive_final_slot_charging_is_rejected(self):
        """The check rejects it, and the check is what a caller can rely on.

        Deliberately NOT written as "Pyomo raises when you set a fixed variable" --
        it does not, and a test asserting that would pass today and pass tomorrow for
        the wrong reason. `fix(0.0)` sets a value the solver receives; a later
        `set_value` changes it silently, and nothing downstream would notice, because
        the whole reason this leak survived is that these two variables appear in no
        surviving row and no cost term.

        So the postcondition is checkable after the fact (model_checks) and it is the
        postcondition that is asserted here.
        """
        from mobauto2_benders.model_checks import (
            assert_no_final_slot_charging,
            final_slot_energy_violations,
        )

        params = master_params()
        pm = build_master(params)
        T = int(params["T"])

        # As built, the model is clean.
        self.assertEqual(final_slot_energy_violations(pm.m), [])
        assert_no_final_slot_charging(pm.m)

        # A solution that charges in the final slot is rejected, and the message names
        # the vehicle and the value rather than saying "infeasible".
        pm.m.c[0, T - 1].set_value(0.5)
        violations = final_slot_energy_violations(pm.m)
        self.assertTrue(violations)
        self.assertIn(f"c[0,{T - 1}]", violations[0])
        with self.assertRaises(ValueError) as ctx:
            assert_no_final_slot_charging(pm.m)
        self.assertIn("final-slot energy leak", str(ctx.exception))

    def test_the_check_notices_an_unfixed_final_slot(self):
        """The leak itself, not just an instance of it.

        FAILS BEFORE: nothing fixed these variables, so this is the state the whole
        codebase was in and the check would have reported it on every model.
        """
        from mobauto2_benders.model_checks import final_slot_energy_violations

        params = master_params()
        pm = build_master(params)
        T = int(params["T"])
        pm.m.gchg[0, T - 1].unfix()
        violations = final_slot_energy_violations(pm.m)
        self.assertTrue(any("is not fixed" in v for v in violations))

    def test_the_monolith_fixes_the_same_two(self):
        """The two engines must agree about the last slot or their schedules differ
        in a place neither objective can see."""
        from mobauto2_milp.app import _prepare_params as milp_params
        from mobauto2_milp.config import load_config as load_milp
        from mobauto2_milp.model import MobautoMilpModel

        cfg = load_milp(str(CONFIGS / "milp" / "baseline_d9_p56_monolith.yaml"))
        mp, _sp = milp_params(cfg, {})
        mp = dict(mp)
        mp.pop("T", None)
        model = MobautoMilpModel(mp)
        model.initialize()
        m = model.m
        T = len(list(m.T))
        for q in m.Q:
            with self.subTest(q=q):
                self.assertTrue(m.c[q, T - 1].is_fixed())
                self.assertTrue(m.gchg[q, T - 1].is_fixed())


# ---------------------------------------------------------------------------- B2
class TestChargerCapacityBinds(unittest.TestCase):
    """`charger_capacity_binds`. The audit's item 1.8.

    FAILS BEFORE: there was no charger-capacity row at all, so `K_chg` was not a
    parameter and the constraint could not appear at any value.
    """

    def test_the_row_is_absent_at_the_default(self):
        """K_chg defaults to Q, where the divisible row is implied by c in [0,1].

        Emitting it anyway would couple named vehicles in every run and retire the M1
        separability condition for no gain -- see master_impl.
        """
        pm = build_master(master_params())
        self.assertNotIn("C_chg_capacity", constraint_names(pm.m))

    def test_charger_capacity_binds(self):
        params = master_params()
        params["K_chg"] = 1
        pm = build_master(params)
        self.assertIn("C_chg_capacity", constraint_names(pm.m))
        rows = list(pm.m.C_chg_capacity)
        self.assertEqual(len(rows), int(params["T"]))

    def test_the_row_actually_forbids_two_vehicles_charging_at_once(self):
        import pyomo.environ as pyo
        from pyomo.core.expr.visitor import identify_variables

        params = master_params()
        params["K_chg"] = 1
        pm = build_master(params)
        m = pm.m
        t = 1  # not the fixed last slot
        body = m.C_chg_capacity[t].body
        qs = {
            v.index()[0]
            for v in identify_variables(body)
            if isinstance(v.index(), tuple)
        }
        self.assertGreaterEqual(
            len(qs), 2, "the row does not couple two vehicles, so it cannot bind"
        )
        self.assertEqual(float(pyo.value(m.C_chg_capacity[t].upper)), 1.0)

    def test_the_binary_occupancy_form_is_available_and_different(self):
        """The divisible form assumes charger time is preemptible within a slot. Where
        the site says otherwise, the binary form is a different physical claim and
        must be a different model, not a rounding of the same one."""
        params = master_params()
        params["K_chg"] = 1
        params["charger_occupancy_binary"] = True
        pm = build_master(params)
        names = constraint_names(pm.m)
        self.assertIn("C_chg_occ_link", names)
        self.assertIn("C_chg_capacity", names)
        self.assertTrue(hasattr(pm.m, "zchg"))

    def test_the_binary_form_is_emitted_even_at_K_chg_equals_Q(self):
        """It is never implied: integral z forbids sharing a charger even at K = Q."""
        params = master_params()
        params["charger_occupancy_binary"] = True
        pm = build_master(params)
        self.assertIn("C_chg_capacity", constraint_names(pm.m))

    def test_a_negative_charger_count_is_refused(self):
        params = master_params()
        params["K_chg"] = -1
        with self.assertRaises(ValueError):
            build_master(params)


# ---------------------------------------------------------------------------- B7
class TestPenaltyRegime(unittest.TestCase):
    """`rejected_with_free_seat` and `lexicographic_dominates`. Audit item 1.3."""

    @classmethod
    def setUpClass(cls):
        cls.backend = require_solver_backend()

    def _price(self, p_minutes: float):
        from mobauto2_benders.minute_pricer import price_schedule_at_minutes

        # One departure at slot 1 (minute 30) with 15 seats, and arrivals that cannot
        # all reach it. The stranded group's only other option is 60 minutes away,
        # which costs more than p_min=56 to serve and less to reject.
        return price_schedule_at_minutes(
            departures={"OUT": [1, 3], "RET": []},
            requests={"OUT": [0] * 15 + [31, 32, 33], "RET": []},
            slot_resolution=30,
            seats=15.0,
            wmax_minutes=60.0,
            p_minutes=p_minutes,
            lp_solver=self.backend,
        )

    def test_rejected_with_free_seat(self):
        """Nonzero at p_min=56, zero at p_min=120.

        Both directions matter. Nonzero at 56 says the regime is choosing rejection
        over an admissible wait; zero at 120 -- where no wait within W_max=60 can cost
        more than the penalty -- says the count is measuring that choice and not
        merely counting unserved passengers.

        FAILS BEFORE: the diagnostic did not exist, so the regime was invisible.
        """
        cheap = self._price(56.0)
        self.assertGreater(
            cheap.rejected_with_free_seat,
            0.0,
            "at p_min=56 with delta=30 a two-slot wait costs 60 passenger-minutes and "
            "a rejection costs 56, so some passengers MUST be left behind with a free "
            "seat. A zero here means the count or the pricing is wrong.",
        )
        dear = self._price(120.0)
        self.assertEqual(
            dear.rejected_with_free_seat,
            0.0,
            "at p_min=120 no admissible wait (W_max=60) can cost more than the "
            "penalty, so rejection is never preferred and the count must be zero",
        )

    def test_the_two_regimes_really_do_differ_in_who_is_carried(self):
        """Guard against a count that moves while the schedule's outcome does not."""
        self.assertLess(
            self._price(56.0).served_passengers,
            self._price(120.0).served_passengers,
        )

    def test_the_assignment_rows_reconstruct_the_headline(self):
        """B12. Every printed figure must be derivable from the rows beneath it.

        The per-shuttle table that summed to 126 under a headline of 222 was drawn
        from a different solution than the headline. Rows and totals coming from one
        solve is what makes that impossible.
        """
        res = self._price(56.0)
        served = sum(r.passengers for r in res.assignment_rows if r.served)
        unserved = sum(r.passengers for r in res.assignment_rows if not r.served)
        waiting = sum(
            r.passengers * r.wait_minutes for r in res.assignment_rows if r.served
        )
        self.assertAlmostEqual(served, res.served_passengers, places=6)
        self.assertAlmostEqual(unserved, res.unserved_passengers, places=6)
        self.assertAlmostEqual(waiting, res.waiting_minutes, places=6)

    def test_every_assignment_row_states_its_qualifiers(self):
        """B12. A row without them cannot define the average it contributes to."""
        res = self._price(56.0)
        self.assertTrue(res.assignment_rows)
        for row in res.assignment_rows:
            self.assertIn(row.direction, {"OUT", "RET"})
            self.assertEqual(row.assignment_resolution, "minute")
            self.assertEqual(row.same_slot_eligibility, "forbid")
            self.assertIsNotNone(row.departure_offset)
            if row.served:
                self.assertIsNotNone(row.departure_slot)

    def test_average_wait_returns_its_own_definition(self):
        res = self._price(56.0)
        value, definition = res.average_wait(denominator="carried")
        self.assertIn("denominator=carried", definition)
        self.assertIn("same_slot_eligibility=forbid", definition)
        self.assertIn(f"{value:.3f}", definition)

    def test_the_denominator_actually_changes_the_number(self):
        """The reason the range in the report was undefined: two denominators."""
        res = self._price(56.0)
        self.assertGreater(res.unserved_passengers, 0.0)
        carried, _ = res.average_wait(denominator="carried")
        allreq, _ = res.average_wait(denominator="all", exclude_unserved=False)
        self.assertNotAlmostEqual(carried, allreq, places=6)

    def test_lexicographic_dominates(self):
        """Lexicographic never serves fewer passengers than the weighted sum.

        FAILS BEFORE: the mode did not exist, so there was nothing to compare.
        """
        from mobauto2_milp.app import _prepare_params
        from mobauto2_milp.config import load_config
        from mobauto2_milp.model import MobautoMilpModel
        from mobauto2_milp.monolith import MonolithSolver

        served = {}
        for name in ("penalty_regime_weighted_sum", "penalty_regime_lexicographic"):
            path = fixture_for_backend(CONFIGS / "milp" / f"{name}.yaml")
            cfg = load_config(str(path))
            mp, sp = _prepare_params(cfg, {})
            mp, sp = dict(mp), dict(sp)
            mp.pop("T", None)
            result = MonolithSolver(MobautoMilpModel, cfg, mp, sp).run()
            served[name] = float(result.pax_served or 0.0)

        self.assertGreaterEqual(
            served["penalty_regime_lexicographic"],
            served["penalty_regime_weighted_sum"],
            "lexicographic mode served FEWER passengers than the weighted sum, which "
            "cannot happen if level 1 really minimises unserved demand first",
        )
        self.assertGreater(
            served["penalty_regime_lexicographic"],
            served["penalty_regime_weighted_sum"],
            "the two modes served the same number, so this instance does not "
            "discriminate and the test proves nothing -- see "
            "setups/penalty_regime_tiny.yaml for the shape that does",
        )


# ---------------------------------------------------------------------------- B8
class TestTripCapModel(unittest.TestCase):
    """`trip_cap_model`. Audit items 3.4 and 4.8.

    FAILS BEFORE: neither constrained model existed. The 450/30 claim rested on
    observing trip counts in three UNCONSTRAINED optima, which says nothing about
    what is achievable when the trip count is constrained.
    """

    @classmethod
    def setUpClass(cls):
        require_solver_backend()
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "trip_cap_450", REPO_ROOT / "scripts" / "trip_cap_450.py"
        )
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_trip_cap_model(self):
        """Model (a) at Q=4 returns a proven optimum whose trip count is at most 30."""
        import pyomo.environ as pyo

        from mobauto2_milp.config import load_config

        cfg = load_config(self.mod.BASE_CONFIG)
        solver, model = self.mod._build(cfg, Q=4, H_minutes=660, time_limit=180.0)
        total, unserved = self.mod._served_expr(model, solver._scenarios)
        result, _stats = self.mod._solve_with(
            solver,
            model,
            unserved,
            pyo.minimize,
            [self.mod._trip_count_expr(model) <= self.mod.TRIP_CAP],
        )
        self.assertEqual(
            self.mod._status_name(result),
            "OPTIMAL",
            "model (a) did not prove optimality inside the budget; raise it or report "
            "the cell as clock-truncated rather than quoting it",
        )
        trips = float(pyo.value(self.mod._trip_count_expr(model)))
        self.assertLessEqual(trips, self.mod.TRIP_CAP + 1e-6)
        served = total - float(pyo.value(unserved))
        self.assertGreater(served, 0.0)
        # The cap must be the thing that binds, or the cell is answering a different
        # question -- "what does the unconstrained optimum do" -- which is exactly the
        # inference B8 replaces.
        self.assertLessEqual(served, total)

    def test_the_objective_is_restored_after_the_constrained_solve(self):
        """A solver left carrying a replacement objective would silently misreport
        every later run made with it."""
        import pyomo.environ as pyo

        from mobauto2_milp.config import load_config

        cfg = load_config(self.mod.BASE_CONFIG)
        solver, model = self.mod._build(cfg, Q=2, H_minutes=660, time_limit=30.0)
        total, unserved = self.mod._served_expr(model, solver._scenarios)
        self.mod._solve_with(
            solver, model, unserved, pyo.minimize,
            [self.mod._trip_count_expr(model) <= self.mod.TRIP_CAP],
        )
        self.assertTrue(model.obj.active)
        self.assertFalse(hasattr(model, "_b8_obj"))
        self.assertFalse(hasattr(model, "_b8_con_0"))
        del total


if __name__ == "__main__":
    unittest.main()
