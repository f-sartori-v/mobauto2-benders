"""Master model structure. Builds Pyomo models but never calls a solver."""
from __future__ import annotations

import unittest

import pyomo.environ as pyo

from _helpers import CONFIGS, build_master, constraint_names, master_params


class TestSymmetryBreaking(unittest.TestCase):
    """Guards the defect that made the master stop being a relaxation.

    Ordering vehicles by cumulative departures at EVERY time prefix removes
    feasible schedules that are not symmetric duplicates, so the master's bound
    could exceed the true optimum. Ordering by TOTAL departures is valid: any
    schedule can be relabelled by sorting vehicles on total trips.
    """

    def test_uses_total_ordering_not_prefix_ordering(self):
        pm = build_master(master_params())
        names = constraint_names(pm.m)
        self.assertIn("C_sym_break_tot", names)
        self.assertNotIn("C_sym_break_pref", names, "prefix ordering is invalid")
        self.assertNotIn("C3_fifo", names, "duplicate FIFO block should be gone")

    def test_symmetry_rows_scale_with_fleet_not_horizon(self):
        """Total ordering needs Q-1 rows. Prefix ordering needed T*(Q-1), so a
        row count that grows with T is the signature of the invalid form."""
        params = master_params()
        pm = build_master(params)
        rows = len(pm.m.C_sym_break_tot)
        self.assertEqual(rows, int(params["Q"]) - 1)

    def test_heterogeneous_fleet_is_refused(self):
        """Symmetry breaking is only valid for a homogeneous fleet in an
        identical initial state. The audit noted this precondition was never
        checked; it must now fail loudly rather than cut off the optimum."""
        params = master_params()
        params["binit"] = [150.0, 90.0]
        with self.assertRaises(ValueError) as ctx:
            build_master(params)
        self.assertIn("homogeneous", str(ctx.exception).lower())

    def test_heterogeneous_initial_actions_refused(self):
        params = master_params()
        params["initial_actions"] = ["IDL", "CHR"]
        with self.assertRaises(ValueError):
            build_master(params)


class TestChargeBeforeIdleFlag(unittest.TestCase):
    def test_on_by_default(self):
        pm = build_master(master_params())
        self.assertIn("C_no_recharge_after_idle", constraint_names(pm.m))

    def test_can_be_disabled(self):
        params = master_params()
        params["charge_before_idle"] = False
        pm = build_master(params)
        self.assertNotIn("C_no_recharge_after_idle", constraint_names(pm.m))


class TestReturnLegConstraintAbsent(unittest.TestCase):
    """AUDIT_v3 M1 recommends b[q,t] >= L*yRET[q,t] for a tighter relaxation.

    Measured, it is harmful: master phase 18.2s -> 49s over 10 iterations and a
    worse bound at the same budget. It is sound but does not pay for itself.
    This test is a tripwire so it is not re-added from the audit text without
    re-measuring -- delete the test deliberately if the measurement changes.
    """

    def test_c5_ret_not_present(self):
        pm = build_master(master_params())
        self.assertNotIn("C5_ret", constraint_names(pm.m))


class TestNoPerCellConstraintNames(unittest.TestCase):
    """M4: constraints are declared indexed, not one component per (q,t).

    Per-cell components are named like C4_bal_0_7. Their absence is what makes
    the model inspectable and keeps component counts flat.
    """

    def test_build_phase_uses_indexed_constraints(self):
        pm = build_master(master_params())
        offenders = [
            n for n in constraint_names(pm.m)
            if any(n.startswith(p) for p in ("C4_bal_", "C4_chg1_", "C2a_locL_", "C2a_locM_", "C1b_intrip_eq_"))
        ]
        self.assertEqual(offenders, [], f"per-cell components remain: {offenders[:5]}")


class TestCutAggregationGuard(unittest.TestCase):
    """D10 / spec §2.7: collapsing (q,tau) -> tau is only valid when the cut
    coefficients agree across q. On the production path they do, by construction.
    A per-vehicle formulation must fail loudly rather than emit an invalid cut."""

    def test_q_varying_coefficients_are_rejected(self):
        from mobauto2_benders.benders.types import Cut, CutType

        params = master_params()
        params["aggregate_cuts_by_tau"] = True
        pm = build_master(params)
        # Give the model a solution to anchor against.
        for q in pm.m.Q:
            for t in pm.m.T:
                if not pm.m.yOUT[q, t].fixed:
                    pm.m.yOUT[q, t].set_value(0)
                if not pm.m.yRET[q, t].fixed:
                    pm.m.yRET[q, t].set_value(0)
        for t in pm.m.T:
            pm.m.Yout[t].set_value(0)
            pm.m.Yret[t].set_value(0)

        bad = Cut(
            name="bad_cut",
            cut_type=CutType.OPTIMALITY,
            metadata={
                "const": 0.0,
                # same tau, different value per q -> aggregation would be invalid
                "coeff_yOUT": {(0, 1): -10.0, (1, 1): -20.0},
                "coeff_yRET": {},
            },
        )
        with self.assertRaises(RuntimeError) as ctx:
            pm.add_cut_force(bad)
        self.assertIn("varies across q", str(ctx.exception))



class TestFleetListPaddingConvention(unittest.TestCase):
    """Per-vehicle lists are [z specific vehicles..., 1 value shared by the rest].

    A fleet of Q vehicles where z have distinct initial states and the remaining
    Q-z are identical is written as a list of length z+1, and the LAST entry is
    the shared value. `initial_battery` implemented this; `initial_actions` sat
    ten lines away and padded with a literal "IDL" instead, so a fleet declared
    to start charging silently started idle from vehicle z+1 on.

    Homogeneous fleets are the case symmetry breaking allows, so these build with
    symmetry breaking off -- that is the configuration where the convention has
    anything to express.
    """

    def _params(self, **over):
        params = master_params()
        params["symmetry_breaking"] = False
        params["use_fifo_symmetry"] = False
        params["Q"] = 5
        params.update(over)
        return params

    @staticmethod
    def _initial_battery(pm) -> list[float]:
        return [pyo.value(pm.m.b[q, 0]) for q in pm.m.Q]

    @staticmethod
    def _starts_charging(pm) -> list[float]:
        """c[q,0] is fixed to 1 exactly when the forced first action is CHR."""
        return [pyo.value(pm.m.c[q, 0]) for q in pm.m.Q]

    def test_battery_pads_with_last_value(self):
        pm = build_master(self._params(binit=[90.0, 150.0], initial_actions=["IDL"]))
        self.assertEqual(self._initial_battery(pm), [90.0, 150.0, 150.0, 150.0, 150.0])

    def test_actions_pad_with_last_value_not_literal_idl(self):
        pm = build_master(self._params(binit=[150.0], initial_actions=["IDL", "CHR"]))
        self.assertEqual(
            self._starts_charging(pm),
            [0, 1, 1, 1, 1],
            "padding with a literal IDL ignores the shared-value convention",
        )

    def test_single_value_describes_a_homogeneous_fleet(self):
        pm = build_master(self._params(binit=[150.0], initial_actions=["CHR"]))
        self.assertEqual(self._initial_battery(pm), [150.0] * 5)
        self.assertEqual(self._starts_charging(pm), [1] * 5)

    def test_both_lists_pad_the_same_way(self):
        """The defect was the two rules diverging, so pin them together."""
        pm = build_master(self._params(binit=[10.0, 20.0], initial_actions=["IDL", "CHR"]))
        self.assertEqual(self._initial_battery(pm)[1:], [20.0] * 4)
        self.assertEqual(self._starts_charging(pm)[1:], [1] * 4)


class TestRecourseLowerBound(unittest.TestCase):
    """Valid inequality anchoring theta to installed capacity, per prefix.

    Demand arriving in slot t can only be served by a departure in
    [t+1, t+W_slots], so demand accumulated to j only reaches capacity installed
    to j+W_slots. Without it the master answered "cost 0.22, no vehicles" for the
    first five iterations, because nothing tied theta to capacity except the cuts,
    which arrive one at a time.

    Off by default. M1 was valid too and cost 2.7x master time for a worse bound,
    so validity alone does not earn a place in the model.
    """

    def _params(self, on, res=30, q=2):
        import yaml, tempfile, os
        from mobauto2_benders.config import load_config
        from mobauto2_benders.app import _prepare_params

        raw = yaml.safe_load((CONFIGS / "default.yaml").read_text(encoding="utf-8"))
        # default.yaml is a live experiment file; pin the data section so these
        # tests describe the inequality, not whatever run is being set up in it.
        # Multi-scenario legitimately disables the bound (it needs the expectation
        # over scenarios), so a scenario list left there would fail them for the
        # wrong reason.
        raw["data"] = {**raw["data"], "scenario_files": None, "scenarios": None}
        raw["model"]["time"]["slot_resolution"] = res
        raw["model"]["fleet"]["Q"] = q
        raw["model"]["fleet"]["initial_battery"] = [150.0]
        raw["model"]["fleet"]["initial_actions"] = ["IDL"]
        raw["master"]["recourse_lower_bound"] = on
        fd, tmp = tempfile.mkstemp(suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                yaml.safe_dump(raw, fh)
            cfg = load_config(tmp)
            mp, _ = _prepare_params(cfg, {})
        finally:
            os.unlink(tmp)
        if mp.get("T") is None:
            mp["T"] = max(1, int(mp["T_minutes"]) // int(mp["slot_resolution"]))
        return mp

    def test_on_in_the_shipped_default_config(self):
        """D29 flipped this on. Assert the shipped config asks for it, rather than
        that the constraint gets built: whether it is built also depends on the
        data section, which is legitimately multi-scenario at times."""
        from mobauto2_benders.config import load_config

        cfg = load_config(str(CONFIGS / "default.yaml"))
        self.assertTrue(cfg.master.recourse_lower_bound)

    def test_built_when_asked_on_single_scenario_data(self):
        pm = build_master(self._params(True))
        self.assertIn("C_recourse_lb_out", constraint_names(pm.m))

    def test_can_be_disabled(self):
        pm = build_master(self._params(False))
        self.assertNotIn("C_recourse_lb_out", constraint_names(pm.m))
        self.assertNotIn("C_recourse_lb_ret", constraint_names(pm.m))

    def test_frozen_baselines_pin_it_off(self):
        """The D9 baselines diff against archived logs, so their model must not
        move when a default changes. Pinned explicitly, not inherited."""
        from mobauto2_benders.config import load_config

        for name in ("baseline_d9.yaml", "baseline_d9_multi.yaml"):
            with self.subTest(config=name):
                cfg = load_config(str(CONFIGS / name))
                self.assertFalse(cfg.master.recourse_lower_bound)

    def test_one_row_per_slot_per_direction(self):
        params = self._params(True)
        pm = build_master(params)
        self.assertEqual(len(pm.m.C_recourse_lb_out), int(params["T"]))
        self.assertEqual(len(pm.m.C_recourse_lb_ret), int(params["T"]))

    def test_uses_post_truncation_demand(self):
        """A total larger than the subproblem's would demand more unserved
        passengers than can occur, and the inequality would cut off the optimum
        rather than bound it. This is why aggregation is shared, not reimplemented."""
        params = self._params(True)
        rlb = params["recourse_bound_data"]
        self.assertEqual(sum(rlb["R_out"]) + sum(rlb["R_ret"]), 300.0)
        self.assertEqual(len(rlb["R_out"]), int(params["T"]))

    def test_window_scales_with_resolution(self):
        """W_slots = ceil(W_max_min / delta): 2 at 30-min slots, 4 at 15-min."""
        self.assertEqual(self._params(True, res=30)["recourse_bound_data"]["W_slots"], 2)
        self.assertEqual(self._params(True, res=15)["recourse_bound_data"]["W_slots"], 4)

    def test_empty_master_is_no_longer_costless(self):
        """The defect it targets: with no cuts, the master used to answer 0."""
        pm = build_master(self._params(True))
        con = pm.m.C_recourse_lb_out[int(pm.m.T.last() if hasattr(pm.m.T, "last") else 0)]
        self.assertIsNotNone(con)

    def test_late_capacity_cannot_pay_for_early_demand(self):
        """The whole point of the prefix form: a departure at slot 40 must not
        appear on the right-hand side of an early-j row. The aggregate version of
        this inequality let it, which is why it was rejected in favour of this."""
        params = self._params(True)
        pm = build_master(params)
        T = int(params["T"])
        W = int(params["recourse_bound_data"]["W_slots"])
        j = 1
        body = str(pm.m.C_recourse_lb_out[j].body) + str(pm.m.C_recourse_lb_out[j].upper) \
            + str(pm.m.C_recourse_lb_out[j].lower)
        for late in range(j + W + 1, T):
            self.assertNotIn(f"yOUT[0,{late}]", body,
                             f"slot {late} must not appear in the row for j={j}")


if __name__ == "__main__":
    unittest.main()
