"""S4: all four recourse-proxy shapes exist, and the default did not move.

No solver. The formal formulation's recommended baseline (12) is the
scenario-direction multi-cut, `theta[omega,d]`, "the strongest clean baseline". Until S4
it was inexpressible: the master computed `disagg_dir = False if theta_per_scenario`, so
the two disaggregations were mutually exclusive by construction and one cell of D11's
A/B did not exist.

The other half of this file is the part that matters more day to day: **the default did
not move.** `disaggregate_theta_by_direction` was read from a key no config could set and
was hardcoded true, so exposing it risks silently changing every existing
`theta_per_scenario: true` run from |Omega| proxies to 2*|Omega|. That is the
inert-configuration pattern (AUDIT_v4 3.8) with the sign flipped, and the tests below pin
the resolution rather than trusting the comment that describes it.
"""

from __future__ import annotations

import unittest

import _helpers  # noqa: F401  -- puts src/ on sys.path
from _helpers import CONFIGS, build_master, master_params


def _params(theta_per_scenario, theta_by_direction, num_scenarios=3):
    mp = master_params("baseline_d9.yaml")
    mp["theta_per_scenario"] = theta_per_scenario
    mp["num_scenarios"] = num_scenarios
    if theta_by_direction is None:
        # What app.py resolves when the config is silent.
        mp["disaggregate_theta_by_direction"] = not theta_per_scenario
    else:
        mp["disaggregate_theta_by_direction"] = theta_by_direction
    return mp


def _shape(model) -> str:
    if hasattr(model, "theta_out_s") and hasattr(model, "theta_ret_s"):
        return "by_scenario_direction"
    if hasattr(model, "theta_s"):
        return "by_scenario"
    if hasattr(model, "theta_out") and hasattr(model, "theta_ret"):
        return "by_direction"
    if hasattr(model, "theta"):
        return "single"
    return "none"


class AllFourShapesAreReachable(unittest.TestCase):
    def test_single(self):
        m = build_master(_params(False, False)).m
        self.assertEqual(_shape(m), "single")

    def test_by_direction(self):
        m = build_master(_params(False, True)).m
        self.assertEqual(_shape(m), "by_direction")

    def test_by_scenario(self):
        m = build_master(_params(True, False)).m
        self.assertEqual(_shape(m), "by_scenario")

    def test_by_scenario_direction(self):
        """The formulation's recommended baseline (12). Inexpressible before S4."""
        m = build_master(_params(True, True)).m
        self.assertEqual(_shape(m), "by_scenario_direction")

    def test_the_proxy_count_is_two_per_scenario(self):
        m = build_master(_params(True, True, num_scenarios=4)).m
        self.assertEqual(len(list(m.theta_out_s)), 4)
        self.assertEqual(len(list(m.theta_ret_s)), 4)

    def test_per_scenario_shapes_need_scenarios(self):
        """With num_scenarios = 0 the per-scenario request has nothing to index, and
        silently building |Omega|=0 proxies would leave the objective with no recourse
        term at all -- a master whose optimum is the trip cost alone."""
        for by_dir in (False, True):
            m = build_master(_params(True, by_dir, num_scenarios=0)).m
            with self.subTest(theta_by_direction=by_dir):
                self.assertIn(_shape(m), ("single", "by_direction"))


class TheDefaultDidNotMove(unittest.TestCase):
    """Pre-S4 behaviour: theta_per_scenario false -> by_direction, true -> by_scenario."""

    def test_a_silent_config_without_scenarios_gives_the_directional_pair(self):
        m = build_master(_params(False, None)).m
        self.assertEqual(_shape(m), "by_direction")

    def test_a_silent_per_scenario_config_keeps_its_old_shape(self):
        """The regression this file exists for. Before S4 the direction split was FORCED
        OFF whenever per-scenario thetas were on; a config that does not mention
        theta_by_direction must still get |Omega| proxies, not 2*|Omega|."""
        m = build_master(_params(True, None)).m
        self.assertEqual(_shape(m), "by_scenario")

    def test_app_resolves_the_default_that_way(self):
        """Asserted against app.py's own resolution rather than the helper above, so the
        two cannot drift."""
        from mobauto2_benders.app import _prepare_params
        from mobauto2_benders.config import load_config

        for name in ("baseline_d9.yaml", "baseline_d9_multi.yaml"):
            cfg = load_config(CONFIGS / name)
            mp, _sp = _prepare_params(cfg, {})
            with self.subTest(config=name):
                self.assertEqual(
                    bool(mp["disaggregate_theta_by_direction"]),
                    not bool(cfg.master.theta_per_scenario),
                    "a shipped config changed theta shape on this commit",
                )

    def test_no_shipped_config_asks_for_the_new_shape(self):
        """It is opt-in. If one of these ever does, its numbers are on a different model
        and the table quoting them has to say so."""
        from mobauto2_benders.config import load_config

        # Two kinds of file are exempt, both by intent rather than by accumulation:
        #
        #   default.yaml           untracked live experiment file
        #   theta_* / d64/theta_*  configs whose PURPOSE is to select a theta shape --
        #                          the smoke config for the cut routing and the D64 A/B
        #                          cells. Their headers say so, and D64 quotes their
        #                          numbers as being on different models by construction.
        #
        # Everything else is a config whose numbers are published under the assumption of
        # its pre-S4 shape. One of those appearing here means the model moved silently.
        def is_theta_experiment(p) -> bool:
            return p.stem.startswith("theta_") or p.parent.name == "d64"

        for path in sorted(CONFIGS.rglob("*.yaml")):
            if path.name == "default.yaml" or is_theta_experiment(path):
                continue
            try:
                cfg = load_config(path)
            except Exception:
                continue  # milp configs and malformed scratch files are not our business
            if not hasattr(cfg, "master"):
                continue
            with self.subTest(config=path.name):
                self.assertFalse(
                    bool(cfg.master.theta_per_scenario)
                    and bool(cfg.master.theta_by_direction),
                    f"{path.name} selects the scenario-direction shape",
                )


class TheAnchorFollowsTheShape(unittest.TestCase):
    """The recourse anchor (D29) must be written against the proxies that exist.

    Under the scenario-direction shape it is also strictly tighter: each `theta_out_s[s]`
    is bounded by scenario `s`'s OWN unserved cost rather than by the weighted mean's.
    Bounding a weighted sum by the mean is weaker than bounding each term by its own,
    by Jensen -- and on the shipped multi-scenario instance the two scenarios differ
    (150 vs 103 OUT passengers against a mean of 126.5), so this is not a distinction
    without a difference.
    """

    CONFIG = CONFIGS / "phase1" / "theta_sd_smoke.yaml"

    @classmethod
    def setUpClass(cls):
        import os

        from mobauto2_benders.app import _prepare_params
        from mobauto2_benders.config import load_config

        cls._cwd = os.getcwd()
        os.chdir(_helpers.REPO_ROOT)
        cfg = load_config(cls.CONFIG)
        cls.mp, cls.sp = _prepare_params(cfg, {})
        cls.mp["T"] = cls.mp.get("T") or int(cls.mp["T_minutes"]) // int(
            cls.mp["slot_resolution"]
        )

    @classmethod
    def tearDownClass(cls):
        import os

        os.chdir(cls._cwd)

    def test_the_payload_carries_per_scenario_demand(self):
        rlb = self.mp["recourse_bound_data"]
        self.assertIn("R_out_by_scenario", rlb)
        self.assertEqual(len(rlb["R_out_by_scenario"]), int(rlb["num_scenarios"]))
        self.assertEqual(len(rlb["R_ret_by_scenario"]), int(rlb["num_scenarios"]))

    def test_the_scenarios_actually_differ(self):
        """Otherwise the per-scenario anchor is the mean anchor and proves nothing."""
        rlb = self.mp["recourse_bound_data"]
        totals = [sum(v) for v in rlb["R_out_by_scenario"]]
        self.assertGreater(
            max(totals) - min(totals), 1.0, f"scenario OUT totals are ~equal: {totals}"
        )

    def test_the_mean_lies_between_the_scenarios(self):
        """Sanity on the payload: a mean outside the range would mean the weights or the
        aggregation are wrong, and the anchor would be built on the wrong demand."""
        rlb = self.mp["recourse_bound_data"]
        totals = [sum(v) for v in rlb["R_out_by_scenario"]]
        self.assertGreaterEqual(sum(rlb["R_out"]), min(totals) - 1e-9)
        self.assertLessEqual(sum(rlb["R_out"]), max(totals) + 1e-9)

    def test_the_anchor_rows_are_the_scenario_direction_ones(self):
        import pyomo.environ as pyo

        pm = build_master(self.mp)
        names = {
            c.name
            for c in pm.m.component_objects(pyo.Constraint)
            if "recourse_lb" in c.name
        }
        self.assertEqual(names, {"C_recourse_lb_sd_out", "C_recourse_lb_sd_ret"})
        self.assertEqual(getattr(pm, "_anchor_demand_source", None), "per_scenario")

    def test_there_is_one_row_per_scenario_direction_and_prefix(self):
        pm = build_master(self.mp)
        S = int(self.mp["recourse_bound_data"]["num_scenarios"])
        T = int(self.mp["T"])
        self.assertEqual(len(pm.m.C_recourse_lb_sd_out), S * T)
        self.assertEqual(len(pm.m.C_recourse_lb_sd_ret), S * T)

    def test_the_empty_master_is_not_costless_under_this_shape(self):
        """The property `test_empty_master_is_no_longer_costless` asserts for the older
        shapes. Without it here, the anchor would be silently absent under the newest
        shape and the first master solve would return 0 -- which is what it did before
        the sd branch existed."""
        import pyomo.environ as pyo

        pm = build_master(self.mp)
        m = pm.m
        for q in m.Q:
            for t in m.T:
                m.yOUT[q, t].fix(0)
                m.yRET[q, t].fix(0)
        for s in m.Scenarios:
            m.theta_out_s[s].set_value(0.0)
            m.theta_ret_s[s].set_value(0.0)

        # Measure violation without assuming which side Pyomo put theta on. It
        # normalises `theta >= p*(...)` to `p*(...) <= theta`, i.e. body <= upper with
        # upper = 0 -- so a check written against `.lower` reads 0 on every row and
        # would pass for any anchor at all, including none.
        def violation(con) -> float:
            body = pyo.value(con.body, exception=False)
            if body is None:
                return 0.0
            body = float(body)
            worst = 0.0
            lo = pyo.value(con.lower, exception=False)
            if lo is not None:
                worst = max(worst, float(lo) - body)
            up = pyo.value(con.upper, exception=False)
            if up is not None:
                worst = max(worst, body - float(up))
            return worst

        worst = 0.0
        for s in m.Scenarios:
            for j in m.T:
                worst = max(
                    worst,
                    violation(m.C_recourse_lb_sd_out[s, j]),
                    violation(m.C_recourse_lb_sd_ret[s, j]),
                )
        self.assertGreater(
            worst,
            1.0,
            "an idle master with theta = 0 satisfies every anchor row, so the anchor "
            "is absent or vacuous under this shape",
        )

    def test_the_anchor_row_carries_this_scenarios_demand_not_the_mean(self):
        """The reason the sd anchor is tighter, asserted numerically.

        With every `y` fixed to 0 the OUT row for scenario `s` at the last prefix reduces
        to `theta_out_s[s] >= p * total_OUT_demand(s)`. Read that constant back and check
        it is the scenario's own total, not the weighted mean -- which is what the
        aggregated shapes are limited to.
        """
        import pyomo.environ as pyo

        pm = build_master(self.mp)
        m = pm.m
        for q in m.Q:
            for t in m.T:
                m.yOUT[q, t].fix(0)
                m.yRET[q, t].fix(0)
        for s in m.Scenarios:
            m.theta_out_s[s].set_value(0.0)
            m.theta_ret_s[s].set_value(0.0)

        rlb = self.mp["recourse_bound_data"]
        p = float(rlb["p"])
        last = int(self.mp["T"]) - 1
        for s in m.Scenarios:
            implied = abs(float(pyo.value(m.C_recourse_lb_sd_out[s, last].body)))
            own = p * sum(rlb["R_out_by_scenario"][int(s)])
            with self.subTest(scenario=int(s)):
                self.assertAlmostEqual(implied, own, places=6)
        # And the two scenarios' rows must differ, or the shape bought nothing.
        rows = [
            abs(float(pyo.value(m.C_recourse_lb_sd_out[s, last].body)))
            for s in m.Scenarios
        ]
        self.assertGreater(max(rows) - min(rows), 1.0)


class ScenarioWeightsAreAppliedOnce(unittest.TestCase):
    """Handout 87, Failure 5: never apply rho twice.

    The objective used the raw weights while the recourse anchor divided by their sum, so
    a config whose weights did not sum to 1 gave the two different notions of
    expectation. `_scenario_weights` is now the single reader and refuses that config.
    """

    def _master(self, weights, S=3):
        mp = _params(True, False, num_scenarios=S)
        mp["scenario_weights"] = weights
        return build_master(mp)

    def test_normalised_weights_are_accepted_unchanged(self):
        pm = self._master([0.5, 0.25, 0.25])
        self.assertEqual(pm._scenario_weights(3), [0.5, 0.25, 0.25])

    def test_weights_that_do_not_sum_to_one_are_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self._master([0.5, 0.25, 0.9])
        self.assertIn("not 1", str(ctx.exception))

    def test_a_negative_probability_is_refused(self):
        with self.assertRaises(ValueError):
            self._master([1.5, -0.25, -0.25])

    def test_absent_weights_fall_back_to_uniform(self):
        pm = self._master(None, S=4)
        self.assertEqual(pm._scenario_weights(4), [0.25] * 4)

    def test_wrong_length_falls_back_to_uniform(self):
        """A truncated list is a config error, but uniform is the honest reading of "the
        weights do not describe these scenarios" and matches the pre-S4 behaviour."""
        pm = self._master([0.5, 0.5], S=3)
        self.assertEqual(len(pm._scenario_weights(3)), 3)


if __name__ == "__main__":
    unittest.main()
