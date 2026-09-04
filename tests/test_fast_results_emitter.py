"""The emitter's refusals, and the horizon bound. No solver.

Each test here corresponds to a number that was published and later had to be
withdrawn. They are refusals rather than warnings on purpose: a warning next to a
wrong number is still a wrong number in the table.

Run just these:
    python -m unittest tests.test_fast_results_emitter -v
"""

from __future__ import annotations

import unittest

import _helpers  # noqa: F401  (puts src/ on sys.path)

from mobauto2_benders.results_emitter import (
    MANIFEST_FIELDS,
    ManifestMismatch,
    MissingManifest,
    RunRecord,
    StatusMismatch,
    Table,
    manifest_id,
    median_trajectory,
    performance_profile,
    proof_and_gap,
    time_ratio,
)


class TestManifestRequired(unittest.TestCase):
    """`manifest_required`. A table without a manifest id is not written."""

    def test_manifest_required(self):
        t = Table(title="competitiveness", columns=("arm", "obj"))
        t.add(arm="benders", obj=4183.24)
        with self.assertRaises(MissingManifest) as ctx:
            t.render()
        self.assertIn("manifest", str(ctx.exception).lower())

    def test_a_manifest_is_enough_to_render(self):
        t = Table(title="competitiveness", columns=("arm", "obj"), manifest="abc123")
        t.add(arm="benders", obj=4183.24)
        self.assertIn("abc123", t.render())

    def test_legacy_numbers_need_a_label_AND_the_manifest_they_came_from(self):
        """B9. A table that could not be regenerated keeps its numbers and says so.

        The label alone is not enough. "not rerun" tells a reader that someone knows
        the row is stale; naming the manifest tells them WHICH regime produced it,
        which is the thing they need in order to decide whether it is comparable with
        anything else on the page.
        """
        t = Table(
            title="penalty frontier",
            columns=("p", "obj"),
            legacy_label="not rerun: 40-minute budget, out of scope today",
        )
        t.add(p=50, obj=4183.24)
        with self.assertRaises(MissingManifest):
            t.render()
        t.legacy_manifest = "legacy_p1500"
        rendered = t.render()
        self.assertIn("NOT REGENERATED", rendered)
        self.assertIn("legacy_p1500", rendered)

    def test_one_manifest(self):
        t = Table(title="runtime split", columns=("arm", "s"), manifest="m1")
        t.add(arm="a", s=1.0, manifest="m1")
        t.add(arm="b", s=2.0, manifest="m2")
        with self.assertRaises(ManifestMismatch) as ctx:
            t.render()
        self.assertIn("m2", str(ctx.exception))

    def test_a_row_may_not_contradict_the_table(self):
        t = Table(title="runtime split", columns=("arm",), manifest="m1")
        t.add(arm="a", manifest="m9")
        with self.assertRaises(ManifestMismatch):
            t.render()


class TestManifestId(unittest.TestCase):
    def test_the_contract_fields_are_all_present(self):
        """The shared contract with Agent CP-LBBD names these, in this order."""
        for name in (
            "H", "delta", "Q", "S", "Emax", "b0", "c_trip", "rho", "tau_trip",
            "Wmax", "p_min", "epsilon", "kappa", "K_chg", "o",
            "same_slot_eligibility",
        ):
            self.assertIn(name, MANIFEST_FIELDS)
        self.assertEqual(MANIFEST_FIELDS[:5], ("H", "delta", "Q", "S", "Emax"))

    def test_a_missing_field_does_not_collide_with_a_present_one(self):
        """An absent field is hashed as null, not skipped.

        Skipping would let a run that never set `K_chg` share an id with one that set
        it explicitly, and the id's entire purpose is that two rows sharing it were
        produced under the same conditions.
        """
        a = manifest_id({"H": 660, "Q": 2})
        b = manifest_id({"H": 660, "Q": 2, "K_chg": 2})
        self.assertNotEqual(a, b)

    def test_the_id_moves_with_the_eligibility_convention(self):
        base = {"H": 660, "delta": 1, "same_slot_eligibility": "forbid"}
        other = dict(base, same_slot_eligibility="allow")
        self.assertNotEqual(manifest_id(base), manifest_id(other))

    def test_the_id_is_stable(self):
        fields = {"H": 660, "delta": 30, "Q": 2}
        self.assertEqual(manifest_id(fields), manifest_id(dict(fields)))


class TestRatioRefusedOnStatusMismatch(unittest.TestCase):
    """`ratio_refused_on_status_mismatch`. The "390x slower" claim, blocked."""

    def setUp(self):
        self.proved = RunRecord(
            label="minute monolith",
            manifest="m1",
            status="OPTIMAL",
            wall_time_s=0.8,
            lower_bound=293.37,
            upper_bound=293.37,
        )
        self.truncated = RunRecord(
            label="benders minute",
            manifest="m1",
            status="MAX_TIME",
            wall_time_s=301.4,
            clock_truncated_master_solves=3,
            lower_bound=219.74,
            upper_bound=299.37,
        )

    def test_ratio_refused_on_status_mismatch(self):
        with self.assertRaises(StatusMismatch) as ctx:
            time_ratio(self.proved, self.truncated)
        msg = str(ctx.exception)
        self.assertIn("did not terminate the same way", msg)
        self.assertIn("proof_and_gap", msg)

    def test_the_refusal_also_fires_on_censoring_alone(self):
        """A run marked OPTIMAL that stopped a master solve on the clock reached that
        status through a machine-load dependent search. Its wall time is one draw."""
        censored_optimal = RunRecord(
            label="benders",
            manifest="m1",
            status="OPTIMAL",
            wall_time_s=120.0,
            clock_truncated_master_solves=1,
        )
        with self.assertRaises(StatusMismatch):
            time_ratio(self.proved, censored_optimal)

    def test_a_ratio_between_two_proofs_is_computed(self):
        """The guard must not refuse everything, or it guards nothing."""
        slow_proof = RunRecord(
            label="benders", manifest="m1", status="OPTIMAL", wall_time_s=8.0
        )
        self.assertAlmostEqual(time_ratio(self.proved, slow_proof), 10.0, places=9)

    def test_the_honest_pair_is_emitted_instead(self):
        pair = proof_and_gap(self.truncated)
        self.assertIsNone(
            pair["time_to_proof_s"],
            "an arm that never proved optimality has no time-to-proof; reporting its "
            "wall clock under that name is the claim being withdrawn",
        )
        self.assertTrue(pair["censored"])
        self.assertGreater(pair["terminal_gap"], 0.2)
        self.assertEqual(pair["wall_time_s"], 301.4)


class TestCensoringRatherThanDiscarding(unittest.TestCase):
    """B10. The 228-iteration run is kept and labelled, not thrown away."""

    def test_a_censored_run_is_still_a_row(self):
        t = Table(title="convergence", columns=("run", "gap"), manifest="m1")
        t.add(run="228-iteration", gap=0.118, censored=True, manifest="m1")
        t.add(run="short", gap=0.47, censored=False, manifest="m1")
        rendered = t.render()
        self.assertIn("228-iteration", rendered)
        self.assertIn("censored rows: 1 of 2", rendered)

    def test_censored_is_derived_from_the_truncation_count(self):
        self.assertFalse(RunRecord("a", "m", "OPTIMAL").censored)
        self.assertTrue(
            RunRecord("a", "m", "OPTIMAL", clock_truncated_master_solves=1).censored
        )

    def test_median_trajectory_summarises_repetitions(self):
        traj = median_trajectory([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
        self.assertEqual(traj, [2.0, 3.0, 4.0])

    def test_median_trajectory_truncates_rather_than_extrapolating(self):
        """Extending a short run's trajectory would invent iterations it never ran."""
        traj = median_trajectory([[1.0, 2.0, 3.0], [2.0, 3.0]])
        self.assertEqual(len(traj), 2)

    def test_performance_profile_includes_censored_runs(self):
        records = [
            RunRecord("r1", "m", "OPTIMAL", wall_time_s=10.0),
            RunRecord("r2", "m", "MAX_TIME", wall_time_s=20.0,
                      clock_truncated_master_solves=2),
            RunRecord("r3", "m", "OPTIMAL", wall_time_s=40.0),
        ]
        profile = dict(performance_profile(records, ratios=(1.0, 2.0, 4.0)))
        self.assertAlmostEqual(profile[1.0], 1 / 3)
        self.assertAlmostEqual(profile[2.0], 2 / 3)
        self.assertAlmostEqual(profile[4.0], 1.0)


class TestHorizonNecessaryBound(unittest.TestCase):
    """`horizon_necessary_bound`. The 14Q cap, and what it does and does not say."""

    def setUp(self):
        import importlib.util
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[1] / "scripts" / "trip_cap_450.py"
        )
        spec = importlib.util.spec_from_file_location("trip_cap_450", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.cap = module.horizon_trip_cap

    def test_horizon_necessary_bound(self):
        """At H=660 with the shipped parameters, one vehicle caps at 14 trips.

        k*30 + (60/70)*max(0, 30k - 150) <= 660.
          k=14 -> 420 + 0.857*270 = 651.4 <= 660   feasible
          k=15 -> 450 + 0.857*300 = 707.1 >  660   not
        """
        self.assertEqual(self.cap(660, 30, 30, 150, 70), 14)

    def test_the_bound_is_monotone_in_the_horizon(self):
        caps = [self.cap(H, 30, 30, 150, 70) for H in (660, 780, 900)]
        self.assertEqual(caps, sorted(caps))
        self.assertGreater(caps[-1], caps[0])

    def test_the_initial_charge_buys_free_trips(self):
        """b0 = 150 at 30 per trip is five trips before any recharge is forced."""
        self.assertGreater(self.cap(150, 30, 30, 150, 70), 4)

    def test_a_faster_charger_raises_the_cap(self):
        self.assertGreater(
            self.cap(660, 30, 30, 150, 140), self.cap(660, 30, 30, 150, 70)
        )


if __name__ == "__main__":
    unittest.main()
