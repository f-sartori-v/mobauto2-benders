"""The safety fixes: mode resolution, the MW/minute refusal, and NO_CUT aggregation.

No solver. Each test here pins a defect that was live in the tree, not a hypothetical:

  * `dual` was UNREACHABLE by configuration -- every shipped config set both legacy
    booleans true and the dispatch is `if mw ... elif dual ...` (AUDIT_v4 3.5).
  * `finite_difference` was the FALL-THROUGH of that pair, so a config omitting both
    selected the one mode with no lower-bound guarantee, silently, and only found out
    after spending its budget.
  * Magnanti-Wong with the minute recourse duals a DIFFERENT LP than the primal being
    solved -- the D30 failure mode -- and nothing refused the combination.
  * A scenario whose theta early-exit fired poisoned the multi-scenario aggregate to
    `unknown`, dropping the run's lower bound (D64 4b).
"""

from __future__ import annotations

import unittest

import _helpers  # noqa: F401  -- puts src/ on sys.path
from _helpers import CONFIGS

from mobauto2_benders.problem.subproblem_impl import (
    CUT_MODE_VALID_LOWER_BOUND,
    resolve_cut_mode,
)


class CutModeResolution(unittest.TestCase):
    def test_cut_mode_wins_when_present(self):
        for mode in ("mw", "dual", "finite_difference"):
            with self.subTest(mode=mode):
                self.assertEqual(resolve_cut_mode({"cut_mode": mode}), mode)

    def test_legacy_pair_keeps_its_historical_precedence(self):
        """mw, then dual, then finite differences -- unchanged, so hand-built params
        in tests and scripts behave exactly as before."""
        self.assertEqual(
            resolve_cut_mode({"use_magnanti_wong": True, "use_dual_slopes": True}), "mw"
        )
        self.assertEqual(
            resolve_cut_mode({"use_magnanti_wong": False, "use_dual_slopes": True}),
            "dual",
        )
        self.assertEqual(resolve_cut_mode({}), "finite_difference")

    def test_an_unimplemented_mode_raises(self):
        with self.assertRaises(RuntimeError):
            resolve_cut_mode({"cut_mode": "logic_based"})

    def test_every_resolvable_mode_has_a_validity_entry(self):
        for params in ({"cut_mode": "mw"}, {"cut_mode": "dual"}, {}):
            self.assertIn(resolve_cut_mode(params), CUT_MODE_VALID_LOWER_BOUND)


class ConfigRefusals(unittest.TestCase):
    """Load-time refusals. Each replaces a failure that used to surface hours later."""

    def _cfg(self, sub_overrides: dict, tmpname: str):
        import yaml
        from pathlib import Path

        from mobauto2_benders.config import load_config

        raw = yaml.safe_load(
            (CONFIGS / "baseline_d9.yaml").read_text(encoding="utf-8-sig")
        )
        sub = raw.setdefault("subproblem", {})
        # baseline_d9.yaml ships BOTH legacy booleans, so a test that sets `cut_mode`
        # would trip the both-forms refusal rather than exercise what it means to.
        # Dropping them here is what a config author would do when migrating.
        if "cut_mode" in sub_overrides:
            sub.pop("use_magnanti_wong", None)
            sub.pop("use_dual_slopes", None)
        sub.update(sub_overrides)
        out = Path(_helpers.REPO_ROOT) / "configs" / tmpname
        out.write_text(yaml.safe_dump(raw), encoding="utf-8")
        try:
            return load_config(out)
        finally:
            out.unlink(missing_ok=True)

    def test_mw_with_minute_recourse_is_allowed(self):
        """B2 (handout item, 2026-09-02): this combination used to be refused at
        load, because solve_mw_dual built only the dual of the SLOT primal, a
        different LP from the minute recourse's own. solve_mw_dual_minute
        (minute_pricer.py) is that LP's own dual, so the combination is no longer
        refused -- see tests/test_minute_mw.py for the port's own soundness
        checks."""
        cfg = self._cfg(
            {
                "use_magnanti_wong": True,
                "recourse_resolution": "minute",
                "Wmax_minutes": 60,
            },
            "_tmp_mw_minute.yaml",
        )
        self.assertEqual(cfg.subproblem.cut_mode, "mw")

    def test_plain_dual_with_minute_recourse_is_allowed(self):
        """The valid combination must stay reachable, or the refusal has removed the
        minute recourse rather than guarded it."""
        cfg = self._cfg(
            {
                "use_magnanti_wong": False,
                "use_dual_slopes": True,
                "recourse_resolution": "minute",
                "Wmax_minutes": 60,
            },
            "_tmp_dual_minute.yaml",
        )
        self.assertEqual(cfg.subproblem.cut_mode, "dual")

    def test_finite_difference_is_refused_without_acknowledgement(self):
        with self.assertRaises(ValueError) as ctx:
            self._cfg(
                {"use_magnanti_wong": False, "use_dual_slopes": False},
                "_tmp_fdiff.yaml",
            )
        self.assertIn("no lower-bound guarantee", str(ctx.exception))

    def test_finite_difference_runs_when_acknowledged(self):
        cfg = self._cfg(
            {
                "use_magnanti_wong": False,
                "use_dual_slopes": False,
                "acknowledge_no_lower_bound": True,
            },
            "_tmp_fdiff_ok.yaml",
        )
        self.assertEqual(cfg.subproblem.cut_mode, "finite_difference")
        self.assertFalse(CUT_MODE_VALID_LOWER_BOUND[cfg.subproblem.cut_mode])

    def test_setting_both_forms_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self._cfg(
                {"cut_mode": "dual", "use_magnanti_wong": True},
                "_tmp_both_forms.yaml",
            )
        self.assertIn("two forms", str(ctx.exception))

    def test_cut_mode_alone_selects_the_previously_unreachable_dual(self):
        """The whole point of S1b. `dual` could not be selected before: every shipped
        config set both booleans and mw wins the precedence."""
        cfg = self._cfg({"cut_mode": "dual"}, "_tmp_cutmode_dual.yaml")
        self.assertEqual(cfg.subproblem.cut_mode, "dual")

    def test_shipped_configs_still_load(self):
        """The refusals must not reject the tree's own TRACKED configs.

        Tracked only, deliberately. `configs/sweep/` and `configs/bench/` are gitignored
        scratch and carry configs that already fail to load for unrelated reasons --
        `cap102.yaml` still sets `fill_first_epsilon`, deleted in D30. Asserting over
        those would make this test fail for the state of somebody's scratch directory
        rather than for anything these refusals did.
        """
        import subprocess

        from mobauto2_benders.config import load_config

        tracked = subprocess.run(
            ["git", "ls-files", "configs"],
            cwd=str(_helpers.REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        tracked = {
            (_helpers.REPO_ROOT / f).resolve()
            for f in tracked
            if f.endswith(".yaml")
        }

        checked = 0
        for path in sorted(CONFIGS.rglob("*.yaml")):
            # configs/milp/ is the monolith's schema, parsed by mobauto2_milp.config.
            # Feeding it to the Benders loader tests nothing about these refusals.
            if (
                path.resolve() not in tracked
                or path.name.startswith("_tmp_")
                or path.parent.name == "milp"
            ):
                continue
            try:
                cfg = load_config(path)
            except Exception as exc:
                raise AssertionError(f"{path.name} no longer loads: {exc}") from exc
            if hasattr(cfg, "subproblem"):
                self.assertIn(cfg.subproblem.cut_mode, CUT_MODE_VALID_LOWER_BOUND)
                checked += 1
        self.assertGreater(checked, 10, "sweep found suspiciously few configs")


class NoCutAggregation(unittest.TestCase):
    """D64 4b. A scenario that produced NO cut is not a scenario with an unknown cut.

    The theta early-exit skips cut generation for a scenario whose theta already covers
    its recourse. That scenario's diagnostics carry neither `cut_generation_mode` nor
    `cut_valid_lower_bound`, and reading the absence as "unknown" made the conjunction
    False and dropped the run's lower bound -- measured on `theta_by_scen`, 2 of 96
    iterations, reported as `mixed(mw+unknown)`.

    The aggregation is inline in `evaluate`, so these tests pin the RULE it implements.
    """

    @staticmethod
    def _aggregate(scenario_diags):
        contributors = [d for d in scenario_diags if "cut_valid_lower_bound" in d]
        abstained = len(scenario_diags) - len(contributors)
        valid = (
            all(bool(d["cut_valid_lower_bound"]) for d in contributors)
            if contributors
            else False
        )
        modes = sorted(
            {str(d.get("cut_generation_mode", "unknown")) for d in contributors}
        )
        joined = "+".join(modes)
        label = modes[0] if len(modes) == 1 else (f"mixed({joined})" if modes else "no_cut")
        if abstained:
            label = f"{label}+no_cut({abstained})"
        return valid, label

    def test_an_abstaining_scenario_no_longer_voids_the_bound(self):
        valid, label = self._aggregate(
            [
                {"cut_generation_mode": "mw", "cut_valid_lower_bound": True},
                {},  # theta early-exit: produced nothing
            ]
        )
        self.assertTrue(valid, "an abstention still poisons the aggregate")
        self.assertEqual(label, "mw+no_cut(1)")

    def test_a_genuinely_invalid_contributor_still_voids_it(self):
        valid, _ = self._aggregate(
            [
                {"cut_generation_mode": "mw", "cut_valid_lower_bound": True},
                {
                    "cut_generation_mode": "finite_difference",
                    "cut_valid_lower_bound": False,
                },
            ]
        )
        self.assertFalse(valid)

    def test_all_abstaining_is_not_valid(self):
        """No contributor means nothing certified the bound. Fail closed."""
        valid, label = self._aggregate([{}, {}])
        self.assertFalse(valid)
        self.assertEqual(label, "no_cut+no_cut(2)")

    def test_mixed_contributors_are_labelled(self):
        _valid, label = self._aggregate(
            [
                {"cut_generation_mode": "mw", "cut_valid_lower_bound": True},
                {
                    "cut_generation_mode": "mw_dual_fallback",
                    "cut_valid_lower_bound": True,
                },
            ]
        )
        self.assertEqual(label, "mixed(mw+mw_dual_fallback)")


if __name__ == "__main__":
    unittest.main()
