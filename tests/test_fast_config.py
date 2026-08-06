"""Config schema. No solver required."""
from __future__ import annotations

import unittest

from _helpers import CONFIGS, load_cfg


class TestConfigSchema(unittest.TestCase):
    def test_all_shipped_configs_load(self):
        for path in sorted(CONFIGS.glob("*.yaml")):
            with self.subTest(config=path.name):
                load_cfg(path.name)

    def test_unknown_key_is_rejected(self):
        """The allow-list is what stops silent typos in a config."""
        import yaml
        from mobauto2_benders.config import load_config
        import tempfile, os

        raw = yaml.safe_load((CONFIGS / "baseline_d9.yaml").read_text(encoding="utf-8"))
        raw["subproblem"]["not_a_real_key"] = 1.0
        fd, tmp = tempfile.mkstemp(suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                yaml.safe_dump(raw, fh)
            with self.assertRaises(ValueError) as ctx:
                load_config(tmp)
            self.assertIn("not_a_real_key", str(ctx.exception))
        finally:
            os.unlink(tmp)

    def test_unused_capacity_penalty_is_gone(self):
        """Spec §4: it was validated, threaded through, and read by nothing.

        Deleted rather than wired in. Setting it must now be an error, not a
        silently inert knob -- the previous behaviour already misled someone into
        setting it to 0.5 expecting an effect.
        """
        import yaml
        from mobauto2_benders.config import load_config
        import tempfile, os

        raw = yaml.safe_load((CONFIGS / "baseline_d9.yaml").read_text(encoding="utf-8"))
        raw["subproblem"]["unused_capacity_penalty"] = 0.5
        fd, tmp = tempfile.mkstemp(suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                yaml.safe_dump(raw, fh)
            with self.assertRaises(ValueError):
                load_config(tmp)
        finally:
            os.unlink(tmp)

    def test_emit_reports_defaults_off(self):
        """M5: a normal run must not litter LP and solver-log files.

        Previously derived as log_level != "REPORT", i.e. on by default and
        disabled only by a log level that reads as if it would enable reports.
        """
        cfg = load_cfg("baseline_d9.yaml")
        self.assertFalse(cfg.run.emit_reports)

    def test_charge_before_idle_defaults_on(self):
        """M2: the canonicalisation stays on unless explicitly disabled."""
        cfg = load_cfg("baseline_d9.yaml")
        self.assertTrue(cfg.master.charge_before_idle)

    def test_concurrency_penalty_is_reported(self):
        """It is active in the objective and not in the published formulation,
        so its value has to be visible on every table. Guard that it is at least
        carried in the parsed config rather than being an invisible default."""
        cfg = load_cfg("baseline_d9.yaml")
        self.assertIsNotNone(cfg.model.costs.concurrency_penalty)


if __name__ == "__main__":
    unittest.main()
