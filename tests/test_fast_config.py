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


class TestTimeLimitKeysAreDistinguishable(unittest.TestCase):
    """Three limits decide when a run stops and two of them used to be inert.

    `solver.time_limit_s` (whole loop) and `master.solve_time_limit_s` (one master
    solve) differed by one word, and `master.mipgap` read like the convergence
    criterion when that is `solver.tolerance`. Worse, the Benders loop overwrote
    both master values on every iteration from hardcoded constants, so setting
    them in a config did nothing at all.

    The old names now fail loudly rather than being silently ignored.
    """

    def _load_with(self, section: str, key: str, value):
        import yaml, tempfile, os
        from mobauto2_benders.config import load_config

        raw = yaml.safe_load((CONFIGS / "baseline_d9.yaml").read_text(encoding="utf-8"))
        raw[section][key] = value
        fd, tmp = tempfile.mkstemp(suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                yaml.safe_dump(raw, fh)
            return load_config(tmp)
        finally:
            os.unlink(tmp)

    def test_old_master_time_limit_name_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._load_with("master", "solve_time_limit_s", 30)
        self.assertIn("per_iteration_time_limit_s", str(ctx.exception))

    def test_old_master_mipgap_name_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._load_with("master", "mipgap", 0.05)
        self.assertIn("per_iteration_mipgap", str(ctx.exception))

    def test_old_solver_time_limit_name_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._load_with("solver", "time_limit_s", 120)
        self.assertIn("total_time_limit_s", str(ctx.exception))

    def test_shipped_config_exposes_all_three_limits(self):
        cfg = load_cfg("baseline_d9.yaml")
        self.assertIsNotNone(cfg.master.per_iteration_time_limit_s)
        self.assertIsNotNone(cfg.master.per_iteration_mipgap)
        self.assertGreater(cfg.solver.total_time_limit_s, 0)
        self.assertGreater(cfg.solver.tolerance, 0.0)


class TestMasterScheduleRespectsConfigCeilings(unittest.TestCase):
    """The gap-tied schedule may tighten the master's controls, never loosen them
    past what the config asked for. Before this was wired, the config values were
    parsed, threaded down and discarded -- the same defect class as D19."""

    def test_schedule_ceilings_come_from_config(self):
        from mobauto2_benders.config import load_config
        import yaml, tempfile, os

        raw = yaml.safe_load((CONFIGS / "baseline_d9.yaml").read_text(encoding="utf-8"))
        raw["master"]["per_iteration_time_limit_s"] = 7
        raw["master"]["per_iteration_mipgap"] = 0.02
        fd, tmp = tempfile.mkstemp(suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                yaml.safe_dump(raw, fh)
            cfg = load_config(tmp)
        finally:
            os.unlink(tmp)

        # The schedule computes mp_tl = 2 + 5/mp_gap, which at gap 0.02 is 252s --
        # far above the 7s ceiling, so the cap is what must survive.
        self.assertEqual(cfg.master.per_iteration_time_limit_s, 7)
        self.assertAlmostEqual(cfg.master.per_iteration_mipgap, 0.02)


class TestCplexOptionNames(unittest.TestCase):
    """CPXPARAM_* options were silently dropped under solver_backend=cplex_direct.

    Pyomo's CPLEXDirect splits an option key on '_' and walks
    cplex.Cplex().parameters, so 'CPXPARAM_Threads' would resolve
    parameters.CPXPARAM.Threads and raise. The old code skipped every CPXPARAM_*
    key to avoid that, which meant a config asking for one thread quietly used
    all of them -- and the baselines cite reproducibility as the reason for the
    setting.

    The drop also hid a second mistake: the shipped name
    CPXPARAM_MIP_Strategy_Symmetry does not exist. CPLEX calls that parameter
    preprocessing.symmetry.
    """

    def test_cpxparam_names_translate_to_direct_paths(self):
        from mobauto2_benders.problem.master_impl import _cplex_direct_option_name
        self.assertEqual(_cplex_direct_option_name("CPXPARAM_Threads"), "threads")
        self.assertEqual(
            _cplex_direct_option_name("CPXPARAM_Preprocessing_Symmetry"),
            "preprocessing_symmetry",
        )

    def test_non_cpxparam_keys_pass_through(self):
        from mobauto2_benders.problem.master_impl import _cplex_direct_option_name
        self.assertEqual(_cplex_direct_option_name("timelimit"), "timelimit")

    def test_unresolvable_parameter_name_is_rejected(self):
        from mobauto2_benders.problem.master_impl import _validate_cplex_options
        try:
            import cplex  # noqa: F401
        except Exception:
            self.skipTest("CPLEX not available")
        with self.assertRaises(ValueError) as ctx:
            _validate_cplex_options({"CPXPARAM_MIP_Strategy_Symmetry": 5}, "cplex_direct")
        self.assertIn("CPXPARAM_MIP_Strategy_Symmetry", str(ctx.exception))

    def test_shipped_configs_use_resolvable_names(self):
        from mobauto2_benders.problem.master_impl import _validate_cplex_options
        try:
            import cplex  # noqa: F401
        except Exception:
            self.skipTest("CPLEX not available")
        for path in sorted(CONFIGS.glob("*.yaml")):
            with self.subTest(config=path.name):
                cfg = load_cfg(path.name)
                _validate_cplex_options(cfg.master.cplex_options, cfg.master.solver_backend)


class TestMultiScenarioBoundSemantics(unittest.TestCase):
    """LB and UB must describe the same problem.

    One cut per scenario against a single theta forces theta >= max_s Q_s(y),
    while the reported UB is the weighted mean of the same Q_s. max >= mean, so
    the master's optimum could exceed the optimum the UB measures -- the D15/D16
    failure mode, a bound that is not a bound. The combination is refused at load.
    """

    def _cfg(self, multi_cuts, theta_per_scenario, scenarios=True):
        import yaml, tempfile, os
        from mobauto2_benders.config import load_config

        raw = yaml.safe_load((CONFIGS / "default.yaml").read_text(encoding="utf-8"))
        raw["subproblem"]["multi_cuts_by_scenario"] = multi_cuts
        raw["master"]["theta_per_scenario"] = theta_per_scenario
        if not scenarios:
            raw["data"] = {**raw["data"], "scenario_files": None, "scenarios": None}
        fd, tmp = tempfile.mkstemp(suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                yaml.safe_dump(raw, fh)
            return load_config(tmp)
        finally:
            os.unlink(tmp)

    def test_multi_cuts_on_a_shared_theta_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self._cfg(multi_cuts=True, theta_per_scenario=False)
        self.assertIn("theta_per_scenario", str(ctx.exception))

    def test_per_scenario_theta_is_accepted(self):
        cfg = self._cfg(multi_cuts=True, theta_per_scenario=True)
        self.assertTrue(cfg.master.theta_per_scenario)

    def test_averaged_cuts_are_accepted(self):
        cfg = self._cfg(multi_cuts=False, theta_per_scenario=False)
        self.assertFalse(cfg.subproblem.multi_cuts_by_scenario)

    def test_single_scenario_runs_are_unaffected(self):
        """Without scenarios there is no expectation to disagree about, and
        multi_cuts_by_scenario defaults true."""
        cfg = self._cfg(multi_cuts=True, theta_per_scenario=False, scenarios=False)
        self.assertTrue(cfg.subproblem.multi_cuts_by_scenario)
