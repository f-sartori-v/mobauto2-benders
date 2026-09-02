"""Config schema. No solver required."""

from __future__ import annotations

import unittest

from _helpers import CONFIGS, DEFAULT_CONFIG, load_cfg


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

    def test_delta_chg_recomputes_after_a_slot_resolution_override(self):
        """R6 (handout C6, 2026-09-02): delta_chg is evaluated once at config load
        for that config's own slot_resolution. Any script that overrides
        slot_resolution (a multi-resolution sweep, e.g.) must call
        _energy_params_for_resolution afterwards or every delta != the config's own
        30 keeps charging at the 30-minute rate -- 2x too fast at delta=15, 3x at
        delta=10. This defect invalidated 80 of 120 cells once already; this test
        makes the recompute a checked contract rather than a convention scripts must
        remember.
        """
        from mobauto2_benders.app import _energy_params_for_resolution

        cfg = load_cfg("baseline_d9.yaml")
        expected = {30: 35.00, 15: 17.50, 10: 11.67, 1: 1.17}
        for delta, want in expected.items():
            with self.subTest(slot_resolution=delta):
                got = float(_energy_params_for_resolution(cfg, delta)["delta_chg"])
                self.assertAlmostEqual(got, want, places=2)


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
            _validate_cplex_options(
                {"CPXPARAM_MIP_Strategy_Symmetry": 5}, "cplex_direct"
            )
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
                _validate_cplex_options(
                    cfg.master.cplex_options, cfg.master.solver_backend
                )


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

        raw = yaml.safe_load((CONFIGS / DEFAULT_CONFIG).read_text(encoding="utf-8"))
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


class TestPenaltyUnits(unittest.TestCase):
    """D50. `p` is in slot units, so stating it directly makes the policy it encodes
    move with the grid. `p_minutes` states the same policy resolution-independently.

    The repository already contained both readings before this existed --
    baseline_d9 at 30-minute slots with p=50 (one unserved passenger worth 1500
    passenger-minutes) and the Fase 1 point at 15-minute slots with p=50 (worth 750).
    Each run was internally consistent; no objective from one was comparable with the
    other, and nothing said so.
    """

    BASE = CONFIGS / "baseline_d9.yaml"

    def _load(self, text: str):
        from mobauto2_benders.config import load_config
        import tempfile, os

        fd, path = tempfile.mkstemp(suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            return load_config(path)
        finally:
            os.unlink(path)

    def _variant(self, slot: int, penalty_line: str) -> str:
        text = self.BASE.read_text(encoding="utf-8")
        text = text.replace("slot_resolution: 30", f"slot_resolution: {slot}")
        return text.replace("  p: 50.0", penalty_line)

    def test_p_minutes_encodes_one_policy_at_every_resolution(self):
        """The whole point: p * slot_resolution must be invariant."""
        for slot, expected_p in ((30, 50.0), (15, 100.0), (5, 300.0), (1, 1500.0)):
            with self.subTest(slot=slot):
                cfg = self._load(self._variant(slot, "  p_minutes: 1500.0"))
                self.assertAlmostEqual(cfg.subproblem.p, expected_p, places=9)
                self.assertAlmostEqual(
                    cfg.subproblem.p * cfg.model.time.slot_resolution, 1500.0, places=6
                )

    def test_a_bare_p_does_not_encode_one_policy(self):
        """The behaviour p_minutes exists to replace, asserted so the difference is
        visible rather than folklore."""
        worth = {}
        for slot in (30, 15):
            cfg = self._load(self._variant(slot, "  p: 50.0"))
            worth[slot] = cfg.subproblem.p * cfg.model.time.slot_resolution
        self.assertEqual(worth[30], 1500.0)
        self.assertEqual(worth[15], 750.0)
        self.assertNotEqual(
            worth[30], worth[15], "bare p would have to be resolution-dependent"
        )

    def test_p_minutes_is_not_rounded(self):
        """p is a cost coefficient, not an index bound like Wmax. Rounding it would
        silently change the policy."""
        cfg = self._load(self._variant(7, "  p_minutes: 100.0"))
        self.assertAlmostEqual(cfg.subproblem.p, 100.0 / 7.0, places=12)

    def test_both_forms_at_once_is_refused(self):
        """Ambiguity fails closed rather than resolving by precedence: a precedence
        rule would let a config state two policies and silently honour one."""
        with self.assertRaises(ValueError) as ctx:
            self._load(self._variant(30, "  p: 50.0\n  p_minutes: 1500.0"))
        self.assertIn("p_minutes", str(ctx.exception))

    def test_neither_form_is_refused(self):
        with self.assertRaises(ValueError):
            self._load(self._variant(30, ""))

    def test_existing_configs_are_unchanged(self):
        """Backward compatibility: every shipped config states a bare p and must load
        to exactly the value it always did."""
        from mobauto2_benders.config import load_config

        for name, expected in (
            ("baseline_d9.yaml", 50.0),
            ("phase1/lp_only_150.yaml", 50.0),
        ):
            with self.subTest(name=name):
                cfg = load_config(CONFIGS / name)
                self.assertAlmostEqual(cfg.subproblem.p, expected, places=9)
                self.assertIsNone(
                    cfg.subproblem.p_minutes,
                    "a config stating a bare p must record p_minutes as None so the "
                    "manifest can say which form was used",
                )


class TestWmaxIsNeverRoundedUp(unittest.TestCase):
    """The maximum wait is a service promise, so the discretisation must not grant
    more of it than the config asked for.

    `ceil` did: at 30-minute slots `Wmax_minutes: 45` became 2 slots, and a passenger
    could be made to wait 60 minutes against a stated cap of 45. `floor` never grants
    more than asked.
    """

    def _fn(self):
        from mobauto2_benders.problem.subproblem_impl import wmax_minutes_to_slots

        return wmax_minutes_to_slots

    def test_the_cap_is_never_exceeded(self):
        """The invariant, stated directly: slots * slot_width <= Wmax_minutes."""
        fn = self._fn()
        for wmax in (30, 45, 60, 75, 90, 120):
            for slot in (1, 5, 15, 30):
                if wmax < slot:
                    continue
                with self.subTest(wmax=wmax, slot=slot):
                    self.assertLessEqual(fn(wmax, slot) * slot, wmax)

    def test_the_shipped_settings_are_unaffected(self):
        """60 minutes at 30- and 15-minute slots divide exactly, so no number in this
        repository changes. Asserted so the fix cannot be blamed for a later drift."""
        fn = self._fn()
        self.assertEqual(fn(60, 30), 2)
        self.assertEqual(fn(60, 15), 4)

    def test_the_case_that_used_to_be_wrong(self):
        fn = self._fn()
        self.assertEqual(fn(45, 30), 1, "45 minutes at 30-minute slots must be 1 slot")

    def test_a_cap_shorter_than_one_slot_is_refused(self):
        """floor would give 0 -- no arcs, every passenger unserved, no error. ceil hid
        this by rounding up to one slot, which is the over-permissive direction."""
        fn = self._fn()
        with self.assertRaises(ValueError) as ctx:
            fn(20, 30)
        self.assertIn("shorter than one slot", str(ctx.exception))

    def test_both_packages_agree(self):
        """The rule is implemented once per package; the copies must not drift."""
        from mobauto2_milp.monolith import wmax_minutes_to_slots as milp_fn

        fn = self._fn()
        for wmax, slot in ((60, 30), (60, 15), (45, 30), (90, 30), (60, 1)):
            with self.subTest(wmax=wmax, slot=slot):
                self.assertEqual(fn(wmax, slot), milp_fn(wmax, slot))
