"""Phase 5: the decomposition reaches the SAME optimum as the extensive form.

Handout section 86 phase 5: `|z*_Benders - z*_extensive| <= eps`. This is the rung of
the validation ladder that was missing. Everything else in this repository asserts an
INEQUALITY -- `LB <= (a known feasible objective)` -- which catches a bound that is not
a bound (the D30 class) and nothing else. It is satisfied by:

  * a decomposition that is valid but converges to the wrong place;
  * a master missing a constraint the monolith has;
  * cuts so weak the run never closes at all.

An equality, at a size where BOTH sides prove optimality, catches all three.

Two instances, deliberately:

  slack  setups/phase5_tiny.yaml       6 pax per direction against S=15. Capacity never
                                        binds, so pi can be 0 everywhere -- measured, the
                                        final cut on this cell has nnz=0.
  tight  setups/phase5_tiny_tight.yaml 17 OUT in one slot against S=15. Capacity binds,
                                        so pi is strictly negative somewhere and the
                                        SLOPE vector is under test, not just the constant.

The slack cell alone would be a weak gate: a nearly flat cut can still drive a trivial
master to the right answer, so the equality could hold while carrying almost no cut
information. Hence the pair.

Two cut modes per instance: `dual` (the plain capacity duals -- the baseline) and `mw`.
Formal formulation 16.6 says MW changes the SHAPE of the cut away from the incumbent and
not its exactness at the incumbent, so both must reach the same optimum. Asserting that
is what turns 16.6 from a claim into a test.

REQUIRES CPLEX. Six solves, a few seconds total.
    p310 -m unittest tests.test_phase5_exactness -v
"""

from __future__ import annotations

import unittest

import _helpers  # noqa: F401  (puts src/ on sys.path)
from _helpers import require_solver_backend
from _helpers import CONFIGS

BENDERS = {
    "slack": {"dual": "phase5/tiny_dual.yaml", "mw": "phase5/tiny_mw.yaml"},
    "tight": {
        "dual": "phase5/tiny_dual_tight.yaml",
        "mw": "phase5/tiny_mw_tight.yaml",
    },
}
MONOLITH = {
    "slack": CONFIGS / "milp" / "phase5_tiny_monolith.yaml",
    "tight": CONFIGS / "milp" / "phase5_tiny_tight_monolith.yaml",
}

# Absolute tolerance on the equality.
#
# It has to admit the optimal-face slack. `solve_mw_dual` states the optimal face as an
# inequality with `face_tol = max(1e-6, 1e-9*|Q|)`, so a Magnanti-Wong cut may sit up to
# 1e-6 inside the face and the lower bound derived from it lands that much low. Measured
# on the slack cell: monolith 12.02, MW LB 12.019998999999995 -- short by 1e-6 exactly.
#
# That gap is S2 working rather than a defect. Before S2 the intercept was IMPOSED as
# `Q(y) - sum(dm*y_inc)`, so the LB printed 12.02 on the nose while nothing had verified
# the dual actually supported it. Deriving the intercept from alpha costs 1e-6 of bound
# and buys the guarantee. A tolerance tighter than the face slack would fail the honest
# version and pass the dishonest one.
EPS = 1.0e-5


def _declared_demand(cell: str) -> float:
    """Passenger count from the scenario file's own `n:` field.

    Deliberately not via the loader: the bracket this feeds is meant to be an
    independent expectation, and a number taken from the code under test would only
    prove that code agrees with itself.
    """
    import yaml

    from mobauto2_milp.config import load_config as load_milp

    rel = str((load_milp(MONOLITH[cell]).data.scenario_files or [None])[0])
    doc = yaml.safe_load((_helpers.REPO_ROOT / rel).read_text(encoding="utf-8"))
    return float(doc["n"])


class TestTheArmsDescribeTheSameInstance(unittest.TestCase):
    """No solver. An equality between two models is only a test if it is one instance.

    Without this the files could drift -- a different Q, horizon, penalty or demand on
    either side -- and the gate would compare the optima of two different problems and
    report the disagreement as a decomposition defect. `test_monolith_reference` needed
    the same guard for the same reason.
    """

    def _pair(self, cell: str, mode: str):
        from mobauto2_benders.config import load_config as load_benders
        from mobauto2_milp.config import load_config as load_milp

        return (
            load_benders(CONFIGS / BENDERS[cell][mode]),
            load_milp(MONOLITH[cell]),
        )

    def test_instance_fields_agree(self):
        for cell in BENDERS:
            for mode in BENDERS[cell]:
                b, m = self._pair(cell, mode)
                with self.subTest(cell=cell, mode=mode):
                    self.assertEqual(b.model.time.T_minutes, m.model.time.T_minutes)
                    self.assertEqual(
                        b.model.time.slot_resolution, m.model.time.slot_resolution
                    )
                    self.assertEqual(
                        b.model.time.trip_duration_minutes,
                        m.model.time.trip_duration_minutes,
                    )
                    self.assertEqual(b.model.fleet.Q, m.model.fleet.Q)
                    self.assertEqual(b.model.energy.Emax, m.model.energy.Emax)
                    self.assertEqual(b.model.energy.L, m.model.energy.L)
                    self.assertAlmostEqual(
                        float(b.model.costs.start_cost_epsilon),
                        float(m.model.costs.start_cost_epsilon),
                        places=9,
                    )
                    self.assertAlmostEqual(
                        float(b.model.costs.concurrency_penalty),
                        float(m.model.costs.concurrency_penalty),
                        places=9,
                    )

    def test_service_policy_agrees(self):
        for cell in BENDERS:
            for mode in BENDERS[cell]:
                b, m = self._pair(cell, mode)
                with self.subTest(cell=cell, mode=mode):
                    self.assertAlmostEqual(
                        float(b.subproblem.S), float(m.service.S), places=9
                    )
                    self.assertAlmostEqual(
                        float(b.subproblem.p), float(m.service.p), places=9
                    )
                    self.assertEqual(
                        b.subproblem.Wmax_minutes, m.service.Wmax_minutes
                    )

    def test_both_arms_read_the_same_demand_file(self):
        from mobauto2_benders.config import load_config as load_benders
        from mobauto2_milp.config import load_config as load_milp

        for cell in BENDERS:
            m = load_milp(MONOLITH[cell])
            mono_files = [str(x) for x in (m.data.scenario_files or [])]
            self.assertEqual(len(mono_files), 1, "the gate is single-scenario")
            for mode in BENDERS[cell]:
                b = load_benders(CONFIGS / BENDERS[cell][mode])
                with self.subTest(cell=cell, mode=mode):
                    self.assertEqual(str(b.data.demand_file), mono_files[0])

    def test_the_two_cells_are_different_instances(self):
        """Guards against both cells silently pointing at the same demand file.

        If they did, "slack" and "tight" would be one measurement reported twice and
        the slope vector would never be exercised.
        """
        from mobauto2_milp.config import load_config as load_milp

        files = {
            cell: str((load_milp(MONOLITH[cell]).data.scenario_files or [None])[0])
            for cell in MONOLITH
        }
        self.assertNotEqual(files["slack"], files["tight"])


class TestBendersReachesTheMonolithOptimum(unittest.TestCase):
    """REQUIRES AN LP/MIP BACKEND. Six solves: two monoliths and four Benders arms."""

    @classmethod
    def setUpClass(cls):
        require_solver_backend()
        import os

        from mobauto2_benders.app import run as benders_run
        from mobauto2_milp.app import _prepare_params as milp_params
        from mobauto2_milp.config import load_config as load_milp
        from mobauto2_milp.model import MobautoMilpModel
        from mobauto2_milp.monolith import MonolithSolver

        # Both packages resolve `setups/...` against the process CWD, so the gate would
        # pass under `discover -s tests` from the repo root and fail when the file is
        # run on its own. Pin it instead of relying on how the suite was invoked.
        cls._cwd = os.getcwd()
        os.chdir(_helpers.REPO_ROOT)

        cls.mono = {}
        cls.all_unserved = {}
        for cell, path in MONOLITH.items():
            cfg = load_milp(_helpers.fixture_for_backend(path))
            mp, sp = milp_params(cfg, {})
            cls.mono[cell] = MonolithSolver(MobautoMilpModel, cfg, mp, sp).run()
            # All-unserved cost, for the non-degeneracy bracket. `p` is in slot units
            # after the load-time conversion from p_minutes.
            #
            # The head count comes from the scenario file's own `n:` field rather than
            # from a loader: `_prepare_params` leaves R_out/R_ret as None here (the
            # monolith reads demand inside run()), and taking the number from the same
            # code path under test would make the bracket agree with itself. It is
            # cross-checked against what the subproblem reports in
            # test_the_demand_count_agrees_with_the_scenario_file.
            cls.all_unserved[cell] = _declared_demand(cell) * float(sp["p"])

        cls.bd = {}
        for cell, modes in BENDERS.items():
            for mode, rel in modes.items():
                cls.bd[(cell, mode)] = benders_run(
                    str(_helpers.fixture_for_backend(CONFIGS / rel)), {}
                )

    @classmethod
    def tearDownClass(cls):
        import os

        os.chdir(cls._cwd)

    # ---- the gate itself -------------------------------------------------------

    def test_every_arm_reaches_the_monolith_optimum(self):
        for (cell, mode), res in self.bd.items():
            z = float(self.mono[cell].best_upper_bound)
            with self.subTest(cell=cell, mode=mode):
                self.assertIsNotNone(res.best_upper_bound)
                self.assertIsNotNone(res.best_lower_bound)
                self.assertLessEqual(
                    abs(float(res.best_upper_bound) - z),
                    EPS,
                    f"{cell}/{mode}: Benders UB {res.best_upper_bound} vs monolith {z}",
                )
                self.assertLessEqual(
                    abs(float(res.best_lower_bound) - z),
                    EPS,
                    f"{cell}/{mode}: Benders LB {res.best_lower_bound} vs monolith {z}",
                )

    def test_the_lower_bound_never_exceeds_the_optimum(self):
        """One-sided and stricter than the equality: an LB above the true optimum is
        not a tolerance question, it is the D30 defect. No EPS slack on this side."""
        for (cell, mode), res in self.bd.items():
            z = float(self.mono[cell].best_upper_bound)
            with self.subTest(cell=cell, mode=mode):
                self.assertLessEqual(float(res.best_lower_bound), z + EPS)

    def test_both_sides_prove_optimality_rather_than_stopping_on_the_clock(self):
        """A clock-truncated MIP is machine-dependent (D26). A gate that stops on the
        clock passes on a quiet machine and fails on a busy one, which is worse than
        no gate at all."""
        from mobauto2_benders.benders.types import SolveStatus as BdStatus
        from mobauto2_milp.types import SolveStatus as MilpStatus

        for cell, r in self.mono.items():
            with self.subTest(arm=f"monolith/{cell}"):
                self.assertEqual(r.status, MilpStatus.OPTIMAL)
        for (cell, mode), res in self.bd.items():
            with self.subTest(cell=cell, mode=mode):
                self.assertEqual(res.status, BdStatus.OPTIMAL)
                self.assertEqual(
                    int(res.clock_truncated_master_solves or 0),
                    0,
                    "a master solve stopped on the clock; this gate is then one draw",
                )

    def test_the_cuts_carried_a_lower_bound(self):
        """The equality means nothing if the run reported no bound (S1).

        Both modes here are in CUT_MODE_VALID_LOWER_BOUND, so `mw` degrading to
        `mw_dual_fallback` is acceptable and still valid -- but `finite_difference`
        would not be, and before S1 that was where a failed MW landed.
        """
        from mobauto2_benders.problem.subproblem_impl import (
            CUT_MODE_VALID_LOWER_BOUND,
        )

        for (cell, mode), res in self.bd.items():
            with self.subTest(cell=cell, mode=mode):
                self.assertTrue(res.cut_valid_lower_bound)
                self.assertIn(
                    str(res.cut_generation_mode), CUT_MODE_VALID_LOWER_BOUND
                )
                self.assertTrue(
                    CUT_MODE_VALID_LOWER_BOUND[str(res.cut_generation_mode)]
                )

    # ---- guards against the gate passing for the wrong reason -------------------

    def test_the_optima_are_non_degenerate(self):
        """Strictly between 0 and the all-unserved cost.

        At 0 the instance has no demand and every schedule is optimal. At the
        all-unserved cost the optimum is "run nothing", the recourse does no work, and
        the equality would hold without the cut geometry being exercised at all.
        """
        for cell, r in self.mono.items():
            z = float(r.best_upper_bound)
            with self.subTest(cell=cell):
                self.assertGreater(z, 0.0)
                self.assertLess(z, self.all_unserved[cell])

    def test_the_demand_count_agrees_with_the_scenario_file(self):
        """Closes the loop on the independent count used for the bracket above.

        If the loader silently dropped requests outside the horizon (D25) the bracket
        would be computed against more demand than the model actually saw, and it
        would loosen without saying so. These instances are built to fit entirely
        inside the horizon, and this is what asserts they still do.
        """
        for (cell, mode), res in self.bd.items():
            with self.subTest(cell=cell, mode=mode):
                self.assertIsNotNone(res.sp_total_demand)
                self.assertAlmostEqual(
                    float(res.sp_total_demand), _declared_demand(cell), places=6
                )

    def test_the_two_cells_exercise_different_optima(self):
        self.assertNotAlmostEqual(
            float(self.mono["slack"].best_upper_bound),
            float(self.mono["tight"].best_upper_bound),
            places=3,
        )

    def test_the_tight_cell_really_binds_capacity(self):
        """Otherwise "tight" is a second slack cell and no slope is ever tested.

        Priced directly rather than inferred from the objective: the subproblem is
        asked for its capacity duals at a one-departure-per-direction schedule, and at
        least one must be strictly negative. A dual of 0 everywhere means extra seats
        are worth nothing, i.e. capacity does not bind.
        """
        from mobauto2_benders.app import _prepare_params
        from mobauto2_benders.config import load_config
        from mobauto2_benders.problem.subproblem_impl import (
            SPParams,
            aggregate_requests,
            load_demand_doc,
            slopes_from_capacity_duals,
            solve_subproblem,
            wmax_minutes_to_slots,
        )

        cfg = load_config(CONFIGS / BENDERS["tight"]["dual"])
        mp, sp = _prepare_params(cfg, {})
        res_min = int(mp["slot_resolution"])
        T = int(mp.get("T") or (int(mp["T_minutes"]) // res_min))
        S = float(sp["S"])
        # `Wmax_slots` is absent from the params by design when the config states the
        # policy in minutes (D53), so convert the same way the subproblem does --
        # floor, not ceil: ceil would GRANT more waiting than the config asked for.
        W = wmax_minutes_to_slots(float(sp["Wmax_minutes"]), res_min)

        doc = load_demand_doc(_helpers.REPO_ROOT / str(cfg.data.demand_file))
        R_out, R_ret = aggregate_requests(doc, T, res_min)

        # One OUT at tau=1 (reaches the slot-0 arrivals) and one RET at tau=3.
        C_out = [0.0] * T
        C_ret = [0.0] * T
        C_out[1] = S
        C_ret[3] = S
        P = SPParams(
            T=T,
            Wmax_slots=W,
            p=float(sp["p"]),
            lp_solver=require_solver_backend(),
            S=S,
            K_out=[1 if c > 0 else 0 for c in C_out],
            K_ret=[1 if c > 0 else 0 for c in C_ret],
            slot_resolution=res_min,
        )
        duals, _obj = solve_subproblem(P, C_out, C_ret, list(R_out), list(R_ret))
        mw = slopes_from_capacity_duals(duals, S, T)
        worst = min(list(mw.dm_out.values()) + list(mw.dm_ret.values()))
        self.assertLess(
            worst,
            -1e-9,
            "no capacity dual is strictly negative on the 'tight' cell, so it is not "
            "tight and the slope vector is untested by this gate",
        )
        # And 20.2: no slope may be positive, in either direction.
        for d, slopes in (("OUT", mw.dm_out), ("RET", mw.dm_ret)):
            for tau, v in slopes.items():
                with self.subTest(direction=d, tau=tau):
                    self.assertLessEqual(v, 1e-9)


if __name__ == "__main__":
    unittest.main()
