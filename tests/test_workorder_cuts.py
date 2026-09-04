"""Cut-level acceptance tests: scenario weights, degenerate duals, tiny enumeration.

The three tests in the work order's table that are about the CUT rather than about the
model. Each fails on the pre-change code; the docstrings say how.

Run just these:
    python -m unittest discover -s tests -p "test_workorder_cuts.py" -v
"""

from __future__ import annotations

import itertools
import unittest

import _helpers  # noqa: F401  (puts src/ on sys.path)
from _helpers import require_solver_backend

from mobauto2_benders.signature import candidate_signature

T = 6
SEATS = 15.0
WMAX_SLOTS = 2
TRIP_SLOTS = 1
Q_FLEET = 2

# Two scenarios that genuinely disagree about when demand arrives. If they agreed, a
# dropped scenario index would produce the same cut and `cut_scenario_weights` would
# pass on broken code.
SCENARIO_A = {"R_out": [0, 20, 4, 0, 0, 0], "R_ret": [0, 0, 18, 3, 0, 0]}
SCENARIO_B = {"R_out": [0, 3, 25, 0, 0, 0], "R_ret": [0, 0, 2, 22, 0, 0]}
WEIGHTS = [0.75, 0.25]  # deliberately unequal, and not each other's complement by 0.5


def _sp(backend: str, **overrides) -> dict:
    params = {
        "T": T,
        "S": SEATS,
        "Wmax_slots": WMAX_SLOTS,
        "p": 56.0 / 30.0,
        "slot_resolution": 30,
        "trip_slots": TRIP_SLOTS,
        "Q": Q_FLEET,
        "lp_solver": backend,
        "cut_mode": "dual",
        "eps_cut": 1e-8,
    }
    params.update(overrides)
    return params


def _candidate(Y_out, Y_ret, Q: int = Q_FLEET) -> dict:
    """A per-vehicle schedule with the given signature.

    Sized from the signature rather than from a module constant: this file exercises
    two horizons (T=6 for the cut tests, T=5 for the enumeration) and a candidate built
    to the wrong length is an index error at best and a silently truncated schedule at
    worst.

    Vehicles are filled in index order. The recourse cannot tell the difference (E1),
    which is the point -- the cut is a function of the signature.
    """
    horizon = len(list(Y_out))
    cand: dict[str, float] = {}
    for q in range(Q):
        for t in range(horizon):
            cand[f"yOUT[{q},{t}]"] = 0.0
            cand[f"yRET[{q},{t}]"] = 0.0
    for t in range(horizon):
        for q in range(int(Y_out[t])):
            cand[f"yOUT[{q},{t}]"] = 1.0
        for q in range(int(Y_ret[t])):
            cand[f"yRET[{q},{t}]"] = 1.0
    return cand


def _evaluate(params: dict, cand: dict):
    from mobauto2_benders.problem.subproblem_impl import ProblemSubproblem

    return ProblemSubproblem(dict(params)).evaluate(dict(cand))


def _cut_value_at(meta: dict, Y_out, Y_ret) -> float:
    """Evaluate an assembled cut at a signature.

    The coefficients are keyed by (q, tau) and the production path broadcasts one
    per-slot dual to every q, so summing coefficient * Y[tau] / Q would double count.
    Instead the cut is evaluated on the per-vehicle candidate the signature implies,
    exactly as the master would.
    """
    cand = _candidate(Y_out, Y_ret)
    total = float(meta["const"])
    for (q, tau), v in dict(meta.get("coeff_yOUT") or {}).items():
        total += float(v) * cand.get(f"yOUT[{int(q)},{int(tau)}]", 0.0)
    for (q, tau), v in dict(meta.get("coeff_yRET") or {}).items():
        total += float(v) * cand.get(f"yRET[{int(q)},{int(tau)}]", 0.0)
    return total


# ---------------------------------------------------------------------------- B1
class TestCutScenarioWeights(unittest.TestCase):
    """`cut_scenario_weights`. The audit's item 1.4.

    FAILS BEFORE, in two distinct ways:

      1. The aggregated cut was assembled by `zip(weights, consts)` over five parallel
         lists that the per-scenario theta early-exit could shorten with a `continue`.
         A shortened list pairs every scenario with its predecessor's weight and drops
         the last one entirely.
      2. Nothing reconciled the assembled cut against `sum_s w_s Q_d(s, Ybar)` at
         insertion. The per-scenario tightness check runs BEFORE the mix, so every way
         of losing `s` or `w_s` in the aggregation survived it.
    """

    @classmethod
    def setUpClass(cls):
        cls.backend = require_solver_backend()
        cls.Y_out = [0, 2, 1, 0, 0, 0]
        cls.Y_ret = [0, 0, 1, 1, 0, 0]
        cls.cand = _candidate(cls.Y_out, cls.Y_ret)

    def _aggregated(self, weights):
        params = _sp(
            self.backend,
            scenarios=[dict(SCENARIO_A), dict(SCENARIO_B)],
            scenario_weights=list(weights),
            multi_cuts_by_scenario=False,
        )
        return _evaluate(params, self.cand)

    def _per_scenario_recourse(self):
        """Each scenario's own optimal recourse, solved separately.

        Computed here rather than read from the multi-scenario diagnostics on purpose:
        an independent source is what makes the reconciliation a check rather than a
        restatement.
        """
        out = []
        for scen in (SCENARIO_A, SCENARIO_B):
            params = _sp(self.backend, R_out=scen["R_out"], R_ret=scen["R_ret"])
            out.append(float(_evaluate(params, self.cand).upper_bound))
        return out

    def test_cut_scenario_weights(self):
        """The inserted cut's intercept reconciles to `sum_s w_s Q_d(s, Ybar)`."""
        res = self._aggregated(WEIGHTS)
        cut = (res.cuts or [res.cut])[0]
        meta = cut.metadata
        self.assertEqual(meta.get("cut_architecture"), "aggregated")

        singles = self._per_scenario_recourse()
        expected = sum(w * q for w, q in zip(WEIGHTS, singles))

        at_incumbent = _cut_value_at(meta, self.Y_out, self.Y_ret)
        self.assertAlmostEqual(
            at_incumbent,
            expected,
            places=6,
            msg=(
                "the aggregated cut does not evaluate to the weighted recourse at the "
                f"incumbent: cut={at_incumbent:.9g}, "
                f"sum_s w_s Q_s={expected:.9g} from independent per-scenario solves"
            ),
        )
        self.assertAlmostEqual(float(res.upper_bound), expected, places=6)

    def test_the_weights_actually_reach_the_cut(self):
        """Swap the weights and the cut must move.

        Without this, a cut that ignored `w_s` entirely -- using a plain mean, say --
        would still reconcile against a plain mean and pass the test above.
        """
        a = self._aggregated(WEIGHTS)
        b = self._aggregated(list(reversed(WEIGHTS)))
        meta_a = (a.cuts or [a.cut])[0].metadata
        meta_b = (b.cuts or [b.cut])[0].metadata
        self.assertNotAlmostEqual(
            float(meta_a["const"]),
            float(meta_b["const"]),
            places=6,
            msg="reversing the scenario weights left the cut unchanged, so the "
            "weights are not reaching it",
        )
        self.assertNotAlmostEqual(
            float(a.upper_bound), float(b.upper_bound), places=6
        )

    def test_the_cut_records_which_scenarios_and_weights_produced_it(self):
        """A cut that cannot say what went into it cannot be audited later."""
        res = self._aggregated(WEIGHTS)
        meta = (res.cuts or [res.cut])[0].metadata
        self.assertEqual(meta["scenario_indices"], [0, 1])
        self.assertEqual(
            [round(w, 9) for w in meta["scenario_weights"]],
            [round(w, 9) for w in WEIGHTS],
        )

    def test_unequal_weights_are_not_secretly_uniform(self):
        """Guard the guard: if the two scenarios priced identically, every assertion
        above would hold on broken code."""
        singles = self._per_scenario_recourse()
        self.assertNotAlmostEqual(singles[0], singles[1], places=3)


# ---------------------------------------------------------------------------- B4
class TestDegenerateDualsOk(unittest.TestCase):
    """`degenerate_duals_ok`. Two different optimal duals, both valid.

    WHAT THIS TESTS AND WHY IT IS WRITTEN THIS WAY. The work order asks for an
    instance that "returns two different optimal duals on two solves". Solving the
    same LP twice and hoping the solver pivots differently is not a test -- it passes
    or skips depending on the build, and a skip here reads as a pass.

    Two different optimal duals are obtained deterministically instead: the
    Magnanti-Wong-inspired selection and the plain capacity duals are two DIFFERENT
    dual-feasible solutions of the same degenerate LP. Both must give cuts valid at
    every sampled signature. That is exactly the property B4 says the fibre test was
    wrong to deny, and it is checked rather than assumed.

    FAILS BEFORE: the old fixture ASSERTED dual equality, so the very situation this
    test requires -- two different optimal duals -- was recorded as a defect.
    """

    @classmethod
    def setUpClass(cls):
        cls.backend = require_solver_backend()
        # Degenerate on purpose, and verified so below. All 30 OUT arrivals land in
        # slot 1 and two departures (slots 2 and 3) can each take all of them within
        # W_max; the assignment is not unique and neither is the dual. Measured, the
        # plain capacity duals and the Magnanti-Wong-inspired selection return
        # genuinely different optimal duals here -- const 60 with zero slopes against
        # const 112 with a -13 slope at tau=2 -- which is exactly the situation the old
        # fibre fixture recorded as a defect.
        cls.R_out = [0, 30, 0, 0, 0, 0]
        cls.R_ret = [0, 0, 30, 0, 0, 0]
        cls.Y_out = [0, 0, 2, 2, 0, 0]
        cls.Y_ret = [0, 0, 0, 2, 2, 0]
        cls.cand = _candidate(cls.Y_out, cls.Y_ret)

    def _cut(self, cut_mode: str):
        params = _sp(
            self.backend,
            R_out=self.R_out,
            R_ret=self.R_ret,
            cut_mode=cut_mode,
            use_magnanti_wong=(cut_mode == "mw"),
            use_dual_slopes=True,
            mw_core_point={"Yout": [0.5] * T, "Yret": [0.3] * T},
        )
        res = _evaluate(params, self.cand)
        return res, (res.cuts or [res.cut])[0].metadata

    def _sampled_signatures(self):
        """Twelve feasible signatures, including the incumbent and the empty one."""
        samples = [
            ([0, 0, 2, 2, 0, 0], [0, 0, 0, 2, 2, 0]),  # the incumbent
            ([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),  # all idle
            ([0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0]),
            ([0, 0, 2, 0, 0, 0], [0, 0, 0, 2, 0, 0]),
            ([0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 2, 0]),
            ([0, 0, 1, 2, 0, 0], [0, 0, 0, 2, 1, 0]),
            ([0, 0, 2, 1, 0, 0], [0, 0, 0, 1, 2, 0]),
            ([0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0]),
            ([0, 2, 2, 2, 0, 0], [0, 0, 2, 2, 2, 0]),
            ([0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]),
            ([0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0]),
            ([0, 2, 0, 2, 0, 0], [0, 0, 2, 0, 2, 0]),
        ]
        self.assertEqual(len(samples), 12)
        return samples

    def _true_recourse(self, Y_out, Y_ret) -> float:
        params = _sp(self.backend, R_out=self.R_out, R_ret=self.R_ret)
        return float(_evaluate(params, _candidate(Y_out, Y_ret)).upper_bound)

    def test_the_two_selections_really_are_different_duals(self):
        _res_mw, meta_mw = self._cut("mw")
        _res_dual, meta_dual = self._cut("dual")
        same_const = abs(
            float(meta_mw["const"]) - float(meta_dual["const"])
        ) <= 1e-9
        same_slopes = dict(meta_mw["coeff_yOUT"]) == dict(meta_dual["coeff_yOUT"])
        self.assertFalse(
            same_const and same_slopes,
            "the two dual selections returned the same cut, so this instance is not "
            "degenerate enough to exercise the property under test",
        )

    def test_degenerate_duals_ok(self):
        """Both cuts underestimate the true recourse at every sampled signature."""
        for mode in ("mw", "dual"):
            _res, meta = self._cut(mode)
            for Y_out, Y_ret in self._sampled_signatures():
                cut_val = _cut_value_at(meta, Y_out, Y_ret)
                true_val = self._true_recourse(Y_out, Y_ret)
                with self.subTest(mode=mode, Y_out=tuple(Y_out)):
                    self.assertLessEqual(
                        cut_val,
                        true_val + 1e-6,
                        f"the {mode} cut OVERESTIMATES the recourse at "
                        f"Y_out={Y_out}, Y_ret={Y_ret} "
                        f"(cut={cut_val:.6f} > true={true_val:.6f}). A cut that does "
                        "this excludes feasible schedules and can cut off the optimum.",
                    )

    def test_the_validity_check_is_not_vacuous(self):
        """A cut that was identically minus-infinity would pass the test above."""
        _res, meta = self._cut("dual")
        incumbent = _cut_value_at(meta, self.Y_out, self.Y_ret)
        true_val = self._true_recourse(self.Y_out, self.Y_ret)
        self.assertAlmostEqual(
            incumbent,
            true_val,
            places=5,
            msg="the cut is not tight at the incumbent, so the underestimation checks "
            "elsewhere could be passing on a cut that says nothing",
        )


# ---------------------------------------------------------------------------- gate
class TestTinyEnumeration(unittest.TestCase):
    """`tiny_enumeration`. Every generated cut valid at every feasible signature.

    The gate the work order asks for, at a size where "every feasible signature" is a
    finite set that can actually be enumerated rather than sampled. It is run across
    the boundary conditions named in the table: `p_min` below, equal to and above
    `W_max`; arrivals on a slot boundary; zero-wait boarding under both eligibility
    conventions.

    FAILS BEFORE: at `same_slot_eligibility="allow"` the slot recourse could not be
    built at all -- the convention was hard-coded to `forbid` in `solve_subproblem`
    and to `allow` in the minute path, and no run could set either.
    """

    T_TINY = 5

    @classmethod
    def setUpClass(cls):
        cls.backend = require_solver_backend()

    def _params(self, Q: int, p_minutes: float, eligibility: str) -> dict:
        return {
            "T": self.T_TINY,
            "S": 4.0,
            "Wmax_slots": 2,
            "p": float(p_minutes) / 30.0,
            "slot_resolution": 30,
            "trip_slots": 1,
            "Q": Q,
            "lp_solver": self.backend,
            "cut_mode": "dual",
            "eps_cut": 1e-8,
            "same_slot_eligibility": eligibility,
            # Arrivals include slot 0 (the boundary bucket) and a slot that can only be
            # reached at the far end of the window.
            "R_out": [5, 3, 0, 2, 0],
            "R_ret": [0, 4, 3, 0, 0],
        }

    def _signatures(self, Q: int):
        """Every signature the master could present, at this size.

        Departures are impossible in the last `trip_slots` slots and OUT is further
        restricted, so the enumeration is over the slots that can actually carry one.
        Enumerating more would test the cut at points no master can reach; enumerating
        fewer would leave the claim unproven where it matters.
        """
        from mobauto2_benders.signature import departures_are_possible

        ok_out, ok_ret = departures_are_possible(self.T_TINY, 1)
        out_slots = [t for t in range(self.T_TINY) if ok_out[t]]
        ret_slots = [t for t in range(self.T_TINY) if ok_ret[t]]
        for out_vals in itertools.product(range(Q + 1), repeat=len(out_slots)):
            for ret_vals in itertools.product(range(Q + 1), repeat=len(ret_slots)):
                Y_out = [0] * self.T_TINY
                Y_ret = [0] * self.T_TINY
                for t, v in zip(out_slots, out_vals):
                    Y_out[t] = v
                for t, v in zip(ret_slots, ret_vals):
                    Y_ret[t] = v
                # A slot cannot start more trips than there are vehicles.
                if any(Y_out[t] + Y_ret[t] > Q for t in range(self.T_TINY)):
                    continue
                yield Y_out, Y_ret

    def test_tiny_enumeration(self):
        # p_min below, equal to, and above W_max (60 minutes).
        for Q in (1, 2):
            for p_minutes in (30.0, 60.0, 120.0):
                for eligibility in ("forbid", "allow"):
                    with self.subTest(Q=Q, p_min=p_minutes, elig=eligibility):
                        self._one_cell(Q, p_minutes, eligibility)

    def _one_cell(self, Q: int, p_minutes: float, eligibility: str):
        params = self._params(Q, p_minutes, eligibility)
        signatures = list(self._signatures(Q))
        self.assertGreater(len(signatures), 4, "the enumeration is too small to mean anything")

        # Generate the cut at the fullest signature, where the recourse is cheapest and
        # the cut therefore has the most room to be wrong lower down.
        gen_out, gen_ret = max(signatures, key=lambda s: sum(s[0]) + sum(s[1]))
        res = _evaluate(params, _candidate(gen_out, gen_ret, Q))
        meta = (res.cuts or [res.cut])[0].metadata

        recourse = {}
        for Y_out, Y_ret in signatures:
            key = (tuple(Y_out), tuple(Y_ret))
            recourse[key] = float(
                _evaluate(params, _candidate(Y_out, Y_ret, Q)).upper_bound
            )

        for (Y_out, Y_ret), true_val in recourse.items():
            cut_val = _cut_value_at(meta, list(Y_out), list(Y_ret))
            self.assertLessEqual(
                cut_val,
                true_val + 1e-6,
                f"cut generated at Y_out={gen_out} overestimates the recourse at "
                f"Y_out={list(Y_out)}, Y_ret={list(Y_ret)}: "
                f"{cut_val:.6f} > {true_val:.6f}",
            )

        # Tight where it was generated, or it is not a Benders cut.
        self.assertAlmostEqual(
            _cut_value_at(meta, gen_out, gen_ret),
            recourse[(tuple(gen_out), tuple(gen_ret))],
            places=5,
        )

        # More capacity never costs more -- the E2 probe, over the whole enumeration
        # rather than a chain of three.
        for (Y_out, Y_ret), val in recourse.items():
            bigger = (
                tuple(min(Q, v + 1) for v in Y_out),
                tuple(Y_ret),
            )
            if bigger in recourse:
                self.assertLessEqual(
                    recourse[bigger],
                    val + 1e-6,
                    f"recourse ROSE when capacity was added: {Y_out} -> {bigger[0]}",
                )

    def test_the_two_conventions_give_different_arc_sets(self):
        """The eligibility flag must actually change the model at this size, or the
        `allow` half of the enumeration above is testing the `forbid` model twice.
        """
        from mobauto2_benders.recourse_fingerprint import (
            recourse_fingerprint,
            slot_recourse_model_for,
        )

        Y_out, Y_ret = [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]
        prints = set()
        for eligibility in ("forbid", "allow"):
            params = self._params(1, 56.0, eligibility)
            model = slot_recourse_model_for(
                params, Y_out, Y_ret, params["R_out"], params["R_ret"]
            )
            prints.add(recourse_fingerprint(model))
        self.assertEqual(
            len(prints), 2, "forbid and allow produced the same recourse LP"
        )

    def test_the_slot_recourse_and_the_validator_agree_on_a_slot_aligned_case(self):
        """The decomposed recourse and the common validator must price one schedule
        identically when the instance gives them nothing to disagree about.

        Arrivals are placed at the START of their slots and the eligibility convention
        is `forbid`, which under `departure_policy="start"` is exactly the slot rule.
        The two then share an arc set, and the only remaining difference -- minute vs
        slot granularity of the wait -- vanishes because every arrival sits on a slot
        boundary. Anything left is a defect in one of them.
        """
        from mobauto2_benders.minute_pricer import price_schedule_at_minutes

        delta = 30
        # 4 arrivals at minute 0 (slot 0), 3 at minute 30 (slot 1).
        arrivals = {"OUT": [0, 0, 0, 0, 30, 30, 30], "RET": []}
        R_out = [4, 3, 0, 0, 0]
        params = {
            "T": self.T_TINY,
            "S": 4.0,
            "Wmax_slots": 2,
            "p": 56.0 / delta,
            "slot_resolution": delta,
            "trip_slots": 1,
            "Q": 1,
            "lp_solver": self.backend,
            "cut_mode": "dual",
            "same_slot_eligibility": "forbid",
            "R_out": R_out,
            "R_ret": [0] * self.T_TINY,
        }
        Y_out, Y_ret = [0, 1, 1, 0, 0], [0] * self.T_TINY
        slot_obj = float(_evaluate(params, _candidate(Y_out, Y_ret, 1)).upper_bound)

        priced = price_schedule_at_minutes(
            departures={"OUT": [1, 2], "RET": []},
            requests=arrivals,
            slot_resolution=delta,
            seats=4.0,
            wmax_minutes=60.0,
            p_minutes=56.0,
            policy="start",
            lp_solver=self.backend,
            same_slot_eligibility="forbid",
        )
        # The slot objective is in slot units; the validator is in passenger-minutes.
        # One multiplication by delta, asserted rather than assumed (the shared
        # contract's objective clause).
        self.assertAlmostEqual(slot_obj * delta, priced.total_cost, places=6)


if __name__ == "__main__":
    unittest.main()
