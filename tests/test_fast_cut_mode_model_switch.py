"""`cut_mode` selects a generator AND a model. The two config forms must agree.

`use_dual_slopes` historically named two unrelated things: the plain-dual cut
GENERATOR, and a MODEL switch flooring the per-slot capacity counts at one so
every tau carries a capacity row and therefore a pi. One key for both means an A/B
on either measures them summed.

S1b replaced the legacy boolean pair with a single `cut_mode` key, and the model
switch was re-keyed onto the RESOLVED mode (`_cut_mode_cfg == "dual"`) rather than
onto the legacy boolean. That is the right fix -- but it is invisible, and the
comments at the two sites still said "if using dual slopes" until D68. Anyone
reading them would conclude that `cut_mode: dual` leaves `use_dual_slopes` False
and therefore builds a DIFFERENT subproblem from the legacy form. It does not, and
this file is what keeps that true.

The failure being guarded is specific and would be silent: two configs naming one
generator, solving two different LPs, and every number from them filed in one
table.

An abandoned branch (`7457e58`, never merged) fixed the same coupling by adding a
separate `min_one_capacity_layer` key that inherited from `use_dual_slopes`. That
approach does not survive `cut_mode`, whose resolution leaves the legacy boolean
False -- inheriting from it would have turned the flooring off for every migrated
config. Recorded here so the abandoned design is not re-proposed.

No solver needed for the config half; the LP half skips without a backend.

Run just these:
    python -m unittest tests.test_fast_cut_mode_model_switch -v
"""

from __future__ import annotations

import copy
import unittest

import yaml

import _helpers
from _helpers import require_solver_backend

FIXTURE = _helpers.REPO_ROOT / "tests" / "fixtures" / "soundness.yaml"

LEGACY = {"use_magnanti_wong": False, "use_dual_slopes": True}
MODERN = {"cut_mode": "dual"}


def _cfg(patch: dict):
    """Load the soundness fixture with the cut-mode keys replaced by `patch`."""
    import tempfile
    from pathlib import Path

    from mobauto2_benders.config import load_config

    raw = copy.deepcopy(yaml.safe_load(FIXTURE.read_text(encoding="utf-8")))
    for key in ("use_magnanti_wong", "use_dual_slopes", "cut_mode"):
        raw["subproblem"].pop(key, None)
    raw["subproblem"].update(patch)
    path = Path(tempfile.mkdtemp(prefix="cut_mode_")) / "c.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_config(path)


class TestBothFormsResolveToTheSameMode(unittest.TestCase):
    def test_they_agree_on_the_generator(self):
        self.assertEqual(_cfg(LEGACY).subproblem.cut_mode, "dual")
        self.assertEqual(_cfg(MODERN).subproblem.cut_mode, "dual")

    def test_the_legacy_boolean_is_not_what_the_model_switch_reads(self):
        """The fact that makes this file necessary rather than obvious.

        `cut_mode: dual` leaves `use_dual_slopes` False. If the flooring were keyed
        off that boolean -- as it once was, and as the abandoned branch's fix would
        have left it -- the two forms would build different LPs.
        """
        self.assertTrue(_cfg(LEGACY).subproblem.use_dual_slopes)
        self.assertFalse(_cfg(MODERN).subproblem.use_dual_slopes)

    def test_the_model_switch_is_keyed_off_the_resolved_mode(self):
        import inspect

        from mobauto2_benders.problem import subproblem_impl

        src = inspect.getsource(subproblem_impl)
        self.assertIn('use_dual = _cut_mode_cfg == "dual"', src)
        self.assertNotIn(
            'use_dual = bool(params.get("use_dual_slopes"',
            src,
            "the model switch is reading the legacy boolean again",
        )


class TestBothFormsBuildTheSameSubproblem(unittest.TestCase):
    """The invariant itself, asserted on the LP rather than on the config."""

    @classmethod
    def setUpClass(cls):
        require_solver_backend()

    def _evaluate(self, patch: dict):
        from mobauto2_benders.app import _prepare_params
        from mobauto2_benders.problem.subproblem_impl import ProblemSubproblem

        cfg = _cfg(patch)
        _helpers.repoint_solvers(cfg)
        mp, sp = _prepare_params(cfg, {})
        T = int(mp.get("T") or (int(mp["T_minutes"]) // int(mp["slot_resolution"])))
        sp["T"] = T
        Q = int(mp.get("Q", 2))

        # Deliberately leave most slots EMPTY. `max(1, K)` differs from `K` only
        # where K is zero, so a dense schedule would make the two forms agree for a
        # reason weaker than the one under test.
        cand = {
            f"y{d}[{q},{t}]": 0.0
            for d in ("OUT", "RET")
            for q in range(Q)
            for t in range(T)
        }
        for t in (3, 6, 9):
            cand[f"yOUT[0,{t}]"] = 1.0

        res = ProblemSubproblem(sp).evaluate(dict(cand))
        cut = (res.cuts or [res.cut])[0]
        slopes = {
            (int(q), int(tau)): round(float(v), 9)
            for (q, tau), v in dict(cut.metadata.get("coeff_yOUT") or {}).items()
        }
        return float(res.upper_bound), slopes

    def test_the_candidate_leaves_slots_empty(self):
        """Guard against the test agreeing vacuously on a dense schedule."""
        from mobauto2_benders.app import _prepare_params

        mp, _sp = _prepare_params(_cfg(MODERN), {})
        T = int(mp.get("T") or (int(mp["T_minutes"]) // int(mp["slot_resolution"])))
        self.assertGreater(T - 3, 5, "too few empty slots for max(1,K) to bite")

    def test_the_recourse_agrees(self):
        self.assertEqual(self._evaluate(LEGACY)[0], self._evaluate(MODERN)[0])

    def test_the_cut_agrees_coefficient_by_coefficient(self):
        """Value equality alone is too weak: two different duals give two different
        cuts at the same recourse, and the master would take whichever it was
        handed. This is the same reasoning E1 uses."""
        self.assertEqual(self._evaluate(LEGACY)[1], self._evaluate(MODERN)[1])


if __name__ == "__main__":
    unittest.main()
