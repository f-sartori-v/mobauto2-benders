"""The fail-closed contract for cuts injected into a branch-and-bound tree (D44).

None of this needs CPLEX, a tree, or a registered callback, which is the point:
the decision that a lazy cut cannot be un-added is a decision about validity
states, and it is cheaper to pin here than to discover from a run whose bound
turned out to exclude the optimum.

The loop's own policy is the opposite one and is tested elsewhere: it adds an
INVALID cut and drops the reported lower bound afterwards. These tests exist to
make sure that policy is never copied across by someone reading solver.py.
"""

from __future__ import annotations

import unittest

from mobauto2_benders.benders.lazy_cuts import (
    CallbackAbort,
    LazyCutStats,
    candidate_from_values,
    cut_to_linear_form,
    vet_cut_for_injection,
)
from mobauto2_benders.benders.types import (
    Cut,
    CutType,
    CutValidity,
    SubproblemResult,
)


def _cut(**meta) -> Cut:
    return Cut(
        name="benders_cut_test",
        cut_type=CutType.OPTIMALITY,
        metadata=dict(meta),
    )


def _valid_cut() -> Cut:
    return _cut(
        const=100.0,
        coeff_yOUT={(0, 0): -3.0, (1, 2): -1.5},
        coeff_yRET={(0, 1): -2.0},
    )


class CutToLinearForm(unittest.TestCase):
    def test_it_reconstructs_constant_and_slopes(self):
        const, coeffs = cut_to_linear_form(_valid_cut())
        self.assertAlmostEqual(const, 100.0)
        self.assertEqual(
            coeffs,
            {
                ("yOUT", 0, 0): -3.0,
                ("yOUT", 1, 2): -1.5,
                ("yRET", 0, 1): -2.0,
            },
        )

    def test_split_thetas_sum_their_constants(self):
        const, _ = cut_to_linear_form(
            _cut(
                const=0.0,
                const_out=40.0,
                const_ret=60.0,
                coeff_yOUT={(0, 0): -1.0},
            )
        )
        # const_out/const_ret take precedence over `const`, which is 0.0 here on
        # purpose: reading the wrong field would give a cut 100 lower and still
        # look like a plausible bound.
        self.assertAlmostEqual(const, 100.0)

    def test_zero_coefficients_are_dropped_but_an_all_zero_cut_aborts(self):
        const, coeffs = cut_to_linear_form(
            _cut(const=5.0, coeff_yOUT={(0, 0): -1.0, (0, 1): 0.0})
        )
        self.assertEqual(coeffs, {("yOUT", 0, 0): -1.0})

        with self.assertRaises(CallbackAbort) as ctx:
            cut_to_linear_form(_cut(const=5.0, coeff_yOUT={(0, 0): 0.0}))
        self.assertIn("zero coefficients", str(ctx.exception))

    def test_missing_slope_metadata_aborts_rather_than_defaulting(self):
        # The all-zero slope vector is the failure HANDLER_CENSUS records. In a
        # tree it reads as `theta >= const`, which is not visibly wrong.
        with self.assertRaises(CallbackAbort):
            cut_to_linear_form(_cut(const=5.0))


class VetCutForInjection(unittest.TestCase):
    def test_a_valid_cut_passes_through(self):
        res = SubproblemResult(
            is_feasible=True,
            cut=_valid_cut(),
            diagnostics={"cut_valid_lower_bound": True},
        )
        const, coeffs = vet_cut_for_injection(res, context="incumbent")
        self.assertAlmostEqual(const, 100.0)
        self.assertEqual(len(coeffs), 3)

    def test_an_invalid_cut_aborts(self):
        res = SubproblemResult(
            is_feasible=True,
            cut=_valid_cut(),
            diagnostics={"cut_valid_lower_bound": False},
        )
        with self.assertRaises(CallbackAbort) as ctx:
            vet_cut_for_injection(res, context="incumbent")
        self.assertIs(ctx.exception.validity, CutValidity.INVALID)

    def test_an_unknown_cut_aborts(self):
        # A cut with no `cut_valid_lower_bound` key at all: the producer said
        # nothing, so nothing is known.
        res = SubproblemResult(is_feasible=True, cut=_valid_cut(), diagnostics={})
        with self.assertRaises(CallbackAbort) as ctx:
            vet_cut_for_injection(res, context="incumbent")
        self.assertIs(ctx.exception.validity, CutValidity.UNKNOWN)

    def test_no_cut_aborts_even_though_the_loop_survives_it(self):
        """This is the case the loop treats as benign and the tree cannot.

        In the loop, NO_CUT means nothing was added, so whatever the bound was
        certified on still holds. In a callback, returning without adding a cut
        asserts the incumbent is acceptable -- and the subproblem has just
        declined to price it.
        """
        res = SubproblemResult(is_feasible=True, cut=None, diagnostics={})
        with self.assertRaises(CallbackAbort) as ctx:
            vet_cut_for_injection(res, context="incumbent")
        self.assertIs(ctx.exception.validity, CutValidity.NO_CUT)

    def test_validity_valid_with_no_cut_attached_aborts(self):
        res = SubproblemResult(
            is_feasible=True, cut=None, diagnostics={"cut_valid_lower_bound": True}
        )
        with self.assertRaises(CallbackAbort) as ctx:
            vet_cut_for_injection(res, context="incumbent")
        self.assertIn("no cut attached", str(ctx.exception))

    def test_every_non_valid_state_aborts(self):
        """Enumerated so a fifth validity state cannot be added silently.

        `CutValidity` growing a member that this module does not name would
        otherwise fall through to whatever branch happens to catch it.
        """
        seen = set()
        for diag, cut in (
            ({"cut_valid_lower_bound": False}, _valid_cut()),
            ({}, _valid_cut()),
            ({}, None),
        ):
            res = SubproblemResult(is_feasible=True, cut=cut, diagnostics=diag)
            with self.assertRaises(CallbackAbort) as ctx:
                vet_cut_for_injection(res, context="incumbent")
            seen.add(ctx.exception.validity)
        self.assertEqual(
            seen | {CutValidity.VALID},
            set(CutValidity),
            "a CutValidity member is not covered by the injection contract",
        )


class CandidateShape(unittest.TestCase):
    def test_it_uses_the_masters_own_key_convention(self):
        cand = candidate_from_values(lambda d, q, t: 1.0 if d == "yOUT" else 0.0, 2, 3)
        self.assertEqual(len(cand), 12)
        self.assertEqual(cand["yOUT[1,2]"], 1.0)
        self.assertEqual(cand["yRET[0,0]"], 0.0)


class Stats(unittest.TestCase):
    def test_validity_counts_accumulate(self):
        st = LazyCutStats()
        st.note_validity(CutValidity.VALID)
        st.note_validity(CutValidity.VALID)
        st.note_validity(CutValidity.INVALID)
        self.assertEqual(st.validity_counts, {"valid": 2, "invalid": 1})
        self.assertIsNone(st.aborted_reason)


if __name__ == "__main__":
    unittest.main()
