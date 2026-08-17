"""Bound recovery from CPLEX log text (audit N2). No solver required.

cplex_log.py is third in the fallback chain
solver_results -> cplex_api -> cplex_log -> computed
and recovers bounds by regex over CPLEX's free-text output, which is inherently
version-fragile. If a CPLEX upgrade changes the log format, best_bound silently
degrades to None and the chain falls through to "computed", which itself needs
both an incumbent and a bound -- three fallbacks that could all miss at once
with no loud failure.

NOTE: these fixtures are written from the format the parser expects, not
captured from a real solver run, because emit_reports is off by default so no
log was to hand. Replacing them with genuinely captured output would make this
a stronger drift detector; the parsing contract it pins is the same either way.
"""

from __future__ import annotations

import unittest

import _helpers  # noqa: F401  (puts src/ on sys.path)
from mobauto2_benders.benders.cplex_log import parse_cplex_log_text

SUMMARY_LOG = """
Version identifier: 22.1.1.0
Populate: phase I
MIP emphasis: balance optimality and feasibility.

Solution pool: 1 solution saved.

Best Integer = 4561.99006     Best Bound = 4558.34955     Gap = 0.08 %
"""

NODE_TABLE_LOG = """
        Nodes                                         Cuts/
   Node  Left     Objective  IInf  Best Integer    Best Bound    ItCnt     Gap

*     0+    0                       5000.0000        0.0000            0  100.00%
      0     2     3000.0000    12   4561.9901     4558.3496      812    0.08%
"""

NO_BOUNDS_LOG = "MIP - Integer infeasible.\nNo solution exists.\n"


class TestCplexLogParsing(unittest.TestCase):
    def test_summary_section(self):
        got = parse_cplex_log_text(SUMMARY_LOG)
        self.assertAlmostEqual(got["best_integer"], 4561.99006, places=4)
        self.assertAlmostEqual(got["best_bound"], 4558.34955, places=4)
        self.assertEqual(got["source"], "summary")

    def test_gap_is_a_ratio_not_a_percentage(self):
        """The parser divides by 100. Getting this wrong would misreport the gap
        by two orders of magnitude while looking entirely plausible."""
        got = parse_cplex_log_text(SUMMARY_LOG)
        self.assertIsNotNone(got["gap"])
        self.assertLess(got["gap"], 1.0)
        self.assertAlmostEqual(got["gap"], 0.0008, places=6)

    def test_node_table_fallback(self):
        got = parse_cplex_log_text(NODE_TABLE_LOG)
        self.assertIsNotNone(got["best_integer"])
        self.assertIsNotNone(got["best_bound"])

    def test_unparseable_log_yields_none_not_garbage(self):
        """Failure must be visible as None so the caller can record
        bound_source "unavailable", never a plausible-looking wrong number."""
        got = parse_cplex_log_text(NO_BOUNDS_LOG)
        self.assertIsNone(got["best_integer"])
        self.assertIsNone(got["best_bound"])

    def test_empty_input_is_safe(self):
        got = parse_cplex_log_text("")
        self.assertIsNone(got["best_bound"])


if __name__ == "__main__":
    unittest.main()


# Genuinely captured output, unlike the fixtures above: these lines are copied from
# the CPLEX log of the monolithic MILP solving the baseline_d9 instance on
# 2026-08-13 (configs/milp/baseline_d9_monolith.yaml), the run that reproduced the
# 4183.24 reference. Trimmed to the node table and the terminal line.
CAPTURED_BASELINE_D9_LOG = """
        Nodes                                         Cuts/
   Node  Left     Objective  IInf  Best Integer    Best Bound    ItCnt     Gap

*     0+    0                        15000.0000      122.0000            99.19%
      0     0     3270.7158    84    15000.0000     3270.7158      342   78.19%
*     0+    0                         4365.2400     3889.0352            10.91%
      0     2     3889.0352    50     4365.2400     3889.0754      500   10.90%
*    31    21      integral     0     4190.7400     3889.4155     1130    7.19%
    778   231     4044.4823    20     4190.7400     4004.3096    15068    4.45%
*   935   237      integral     0     4187.7400     4025.3058    17876    3.88%
*  1752     1      integral     0     4183.2400     4182.2400    27412    0.02%

MIP - Integer optimal solution:  Objective =  4.1832400000e+03
Solution pool: 8 solutions saved.
"""

# The optimum of the baseline_d9 instance, confirmed by the monolith run this log
# came from. The parser must not report anything else from this text.
BASELINE_D9_OPTIMUM = 4183.24

# What the parser returned before the D50 fix: the incumbent as of node 778, because
# every line announcing a NEW incumbent writes the word "integral" in the Objective
# column and the pattern demanded a number there.
STALE_VALUE_BEFORE_FIX = 4190.74


class TestIncumbentLinesAreNotSkipped(unittest.TestCase):
    """D50. The defect that made 4190.74 look like the optimum.

    4190.74 was the project's reference optimum for months and had to be withdrawn
    (D30). It is reproducible from this log as a parsing artifact, which is the most
    likely explanation for how it was recorded in the first place.
    """

    def test_the_optimum_is_reported_not_a_superseded_incumbent(self):
        res = parse_cplex_log_text(CAPTURED_BASELINE_D9_LOG)
        self.assertAlmostEqual(res["best_integer"], BASELINE_D9_OPTIMUM, places=6)
        self.assertNotAlmostEqual(
            res["best_integer"],
            STALE_VALUE_BEFORE_FIX,
            places=6,
            msg="the parser is reporting the node-778 incumbent again",
        )

    def test_a_proven_optimum_pins_the_bound_to_the_objective(self):
        res = parse_cplex_log_text(CAPTURED_BASELINE_D9_LOG)
        self.assertAlmostEqual(res["best_bound"], BASELINE_D9_OPTIMUM, places=6)
        self.assertAlmostEqual(res["gap"], 0.0, places=9)
        self.assertEqual(res["source"], "optimal_line")

    def test_incumbent_lines_are_matched_without_the_terminal_line(self):
        """A run stopped on the clock has a node table and no 'Integer optimal' line.

        This is the path that actually matters in the Benders loop, where master
        solves are truncated. Before the fix it returned 4190.74 here too.
        """
        truncated = CAPTURED_BASELINE_D9_LOG.split("MIP - Integer optimal")[0]
        res = parse_cplex_log_text(truncated)
        self.assertEqual(res["source"], "node_table")
        self.assertAlmostEqual(res["best_integer"], BASELINE_D9_OPTIMUM, places=6)
        self.assertAlmostEqual(res["best_bound"], 4182.24, places=6)

    def test_tolerance_variant_does_not_claim_an_unproven_bound(self):
        """'Integer optimal, tolerance' proves optimality only within a gap.

        Pinning the bound to the objective there would report a tighter bound than
        CPLEX proved -- the one direction that turns a conservative parse error into
        an invalid lower bound.
        """
        text = CAPTURED_BASELINE_D9_LOG.replace(
            "MIP - Integer optimal solution:",
            "MIP - Integer optimal, tolerance (0.0001/1e-06):",
        )
        res = parse_cplex_log_text(text)
        self.assertAlmostEqual(res["best_integer"], BASELINE_D9_OPTIMUM, places=6)
        self.assertAlmostEqual(
            res["best_bound"],
            4182.24,
            places=6,
            msg="claimed the objective as a proven bound on a tolerance-terminated solve",
        )

    def test_the_stale_value_really_is_in_this_log(self):
        """Guard against a fixture that no longer exercises the defect.

        If the 4190.74 lines were ever removed from the fixture, every assertion
        above would still pass while testing nothing.
        """
        self.assertIn("4190.7400", CAPTURED_BASELINE_D9_LOG)
        self.assertIn("integral", CAPTURED_BASELINE_D9_LOG)


class TestBothPackagesShareTheFix(unittest.TestCase):
    """The parser is duplicated in mobauto2_milp; the copies must not drift."""

    def test_milp_copy_parses_the_same_log_the_same_way(self):
        from mobauto2_milp.cplex_log import parse_cplex_log_text as milp_parse

        a = parse_cplex_log_text(CAPTURED_BASELINE_D9_LOG)
        b = milp_parse(CAPTURED_BASELINE_D9_LOG)
        for key in ("best_integer", "best_bound", "gap", "source"):
            self.assertEqual(a[key], b[key], f"{key} differs between the two copies")
