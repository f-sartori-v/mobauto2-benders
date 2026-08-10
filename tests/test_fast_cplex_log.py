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
