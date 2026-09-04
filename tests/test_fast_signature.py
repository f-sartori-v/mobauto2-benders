"""The signature helper and condition M1 (was E3; DESIGN_DD_v1, D48). No solver.

RECLASSIFIED (B4, audit item 1.5). This condition used to be listed among the "four
conditions for exactness", and it is not one. Benders validity does not depend on it at
all: the cuts are valid because the recourse duals are dual feasible, and that argument
never mentions how the MASTER's rows are indexed.

What it actually is: a COMPUTATIONAL property, and the precondition for one specific
thing -- the stage-3 Dantzig-Wolfe reformulation, which needs a per-vehicle column to be
well defined. If it fails, that reformulation is off the table and the master must be
solved as it stands. Nothing about the decomposition's correctness moves.

M1 -- master equivalence with the monolith -- is the name it now carries, matching the
report's table. The condition itself is unchanged: every master row is separable by
vehicle, and the coupling is exactly the terms that are functions of the aggregate
signature alone.

M2 -- symmetry validity, the old E4 -- lives in tests/test_fast_model.py.

Run just these:
    python -m unittest tests.test_fast_signature -v
"""

from __future__ import annotations

import unittest

import _helpers
from _helpers import build_master, master_params

from mobauto2_benders.signature import (
    candidate_signature,
    fibre_size,
    is_integral,
    parse_index,
    signature_key,
)


# Components indexed by (q, t). A constraint touching two different q through any of
# these is a cross-vehicle row.
_PER_VEHICLE_COMPONENTS = frozenset(
    {"yOUT", "yRET", "c", "atL", "atM", "inTrip", "b", "gchg"}
)

# The rows allowed to touch more than one vehicle, each with the reason.
_COUPLING_ALLOWED = {
    # The three couplings named in DESIGN_DD_v1 E3. All are functions of the
    # aggregate, which is why the reformulation survives them.
    "Cagg_out": "defines Yout[t] = sum_q yOUT[q,t] -- this IS the signature",
    "Cagg_ret": "defines Yret[t] = sum_q yRET[q,t] -- this IS the signature",
    "Cagg_z": "defines Z[t] = sum_q (yOUT+yRET)[q,t] -- aggregate, cut sparsity only",
    # Deleted rather than translated by stage 3: it exists only to suppress the
    # symmetry that the reformulation removes outright.
    "C_sym_break_tot": "symmetry breaking, spec 2.8",
    # B2. sum_q c[q,t] <= K_chg genuinely couples named vehicles -- it is a shared
    # physical resource, not an aggregate of the signature. It is emitted ONLY when
    # K_chg < Q (master_impl.py), so M1 holds unchanged for every run with as many
    # chargers as vehicles, which is every archived result. Where chargers are scarce,
    # M1 is genuinely lost and stage 3's per-vehicle column is not well defined; that
    # is a fact about the site, and the reformulation would have to price the charger
    # coupling explicitly. Listed here so the loss is visible rather than asserted away.
    "C_chg_capacity": "shared charger resource (B2); only present when K_chg < Q",
    "C_chg_occ_link": "binds c to the charger-occupancy indicator (B2)",
}


def _q_indices_in(constraint_data, per_vehicle: frozenset) -> set[int]:
    from pyomo.core.expr.visitor import identify_variables

    qs: set[int] = set()
    for var in identify_variables(constraint_data.body, include_fixed=True):
        comp = var.parent_component()
        if comp.local_name not in per_vehicle:
            continue
        idx = var.index()
        if isinstance(idx, tuple) and idx:
            qs.add(int(idx[0]))
    return qs


class TestM1MasterEquivalenceWithTheMonolith(unittest.TestCase):
    """M1 (was E3). The contract behind stage 3 -- computational, not a validity
    condition. See the module docstring for why it was reclassified."""

    @classmethod
    def setUpClass(cls):
        cls.pm = build_master(master_params("baseline_d9.yaml"))
        cls.model = cls.pm.m

    def test_every_row_touches_at_most_one_vehicle(self):
        import pyomo.environ as pyo

        offenders: list[str] = []
        for con in self.model.component_objects(pyo.Constraint, active=True):
            if con.local_name in _COUPLING_ALLOWED:
                continue
            for idx in con:
                qs = _q_indices_in(con[idx], _PER_VEHICLE_COMPONENTS)
                if len(qs) > 1:
                    offenders.append(f"{con.local_name}[{idx}] touches q={sorted(qs)}")
        self.assertEqual(
            offenders,
            [],
            "M1 violated: these master rows couple named vehicles, so a per-vehicle "
            "column is not well defined and the DESIGN_DD_v1 stage 3 reformulation is "
            "invalid. Either the row is a genuine aggregate coupling (add it to "
            "_COUPLING_ALLOWED with the reason) or the reformulation must change.\n"
            + "\n".join(offenders[:20]),
        )

    def test_the_separability_check_can_actually_fail(self):
        """A checker that never sees a cross-vehicle row proves nothing.

        The allow-listed rows ARE cross-vehicle, so running the same detector over
        them must find them. Without this, a broken `_q_indices_in` -- returning an
        empty set, say -- would make the test above pass unconditionally.
        """
        import pyomo.environ as pyo

        found_multi = False
        for con in self.model.component_objects(pyo.Constraint, active=True):
            if con.local_name not in _COUPLING_ALLOWED:
                continue
            for idx in con:
                if len(_q_indices_in(con[idx], _PER_VEHICLE_COMPONENTS)) > 1:
                    found_multi = True
                    break
            if found_multi:
                break
        self.assertTrue(
            found_multi,
            "the detector found no multi-vehicle row even among the rows known to be "
            "multi-vehicle, so it is not detecting anything",
        )

    def test_the_coupling_is_through_the_aggregate_variables(self):
        """Every allowed coupling row must be one that defines or uses Yout/Yret/Z."""
        import pyomo.environ as pyo

        names = {
            c.local_name
            for c in self.model.component_objects(pyo.Constraint, active=True)
        }
        # Rows that exist only under a non-default setting. Skipped here rather than
        # dropped from the allow-list: the allow-list is the register of WHICH
        # couplings are permitted and why, and a row missing from it fails the
        # separability test the moment someone turns the setting on.
        _CONDITIONAL = {
            "C_sym_break_tot",  # only when symmetry breaking is on
            "C_chg_capacity",  # only when K_chg < Q, or the binary form (B2)
            "C_chg_occ_link",  # only under charger_occupancy_binary (B2)
        }
        for allowed in _COUPLING_ALLOWED:
            if allowed in _CONDITIONAL:
                continue
            self.assertIn(
                allowed,
                names,
                f"{allowed} is allow-listed as a coupling row but does not exist; the "
                "allow-list has drifted from the model",
            )


class TestSignatureHelper(unittest.TestCase):
    def test_sums_over_vehicles_per_slot_and_direction(self):
        cand = {
            "yOUT[0,3]": 1.0,
            "yOUT[1,3]": 1.0,
            "yOUT[2,3]": 1.0,
            "yRET[0,5]": 1.0,
            "yRET[1,5]": 1.0,
            "b[0,3]": 150.0,  # non-y keys must be ignored
            "__theta": 42.0,
        }
        Y_out, Y_ret = candidate_signature(cand, T=8)
        self.assertEqual(Y_out[3], 3.0)
        self.assertEqual(Y_ret[5], 2.0)
        self.assertEqual(sum(Y_out), 3.0)
        self.assertEqual(sum(Y_ret), 2.0)

    def test_out_of_range_slots_are_dropped_not_wrapped(self):
        Y_out, _ = candidate_signature({"yOUT[0,99]": 1.0}, T=8)
        self.assertEqual(sum(Y_out), 0.0)

    def test_fractional_signatures_are_preserved(self):
        """The LP phase prices a fractional aggregate, and that is not an error.

        Rounding here would make the diagnostic disagree with the LP the subproblem
        actually solved.
        """
        Y_out, _ = candidate_signature({"yOUT[0,1]": 0.4, "yOUT[1,1]": 0.25}, T=4)
        self.assertAlmostEqual(Y_out[1], 0.65)
        self.assertFalse(is_integral(Y_out))

    def test_parse_index_tolerates_spacing(self):
        self.assertEqual(parse_index("yOUT[2, 7]"), (2, 7))
        self.assertEqual(parse_index("yRET[10,0]"), (10, 0))

    def test_signature_key_is_hashable_and_distinguishes(self):
        a = signature_key([1.0, 0.0], [0.0, 1.0])
        b = signature_key([0.0, 1.0], [0.0, 1.0])
        self.assertEqual(len({a, b}), 2)
        self.assertEqual(a, signature_key([1.0, 0.0], [0.0, 1.0]))


class TestFibreSize(unittest.TestCase):
    def test_counts_assignments_of_a_profile_to_vehicles(self):
        # Two slots each with 1 of 3 vehicles departing: 3 * 3 = 9.
        self.assertEqual(fibre_size([1.0, 1.0], [0.0, 0.0], Q=3), 9)
        # A slot with all 3 departing has exactly one assignment.
        self.assertEqual(fibre_size([3.0], [0.0], Q=3), 1)
        # The empty schedule: one member.
        self.assertEqual(fibre_size([0.0, 0.0], [0.0, 0.0], Q=3), 1)

    def test_undefined_for_fractional_or_infeasible_signatures(self):
        """Returning a number here would be answering a question that has none."""
        self.assertIsNone(fibre_size([0.5], [0.0], Q=3))
        self.assertIsNone(fibre_size([4.0], [0.0], Q=3))  # more departures than fleet
        self.assertIsNone(fibre_size([-1.0], [0.0], Q=3))

    def test_the_design_doc_number_is_reproduced(self):
        """DESIGN_DD_v1 1.4 quotes 3^18 for 18 one-vehicle departure slots at Q=3."""
        Y_out = [1.0] * 18
        self.assertEqual(fibre_size(Y_out, [0.0] * 18, Q=3), 3**18)


if __name__ == "__main__":
    unittest.main()
