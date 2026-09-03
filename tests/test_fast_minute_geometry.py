"""The primal and its Magnanti-Wong dual must be built over the SAME arcs. No solver.

WHY THIS FILE EXISTS. Two changes landed on this function from different directions and
their merge had exactly one correct resolution:

  * D86 (main) extracted the arc set, pools and departure-minute map into
    `_minute_recourse_geometry`, so that `solve_minute_recourse` (the primal) and
    `solve_mw_dual_minute` (its MW dual) could not drift apart.
  * B6 (this branch) added `same_slot_eligibility`, which CHANGES the arc set, and
    originally applied it inside `solve_minute_recourse`.

Taking both sides' text verbatim -- which is what a conflict resolution naturally
produces -- would have left the primal filtering arcs by `w0` while the shared geometry
did not. The dual would then have been built over a strictly larger arc set than the
primal it is restricted to: dual variables for arcs the primal does not have, selecting
on a face that is not the primal's optimal face. That is not a weaker cut, it is an
invalid one, and nothing downstream would have reported it -- the MW path returns `None`
on weak-duality violations and falls back silently to the plain dual.

So the invariant is pinned here rather than trusted: one geometry, both consumers, under
both conventions.

Run just these:
    python -m unittest tests.test_fast_minute_geometry -v
"""

from __future__ import annotations

import unittest

import _helpers  # noqa: F401  (puts src/ on sys.path)

from mobauto2_benders.minute_pricer import (
    OUT,
    RET,
    _minute_recourse_geometry,
    min_wait_minutes,
)

T = 8
SLOT = 30
WMAX = 60.0
# Arrivals ON slot boundaries as well as inside slots. The boundary ones are the only
# arrivals the two conventions treat differently at delta=30, so an instance without
# them cannot tell `forbid` from `allow`.
REQUESTS = {
    OUT: [0, 30, 60, 61, 90, 120, 121, 150],
    RET: [30, 45, 60, 90, 91, 120, 150, 151],
}


def _geo(eligibility: str, offsets=None):
    return _minute_recourse_geometry(
        T, SLOT, WMAX, REQUESTS, "start", offsets, eligibility
    )


class TestTheGeometryCarriesTheConvention(unittest.TestCase):
    def test_the_two_conventions_give_different_arc_sets(self):
        forbid = _geo("forbid")
        allow = _geo("allow")
        self.assertNotEqual(
            set(forbid.arcs[OUT]),
            set(allow.arcs[OUT]),
            "the geometry ignores same_slot_eligibility, so the primal and the MW "
            "dual would agree with each other while both ignoring the convention",
        )
        # `allow` is strictly the larger set: it admits every `forbid` arc plus the
        # zero-wait ones.
        for d in (OUT, RET):
            with self.subTest(direction=d):
                self.assertTrue(set(forbid.arcs[d]) < set(allow.arcs[d]))

    def test_the_extra_arcs_are_exactly_the_zero_wait_ones(self):
        forbid = _geo("forbid")
        allow = _geo("allow")
        for d in (OUT, RET):
            extra = set(allow.arcs[d]) - set(forbid.arcs[d])
            self.assertTrue(extra, f"no zero-wait arcs at all in direction {d}")
            for (m, t, k) in extra:
                with self.subTest(direction=d, arc=(m, t, k)):
                    self.assertEqual(allow.dep_minute[(t, k)] - float(m), 0.0)

    def test_the_convention_travels_on_the_geometry(self):
        self.assertEqual(_geo("forbid").same_slot_eligibility, "forbid")
        self.assertEqual(_geo("allow").same_slot_eligibility, "allow")

    def test_the_default_is_forbid(self):
        default = _minute_recourse_geometry(T, SLOT, WMAX, REQUESTS)
        self.assertEqual(default.same_slot_eligibility, "forbid")
        self.assertEqual(set(default.arcs[OUT]), set(_geo("forbid").arcs[OUT]))

    def test_an_unknown_convention_is_refused(self):
        with self.assertRaises(ValueError):
            _geo("sometimes")

    def test_the_grouped_views_agree_with_the_arc_list(self):
        """`by_tau` and `by_minute` are what the constraint rules iterate.

        A filter applied to `arcs` but not to the groupings would build rows over arcs
        the variable set does not contain -- the same divergence one level down.
        """
        for eligibility in ("forbid", "allow"):
            geo = _geo(eligibility)
            for d in (OUT, RET):
                with self.subTest(elig=eligibility, direction=d):
                    from_tau = {a for group in geo.by_tau[d].values() for a in group}
                    from_min = {a for group in geo.by_minute[d].values() for a in group}
                    self.assertEqual(from_tau, set(geo.arcs[d]))
                    self.assertEqual(from_min, set(geo.arcs[d]))


class TestBothConsumersShareOneGeometry(unittest.TestCase):
    """The merge invariant, read off the source rather than inferred from behaviour.

    A behavioural test would need a solver and would only catch the divergence on an
    instance where it changes the answer. The structural claim -- both functions call
    the one factory, and neither rebuilds arcs itself -- is what actually has to hold.
    """

    @classmethod
    def setUpClass(cls):
        import inspect

        from mobauto2_benders import minute_pricer

        cls.primal = inspect.getsource(minute_pricer.solve_minute_recourse)
        cls.dual = inspect.getsource(minute_pricer.solve_mw_dual_minute)

    def test_both_call_the_shared_geometry(self):
        for name, src in (("primal", self.primal), ("dual", self.dual)):
            with self.subTest(fn=name):
                self.assertIn("_minute_recourse_geometry(", src)

    def test_neither_rebuilds_the_arc_set_itself(self):
        """The literal that would reintroduce the drift."""
        for name, src in (("primal", self.primal), ("dual", self.dual)):
            with self.subTest(fn=name):
                self.assertNotIn(
                    "<= float(wmax_minutes)",
                    src,
                    f"{name} filters arcs itself instead of reading the shared "
                    "geometry; the two will diverge the moment the filter changes on "
                    "one side, which is precisely how B6 and D86 collided",
                )

    def test_both_accept_the_convention(self):
        for name, src in (("primal", self.primal), ("dual", self.dual)):
            with self.subTest(fn=name):
                self.assertIn("same_slot_eligibility", src)


class TestMinWaitMinutes(unittest.TestCase):
    def test_forbid_is_one_minute(self):
        self.assertEqual(min_wait_minutes("forbid"), 1.0)

    def test_allow_is_zero(self):
        self.assertEqual(min_wait_minutes("allow"), 0.0)

    def test_anything_else_is_refused(self):
        for bad in ("Forbid ", "", "none", None):
            with self.subTest(value=bad):
                with self.assertRaises((ValueError, TypeError)):
                    min_wait_minutes(bad)


if __name__ == "__main__":
    unittest.main()
