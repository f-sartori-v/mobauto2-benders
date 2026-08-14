"""Stage 3, step 0: is the per-vehicle trajectory set small enough to ENUMERATE?

    python scripts/stage3_size_enumeration.py

No solver. A counting pass, seconds.

WHY THIS COMES FIRST. `DESIGN_DD_v1.md` stage 3 assumes `J` cannot be enumerated and
therefore goes straight to column generation -- which needs a pricing DP, which needs a
dominance rule the design itself flags as unproven, because `charge_before_idle` makes
charging at `t` depend on `c[t-1]`. That is three unproven things stacked before the
first number.

If `J` is enumerable at the small instance, the Dantzig-Wolfe master can be built
EXACTLY, with no pricing and no dominance rule, and the only question stage 3 actually
asks -- does the reformulated LP root beat the compact one? -- gets answered without any
of that machinery. If the root does not move, stage 3 dies for the price of this script
plus one LP.

WHAT IS COUNTED, AND WHY IT IS AN UPPER BOUND. A Dantzig-Wolfe column is a per-vehicle
departure pattern `y^j`, because the aggregation rows couple only `Y_d[tau] = sum_j
lambda_j y^j_d[tau]`. Battery and charging are internal to a column. So this counts
patterns admitted by the LOCATION dynamics and the horizon fixings alone:

  C2a/C2b/C2c  a vehicle departs OUT only from Longvilliers and RET only from Massy,
               and a departure at `u` arrives `trip_slots` later -- so patterns strictly
               alternate OUT, RET, OUT, ...
  C1b/C1c      no action while in transit
  fixings      no departure at t=0; none at `t >= T - trip_slots`; no OUT at
               `t >= T - 2*trip_slots`; `atL[T-1] = 1` and `inTrip[T-1] = 0`

This is EXACT for those constraints -- the recursion carries the full location state, so
no dominance argument is involved and nothing here can be wrong in the permissive
direction by accident. The battery block (C4, C5) can only REMOVE patterns, so the count
printed is a true upper bound on `|J|`. That is exactly what a sizing question needs: a
bound that cannot be an underestimate.

The initial location comes from `initial_actions`; `IDL` and `OUT` both start at
Longvilliers, `RET` starts at Massy.
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

AT_L = "L"
AT_M = "M"


def count_patterns(T: int, trip_slots: int, start_at_massy: bool) -> tuple[int, dict[int, int]]:
    """(total, {n_trips: count}) of location-feasible single-vehicle patterns."""

    # State: (t, location, arrival_slot_if_in_transit, destination_if_in_transit).
    # Encoded as (t, loc) when settled, or (t, "T", arrival_t, dest) when in transit.
    @lru_cache(maxsize=None)
    def walk(t: int, loc: str, arrive_at: int, dest: str) -> tuple[tuple[int, int], ...]:
        """Counts of completions from slot `t`, keyed by trips taken from here on."""
        # LAND FIRST. A trip arriving exactly at T-1 is feasible -- the master fixes
        # atL[T-1]=1 and inTrip[T-1]=0, and C2a_locL is satisfied by that arrival. An
        # earlier version of this walk tested the terminal condition before processing
        # the arrival and so rejected every pattern whose last RET departs at T-1-trip_slots.
        # That made the pool a strict subset, which RAISES the LP root -- the error
        # flattered the result, and only the column-generation cross-check caught it.
        if arrive_at == t:
            loc, arrive_at, dest = dest, -1, ""

        if t == T - 1:
            # atL[T-1] = 1 and inTrip[T-1] = 0 are fixed.
            if arrive_at >= 0 or loc != AT_L:
                return ()
            return ((0, 1),)

        if arrive_at >= 0:
            # Still in transit. No action is available.
            return walk(t + 1, loc, arrive_at, dest)

        acc: dict[int, int] = {}

        def absorb(res, extra_trips: int) -> None:
            for k, v in res:
                acc[k + extra_trips] = acc.get(k + extra_trips, 0) + v

        # Idle (this also covers charging, which is not part of the pattern).
        absorb(walk(t + 1, loc, -1, ""), 0)

        departure_allowed = t >= 1 and t < T - trip_slots
        if departure_allowed:
            if loc == AT_L and t < max(0, T - 2 * trip_slots):
                # OUT: leaves Longvilliers, arrives at Massy trip_slots later.
                absorb(walk(t + 1, AT_L, t + trip_slots, AT_M), 1)
            elif loc == AT_M:
                absorb(walk(t + 1, AT_M, t + trip_slots, AT_L), 1)

        return tuple(sorted(acc.items()))

    start = AT_M if start_at_massy else AT_L
    res = walk(0, start, -1, "")
    by_trips = {k: v for k, v in res}
    return sum(by_trips.values()), by_trips


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/phase1/rq5_benders_minute_p56.yaml")
    args = ap.parse_args()

    from mobauto2_benders.config import load_config

    cfg = load_config(args.config)
    delta = int(cfg.model.time.slot_resolution)
    T = int(cfg.model.time.T_minutes) // delta
    trip_slots = int(cfg.model.time.trip_duration_minutes) // delta
    Q = int(cfg.model.fleet.Q)
    actions = list(cfg.model.fleet.initial_actions)[:Q]

    print("=" * 78)
    print("Stage 3 sizing -- can the per-vehicle trajectory set be enumerated?")
    print(f"  {args.config}")
    print(f"  T={T} slots of {delta}min   trip_slots={trip_slots}   Q={Q}   "
          f"initial_actions={actions}")
    print("=" * 78)

    total, by_trips = count_patterns(T, trip_slots, start_at_massy=(actions[0] == "RET"))
    print(f"\nlocation-feasible patterns per vehicle: {total:,}")
    print("  (upper bound on |J|: the battery block can only remove patterns)")
    print("\n  trips  patterns")
    for k in sorted(by_trips):
        print(f"  {k:5d}  {by_trips[k]:>12,}")

    print("\n" + "-" * 78)
    print("For contrast, the same count at other operating points:")
    for label, tt, ts in (
        ("T=44, 15-min slots (the Q=3 test point)", 44, 2),
        ("T=44, 30-min slots", 44, 1),
        ("T=22, 30-min slots", 22, 1),
    ):
        tot, _ = count_patterns(tt, ts, start_at_massy=False)
        print(f"  {label:42s} {tot:>22,}")
    print("-" * 78)
    print("Enumeration is open where this number is small enough to hold as columns,")
    print("and closed where it is not. Where it is open, stage 3's only real question --")
    print("does the reformulated LP root beat the compact one -- can be answered with no")
    print("pricing problem and no dominance rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
