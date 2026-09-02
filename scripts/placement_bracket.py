"""What is departure placement actually worth? Bracket it before building anything.

    python scripts/placement_bracket.py [--slot 30] [--Q 2]

THE QUESTION. Today's subproblem prices waiting at minute fidelity but against a
departure instant fixed by convention (`start`/`midpoint`/`end`). Letting it CHOOSE that
instant is a much larger build -- a MIP subproblem, no LP duals, and the vehicle
precedence chain to respect. Before paying for it, measure what it could pay back.

THE INSTRUMENT. Three quantities on one fixed schedule, all in passenger-minutes:

    Q_relaxed   <=   Q_optimal   <=   Q_fixed[start]
    (F2, D74/D76)    (this build)     (today's model, the only honest fixed policy)

  * `Q_fixed[start]` is today's answer, priced under `start` -- the only convention that
    prices what this schedule actually does (D76); `midpoint`/`end` are printed too, but
    only as labelled counterfactuals, not competing baselines.
  * `Q_optimal` chooses one instant per departure, shared by everyone boarding it
    (`price_schedule_optimal_placement`), searching ANTICIPATION only (D76) -- it ignores
    vehicle feasibility of the shift, which can only make it cheaper, so it is a LOWER
    bound on what an implementable optimal placement achieves -- not an achievable cost.
  * `Q_relaxed` is F2's relaxation: every passenger may pick its own instant, over the
    same anticipate-only grid. Physically impossible, strictly cheaper, and the only one
    of the three that may legitimately generate a Benders cut for the optimal-placement
    model, because it is the only one guaranteed to sit at or below the truth.

HOW TO READ THE RESULT. The prize is `best_fixed - Q_optimal`: how much of today's cost
is an artefact of assuming the departure instant rather than choosing it. If it is small
on this instance, the whole direction closes by measurement and nothing needs building.
`best_fixed - Q_relaxed` is the wider, looser envelope -- and its excess over the prize
is the price in cut strength that buying validity on the richer model would cost.

The ordering itself is a standing correctness check on all three implementations
(tests/test_minute_pricer.py::TestOptimalPlacement). If it inverts here, something is
broken, and this script says so rather than printing a number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

BASE_CONFIG = "configs/milp/baseline_d9_p56_monolith.yaml"
DEMAND_FILE = "setups/base.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--slot", type=int, default=None)
    ap.add_argument("--Q", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=300.0)
    args = ap.parse_args()

    from mobauto2_milp.config import load_config
    from mobauto2_milp.app import _prepare_params, _energy_params_for_resolution
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_milp.monolith import MonolithSolver
    from mobauto2_benders.minute_pricer import (
        load_request_minutes,
        price_schedule_at_minutes,
        price_schedule_optimal_placement,
        price_schedule_optimal_placement_with_chain,
        solve_minute_recourse,
    )
    import pyomo.environ as pyo

    cfg = load_config(BASE_CONFIG)
    delta = int(args.slot or cfg.model.time.slot_resolution)
    Q = int(args.Q or cfg.model.fleet.Q)
    seats = float(cfg.service.S)
    wmax = float(cfg.service.Wmax_minutes)
    p_min = float(cfg.service.p_minutes)
    p_slots = p_min / float(delta)

    requests = load_request_minutes(DEMAND_FILE)

    mp, sp = _prepare_params(cfg, {})
    mp, sp = dict(mp), dict(sp)
    mp["slot_resolution"] = delta
    sp["slot_resolution"] = delta
    mp.update(_energy_params_for_resolution(cfg, delta))
    sp.update(_energy_params_for_resolution(cfg, delta))
    sp["p"] = p_slots
    mp["Q"] = Q
    mp["solve_time_limit_s"] = args.time_limit
    mp.pop("T", None)

    print(f"instance={DEMAND_FILE}  delta={delta}  Q={Q}  S={seats:.0f}  "
          f"Wmax={wmax:.0f}min  p_minutes={p_min:.0f}")
    print("solving the first stage (slot monolith) to fix one schedule ...", flush=True)
    solver = MonolithSolver(MobautoMilpModel, cfg, mp, sp)
    result = solver.run()
    m = solver.master.m
    sched: dict[str, list[int]] = {"OUT": [], "RET": []}
    for tau in m.T:
        for q in m.Q:
            if float(pyo.value(m.yOUT[q, tau], exception=False) or 0.0) > 0.5:
                sched["OUT"].append(int(tau))
            if float(pyo.value(m.yRET[q, tau], exception=False) or 0.0) > 0.5:
                sched["RET"].append(int(tau))
    sched["OUT"].sort()
    sched["RET"].sort()

    # The per-vehicle view, which the aggregated schedule above throws away and the
    # chain-constrained evaluator needs: which vehicle flies which trip, in what order,
    # and where it charges.
    vehicle_trips: dict[int, list[tuple[int, str]]] = {}
    charging_slots: dict[int, list[int]] = {}
    for q in m.Q:
        trips: list[tuple[int, str]] = []
        chrs: list[int] = []
        for tau in m.T:
            if float(pyo.value(m.yOUT[q, tau], exception=False) or 0.0) > 0.5:
                trips.append((int(tau), "OUT"))
            if float(pyo.value(m.yRET[q, tau], exception=False) or 0.0) > 0.5:
                trips.append((int(tau), "RET"))
            try:
                if float(pyo.value(m.c[q, tau], exception=False) or 0.0) > 0.5:
                    chrs.append(int(tau))
            except Exception:
                pass
        trips.sort()
        vehicle_trips[int(q)] = trips
        charging_slots[int(q)] = chrs

    print(f"  status={result.status.name}  departures: "
          f"OUT={len(sched['OUT'])} RET={len(sched['RET'])}")
    for q in sorted(vehicle_trips):
        print(f"    vehicle {q}: {len(vehicle_trips[q])} trips, "
              f"{len(charging_slots[q])} charging slots")
    print()

    # D76: only "start" prices what this committed schedule actually does -- see
    # minute_pricer.py's `DeparturePolicy` comment. `midpoint`/`end` are printed
    # alongside it purely as labelled counterfactuals (a slower-departing schedule the
    # master never chose); they are not candidates for "best fixed" any more, so the
    # baseline the prize is measured against is `fixed["start"]`, not min() over all three.
    fixed: dict[str, float] = {}
    for policy in ("start", "midpoint", "end"):
        r = price_schedule_at_minutes(
            sched, requests, delta, seats, wmax, p_min, policy=policy
        )
        fixed[policy] = r.total_cost
        print(f"  Q_fixed[{policy:8s}] = {r.total_cost:10.2f}   "
              f"unserved={r.unserved_passengers:5.0f}"
              + ("   <- honest baseline (D76)" if policy == "start" else "   (counterfactual)"))
    best_policy = "start"
    best_fixed = fixed[best_policy]
    print()

    print("  computing Q_optimal free  (shifts unconstrained) ...", flush=True)
    opt = price_schedule_optimal_placement(
        sched, requests, delta, seats, wmax, p_min
    )
    print(f"  Q_optimal[free ]     = {opt.total_cost:10.2f}   "
          f"unserved={opt.unserved_passengers:5.0f}")

    print("  computing Q_optimal chain (precedence + no charge encroachment) ...",
          flush=True)
    trip_dur = float(cfg.model.time.trip_duration_minutes)
    chain, chosen = price_schedule_optimal_placement_with_chain(
        vehicle_trips, charging_slots, requests, delta, seats, wmax, p_min, trip_dur
    )
    shifted = sum(1 for t, d in chosen.items() if abs(d - t[0] * delta) > 1e-9)
    print(f"  Q_optimal[chain]     = {chain.total_cost:10.2f}   "
          f"unserved={chain.unserved_passengers:5.0f}   "
          f"({shifted}/{len(chosen)} departures actually shifted)")

    T = int(cfg.model.time.T_minutes) // delta
    C_out = [0.0] * T
    C_ret = [0.0] * T
    for t in sched["OUT"]:
        C_out[t] += seats
    for t in sched["RET"]:
        C_ret[t] += seats
    # D76: anticipate-only grid -- tau*delta is the ceiling (the master's own committed
    # instant), not the floor. A positive-offset grid here would price a schedule that
    # departs later than the master ever committed to; see minute_pricer.py::_offset_grid.
    grid = [float(k) for k in range(-delta, 1)]
    print("  computing Q_relaxed (F2, every passenger free, anticipate-only) ...", flush=True)
    _duals, obj_slot_units = solve_minute_recourse(
        T, delta, wmax, p_slots, C_out, C_ret, requests,
        policy="start", placement_offsets=grid,
    )
    relaxed = obj_slot_units * delta
    print(f"  Q_relaxed            = {relaxed:10.2f}")
    print()

    tol = 1e-6
    order = [
        ("Q_relaxed", relaxed),
        ("Q_optimal[free]", opt.total_cost),
        ("Q_optimal[chain]", chain.total_cost),
        (f"Q_fixed[{best_policy}]", best_fixed),
    ]
    print("=" * 78)
    for (na, va), (nb, vb) in zip(order, order[1:]):
        if va > vb + tol:
            print("SANDWICH VIOLATED -- the ordering this module guarantees does not hold.")
            print(f"  {na} = {va:.2f}  >  {nb} = {vb:.2f}")
            print("One of the implementations is wrong. Do not read a prize off this.")
            print("=" * 78)
            return 1

    achievable = best_fixed - chain.total_cost
    upper = best_fixed - opt.total_cost
    envelope = best_fixed - relaxed
    print(f"best fixed policy       : {best_policy} at {best_fixed:.2f}")
    print(f"ACHIEVABLE PRIZE        : {achievable:.2f} passenger-minutes "
          f"({100.0 * achievable / best_fixed:.2f}%)  <- the real one")
    print(f"  upper bound on prize  : {upper:.2f} "
          f"({100.0 * upper / best_fixed:.2f}%)  (free shifting, not implementable)")
    print(f"  eaten by the fleet    : {upper - achievable:.2f} "
          f"-- precedence and charging")
    print(f"relaxation envelope     : {envelope:.2f} "
          f"({100.0 * envelope / best_fixed:.2f}%)")
    print(f"cut strength given up   : {envelope - achievable:.2f} "
          f"-- what a cut generator valid for the")
    print(f"                           optimal-placement model concedes to buy validity.")
    print(f"VERDICT                 : concession / achievable prize = "
          f"{(envelope - achievable) / achievable:.2f}x"
          if achievable > 1e-9 else
          "VERDICT                 : achievable prize is zero -- the chain eats all of it.")
    print("=" * 78)
    print()
    print("Q_optimal[chain] is attainable: departures still choose their instant, but a")
    print("vehicle's trips follow one another and no trip eats into a charging slot. It")
    print("is conservative on charging (encroachment refused outright rather than")
    print("modelled proportionally), so the achievable prize could be slightly larger")
    print("under a finer energy model -- never smaller.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
