"""What is departure placement actually worth? Bracket it before building anything.

    python scripts/placement_bracket.py [--slot 30] [--Q 2]

THE QUESTION. Today's subproblem prices waiting at minute fidelity but against a
departure instant fixed by convention (`start`/`midpoint`/`end`). Letting it CHOOSE that
instant is a much larger build -- a MIP subproblem, no LP duals, and the vehicle
precedence chain to respect. Before paying for it, measure what it could pay back.

THE INSTRUMENT. Three quantities on one fixed schedule, all in passenger-minutes:

    Q_relaxed   <=   Q_optimal   <=   min over policies of Q_fixed
    (F2, D74)        (this build)     (today's model)

  * `Q_fixed` is today's answer -- `price_schedule_at_minutes` under each convention.
  * `Q_optimal` chooses one instant per departure, shared by everyone boarding it
    (`price_schedule_optimal_placement`). It ignores vehicle feasibility of the shift,
    which can only make it cheaper, so it is a LOWER bound on what an implementable
    optimal placement achieves -- not an achievable cost.
  * `Q_relaxed` is F2's relaxation: every passenger may pick its own instant. Physically
    impossible, strictly cheaper, and the only one of the three that may legitimately
    generate a Benders cut for the optimal-placement model, because it is the only one
    guaranteed to sit at or below the truth.

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
    print(f"  status={result.status.name}  departures: "
          f"OUT={len(sched['OUT'])} RET={len(sched['RET'])}")
    print()

    fixed: dict[str, float] = {}
    for policy in ("start", "midpoint", "end"):
        r = price_schedule_at_minutes(
            sched, requests, delta, seats, wmax, p_min, policy=policy
        )
        fixed[policy] = r.total_cost
        print(f"  Q_fixed[{policy:8s}] = {r.total_cost:10.2f}   "
              f"unserved={r.unserved_passengers:5.0f}")
    best_policy = min(fixed, key=fixed.get)
    best_fixed = fixed[best_policy]
    print()

    print("  computing Q_optimal (MIP, one instant per departure) ...", flush=True)
    opt = price_schedule_optimal_placement(
        sched, requests, delta, seats, wmax, p_min
    )
    print(f"  Q_optimal            = {opt.total_cost:10.2f}   "
          f"unserved={opt.unserved_passengers:5.0f}")

    T = int(cfg.model.time.T_minutes) // delta
    C_out = [0.0] * T
    C_ret = [0.0] * T
    for t in sched["OUT"]:
        C_out[t] += seats
    for t in sched["RET"]:
        C_ret[t] += seats
    grid = [float(k) for k in range(delta + 1)]
    print("  computing Q_relaxed (F2, every passenger free) ...", flush=True)
    _duals, obj_slot_units = solve_minute_recourse(
        T, delta, wmax, p_slots, C_out, C_ret, requests,
        policy="midpoint", placement_offsets=grid,
    )
    relaxed = obj_slot_units * delta
    print(f"  Q_relaxed            = {relaxed:10.2f}")
    print()

    ok = relaxed <= opt.total_cost + 1e-6 <= best_fixed + 1e-6
    print("=" * 78)
    if not ok:
        print("SANDWICH VIOLATED -- Q_relaxed <= Q_optimal <= best Q_fixed does not hold.")
        print(f"  {relaxed:.2f} <= {opt.total_cost:.2f} <= {best_fixed:.2f}")
        print("One of the three implementations is wrong. Do not read a prize off this.")
        print("=" * 78)
        return 1

    prize = best_fixed - opt.total_cost
    envelope = best_fixed - relaxed
    print(f"best fixed policy      : {best_policy} at {best_fixed:.2f}")
    print(f"THE PRIZE              : {prize:.2f} passenger-minutes "
          f"({100.0 * prize / best_fixed:.2f}% of today's cost)")
    print(f"relaxation envelope    : {envelope:.2f} "
          f"({100.0 * envelope / best_fixed:.2f}%)")
    print(f"cut strength given up  : {envelope - prize:.2f} "
          f"-- what a cut generator valid for the optimal-placement")
    print(f"                          model concedes versus today's fixed-policy cut.")
    print("=" * 78)
    print()
    print("The prize is a LOWER bound's distance from today's cost: Q_optimal ignores")
    print("whether a shifted departure leaves its vehicle able to make its next trip,")
    print("so an implementable optimal placement costs at least this much and possibly")
    print("more. Read the prize as an upper bound ON THE PRIZE, not a promised saving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
