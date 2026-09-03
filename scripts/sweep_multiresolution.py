"""Does minute-level valuation pay, and where? Sweep over demand shape and placement.

    python scripts/sweep_multiresolution.py [--slot 30] [--Q 2] [--p-minutes 56]
    python scripts/sweep_multiresolution.py --slot 30,15,10 --Q 2,3,4,5   # A4d factorial

For each instance it solves the same first stage twice, monolithically and to proven
optimality -- once with a SLOT recourse, once with a MINUTE recourse -- and prices both
resulting schedules at minute fidelity, which is the only comparison that is fair to
either. See D51/D52.

BOTH placement conventions are swept, because the whole result turned on it: at
`p_minutes = 1500` the minute-optimised schedule is identical to the slot-optimised one
under `start` and 7.2% better under `midpoint`. A single-convention sweep would have
reported the second and missed the first.

THE PREDICTION UNDER TEST (scripts/make_instances.py): the gain should track how much
demand structure lives below one slot -- least on `flat`, most on `burst` and `spiky`.

`--slot` and `--Q` each accept a comma-separated list (a single value works exactly as
before). Together they are the A4d factorial (docs/FORWARD_PLAN_v1.md): delta and Q have
each been swept separately elsewhere in this project -- delta at fixed Q=2 (report Table
grids), Q at fixed delta (Claim 2's fleet-size work) -- but never crossed. This is that
cross, restricted to the two effects that are monolith-only and do not need the
continuous-time CP model: grid refinement (varying delta) and minute-level valuation
(comparing the two solves at each delta). It does not cover continuous departure placement
or the multi-scenario effect -- those need the CP model and D70's stochastic-robustness
construction respectively, not this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

BASE_CONFIG = "configs/milp/baseline_d9_p56_monolith.yaml"
SHAPES = ("flat", "commuter", "bimodal", "burst", "spiky")


def _schedule(model) -> dict[str, list[int]]:
    import pyomo.environ as pyo

    out: dict[str, list[int]] = {"OUT": [], "RET": []}
    for tau in model.T:
        for q in model.Q:
            if float(pyo.value(model.yOUT[q, tau], exception=False) or 0.0) > 0.5:
                out["OUT"].append(int(tau))
            if float(pyo.value(model.yRET[q, tau], exception=False) or 0.0) > 0.5:
                out["RET"].append(int(tau))
    out["OUT"].sort()
    out["RET"].sort()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slot", default=None, help="One value or a comma-separated list.")
    ap.add_argument("--Q", default=None, help="One value or a comma-separated list.")
    ap.add_argument("--p-minutes", type=float, default=None)
    ap.add_argument(
        "--policies", default="start",
        help="Comma-separated placement conventions to sweep. `start` (o=0) is the "
             "committed departure instant and, since D76, the only offset that prices "
             "the schedule the master actually commits to; `midpoint`/`end` remain "
             "computable and are labelled counterfactuals ('what if real dwell and "
             "boarding push this departure o minutes past its committed instant'), "
             "never the schedule's reported cost.",
    )
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument(
        "--time-limit", type=float, default=None,
        help="Per-solve time cap in seconds (both the slot-only monolith and the "
             "minute-recourse master). Defaults to whatever BASE_CONFIG sets "
             "(configs/milp/baseline_d9_p56_monolith.yaml: solve_time_limit_s=900). "
             "Override for finer deltas / larger Q where 900s per solve is too slow "
             "to finish the grid in a reasonable wall-clock budget -- a clock-truncated "
             "cell is still reported (BENDERS_SPEC_v4 section 0.10 caveat applies).",
    )
    args = ap.parse_args()

    from mobauto2_milp.config import load_config
    from mobauto2_milp.app import _prepare_params, _energy_params_for_resolution
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_milp.monolith import MonolithSolver
    from mobauto2_benders.minute_pricer import (
        attach_minute_recourse,
        load_request_minutes,
        price_schedule_at_minutes,
    )

    cfg = load_config(BASE_CONFIG)
    slots = (
        [int(x) for x in args.slot.split(",") if x.strip()]
        if args.slot
        else [int(cfg.model.time.slot_resolution)]
    )
    q_values = (
        [int(x) for x in args.Q.split(",") if x.strip()]
        if args.Q
        else [int(cfg.model.fleet.Q)]
    )
    seats = float(cfg.service.S)
    wmax = float(cfg.service.Wmax_minutes)

    print()
    print(f"slots={slots}  Q={q_values}  S={seats:.0f}  Wmax={wmax:.0f}min")
    print(f"{'delta':>5s} {'Q':>2s} {'shape':10s} {'policy':9s} | {'A slot-opt':>11s} "
          f"{'B minute-opt':>12s} {'gain':>7s} | {'A unserv':>8s} {'B unserv':>8s} "
          f"{'served+':>8s} | same?")
    print("-" * 108)

    for delta in slots:
        p_min = float(
            args.p_minutes
            if args.p_minutes is not None
            else (cfg.service.p_minutes or cfg.service.p * delta)
        )
        for Q in q_values:
            for shape in [x.strip() for x in args.shapes.split(",") if x.strip()]:
                demand = f"setups/generated/{shape}.yaml"
                requests = load_request_minutes(demand)
                for policy in [x.strip() for x in args.policies.split(",") if x.strip()]:
                    mp, sp = _prepare_params(cfg, {})
                    mp, sp = dict(mp), dict(sp)
                    sp["p"] = p_min / float(delta)
                    sp["scenario_files"] = [demand]
                    sp["demand_file"] = demand
                    mp["slot_resolution"] = delta
                    sp["slot_resolution"] = delta
                    # BUG FIX (found while investigating the delta=1 Benders-vs-monolith
                    # comparison): delta_chg (energy gained per charging slot) is baked
                    # into mp/sp by _prepare_params() at the CONFIG's own slot_resolution
                    # (30). Overriding slot_resolution above without recomputing this left
                    # every delta != 30 cell charging at the 30-minute rate regardless of
                    # its actual slot width -- 2x too fast at delta=15, 3x at delta=10.
                    # This is the same recompute app.py's own multi_res path performs.
                    mp.update(_energy_params_for_resolution(cfg, delta))
                    sp.update(_energy_params_for_resolution(cfg, delta))
                    mp["Q"] = Q
                    mp.pop("T", None)
                    if args.time_limit is not None:
                        mp["solve_time_limit_s"] = args.time_limit

                    solver_a = MonolithSolver(MobautoMilpModel, cfg, mp, sp)
                    solver_a.run()
                    sched_a = _schedule(solver_a.master.m)

                    master_b = MobautoMilpModel(dict(mp))
                    master_b.initialize()
                    attach_minute_recourse(
                        master_b.m, requests, delta, seats, wmax, p_min,
                        policy=policy, objective_scale=1.0 / float(delta),
                    )
                    master_b.solve()
                    sched_b = _schedule(master_b.m)

                    ra = price_schedule_at_minutes(
                        sched_a, requests, delta, seats, wmax, p_min, policy=policy
                    )
                    rb = price_schedule_at_minutes(
                        sched_b, requests, delta, seats, wmax, p_min, policy=policy
                    )
                    gain = (
                        100.0 * (ra.total_cost - rb.total_cost) / ra.total_cost
                        if ra.total_cost
                        else 0.0
                    )
                    print(
                        f"{delta:5d} {Q:2d} {shape:10s} {policy:9s} | {ra.total_cost:11.0f} "
                        f"{rb.total_cost:12.0f} {gain:6.2f}% | {ra.unserved_passengers:8.0f} "
                        f"{rb.unserved_passengers:8.0f} "
                        f"{ra.unserved_passengers - rb.unserved_passengers:8.0f} | "
                        f"{'yes' if sched_a == sched_b else 'no'}"
                    )
    print("-" * 108)
    print("gain = how much cheaper the minute-optimised schedule really is, both priced")
    print("at minute fidelity. served+ = extra passengers carried by the minute-optimised")
    print("schedule. 'same?' = the two first stages chose identical departure slots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
