"""Does minute-level valuation pay, and where? Sweep over demand shape and placement.

    python scripts/sweep_multiresolution.py [--slot 30] [--Q 2] [--p-minutes 56]

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
    ap.add_argument("--slot", type=int, default=None)
    ap.add_argument("--Q", type=int, default=None)
    ap.add_argument("--p-minutes", type=float, default=None)
    ap.add_argument(
        "--policies", default="start,midpoint,end",
        help="Comma-separated placement conventions to sweep. `start` assumes the bus "
             "leaves before the passengers it is collecting have arrived and is kept "
             "only as a lower envelope (D54); drop it to halve the runtime.",
    )
    ap.add_argument("--shapes", default=",".join(SHAPES))
    args = ap.parse_args()

    from mobauto2_milp.config import load_config
    from mobauto2_milp.app import _prepare_params
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_milp.monolith import MonolithSolver
    from mobauto2_benders.minute_pricer import (
        attach_minute_recourse,
        load_request_minutes,
        price_schedule_at_minutes,
    )

    cfg = load_config(BASE_CONFIG)
    delta = int(args.slot or cfg.model.time.slot_resolution)
    seats = float(cfg.service.S)
    wmax = float(cfg.service.Wmax_minutes)
    p_min = float(
        args.p_minutes
        if args.p_minutes is not None
        else (cfg.service.p_minutes or cfg.service.p * delta)
    )

    print()
    print(f"slot={delta}min  Q={args.Q or cfg.model.fleet.Q}  S={seats:.0f}  "
          f"Wmax={wmax:.0f}min  p_minutes={p_min:.0f}")
    print(f"{'shape':10s} {'policy':9s} | {'A slot-opt':>11s} {'B minute-opt':>12s} "
          f"{'gain':>7s} | {'A unserv':>8s} {'B unserv':>8s} {'served+':>8s} | same?")
    print("-" * 96)

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
            if args.Q:
                mp["Q"] = args.Q
            mp.pop("T", None)

            solver_a = MonolithSolver(MobautoMilpModel, cfg, mp, sp)
            res_a = solver_a.run()
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
            gain = 100.0 * (ra.total_cost - rb.total_cost) / ra.total_cost if ra.total_cost else 0.0
            print(
                f"{shape:10s} {policy:9s} | {ra.total_cost:11.0f} {rb.total_cost:12.0f} "
                f"{gain:6.2f}% | {ra.unserved_passengers:8.0f} {rb.unserved_passengers:8.0f} "
                f"{ra.unserved_passengers - rb.unserved_passengers:8.0f} | "
                f"{'yes' if sched_a == sched_b else 'no'}"
            )
    print("-" * 96)
    print("gain = how much cheaper the minute-optimised schedule really is, both priced")
    print("at minute fidelity. served+ = extra passengers carried by the minute-optimised")
    print("schedule. 'same?' = the two first stages chose identical departure slots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
