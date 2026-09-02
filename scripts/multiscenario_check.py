"""Does averaging over scenarios wash the multi-resolution effect out?

    python scripts/multiscenario_check.py [--slot 30] [--Q 2] [--p-minutes 56]

THE CONCERN (research note v2, section 9, falsifier 3). With several demand scenarios the
recourse is a WEIGHTED AVERAGE. Each scenario may have sharp sub-slot structure, but if
that structure sits at different minutes in different scenarios, the average can be smooth
even when every member is spiky. The schedule that is best on average would then be the
same under slot and minute valuation, and the effect measured on single scenarios would be
an artifact of looking at one at a time.

WHAT THIS RUNS. The same first stage against the same four scenarios, twice:

  A  slot recourse,   averaged over scenarios  -- what the project does today
  B  minute recourse, averaged over scenarios  -- one shared first stage, per-scenario
                                                  minute evaluation, weights applied

Both are solved monolithically to proven optimality, and both resulting schedules are then
priced at minute fidelity in every scenario and averaged. That last step is the fair
comparison: schedule A optimised a valuation known to be wrong, so it must be judged by
what it really costs.

TWO THINGS ARE REPORTED SEPARATELY, because they can disagree:

  per-scenario   the gain if each scenario were the only one (the D54 measurement)
  averaged       the gain when one schedule must serve all four at once

If `averaged` collapses toward zero while `per-scenario` stays large, falsifier 3 holds
and the effect is an artifact of single-scenario testing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

BASE_CONFIG = "configs/milp/baseline_d9_p56_monolith.yaml"
# Four shapes with structure at DIFFERENT minutes -- the situation most likely to average
# away. Using four copies of one shape would stack the deck the other way.
SCENARIOS = ("commuter", "bimodal", "burst", "spiky")


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
    ap.add_argument("--policy", choices=("start", "midpoint", "end"), default="start")
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
    files = [f"setups/generated/{s}.yaml" for s in SCENARIOS]
    reqs = [load_request_minutes(f) for f in files]
    weight = 1.0 / float(len(reqs))

    mp, sp = _prepare_params(cfg, {})
    mp, sp = dict(mp), dict(sp)
    sp["p"] = p_min / float(delta)
    sp["scenario_files"] = list(files)
    sp.pop("demand_file", None)
    mp["slot_resolution"] = delta
    sp["slot_resolution"] = delta
    if args.Q:
        mp["Q"] = args.Q
    mp.pop("T", None)

    solver_a = MonolithSolver(MobautoMilpModel, cfg, mp, sp)
    solver_a.run()
    sched_a = _schedule(solver_a.master.m)

    master_b = MobautoMilpModel(dict(mp))
    master_b.initialize()
    attach_minute_recourse(
        master_b.m,
        None,
        delta,
        seats,
        wmax,
        p_min,
        policy=args.policy,
        objective_scale=1.0 / float(delta),
        scenarios=[(r, weight) for r in reqs],
    )
    master_b.solve()
    sched_b = _schedule(master_b.m)

    print()
    print(f"slot={delta}min  Q={args.Q or cfg.model.fleet.Q}  p_minutes={p_min:.0f}  "
          f"policy={args.policy}  scenarios={len(reqs)}")
    print(f"schedules identical: {sched_a == sched_b}")
    print()
    print(f"{'scenario':12s} | {'A cost':>9s} {'B cost':>9s} {'gain':>7s} | "
          f"{'A unserv':>8s} {'B unserv':>8s}")
    print("-" * 64)
    tot_a = tot_b = 0.0
    for name, r in zip(SCENARIOS, reqs):
        ra = price_schedule_at_minutes(
            sched_a, r, delta, seats, wmax, p_min, policy=args.policy
        )
        rb = price_schedule_at_minutes(
            sched_b, r, delta, seats, wmax, p_min, policy=args.policy
        )
        g = 100.0 * (ra.total_cost - rb.total_cost) / ra.total_cost if ra.total_cost else 0.0
        tot_a += weight * ra.total_cost
        tot_b += weight * rb.total_cost
        print(f"{name:12s} | {ra.total_cost:9.0f} {rb.total_cost:9.0f} {g:6.2f}% | "
              f"{ra.unserved_passengers:8.0f} {rb.unserved_passengers:8.0f}")
    print("-" * 64)
    gain = 100.0 * (tot_a - tot_b) / tot_a if tot_a else 0.0
    print(f"{'AVERAGED':12s} | {tot_a:9.0f} {tot_b:9.0f} {gain:6.2f}%")
    print()
    print("Read the AVERAGED row against the single-scenario gains in D54. If it collapses")
    print("toward zero, scenario averaging washes the effect out and the single-scenario")
    print("result is an artifact of testing one at a time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
