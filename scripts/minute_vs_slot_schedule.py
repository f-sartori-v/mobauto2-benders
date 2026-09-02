"""Step 3: does minute-level valuation change the SCHEDULE, or only its reported cost?

    python scripts/minute_vs_slot_schedule.py

Solves the same instance twice, monolithically and to proven optimality:

  A. slot master + SLOT recourse    -- what the project does today
  B. slot master + MINUTE recourse  -- the architecture the research idea proposes

B is the multi-resolution model: the first stage stays on slots, the operational
evaluation moves to minutes, and `y` enters the recourse only through the capacity
right-hand side (E2). Both are solved as one MIP, so neither answer is a bound or a
truncation -- any difference between them is the effect of the valuation, not of a
decomposition.

The recourse in B is scaled by 1/slot_resolution so both objectives are in the same
units and the first-stage terms keep the same relative weight in each.

Then both schedules are priced at minute fidelity, which is the only fair comparison:
schedule A optimised against a valuation we now know is wrong by 66-86% on the waiting
term, so it must be judged by what it really costs, not by what its own model claimed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

CONFIG = "configs/milp/baseline_d9_monolith.yaml"


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
    ap.add_argument(
        "--policy", choices=("start", "midpoint", "end"), default="start",
        help="Where inside its slot a departure is assumed to leave. A modelling "
             "assumption, not a fact -- both are worth reporting.",
    )
    ap.add_argument(
        "--p-minutes", type=float, default=None,
        help="Override the unmet-demand penalty, in passenger-minutes. The config's "
             "p is in SLOT units; p_minutes = p * slot_resolution.",
    )
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

    cfg = load_config(CONFIG)
    mp, sp = _prepare_params(cfg, {})
    delta = int(cfg.model.time.slot_resolution)
    seats = float(cfg.service.S)
    wmax = float(cfg.service.Wmax_minutes)
    p_min = (
        float(args.p_minutes)
        if args.p_minutes is not None
        else float(cfg.service.p) * delta
    )
    # p enters the master through service.p in SLOT units, so an override has to be
    # pushed back through the same door or the two arms would price differently.
    mp = dict(mp)
    sp = dict(sp)
    sp["p"] = p_min / float(delta)
    demand = list(cfg.data.scenario_files)[0]
    requests = load_request_minutes(demand)

    # --- A: slot recourse (the current model) ---
    solver_a = MonolithSolver(MobautoMilpModel, cfg, mp, sp)
    res_a = solver_a.run()
    sched_a = _schedule(solver_a.master.m)

    # --- B: minute recourse, same first stage ---
    master_b = MobautoMilpModel(dict(mp))
    master_b.initialize()
    attach_minute_recourse(
        master_b.m, requests, delta, seats, wmax, p_min,
        policy=args.policy, objective_scale=1.0 / float(delta),
    )
    res_b = master_b.solve()
    sched_b = _schedule(master_b.m)

    print("\n" + "=" * 72)
    print(f"instance {CONFIG}   policy={args.policy}   slot={delta}min  S={seats:.0f}  Wmax={wmax:.0f}min  p_min={p_min:.0f}")
    print("=" * 72)
    print(f"A  slot recourse    objective {float(res_a.best_upper_bound):9.2f}  ({res_a.status.name})")
    print(f"B  minute recourse  objective {float(res_b.objective):9.2f}  ({res_b.status.name})")
    print("-" * 72)
    same = sched_a == sched_b
    print(f"schedules identical: {same}")
    for d in ("OUT", "RET"):
        print(f"  {d}  A: {sched_a[d]}")
        print(f"      B: {sched_b[d]}")
        if sched_a[d] != sched_b[d]:
            print(f"      A-only {sorted(set(sched_a[d]) - set(sched_b[d]))}   "
                  f"B-only {sorted(set(sched_b[d]) - set(sched_a[d]))}")
    print("-" * 72)
    print("Both schedules priced at minute fidelity (the fair comparison):")
    for label, sched in (("A (slot-optimised)", sched_a), ("B (minute-optimised)", sched_b)):
        r = price_schedule_at_minutes(sched, requests, delta, seats, wmax, p_min, policy=args.policy)
        avg = r.waiting_minutes / r.served_passengers if r.served_passengers else 0.0
        print(f"  {label:22s} cost {r.total_cost:10.1f} pax-min   "
              f"wait {r.waiting_minutes:7.1f} (avg {avg:5.2f})   unserved {r.unserved_passengers:.0f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
