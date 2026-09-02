"""Validate at the trial's declared service level: 450 passengers/day, up to 30 trips/day.

    python scripts/sweep_target_scale.py [--Q 2,3,4] [--time-limit 300]

THE QUESTION (forward-plan A3, docs/FORWARD_PLAN_v1.md). Every instance reported elsewhere
in this project runs 300-400 requests. The project's own declared trial service level
(report Section intro-corridor, deliverable 1.4.3) is up to 450 passengers/day over up to
30 trips/day, and it has never been exercised. This sweeps fleet size on
setups/base_scale450.yaml (scripts/scale_baseline_demand.py) and reports, per Q: whether the
solve reaches a certified optimum inside the given time budget, passengers served/unserved,
and total departures against the declared 30-trip ceiling.

WHAT "DONE" LOOKS LIKE EITHER WAY (A3's own bar). A run exists at this scale with a recorded
outcome. If a Q value does not reach a certified bound inside the budget, that connects to
Claim 2 (the decomposition is not competitive on this family) rather than being a failed
experiment to hide -- report the best bound and the gap, not just a blank cell.

BUDGET. Each Q is capped at --time-limit seconds (default 300), stopping on the gap or the
clock, whichever comes first. A clock-truncated cell is reported as such
(status != OPTIMAL) and is not quotable as a certified figure -- only as evidence that the
budget was insufficient at that Q, which is itself part of the answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

BASE_CONFIG = "configs/milp/target_scale450_monolith.yaml"


def _schedule(model) -> dict[str, list[int]]:
    import pyomo.environ as pyo

    out: dict[str, list[int]] = {"OUT": [], "RET": []}
    for tau in model.T:
        for q in model.Q:
            if float(pyo.value(model.yOUT[q, tau], exception=False) or 0.0) > 0.5:
                out["OUT"].append(int(tau))
            if float(pyo.value(model.yRET[q, tau], exception=False) or 0.0) > 0.5:
                out["RET"].append(int(tau))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Q", default="2,3,4")
    ap.add_argument("--time-limit", type=float, default=300.0)
    args = ap.parse_args()

    from mobauto2_milp.app import _prepare_params
    from mobauto2_milp.config import load_config
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_milp.monolith import MonolithSolver

    q_values = [int(x) for x in args.Q.split(",")]
    cfg = load_config(BASE_CONFIG)

    print(f"instance: {cfg.data.scenario_files}, p_minutes={cfg.service.p * cfg.model.time.slot_resolution}, "
          f"time_limit_s={args.time_limit} per Q")
    print(f"{'Q':>3s} | {'status':>10s} | {'served':>7s} | {'unserved':>8s} | "
          f"{'trips':>5s} | {'<=30 trips':>10s} | {'obj':>10s} | {'bound':>10s} | {'gap':>7s}")
    print("-" * 90)

    rows = []
    for Q in q_values:
        mp, sp = _prepare_params(cfg, {})
        mp, sp = dict(mp), dict(sp)
        mp["Q"] = Q
        mp["solve_time_limit_s"] = args.time_limit
        mp.pop("T", None)
        solver = MonolithSolver(MobautoMilpModel, cfg, mp, sp)
        result = solver.run()
        schedule = _schedule(solver.master.m)
        trips = len(schedule["OUT"]) + len(schedule["RET"])
        served = result.pax_served if result.pax_served is not None else float("nan")
        total = result.pax_total if result.pax_total is not None else float("nan")
        unserved = total - served if total == total and served == served else float("nan")
        obj = result.best_upper_bound
        bound = result.best_lower_bound
        gap = (
            100.0 * abs(obj - bound) / max(1.0, abs(obj))
            if obj is not None and bound is not None
            else float("nan")
        )
        within_30 = "yes" if trips <= 30 else "NO"
        rows.append(
            dict(Q=Q, status=result.status.name, served=served, unserved=unserved,
                 trips=trips, within_30=within_30, obj=obj, bound=bound, gap=gap)
        )
        print(f"{Q:3d} | {result.status.name:>10s} | {served:7.0f} | {unserved:8.0f} | "
              f"{trips:5d} | {within_30:>10s} | {obj:10.2f} | {bound:10.2f} | {gap:6.2f}%")

    print()
    print("A cell whose status is not OPTIMAL is a clock-truncated bound (BENDERS_SPEC_v4")
    print("section 0.10): reproducible only in the sense that a non-binding budget would")
    print("give the true optimum, not that this exact number will regenerate on another")
    print("machine or thread count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
