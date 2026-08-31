"""Delta=1 sanity pilot: does slot-recourse collapse into minute-recourse at 1-minute slots?

    python scripts/delta1_monolith_pilot.py [--Q 2] [--time-limit 300] [--policy end]

THE POINT (forward-plan A4d follow-up, requested inline after the delta in {30,15,10}
factorial). At slot_resolution=1, a slot IS a minute: the slot-only recourse and the
minute-level recourse are the same model, so the two solves this script runs (mirroring
scripts/sweep_multiresolution.py's inner loop) should produce IDENTICAL schedules and
costs -- `same?` should read `yes` and `gain` should read ~0.00%. That is the sanity
check. `--policy end` is used (not the default 3-way sweep): at delta=1 the three
placement conventions (start/midpoint/end) coincide, so sweeping them is redundant here.

THE OTHER POINT -- tractability. Every other resolution in this project's monolith
results uses slot_resolution in {10, 15, 30}: T = T_minutes/delta in {66, 44, 22}. Here
T = 660, a 10-30x larger monolith. If this does not converge inside --time-limit seconds
per solve, or does not even finish building inside a reasonable wall clock, THAT is a
finding, not a failure to hide: it is the point past which this MILP formulation stops
being viable and a decomposition (Benders) is the only way forward. Report the outcome
either way. A companion script (once this pilot's outcome is known) runs the equivalent
check through the Benders solver on the same instance for the direct comparison.

INSTANCE. Uses the config's own demand (`setups/base.yaml`, the project's real baseline,
~300 requests) rather than the synthetic shapes in setups/generated/ -- those were built
to test sensitivity to demand SHAPE (report Table grids / D51-52), which is not the
question here.

Prints progress with explicit before/after timestamps and flush=True at each build/solve
boundary, specifically so that a hang during model construction (not just a slow solve)
is visible in the output rather than silent.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

BASE_CONFIG = "configs/milp/baseline_d9_p56_monolith.yaml"
DEMAND_FILE = "setups/base.yaml"


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Q", type=int, default=2)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--policy", default="end")
    ap.add_argument("--p-minutes", type=float, default=None)
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

    delta = 1
    Q = args.Q
    print(f"[{_now()}] delta1 pilot starting: Q={Q}, policy={args.policy}, "
          f"time_limit={args.time_limit}s per solve, instance={DEMAND_FILE}", flush=True)

    cfg = load_config(BASE_CONFIG)
    p_min = float(args.p_minutes if args.p_minutes is not None else cfg.service.p_minutes)
    seats = float(cfg.service.S)
    wmax = float(cfg.service.Wmax_minutes)

    print(f"[{_now()}] loading request minutes from {DEMAND_FILE} ...", flush=True)
    t0 = time.monotonic()
    requests = load_request_minutes(DEMAND_FILE)
    print(f"[{_now()}] loaded {sum(len(v) for v in requests.values())} request-minutes "
          f"in {time.monotonic()-t0:.1f}s", flush=True)

    mp, sp = _prepare_params(cfg, {})
    mp, sp = dict(mp), dict(sp)
    sp["p"] = p_min / float(delta)
    mp["slot_resolution"] = delta
    sp["slot_resolution"] = delta
    mp["Q"] = Q
    mp["solve_time_limit_s"] = args.time_limit
    mp.pop("T", None)

    print(f"[{_now()}] === solver A: slot-only monolith, T={cfg.model.time.T_minutes // delta} slots ===",
          flush=True)
    t0 = time.monotonic()
    print(f"[{_now()}] building + solving MonolithSolver ...", flush=True)
    solver_a = MonolithSolver(MobautoMilpModel, cfg, mp, sp)
    result_a = solver_a.run()
    dt_a = time.monotonic() - t0
    print(f"[{_now()}] solver A done in {dt_a:.1f}s, status={result_a.status.name}, "
          f"obj={result_a.best_upper_bound}, bound={result_a.best_lower_bound}", flush=True)
    sched_a = _schedule(solver_a.master.m)

    print(f"[{_now()}] === solver B: minute-recourse master ===", flush=True)
    t0 = time.monotonic()
    print(f"[{_now()}] building master ...", flush=True)
    master_b = MobautoMilpModel(dict(mp))
    master_b.initialize()
    print(f"[{_now()}] attaching minute recourse ...", flush=True)
    attach_minute_recourse(
        master_b.m, requests, delta, seats, wmax, p_min,
        policy=args.policy, objective_scale=1.0 / float(delta),
    )
    print(f"[{_now()}] solving ...", flush=True)
    master_b.solve()
    dt_b = time.monotonic() - t0
    print(f"[{_now()}] solver B done in {dt_b:.1f}s", flush=True)
    sched_b = _schedule(master_b.m)

    ra = price_schedule_at_minutes(sched_a, requests, delta, seats, wmax, p_min, policy=args.policy)
    rb = price_schedule_at_minutes(sched_b, requests, delta, seats, wmax, p_min, policy=args.policy)
    gain = 100.0 * (ra.total_cost - rb.total_cost) / ra.total_cost if ra.total_cost else 0.0

    print()
    print("=" * 90)
    print(f"delta=1  Q={Q}  policy={args.policy}")
    print(f"  A (slot-only, priced @ minute): cost={ra.total_cost:.2f}  unserved={ra.unserved_passengers:.0f}")
    print(f"  B (minute-recourse):            cost={rb.total_cost:.2f}  unserved={rb.unserved_passengers:.0f}")
    print(f"  gain = {gain:.4f}%   same schedule? {'yes' if sched_a == sched_b else 'no'}")
    print(f"  solver A: {dt_a:.1f}s, status={result_a.status.name}")
    print(f"  solver B: {dt_b:.1f}s")
    print("=" * 90)
    print()
    if abs(gain) < 0.01 and sched_a == sched_b:
        print("SANITY CHECK PASSES: at delta=1, slot-recourse and minute-recourse coincide.")
    else:
        print("SANITY CHECK DOES NOT PASS AS EXPECTED -- gain is non-trivial or schedules differ")
        print("at delta=1. Worth a closer look before trusting this cell (could be a clock-")
        print("truncation artifact if either solver hit the time limit -- check status above).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
