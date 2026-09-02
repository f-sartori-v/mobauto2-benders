"""B1 Route 2 (HANDOUT_V02_CONSOLIDATION.md): the delta=1 sanity pilot on a SMALLER
instance, after three attempts on the full T_minutes=660 instance did not prove either
arm (300s/arm: ~47-48% gaps; 3600s/arm, D76: 6.14%/7.02%, neither proven).

    python scripts/delta1_short_horizon_pilot.py [--Q 2] [--time-limit 900] [--policy start]

Identical structure to `delta1_monolith_pilot.py` (same two arms: A slot-only recourse,
B minute recourse, both priced at minute fidelity for the fair comparison) but against
`configs/milp/baseline_d9_p56_monolith_delta1_t180.yaml` -- T_minutes cut from 660 to
180 (see that config's own header for exactly what this drops and keeps). A smaller
instance that proves is worth more than the full instance that does not.
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

BASE_CONFIG = "configs/milp/baseline_d9_p56_monolith_delta1_t180.yaml"
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
    ap.add_argument("--time-limit", type=float, default=900.0)
    ap.add_argument("--policy", default="start")
    ap.add_argument("--p-minutes", type=float, default=None)
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

    delta = 1
    Q = args.Q
    print(f"[{_now()}] delta1 SHORT-HORIZON pilot starting: Q={Q}, policy={args.policy}, "
          f"time_limit={args.time_limit}s per solve, config={BASE_CONFIG}", flush=True)

    cfg = load_config(BASE_CONFIG)
    T_minutes = int(cfg.model.time.T_minutes)
    p_min = float(args.p_minutes if args.p_minutes is not None else cfg.service.p_minutes)
    seats = float(cfg.service.S)
    wmax = float(cfg.service.Wmax_minutes)

    print(f"[{_now()}] loading request minutes from {DEMAND_FILE} (full file; requests "
          f">= T_minutes={T_minutes} are outside this instance's horizon and will be "
          f"dropped+warned by aggregation/pricing, not silently) ...", flush=True)
    t0 = time.monotonic()
    requests_full = load_request_minutes(DEMAND_FILE)
    # Truncate to this instance's own horizon explicitly, so arm B (which does not
    # aggregate through aggregate_requests) sees the same reduced demand as arm A,
    # rather than the full 300-request file with everything past 180 min guaranteed
    # unserved by construction (that would still price correctly, since no arc can
    # reach past the horizon, but it silently changes what "unserved" means here --
    # explicit truncation keeps both arms honestly comparable to the stated reduction).
    requests = {
        d: [m for m in times if m < T_minutes] for d, times in requests_full.items()
    }
    kept = sum(len(v) for v in requests.values())
    dropped = sum(len(v) for v in requests_full.values()) - kept
    print(f"[{_now()}] kept {kept} request(s) within [0,{T_minutes}) minutes, "
          f"dropped {dropped} outside it", flush=True)

    mp, sp = _prepare_params(cfg, {})
    mp, sp = dict(mp), dict(sp)
    sp["p"] = p_min / float(delta)
    mp["slot_resolution"] = delta
    sp["slot_resolution"] = delta
    mp.update(_energy_params_for_resolution(cfg, delta))
    sp.update(_energy_params_for_resolution(cfg, delta))
    mp["Q"] = Q
    mp["solve_time_limit_s"] = args.time_limit
    mp.pop("T", None)

    print(f"[{_now()}] === solver A: slot-only monolith, T={T_minutes // delta} slots ===",
          flush=True)
    t0 = time.monotonic()
    solver_a = MonolithSolver(MobautoMilpModel, cfg, mp, sp)
    result_a = solver_a.run()
    dt_a = time.monotonic() - t0
    print(f"[{_now()}] solver A done in {dt_a:.1f}s, status={result_a.status.name}, "
          f"obj={result_a.best_upper_bound}, bound={result_a.best_lower_bound}", flush=True)
    sched_a = _schedule(solver_a.master.m)

    print(f"[{_now()}] === solver B: minute-recourse master ===", flush=True)
    t0 = time.monotonic()
    master_b = MobautoMilpModel(dict(mp))
    master_b.initialize()
    attach_minute_recourse(
        master_b.m, requests, delta, seats, wmax, p_min,
        policy=args.policy, objective_scale=1.0 / float(delta),
    )
    result_b = master_b.solve()
    dt_b = time.monotonic() - t0
    print(f"[{_now()}] solver B done in {dt_b:.1f}s, status={result_b.status.name}, "
          f"obj={result_b.objective}, bound={getattr(result_b, 'best_bound', None)}", flush=True)
    sched_b = _schedule(master_b.m)

    ra = price_schedule_at_minutes(sched_a, requests, delta, seats, wmax, p_min, policy=args.policy)
    rb = price_schedule_at_minutes(sched_b, requests, delta, seats, wmax, p_min, policy=args.policy)
    gain = 100.0 * (ra.total_cost - rb.total_cost) / ra.total_cost if ra.total_cost else 0.0

    print()
    print("=" * 90)
    print(f"delta=1 T_minutes={T_minutes}  Q={Q}  policy={args.policy}")
    print(f"  A (slot-only, priced @ minute): cost={ra.total_cost:.2f}  unserved={ra.unserved_passengers:.0f}"
          f"  status={result_a.status.name}")
    print(f"  B (minute-recourse):            cost={rb.total_cost:.2f}  unserved={rb.unserved_passengers:.0f}"
          f"  status={result_b.status.name}")
    print(f"  A own claimed cost (rescaled):   {float(result_a.best_upper_bound) * delta:.4f}")
    print(f"  B own claimed cost (rescaled):   {float(result_b.objective) * delta:.4f}")
    print(f"  gain = {gain:.4f}%   same schedule? {'yes' if sched_a == sched_b else 'no'}")
    print(f"  solver A: {dt_a:.1f}s")
    print(f"  solver B: {dt_b:.1f}s")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
