"""Benders with the MASTER also at slot_resolution=1 -- the true apples-to-apples test
against the delta=1 monolith.

    python scripts/delta1_benders_master.py [--time-limit 300]

THE POINT. Every earlier Benders-vs-monolith comparison this session ran used the
project's standard master (slot_resolution=30, T=22 -- 22 candidate departure slots).
That is NOT the same decision space as the delta=1 monolith (T=660): the master could
never reach the schedules the monolith reaches, no matter how long it ran, because its
own first-stage variables are coarser. This script gives the Benders MASTER the same
T=660 first-stage grid, decomposed the normal way (minute-level recourse via LPs/duals,
not embedded in the master's own MIP). If the master alone -- without the monolith's
combined recourse constraints -- is tractable at this size, that is the actual "value of
Benders" answer for this instance; if it is not, Benders offers nothing here either, and
the bottleneck is confirmed to be the first-stage grid itself, not the recourse.

BUG FIX CARRIED OVER (found while building this script's monolith-side siblings):
delta_chg (energy per charging slot) is baked into mp/sp by _prepare_params() at the
CONFIG's own slot_resolution (30) and must be recomputed for the overridden resolution
via _energy_params_for_resolution -- otherwise charging runs at 30x the correct rate at
delta=1.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

BENDERS_CONFIG = "configs/phase1/rq5_benders_minute_p56.yaml"
DEMAND_FILE = "setups/base.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--Q", type=int, default=2)
    args = ap.parse_args()

    from mobauto2_benders.app import _prepare_params, _run_single, _energy_params_for_resolution
    from mobauto2_benders.config import load_config
    from mobauto2_benders.minute_pricer import load_request_minutes, price_schedule_at_minutes
    import pyomo.environ as pyo

    def _schedule(model):
        out = {"OUT": [], "RET": []}
        for tau in model.T:
            for q in model.Q:
                if float(pyo.value(model.yOUT[q, tau], exception=False) or 0.0) > 0.5:
                    out["OUT"].append(int(tau))
                if float(pyo.value(model.yRET[q, tau], exception=False) or 0.0) > 0.5:
                    out["RET"].append(int(tau))
        out["OUT"] = sorted(set(out["OUT"]))
        out["RET"] = sorted(set(out["RET"]))
        return out

    delta = 1
    cfg = load_config(BENDERS_CONFIG)
    cfg.solver.total_time_limit_s = float(args.time_limit)

    mp, sp = _prepare_params(cfg, {})
    mp, sp = dict(mp), dict(sp)
    mp["slot_resolution"] = delta
    sp["slot_resolution"] = delta
    mp.update(_energy_params_for_resolution(cfg, delta))
    sp.update(_energy_params_for_resolution(cfg, delta))
    mp["Q"] = int(args.Q)
    mp.pop("T", None)
    sp.pop("T", None)

    print(f"[start] Benders master at slot_resolution={delta} (T=660), Q={args.Q}, "
          f"total_time_limit_s={args.time_limit}s, delta_chg={mp.get('delta_chg')}", flush=True)

    t0 = time.monotonic()
    result, master = _run_single(cfg, mp, sp, emit_cli_output=True)
    dt = time.monotonic() - t0

    requests = load_request_minutes(DEMAND_FILE)
    seats = float(cfg.subproblem.S)
    wmax = float(cfg.subproblem.Wmax_minutes)
    p_min = float(cfg.subproblem.p_minutes)
    policy = str(cfg.subproblem.departure_policy)
    sched = _schedule(master.m)
    priced = price_schedule_at_minutes(sched, requests, delta, seats, wmax, p_min, policy=policy)

    print()
    print("=" * 90)
    print(f"Benders master@delta=1 result: LB={result.best_lower_bound} UB={result.best_upper_bound} "
          f"iters={result.iterations} pax_served={result.pax_served}/{result.pax_total} "
          f"clock_truncated_master_solves={result.clock_truncated_master_solves}")
    print(f"wall time: {dt:.1f}s")
    print(f"repriced total_cost={priced.total_cost:.2f}  unserved={priced.unserved_passengers:.0f}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
