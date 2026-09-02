"""Push the delta=1, Q=2 minute-recourse monolith toward optimality, tee'd for external monitoring.

    python scripts/delta1_to_optimality.py [--Q 2] [--outer-time-limit 10800]

Runs ONLY the minute-recourse arm (the counterpart to Benders' minute-level subproblem --
see scripts/delta1_monolith_pilot.py for the slot-only arm and the original two-arm
sanity check). `--outer-time-limit` is a safety net, not the real stopping rule: the real
rule is external -- a monitor script polls this process's tee'd CPLEX log every 5 minutes
and kills it if the gap has not dropped by at least 1 percentage point since the last
check (docs_decisions.md entry to follow records both numbers either way).

`solver_tee=True` so CPLEX's node log streams to stdout, where the external monitor can
read the running gap.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--Q", type=int, default=2)
    ap.add_argument("--outer-time-limit", type=float, default=10800.0)
    ap.add_argument("--policy", default="start")
    args = ap.parse_args()

    from mobauto2_milp.config import load_config
    from mobauto2_milp.app import _prepare_params, _energy_params_for_resolution
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_benders.minute_pricer import (
        attach_minute_recourse,
        load_request_minutes,
        price_schedule_at_minutes,
    )
    import pyomo.environ as pyo

    delta = 1
    Q = args.Q
    print(f"[{_now()}] delta1-to-optimality starting: Q={Q}, policy={args.policy}, "
          f"outer_time_limit={args.outer_time_limit}s, instance={DEMAND_FILE}", flush=True)

    cfg = load_config(BASE_CONFIG)
    p_min = float(cfg.service.p_minutes)
    seats = float(cfg.service.S)
    wmax = float(cfg.service.Wmax_minutes)

    requests = load_request_minutes(DEMAND_FILE)

    mp, sp = _prepare_params(cfg, {})
    mp, sp = dict(mp), dict(sp)
    sp["p"] = p_min / float(delta)
    mp["slot_resolution"] = delta
    sp["slot_resolution"] = delta
    # BUG FIX: recompute delta_chg for the overridden resolution (see
    # delta1_monolith_pilot.py / sweep_multiresolution.py for the full explanation).
    mp.update(_energy_params_for_resolution(cfg, delta))
    sp.update(_energy_params_for_resolution(cfg, delta))
    mp["Q"] = Q
    mp["solve_time_limit_s"] = args.outer_time_limit
    mp["solver_tee"] = True
    mp.pop("T", None)

    print(f"[{_now()}] building master (minute-recourse, T=660 slots) ...", flush=True)
    t0 = time.monotonic()
    master = MobautoMilpModel(dict(mp))
    master.initialize()
    print(f"[{_now()}] attaching minute recourse ...", flush=True)
    attach_minute_recourse(
        master.m, requests, delta, seats, wmax, p_min,
        policy=args.policy, objective_scale=1.0 / float(delta),
    )
    print(f"[{_now()}] solving (outer cap {args.outer_time_limit:.0f}s; the real stopping rule is "
          f"the external monitor) ...", flush=True)
    master.solve()
    dt = time.monotonic() - t0
    print(f"[{_now()}] solve returned after {dt:.1f}s", flush=True)

    out: dict[str, list[int]] = {"OUT": [], "RET": []}
    for tau in master.m.T:
        for q in master.m.Q:
            if float(pyo.value(master.m.yOUT[q, tau], exception=False) or 0.0) > 0.5:
                out["OUT"].append(int(tau))
            if float(pyo.value(master.m.yRET[q, tau], exception=False) or 0.0) > 0.5:
                out["RET"].append(int(tau))
    out["OUT"].sort()
    out["RET"].sort()

    priced = price_schedule_at_minutes(out, requests, delta, seats, wmax, p_min, policy=args.policy)
    print()
    print("=" * 90)
    print(f"FINAL (or last-observed-before-kill): delta=1 Q={Q} policy={args.policy}")
    print(f"  repriced total_cost={priced.total_cost:.2f}  unserved={priced.unserved_passengers:.0f}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
