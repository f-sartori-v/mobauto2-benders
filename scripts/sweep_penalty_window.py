"""Sweep p_minutes and Wmax jointly; publish the frontier an operator would need.

    python scripts/sweep_penalty_window.py [--p-minutes 14,28,56,112,224] [--wmax 30,45,60,90,120]

THE QUESTION (forward-plan A2, docs/FORWARD_PLAN_v1.md). p_minutes ~= 56 is an assumption
adopted in this work -- an indifference statement about the operator's tradeoff between one
more passenger carried and delay to those already carried -- not an elicited preference
(report Section res-penalty). Wmax is a companion policy choice: how long a passenger may be
made to wait before being counted unserved at all. Both jointly set the whole order-of-service
behaviour (report Section meth-priority), and the report's own next step is "to sweep it
jointly with Wmax and put the resulting frontier in front of whoever will operate the
service." Neither parameter has been swept, let alone jointly, until this script.

WHAT THIS RUNS. The baseline single-scenario instance (Q=2, T=22, delta=30,
configs/milp/baseline_d9_p56_monolith.yaml), solved to proven optimality once per
(p_minutes, Wmax) cell -- a monolithic MILP, so there is no cut machinery to interact with
the sweep and no ambiguity about which bound produced the number.

WHY EVERY CELL IS RE-PRICED AT MINUTE FIDELITY. Each cell's own reported (slot) objective is
known to overstate cost by 28.5% and to misprice waiting alone by 66-86% (D51). A frontier an
operator will actually read a decision off is exactly the kind of table that error must not
be allowed into, so every cell's resulting schedule is re-priced against the demand's actual
arrival minutes (minute_pricer.price_schedule_at_minutes) before being reported.

A NOTE ON WHAT THIS DOES AND DOES NOT SHOW. Every cell here is a schedule OPTIMISED under
slot-level valuation at that cell's (p, Wmax) -- not the minute-optimal schedule for that
cell. That is the same construction A1/D70 used for the same reason: it is what the model
today actually runs, and it is fair to itself (unlike quoting a slot-reported number, which
is not fair to the reader). It does not, by itself, say what a minute-level recourse would
choose at each cell -- that is A5-adjacent future work, not this sweep's question.
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

DEFAULT_P_MINUTES = (14.0, 28.0, 56.0, 112.0, 224.0)
DEFAULT_WMAX = (30.0, 45.0, 60.0, 90.0, 120.0)


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


def _parse_grid(raw: str) -> tuple[float, ...]:
    return tuple(float(x) for x in raw.split(","))


def _solve_cell(p_minutes: float, wmax_minutes: float, Q: int | None, delta: int):
    from mobauto2_milp.app import _prepare_params
    from mobauto2_milp.config import load_config
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_milp.monolith import MonolithSolver

    cfg = load_config(BASE_CONFIG)
    mp, sp = _prepare_params(cfg, {})
    mp, sp = dict(mp), dict(sp)
    sp["p"] = float(p_minutes) / float(delta)
    sp["Wmax_minutes"] = float(wmax_minutes)
    sp.pop("Wmax_slots", None)
    mp["slot_resolution"] = delta
    sp["slot_resolution"] = delta
    if Q:
        mp["Q"] = Q
    mp.pop("T", None)

    solver = MonolithSolver(MobautoMilpModel, cfg, mp, sp)
    result = solver.run()
    if result.best_upper_bound is None:
        raise RuntimeError(
            f"no incumbent at p_minutes={p_minutes} Wmax={wmax_minutes} -- check the solver "
            "backend is available (see docs/PROJECT_STATE_v6.md section 6.2)"
        )
    return result, _schedule(solver.master.m)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--p-minutes", default=",".join(str(v) for v in DEFAULT_P_MINUTES))
    ap.add_argument("--wmax", default=",".join(str(v) for v in DEFAULT_WMAX))
    ap.add_argument("--slot", type=int, default=None)
    ap.add_argument("--Q", type=int, default=None)
    ap.add_argument("--policy", choices=("start", "midpoint", "end"), default="start")
    args = ap.parse_args()

    from mobauto2_benders.minute_pricer import load_request_minutes, price_schedule_at_minutes
    from mobauto2_milp.config import load_config

    cfg = load_config(BASE_CONFIG)
    delta = int(args.slot or cfg.model.time.slot_resolution)
    seats = float(cfg.service.S)
    p_grid = _parse_grid(args.p_minutes)
    wmax_grid = _parse_grid(args.wmax)
    requests = load_request_minutes(DEMAND_FILE)
    total_pax = float(len(requests.get("OUT", [])) + len(requests.get("RET", [])))

    print(f"instance: {DEMAND_FILE}, Q={args.Q or cfg.model.fleet.Q}, slot={delta}min, "
          f"policy={args.policy}, {total_pax:.0f} requests")
    print(f"p_minutes grid: {p_grid}")
    print(f"Wmax grid (min): {wmax_grid}")
    print()

    cells: list[dict] = []
    header = f"{'Wmax':>6s} | " + " | ".join(f"p={p:>6.0f}" for p in p_grid)
    print("Served / total, minute-honest:")
    print(header)
    print("-" * len(header))
    for wmax in wmax_grid:
        row_served = []
        for p in p_grid:
            result, schedule = _solve_cell(p, wmax, args.Q, delta)
            priced = price_schedule_at_minutes(
                schedule, requests, delta, seats, wmax, p, policy=args.policy
            )
            cells.append(
                {
                    "p_minutes": p,
                    "wmax_minutes": wmax,
                    "served": priced.served_passengers,
                    "unserved": priced.unserved_passengers,
                    "waiting_minutes": priced.waiting_minutes,
                    "total_cost": priced.total_cost,
                    "departures": len(schedule["OUT"]) + len(schedule["RET"]),
                    "status": result.status.name,
                }
            )
            row_served.append(f"{priced.served_passengers:5.0f}/{total_pax:.0f}")
        print(f"{wmax:6.0f} | " + " | ".join(f"{v:>8s}" for v in row_served))

    print()
    print("Average wait per served passenger (minutes, minute-honest):")
    print(header)
    print("-" * len(header))
    idx = 0
    for wmax in wmax_grid:
        row = []
        for _p in p_grid:
            c = cells[idx]
            idx += 1
            avg_wait = c["waiting_minutes"] / c["served"] if c["served"] else 0.0
            row.append(f"{avg_wait:8.1f}")
        print(f"{wmax:6.0f} | " + " | ".join(row))

    print()
    print("Total cost, passenger-minutes (waiting + p_minutes x unserved):")
    print(header)
    print("-" * len(header))
    idx = 0
    for wmax in wmax_grid:
        row = []
        for _p in p_grid:
            c = cells[idx]
            idx += 1
            row.append(f"{c['total_cost']:8.0f}")
        print(f"{wmax:6.0f} | " + " | ".join(row))

    print()
    print("Read served/total and average wait together, not the total cost column alone:")
    print("total cost is dominated by whichever term p_minutes makes expensive, exactly the")
    print("mechanism Section res-penalty describes -- it is not a single-number summary of")
    print("service quality on its own.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
