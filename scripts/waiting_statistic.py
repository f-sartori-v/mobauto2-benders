"""B12. Every waiting-time statistic, with the definition that makes it a number.

    python scripts/waiting_statistic.py [--Q 2] [--out FILE.json]

THE DEFECT (audit item 3.6). The report quoted "13.9-15.9 min" average waiting for ONE
FIXED SCHEDULE. A range implies something varies. Nothing in the sentence said what, and
for a fixed schedule the honest possibilities are all choices of DEFINITION rather than
of schedule:

    scenario                which day's arrivals are being averaged over
    direction               OUT, RET, or both pooled
    assignment resolution   slot-level waits (tau - t) or minute-level (dep - arrival)
    departure offset o      where inside its slot the departure is assumed to leave
    same-slot eligibility   whether a zero-wait boarding is admissible at all
    denominator             carried passengers, or all requests
    unserved                excluded from the average, or counted as some wait

This script enumerates those choices against one fixed schedule and prints the average
each one produces, with the choice on the same line as the number. The range then either
reconstructs -- and the script says which two definitions are its endpoints -- or it does
not, and the range should be replaced by a single defined number.

WHAT IT DOES NOT DO. It does not average over definitions. A range over definitional
choices is not an uncertainty interval; it is a list of different quantities, and
reporting their spread as if it were measurement error is the error being corrected.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

BASE_CONFIG = "configs/milp/baseline_d9_p56_monolith.yaml"
SCENARIOS = {
    "base": "setups/base.yaml",
    "temporal_noise": "setups/base_vol20_pm60.yaml",
    "return_peak_advanced": "setups/base_ret_peak_adv.yaml",
    "midday_surge": "setups/base_plus100_out_noon.yaml",
}
REPORTED_RANGE = (13.9, 15.9)


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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--Q", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mobauto2_milp.app import _prepare_params
    from mobauto2_milp.config import load_config
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_milp.monolith import MonolithSolver
    from mobauto2_benders.minute_pricer import (
        load_request_minutes,
        price_schedule_at_minutes,
    )

    cfg = load_config(BASE_CONFIG)
    delta = int(cfg.model.time.slot_resolution)
    seats = float(cfg.service.S)
    wmax = float(cfg.service.Wmax_minutes)
    p_min = float(cfg.service.p * delta)

    mp, sp = _prepare_params(cfg, {})
    mp, sp = dict(mp), dict(sp)
    if args.Q:
        mp["Q"] = args.Q
    mp.pop("T", None)
    solver = MonolithSolver(MobautoMilpModel, cfg, mp, sp)
    result = solver.run()
    sched = _schedule(solver.master.m)

    print("B12 -- one fixed schedule, every waiting statistic it admits")
    print("=" * 78)
    print(
        f"schedule: {len(sched['OUT'])} OUT + {len(sched['RET'])} RET departures, "
        f"from the monolith optimum ({result.status.name}, obj="
        f"{result.best_upper_bound:.3f})"
    )
    print(
        f"regime: delta={delta}, p_min={p_min:g}, W_max={wmax:g}, S={seats:g}, "
        f"Q={mp['Q']}"
    )
    print()

    rows: list[dict] = []
    header = (
        f"{'scenario':>20s} | {'dir':>4s} | {'res':>6s} | {'policy':>8s} | "
        f"{'elig':>6s} | {'denom':>8s} | {'avg_min':>8s} | {'n':>5s}"
    )
    print(header)
    print("-" * len(header))

    for scen_name, path in SCENARIOS.items():
        requests = load_request_minutes(path)
        for policy, eligibility in itertools.product(
            ("start", "midpoint", "end"), ("forbid", "allow")
        ):
            priced = price_schedule_at_minutes(
                departures=sched,
                requests=requests,
                slot_resolution=delta,
                seats=seats,
                wmax_minutes=wmax,
                p_minutes=p_min,
                policy=policy,
                same_slot_eligibility=eligibility,
                scenario_label=scen_name,
            )
            for direction in ("both", "OUT", "RET"):
                if direction == "both":
                    sel = list(priced.assignment_rows)
                else:
                    sel = [
                        r for r in priced.assignment_rows if r.direction == direction
                    ]
                served_rows = [r for r in sel if r.served]
                carried = sum(r.passengers for r in served_rows)
                waiting = sum(r.passengers * r.wait_minutes for r in served_rows)
                allreq = sum(r.passengers for r in sel)
                for denom_name, n in (("carried", carried), ("all", allreq)):
                    avg = waiting / n if n else 0.0
                    rows.append(
                        {
                            "scenario": scen_name,
                            "direction": direction,
                            "assignment_resolution": "minute",
                            "departure_policy": policy,
                            "departure_offset": (
                                0.0
                                if policy == "start"
                                else (delta / 2.0 if policy == "midpoint" else delta)
                            ),
                            "same_slot_eligibility": eligibility,
                            "denominator": denom_name,
                            "exclude_unserved": denom_name == "carried",
                            "n": n,
                            "waiting_minutes": waiting,
                            "avg_wait_min": avg,
                        }
                    )
                    print(
                        f"{scen_name:>20s} | {direction:>4s} | {'minute':>6s} | "
                        f"{policy:>8s} | {eligibility:>6s} | {denom_name:>8s} | "
                        f"{avg:8.2f} | {n:5.0f}"
                    )

    print()
    lo, hi = REPORTED_RANGE
    matches_lo = [r for r in rows if abs(r["avg_wait_min"] - lo) < 0.1]
    matches_hi = [r for r in rows if abs(r["avg_wait_min"] - hi) < 0.1]
    print(f"RECONSTRUCTING THE REPORTED {lo}-{hi} MIN RANGE")
    print("-" * 78)
    if matches_lo and matches_hi:
        print(f"  lower endpoint {lo}: " + _describe(matches_lo[0]))
        print(f"  upper endpoint {hi}: " + _describe(matches_hi[0]))
        print("  What varies across the two endpoints:")
        for key in (
            "scenario", "direction", "departure_policy", "same_slot_eligibility",
            "denominator",
        ):
            a, b = matches_lo[0][key], matches_hi[0][key]
            if a != b:
                print(f"    {key}: {a} -> {b}")
    else:
        print(
            f"  NOT RECONSTRUCTED. No definition on this schedule yields "
            f"{lo if not matches_lo else hi} min."
        )
        vals = sorted(r["avg_wait_min"] for r in rows)
        print(
            f"  This schedule's averages span {vals[0]:.2f}-{vals[-1]:.2f} min across "
            f"{len(rows)} definitions."
        )
        print(
            "  So the published range cannot be a range over definitions on THIS "
            "schedule."
        )
        print(
            "  Replace it with a single defined number. The one this regime supports "
            "is:"
        )
        pick = [
            r
            for r in rows
            if r["scenario"] == "base"
            and r["direction"] == "both"
            and r["departure_policy"] == "start"
            and r["same_slot_eligibility"] == "forbid"
            and r["denominator"] == "carried"
        ]
        if pick:
            print("    " + _describe(pick[0]))
            print(
                "    -- the consolidated regime's own conventions: minute assignment, "
                "departure at its committed instant (o=0), the master's own tau>=t+1 "
                "eligibility, averaged over carried passengers."
            )

    if args.out:
        Path(args.out).write_text(
            json.dumps({"schedule": sched, "rows": rows}, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.out}")
    return 0


def _describe(row: dict) -> str:
    return (
        f"avg_wait_min={row['avg_wait_min']:.2f} scenario={row['scenario']} "
        f"direction={row['direction']} "
        f"assignment_resolution={row['assignment_resolution']} "
        f"departure_offset={row['departure_offset']:g} "
        f"same_slot_eligibility={row['same_slot_eligibility']} "
        f"denominator={row['denominator']} n={row['n']:.0f} "
        f"exclude_unserved={row['exclude_unserved']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
