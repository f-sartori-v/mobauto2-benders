"""B6. How big is the same-slot eligibility convention effect, on its own?

    python scripts/same_slot_convention.py [--delta 1] [--Q 2] [--time-limit 120]
                                           [--horizon-minutes 660] [--out FILE.json]

THE DEFECT THIS MEASURES (audit item 2.4). The slot recourse forbids same-slot boarding
structurally, through `tau >= t+1`. The minute recourse admitted a zero-minute wait,
because its only reachability filter was `0 <= dep - m <= Wmax`. Nothing recorded the
difference. At delta = 30 it touches only arrivals landing exactly on a slot boundary;
at delta = 1 the two arc sets differ everywhere, so the reported 1.40% and 1.60%
delta=1 figures compared eligibility CONVENTIONS and solution STRATEGIES at the same
time and cannot be read as a decomposition result.

TWO MEASUREMENTS, AND THE FIRST IS THE CLEAN ONE.

  Part 1 -- CONVENTION EFFECT AT A FIXED SCHEDULE. One schedule, priced twice. Nothing
  is optimised, so nothing but the convention can move, and the difference IS the
  convention effect. This is exact, costs two LPs, and is the number B6 calls "a result
  in its own right".

  Part 2 -- OPTIMISED ARMS. Both arms re-optimised under their own convention. This is
  the like-for-like comparison the delta=1 experiment was supposed to be, and it is
  reported with proof status per arm: at delta=1 the monolith does not close in any
  budget this script would impose, so a truncated arm is labelled, not quoted.

WHY THE PARTS ARE SEPARATE. Part 2 alone cannot attribute its difference -- an
optimiser that stops on the clock differs from another that stops on the clock for
reasons that have nothing to do with arc sets. Part 1 has no optimiser in it at all.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

BASE_CONFIG = "configs/milp/baseline_d9_p56_monolith.yaml"
DEMAND_FILE = "setups/base.yaml"
CONVENTIONS = ("forbid", "allow")


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


def _reference_schedule(delta: int, T: int, Q: int, trip_slots: int) -> dict:
    """A fixed, legal schedule to price under both conventions.

    Deliberately NOT an optimum of either arm. An optimum of one convention is a
    schedule chosen to suit that convention's arc set, so pricing it under the other
    would report the convention effect plus a selection effect. A neutral schedule --
    departures spread evenly across the slots that can carry one -- is chosen by
    neither and measures only the arcs.
    """
    from mobauto2_benders.signature import departures_are_possible

    ok_out, ok_ret = departures_are_possible(T, trip_slots)
    out_slots = [t for t in range(T) if ok_out[t]]
    ret_slots = [t for t in range(T) if ok_ret[t]]
    # Roughly one round trip per vehicle per hour, spread evenly.
    n_trips = max(1, int(Q * (T * delta) / 60.0))
    step_out = max(1, len(out_slots) // max(1, n_trips))
    step_ret = max(1, len(ret_slots) // max(1, n_trips))
    return {
        "OUT": out_slots[::step_out][:n_trips],
        "RET": ret_slots[::step_ret][:n_trips],
    }


def _solve_minute(cfg, delta: int, Q: int, T_minutes: int, policy: str,
                  eligibility: str, time_limit: float):
    from mobauto2_milp.app import _prepare_params, _energy_params_for_resolution
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_benders.minute_pricer import (
        attach_minute_recourse,
        load_request_minutes,
    )

    mp, _sp = _prepare_params(cfg, {})
    mp = dict(mp)
    mp["slot_resolution"] = delta
    mp["T_minutes"] = int(T_minutes)
    mp.pop("T", None)
    mp["Q"] = int(Q)
    mp["solve_time_limit_s"] = float(time_limit)
    mp.update(_energy_params_for_resolution(cfg, delta))

    master = MobautoMilpModel(mp)
    master.initialize()
    attach_minute_recourse(
        master.m,
        requests=load_request_minutes(DEMAND_FILE),
        slot_resolution=delta,
        seats=float(cfg.service.S),
        wmax_minutes=float(cfg.service.Wmax_minutes),
        p_minutes=float(cfg.service.p * cfg.model.time.slot_resolution),
        policy=policy,
        objective_scale=1.0 / float(delta),
        same_slot_eligibility=eligibility,
    )
    t0 = time.perf_counter()
    res = master.solve()
    seconds = time.perf_counter() - t0
    stats = master.last_solve_stats()
    return master, res, seconds, stats


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--delta", type=int, default=1)
    ap.add_argument("--Q", type=int, default=2)
    ap.add_argument("--horizon-minutes", type=int, default=660)
    ap.add_argument("--policy", default="start")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--skip-optimisation", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mobauto2_milp.config import load_config
    from mobauto2_benders.minute_pricer import (
        load_request_minutes,
        price_schedule_at_minutes,
    )

    cfg = load_config(BASE_CONFIG)
    delta = int(args.delta)
    T = max(1, int(args.horizon_minutes) // delta)
    seats = float(cfg.service.S)
    wmax = float(cfg.service.Wmax_minutes)
    p_min = float(cfg.service.p * cfg.model.time.slot_resolution)
    trip_slots = max(
        1,
        int(round(float(cfg.model.time.trip_duration_minutes) / float(delta))),
    )
    requests = load_request_minutes(DEMAND_FILE)

    record: dict = {
        "delta": delta,
        "Q": args.Q,
        "horizon_minutes": args.horizon_minutes,
        "T": T,
        "policy": args.policy,
        "p_minutes": p_min,
        "Wmax_minutes": wmax,
        "seats": seats,
    }

    print("B6 -- the same-slot eligibility convention, measured on its own")
    print("=" * 78)
    print(
        f"delta={delta} min, T={T} slots, Q={args.Q}, policy={args.policy}, "
        f"p_min={p_min:g}, W_max={wmax:g}, S={seats:g}"
    )
    print()
    print("PART 1 -- one FIXED schedule, priced under both conventions.")
    print("Nothing is optimised here, so nothing but the convention can move.")
    print()

    sched = _reference_schedule(delta, T, int(args.Q), trip_slots)
    print(f"reference schedule: {len(sched['OUT'])} OUT + {len(sched['RET'])} RET "
          f"departures at slots OUT={sched['OUT'][:8]}... RET={sched['RET'][:8]}...")
    print()
    header = (
        f"{'convention':>10s} | {'cost':>11s} | {'wait_min':>10s} | "
        f"{'served':>7s} | {'unserved':>8s} | {'free-seat':>9s}"
    )
    print(header)
    print("-" * len(header))
    priced = {}
    for eligibility in CONVENTIONS:
        r = price_schedule_at_minutes(
            departures=sched,
            requests=requests,
            slot_resolution=delta,
            seats=seats,
            wmax_minutes=wmax,
            p_minutes=p_min,
            policy=args.policy,
            same_slot_eligibility=eligibility,
        )
        priced[eligibility] = r
        print(
            f"{eligibility:>10s} | {r.total_cost:11.1f} | {r.waiting_minutes:10.1f} | "
            f"{r.served_passengers:7.0f} | {r.unserved_passengers:8.0f} | "
            f"{r.rejected_with_free_seat:9.0f}"
        )
    a, b = priced["forbid"], priced["allow"]
    effect = (
        100.0 * (a.total_cost - b.total_cost) / b.total_cost if b.total_cost else 0.0
    )
    print("-" * len(header))
    print(
        f"CONVENTION EFFECT at a fixed schedule: forbid costs {effect:+.2f}% of allow "
        f"({a.total_cost - b.total_cost:+.1f} passenger-minutes)"
    )
    record["fixed_schedule"] = {
        e: {
            "total_cost": r.total_cost,
            "waiting_minutes": r.waiting_minutes,
            "served": r.served_passengers,
            "unserved": r.unserved_passengers,
            "rejected_with_free_seat": r.rejected_with_free_seat,
        }
        for e, r in priced.items()
    }
    record["convention_effect_pct_fixed_schedule"] = effect
    print()

    if args.skip_optimisation:
        print("PART 2 skipped (--skip-optimisation).")
    else:
        print("PART 2 -- both arms RE-OPTIMISED under their own convention.")
        print(f"Budget {args.time_limit:g} s per arm. An arm that stops on the clock is")
        print("labelled: its objective is an upper bound, not an optimum, and the two")
        print("arms are then not comparable as optima.")
        print()
        header2 = (
            f"{'convention':>10s} | {'status':>12s} | {'obj(slot)':>11s} | "
            f"{'bound':>11s} | {'gap':>7s} | {'seconds':>8s} | proof"
        )
        print(header2)
        print("-" * len(header2))
        record["optimised"] = {}
        for eligibility in CONVENTIONS:
            try:
                master, res, seconds, stats = _solve_minute(
                    cfg, delta, int(args.Q), int(args.horizon_minutes),
                    args.policy, eligibility, args.time_limit,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"{eligibility:>10s} | {'ERROR':>12s} | {type(exc).__name__}: {exc}")
                record["optimised"][eligibility] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            status = res.status.name
            gap = stats.get("gap")
            on_clock = seconds >= float(args.time_limit) * 0.99
            proven = status == "OPTIMAL" and not on_clock
            obj = res.objective
            bound = res.lower_bound
            print(
                f"{eligibility:>10s} | {status:>12s} | "
                f"{(obj if obj is not None else float('nan')):11.2f} | "
                f"{(bound if bound is not None else float('nan')):11.2f} | "
                f"{(100.0 * gap if gap is not None else float('nan')):6.2f}% | "
                f"{seconds:8.1f} | "
                + ("proven" if proven else "CLOCK-TRUNCATED: obj is an upper bound only")
            )
            record["optimised"][eligibility] = {
                "status": status,
                "objective_slot_units": obj,
                "bound": bound,
                "gap": None if gap is None else float(gap),
                "seconds": seconds,
                "clock_truncated": bool(on_clock),
                "proven": bool(proven),
                "schedule": _schedule(master.m),
            }
        both = record["optimised"]
        if all(
            isinstance(v, dict) and v.get("proven") for v in both.values()
        ):
            fo = both["forbid"]["objective_slot_units"]
            al = both["allow"]["objective_slot_units"]
            if al:
                print()
                print(
                    f"OPTIMISED convention effect: forbid is {100.0 * (fo - al) / al:+.2f}% "
                    "of allow, both proven."
                )
        else:
            print()
            print("NOT COMPARABLE AS OPTIMA: at least one arm stopped on the clock.")
            print("Report Part 1 as the convention effect; Part 2's arms are bounds.")

    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
