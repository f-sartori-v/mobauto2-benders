"""B2, second half. Does the charger constraint's cost grow or shrink with the fleet?

    python scripts/charger_capacity_sweep.py [--time-limit 60] [--out FILE.json]

THE QUESTION LEFT OPEN. `docs/CLOSEOUT_T54.md` section 3b measured `K_chg` at Q=2 on the
consolidated regime: one charger for two vehicles costs +1.52 % of the objective. One
point is not a trend. Two readings of it are equally plausible a priori:

  * the cost GROWS with the fleet -- more vehicles competing for the same charger, so
    the queue lengthens and the constraint bites harder;
  * the cost SHRINKS -- a larger fleet has more slack to reschedule around a charger,
    so the same absolute shortage is easier to absorb.

They imply opposite things about what a site should buy, so the sweep is worth its few
minutes.

WHAT IS SWEPT. Two instances, because the answer may not be a property of the model:
the baseline (300 requests) and the 450-passenger target (`setups/base_scale450.yaml`),
which is the scale the trial actually declares. `K_chg` runs from 1 to Q at each fleet
size; `K_chg = Q` is the unconstrained reference each row is measured against.

The divisible form is swept. The indivisible (`charger_occupancy_binary`) form is run at
the extreme point only -- it is a different physical claim, not a variant of the same
number, and mixing them in one trend line would be exactly the manifest violation the
emitter refuses.
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

INSTANCES = {
    "baseline_300": "configs/milp/baseline_d9_p56_monolith.yaml",
    "target_450": "configs/milp/target_scale450_monolith.yaml",
}


def _schedule_key(model) -> tuple:
    import pyomo.environ as pyo

    out = []
    for tau in model.T:
        for q in model.Q:
            if float(pyo.value(model.yOUT[q, tau], exception=False) or 0.0) > 0.5:
                out.append(("O", int(q), int(tau)))
            if float(pyo.value(model.yRET[q, tau], exception=False) or 0.0) > 0.5:
                out.append(("R", int(q), int(tau)))
    return tuple(sorted(out))


def _charging_slots(model) -> float:
    import pyomo.environ as pyo

    total = 0.0
    for q in model.Q:
        for t in model.T:
            total += float(pyo.value(model.c[q, t], exception=False) or 0.0)
    return total


def _run(cfg, Q: int, K_chg, binary: bool, time_limit: float):
    from mobauto2_milp.app import _prepare_params
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_milp.monolith import MonolithSolver

    mp, sp = _prepare_params(cfg, {})
    mp, sp = dict(mp), dict(sp)
    mp["Q"] = int(Q)
    mp.pop("T", None)
    mp["solve_time_limit_s"] = float(time_limit)
    if K_chg is not None:
        mp["K_chg"] = int(K_chg)
    mp["charger_occupancy_binary"] = bool(binary)

    t0 = time.perf_counter()
    solver = MonolithSolver(MobautoMilpModel, cfg, mp, sp)
    result = solver.run()
    seconds = time.perf_counter() - t0
    stats = solver.master.last_solve_stats()
    return solver, result, seconds, stats


def main() -> int:
    from mobauto2_milp.config import load_config

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--Q-baseline", default="2,3,4,5")
    ap.add_argument("--Q-target", default="4,6,8")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    plan = {
        "baseline_300": [int(x) for x in args.Q_baseline.split(",")],
        "target_450": [int(x) for x in args.Q_target.split(",")],
    }

    rows: list[dict] = []
    print("B2 -- charger capacity across fleet sizes and instances")
    print("=" * 84)
    print(f"budget {args.time_limit:g} s per cell; divisible form unless marked binary")
    print()
    header = (
        f"{'instance':>13s} | {'Q':>2s} | {'K_chg':>6s} | {'status':>9s} | "
        f"{'objective':>10s} | {'vs K=Q':>8s} | {'%':>7s} | {'chg slots':>9s} | sched"
    )
    print(header)
    print("-" * len(header))

    for inst, config_path in INSTANCES.items():
        cfg = load_config(config_path)
        for Q in plan[inst]:
            # Reference first: K_chg = Q is the unconstrained model every row below is
            # measured against, and it must be the SAME fleet size or the comparison
            # is against a different problem.
            _s, ref, ref_secs, ref_stats = _run(cfg, Q, None, False, args.time_limit)
            ref_obj = ref.best_upper_bound
            ref_key = _schedule_key(_s.master.m)
            ref_proven = ref.status.name == "OPTIMAL"
            rows.append(
                dict(instance=inst, Q=Q, K_chg=Q, form="divisible",
                     status=ref.status.name, objective=ref_obj, delta=0.0, pct=0.0,
                     charging_slots=_charging_slots(_s.master.m),
                     seconds=ref_secs, proven=ref_proven, schedule_changed=False)
            )
            print(
                f"{inst:>13s} | {Q:2d} | {'Q=' + str(Q):>6s} | "
                f"{ref.status.name:>9s} | {ref_obj:10.4f} | {'-':>8s} | {'-':>7s} | "
                f"{_charging_slots(_s.master.m):9.2f} | reference"
            )

            for K in range(1, Q):
                s2, res, secs, stats = _run(cfg, Q, K, False, args.time_limit)
                obj = res.best_upper_bound
                proven = res.status.name == "OPTIMAL"
                delta = (obj - ref_obj) if (obj is not None and ref_obj) else float("nan")
                pct = 100.0 * delta / ref_obj if ref_obj else float("nan")
                changed = _schedule_key(s2.master.m) != ref_key
                rows.append(
                    dict(instance=inst, Q=Q, K_chg=K, form="divisible",
                         status=res.status.name, objective=obj, delta=delta, pct=pct,
                         charging_slots=_charging_slots(s2.master.m),
                         seconds=secs, proven=proven and ref_proven,
                         schedule_changed=bool(changed))
                )
                print(
                    f"{inst:>13s} | {Q:2d} | {K:6d} | {res.status.name:>9s} | "
                    f"{obj:10.4f} | {delta:+8.4f} | {pct:+6.2f}% | "
                    f"{_charging_slots(s2.master.m):9.2f} | "
                    + ("changed" if changed else "same")
                    + ("" if (proven and ref_proven) else "  [NOT PROVEN]")
                )

            # The indivisible form at the tightest setting only.
            sb, resb, secsb, _st = _run(cfg, Q, 1, True, args.time_limit)
            objb = resb.best_upper_bound
            deltab = (objb - ref_obj) if (objb is not None and ref_obj) else float("nan")
            rows.append(
                dict(instance=inst, Q=Q, K_chg=1, form="binary_occupancy",
                     status=resb.status.name, objective=objb, delta=deltab,
                     pct=100.0 * deltab / ref_obj if ref_obj else float("nan"),
                     charging_slots=_charging_slots(sb.master.m), seconds=secsb,
                     proven=resb.status.name == "OPTIMAL" and ref_proven,
                     schedule_changed=_schedule_key(sb.master.m) != ref_key)
            )
            print(
                f"{inst:>13s} | {Q:2d} | {'1 bin':>6s} | {resb.status.name:>9s} | "
                f"{objb:10.4f} | {deltab:+8.4f} | "
                f"{(100.0 * deltab / ref_obj if ref_obj else float('nan')):+6.2f}% | "
                f"{_charging_slots(sb.master.m):9.2f} | indivisible form"
            )
        print("-" * len(header))

    print()
    print("READING THE TREND")
    print("  PROVEN CELLS ONLY. A trend line drawn through clock-truncated cells is a")
    print("  trend through upper bounds, and the truncation is not even in a consistent")
    print("  direction -- a truncated reference makes its own row's delta look SMALLER")
    print("  while a truncated constrained cell makes it look LARGER.")
    for inst in INSTANCES:
        tight = [
            r for r in rows
            if r["instance"] == inst and r["form"] == "divisible"
            and r["K_chg"] == 1 and r["proven"]
        ]
        skipped = [
            r for r in rows
            if r["instance"] == inst and r["form"] == "divisible"
            and r["K_chg"] == 1 and not r["proven"]
        ]
        desc = ", ".join(f"Q={r['Q']}: {r['pct']:+.2f}%" for r in sorted(tight, key=lambda r: r["Q"]))
        if len(tight) >= 2:
            trend = sorted(tight, key=lambda r: r["Q"])
            first, last = trend[0]["pct"], trend[-1]["pct"]
            direction = (
                "GROWS with the fleet" if last > first + 0.01
                else "SHRINKS with the fleet" if last < first - 0.01
                else "is flat in the fleet"
            )
            print(f"  {inst}: cost of K_chg=1 {direction} -- {desc}")
        elif tight:
            print(f"  {inst}: only one proven cell ({desc}); no trend can be read")
        else:
            print(f"  {inst}: NO proven cell at K_chg=1; no trend can be read")
        if skipped:
            qs = ", ".join(f"Q={r['Q']}" for r in sorted(skipped, key=lambda r: r["Q"]))
            print(f"    excluded as clock-truncated: {qs}")

        # The operationally useful number is not the trend, it is the threshold.
        binding = sorted(
            {
                r["K_chg"]
                for r in rows
                if r["instance"] == inst and r["form"] == "divisible"
                and r["proven"] and r["K_chg"] != r["Q"] and abs(r["pct"]) > 0.01
            }
        )
        free = sorted(
            {
                r["K_chg"]
                for r in rows
                if r["instance"] == inst and r["form"] == "divisible"
                and r["proven"] and r["K_chg"] != r["Q"] and abs(r["pct"]) <= 0.01
            }
        )
        if binding or free:
            print(
                f"    proven: K_chg in {binding} costs something, K_chg in {free} is free"
            )
    print()
    print(
        "  A cell marked [NOT PROVEN] is clock-truncated: its objective is an upper "
        "bound, so its\n  delta is not a measurement and must not be read as one."
    )

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
