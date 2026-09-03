"""One stochastic schedule against scenario-specific optima, at p_minutes=56.

    python scripts/stochastic_robustness.py [--Q 2] [--p-minutes 56] [--policy start]
                                            [--comparator slot|minute]

B13 (audit item 3.5) -- WHAT THIS TABLE IS CALLED, AND WHY IT MATTERS.

The default comparator (`--comparator slot`) builds each scenario's own optimum with the
SLOT recourse and then prices it at minute fidelity. The resulting gap is therefore
NEITHER of the two things it was reported as:

  * it is not EVPI, because EVPI compares against the optimum under the SAME valuation
    used for the evaluation, and these comparators were optimised under a different one;
  * it is not "the cost of hedging", for the same reason -- part of the gap is the
    comparator's own slot-vs-minute mismatch, not the price of committing to one
    schedule before the day is known.

The proof that it is neither is already in the old table: in one scenario the HEDGED
schedule beat its own scenario-specific comparator. A schedule optimised for scenario s
cannot be beaten on s by a schedule optimised for four scenarios at once -- unless the
comparator was not optimal for s under the valuation being applied. It was not.

So the table is named for what it is: "one stochastic schedule against scenario-specific
SLOT-optimal schedules". The number it reports is a real measurement of a real pair of
schedules; it just does not carry the interpretation that was attached to it.

`--comparator minute` computes the honest version. Each scenario's optimum is solved
under the SAME minute recourse used for the evaluation (minute_pricer.attach_minute_
recourse, pinned to theta by equality), so the comparison is like-for-like and VSS and
EVPI are then defined:

    EVPI = E_s[ Q(x*_s, s) ]  -  E_s[ Q(x*_hedged, s) ]      (both under minute valuation)
    VSS  = E_s[ Q(x*_mean-value, s) ] - E_s[ Q(x*_hedged, s) ]

This script reports EVPI under `--comparator minute`. VSS needs the mean-value solution
as a third arm and is not computed here; the table says so rather than labelling the
EVPI column as if it were both.

THE UNDERLYING QUESTION (forward-work item A1, docs/FORWARD_PLAN_v1.md). The conference-era result
compared one schedule built to serve four demand scenarios against the four per-scenario
deterministic optima, and reported that planning under uncertainty spreads departures
rather than chasing peaks, trading served passengers for waiting regularity. That result
predates two corrections this repository has since made: p_minutes was 1500 there, 27x
the operator's stated indifference (D53), and waiting was priced at slot resolution, known
since D51 to overstate the objective by 28.5% and waiting alone by 66-86%. The mechanism is
not in dispute; the NUMBERS are void until regenerated under the corrected regime
(docs/PROJECT_STATE_v6.md section 3, "Not superseded, and not quotable either" in the
report's Results section). This script does that regeneration.

THE FOUR SCENARIOS. Exactly the instance family the T.5.4 report's Results section states,
each carrying weight 0.25, and the same four files configs/default.example.yaml lists:

    base                    setups/base.yaml                 the baseline demand day
    temporal_noise          setups/base_vol20_pm60.yaml       20% of requests shifted +-60min
    return_peak_advanced    setups/base_ret_peak_adv.yaml     the RET peak moved 2h earlier
    midday_surge            setups/base_plus100_out_noon.yaml +100 OUT requests, 11:00-13:00

WHAT THIS RUNS. Five monolithic solves at p_minutes=56, each to proven optimality:

  A  one schedule against all four scenarios jointly (weight 0.25 each)   -- "hedged"
  B1..B4  one schedule against each scenario alone                        -- "oracle" per day

Then every one of the five schedules is priced, at MINUTE fidelity, against every
scenario's actual arrival minutes (minute_pricer.price_schedule_at_minutes -- the same
construction scripts/multiscenario_check.py uses). Two comparisons are reported for each
scenario s: the hedged schedule's minute-cost under s, against oracle-s's own minute-cost
under s (which is oracle-s's true achievable cost, since it was optimised for exactly that
day). The weighted average of the gap across all four scenarios is the price of not
knowing in advance which day it will be -- the quantity a "value of the stochastic
solution" argument is normally built on.

WHAT THIS DOES NOT DO. It does not re-litigate the slot-level version of this comparison;
that number is void (docs/PROJECT_STATE_v6.md section 3) and this script does not
regenerate it, because the report asks for minute-level valuation specifically, not for a
slot-level replacement. Read the AVERAGED row, not any single scenario in isolation --
docs/RESEARCH_NOTE_v2.md section 3 (falsifier 3) already showed averaging across scenarios
attenuates a single-scenario effect sharply (3.84% against 8-50%), so this script's
weighted average is the operational figure and any per-scenario cell is a diagnostic, not
the result.
"""

from __future__ import annotations

import argparse
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


def _schedule(model) -> dict[str, list[int]]:
    """Aggregate departure slots as a multiset: schedule[d].count(tau) == Y_d[tau].

    Matches scripts/multiscenario_check.py's helper of the same name, so a schedule
    captured here is interchangeable with price_schedule_at_minutes exactly as that
    script already exercises it.
    """
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


def _solve(scenario_files: list[str], Q: int | None, p_minutes: float, delta: int,
           time_limit: float | None = None):
    from mobauto2_milp.app import _prepare_params
    from mobauto2_milp.config import load_config
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_milp.monolith import MonolithSolver

    cfg = load_config(BASE_CONFIG)
    mp, sp = _prepare_params(cfg, {})
    mp, sp = dict(mp), dict(sp)
    sp["scenario_files"] = list(scenario_files)
    sp.pop("demand_file", None)
    sp["p"] = float(p_minutes) / float(delta)
    mp["slot_resolution"] = delta
    sp["slot_resolution"] = delta
    if Q:
        mp["Q"] = Q
    mp.pop("T", None)
    if time_limit is not None:
        mp["solve_time_limit_s"] = float(time_limit)

    solver = MonolithSolver(MobautoMilpModel, cfg, mp, sp)
    result = solver.run()
    if result.best_upper_bound is None:
        raise RuntimeError(
            f"no incumbent for scenario_files={scenario_files} -- check the solver "
            "backend is available (see docs/PROJECT_STATE_v6.md section 6.2)"
        )
    return solver, result, _schedule(solver.master.m)


def _solve_minute(scenario_files: list[str], Q: int | None, p_minutes: float,
                  delta: int, policy: str, weights: list[float] | None = None,
                  time_limit: float | None = None):
    """B13. The scenario's optimum under the SAME minute recourse used to evaluate it.

    This is what makes the comparison like-for-like. `attach_minute_recourse` pins
    theta to the minute-level recourse by EQUALITY, so the returned schedule is optimal
    for exactly the valuation the pricing step then applies -- which is the property
    the slot comparator lacked and the reason its gap was neither EVPI nor a hedging
    cost.

    `objective_scale = 1/delta` keeps the recourse in slot-equivalent units so the
    first-stage epsilon and kappa terms keep the relative weight they have in every
    other run; without it a minute-scale theta would outweigh them by `delta` and
    quietly break ties differently.
    """
    from mobauto2_milp.app import _prepare_params
    from mobauto2_milp.config import load_config
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_benders.minute_pricer import (
        attach_minute_recourse,
        load_request_minutes,
    )

    cfg = load_config(BASE_CONFIG)
    mp, _sp = _prepare_params(cfg, {})
    mp = dict(mp)
    mp["slot_resolution"] = delta
    if Q:
        mp["Q"] = Q
    mp.pop("T", None)
    if time_limit is not None:
        mp["solve_time_limit_s"] = float(time_limit)

    master = MobautoMilpModel(mp)
    master.initialize()
    reqs = [load_request_minutes(f) for f in scenario_files]
    w = weights or [1.0 / len(reqs)] * len(reqs)
    attach_minute_recourse(
        master.m,
        requests=None,
        slot_resolution=delta,
        seats=float(cfg.service.S),
        wmax_minutes=float(cfg.service.Wmax_minutes),
        p_minutes=float(p_minutes),
        policy=policy,
        objective_scale=1.0 / float(delta),
        scenarios=list(zip(reqs, w)),
    )
    res = master.solve()
    if res.objective is None:
        raise RuntimeError(
            f"no incumbent for the minute-recourse arm on {scenario_files}"
        )
    return master, res, _schedule(master.m)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--comparator",
        choices=("slot", "minute"),
        default="slot",
        help=(
            "slot: scenario optima built with the SLOT recourse, priced at minutes -- "
            "the existing table, whose gap is neither EVPI nor a hedging cost. "
            "minute: scenario optima built under the SAME minute recourse used to "
            "evaluate them, which makes EVPI defined (B13)."
        ),
    )
    ap.add_argument("--slot", type=int, default=None, help="Slot width in minutes; default from the base config.")
    ap.add_argument("--Q", type=int, default=None, help="Fleet size; default from the base config (2).")
    ap.add_argument("--p-minutes", type=float, default=56.0)
    ap.add_argument("--policy", choices=("start", "midpoint", "end"), default="start")
    ap.add_argument(
        "--time-limit", type=float, default=None,
        help="Per-solve ceiling in seconds. Five solves run here, so the config's own "
             "1800 s is a 2.5 h worst case; cap it explicitly. A solve that stops on "
             "the clock makes its arm a bound, not an optimum, and the table says so.",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mobauto2_benders.minute_pricer import load_request_minutes, price_schedule_at_minutes
    from mobauto2_milp.config import load_config

    cfg = load_config(BASE_CONFIG)
    delta = int(args.slot or cfg.model.time.slot_resolution)
    seats = float(cfg.service.S)
    wmax = float(cfg.service.Wmax_minutes)
    p_minutes = float(args.p_minutes)

    names = list(SCENARIOS.keys())
    files = [SCENARIOS[n] for n in names]
    weight = 1.0 / float(len(names))
    requests = {n: load_request_minutes(f) for n, f in zip(names, files)}

    print(f"p_minutes={p_minutes:.0f}  slot={delta}min  Q={args.Q or cfg.model.fleet.Q}  policy={args.policy}")
    print(f"scenarios: {', '.join(names)} (weight {weight:.2f} each)")
    print()

    print("--- solving the hedged schedule (one schedule, all four scenarios) ---")
    # The hedged arm must be built under the SAME recourse as its comparators, or the
    # gap is again measuring the valuation rather than the hedging (B13).
    if args.comparator == "minute":
        _, hedged_result, hedged_schedule = _solve_minute(
            files, args.Q, p_minutes, delta, args.policy, time_limit=args.time_limit
        )
    else:
        _, hedged_result, hedged_schedule = _solve(
            files, args.Q, p_minutes, delta, time_limit=args.time_limit
        )
    print(f"hedged: {hedged_result.status.name}, "
          f"{len(hedged_schedule['OUT'])} OUT + {len(hedged_schedule['RET'])} RET departures")
    print()

    oracle_schedules: dict[str, dict[str, list[int]]] = {}
    for name, f in zip(names, files):
        print(f"--- solving the scenario-specific schedule for {name} alone "
              f"({args.comparator} recourse) ---")
        if args.comparator == "minute":
            _, oracle_result, oracle_schedule = _solve_minute(
                [f], args.Q, p_minutes, delta, args.policy,
                time_limit=args.time_limit,
            )
        else:
            _, oracle_result, oracle_schedule = _solve(
                [f], args.Q, p_minutes, delta, time_limit=args.time_limit
            )
        oracle_schedules[name] = oracle_schedule
        print(f"{name}: {oracle_result.status.name}, "
              f"{len(oracle_schedule['OUT'])} OUT + {len(oracle_schedule['RET'])} RET departures")
    print()

    comparator_name = (
        "minute-optimal" if args.comparator == "minute" else "slot-optimal"
    )
    print(
        f"One stochastic schedule against scenario-specific {comparator_name} schedules"
    )
    print(f"{'scenario':22s} | {'hedged cost':>11s} {comparator_name[:11]:>11s} "
          f"{'gap':>7s} | {'hedged unserv':>13s} {'cmp unserv':>13s}")
    print("-" * 92)
    hedged_total = oracle_total = 0.0
    for name in names:
        r = requests[name]
        hedged_priced = price_schedule_at_minutes(hedged_schedule, r, delta, seats, wmax, p_minutes, policy=args.policy)
        oracle_priced = price_schedule_at_minutes(oracle_schedules[name], r, delta, seats, wmax, p_minutes, policy=args.policy)
        gap = (
            100.0 * (hedged_priced.total_cost - oracle_priced.total_cost) / oracle_priced.total_cost
            if oracle_priced.total_cost
            else 0.0
        )
        hedged_total += weight * hedged_priced.total_cost
        oracle_total += weight * oracle_priced.total_cost
        print(f"{name:22s} | {hedged_priced.total_cost:11.0f} {oracle_priced.total_cost:11.0f} {gap:6.1f}% | "
              f"{hedged_priced.unserved_passengers:13.0f} {oracle_priced.unserved_passengers:13.0f}")
    print("-" * 92)
    hedging_cost = 100.0 * (hedged_total - oracle_total) / oracle_total if oracle_total else 0.0
    print(f"{'AVERAGED':22s} | {hedged_total:11.0f} {oracle_total:11.0f} {hedging_cost:6.1f}%")
    print()
    print("Read AVERAGED as the operational figure: what it costs, in passenger-minutes at")
    print(f"p_minutes={p_minutes:.0f}, to run ONE schedule across all four scenarios instead of")
    print("re-planning perfectly for whichever day actually happens. Per-scenario cells are")
    print("diagnostic only (docs/RESEARCH_NOTE_v2.md section 3, falsifier 3).")
    print()
    if args.comparator == "slot":
        print("B13: THIS GAP IS NOT EVPI AND NOT THE COST OF HEDGING.")
        print("The comparators are SLOT-optimal schedules priced at minute fidelity, so")
        print("part of the gap is their own slot-vs-minute mismatch rather than the price")
        print("of committing before the day is known. A scenario in which the HEDGED")
        print("schedule WINS is the proof: a schedule optimised for one day cannot lose on")
        print("that day to one optimised for four, unless it was not optimal under the")
        print("valuation being applied. Re-run with --comparator minute for the defined")
        print("quantity.")
    else:
        evpi = oracle_total - hedged_total
        print("B13: EVPI, defined. Both arms are optimal under the SAME minute recourse")
        print("that prices them, so the difference is the value of knowing the day in")
        print(f"advance: EVPI = {abs(evpi):.1f} passenger-minutes "
              f"({abs(hedging_cost):.1f}% of the perfect-information cost).")
        print("VSS is NOT computed here: it needs the mean-value solution as a third arm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
