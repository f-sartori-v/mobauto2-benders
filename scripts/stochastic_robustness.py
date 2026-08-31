"""Regenerate the stochastic-robustness result set at p_minutes=56, minute-level valuation.

    python scripts/stochastic_robustness.py [--Q 2] [--p-minutes 56] [--policy midpoint]

THE QUESTION (forward-work item A1, docs/FORWARD_PLAN_v1.md). The conference-era result
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


def _solve(scenario_files: list[str], Q: int | None, p_minutes: float, delta: int):
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

    solver = MonolithSolver(MobautoMilpModel, cfg, mp, sp)
    result = solver.run()
    if result.best_upper_bound is None:
        raise RuntimeError(
            f"no incumbent for scenario_files={scenario_files} -- check the solver "
            "backend is available (see docs/PROJECT_STATE_v6.md section 6.2)"
        )
    return solver, result, _schedule(solver.master.m)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slot", type=int, default=None, help="Slot width in minutes; default from the base config.")
    ap.add_argument("--Q", type=int, default=None, help="Fleet size; default from the base config (2).")
    ap.add_argument("--p-minutes", type=float, default=56.0)
    ap.add_argument("--policy", choices=("start", "midpoint", "end"), default="midpoint")
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
    _, hedged_result, hedged_schedule = _solve(files, args.Q, p_minutes, delta)
    print(f"hedged: {hedged_result.status.name}, "
          f"{len(hedged_schedule['OUT'])} OUT + {len(hedged_schedule['RET'])} RET departures")
    print()

    oracle_schedules: dict[str, dict[str, list[int]]] = {}
    for name, f in zip(names, files):
        print(f"--- solving the oracle schedule for {name} alone ---")
        _, oracle_result, oracle_schedule = _solve([f], args.Q, p_minutes, delta)
        oracle_schedules[name] = oracle_schedule
        print(f"{name}: {oracle_result.status.name}, "
              f"{len(oracle_schedule['OUT'])} OUT + {len(oracle_schedule['RET'])} RET departures")
    print()

    print(f"{'scenario':22s} | {'hedged cost':>11s} {'oracle cost':>11s} {'gap':>7s} | "
          f"{'hedged unserv':>13s} {'oracle unserv':>13s}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
