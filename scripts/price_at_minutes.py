"""Price a monolith-optimal schedule at minute fidelity and report waiting honestly.

    python scripts/price_at_minutes.py [--config configs/milp/baseline_d9_monolith.yaml]

Solves the instance with the monolithic MILP, then takes the schedule it proved optimal
and prices it against the demand's ACTUAL arrival minutes. Reports the waiting term
separately from the objective, because the objective cannot see it: on `baseline_d9`
waiting is 6.8% of the objective and the unmet-demand penalty is 93.2%, so an error that
nearly doubles the waiting figure moves the total by ~1%.

This is a measuring instrument. It changes no model and optimises nothing -- it re-prices
a fixed schedule, so any difference it reports is the cost of the slot abstraction and
nothing else.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/milp/baseline_d9_monolith.yaml")
    ap.add_argument(
        "--demand",
        default=None,
        help="Demand file with per-request minutes. Defaults to the config's first "
        "scenario file.",
    )
    args = ap.parse_args()

    import pyomo.environ as pyo

    from mobauto2_milp.config import load_config
    from mobauto2_milp.app import _prepare_params
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_milp.monolith import MonolithSolver
    from mobauto2_benders.minute_pricer import honest_waiting, load_request_minutes

    cfg = load_config(args.config)
    mp, sp = _prepare_params(cfg, {})
    solver = MonolithSolver(MobautoMilpModel, cfg, mp, sp)
    result = solver.run()
    model = solver.master.m
    diag = solver._last_diagnostics

    demand_path = args.demand or list(cfg.data.scenario_files)[0]
    if len(list(cfg.data.scenario_files)) > 1 and args.demand is None:
        print(
            f"[warn] config has {len(list(cfg.data.scenario_files))} scenarios; pricing "
            f"only the first ({demand_path}). Minute-level pricing of a multi-scenario "
            "recourse is not the same quantity and is not attempted here."
        )

    delta = int(cfg.model.time.slot_resolution)
    departures: dict[str, list[int]] = {"OUT": [], "RET": []}
    for tau in model.T:
        for q in model.Q:
            if float(pyo.value(model.yOUT[q, tau], exception=False) or 0.0) > 0.5:
                departures["OUT"].append(int(tau))
            if float(pyo.value(model.yRET[q, tau], exception=False) or 0.0) > 0.5:
                departures["RET"].append(int(tau))

    report = honest_waiting(
        departures=departures,
        requests=load_request_minutes(demand_path),
        slot_resolution=delta,
        seats=float(cfg.service.S),
        wmax_minutes=float(cfg.service.Wmax_minutes),
        p_minutes=float(cfg.service.p) * delta,
        slot_waiting_cost_slots=float(diag["sp_wait_cost_slots"]),
        slot_unserved=float(diag["sp_penalty_pax"]),
        served_slot=float(diag["pax_served"]),
    )

    print()
    print(f"instance : {args.config}")
    print(f"demand   : {demand_path}")
    print(
        f"schedule : {len(departures['OUT'])} OUT + {len(departures['RET'])} RET "
        f"departures, objective {result.best_upper_bound:.2f} ({result.status.name})"
    )
    print()
    print(report.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
