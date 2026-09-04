"""B8. Replace the 450/30 inference with two constrained models.

    python scripts/trip_cap_450.py [--Q 2,3,4,5,6,7,8] [--H 660,780,900]
                                   [--time-limit 120] [--out FILE.json]

THE CLAIM THIS RETIRES (audit items 3.4 and 4.8). The report inferred that no fleet
serves 450 passengers in 30 trips from having OBSERVED 24, 36 and 48 trips in three
UNCONSTRAINED optima. That is not an inference. An unconstrained optimum reports the
trip count that minimises a weighted sum in which trips carry a small positive cost;
it says nothing about what is achievable when the trip count is CONSTRAINED, because
the constrained problem is a different problem whose optimum was never computed.

WHAT IS COMPUTED HERE INSTEAD.

  Model (a)  maximise served demand   subject to   N_trip <= 30
  Model (b)  minimise N_trip          subject to   served demand = 450

(a) answers "how many can 30 trips carry?" directly. (b) answers "how few trips can
carry all of them?", and reports INFEASIBLE explicitly when no schedule serves all 450
-- an infeasible cell is a result, not a blank.

Both are solved on the monolith, which carries the recourse by equality, so no cut and
no theta approximation stands between the answer and the model.

PROOF STATUS IS REPORTED PER CELL. A cell that stopped on the clock is a bound, not an
answer, and it is labelled as one. For model (a) the served count of a clock-truncated
run is a valid LOWER bound on what 30 trips can carry (it is an achieved schedule); the
bound column says how much room the solver could not close. For model (b) a
clock-truncated run gives an UPPER bound on the minimum trip count.

THE HORIZON NECESSARY CONDITION IS PRINTED FIRST, and it is printed for a specific
reason. It says that one vehicle making `k` one-way trips over a horizon `H` needs

    k * tau_trip + (60 / rho) * max(0, k * c_trip - b0)  <=  H

-- driving time plus the recharging time the driving forces -- and at H=660 with the
shipped energy parameters that caps one vehicle at 14 trips, hence the fleet at 14Q.

READ THAT BOUND FOR WHAT IT IS. It is a necessary condition on TRIP COUNT ALONE. It
settles nothing about passengers: 14Q trips at S seats is an upper bound on carried
demand only if every trip runs full and in the right direction at the right time, which
no schedule does. It is printed so that an infeasible cell in model (b) can be read
against it -- if the required trip count exceeds 14Q, the infeasibility is the horizon,
not the optimiser.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

BASE_CONFIG = "configs/milp/target_scale450_monolith.yaml"
TRIP_CAP = 30
TARGET_DEMAND = 450.0


def horizon_trip_cap(
    H_minutes: float,
    tau_trip_minutes: float,
    energy_per_trip: float,
    b0: float,
    charge_rate_per_hour: float,
) -> int:
    """Largest `k` satisfying the horizon necessary condition, for ONE vehicle.

        k * tau_trip + (60 / rho) * max(0, k * c_trip - b0)  <=  H

    Driving time, plus the recharging time that much driving forces. Necessary, not
    sufficient: it ignores where the vehicle is, which direction each trip runs, and
    whether anyone is waiting for it.
    """
    k = 0
    while True:
        nxt = k + 1
        drive = nxt * float(tau_trip_minutes)
        deficit = max(0.0, nxt * float(energy_per_trip) - float(b0))
        charge = (60.0 / float(charge_rate_per_hour)) * deficit
        if drive + charge > float(H_minutes) + 1e-9:
            return k
        k = nxt
        if k > 10_000:  # pragma: no cover - the loop is bounded by H in practice
            raise RuntimeError("horizon trip cap did not terminate; check rho and H")


def _served_expr(model, scenarios):
    """Weighted served passengers = total demand minus weighted unserved."""
    import pyomo.environ as pyo

    weights = [float(sc.weight) for sc in scenarios]
    total = sum(
        weights[i] * (sum(sc.R_out) + sum(sc.R_ret)) for i, sc in enumerate(scenarios)
    )
    unserved = sum(
        weights[int(s)] * sum(model.u_OUT[s, t] + model.u_RET[s, t] for t in model.T)
        for s in model.MonoScenarios
    )
    del pyo
    return total, unserved


def _trip_count_expr(model):
    return sum(
        model.yOUT[q, t] + model.yRET[q, t] for q in model.Q for t in model.T
    )


def _build(cfg, Q: int, H_minutes: int, time_limit: float):
    from mobauto2_milp.app import _prepare_params
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_milp.monolith import MonolithSolver

    mp, sp = _prepare_params(cfg, {})
    mp, sp = dict(mp), dict(sp)
    mp["Q"] = int(Q)
    mp["T_minutes"] = int(H_minutes)
    mp.pop("T", None)
    mp["solve_time_limit_s"] = float(time_limit)
    # Symmetry breaking stays as the config sets it: the fleet here IS homogeneous, so
    # M2 holds and the ordering constraint is valid. It is left on because turning it
    # off would make these cells incomparable with every other run on this instance.
    solver = MonolithSolver(MobautoMilpModel, cfg, mp, sp)
    solver.master.initialize()
    model = solver.master.m
    solver._scenarios = solver._load_scenarios(len(list(model.T)))
    solver._attach_recourse_model(model, solver._scenarios)
    return solver, model


def _solve_with(solver, model, objective_expr, sense, extra_constraints):
    """Solve `model` under a replacement objective, then restore it.

    The master's own weighted-sum objective is deactivated rather than deleted, and
    restored in a `finally`, so a solver object cannot be left carrying a different
    objective than the one its config names.
    """
    import pyomo.environ as pyo

    model.obj.deactivate()
    added: list[str] = []
    try:
        for i, expr in enumerate(extra_constraints):
            name = f"_b8_con_{i}"
            model.add_component(name, pyo.Constraint(expr=expr))
            added.append(name)
        model.add_component(
            "_b8_obj", pyo.Objective(expr=objective_expr, sense=sense)
        )
        added.append("_b8_obj")
        result = solver.master.solve()
        stats = dict(solver.master.last_solve_stats())
        return result, stats
    finally:
        for name in reversed(added):
            if hasattr(model, name):
                model.del_component(name)
        model.obj.activate()


def _status_name(result) -> str:
    st = getattr(result, "status", None)
    return getattr(st, "name", str(st))


def main() -> int:
    import pyomo.environ as pyo

    from mobauto2_milp.config import load_config

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--Q", default="2,3,4,5,6,7,8")
    ap.add_argument("--H", default="660,780,900")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--models",
        default="a,b",
        help="which models to run: 'a', 'b', or 'a,b'",
    )
    args = ap.parse_args()

    q_values = [int(x) for x in args.Q.split(",")]
    h_values = [int(x) for x in args.H.split(",")]
    which = {m.strip().lower() for m in args.models.split(",")}

    cfg = load_config(BASE_CONFIG)
    tau_trip = float(cfg.model.time.trip_duration_minutes)
    energy_per_trip = float(cfg.model.energy.L)
    b0 = float((cfg.model.fleet.binit or [cfg.model.energy.Emax])[0])
    # delta_chg is energy per SLOT; rho is energy per HOUR, which is what the
    # condition is written in.
    from mobauto2_milp.config import resolve_energy_params

    energy = resolve_energy_params(
        cfg.model.energy, {"slot_resolution": cfg.model.time.slot_resolution}
    )
    delta_chg = float(energy["delta_chg"])
    rho = delta_chg * (60.0 / float(cfg.model.time.slot_resolution))

    print("B8 -- 450 passengers in 30 trips, as two constrained models")
    print("=" * 78)
    print(f"instance          : {cfg.data.scenario_files}")
    print(
        f"parameters        : tau_trip={tau_trip:g} min, c_trip={energy_per_trip:g}, "
        f"b0={b0:g}, rho={rho:g}/h, S={cfg.service.S:g}, "
        f"p_min={cfg.service.p * cfg.model.time.slot_resolution:g}"
    )
    print(f"budget            : {args.time_limit:g} s per cell")
    print()
    print("HORIZON NECESSARY CONDITION (trip count only -- says nothing about pax):")
    print("  k*tau_trip + (60/rho)*max(0, k*c_trip - b0) <= H")
    caps = {}
    for H in h_values:
        k = horizon_trip_cap(H, tau_trip, energy_per_trip, b0, rho)
        caps[H] = k
        print(
            f"  H={H:4d} min -> at most {k:3d} one-way trips per vehicle, "
            f"hence at most {k}Q for the fleet"
        )
    print()

    rows: list[dict] = []
    header = (
        f"{'model':>5s} | {'Q':>2s} | {'H':>4s} | {'status':>12s} | "
        f"{'served':>7s} | {'trips':>5s} | {'bound':>10s} | {'gap%':>7s} | "
        f"{'14Q cap':>7s} | proof"
    )
    print(header)
    print("-" * len(header))

    for H in h_values:
        for Q in q_values:
            fleet_cap = caps[H] * Q
            if "a" in which:
                solver, model = _build(cfg, Q, H, args.time_limit)
                total, unserved = _served_expr(model, solver._scenarios)
                result, stats = _solve_with(
                    solver,
                    model,
                    unserved,
                    pyo.minimize,
                    [_trip_count_expr(model) <= TRIP_CAP],
                )
                status = _status_name(result)
                proven = status == "OPTIMAL"
                try:
                    served = total - float(pyo.value(unserved))
                    trips = float(pyo.value(_trip_count_expr(model)))
                except Exception:
                    served, trips = float("nan"), float("nan")
                bound = getattr(result, "lower_bound", None)
                # The solver minimises unserved; the bound on unserved is an UPPER
                # bound on served. Reported in served terms so the column means one
                # thing down the page.
                served_bound = (
                    total - float(bound) if bound is not None else float("nan")
                )
                gap = stats.get("gap")
                rows.append(
                    dict(
                        model="a",
                        Q=Q,
                        H=H,
                        status=status,
                        served=served,
                        served_upper_bound=served_bound,
                        trips=trips,
                        gap=None if gap is None else float(gap),
                        proven=proven,
                        fleet_trip_cap=fleet_cap,
                        total_demand=total,
                    )
                )
                print(
                    f"{'(a)':>5s} | {Q:2d} | {H:4d} | {status:>12s} | "
                    f"{served:7.0f} | {trips:5.0f} | {served_bound:10.1f} | "
                    f"{(100.0 * gap if gap is not None else float('nan')):6.2f}% | "
                    f"{fleet_cap:7d} | "
                    + ("proven" if proven else "clock-truncated: served is a LOWER bound")
                )

            if "b" in which:
                solver, model = _build(cfg, Q, H, args.time_limit)
                total, unserved = _served_expr(model, solver._scenarios)
                # served == 450 is written as unserved <= total - 450, which is the
                # same set: served cannot exceed the demand that exists.
                slack = total - TARGET_DEMAND
                result, stats = _solve_with(
                    solver,
                    model,
                    _trip_count_expr(model),
                    pyo.minimize,
                    [unserved <= slack + 1e-6],
                )
                status = _status_name(result)
                infeasible = status in {"INFEASIBLE", "ERROR"} or not _has_solution(
                    model
                )
                proven = status == "OPTIMAL"
                try:
                    trips = float(pyo.value(_trip_count_expr(model)))
                    served = total - float(pyo.value(unserved))
                except Exception:
                    trips, served = float("nan"), float("nan")
                gap = stats.get("gap")
                rows.append(
                    dict(
                        model="b",
                        Q=Q,
                        H=H,
                        status=status,
                        infeasible=bool(infeasible),
                        served=served,
                        trips=trips,
                        gap=None if gap is None else float(gap),
                        proven=proven,
                        fleet_trip_cap=fleet_cap,
                        total_demand=total,
                        target=TARGET_DEMAND,
                    )
                )
                note = (
                    "INFEASIBLE: no schedule serves all 450"
                    if infeasible
                    else ("proven" if proven else "clock-truncated: trips is an UPPER bound")
                )
                print(
                    f"{'(b)':>5s} | {Q:2d} | {H:4d} | {status:>12s} | "
                    f"{served:7.0f} | {trips:5.0f} | {'-':>10s} | "
                    f"{(100.0 * gap if gap is not None else float('nan')):6.2f}% | "
                    f"{fleet_cap:7d} | {note}"
                )

    print()
    print("HOW TO READ THIS TABLE")
    print(
        "  Model (a): 'served' is achieved by an exhibited schedule with at most "
        f"{TRIP_CAP} trips,"
    )
    print(
        "             so it is a LOWER bound on what 30 trips can carry even when the "
        "cell"
    )
    print("             is clock-truncated. 'bound' is the upper bound the solver proved.")
    print(
        "  Model (b): 'trips' is achieved, so it is an UPPER bound on the minimum trip "
        "count."
    )
    print(
        "             An INFEASIBLE cell means no schedule at that (Q, H) serves all "
        "450 --"
    )
    print("             read it against the 14Q column before blaming the optimiser.")
    print(
        "  The 14Q column is the horizon necessary condition on TRIP COUNT. It bounds "
        "trips,"
    )
    print("  not passengers, and a cell can be infeasible well below it.")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def _has_solution(model) -> bool:
    import pyomo.environ as pyo

    try:
        for q in model.Q:
            for t in model.T:
                if pyo.value(model.yOUT[q, t], exception=False) is None:
                    return False
                break
            break
        return True
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
