"""RQ5: is decomposed minute-level Benders faster than the minute-level MONOLITH?

    python scripts/rq5_minute_vs_monolith.py

THE POINT OF THIS SCRIPT IS THE BASELINE. Every speed comparison this project has run so
far used a model that is not the one being solved:

  D52  decomposition vs a SLOT recourse   -- different model
  D54  decomposition vs the SLOT monolith -- different model, and its optimum is an
                                             UPPER bound on this one (note v2, section 3)

Here both arms solve THE SAME model -- slot first stage, minute recourse, `y` entering the
recourse only through the capacity right-hand side (E2) -- so the only thing that differs
is the algorithm:

  M  monolith  first stage + minute recourse in one MIP, theta pinned by EQUALITY.
               No cut, no theta approximation, so the whole D30 defect class is out of
               reach. Solved to proven optimality.
  B  Benders   the same model decomposed: slot master, minute subproblem, optimality cuts
               built from duals the minute LP returns one-per-departure-slot.

Both single-threaded, both in the policy regime p_minutes = 56 (D53), both `midpoint`.

WHAT AN HONEST ANSWER LOOKS LIKE. If B does not close its gap inside the budget, the
answer is "no, and here is how far it got" -- not "inconclusive". A decomposition that
cannot match a monolith on an instance this small has been told something, and reporting
the LB it reached against M's proven optimum is the measurement.

The two arms are built from two different packages (`mobauto2_milp` and
`mobauto2_benders`) whose first stages are hand-synced copies. If they have drifted, this
comparison is confounded rather than wrong-looking, so the instance parameters are checked
against each other before anything is solved and the run aborts on a mismatch.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

MONOLITH_CONFIG = "configs/milp/baseline_d9_p56_monolith.yaml"
BENDERS_CONFIG = "configs/phase1/rq5_benders_minute_p56.yaml"

# Independent expectation, recorded before this script existed: the monolithic minute
# answer on this instance at p_minutes = 1500 is 3822.97 slot-equivalent units, i.e.
# 114 682 passenger-minutes (D51, and the header of configs/phase1/d9_recourse_minute.yaml).
# Checking against it is what makes arm M the same instrument D51 measured rather than a
# new one that happens to run.
D51_TARGET_P1500 = 3822.97
D51_TARGET_TOL = 0.05


def _instance_fingerprint_milp(cfg) -> dict[str, float]:
    return {
        "T_minutes": float(cfg.model.time.T_minutes),
        "slot_resolution": float(cfg.model.time.slot_resolution),
        "trip_duration_minutes": float(cfg.model.time.trip_duration_minutes),
        "Q": float(cfg.model.fleet.Q),
        "Emax": float(cfg.model.energy.Emax),
        "L": float(cfg.model.energy.L),
        "S": float(cfg.service.S),
        "Wmax_minutes": float(cfg.service.Wmax_minutes),
    }


def _instance_fingerprint_benders(cfg) -> dict[str, float]:
    return {
        "T_minutes": float(cfg.model.time.T_minutes),
        "slot_resolution": float(cfg.model.time.slot_resolution),
        "trip_duration_minutes": float(cfg.model.time.trip_duration_minutes),
        "Q": float(cfg.model.fleet.Q),
        "Emax": float(cfg.model.energy.Emax),
        "L": float(cfg.model.energy.L),
        "S": float(cfg.subproblem.S),
        "Wmax_minutes": float(cfg.subproblem.Wmax_minutes),
    }


def _solve_minute_monolith(cfg, p_minutes: float, policy: str, time_limit: float):
    """Arm M. Returns (objective, status_name, seconds, terminated_on_clock)."""
    from mobauto2_milp.app import _prepare_params
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_benders.minute_pricer import attach_minute_recourse, load_request_minutes

    mp, _sp = _prepare_params(cfg, {})
    mp = dict(mp)
    mp["solve_time_limit_s"] = float(time_limit)

    delta = int(cfg.model.time.slot_resolution)
    requests = load_request_minutes(list(cfg.data.scenario_files)[0])

    master = MobautoMilpModel(mp)
    master.initialize()
    attach_minute_recourse(
        master.m,
        requests,
        delta,
        float(cfg.service.S),
        float(cfg.service.Wmax_minutes),
        float(p_minutes),
        policy=policy,
        # Slot-equivalent units, so the objective is directly comparable with the
        # Benders arm and the first-stage terms keep their relative weight (note v2 s5).
        objective_scale=1.0 / float(delta),
    )

    t0 = time.perf_counter()
    res = master.solve()
    seconds = time.perf_counter() - t0

    status = res.status.name
    on_clock = seconds >= float(time_limit) * 0.99
    return float(res.objective), status, seconds, on_clock


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", choices=("start", "midpoint", "end"), default="midpoint")
    ap.add_argument("--p-minutes", type=float, default=56.0)
    ap.add_argument(
        "--monolith-time-limit", type=float, default=300.0,
        help="Safety cap on arm M. It should terminate on the gap well inside this; "
             "if it stops on the clock the comparison is against a truncated optimum "
             "and says so.",
    )
    ap.add_argument(
        "--skip-verify", action="store_true",
        help="Skip the D51 instrument check (one extra monolith solve at p_minutes=1500).",
    )
    args = ap.parse_args()

    from mobauto2_milp.config import load_config as load_milp_config
    from mobauto2_benders.config import load_config as load_benders_config

    mono_cfg = load_milp_config(MONOLITH_CONFIG)
    bend_cfg = load_benders_config(BENDERS_CONFIG)

    fp_m = _instance_fingerprint_milp(mono_cfg)
    fp_b = _instance_fingerprint_benders(bend_cfg)
    drift = {k: (fp_m[k], fp_b[k]) for k in fp_m if abs(fp_m[k] - fp_b[k]) > 1e-9}
    if drift:
        print("ABORT: the two arms do not describe the same instance.")
        for k, (a, b) in sorted(drift.items()):
            print(f"  {k}: monolith {a}  benders {b}")
        print("A speed comparison between two different instances measures nothing.")
        return 2

    delta = int(fp_m["slot_resolution"])
    print("=" * 78)
    print("RQ5  decomposed minute Benders  vs  minute monolith")
    print(f"     same model, same instance, single-threaded, policy={args.policy}, "
          f"p_minutes={args.p_minutes:.0f}, slot={delta}min")
    print("=" * 78)

    # --- instrument check: does arm M reproduce the recorded D51 answer? ---
    if not args.skip_verify:
        obj, status, secs, _clock = _solve_minute_monolith(
            mono_cfg, 1500.0, "midpoint", args.monolith_time_limit
        )
        ok = abs(obj - D51_TARGET_P1500) <= D51_TARGET_TOL
        print(f"instrument check  p_minutes=1500 -> {obj:10.2f}  "
              f"(D51 recorded {D51_TARGET_P1500:.2f})  {'OK' if ok else 'MISMATCH'}  "
              f"[{status}, {secs:.1f}s]")
        if not ok:
            print("ABORT: arm M is not the instrument D51 measured. Fix that before")
            print("       reading any timing below as a comparison with the record.")
            return 3
        print("-" * 78)

    # --- arm M ---
    m_obj, m_status, m_secs, m_on_clock = _solve_minute_monolith(
        mono_cfg, args.p_minutes, args.policy, args.monolith_time_limit
    )
    print(f"M  monolith   objective {m_obj:10.2f}   {m_status:12s}  {m_secs:8.1f} s")
    if m_on_clock:
        print("   WARNING: arm M stopped on the clock. Its objective is not a proven")
        print("            optimum and the comparison below is against a truncation.")

    # --- arm B ---
    from mobauto2_benders import app as benders_app

    t0 = time.perf_counter()
    res = benders_app.run(BENDERS_CONFIG, {})
    b_secs = time.perf_counter() - t0

    lb = res.best_lower_bound
    ub = res.best_upper_bound
    print(f"B  Benders    LB {lb if lb is None else f'{lb:.2f}':>10}   "
          f"UB {ub if ub is None else f'{ub:.2f}':>10}   "
          f"{res.status.name:12s}  {b_secs:8.1f} s   ({res.iterations} iterations)")

    clocked = getattr(res, "master_time_limited_solves", None)
    if clocked:
        print(f"   {clocked} master solve(s) stopped on the clock -- this run is not "
              f"bit-reproducible (D26).")
    if getattr(res, "cut_valid_lower_bound", None) is False:
        print("   WARNING: the cut generator does not support a valid lower bound, so "
              "LB above is not one.")

    print("-" * 78)
    if lb is not None and m_obj:
        print(f"B reached {100.0 * float(lb) / float(m_obj):.1f}% of M's objective "
              f"as a lower bound, in {b_secs / max(m_secs, 1e-9):.1f}x M's wall time.")
    print("=" * 78)
    print("Read this as: same model, two algorithms. Whichever way it comes out, it is")
    print("the first comparison in this repository entitled to say anything about the")
    print("speed of a minute-level method.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
