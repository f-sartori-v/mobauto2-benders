"""F2's falsifier: does placement freedom improve cut strength at a fixed budget?

    python scripts/f2_placement_freedom_check.py

THE CLAIM UNDER TEST (docs/PROJECT_STATE_v6.md section 5, F2). A fixed offset grid
`O subset [0, delta]`, chosen once at load, lets the minute recourse treat a departure as
reachable at any minute in `O` rather than only at `departure_policy`'s single instant. It
is a RELAXATION (Q_relaxed <= Q_true), so a cut derived from it is a valid lower bound, and
the design's whole point is that it might be a TIGHTER one -- more of the arc set the true
minute-level problem has, without adding a single row to the master or changing the cut
interface.

THE FALSIFIER, STATED IN ADVANCE, before this ran: settled negatively if cut strength at a
fixed budget does not improve beyond run-to-run noise.

THE TWO CONFIGS. configs/f2/check_baseline.yaml and configs/f2/check_offsets.yaml are
identical except for exactly one key, subproblem.placement_offsets, so any LB difference
is attributable to that key alone. Both are ITERATION-budgeted (15 iterations), not
wall-clock-budgeted (BENDERS_SPEC_v4 section 0.10), with a generous total_time_limit_s so
the clock does not bind first -- which is what makes them bit-reproducible and therefore
makes "beyond run-to-run noise" a meaningful bar: an iteration-budgeted, non-clock-truncated
run has no run-to-run noise to attribute a small gap to.

WHAT THIS DOES NOT DO. It does not run either arm to convergence, and it does not compare
against the monolith -- that is Claim 2's question, not F2's. It also does not compare
UPPER bounds: F2 is explicitly not licensed to report one (a schedule priced under the
relaxed recourse is not its true cost), so only best_lower_bound is read here.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _run(config_path: str, label: str):
    from mobauto2_benders import app

    print(f"--- {label} ({config_path}) ---")
    result = app.run(config_path, {"emit_cli_output": False})
    truncated = result.clock_truncated_master_solves or 0
    if truncated:
        print(
            f"[WARN] {truncated} master solve(s) stopped on the clock -- this run is NOT "
            "bit-reproducible, and any LB difference against the other arm may be noise, "
            "not the mechanism."
        )
    print(f"status={result.status.name} LB={result.best_lower_bound} UB={result.best_upper_bound} "
          f"iterations={result.iterations} clock_truncated={truncated}")
    print()
    return result


def main() -> int:
    baseline = _run("configs/f2/check_baseline.yaml", "baseline (single offset, today's model)")
    f2 = _run("configs/f2/check_offsets.yaml", "F2 (placement freedom, offsets 0/15/30)")

    lb_base = baseline.best_lower_bound
    lb_f2 = f2.best_lower_bound
    print("=" * 60)
    print(f"LB baseline : {lb_base}")
    print(f"LB F2       : {lb_f2}")
    if lb_base is not None and lb_f2 is not None:
        gain_abs = lb_f2 - lb_base
        gain_pct = 100.0 * gain_abs / abs(lb_base) if lb_base else float("nan")
        print(f"gain        : {gain_abs:+.4f} ({gain_pct:+.3f}%)")
        both_reproducible = not (
            (baseline.clock_truncated_master_solves or 0)
            or (f2.clock_truncated_master_solves or 0)
        )
        print(f"both bit-reproducible (clock_truncated_master_solves == 0 on both): "
              f"{both_reproducible}")
        if not both_reproducible:
            print("VERDICT: inconclusive -- at least one arm was clock-truncated, so a gap "
                  "cannot be attributed to F2 rather than machine load.")
        elif abs(gain_pct) < 0.05:
            print("VERDICT: falsifier triggers -- LB does not move beyond noise-scale, and "
                  "there is no run-to-run noise in a reproducible, iteration-budgeted run to "
                  "attribute a near-zero gap to. F2 is settled negatively at this budget.")
        elif gain_pct < 0.0:
            print("VERDICT: falsifier triggers, more sharply than 'no improvement' -- LB is "
                  "reproducibly WORSE under F2, not merely flat. Settled negatively at this "
                  "budget: placement freedom is actively harmful here, not neutral.")
        else:
            print("VERDICT: LB moves UP by a reproducible, non-trivial amount. F2 is not "
                  "falsified at this budget; see whether it holds at other budgets/instances "
                  "before calling it settled positively.")
    else:
        print("gain        : unavailable (no lower bound on at least one arm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
