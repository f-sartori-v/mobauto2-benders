#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import pyomo.environ as pyo

# Ensure `src/` is on sys.path for direct script execution
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobauto2_benders.params import Params
from mobauto2_benders.master_pyomo import build_master, solve_mip, add_benders_cut
from mobauto2_benders.subproblem_lp import solve_subproblem


def benders_loop(P: Params) -> Dict[str, float]:
    m = build_master(P)
    best_lb, best_ub = float("inf"), float("inf")

    for it in range(1, P.max_iters + 1):
        # Solve master
        solve_mip(m, P.mip_solver, executable=P.mip_solver_executable)
        theta_val = float(pyo.value(m.theta))
        lb = theta_val
        best_lb = max(best_lb, lb) if best_lb != float("inf") else lb

        # Extract y decisions
        y_out = {(q, t): float(pyo.value(m.yOUT[q, t])) for q in range(P.Q) for t in range(P.T)}
        y_ret = {(q, t): float(pyo.value(m.yRET[q, t])) for q in range(P.Q) for t in range(P.T)}

        # Build capacities C_{tau}
        C_out = {tau: P.S * sum(y_out[q, tau] for q in range(P.Q)) for tau in range(P.T)}
        C_ret = {tau: P.S * sum(y_ret[q, tau] for q in range(P.Q)) for tau in range(P.T)}

        # Solve subproblem for the single scenario
        duals, ub_val = solve_subproblem(P, [C_out[tau] for tau in range(P.T)], [C_ret[tau] for tau in range(P.T)], P.R_out, P.R_ret)

        # Update UB
        best_ub = min(best_ub, ub_val) if best_ub != float("inf") else ub_val

        # Build Benders cut
        const = sum(duals['alpha_OUT'][t] * P.R_out[t] for t in range(P.T)) + \
                sum(duals['alpha_RET'][t] * P.R_ret[t] for t in range(P.T))
        coeff_out = {(q, tau): P.S * duals['pi_OUT'][tau] for q in range(P.Q) for tau in range(P.T)}
        coeff_ret = {(q, tau): P.S * duals['pi_RET'][tau] for q in range(P.Q) for tau in range(P.T)}
        add_benders_cut(m, const, coeff_out, coeff_ret)

        # Gap check
        if best_lb < float("inf") and best_ub < float("inf"):
            gap = abs(best_ub - best_lb)
            rel_gap = gap / max(1.0, abs(best_ub))
            print(f"it={it} lb={best_lb:.6g} ub={best_ub:.6g} gap={gap:.6g} rel={rel_gap:.3g}")
            if rel_gap <= P.tolerance:
                break
        else:
            print(f"it={it} lb={best_lb} ub={best_ub}")

    return {"iterations": it, "best_lb": best_lb, "best_ub": best_ub}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/small_demo.yaml"))
    args = p.parse_args(argv)

    P = Params.load(args.config)
    res = benders_loop(P)
    print(f"status=done iterations={res['iterations']} best_lb={res['best_lb']:.6g} best_ub={res['best_ub']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
