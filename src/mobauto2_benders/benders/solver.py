from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from ..config import BendersConfig
from .master import MasterProblem
from .subproblem import Subproblem
from .types import SolveStatus, SubproblemResult

log = logging.getLogger(__name__)


@dataclass(slots=True)
class BendersRunResult:
    status: SolveStatus
    iterations: int
    best_lower_bound: Optional[float]
    best_upper_bound: Optional[float]


class BendersSolver:
    def __init__(self, master: MasterProblem, subproblem: Subproblem, cfg: BendersConfig):
        self.master = master
        self.subproblem = subproblem
        self.cfg = cfg

    def run(self) -> BendersRunResult:
        t0 = time.time()
        max_it = self.cfg.run.max_iterations
        tol = self.cfg.run.tolerance
        self.master.initialize()

        best_lb: Optional[float] = None
        best_ub: Optional[float] = None

        for it in range(1, max_it + 1):
            if time.time() - t0 > self.cfg.run.time_limit_s:
                log.warning("Time limit reached after %d iterations", it - 1)
                return BendersRunResult(
                    status=SolveStatus.UNKNOWN,
                    iterations=it - 1,
                    best_lower_bound=best_lb,
                    best_upper_bound=best_ub,
                )

            mres = self.master.solve()
            log.info(
                "iter=%d master status=%s obj=%s lb=%s",
                it,
                mres.status,
                f"{mres.objective:.6g}" if mres.objective is not None else None,
                f"{mres.lower_bound:.6g}" if mres.lower_bound is not None else None,
            )

            if mres.status == SolveStatus.INFEASIBLE:
                log.error("Master problem infeasible")
                return BendersRunResult(
                    status=mres.status,
                    iterations=it,
                    best_lower_bound=best_lb,
                    best_upper_bound=best_ub,
                )

            # Update lower bound if provided
            if mres.lower_bound is not None:
                best_lb = mres.lower_bound if best_lb is None else max(best_lb, mres.lower_bound)

            if not mres.candidate:
                log.error("Master did not return a candidate solution")
                return BendersRunResult(
                    status=SolveStatus.UNKNOWN,
                    iterations=it,
                    best_lower_bound=best_lb,
                    best_upper_bound=best_ub,
                )

            sres: SubproblemResult = self.subproblem.evaluate(mres.candidate)
            # Update UB if provided
            if sres.upper_bound is not None:
                best_ub = sres.upper_bound if best_ub is None else min(best_ub, sres.upper_bound)
            # Add cut(s) if provided (optimality or feasibility)
            added = 0
            if sres.cut is not None:
                self.master.add_cut(sres.cut)
                added += 1
                log.info("added %s cut '%s' violation=%s", sres.cut.cut_type, sres.cut.name, sres.violation)
            for c in getattr(sres, "cuts", []) or []:
                self.master.add_cut(c)
                added += 1
            if added:
                log.info("added %d cut(s)", added)

            # Check gap if we have both bounds
            if best_lb is not None and best_ub is not None:
                gap = abs(best_ub - best_lb)
                rel_gap = gap / max(1.0, abs(best_ub))
                log.info("bounds: best_lb=%.6g best_ub=%.6g gap=%.6g rel=%.3g", best_lb, best_ub, gap, rel_gap)
                if rel_gap <= tol:
                    log.info("Optimality reached within tolerance after %d iterations", it)
                    return BendersRunResult(
                        status=SolveStatus.OPTIMAL,
                        iterations=it,
                        best_lower_bound=best_lb,
                        best_upper_bound=best_ub,
                    )

        log.warning("Max iterations reached: %d", max_it)
        return BendersRunResult(
            status=SolveStatus.UNKNOWN,
            iterations=max_it,
            best_lower_bound=best_lb,
            best_upper_bound=best_ub,
        )


__all__ = ["BendersSolver", "BendersRunResult"]
