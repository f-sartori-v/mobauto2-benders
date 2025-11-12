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
        print("Initialized master problem.")

        # Optionally install lazy constraints callback if supported
        use_lazy = bool(self.cfg.master.params.get("use_lazy_cuts", False))
        if use_lazy and hasattr(self.master, "install_lazy_callback"):
            # Pass the subproblem instance so the callback can generate cuts
            getattr(self.master, "install_lazy_callback")(self.subproblem)
            # If install failed (no callback), fallback to non-lazy
            try:
                if not getattr(self.master, "lazy_installed", lambda: False)():
                    use_lazy = False
            except Exception:
                use_lazy = False

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

            print(f"\n=== Iteration {it} ===")
            print("Solving Master (MP)...")
            mres = self.master.solve()
            log.info(
                "iter=%d master status=%s obj=%s lb=%s",
                it,
                mres.status,
                f"{mres.objective:.6g}" if mres.objective is not None else None,
                f"{mres.lower_bound:.6g}" if mres.lower_bound is not None else None,
            )
            print(
                "MP result: status=%s obj=%s lb=%s"
                % (
                    mres.status,
                    (f"{mres.objective:.6g}" if mres.objective is not None else "-"),
                    (f"{mres.lower_bound:.6g}" if mres.lower_bound is not None else "-"),
                )
            )

            # Instrumentation: per-iteration summary
            try:
                cuts_in_model = getattr(self.master, "cuts_count", lambda: None)()
            except Exception:
                cuts_in_model = None
            last_const, last_nnz = (None, None)
            try:
                last_const, last_nnz = getattr(self.master, "last_cut_info", lambda: (None, None))()
            except Exception:
                pass
            if mres.objective is not None:
                print(
                    f"[MP] iter={it} cuts={cuts_in_model if cuts_in_model is not None else '-'} theta={mres.objective:.3f}"
                    + (f" last_const={last_const:.3f} last_nnz={last_nnz}" if last_const is not None else "")
                )

            # (Diagnostic only) We no longer assert binding; the MP is free to change y

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

            print("Evaluating Subproblem (SP) at candidate...")
            sres: SubproblemResult = self.subproblem.evaluate(mres.candidate)
            # Update UB if provided
            if sres.upper_bound is not None:
                best_ub = sres.upper_bound if best_ub is None else min(best_ub, sres.upper_bound)
            print(
                "SP result: ub=%s feasible=%s"
                % (
                    (f"{sres.upper_bound:.6g}" if sres.upper_bound is not None else "-"),
                    sres.is_feasible,
                )
            )
            # Add cut(s) if provided (optimality or feasibility) unless using lazy cuts
            added = 0
            cut_names: list[str] = []
            if not use_lazy:
                # Capture current number of cuts before adding
                try:
                    cuts_before = getattr(self.master, "cuts_count", lambda: 0)()
                except Exception:
                    cuts_before = 0

                # Force-accept the first violated cut at this incumbent
                forced_added = False
                if sres.cut is not None and hasattr(self.master, "add_cut_force"):
                    try:
                        forced_added = bool(getattr(self.master, "add_cut_force")(sres.cut))
                    except Exception:
                        forced_added = False
                    if forced_added:
                        cut_names.append(sres.cut.name)
                        log.info("force-added %s cut '%s'", sres.cut.cut_type, sres.cut.name)

                # If nothing forced yet, try forcing the first in sres.cuts
                if (not forced_added) and getattr(sres, "cuts", None) and hasattr(self.master, "add_cut_force"):
                    for c in sres.cuts or []:
                        try:
                            forced_added = bool(getattr(self.master, "add_cut_force")(c))
                        except Exception:
                            forced_added = False
                        if forced_added:
                            cut_names.append(c.name)
                            log.info("force-added %s cut '%s'", c.cut_type, c.name)
                            break

                # Add remaining cuts through the normal filtered path
                if sres.cut is not None and not forced_added:
                    self.master.add_cut(sres.cut)
                    cut_names.append(sres.cut.name)
                for c in getattr(sres, "cuts", []) or []:
                    # Avoid re-adding the same cut if it was force-added above
                    if forced_added and cut_names and c.name == cut_names[-1]:
                        continue
                    self.master.add_cut(c)
                    cut_names.append(c.name)

                # Compute how many were actually added
                try:
                    cuts_after = getattr(self.master, "cuts_count", lambda: 0)()
                except Exception:
                    cuts_after = cuts_before
                added = int(cuts_after) - int(cuts_before)
                if added > 0:
                    log.info("added %d cut(s)", added)
                    names_str = ", ".join(cut_names) if cut_names else "(unnamed)"
                    print(f"Master updated: added {added} cut(s): {names_str}")
                else:
                    if cut_names:
                        print("Master updated: no new cuts (all skipped / duplicates)")

            # Check gap if we have both bounds
            if best_lb is not None and best_ub is not None:
                gap = abs(best_ub - best_lb)
                rel_gap = gap / max(1.0, abs(best_ub))
                log.info("bounds: best_lb=%.6g best_ub=%.6g gap=%.6g rel=%.3g", best_lb, best_ub, gap, rel_gap)
                print(f"Bounds: LB={best_lb:.6g} UB={best_ub:.6g} gap={gap:.6g} rel={rel_gap:.3g}")
                if rel_gap <= tol:
                    log.info("Optimality reached within tolerance after %d iterations", it)
                    print(f"\nOptimality reached within tolerance after {it} iterations.")
                    # If the master has a formatter for the current solution, print it
                    try:
                        fmt = getattr(self.master, "format_solution", None)
                        if callable(fmt):
                            print("\nBest Master Solution:")
                            print(fmt())
                    except Exception:
                        pass
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
