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
        # Stall detection on gap improvement
        stall_max = int(getattr(self.cfg.run, "stall_max_no_improve_iters", 0) or 0)
        stall_min_abs = float(getattr(self.cfg.run, "stall_min_abs_improve", 0.0) or 0.0)
        stall_min_rel = float(getattr(self.cfg.run, "stall_min_rel_improve", 0.0) or 0.0)
        stall_ctr = 0
        prev_gap: Optional[float] = None

        last_diag: dict | None = None
        for it in range(1, max_it + 1):
            if time.time() - t0 > self.cfg.run.time_limit_s:
                log.warning("Time limit reached after %d iterations", it - 1)
                # Print the best incumbent information we have so far
                try:
                    elapsed = time.time() - t0
                    print(f"\nTime limit reached after {it - 1} iterations.")
                    print(f"Total solve time: {elapsed:.3f} seconds")
                    if best_lb is not None and best_ub is not None:
                        gap = abs(best_ub - best_lb)
                        rel_gap = gap / max(1.0, abs(best_ub))
                        print(
                            f"Best bounds: LB={best_lb:.6g} UB={best_ub:.6g} gap={gap:.6g} rel={rel_gap:.3g}"
                        )
                    # If the master has a formatter for the current solution, print it
                    try:
                        fmt = getattr(self.master, "format_solution", None)
                        if callable(fmt):
                            print("\nBest Master Solution:")
                            print(fmt())
                    except Exception:
                        pass
                    # Optional diagnostics from last evaluated subproblem (if available)
                    try:
                        if last_diag:
                            T = int(last_diag.get("T")) if "T" in last_diag else None
                            R_out = last_diag.get("R_out")
                            R_ret = last_diag.get("R_ret")
                            pax_out = last_diag.get("pax_out_by_tau")
                            pax_ret = last_diag.get("pax_ret_by_tau")
                            if isinstance(pax_out, list) and isinstance(pax_ret, list):
                                n = len(pax_out)
                                if T is None:
                                    T = n
                                header = "       " + " ".join(f"{t:>3d}" for t in range(T))
                                def fmt_row_floats(vals: list[float]) -> str:
                                    return " ".join(f"{float(v):>3.0f}" for v in (vals + [0.0] * T)[:T])
                                if isinstance(R_out, list) and isinstance(R_ret, list):
                                    print("\nDemand per slot (OUT/RET):")
                                    print(header)
                                    print(f"  OUT: {fmt_row_floats(R_out)}")
                                    print(f"  RET: {fmt_row_floats(R_ret)}")
                                try:
                                    m = getattr(self.master, "m", None)
                                    if m is not None:
                                        Q = list(m.Q)
                                        served_qt = [[0.0 for _ in range(T)] for _ in Q]
                                        for tau in range(T):
                                            out_qs = [q for q in Q if float(m.yOUT[q, tau].value or 0.0) >= 0.5]
                                            ret_qs = [q for q in Q if float(m.yRET[q, tau].value or 0.0) >= 0.5]
                                            k_out = len(out_qs)
                                            k_ret = len(ret_qs)
                                            share_out = (float(pax_out[tau]) / k_out) if k_out > 0 else 0.0
                                            share_ret = (float(pax_ret[tau]) / k_ret) if k_ret > 0 else 0.0
                                            for q in out_qs:
                                                served_qt[q][tau] += share_out
                                            for q in ret_qs:
                                                served_qt[q][tau] += share_ret
                                        print("\nPax per shuttle and slot (total):")
                                        print(header)
                                        for q in Q:
                                            print(f"  q={q}: {fmt_row_floats(served_qt[q])}")
                                except Exception:
                                    pass
                    except Exception:
                        pass
                except Exception:
                    pass
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
                    f"[MP] iter={it} cuts={cuts_in_model if cuts_in_model is not None else '-'} obj={mres.objective:.3f}"
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
            # Update UB if provided (include first-stage cost from master to compare totals)
            if sres.upper_bound is not None:
                try:
                    fcost_fn = getattr(self.master, "first_stage_cost", None)
                    fcost = float(fcost_fn(mres.candidate)) if callable(fcost_fn) else 0.0
                except Exception:
                    fcost = 0.0
                total_ub = float(sres.upper_bound) + float(fcost)
                best_ub = total_ub if best_ub is None else min(best_ub, total_ub)
            # Keep last diagnostics for end-of-run reporting
            try:
                last_diag = dict(getattr(sres, "diagnostics", {}) or {})
            except Exception:
                last_diag = None
            try:
                _ub_print = f"{sres.upper_bound:.6g}" if sres.upper_bound is not None else "-"
            except Exception:
                _ub_print = "-"
            print(f"SP result: ub={_ub_print} feasible={sres.is_feasible}")
            # Suppress repetitive demand printouts; diagnostics still available at the end
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
                    elapsed = time.time() - t0
                    print(f"\nOptimality reached within tolerance after {it} iterations.")
                    print(f"Total solve time: {elapsed:.3f} seconds")
                    # If the master has a formatter for the current solution, print it
                    try:
                        fmt = getattr(self.master, "format_solution", None)
                        if callable(fmt):
                            print("\nBest Master Solution:")
                            print(fmt())
                    except Exception:
                        pass
                    # Additional matrices from subproblem diagnostics (if available)
                    try:
                        if last_diag:
                            T = int(last_diag.get("T")) if "T" in last_diag else None
                            R_out = last_diag.get("R_out")
                            R_ret = last_diag.get("R_ret")
                            pax_out = last_diag.get("pax_out_by_tau")
                            pax_ret = last_diag.get("pax_ret_by_tau")
                            if isinstance(pax_out, list) and isinstance(pax_ret, list):
                                n = len(pax_out)
                                if T is None:
                                    T = n
                                # Header aligned as in master formatter
                                header = "       " + " ".join(f"{t:>3d}" for t in range(T))
                                def fmt_row_floats(vals: list[float]) -> str:
                                    return " ".join(f"{float(v):>3.0f}" for v in (vals + [0.0] * T)[:T])
                                # Demand by direction
                                if isinstance(R_out, list) and isinstance(R_ret, list):
                                    print("\nDemand per slot (OUT/RET):")
                                    print(header)
                                    print(f"  OUT: {fmt_row_floats(R_out)}")
                                    print(f"  RET: {fmt_row_floats(R_ret)}")
                                # Pax per shuttle and slot (total), evenly allocate across starting shuttles at that slot
                                try:
                                    m = getattr(self.master, "m", None)
                                    if m is not None:
                                        Q = list(m.Q)
                                        served_qt = [[0.0 for _ in range(T)] for _ in Q]
                                        for tau in range(T):
                                            out_qs = [q for q in Q if float(m.yOUT[q, tau].value or 0.0) >= 0.5]
                                            ret_qs = [q for q in Q if float(m.yRET[q, tau].value or 0.0) >= 0.5]
                                            k_out = len(out_qs)
                                            k_ret = len(ret_qs)
                                            share_out = (float(pax_out[tau]) / k_out) if k_out > 0 else 0.0
                                            share_ret = (float(pax_ret[tau]) / k_ret) if k_ret > 0 else 0.0
                                            for q in out_qs:
                                                served_qt[q][tau] += share_out
                                            for q in ret_qs:
                                                served_qt[q][tau] += share_ret
                                        print("\nPax per shuttle and slot (total):")
                                        print(header)
                                        for q in Q:
                                            print(f"  q={q}: {fmt_row_floats(served_qt[q])}")
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    return BendersRunResult(
                        status=SolveStatus.OPTIMAL,
                        iterations=it,
                        best_lower_bound=best_lb,
                        best_upper_bound=best_ub,
                    )
                # Stall stopping: if gap does not improve sufficiently for several iterations
                if stall_max > 0:
                    improved = False
                    if prev_gap is None:
                        improved = True
                    else:
                        abs_impr = float(prev_gap - gap)
                        rel_impr = abs_impr / max(1.0, abs(prev_gap))
                        improved = (abs_impr >= max(0.0, stall_min_abs)) or (rel_impr >= max(0.0, stall_min_rel))
                    if improved:
                        stall_ctr = 0
                    else:
                        stall_ctr += 1
                    prev_gap = gap
                if stall_ctr >= stall_max:
                    log.info(
                        "Stopping due to stall: no gap improvement for %d iterations (gap=%.6g)",
                        stall_ctr,
                        gap,
                    )
                    print(
                        f"\nStopped early after {it} iterations due to stall: no gap improvement for {stall_ctr} iterations."
                    )
                    # Print best-known incumbent details and formatted solution
                    try:
                        elapsed = time.time() - t0
                        print(f"Total solve time: {elapsed:.3f} seconds")
                        if best_lb is not None and best_ub is not None:
                            _gap = abs(best_ub - best_lb)
                            _rel = _gap / max(1.0, abs(best_ub))
                            print(
                                f"Best bounds: LB={best_lb:.6g} UB={best_ub:.6g} gap={_gap:.6g} rel={_rel:.3g}"
                            )
                        fmt = getattr(self.master, "format_solution", None)
                        if callable(fmt):
                            print("\nBest Master Solution:")
                            print(fmt())
                        # Optional diagnostics from last evaluated subproblem (if available)
                        try:
                            if last_diag:
                                T = int(last_diag.get("T")) if "T" in last_diag else None
                                R_out = last_diag.get("R_out")
                                R_ret = last_diag.get("R_ret")
                                pax_out = last_diag.get("pax_out_by_tau")
                                pax_ret = last_diag.get("pax_ret_by_tau")
                                if isinstance(pax_out, list) and isinstance(pax_ret, list):
                                    n = len(pax_out)
                                    if T is None:
                                        T = n
                                    header = "       " + " ".join(f"{t:>3d}" for t in range(T))
                                    def fmt_row_floats(vals: list[float]) -> str:
                                        return " ".join(f"{float(v):>3.0f}" for v in (vals + [0.0] * T)[:T])
                                    if isinstance(R_out, list) and isinstance(R_ret, list):
                                        print("\nDemand per slot (OUT/RET):")
                                        print(header)
                                        print(f"  OUT: {fmt_row_floats(R_out)}")
                                        print(f"  RET: {fmt_row_floats(R_ret)}")
                                    try:
                                        m = getattr(self.master, "m", None)
                                        if m is not None:
                                            Q = list(m.Q)
                                            served_qt = [[0.0 for _ in range(T)] for _ in Q]
                                            for tau in range(T):
                                                out_qs = [q for q in Q if float(m.yOUT[q, tau].value or 0.0) >= 0.5]
                                                ret_qs = [q for q in Q if float(m.yRET[q, tau].value or 0.0) >= 0.5]
                                                k_out = len(out_qs)
                                                k_ret = len(ret_qs)
                                                share_out = (float(pax_out[tau]) / k_out) if k_out > 0 else 0.0
                                                share_ret = (float(pax_ret[tau]) / k_ret) if k_ret > 0 else 0.0
                                                for q in out_qs:
                                                    served_qt[q][tau] += share_out
                                                for q in ret_qs:
                                                    served_qt[q][tau] += share_ret
                                            print("\nPax per shuttle and slot (total):")
                                            print(header)
                                            for q in Q:
                                                print(f"  q={q}: {fmt_row_floats(served_qt[q])}")
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                    except Exception:
                        pass
                    return BendersRunResult(
                        status=SolveStatus.FEASIBLE,
                        iterations=it,
                        best_lower_bound=best_lb,
                        best_upper_bound=best_ub,
                    )

        log.warning("Max iterations reached: %d", max_it)
        # Print incumbent solution if available at max-iterations stop
        try:
            elapsed = time.time() - t0
            print(f"\nMax iterations reached: {max_it}")
            print(f"Total solve time: {elapsed:.3f} seconds")
            if best_lb is not None and best_ub is not None:
                gap = abs(best_ub - best_lb)
                rel_gap = gap / max(1.0, abs(best_ub))
                print(f"Best bounds: LB={best_lb:.6g} UB={best_ub:.6g} gap={gap:.6g} rel={rel_gap:.3g}")
            fmt = getattr(self.master, "format_solution", None)
            if callable(fmt):
                print("\nBest Master Solution:")
                print(fmt())
        except Exception:
            pass
        return BendersRunResult(
            status=SolveStatus.UNKNOWN,
            iterations=max_it,
            best_lower_bound=best_lb,
            best_upper_bound=best_ub,
        )


__all__ = ["BendersSolver", "BendersRunResult"]
