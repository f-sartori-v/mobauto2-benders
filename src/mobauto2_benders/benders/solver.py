from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Iterable, Mapping, Any

from ..config import BendersConfig
from .core import CorePoint
from .master import MasterProblem
from .subproblem import Subproblem
from .types import SolveStatus, SubproblemResult

log = logging.getLogger(__name__)


# --- Global cut filtering knobs ---
# relative violation threshold
VIOL_TOL_REL: float = 1e-3
# rounding digits for coefficients in signatures
COEFF_ROUND_DIGITS: int = 5
# treat smaller coefficients as zero
COEFF_ZERO_TOL: float = 1e-8

# Global pool of cut signatures to avoid numerical duplicates
_cut_signatures: set[tuple] = set()

# Debug logging for cuts (to avoid log explosion)
DEBUG_CUTS: bool = True
DEBUG_MAX_CUTS: int = 50
_debug_cut_count: int = 0
_debug_suppressed_notice_done: bool = False
# Limit how many coefficients to list per cut in debug
DEBUG_COEFFS_TOP_K: int = 10


def _key_for_sort(idx: Any) -> Any:
    """Stable key for sorting heterogeneous index types in signatures."""
    if isinstance(idx, (int, float, str)):
        return idx
    try:
        return tuple(idx)  # type: ignore[arg-type]
    except Exception:
        return repr(idx)


def make_cut_signature(const: float, slopes: Mapping[Any, float] | Iterable[tuple[Any, float]], scope: Any | None = None) -> tuple:
    """Build a canonical signature for a cut: (rounded_const, ((idx, rounded_beta), ...)).

    - Only include coefficients with |beta| > COEFF_ZERO_TOL.
    - Round constant and betas to COEFF_ROUND_DIGITS.
    - Sort entries by index for determinism.
    """
    if not isinstance(slopes, Mapping):
        slopes = dict(slopes)  # type: ignore[arg-type]
    rc = round(float(const), COEFF_ROUND_DIGITS)
    items: list[tuple[Any, float]] = []
    for k, v in slopes.items():
        vv = float(v)
        if abs(vv) <= COEFF_ZERO_TOL:
            continue
        items.append((k, round(vv, COEFF_ROUND_DIGITS)))
    items.sort(key=lambda kv: _key_for_sort(kv[0]))
    # Include an optional scope key (e.g., scenario id) to avoid cross-scope dedup
    if scope is None:
        return (rc, tuple(items))
    try:
        scope_key = tuple(scope) if not isinstance(scope, (int, float, str)) else scope
    except Exception:
        scope_key = repr(scope)
    return (scope_key, rc, tuple(items))


def add_benders_cut(
    iteration: int,
    const: float,
    slopes: Mapping[Any, float] | Iterable[tuple[Any, float]],
    lhs_value: float,
    rhs_value: float,
    cut_type: str = "optimality",
    signature_scope: Any | None = None,
) -> bool:
    """Common filter for adding a Benders cut.

    - Computes violation and compares to relative threshold.
    - Skips numerically duplicate cuts using a global signature set.
    - Logs a concise status message and returns True if the cut should be added.
    """
    viol = float(rhs_value) - float(lhs_value)
    thr = VIOL_TOL_REL * (abs(float(rhs_value)) + 1.0)
    def _dbg(msg: str) -> None:
        global _debug_cut_count, _debug_suppressed_notice_done
        if not DEBUG_CUTS:
            return
        if _debug_cut_count < DEBUG_MAX_CUTS:
            log.info(msg)
            _debug_cut_count += 1
        else:
            if not _debug_suppressed_notice_done:
                log.info("[BENDERS] cut debug: further messages suppressed")
                _debug_suppressed_notice_done = True

    if viol < thr:
        _dbg(
            f"[BENDERS] cut skipped: small violation (it={iteration} type={cut_type} lhs={float(lhs_value):.6g} rhs={float(rhs_value):.6g} viol={viol:.3g} thr={thr:.3g})"
        )
        return False

    # Prepare slopes as a dictionary for debug printing and signature computation
    if isinstance(slopes, Mapping):
        _slopes_dict: dict[Any, float] = dict(slopes)
    else:
        _slopes_dict = dict(slopes)  # type: ignore[arg-type]

    # Verbose cut debug: print summary + top-K coefficients using module logger
    if DEBUG_CUTS:
        global _debug_cut_count
        _debug_cut_count += 1
        if _debug_cut_count <= DEBUG_MAX_CUTS:
            # Only report coefficients above zero tolerance
            nz_items = [(k, float(v)) for k, v in _slopes_dict.items() if abs(float(v)) > COEFF_ZERO_TOL]
            nnz = len(nz_items)
            # Summary: range of coefficients if any
            if nz_items:
                vals = [abs(v) for _, v in nz_items]
                # Sort by descending magnitude for top-K printout
                nz_items.sort(key=lambda kv: abs(kv[1]), reverse=True)
                vmin = min(nz_items, key=lambda kv: kv[1])[1]
                vmax = max(nz_items, key=lambda kv: kv[1])[1]
                log.info(
                    "[CUT DEBUG] it=%s type=%s const=%.6g nnz=%s range=[%.3g,%.3g]",
                    str(iteration), str(cut_type), float(const), nnz, vmin, vmax,
                )
            else:
                log.info(
                    "[CUT DEBUG] it=%s type=%s const=%.6g nnz=0",
                    str(iteration), str(cut_type), float(const),
                )
            # Print top-K by absolute value
            kmax = max(0, int(DEBUG_COEFFS_TOP_K))
            for k, v in nz_items[:kmax]:
                try:
                    log.info("beta[%s] = %.6g", str(k), float(v))
                except Exception:
                    log.info("beta[%r] = %.6g", k, float(v))
            if nnz > kmax > 0:
                log.info("... (%d coefficient(s) omitted)", nnz - kmax)

    sig = make_cut_signature(const, _slopes_dict, scope=signature_scope)
    if sig in _cut_signatures:
        nnz = len(sig[-1]) if isinstance(sig, tuple) and len(sig) > 0 else "-"
        _dbg(f"[BENDERS] cut skipped: duplicate (it={iteration} type={cut_type} nnz={nnz})")
        return False

    _cut_signatures.add(sig)
    nnz = len(sig[-1]) if isinstance(sig, tuple) and len(sig) > 0 else "-"
    _dbg(f"[BENDERS] cut added (it={iteration} type={cut_type} nnz={nnz})")
    return True


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

        # Helper: print diagnostics from the most recent subproblem evaluation
        def _print_sp_diagnostics(diag: dict | None) -> None:
            if not diag:
                return
            # If multiple scenarios were evaluated, expect a list under 'scenarios'
            scenarios = diag.get("scenarios") if isinstance(diag, dict) else None
            try:
                m = getattr(self.master, "m", None)
            except Exception:
                m = None
            def _fmt_header(T: int) -> str:
                return "       " + " ".join(f"{t:>3d}" for t in range(int(T)))
            def _fmt_row(vals: list[float], T: int) -> str:
                return " ".join(f"{float(v):>3.0f}" for v in (list(vals) + [0.0] * int(T))[: int(T)])
            def _map_layers_to_shuttles(T: int, pax_by_tau_k: list[list[float]], dir_: str) -> list[list[float]]:
                # Build per-shuttle matrix [q][tau] from per-layer at each tau using master starts
                if m is None:
                    # Fallback: evenly distribute across layers if no model available
                    # Keep same shape as Q x T with zeros
                    try:
                        Q = list(getattr(m, "Q", []))
                        qn = len(Q)
                    except Exception:
                        qn = 0
                    return [[0.0 for _ in range(T)] for _ in range(qn)]
                Q = list(m.Q)
                per_q_tau = [[0.0 for _ in range(T)] for _ in Q]
                for tau in range(T):
                    # Determine which shuttles start at tau in the given direction
                    if dir_.upper() == "OUT":
                        qs = [q for q in Q if float(m.yOUT[q, tau].value or 0.0) >= 0.5]
                    else:
                        qs = [q for q in Q if float(m.yRET[q, tau].value or 0.0) >= 0.5]
                    kmax = min(len(qs), len(pax_by_tau_k[tau]) if tau < len(pax_by_tau_k) else 0)
                    for k in range(kmax):
                        q = qs[k]
                        per_q_tau[q][tau] = float(pax_by_tau_k[tau][k] or 0.0)
                return per_q_tau
            # Multi-scenario path
            if isinstance(scenarios, list) and scenarios:
                try:
                    T = int(diag.get("T")) if "T" in diag else None
                except Exception:
                    T = None
                for idx, sdiag in enumerate(scenarios, start=1):
                    try:
                        label = sdiag.get("label", f"scenario {idx}")
                    except Exception:
                        label = f"scenario {idx}"
                    try:
                        R_out = sdiag.get("R_out")
                        R_ret = sdiag.get("R_ret")
                        pax_out = sdiag.get("pax_out_by_tau")
                        pax_ret = sdiag.get("pax_ret_by_tau")
                        pax_out_k = sdiag.get("pax_out_by_tau_k") or []
                        pax_ret_k = sdiag.get("pax_ret_by_tau_k") or []
                        if T is None:
                            if isinstance(pax_out, list):
                                T = len(pax_out)
                            elif isinstance(R_out, list):
                                T = len(R_out)
                        if not isinstance(T, int):
                            continue
                        header = _fmt_header(T)
                        print(f"\nScenario {idx}: {label}")
                        if isinstance(R_out, list) and isinstance(R_ret, list):
                            print("Demand per slot (OUT/RET):")
                            print(header)
                            print(f"  OUT: {_fmt_row(R_out, T)}")
                            print(f"  RET: {_fmt_row(R_ret, T)}")
                        # Per-shuttle passengers using per-layer flows mapped to shuttles
                        try:
                            per_q_out = _map_layers_to_shuttles(T, pax_out_k, "OUT")
                            per_q_ret = _map_layers_to_shuttles(T, pax_ret_k, "RET")
                            # Totals (OUT+RET)
                            Q = list(m.Q) if m is not None else list(range(len(per_q_out)))
                            print("Passengers per shuttle and slot (OUT):")
                            print(header)
                            for q in Q:
                                print(f"  q={q}: {_fmt_row(per_q_out[q] if q < len(per_q_out) else [0.0]*T, T)}")
                            print("Passengers per shuttle and slot (RET):")
                            print(header)
                            for q in Q:
                                print(f"  q={q}: {_fmt_row(per_q_ret[q] if q < len(per_q_ret) else [0.0]*T, T)}")
                            # Combined total per shuttle (optional)
                            try:
                                print("Passengers per shuttle and slot (TOTAL):")
                                print(header)
                                for q in Q:
                                    row = [0.0 for _ in range(T)]
                                    if q < len(per_q_out):
                                        for t in range(T):
                                            row[t] += float(per_q_out[q][t] if t < len(per_q_out[q]) else 0.0)
                                    if q < len(per_q_ret):
                                        for t in range(T):
                                            row[t] += float(per_q_ret[q][t] if t < len(per_q_ret[q]) else 0.0)
                                    print(f"  q={q}: {_fmt_row(row, T)}")
                            except Exception:
                                pass
                        except Exception:
                            pass
                    except Exception:
                        continue
                return
            # Single-scenario path (legacy)
            try:
                T = int(diag.get("T")) if "T" in diag else None
                R_out = diag.get("R_out")
                R_ret = diag.get("R_ret")
                pax_out = diag.get("pax_out_by_tau")
                pax_ret = diag.get("pax_ret_by_tau")
                if isinstance(pax_out, list) and isinstance(pax_ret, list):
                    n = len(pax_out)
                    if T is None:
                        T = n
                    header = _fmt_header(T)
                    if isinstance(R_out, list) and isinstance(R_ret, list):
                        print("\nDemand per slot (OUT/RET):")
                        print(header)
                        print(f"  OUT: {_fmt_row(R_out, T)}")
                        print(f"  RET: {_fmt_row(R_ret, T)}")
                    try:
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
                                print(f"  q={q}: {_fmt_row(served_qt[q], T)}")
                    except Exception:
                        pass
            except Exception:
                pass

        # Optionally install lazy constraints callback if supported
        use_lazy = bool(self.cfg.master.params.get("use_lazy_cuts", False))
        # Magnanti–Wong requires explicit separation to access the core point; disable lazy when enabled
        try:
            if bool(self.cfg.subproblem.params.get("use_magnanti_wong", False)):
                use_lazy = False
        except Exception:
            pass
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
                        _print_sp_diagnostics(last_diag)
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

            # Optional: update and pass a core point for Magnanti–Wong selection
            try:
                mw_enabled = bool(self.cfg.subproblem.params.get("use_magnanti_wong", False))
            except Exception:
                mw_enabled = False
            if mw_enabled:
                # Lazy-init the core point helper with configured alpha
                try:
                    alpha = float(self.cfg.subproblem.params.get("mw_core_alpha", 0.3) or 0.3)
                except Exception:
                    alpha = 0.3
                if not hasattr(self, "_core_point") or self._core_point is None:
                    self._core_point = CorePoint(alpha=alpha)
                # Update core from current incumbent (moving average)
                try:
                    # Provide a hint for T from last diagnostics if available
                    T_hint = None
                    if last_diag and isinstance(last_diag, dict) and "T" in last_diag:
                        try:
                            T_hint = int(last_diag.get("T"))
                        except Exception:
                            T_hint = None
                    self._core_point.update_from_candidate(mres.candidate, T_hint=T_hint)
                    # Attach to subproblem params for MW dual selection
                    if isinstance(getattr(self.subproblem, "params", None), dict):
                        self.subproblem.params["mw_core_point"] = self._core_point.as_params()
                        self.subproblem.params["use_magnanti_wong"] = True
                except Exception:
                    pass

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
            # Cut tightness check: evaluate line(y) from the raw cut metadata and compare to SP upper bound
            try:
                if sres.cut is not None and sres.upper_bound is not None and mres.candidate is not None:
                    cmeta = getattr(sres.cut, "metadata", {}) or {}
                    const = float(cmeta.get("const", 0.0))
                    coeff_yout = cmeta.get("coeff_yOUT") or {}
                    coeff_yret = cmeta.get("coeff_yRET") or {}
                    line_val = float(const)
                    # Candidate has keys like 'yOUT[q,t]' and 'yRET[q,t]'
                    def _cand_val(prefix: str, q: int, t: int) -> float:
                        return float(mres.candidate.get(f"{prefix}[{int(q)},{int(t)}]", 0.0))
                    if isinstance(coeff_yout, dict):
                        for (q, tau), v in coeff_yout.items():
                            line_val += float(v) * _cand_val("yOUT", int(q), int(tau))
                    if isinstance(coeff_yret, dict):
                        for (q, tau), v in coeff_yret.items():
                            line_val += float(v) * _cand_val("yRET", int(q), int(tau))
                    diff = float(line_val) - float(sres.upper_bound)
                    print(
                        f"[CUT TIGHTNESS] line(y)={line_val:.6g}  SP_ub={float(sres.upper_bound):.6g}  diff={diff:.3g}"
                    )
            except Exception:
                pass
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
                        _print_sp_diagnostics(last_diag)
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
                if stall_max > 0 and stall_ctr >= stall_max:
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
                            _print_sp_diagnostics(last_diag)
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
            # Diagnostics from last evaluated subproblem, if any
            try:
                _print_sp_diagnostics(last_diag)
            except Exception:
                pass
        except Exception:
            pass
        return BendersRunResult(
            status=SolveStatus.UNKNOWN,
            iterations=max_it,
            best_lower_bound=best_lb,
            best_upper_bound=best_ub,
        )


__all__ = ["BendersSolver", "BendersRunResult"]
