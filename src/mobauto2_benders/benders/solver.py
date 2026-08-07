from __future__ import annotations

import logging
import time
import math
import copy
from dataclasses import dataclass
from typing import Optional, Iterable, Mapping, Any

from ..config import RootConfig
from .master import MasterProblem
from .subproblem import Subproblem
from .types import SolveStatus, SubproblemResult
from .cplex_log import parse_cplex_log_bounds
import pyomo.environ as pyo

log = logging.getLogger(__name__)


# --- Global cut filtering knobs ---
# relative violation threshold
VIOL_TOL_REL: float = 1e-8
# rounding digits for coefficients in signatures
COEFF_ROUND_DIGITS: int = 6
# treat smaller coefficients as zero
COEFF_ZERO_TOL: float = 1e-6


def _get_default_signature_set() -> set[tuple]:
    # Fallback only; callers should pass a per-run set.
    return set()


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


def make_cut_signature(
    const: float,
    slopes: Mapping[Any, float] | Iterable[tuple[Any, float]],
    scope: Any | None = None,
) -> tuple:
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


def make_slope_signature(
    slopes: Mapping[Any, float] | Iterable[tuple[Any, float]], scope: Any | None = None
) -> tuple:
    """Build a canonical signature based only on slopes (ignore const)."""
    if not isinstance(slopes, Mapping):
        slopes = dict(slopes)  # type: ignore[arg-type]
    items: list[tuple[Any, float]] = []
    for k, v in slopes.items():
        vv = float(v)
        if abs(vv) <= COEFF_ZERO_TOL:
            continue
        items.append((k, round(vv, COEFF_ROUND_DIGITS)))
    items.sort(key=lambda kv: _key_for_sort(kv[0]))
    if scope is None:
        return tuple(items)
    try:
        scope_key = tuple(scope) if not isinstance(scope, (int, float, str)) else scope
    except Exception:
        scope_key = repr(scope)
    return (scope_key, tuple(items))


def set_cut_tolerances(eps_cut: float, eps_hash: float) -> None:
    global VIOL_TOL_REL, COEFF_ZERO_TOL, COEFF_ROUND_DIGITS
    try:
        VIOL_TOL_REL = float(eps_cut)
    except Exception:
        pass
    try:
        COEFF_ZERO_TOL = float(eps_hash)
    except Exception:
        pass
    try:
        # round digits consistent with eps_hash (e.g., 1e-6 -> 6 digits)
        if eps_hash > 0:
            COEFF_ROUND_DIGITS = max(0, int(round(-math.log10(float(eps_hash)))))
    except Exception:
        pass


def add_benders_cut(
    iteration: int,
    const: float,
    slopes: Mapping[Any, float] | Iterable[tuple[Any, float]],
    lhs_value: float,
    rhs_value: float,
    cut_type: str = "optimality",
    signature_scope: Any | None = None,
    cuts_in_model: int | None = None,
    signature_set: set[tuple] | None = None,
    slope_const_map: dict[tuple, float] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> bool:
    """Common filter for adding a Benders cut.

    - Computes violation and compares to relative threshold.
    - Skips numerically duplicate cuts using a global signature set.
    - Logs a concise status message and returns True if the cut should be added.
    """
    viol = float(rhs_value) - float(lhs_value)
    thr = VIOL_TOL_REL * (abs(float(rhs_value)) + 1.0)
    if diagnostics is not None:
        diagnostics.update(
            {
                "iteration": int(iteration),
                "cut_type": str(cut_type),
                "lhs_value": float(lhs_value),
                "rhs_value": float(rhs_value),
                "violation": float(viol),
                "violation_threshold": float(thr),
                "const": float(const),
                "cuts_in_model": cuts_in_model,
            }
        )

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
            nz_items = [
                (k, float(v))
                for k, v in _slopes_dict.items()
                if abs(float(v)) > COEFF_ZERO_TOL
            ]
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
                    str(iteration),
                    str(cut_type),
                    float(const),
                    nnz,
                    vmin,
                    vmax,
                )
            else:
                log.info(
                    "[CUT DEBUG] it=%s type=%s const=%.6g nnz=0",
                    str(iteration),
                    str(cut_type),
                    float(const),
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

    if not _slopes_dict:
        if diagnostics is not None:
            diagnostics["decision"] = "skip"
            diagnostics["skip_reason"] = "invalid_slopes"
        log.info(
            "[BENDERS] skip reason=invalid_slopes signature=None cuts_in_model=%s",
            str(cuts_in_model),
        )
        return False
    # If effectively empty (all slopes ~0 and const ~0), skip
    if abs(float(const)) <= COEFF_ZERO_TOL and all(
        abs(float(v)) <= COEFF_ZERO_TOL for v in _slopes_dict.values()
    ):
        if diagnostics is not None:
            diagnostics["decision"] = "skip"
            diagnostics["skip_reason"] = "numerically_empty"
        log.info(
            "[BENDERS] skip reason=numerically_empty signature=None cuts_in_model=%s",
            str(cuts_in_model),
        )
        return False

    # Dominance pruning by slope pattern: keep only the largest const for a given slope signature
    if slope_const_map is not None:
        try:
            slope_sig = make_slope_signature(_slopes_dict, scope=signature_scope)
            prev_const = slope_const_map.get(slope_sig)
            if prev_const is not None and float(const) <= float(prev_const) + 1e-9:
                if diagnostics is not None:
                    diagnostics["decision"] = "skip"
                    diagnostics["skip_reason"] = "dominated_by_slope"
                    diagnostics["slope_signature"] = slope_sig
                    diagnostics["previous_const"] = float(prev_const)
                log.info(
                    "[BENDERS] skip reason=dominated_by_slope signature=%s cuts_in_model=%s",
                    str(slope_sig),
                    str(cuts_in_model),
                )
                return False
            slope_const_map[slope_sig] = float(const)
        except Exception:
            pass

    # If this is the very first cut, accept (unless numerically empty)
    if cuts_in_model == 0:
        sigset = (
            signature_set if signature_set is not None else _get_default_signature_set()
        )
        sig = make_cut_signature(const, _slopes_dict, scope=signature_scope)
        if diagnostics is not None:
            diagnostics["decision"] = "add"
            diagnostics["signature"] = sig
        log.info("[BENDERS] signature=%s sigset_size=%s", str(sig), str(len(sigset)))
        sigset.add(sig)
        nnz = len(sig[-1]) if isinstance(sig, tuple) and len(sig) > 0 else "-"
        _dbg(f"[BENDERS] cut added (it={iteration} type={cut_type} nnz={nnz})")
        return True

    if viol < thr:
        if diagnostics is not None:
            diagnostics["decision"] = "skip"
            diagnostics["skip_reason"] = "numerically_small"
        _dbg(
            f"[BENDERS] cut skipped: small violation (it={iteration} type={cut_type} lhs={float(lhs_value):.6g} rhs={float(rhs_value):.6g} viol={viol:.3g} thr={thr:.3g})"
        )
        log.info(
            "[BENDERS] skip reason=numerically_small signature=None cuts_in_model=%s",
            str(cuts_in_model),
        )
        return False

    sigset = (
        signature_set if signature_set is not None else _get_default_signature_set()
    )
    sig = make_cut_signature(const, _slopes_dict, scope=signature_scope)
    if diagnostics is not None:
        diagnostics["signature"] = sig
    log.info("[BENDERS] signature=%s sigset_size=%s", str(sig), str(len(sigset)))
    if sig in sigset:
        if diagnostics is not None:
            diagnostics["decision"] = "skip"
            diagnostics["skip_reason"] = "duplicate_by_signature"
        nnz = len(sig[-1]) if isinstance(sig, tuple) and len(sig) > 0 else "-"
        _dbg(
            f"[BENDERS] cut skipped: duplicate (it={iteration} type={cut_type} nnz={nnz})"
        )
        log.info(
            "[BENDERS] skip reason=duplicate_by_signature signature=%s cuts_in_model=%s",
            str(sig),
            str(cuts_in_model),
        )
        return False

    sigset.add(sig)
    if diagnostics is not None:
        diagnostics["decision"] = "add"
    nnz = len(sig[-1]) if isinstance(sig, tuple) and len(sig) > 0 else "-"
    _dbg(f"[BENDERS] cut added (it={iteration} type={cut_type} nnz={nnz})")
    return True


@dataclass(slots=True)
class BendersRunResult:
    status: SolveStatus
    iterations: int
    best_lower_bound: Optional[float]
    best_upper_bound: Optional[float]
    pax_served: Optional[float] = None
    pax_total: Optional[float] = None
    subproblem_obj: Optional[float] = None
    sp_wait_cost_slots: Optional[float] = None
    sp_fill_eps_cost: Optional[float] = None
    sp_penalty_cost: Optional[float] = None
    sp_penalty_pax: Optional[float] = None
    sp_total_demand: Optional[float] = None
    sp_slot_resolution: Optional[int] = None
    # Which generator actually produced the cuts, and whether they support a
    # lower bound at all. Both were previously invisible: Magnanti-Wong failed
    # silently to finite differences while still claiming a valid bound, and
    # nothing in any output said so. The manifest records these.
    cut_generation_mode: Optional[str] = None
    cut_valid_lower_bound: Optional[bool] = None
    # How many master solves stopped on the clock instead of on the gap. Any value
    # above zero means the run is NOT bit-reproducible: node counts inside a
    # time-limited solve depend on machine load, and every later iteration inherits
    # the difference. configs/baseline_d9.yaml lists this as a determinism
    # requirement; measured, a binding limit moved the LB by 8% between two runs of
    # one config while a non-binding one reproduced to the last digit.
    clock_truncated_master_solves: Optional[int] = None


@dataclass(slots=True)
class IterReport:
    k: int
    best_lb: Optional[float] = None
    best_ub: Optional[float] = None
    gap_abs: Optional[float] = None
    gap_rel_pct: Optional[float] = None
    mp_status: Optional[str] = None
    mp_obj: Optional[float] = None
    mp_best_bound: Optional[float] = None
    mp_gap_pct: Optional[float] = None
    mp_nodes: Optional[float] = None
    sp_feasible: Optional[bool] = None
    sp_recourse: Optional[float] = None
    candidate_ub: Optional[float] = None
    pax_served: Optional[float] = None
    pax_total: Optional[float] = None
    penalty_pax: Optional[float] = None
    avg_wait_min: Optional[float] = None
    cuts_added: Optional[int] = None
    cuts_total: Optional[int] = None
    last_cut_type: Optional[str] = None
    last_cut_nnz: Optional[int] = None
    t_master_s: Optional[float] = None
    t_sp_solve_s: Optional[float] = None
    t_cutgen_s: Optional[float] = None
    t_iter_s: Optional[float] = None


def _fmt_float(val: Optional[float], digits: int) -> str:
    if val is None:
        return "NA"
    try:
        return f"{float(val):.{digits}f}"
    except Exception:
        return "NA"


def _fmt_int(val: Optional[float | int]) -> str:
    if val is None:
        return "NA"
    try:
        return str(int(val))
    except Exception:
        return "NA"


def format_report_line(rep: IterReport) -> str:
    if rep.best_lb is not None and rep.best_ub is not None:
        gap_abs = abs(float(rep.best_ub) - float(rep.best_lb))
        gap_rel = gap_abs / max(1.0, abs(float(rep.best_ub)))
        rep.gap_abs = gap_abs
        rep.gap_rel_pct = 100.0 * gap_rel
    served = "NA"
    if rep.pax_served is not None and rep.pax_total is not None:
        try:
            served = f"{float(rep.pax_served):.0f}/{float(rep.pax_total):.0f}"
        except Exception:
            served = "NA"
    feas = "NA" if rep.sp_feasible is None else str(bool(rep.sp_feasible))
    return (
        f"k={rep.k} | "
        f"BND LB={_fmt_float(rep.best_lb, 2)} UB={_fmt_float(rep.best_ub, 2)} "
        f"gap={_fmt_float(rep.gap_abs, 2)} rel={_fmt_float(rep.gap_rel_pct, 2)}% | "
        f"MP st={rep.mp_status or 'NA'} obj={_fmt_float(rep.mp_obj, 2)} bd={_fmt_float(rep.mp_best_bound, 2)} "
        f"gap={_fmt_float(rep.mp_gap_pct, 2)}% nodes={_fmt_int(rep.mp_nodes)} | "
        f"SP feas={feas} rec={_fmt_float(rep.sp_recourse, 2)} ub={_fmt_float(rep.candidate_ub, 2)} "
        f"served={served} pen={_fmt_float(rep.penalty_pax, 2)} avgW={_fmt_float(rep.avg_wait_min, 2)} | "
        f"CUT add={_fmt_int(rep.cuts_added)} tot={_fmt_int(rep.cuts_total)} "
        f"type={rep.last_cut_type or 'NA'} nnz={_fmt_int(rep.last_cut_nnz)} | "
        f"TIME mp={_fmt_float(rep.t_master_s, 3)} sp={_fmt_float(rep.t_sp_solve_s, 3)} "
        f"cut={_fmt_float(rep.t_cutgen_s, 3)} it={_fmt_float(rep.t_iter_s, 3)}"
    )


class BendersSolver:
    def __init__(self, master: MasterProblem, subproblem: Subproblem, cfg: RootConfig):
        self.master = master
        self.subproblem = subproblem
        self.cfg = cfg
        # MW core point state (aggregate space)
        self._mw_core_out = None
        self._mw_core_ret = None

    def run(self) -> BendersRunResult:
        t0 = time.time()
        set_cut_tolerances(self.cfg.tolerances.eps_cut, self.cfg.tolerances.eps_hash)
        report_mode = str(self.cfg.run.log_level).upper() == "REPORT"

        def _vprint(*args, **kwargs) -> None:
            if not report_mode:
                print(*args, **kwargs)

        max_it = self.cfg.solver.max_iterations
        tol = self.cfg.solver.tolerance
        self.master.initialize()
        _vprint("Initialized master problem.")

        total_master_time = 0.0
        total_sp_solve_time = 0.0
        total_cutgen_time = 0.0
        total_cutadd_time = 0.0
        total_iter = 0

        last_pax_served: Optional[float] = None
        last_pax_total: Optional[float] = None
        no_cut_streak = 0
        last_mp_gap: Optional[float] = None
        last_mp_term: Optional[str] = None
        clock_truncated_master_solves = 0
        # Master schedule parameters tied to global BD gap.
        #
        # The schedule below overwrites the master's per-iteration time limit and
        # mipgap on every iteration. Before these two bounds were wired, that made
        # master.solve_time_limit_s and master.mipgap inert: the config was parsed,
        # passed down, and discarded (same class as D19). The config now supplies the
        # ceilings, so the schedule can only tighten, never loosen past what was asked
        # for.
        mp_gap_min = 0.001
        mp_gap_max = float(
            self.cfg.master.per_iteration_mipgap
            if self.cfg.master.per_iteration_mipgap is not None
            else 0.05
        )
        mp_gap_scale = 1.0
        mp_time_base = 2.0
        mp_time_k = 5.0
        mp_tl_cap = (
            float(self.cfg.master.per_iteration_time_limit_s)
            if self.cfg.master.per_iteration_time_limit_s is not None
            else None
        )
        if mp_gap_min > mp_gap_max:
            # A config asking for a tighter ceiling than the schedule's floor wins.
            mp_gap_min = mp_gap_max
        # MW config (default enabled)
        sp_params = self.subproblem.params or {}
        mw_enabled = bool(
            sp_params.get("use_magnanti_wong", self.cfg.subproblem.use_magnanti_wong)
        )
        mw_alpha = float(
            sp_params.get("mw_core_alpha", self.cfg.subproblem.mw_core_alpha) or 0.10
        )
        mw_eps = float(
            sp_params.get(
                "mw_core_eps", getattr(self.cfg.subproblem, "mw_core_eps", 1e-3)
            )
            or 1e-3
        )
        if not (0.0 < mw_alpha <= 1.0):
            mw_alpha = 0.10
        if mw_eps <= 0.0:
            mw_eps = 1e-3
        _vprint(f"[MW] enabled={mw_enabled} alpha={mw_alpha:.3g} eps={mw_eps:.3g}")

        def _calc_pax_totals_from_diag(
            diag: dict | None,
        ) -> tuple[Optional[float], Optional[float]]:
            if not isinstance(diag, dict):
                return None, None
            scenarios = diag.get("scenarios")
            if isinstance(scenarios, list) and scenarios:
                weights = diag.get("scenario_weights")
                served_vals = []
                total_vals = []
                for sdiag in scenarios:
                    try:
                        R_out = sdiag.get("R_out") or []
                        R_ret = sdiag.get("R_ret") or []
                        pax_out = sdiag.get("pax_out_by_tau") or []
                        pax_ret = sdiag.get("pax_ret_by_tau") or []
                        total = float(sum(R_out) + sum(R_ret))
                        served = float(sum(pax_out) + sum(pax_ret))
                        total_vals.append(total)
                        served_vals.append(served)
                    except Exception:
                        continue
                if not served_vals or not total_vals:
                    return None, None
                if isinstance(weights, list) and len(weights) == len(served_vals):
                    served_w = sum(float(w) * s for w, s in zip(weights, served_vals))
                    total_w = sum(float(w) * t for w, t in zip(weights, total_vals))
                    return served_w, total_w
                # Default to simple average if weights missing
                n = float(len(served_vals))
                return sum(served_vals) / n, sum(total_vals) / n
            try:
                R_out = diag.get("R_out") or []
                R_ret = diag.get("R_ret") or []
                pax_out = diag.get("pax_out_by_tau") or []
                pax_ret = diag.get("pax_ret_by_tau") or []
                total = float(sum(R_out) + sum(R_ret))
                served = float(sum(pax_out) + sum(pax_ret))
                return served, total
            except Exception:
                return None, None

        # Helper: print diagnostics from the most recent subproblem evaluation
        def _report_y(dir_: str, q: int, tau: int) -> float:
            """Departure indicator for the solution the end-of-run report describes.

            Prefers the incumbent snapshot (best_candidate) so the passenger tables
            agree with the timeline and the reported objective. Falls back to the
            live model only when no incumbent was recorded, e.g. a run that ended
            before the first improving solution.
            """
            key = (
                f"yOUT[{int(q)},{int(tau)}]"
                if dir_.upper() == "OUT"
                else f"yRET[{int(q)},{int(tau)}]"
            )
            if isinstance(best_candidate, dict) and key in best_candidate:
                try:
                    return float(best_candidate[key])
                except Exception:
                    pass
            mm = getattr(self.master, "m", None)
            if mm is None:
                return 0.0
            try:
                var = mm.yOUT if dir_.upper() == "OUT" else mm.yRET
                return float(var[q, tau].value or 0.0)
            except Exception:
                return 0.0

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
                return " ".join(
                    f"{float(v):>3.0f}" for v in (list(vals) + [0.0] * int(T))[: int(T)]
                )

            def _map_layers_to_shuttles(
                T: int, pax_by_tau_k: list[list[float]], dir_: str
            ) -> list[list[float]]:
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
                    # Determine which shuttles start at tau in the given direction.
                    # Read the SAME solution the rest of the report describes (the
                    # incumbent snapshot), not the live model, which holds the last
                    # master solve. Mixing the two silently drops every passenger
                    # assigned to a slot where the last solve had no departure.
                    qs = [q for q in Q if _report_y(dir_, q, tau) >= 0.5]
                    kmax = min(
                        len(qs),
                        len(pax_by_tau_k[tau]) if tau < len(pax_by_tau_k) else 0,
                    )
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
                            Q = (
                                list(m.Q)
                                if m is not None
                                else list(range(len(per_q_out)))
                            )
                            print("Passengers per shuttle and slot (OUT):")
                            print(header)
                            for q in Q:
                                print(
                                    f"  q={q}: {_fmt_row(per_q_out[q] if q < len(per_q_out) else [0.0]*T, T)}"
                                )
                            print("Passengers per shuttle and slot (RET):")
                            print(header)
                            for q in Q:
                                print(
                                    f"  q={q}: {_fmt_row(per_q_ret[q] if q < len(per_q_ret) else [0.0]*T, T)}"
                                )
                            # Combined total per shuttle (optional)
                            try:
                                print("Passengers per shuttle and slot (TOTAL):")
                                print(header)
                                for q in Q:
                                    row = [0.0 for _ in range(T)]
                                    if q < len(per_q_out):
                                        for t in range(T):
                                            row[t] += float(
                                                per_q_out[q][t]
                                                if t < len(per_q_out[q])
                                                else 0.0
                                            )
                                    if q < len(per_q_ret):
                                        for t in range(T):
                                            row[t] += float(
                                                per_q_ret[q][t]
                                                if t < len(per_q_ret[q])
                                                else 0.0
                                            )
                                    print(f"  q={q}: {_fmt_row(row, T)}")
                            except Exception:
                                pass
                        except Exception:
                            pass
                        try:
                            if (
                                isinstance(pax_out, list)
                                and isinstance(pax_ret, list)
                                and isinstance(R_out, list)
                                and isinstance(R_ret, list)
                            ):
                                served = float(sum(pax_out) + sum(pax_ret))
                                total = float(sum(R_out) + sum(R_ret))
                                print(f"Pax served: {served:.0f}/{total:.0f}")
                        except Exception:
                            pass
                    except Exception:
                        continue
                nonlocal last_pax_served, last_pax_total
                last_pax_served, last_pax_total = _calc_pax_totals_from_diag(diag)
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
                            # Use the subproblem's true per-layer counts and the
                            # incumbent's departures, exactly as the multi-scenario
                            # path does. The previous version split pax_out[tau]
                            # EQUALLY across the vehicles departing at tau -- a
                            # fiction that made shuttles look identical -- and read
                            # those departures from the live model, so passengers
                            # assigned to slots the last master solve left empty
                            # were dropped from the table while still counted in
                            # the "Pax served" total below.
                            pax_out_k = diag.get("pax_out_by_tau_k")
                            pax_ret_k = diag.get("pax_ret_by_tau_k")
                            served_qt: list[list[float]] | None = None
                            if isinstance(pax_out_k, list) and isinstance(
                                pax_ret_k, list
                            ):
                                per_q_out = _map_layers_to_shuttles(T, pax_out_k, "OUT")
                                per_q_ret = _map_layers_to_shuttles(T, pax_ret_k, "RET")
                                served_qt = [
                                    [
                                        float(
                                            per_q_out[q][t]
                                            if q < len(per_q_out)
                                            and t < len(per_q_out[q])
                                            else 0.0
                                        )
                                        + float(
                                            per_q_ret[q][t]
                                            if q < len(per_q_ret)
                                            and t < len(per_q_ret[q])
                                            else 0.0
                                        )
                                        for t in range(T)
                                    ]
                                    for q in Q
                                ]
                            if served_qt is not None:
                                print("\nPax per shuttle and slot (total):")
                                print(header)
                                for q in Q:
                                    print(f"  q={q}: {_fmt_row(served_qt[q], T)}")
                                # Cross-check: the table must account for every served
                                # passenger. A mismatch means the table and the total
                                # describe different solutions -- say so rather than
                                # printing a total that appears to validate the table.
                                try:
                                    table_sum = float(
                                        sum(sum(row) for row in served_qt)
                                    )
                                    served_chk = float(sum(pax_out) + sum(pax_ret))
                                    if abs(table_sum - served_chk) > 1e-6 * max(
                                        1.0, abs(served_chk)
                                    ):
                                        print(
                                            f"  [WARN] per-shuttle table sums to {table_sum:.0f} but "
                                            f"{served_chk:.0f} passengers were served; the table and the "
                                            "totals describe different solutions."
                                        )
                                except Exception:
                                    pass
                            try:
                                served = float(sum(pax_out) + sum(pax_ret))
                                total = float(sum(R_out) + sum(R_ret))
                                print(f"Pax served: {served:.0f}/{total:.0f}")
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass
            last_pax_served, last_pax_total = _calc_pax_totals_from_diag(diag)

        best_lb: Optional[float] = None
        best_ub: Optional[float] = None
        best_candidate: dict[str, float] | None = None
        best_diag: dict[str, Any] | None = None
        best_pax_served: Optional[float] = None
        best_pax_total: Optional[float] = None
        prev_best_lb: Optional[float] = None
        prev_best_ub: Optional[float] = None
        lower_bound_semantics_valid = True
        # Stall detection on gap improvement
        stall_max = int(getattr(self.cfg.solver, "stall_max_no_improve_iters", 0) or 0)
        stall_min_abs = float(
            getattr(self.cfg.solver, "stall_min_abs_improve", 0.0) or 0.0
        )
        stall_min_rel = float(
            getattr(self.cfg.solver, "stall_min_rel_improve", 0.0) or 0.0
        )
        stall_ctr = 0
        prev_gap: Optional[float] = None

        # LP phase. The early iterations spend thousands of branch-and-bound nodes
        # producing one cut from a cut set that is still nearly empty. Solving the
        # master without integrality is far cheaper and yields a fractional y whose
        # subproblem dual is still a valid cut, because the recourse value function
        # is convex in y (it is an LP parameterised by its right-hand side).
        #
        # What the phase must NOT do is claim an upper bound: a fractional schedule
        # is not exhibitable, so its recourse cost bounds nothing. The guards below
        # keep best_ub, the incumbent and the reported pax counts untouched while
        # the phase is active. The LB is safe -- the LP master's optimum is a lower
        # bound on the MIP master's, which is a lower bound on the true optimum.
        lp_phase_enabled = bool(getattr(self.cfg.master, "lp_phase", False))
        lp_phase_max_iters = int(getattr(self.cfg.master, "lp_phase_max_iters", 0) or 0)
        lp_phase_stall_iters = int(
            getattr(self.cfg.master, "lp_phase_stall_iters", 0) or 0
        )
        lp_phase_min_rel_improve = float(
            getattr(self.cfg.master, "lp_phase_min_rel_improve", 0.0) or 0.0
        )
        lp_phase_active = bool(lp_phase_enabled and lp_phase_max_iters > 0)
        lp_phase_iters = 0
        lp_phase_stall_ctr = 0
        lp_phase_prev_obj: Optional[float] = None
        lp_phase_exit_reason: Optional[str] = None
        if lp_phase_active:
            _vprint(
                f"[LP-PHASE] on: max_iters={lp_phase_max_iters} "
                f"stall_iters={lp_phase_stall_iters} min_rel_improve={lp_phase_min_rel_improve:g}"
            )

        last_diag: dict | None = None

        def _report_diag() -> dict[str, Any] | None:
            if isinstance(best_diag, dict):
                return best_diag
            if isinstance(last_diag, dict):
                return last_diag
            return None

        def _set_master_report_snapshot() -> None:
            setter = getattr(self.master, "set_report_solution", None)
            if not callable(setter):
                return
            realized_map: dict[tuple[int, int], float] | None = None
            diag = _report_diag()
            if isinstance(diag, dict):
                try:
                    raw_map = diag.get("realized_departure_min_map", {})
                    if isinstance(raw_map, dict):
                        realized_map = dict(raw_map)
                except Exception:
                    realized_map = None
            try:
                setter(best_candidate, realized_map)
            except Exception:
                pass

        def _extract_sp_diag(diag: dict | None) -> dict[str, Any]:
            if not isinstance(diag, dict):
                return {}
            return {
                "subproblem_obj": diag.get("objective_value"),
                "sp_wait_cost_slots": diag.get("waiting_cost_slots"),
                "sp_fill_eps_cost": diag.get("fill_eps_cost"),
                "sp_penalty_cost": diag.get("penalty_cost"),
                "sp_penalty_pax": diag.get("penalty_pax"),
                "sp_total_demand": diag.get("total_demand"),
                "sp_slot_resolution": diag.get("slot_resolution"),
            }

        def _make_result(status: SolveStatus, iterations: int) -> BendersRunResult:
            if lp_phase_active:
                # The run never left the LP phase, so no integer-feasible schedule
                # was ever priced. Everything downstream that reads like a solution
                # -- the printed schedule, the served-passenger counts -- belongs to
                # a fractional y. Say so rather than let it read as a result.
                msg = (
                    f"Run ended inside the LP phase after {iterations} iteration(s): no "
                    "integer-feasible schedule was evaluated, so there is no upper "
                    "bound and the printed schedule is fractional. The lower bound is "
                    "valid. Lower master.lp_phase_max_iters or raise "
                    "solver.total_time_limit_s."
                )
                log.warning(msg)
                print(f"\n[LP-PHASE] {msg}")
            extra = _extract_sp_diag(_report_diag())
            _diag = _report_diag()
            _mode = (
                _diag.get("cut_generation_mode") if isinstance(_diag, dict) else None
            )
            return BendersRunResult(
                cut_generation_mode=_mode,
                cut_valid_lower_bound=bool(lower_bound_semantics_valid),
                status=status,
                iterations=iterations,
                best_lower_bound=best_lb,
                best_upper_bound=best_ub,
                pax_served=(
                    best_pax_served if best_pax_served is not None else last_pax_served
                ),
                pax_total=(
                    best_pax_total if best_pax_total is not None else last_pax_total
                ),
                subproblem_obj=extra.get("subproblem_obj"),
                sp_wait_cost_slots=extra.get("sp_wait_cost_slots"),
                sp_fill_eps_cost=extra.get("sp_fill_eps_cost"),
                sp_penalty_cost=extra.get("sp_penalty_cost"),
                sp_penalty_pax=extra.get("sp_penalty_pax"),
                sp_total_demand=extra.get("sp_total_demand"),
                sp_slot_resolution=extra.get("sp_slot_resolution"),
                clock_truncated_master_solves=int(clock_truncated_master_solves),
            )

        for it in range(1, max_it + 1):
            iter_t0 = time.perf_counter()
            rep = IterReport(k=it)
            lb_before_iter = best_lb
            ub_before_iter = best_ub
            if time.time() - t0 > self.cfg.solver.total_time_limit_s:
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
                        _set_master_report_snapshot()
                        fmt = getattr(self.master, "format_solution", None)
                        if callable(fmt):
                            print("\nBest Master Solution:")
                            print(fmt())
                    except Exception:
                        pass
                    # Optional diagnostics from last evaluated subproblem (if available)
                    try:
                        _print_sp_diagnostics(_report_diag())
                    except Exception:
                        pass
                except Exception:
                    pass
                # Summary timing
                try:
                    total_time = (
                        total_master_time
                        + total_sp_time
                        + total_cutgen_time
                        + total_cutadd_time
                    )
                    if total_time > 0 and total_iter > 0:
                        print("\n=== Timing Summary ===")
                        print(
                            "Total: master=%.3fs (%.1f%%) sp_solve=%.3fs (%.1f%%) cutgen=%.3fs (%.1f%%) cutadd=%.3fs (%.1f%%)"
                            % (
                                total_master_time,
                                100.0 * total_master_time / total_time,
                                total_sp_solve_time,
                                100.0 * total_sp_solve_time / total_time,
                                total_cutgen_time,
                                100.0 * total_cutgen_time / total_time,
                                total_cutadd_time,
                                100.0 * total_cutadd_time / total_time,
                            )
                        )
                        print(
                            "Avg/iter: master=%.3fs sp_solve=%.3fs cutgen=%.3fs cutadd=%.3fs"
                            % (
                                total_master_time / total_iter,
                                total_sp_solve_time / total_iter,
                                total_cutgen_time / total_iter,
                                total_cutadd_time / total_iter,
                            )
                        )
                except Exception:
                    pass
                return _make_result(SolveStatus.UNKNOWN, it - 1)

            _vprint(f"\n=== Iteration {it} ===")
            _vprint("Solving Master (MP)...")
            try:
                if isinstance(getattr(self.master, "params", None), dict):
                    self.master.params["iteration"] = int(it)
            except Exception:
                pass
            # Tightening phase: if no new cuts are being added, solve MP harder to raise LB
            if no_cut_streak >= 2:
                try:
                    if isinstance(getattr(self.master, "params", None), dict):
                        cur_tl = float(
                            self.master.params.get("solve_time_limit_s", 0) or 0.0
                        )
                        cur_gap = self.master.params.get("mipgap")
                        cur_gap = float(cur_gap) if cur_gap is not None else 1.0
                        if no_cut_streak >= 3:
                            tl = max(cur_tl, 60.0)
                            mg = min(cur_gap, 0.001)
                        else:
                            tl = max(cur_tl, 30.0)
                            mg = min(cur_gap, 0.01)
                        self.master.params["solve_time_limit_s"] = tl
                        self.master.params["mipgap"] = mg
                        _vprint(
                            f"[TIGHTEN] no new cuts for {no_cut_streak} iters: time_limit_s={tl:.0f} mipgap={mg:.3g}"
                        )
                except Exception:
                    pass
            # Dynamic master solve schedule based on current global BD gap
            if no_cut_streak < 2:
                try:
                    g_bd = None
                    if (
                        lower_bound_semantics_valid
                        and best_lb is not None
                        and best_ub is not None
                    ):
                        gap = float(best_ub) - float(best_lb)
                        if gap < 0.0:
                            gap = 0.0
                        g_bd = gap / max(1.0, abs(float(best_ub)))
                    if g_bd is None:
                        g_bd = float(mp_gap_max)
                    mp_gap = min(
                        float(mp_gap_max),
                        max(float(mp_gap_min), float(mp_gap_scale) * float(g_bd)),
                    )
                    mp_tl = float(mp_time_base) + (
                        float(mp_time_k) / max(float(mp_gap), 1e-4)
                    )
                    if mp_tl_cap is not None:
                        mp_tl = min(mp_tl, mp_tl_cap)
                    # The scheduled value depends only on the gap, so this line is
                    # reproducible and the baseline log-diff regression (D9) stays valid.
                    _vprint(
                        f"[SCHEDULE] master gap-tied: time_limit_s={mp_tl:.0f} mipgap={mp_gap:.2g} g_bd={g_bd:.3g}"
                    )
                    # Never let one master solve outlast the run budget it sits inside.
                    # This clamp reads the wall clock, so it is NOT reproducible -- log it
                    # only when it actually bites, where its presence is itself the signal
                    # that the run became machine-dependent.
                    remaining = float(self.cfg.solver.total_time_limit_s) - (
                        time.time() - t0
                    )
                    if remaining < mp_tl:
                        mp_tl = max(1.0, remaining)
                        _vprint(
                            f"[SCHEDULE] budget clamp: time_limit_s={mp_tl:.0f} "
                            "(run budget nearly spent; this run is not reproducible)"
                        )
                    if isinstance(getattr(self.master, "params", None), dict):
                        self.master.params["solve_time_limit_s"] = mp_tl
                        self.master.params["mipgap"] = mp_gap
                except Exception:
                    pass
            try:
                if isinstance(getattr(self.master, "params", None), dict):
                    self.master.params["lp_relaxation"] = bool(lp_phase_active)
            except Exception:
                pass
            mp_t0 = time.perf_counter()
            mres = self.master.solve()
            mp_t1 = time.perf_counter()
            mp_time = mp_t1 - mp_t0
            total_master_time += mp_time
            log.info(
                "iter=%d master status=%s obj=%s lb=%s",
                it,
                mres.status,
                f"{mres.objective:.6g}" if mres.objective is not None else None,
                f"{mres.lower_bound:.6g}" if mres.lower_bound is not None else None,
            )
            _vprint(
                "MP result: status=%s obj=%s lb=%s"
                % (
                    mres.status,
                    (f"{mres.objective:.6g}" if mres.objective is not None else "-"),
                    (
                        f"{mres.lower_bound:.6g}"
                        if mres.lower_bound is not None
                        else "-"
                    ),
                )
            )
            # [CHECK] MP decomposition at incumbent
            try:
                fcost_fn = getattr(self.master, "first_stage_cost", None)
                first_stage_cost_inc = (
                    float(fcost_fn(mres.candidate))
                    if callable(fcost_fn) and mres.candidate
                    else 0.0
                )
            except Exception:
                first_stage_cost_inc = 0.0
            theta_inc = None
            theta_out_inc = None
            theta_ret_inc = None
            try:

                def _safe_value(v) -> float | None:
                    try:
                        return pyo.value(v, exception=False)
                    except Exception:
                        return None

                def _read_scalar_or_indexed(v) -> float | None:
                    try:
                        if hasattr(v, "is_indexed") and v.is_indexed():
                            if None in v:
                                return _safe_value(v[None])
                            if len(v) == 1:
                                return _safe_value(next(iter(v.values())))
                            return None
                        return _safe_value(v)
                    except Exception:
                        return None

                m = getattr(self.master, "m", None)
                if m is not None:
                    if hasattr(m, "theta"):
                        theta_inc = _read_scalar_or_indexed(m.theta)
                    if hasattr(m, "theta_out"):
                        theta_out_inc = _read_scalar_or_indexed(m.theta_out)
                    if hasattr(m, "theta_ret"):
                        theta_ret_inc = _read_scalar_or_indexed(m.theta_ret)
                    if (theta_inc is None) and hasattr(m, "theta_s"):
                        try:
                            vals = []
                            for s in m.theta_s:
                                v = _safe_value(m.theta_s[s])
                                if v is not None:
                                    vals.append(float(v))
                            if vals:
                                theta_inc = float(sum(vals))
                        except Exception:
                            pass

                # Prefer a master helper if it can add extra fields, but do not overwrite model values
                get_theta = getattr(self.master, "get_theta_values", None)
                if callable(get_theta):
                    try:
                        tv = get_theta()
                    except Exception:
                        tv = None
                    if isinstance(tv, dict):
                        if theta_inc is None and tv.get("theta") is not None:
                            theta_inc = float(tv.get("theta"))
                        if theta_out_inc is None and tv.get("theta_out") is not None:
                            theta_out_inc = float(tv.get("theta_out"))
                        if theta_ret_inc is None and tv.get("theta_ret") is not None:
                            theta_ret_inc = float(tv.get("theta_ret"))
            except Exception:
                pass
            if theta_inc is None:
                # Fall back to sum of theta_out/ret if present
                if theta_out_inc is not None or theta_ret_inc is not None:
                    theta_inc = float(theta_out_inc or 0.0) + float(
                        theta_ret_inc or 0.0
                    )
                else:
                    theta_inc = 0.0
            mp_inc_obj = float(mres.objective) if mres.objective is not None else None
            theta_read = float(theta_inc)
            mp_total_inc = float(first_stage_cost_inc) + float(theta_inc)
            # If theta readback is missing or inconsistent, align with objective for checking
            if mp_inc_obj is not None and abs(mp_total_inc - mp_inc_obj) > 1e-6:
                # Derive implied theta from objective (diagnostic only)
                theta_implied = float(mp_inc_obj) - float(first_stage_cost_inc)
                log.warning(
                    "[CHECK WARN] MP theta mismatch: first=%.6g theta_read=%.6g obj=%.6g -> theta_implied=%.6g",
                    float(first_stage_cost_inc),
                    float(theta_inc),
                    float(mp_inc_obj),
                    float(theta_implied),
                )
                theta_inc = float(theta_implied)
                mp_total_inc = float(first_stage_cost_inc) + float(theta_inc)
                log.warning(
                    "[CHECK WARN] theta vars: theta=%.6g theta_out=%s theta_ret=%s",
                    float(theta_inc),
                    (
                        f"{float(theta_out_inc):.6g}"
                        if theta_out_inc is not None
                        else "-"
                    ),
                    (
                        f"{float(theta_ret_inc):.6g}"
                        if theta_ret_inc is not None
                        else "-"
                    ),
                )
                if abs(float(theta_read) - float(theta_implied)) > 1e-3:
                    log.warning(
                        "[CHECK WARN] theta_read differs from implied by >1e-3; verify MP theta vars."
                    )
            # Best bound from solver stats if available
            mp_best_bound = None
            mp_best_integer = None
            mp_term = None
            mp_status = None
            mp_incumbent = None
            mp_gap = None
            bb_source = None
            inc_source = None
            gap_source = None
            bb_reason = None
            try:
                stats = getattr(self.master, "last_solve_stats", lambda: {})()
                mp_best_bound = stats.get("best_bound")
                mp_best_integer = stats.get("best_integer")
                mp_incumbent = stats.get("incumbent")
                mp_gap = stats.get("gap")
                mp_term = stats.get("termination_condition")
                mp_status = stats.get("status")
                bb_source = stats.get("best_bound_source")
                inc_source = stats.get("incumbent_source")
                gap_source = stats.get("gap_source")
                bb_reason = stats.get("best_bound_reason")
            except Exception:
                pass
            if mp_incumbent is None:
                mp_incumbent = mp_best_integer
            if (
                mp_gap is None
                and mp_incumbent is not None
                and mp_best_bound is not None
            ):
                try:
                    gap_val = (float(mp_incumbent) - float(mp_best_bound)) / max(
                        1.0, abs(float(mp_incumbent))
                    )
                    if gap_val < 0.0:
                        gap_val = 0.0
                    mp_gap = gap_val
                    gap_source = "computed"
                except Exception:
                    pass
            _vprint(
                "[CHECK] MP: first=%.6g theta=%.6g mp_total=%.6g inc_obj=%s best_bound=%s"
                % (
                    float(first_stage_cost_inc),
                    float(theta_inc),
                    float(mp_total_inc),
                    (f"{mp_inc_obj:.6g}" if mp_inc_obj is not None else "-"),
                    (
                        f"{float(mp_best_bound):.6g}"
                        if mp_best_bound is not None
                        else "-"
                    ),
                )
            )
            _vprint(
                "[CHECK] MP bounds: mp_obj=%s mp_best_bound=%s mp_best_integer=%s term=%s status=%s gap=%s"
                % (
                    (f"{mp_inc_obj:.6g}" if mp_inc_obj is not None else "-"),
                    (
                        f"{float(mp_best_bound):.6g}"
                        if mp_best_bound is not None
                        else "-"
                    ),
                    (
                        f"{float(mp_best_integer):.6g}"
                        if mp_best_integer is not None
                        else "-"
                    ),
                    (str(mp_term) if mp_term is not None else "-"),
                    (str(mp_status) if mp_status is not None else "-"),
                    (f"{float(mp_gap):.6g}" if mp_gap is not None else "-"),
                )
            )
            try:
                last_mp_gap = float(mp_gap) if mp_gap is not None else None
            except Exception:
                last_mp_gap = None
            last_mp_term = str(mp_term) if mp_term is not None else None
            # A master solve that stops on the clock rather than on the gap makes the
            # whole run machine-dependent: how many nodes it explored in those seconds
            # depends on machine load, and every later iteration inherits the difference.
            # configs/baseline_d9.yaml states this as a determinism requirement; a run
            # that breaks it must say so rather than look reproducible.
            if (
                last_mp_term is not None
                and "maxtimelimit" in last_mp_term.lower().replace("_", "")
            ):
                clock_truncated_master_solves += 1
            _vprint(
                "[MP] iter=%d incumbent=%s best_bound=%s gap=%s (src: inc=%s bb=%s gap=%s)"
                % (
                    it,
                    (f"{float(mp_incumbent):.6g}" if mp_incumbent is not None else "-"),
                    (
                        f"{float(mp_best_bound):.6g}"
                        if mp_best_bound is not None
                        else "-"
                    ),
                    (f"{float(mp_gap):.6g}" if mp_gap is not None else "-"),
                    (str(inc_source) if inc_source is not None else "-"),
                    (str(bb_source) if bb_source is not None else "-"),
                    (str(gap_source) if gap_source is not None else "-"),
                )
            )

            try:
                stats_nodes = stats.get("nodes") if isinstance(stats, dict) else None
            except Exception:
                stats_nodes = None
            rep.t_master_s = mp_time
            rep.mp_status = str(mp_status) if mp_status is not None else None
            rep.mp_obj = mp_inc_obj
            rep.mp_best_bound = (
                float(mp_best_bound) if mp_best_bound is not None else None
            )
            rep.mp_gap_pct = (float(mp_gap) * 100.0) if mp_gap is not None else None
            rep.mp_nodes = stats_nodes

            # Instrumentation: per-iteration summary
            try:
                cuts_in_model = getattr(self.master, "cuts_count", lambda: None)()
            except Exception:
                cuts_in_model = None
            last_const, last_nnz = (None, None)
            try:
                last_const, last_nnz = getattr(
                    self.master, "last_cut_info", lambda: (None, None)
                )()
            except Exception:
                pass
            if mres.objective is not None:
                _vprint(
                    f"[MP] iter={it} cuts={cuts_in_model if cuts_in_model is not None else '-'} obj={mres.objective:.3f}"
                    + (
                        f" last_const={last_const:.3f} last_nnz={last_nnz}"
                        if last_const is not None
                        else ""
                    )
                )

            # (Diagnostic only) We no longer assert binding; the MP is free to change y

            if mres.status == SolveStatus.INFEASIBLE:
                log.error("Master problem infeasible")
                return _make_result(mres.status, it)

            # Update lower bound if provided
            # Prefer solver best bound when available
            if mp_best_bound is None:
                try:
                    log_path = getattr(
                        self.master, "last_solver_log_path", lambda: None
                    )()
                except Exception:
                    log_path = None
                parsed = parse_cplex_log_bounds(log_path)
                if parsed.get("best_bound") is not None:
                    mp_best_bound = parsed.get("best_bound")
                    bb_source = "cplex_log"
                elif bb_reason is None:
                    bb_reason = "not_in_solver_or_log"
            if lower_bound_semantics_valid and mp_best_bound is not None:
                best_lb = (
                    float(mp_best_bound)
                    if best_lb is None
                    else max(best_lb, float(mp_best_bound))
                )
            elif lower_bound_semantics_valid:
                reason = f" reason={bb_reason}" if bb_reason is not None else ""
                log.warning(
                    "[CHECK] MP best bound unavailable; LB not updated.%s", reason
                )
                _vprint(f"[MP] best_bound unavailable{reason}")
            if mp_best_bound is not None:
                rep.mp_best_bound = float(mp_best_bound)

            if not mres.candidate:
                log.error("Master did not return a candidate solution")
                _vprint(
                    "Master did not return a usable incumbent; skipping SP and increasing MP time limit."
                )
                try:
                    if isinstance(getattr(self.master, "params", None), dict):
                        cur_tl = float(
                            self.master.params.get("solve_time_limit_s", 0) or 0
                        )
                        self.master.params["solve_time_limit_s"] = max(cur_tl, 10.0)
                except Exception:
                    pass
                try:
                    rep.cuts_added = 0
                    rep.cuts_total = getattr(self.master, "cuts_count", lambda: None)()
                except Exception:
                    rep.cuts_total = None
                rep.best_lb = best_lb
                rep.best_ub = best_ub
                rep.t_iter_s = time.perf_counter() - iter_t0
                if report_mode:
                    print(format_report_line(rep))
                continue

            # Optional: update and pass a core point for Magnanti–Wong selection
            if mw_enabled:
                # Determine Q and T safely
                Qn = None
                Tn = None
                try:
                    m = getattr(self.master, "m", None)
                    if m is not None:
                        try:
                            Qn = len(list(m.Q))
                        except Exception:
                            Qn = None
                        try:
                            Tn = len(list(m.T))
                        except Exception:
                            Tn = None
                except Exception:
                    pass
                if Qn is None:
                    try:
                        Qn = int(getattr(self.master, "params", {}).get("Q"))
                    except Exception:
                        Qn = None
                if Tn is None:
                    try:
                        Tn = int(getattr(self.master, "params", {}).get("T"))
                    except Exception:
                        Tn = None

                # Initialize neutral interior core if missing
                if self._mw_core_out is None or self._mw_core_ret is None:
                    if Qn is not None and Tn is not None:
                        eps = float(mw_eps)
                        cap = float(Qn) - eps
                        seed = min(max(float(Qn) / 2.0, eps), cap)
                        self._mw_core_out = [seed for _ in range(Tn)]
                        self._mw_core_ret = [seed for _ in range(Tn)]

                # Update core from current incumbent (moving average)
                if mres.candidate and Qn is not None and Tn is not None:
                    cur_out = [0.0 for _ in range(Tn)]
                    cur_ret = [0.0 for _ in range(Tn)]
                    for name, val in mres.candidate.items():
                        if not isinstance(name, str):
                            continue
                        try:
                            if name.startswith("yOUT["):
                                inside = name[name.find("[") + 1 : name.find("]")]
                                _, t_str = inside.split(",")
                                t = int(t_str.strip())
                                if 0 <= t < Tn:
                                    cur_out[t] += float(val)
                            elif name.startswith("yRET["):
                                inside = name[name.find("[") + 1 : name.find("]")]
                                _, t_str = inside.split(",")
                                t = int(t_str.strip())
                                if 0 <= t < Tn:
                                    cur_ret[t] += float(val)
                        except Exception:
                            continue
                    if self._mw_core_out is None or self._mw_core_ret is None:
                        self._mw_core_out = list(cur_out)
                        self._mw_core_ret = list(cur_ret)
                    else:
                        for t in range(Tn):
                            self._mw_core_out[t] = (1.0 - mw_alpha) * float(
                                self._mw_core_out[t]
                            ) + mw_alpha * float(cur_out[t])
                            self._mw_core_ret[t] = (1.0 - mw_alpha) * float(
                                self._mw_core_ret[t]
                            ) + mw_alpha * float(cur_ret[t])
                    # Keep strictly interior
                    eps = float(mw_eps)
                    cap = float(Qn) - eps
                    for t in range(Tn):
                        self._mw_core_out[t] = min(
                            max(float(self._mw_core_out[t]), eps), cap
                        )
                        self._mw_core_ret[t] = min(
                            max(float(self._mw_core_ret[t]), eps), cap
                        )
                    try:
                        vals = list(self._mw_core_out or []) + list(
                            self._mw_core_ret or []
                        )
                        if vals:
                            vmin = min(vals)
                            vmax = max(vals)
                            vmean = sum(vals) / float(len(vals))
                            _vprint(
                                f"[MW] core updated (t=0..): min={vmin:.3g} max={vmax:.3g} mean={vmean:.3g}"
                            )
                    except Exception:
                        pass

                # Attach to subproblem params for MW dual selection
                try:
                    if isinstance(getattr(self.subproblem, "params", None), dict):
                        if (
                            self._mw_core_out is not None
                            and self._mw_core_ret is not None
                        ):
                            self.subproblem.params["mw_core_point"] = {
                                "Yout": list(self._mw_core_out),
                                "Yret": list(self._mw_core_ret),
                            }
                        self.subproblem.params["use_magnanti_wong"] = mw_enabled
                except Exception:
                    pass
            else:
                try:
                    if isinstance(getattr(self.subproblem, "params", None), dict):
                        pass
                except Exception:
                    pass

            try:
                if isinstance(getattr(self.subproblem, "params", None), dict):
                    self.subproblem.params["debug_current_iteration"] = int(it)
                    remaining_time = max(
                        1.0,
                        float(self.cfg.solver.total_time_limit_s) - (time.time() - t0),
                    )
                    self.subproblem.params["solve_time_limit_s"] = float(remaining_time)
            except Exception:
                pass

            _vprint("Evaluating Subproblem (SP) at candidate...")
            sp_t0 = time.perf_counter()
            sres: SubproblemResult = self.subproblem.evaluate(mres.candidate)
            sp_t1 = time.perf_counter()
            sp_time = sp_t1 - sp_t0
            # Update UB if provided (include first-stage cost from master to compare totals)
            sp_total_obj = None
            sp_recourse_obj = None
            sp_first_stage = None
            is_new_best_incumbent = False
            if sres.upper_bound is not None:
                try:
                    fcost_fn = getattr(self.master, "first_stage_cost", None)
                    fcost = (
                        float(fcost_fn(mres.candidate)) if callable(fcost_fn) else 0.0
                    )
                except Exception:
                    fcost = 0.0
                sp_recourse_obj = float(sres.upper_bound)
                sp_first_stage = float(fcost)
                sp_total_obj = float(sp_recourse_obj) + float(sp_first_stage)
                if lp_phase_active:
                    # The candidate is fractional. Its recourse cost is a real number
                    # for a schedule nobody can operate, so it is not an upper bound
                    # on anything and must not become the incumbent. The cut derived
                    # from the same solve is still valid and is still added below.
                    is_new_best_incumbent = False
                else:
                    prev_ub = best_ub
                    is_new_best_incumbent = (
                        prev_ub is None or float(sp_total_obj) < float(prev_ub) - 1e-9
                    )
                    # Use sp_total_obj consistently for UB updates
                    best_ub = (
                        sp_total_obj if best_ub is None else min(best_ub, sp_total_obj)
                    )
            # Keep last diagnostics for end-of-run reporting
            try:
                last_diag = dict(getattr(sres, "diagnostics", {}) or {})
            except Exception:
                last_diag = None
            try:
                current_cut_lb_valid = (
                    bool(last_diag.get("cut_valid_lower_bound", True))
                    if isinstance(last_diag, dict)
                    else True
                )
            except Exception:
                current_cut_lb_valid = True
            if not current_cut_lb_valid:
                lower_bound_semantics_valid = False
                best_lb = None
            rep.sp_feasible = (
                bool(sres.is_feasible) if sres.is_feasible is not None else None
            )
            rep.sp_recourse = sp_recourse_obj
            # Reporting a candidate UB for a fractional schedule would put a number
            # in the UB column that no schedule achieves.
            rep.candidate_ub = None if lp_phase_active else sp_total_obj
            try:
                served, total = _calc_pax_totals_from_diag(last_diag)
            except Exception:
                served, total = (None, None)
            rep.pax_served = served
            rep.pax_total = total
            if sres.upper_bound is not None and is_new_best_incumbent:
                try:
                    best_candidate = copy.deepcopy(dict(mres.candidate))
                except Exception:
                    best_candidate = dict(mres.candidate)
                try:
                    best_diag = (
                        copy.deepcopy(last_diag)
                        if isinstance(last_diag, dict)
                        else None
                    )
                except Exception:
                    best_diag = dict(last_diag) if isinstance(last_diag, dict) else None
                best_pax_served = served
                best_pax_total = total
            try:
                pen_pax = (
                    last_diag.get("penalty_pax")
                    if isinstance(last_diag, dict)
                    else None
                )
                rep.penalty_pax = float(pen_pax) if pen_pax is not None else None
            except Exception:
                rep.penalty_pax = None
            try:
                wait_slots = (
                    last_diag.get("waiting_cost_slots")
                    if isinstance(last_diag, dict)
                    else None
                )
                slot_res = (
                    last_diag.get("slot_resolution")
                    if isinstance(last_diag, dict)
                    else None
                )
                if wait_slots is not None and rep.pax_served and rep.pax_served > 0:
                    res = int(slot_res or 1)
                    rep.avg_wait_min = (float(wait_slots) * float(res)) / float(
                        rep.pax_served
                    )
            except Exception:
                rep.avg_wait_min = None
            try:
                _ub_print = (
                    f"{sres.upper_bound:.6g}" if sres.upper_bound is not None else "-"
                )
            except Exception:
                _ub_print = "-"
            # Clarify that SP result is recourse-only
            _vprint(f"SP result: recourse={_ub_print} feasible={sres.is_feasible}")
            try:
                if isinstance(last_diag, dict):
                    _vprint(
                        "[SP PIPELINE] iter=%s build=%.3fs solve=%.3fs extract=%.3fs post=%.3fs cutgen=%.3fs mode=%s lb_valid=%s tl_inc=%s vars=%s bin=%s cons=%s lp=%s"
                        % (
                            str(it),
                            float(last_diag.get("timing_build_s", 0.0) or 0.0),
                            float(last_diag.get("timing_solve_s", 0.0) or 0.0),
                            float(last_diag.get("timing_extract_s", 0.0) or 0.0),
                            float(last_diag.get("timing_postprocess_s", 0.0) or 0.0),
                            float(last_diag.get("timing_cutgen_s", 0.0) or 0.0),
                            str(last_diag.get("cut_generation_mode", "-")),
                            str(bool(last_diag.get("cut_valid_lower_bound", True))),
                            str(bool(last_diag.get("time_limited_incumbent", False))),
                            str(last_diag.get("model_num_variables", "-")),
                            str(last_diag.get("model_num_binary_variables", "-")),
                            str(last_diag.get("model_num_constraints", "-")),
                            str(last_diag.get("exported_lp_path", "-")),
                        )
                    )
            except Exception:
                pass
            if sp_recourse_obj is not None:
                _vprint(
                    "[CHECK] SP: recourse=%.6g first=%.6g sp_total=%.6g"
                    % (
                        float(sp_recourse_obj),
                        float(sp_first_stage or 0.0),
                        float(sp_total_obj or 0.0),
                    )
                )
            # Correctness checks (warnings only)
            try:
                if (
                    lower_bound_semantics_valid
                    and mp_total_inc is not None
                    and sp_total_obj is not None
                ):
                    if float(mp_total_inc) > float(sp_total_obj) + 1e-6:
                        log.warning(
                            "[CHECK FAIL] MP total exceeds SP total at same y: mp_total=%.6g sp_total=%.6g (objective mismatch or theta overestimates)",
                            float(mp_total_inc),
                            float(sp_total_obj),
                        )
                if (
                    lower_bound_semantics_valid
                    and best_lb is not None
                    and best_ub is not None
                ):
                    if float(best_lb) > float(best_ub) + 1e-6:
                        log.warning(
                            "[CHECK FAIL] LB exceeds UB: LB=%.6g UB=%.6g",
                            float(best_lb),
                            float(best_ub),
                        )
                        # Keep previous valid LB if available
                        if lb_before_iter is not None:
                            best_lb = lb_before_iter
                            log.warning(
                                "[CHECK] LB invalid; reverting to previous LB=%.6g",
                                float(best_lb),
                            )
                        else:
                            best_lb = None
                            log.warning(
                                "[CHECK] LB invalid; cleared LB (no previous valid value)."
                            )
                if (
                    lower_bound_semantics_valid
                    and prev_best_lb is not None
                    and best_lb is not None
                ):
                    if float(best_lb) + 1e-6 < float(prev_best_lb):
                        log.warning(
                            "[CHECK FAIL] LB decreased: prev=%.6g now=%.6g",
                            float(prev_best_lb),
                            float(best_lb),
                        )
                if prev_best_ub is not None and best_ub is not None:
                    if float(best_ub) > float(prev_best_ub) + 1e-6:
                        log.warning(
                            "[CHECK FAIL] UB increased: prev=%.6g now=%.6g",
                            float(prev_best_ub),
                            float(best_ub),
                        )
            except Exception:
                pass
            # Cut tightness check: evaluate line(y) from the raw cut metadata and compare to SP upper bound
            try:
                if (
                    sres.cut is not None
                    and sres.upper_bound is not None
                    and mres.candidate is not None
                ):
                    cmeta = getattr(sres.cut, "metadata", {}) or {}
                    const = float(cmeta.get("const", 0.0))
                    coeff_yout = cmeta.get("coeff_yOUT") or {}
                    coeff_yret = cmeta.get("coeff_yRET") or {}
                    line_val = float(const)

                    # Candidate has keys like 'yOUT[q,t]' and 'yRET[q,t]'
                    def _cand_val(prefix: str, q: int, t: int) -> float:
                        return float(
                            mres.candidate.get(f"{prefix}[{int(q)},{int(t)}]", 0.0)
                        )

                    if isinstance(coeff_yout, dict):
                        for (q, tau), v in coeff_yout.items():
                            line_val += float(v) * _cand_val("yOUT", int(q), int(tau))
                    if isinstance(coeff_yret, dict):
                        for (q, tau), v in coeff_yret.items():
                            line_val += float(v) * _cand_val("yRET", int(q), int(tau))
                    diff = float(line_val) - float(sres.upper_bound)
                    _vprint(
                        f"[CUT TIGHTNESS] raw_line(y)={line_val:.6g}  SP_ub={float(sres.upper_bound):.6g}  diff={diff:.3g}"
                    )
            except Exception:
                pass
            # Suppress repetitive demand printouts; diagnostics still available at the end
            # Add cut(s) if provided (optimality or feasibility)
            cut_t0 = time.perf_counter()
            added = 0
            cut_names: list[str] = []
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
                    log.info(
                        "force-added %s cut '%s'", sres.cut.cut_type, sres.cut.name
                    )

            # If nothing forced yet, try forcing the first in sres.cuts
            if (
                (not forced_added)
                and getattr(sres, "cuts", None)
                and hasattr(self.master, "add_cut_force")
            ):
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
                _vprint(f"Master updated: added {added} cut(s): {names_str}")
                no_cut_streak = 0
            else:
                if cut_names:
                    _vprint("Master updated: no new cuts (all skipped / duplicates)")
                    try:
                        attempt = getattr(
                            self.master, "last_cut_attempt", lambda: None
                        )()
                    except Exception:
                        attempt = None
                    if isinstance(attempt, dict):
                        _vprint(
                            "[CUT ATTEMPT] it=%s cand=%s theta_out=%s theta_ret=%s theta_total=%s rec_out=%s rec_ret=%s rec_total=%s raw_rhs_out=%s raw_rhs_ret=%s raw_rhs_total=%s raw_gap=%s lhs_out=%s lhs_ret=%s lhs_total=%s rhs_out=%s rhs_ret=%s rhs_total=%s tol_master=%s tol_filter=%s skip=%s"
                            % (
                                str(attempt.get("iteration")),
                                str(attempt.get("candidate_id")),
                                str(attempt.get("theta_out")),
                                str(attempt.get("theta_ret")),
                                str(attempt.get("theta_total")),
                                str(attempt.get("recourse_out")),
                                str(attempt.get("recourse_ret")),
                                str(attempt.get("recourse_total")),
                                str(attempt.get("raw_rhs_out")),
                                str(attempt.get("raw_rhs_ret")),
                                str(attempt.get("raw_rhs_total")),
                                str(attempt.get("aggregate_cut_logging_gap")),
                                str(attempt.get("lhs_out")),
                                str(attempt.get("lhs_ret")),
                                str(attempt.get("lhs_total")),
                                str(
                                    attempt.get(
                                        "transformed_rhs_out_after_scaling",
                                        attempt.get("transformed_rhs_after_scaling"),
                                    )
                                ),
                                str(attempt.get("transformed_rhs_ret_after_scaling")),
                                str(
                                    attempt.get(
                                        "transformed_rhs_total_after_scaling",
                                        attempt.get("transformed_rhs_after_scaling"),
                                    )
                                ),
                                str(attempt.get("violation_tolerance_master")),
                                str(attempt.get("violation_tolerance_filter_rel")),
                                str(attempt.get("skip_reason")),
                            )
                        )
                        if (
                            attempt.get("filter_out") is not None
                            or attempt.get("filter_ret") is not None
                            or attempt.get("filter_total") is not None
                        ):
                            _vprint(
                                "[CUT FILTER] out=%s ret=%s total=%s"
                                % (
                                    str(attempt.get("filter_out")),
                                    str(attempt.get("filter_ret")),
                                    str(attempt.get("filter_total")),
                                )
                            )
                        if attempt.get("mismatch_explanation"):
                            log.warning(
                                "[CUT MISMATCH] it=%s cand=%s raw_gap=%s skip_reason=%s explanation=%s",
                                str(attempt.get("iteration")),
                                str(attempt.get("candidate_id")),
                                str(attempt.get("aggregate_cut_logging_gap")),
                                str(attempt.get("skip_reason")),
                                str(attempt.get("mismatch_explanation")),
                            )
                no_cut_streak += 1
            try:
                attempt = getattr(self.master, "last_cut_attempt", lambda: None)()
            except Exception:
                attempt = None
            if isinstance(attempt, dict) and cut_names:
                _vprint(
                    "[CUT SUMMARY] it=%s cand=%s decision=%s skip=%s theta_total=%s rec_total=%s raw_rhs_total=%s rhs_total=%s lhs_total=%s"
                    % (
                        str(attempt.get("iteration")),
                        str(attempt.get("candidate_id")),
                        str(attempt.get("decision")),
                        str(attempt.get("skip_reason")),
                        str(attempt.get("theta_total")),
                        str(attempt.get("recourse_total")),
                        str(attempt.get("raw_rhs_total")),
                        str(
                            attempt.get(
                                "transformed_rhs_total_after_scaling",
                                attempt.get("transformed_rhs_after_scaling"),
                            )
                        ),
                        str(attempt.get("lhs_total")),
                    )
                )
            rep.cuts_added = added
            rep.cuts_total = cuts_after
            if added > 0:
                try:
                    last_cut_type, last_cut_nnz = getattr(
                        self.master, "last_cut_meta", lambda: (None, None)
                    )()
                except Exception:
                    last_cut_type, last_cut_nnz = (None, None)
                rep.last_cut_type = last_cut_type
                rep.last_cut_nnz = last_cut_nnz
            # Invariant: first iteration with finite SP UB must add at least one cut
            if (
                it == 1
                and (sres.upper_bound is not None)
                and (mres.candidate is not None)
                and int(getattr(self.master, "cuts_count", lambda: 0)()) == 0
            ):
                cmeta = (
                    getattr(sres.cut, "metadata", {}) or {}
                    if sres.cut is not None
                    else {}
                )
                const = cmeta.get("const") if isinstance(cmeta, dict) else None
                coeff_yout = (
                    cmeta.get("coeff_yOUT") if isinstance(cmeta, dict) else None
                )
                coeff_yret = (
                    cmeta.get("coeff_yRET") if isinstance(cmeta, dict) else None
                )
                nnz = 0
                try:
                    nnz += len(
                        [v for v in (coeff_yout or {}).values() if abs(float(v)) > 0]
                    )
                except Exception:
                    pass
                try:
                    nnz += len(
                        [v for v in (coeff_yret or {}).values() if abs(float(v)) > 0]
                    )
                except Exception:
                    pass
                raise RuntimeError(
                    "Invariant failed: SP UB is finite but no cut was added in iteration 1. "
                    f"const={const} nnz={nnz} cut_names={cut_names}."
                )
            cut_t1 = time.perf_counter()
            cut_add_time = cut_t1 - cut_t0
            total_cutadd_time += cut_add_time

            sp_solve_time = None
            cutgen_time = None
            if isinstance(last_diag, dict):
                sp_solve_time = last_diag.get("timing_sp_solve_s")
                cutgen_time = last_diag.get("timing_cutgen_s")
            if sp_solve_time is None:
                sp_solve_time = sp_time
            if cutgen_time is None:
                cutgen_time = 0.0
            total_sp_solve_time += float(sp_solve_time)
            total_cutgen_time += float(cutgen_time)

            total_iter += 1
            rep.t_sp_solve_s = (
                float(sp_solve_time) if sp_solve_time is not None else None
            )
            rep.t_cutgen_s = float(cutgen_time) if cutgen_time is not None else None

            # Per-iteration timing + stats
            stats = {}
            try:
                stats = getattr(self.master, "last_solve_stats", lambda: {})()
            except Exception:
                stats = {}
            nodes = stats.get("nodes")
            gap = stats.get("gap")
            best_bound = stats.get("best_bound")
            incumbent = stats.get("incumbent")
            _vprint(
                "Timing: master=%.3fs sp_solve=%.3fs cutgen=%.3fs cutadd=%.3fs cuts_added=%s total_cuts=%s"
                % (
                    mp_time,
                    float(sp_solve_time),
                    float(cutgen_time),
                    cut_add_time,
                    added,
                    getattr(self.master, "cuts_count", lambda: "-")(),
                )
            )
            if (
                nodes is not None
                or best_bound is not None
                or incumbent is not None
                or gap is not None
            ):
                _vprint(
                    "MP stats: nodes=%s best_bound=%s incumbent=%s gap=%s"
                    % (
                        str(nodes),
                        str(best_bound),
                        str(incumbent),
                        str(gap),
                    )
                )

            rep.best_lb = best_lb
            rep.best_ub = best_ub
            rep.t_iter_s = time.perf_counter() - iter_t0
            if report_mode:
                print(format_report_line(rep))

            # Decide whether the LP phase has stopped paying. Evaluated at the end
            # of the iteration so that the candidate just priced, the cut just
            # added and the bound just recorded all belong to the same regime.
            if lp_phase_active:
                lp_phase_iters += 1
                cur_obj = float(mres.objective) if mres.objective is not None else None
                rel_improve = None
                if cur_obj is not None and lp_phase_prev_obj is not None:
                    rel_improve = (cur_obj - float(lp_phase_prev_obj)) / max(
                        1.0, abs(cur_obj)
                    )
                    if rel_improve < lp_phase_min_rel_improve:
                        lp_phase_stall_ctr += 1
                    else:
                        lp_phase_stall_ctr = 0
                lp_phase_prev_obj = cur_obj
                if added <= 0:
                    lp_phase_exit_reason = "no cut generated"
                elif lp_phase_iters >= lp_phase_max_iters:
                    lp_phase_exit_reason = f"iteration budget ({lp_phase_max_iters})"
                elif (
                    lp_phase_stall_iters > 0
                    and lp_phase_stall_ctr >= lp_phase_stall_iters
                ):
                    lp_phase_exit_reason = (
                        f"LP objective stalled for {lp_phase_stall_ctr} iters "
                        f"(last rel improve {rel_improve:.3g})"
                        if rel_improve is not None
                        else f"LP objective stalled for {lp_phase_stall_ctr} iters"
                    )
                _vprint(
                    "[LP-PHASE] it=%d obj=%s rel_improve=%s stall=%d/%d cuts=%d"
                    % (
                        it,
                        (f"{cur_obj:.6g}" if cur_obj is not None else "-"),
                        (f"{rel_improve:.3g}" if rel_improve is not None else "-"),
                        lp_phase_stall_ctr,
                        lp_phase_stall_iters,
                        int(added),
                    )
                )
                if lp_phase_exit_reason is not None:
                    lp_phase_active = False
                    print(
                        f"[LP-PHASE] off after {lp_phase_iters} iteration(s): {lp_phase_exit_reason}. "
                        f"{int(getattr(self.master, 'cuts_count', lambda: 0)())} cut(s) carried into the MIP phase. "
                        "No upper bound was claimed while it was on."
                    )

            # Check gap if we have both bounds
            if (
                lower_bound_semantics_valid
                and best_lb is not None
                and best_ub is not None
            ):
                gap = float(best_ub) - float(best_lb)
                if gap < 0.0:
                    gap = 0.0
                rel_gap = gap / max(1.0, abs(best_ub))
                log.info(
                    "bounds: best_lb=%.6g best_ub=%.6g gap=%.6g rel=%.3g",
                    best_lb,
                    best_ub,
                    gap,
                    rel_gap,
                )
                _vprint(
                    f"[CHECK] GAP: LB={best_lb:.6g} UB={best_ub:.6g} gap={gap:.6g} rel={rel_gap:.3g}"
                )
                prev_best_lb = best_lb
                prev_best_ub = best_ub
                if gap <= (tol * max(1.0, abs(best_ub))):
                    log.info(
                        "Optimality reached within tolerance after %d iterations", it
                    )
                    elapsed = time.time() - t0
                    print(
                        f"\nOptimality reached within tolerance after {it} iterations."
                    )
                    print(f"Total solve time: {elapsed:.3f} seconds")
                    # If the master has a formatter for the current solution, print it
                    try:
                        _set_master_report_snapshot()
                        fmt = getattr(self.master, "format_solution", None)
                        if callable(fmt):
                            print("\nBest Master Solution:")
                            print(fmt())
                    except Exception:
                        pass
                    # Additional matrices from subproblem diagnostics (if available)
                    try:
                        _print_sp_diagnostics(_report_diag())
                    except Exception:
                        pass
                    # Summary timing
                    try:
                        total_time = (
                            total_master_time
                            + total_sp_solve_time
                            + total_cutgen_time
                            + total_cutadd_time
                        )
                        if total_time > 0 and total_iter > 0:
                            print("\n=== Timing Summary ===")
                            print(
                                "Total: master=%.3fs (%.1f%%) sp_solve=%.3fs (%.1f%%) cutgen=%.3fs (%.1f%%) cutadd=%.3fs (%.1f%%)"
                                % (
                                    total_master_time,
                                    100.0 * total_master_time / total_time,
                                    total_sp_solve_time,
                                    100.0 * total_sp_solve_time / total_time,
                                    total_cutgen_time,
                                    100.0 * total_cutgen_time / total_time,
                                    total_cutadd_time,
                                    100.0 * total_cutadd_time / total_time,
                                )
                            )
                            print(
                                "Avg/iter: master=%.3fs sp_solve=%.3fs cutgen=%.3fs cutadd=%.3fs"
                                % (
                                    total_master_time / total_iter,
                                    total_sp_solve_time / total_iter,
                                    total_cutgen_time / total_iter,
                                    total_cutadd_time / total_iter,
                                )
                            )
                    except Exception:
                        pass
                        print(
                            "Avg/iter: master=%.3fs sp_solve=%.3fs cutgen=%.3fs cutadd=%.3fs"
                            % (
                                total_master_time / total_iter,
                                total_sp_solve_time / total_iter,
                                total_cutgen_time / total_iter,
                                total_cutadd_time / total_iter,
                            )
                        )
                    except Exception:
                        pass
                    return _make_result(SolveStatus.OPTIMAL, it)
            elif not lower_bound_semantics_valid and best_ub is not None:
                _vprint(
                    "[CHECK] Bounds are heuristic only; LB/gap optimality logic disabled for current cut mode."
                )
                # Stall stopping: if gap does not improve sufficiently for several iterations
                if stall_max > 0:
                    improved = False
                    if prev_gap is None:
                        improved = True
                    else:
                        abs_impr = float(prev_gap - gap)
                        rel_impr = abs_impr / max(1.0, abs(prev_gap))
                        improved = (abs_impr >= max(0.0, stall_min_abs)) or (
                            rel_impr >= max(0.0, stall_min_rel)
                        )
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
                        _set_master_report_snapshot()
                        fmt = getattr(self.master, "format_solution", None)
                        if callable(fmt):
                            print("\nBest Master Solution:")
                            print(fmt())
                        # Optional diagnostics from last evaluated subproblem (if available)
                        try:
                            _print_sp_diagnostics(_report_diag())
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # Summary timing
                    try:
                        total_time = (
                            total_master_time
                            + total_sp_time
                            + total_cutgen_time
                            + total_cutadd_time
                        )
                        if total_time > 0 and total_iter > 0:
                            print("\n=== Timing Summary ===")
                            print(
                                "Total: master=%.3fs (%.1f%%) sp_solve=%.3fs (%.1f%%) cutgen=%.3fs (%.1f%%) cutadd=%.3fs (%.1f%%)"
                                % (
                                    total_master_time,
                                    100.0 * total_master_time / total_time,
                                    total_sp_solve_time,
                                    100.0 * total_sp_solve_time / total_time,
                                    total_cutgen_time,
                                    100.0 * total_cutgen_time / total_time,
                                    total_cutadd_time,
                                    100.0 * total_cutadd_time / total_time,
                                )
                            )
                        print(
                            "Avg/iter: master=%.3fs sp_solve=%.3fs cutgen=%.3fs cutadd=%.3fs"
                            % (
                                total_master_time / total_iter,
                                total_sp_solve_time / total_iter,
                                total_cutgen_time / total_iter,
                                total_cutadd_time / total_iter,
                            )
                        )
                    except Exception:
                        pass
                    return _make_result(SolveStatus.FEASIBLE, it)

        log.warning("Max iterations reached: %d", max_it)
        # Print incumbent solution if available at max-iterations stop
        try:
            elapsed = time.time() - t0
            print(f"\nMax iterations reached: {max_it}")
            print(f"Total solve time: {elapsed:.3f} seconds")
            if best_lb is not None and best_ub is not None:
                gap = abs(best_ub - best_lb)
                rel_gap = gap / max(1.0, abs(best_ub))
                print(
                    f"Best bounds: LB={best_lb:.6g} UB={best_ub:.6g} gap={gap:.6g} rel={rel_gap:.3g}"
                )
            _set_master_report_snapshot()
            fmt = getattr(self.master, "format_solution", None)
            if callable(fmt):
                print("\nBest Master Solution:")
                print(fmt())
            # Diagnostics from last evaluated subproblem, if any
            try:
                _print_sp_diagnostics(_report_diag())
            except Exception:
                pass
        except Exception:
            pass
        # Summary timing
        try:
            total_time = (
                total_master_time
                + total_sp_time
                + total_cutgen_time
                + total_cutadd_time
            )
            if total_time > 0 and total_iter > 0:
                print("\n=== Timing Summary ===")
                print(
                    "Total: master=%.3fs (%.1f%%) sp_solve=%.3fs (%.1f%%) cutgen=%.3fs (%.1f%%) cutadd=%.3fs (%.1f%%)"
                    % (
                        total_master_time,
                        100.0 * total_master_time / total_time,
                        total_sp_solve_time,
                        100.0 * total_sp_solve_time / total_time,
                        total_cutgen_time,
                        100.0 * total_cutgen_time / total_time,
                        total_cutadd_time,
                        100.0 * total_cutadd_time / total_time,
                    )
                )
                print(
                    "Avg/iter: master=%.3fs sp_solve=%.3fs cutgen=%.3fs cutadd=%.3fs"
                    % (
                        total_master_time / total_iter,
                        total_sp_solve_time / total_iter,
                        total_cutgen_time / total_iter,
                        total_cutadd_time / total_iter,
                    )
                )
        except Exception:
            pass
        return _make_result(SolveStatus.UNKNOWN, max_it)


__all__ = ["BendersSolver", "BendersRunResult"]
