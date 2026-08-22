from __future__ import annotations

from typing import Any, Dict, Tuple, Iterable, Optional, Mapping
from bisect import bisect_right
import time
from pathlib import Path
import json
import math

try:
    import yaml as _yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _yaml = None

import pyomo.environ as pyo
from dataclasses import dataclass, field

from ..benders.subproblem import Subproblem
from ..benders.types import Candidate, Cut, CutType, SubproblemResult
from ..signature import project_core_point


# Whether a cut-generation mode carries a lower-bound guarantee.
#
# One table rather than a boolean set by hand in each dispatch branch. The branches
# and the flag were separate facts about the same thing, and they disagreed: the MW
# fallback set `cut_lb_valid = False` in one place while `solver.py` read the missing
# diagnostics key as True in another, which is the defect `CutValidity` was introduced
# to describe (benders/types.py). A mode whose name is not in this table raises rather
# than defaulting, so adding a generator without deciding its guarantee is not possible.
#
#   mw                 Pareto-optimal dual on the optimal face. Valid.
#   mw_dual_fallback   MW declined; plain capacity duals instead. Valid but not Pareto
#                      optimal -- validity comes from dual feasibility (handout 77).
#   dual               plain capacity duals by configuration. The Level A/B baseline.
#   finite_difference  perturbation estimates. NOT a certified lower bound
#                      (handout 75/76): a heuristic subproblem value bounds Q(x) from
#                      ABOVE, so a cut built from it can exclude the optimum.
CUT_MODE_VALID_LOWER_BOUND: Dict[str, bool] = {
    "mw": True,
    "mw_dual_fallback": True,
    "dual": True,
    "finite_difference": False,
}


def cut_mode_carries_lower_bound(mode: str) -> bool:
    """Look up a mode's lower-bound guarantee, refusing unknown modes.

    Fail-closed is not enough here: defaulting an unrecognised mode to False would
    silently drop `best_lb` for a run whose cuts may be perfectly valid, and
    defaulting to True would report a bound that nothing certified. Both are wrong
    answers to "I do not know", so the only correct response is to refuse.
    """
    try:
        return CUT_MODE_VALID_LOWER_BOUND[str(mode)]
    except KeyError:
        raise RuntimeError(
            f"cut generation reported mode {mode!r}, which has no recorded "
            "lower-bound guarantee. Add it to CUT_MODE_VALID_LOWER_BOUND with an "
            "explicit decision; neither True nor False is a safe default for a "
            "generator whose validity nobody has stated."
        ) from None


@dataclass(frozen=True, slots=True)
class MWDual:
    """One dual solution of the passenger-assignment LP, as a cut needs it.

    `dm_d[tau]` are the cut slopes, already scaled by seats (`S * pi_d[tau] <= 0`).

    `intercept_d` is the demand side of the dual objective, `sum_i alpha_d[i] * R_d[i]`,
    as a SCALAR. It is a scalar rather than a vector because the index of the demand
    rows is not the index of the capacity rows and need not be: the slot recourse has
    one demand row per slot, the minute recourse one per arrival MINUTE (D51), and the
    cut only ever needs the sum. Making the scalar the interface is what lets both
    resolutions share this code -- and its absence is what broke the minute path when
    S2 started deriving the intercept from a slot-indexed `alpha` the minute LP does
    not have.

    `alpha_d` is kept for diagnostics only, keyed by whatever the demand rows are keyed
    by. Never sum it against a slot-indexed demand vector.

    Carried as one object rather than loose dicts because the slopes and the intercept
    they must agree with are only meaningful together: `derive_cut_intercepts` checks
    one against the other, and a caller holding slopes from one solve and an intercept
    from another would pass that check while producing a cut tight at nothing.
    """

    dm_out: Dict[int, float]
    dm_ret: Dict[int, float]
    intercept_out: float
    intercept_ret: float
    alpha_out: Dict[int, float] = field(default_factory=dict)
    alpha_ret: Dict[int, float] = field(default_factory=dict)


def resolve_cut_mode(params: Mapping[str, Any]) -> str:
    """The generator this run uses, as one value (S1b).

    `cut_mode` is authoritative when present; `config.py` always sets it. The legacy
    `use_magnanti_wong` / `use_dual_slopes` pair is honoured only for callers that build
    params by hand -- tests and scripts -- and is resolved with the dispatch's historical
    precedence: mw, then dual, then finite differences.

    That precedence is why the enum exists. Every shipped config set BOTH booleans true,
    so `dual` -- the valid, non-Pareto ablation baseline the formal formulation's Level
    A/B needs -- was unreachable (AUDIT_v4 3.5). Two booleans cannot carry three
    mutually exclusive states without one going dead.
    """
    mode = params.get("cut_mode")
    if mode:
        mode = str(mode).strip().lower()
        if mode not in CUT_MODE_VALID_LOWER_BOUND and mode != "mw_dual_fallback":
            raise RuntimeError(
                f"cut_mode is {mode!r}, which no generator implements. Expected one of "
                f"{', '.join(sorted(CUT_MODE_VALID_LOWER_BOUND))}."
            )
        return mode
    if bool(params.get("use_magnanti_wong", False)):
        return "mw"
    if bool(params.get("use_dual_slopes", False)):
        return "dual"
    return "finite_difference"


def slopes_from_capacity_duals(
    duals: Mapping[str, Any], S: float, T: int
) -> MWDual:
    """Build the plain-dual (non-Pareto) cut data straight from `solve_subproblem`.

    `dm_d[tau] = S * pi_d[tau]`, with `pi <= 0` for a `<=` row in a minimisation.
    This is a **valid** lower-bounding cut -- validity comes from dual feasibility
    (handout 77) -- it is simply not Pareto-optimal. That makes it the correct
    fallback when Magnanti-Wong declines, and the ablation baseline the formal
    formulation's Level A/B comparison needs.
    """
    pi_out = dict(duals.get("pi_OUT", {}) or {})
    pi_ret = dict(duals.get("pi_RET", {}) or {})
    # The demand side of the dual objective, as a scalar. Both recourse resolutions
    # publish it under these keys; neither is asked to express its demand duals on the
    # slot index, because the minute recourse's are keyed by arrival minute (D51) and
    # summing those against a slot-indexed demand vector is meaningless.
    if "intercept_out" not in duals or "intercept_ret" not in duals:
        raise KeyError(
            "the subproblem returned duals without 'intercept_out'/'intercept_ret'. "
            "That scalar is the demand side of the dual objective, sum_i alpha[i]*R[i], "
            "and the cut constant is derived from it (S2). A recourse that cannot "
            "produce it cannot produce a cut whose tightness is checkable, so it must "
            "say so rather than let the constant be imposed silently."
        )
    return MWDual(
        dm_out={t: float(S) * float(pi_out.get(t, 0.0)) for t in range(T)},
        dm_ret={t: float(S) * float(pi_ret.get(t, 0.0)) for t in range(T)},
        intercept_out=float(duals["intercept_out"]),
        intercept_ret=float(duals["intercept_ret"]),
        alpha_out=dict(duals.get("alpha_OUT", {}) or {}),
        alpha_ret=dict(duals.get("alpha_RET", {}) or {}),
    )


def expand_slopes_to_candidate(
    candidate: Candidate,
    dm_out: Mapping[int, float],
    dm_ret: Mapping[int, float],
    T: int,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """Broadcast per-slot slopes onto the master's per-(q,tau) variable names.

    The slope is identical across `q` by construction -- the recourse sees only
    `Y_d[tau] = sum_q y_d[q,tau]` (D48/E1) -- which is what makes the master's
    `aggregate_cuts_by_tau` collapse an identity rather than an approximation.
    `_assert_q_invariant` enforces it on the other side.

    Extracted because this block was written out four times, twice per dispatch
    site. Duplicated cut-generation code is how C3 and N6 each stayed invisible
    for months: a fix applied to one copy leaves the other producing the old cut.
    """
    c_out: Dict[Tuple[int, int], float] = {}
    c_ret: Dict[Tuple[int, int], float] = {}
    for name in candidate.keys():
        if not isinstance(name, str):
            continue
        if name.startswith("yOUT["):
            target, slopes = c_out, dm_out
        elif name.startswith("yRET["):
            target, slopes = c_ret, dm_ret
        else:
            continue
        inside = name[name.find("[") + 1 : name.find("]")]
        q_str, tau_str = inside.split(",")
        q = int(q_str.strip())
        tau = int(tau_str.strip())
        if 0 <= tau < T:
            target[(q, tau)] = float(slopes.get(tau, 0.0))
    return c_out, c_ret


def derive_cut_intercepts(
    mw: MWDual,
    y_inc_out: Iterable[float],
    y_inc_ret: Iterable[float],
    ub_out: float,
    ub_ret: float,
    T: int,
    eps_dual: float,
) -> Tuple[float, float, Dict[str, float]]:
    """Derive the cut intercepts from the demand duals, and verify strong duality.

    Returns `(a_out, a_ret, diagnostics)` with `a_d = sum_t alpha_d[t] * R_d[t]`.

    The cut for direction `d` is `theta_d >= a_d + sum_tau dm_d[tau] * Y_d[tau]`.
    At the incumbent `Y_d = C_d / S` this evaluates to
    `sum_t alpha_d R_d + sum_tau C_d[tau] pi_d[tau]`, which strong duality equates
    to `Q_d(Y_inc)`. So the tightness property the formal formulation checks in 20.1
    now *follows* from the dual rather than being imposed.

    Raises when the two disagree by more than `eps_dual`. That disagreement is the
    live version of formulation 20.3, and it is the detector this code did not have:
    the previous intercept was written as `ub - sum(dm*y_inc)`, so 20.1 could not
    fail, and an overestimating cut -- the only kind that can exclude the optimum,
    and the D30 defect -- had nothing watching for it.
    """
    y_inc_out = list(y_inc_out)
    y_inc_ret = list(y_inc_ret)

    a_out = float(mw.intercept_out)
    a_ret = float(mw.intercept_ret)

    imposed_out = float(ub_out) - sum(
        float(mw.dm_out.get(tau, 0.0)) * float(y_inc_out[tau]) for tau in range(T)
    )
    imposed_ret = float(ub_ret) - sum(
        float(mw.dm_ret.get(tau, 0.0)) * float(y_inc_ret[tau]) for tau in range(T)
    )

    d_out = a_out - imposed_out
    d_ret = a_ret - imposed_ret
    diagnostics = {
        "intercept_out_from_alpha": float(a_out),
        "intercept_ret_from_alpha": float(a_ret),
        "intercept_out_imposed": float(imposed_out),
        "intercept_ret_imposed": float(imposed_ret),
        "intercept_gap_out": float(d_out),
        "intercept_gap_ret": float(d_ret),
        "intercept_eps_dual": float(eps_dual),
    }
    if abs(d_out) > eps_dual or abs(d_ret) > eps_dual:
        raise RuntimeError(
            "strong duality failed on the selected dual: the intercept derived from "
            f"alpha is OUT={a_out:.10g} / RET={a_ret:.10g}, while Q(y) - sum(dm*y_inc) "
            f"gives OUT={imposed_out:.10g} / RET={imposed_ret:.10g} "
            f"(diff OUT={d_out:.3g}, RET={d_ret:.3g}, tol={eps_dual:.3g}). "
            "Either the dual LP does not represent the dual of the primal (a wrong "
            "row, a wrong sign, a stale right-hand side), or the primal objective "
            "used to anchor the cut is not the one the duals came from. A cut built "
            "through this disagreement can overestimate the recourse, which is the "
            "one error that excludes the optimum (D30)."
        )
    return a_out, a_ret, diagnostics


def _tightness_tolerance(
    eps_cut: float, target_val: float, dual_diag: Mapping[str, float]
) -> float:
    """Tolerance for "the assembled cut equals Q(y) at the incumbent" (20.1).

    Must be at least as loose as the slack the cut generator is ALLOWED to have, or
    the code rejects duals it just admitted. That inconsistency was live and hidden:

      - `solve_mw_dual` states the optimal face as an inequality with
        `face_tol = max(1e-6, 1e-9*|Q|)` of slack, deliberately, because a float
        equality against a separately computed primal optimum is infeasible for a
        few ulps of disagreement.
      - this check used `eps_cut * max(1, |Q|)`, i.e. `1e-8 * 22.4 = 2.24e-7` on the
        Phase 5 instance.

    So MW could legitimately return a dual 1e-6 inside the face and the tightness
    check would refuse the cut built from it. Nothing noticed, because the intercept
    was IMPOSED as `Q(y) - sum(dm*y_inc)`, which made this check an identity that
    could not fail (S2). Deriving the intercept from alpha exposed it immediately:
    the Phase 5 MW arm aborted with a measured gap of exactly -1e-06 = face_tol.

    The two jobs are now separated. `derive_cut_intercepts` checks the CONSTANT
    against the duals at `eps_dual`, which is the tolerance that has to admit the
    face slack. This function checks the ASSEMBLED cut -- constant plus broadcast
    slopes -- so its remaining job is to catch a bad broadcast, and it inherits
    `eps_dual` so it cannot fire on a constant already accepted upstream.
    """
    floor = float(eps_cut) * max(1.0, abs(float(target_val)))
    eps_dual = float(dual_diag.get("intercept_eps_dual", 0.0) or 0.0)
    return max(floor, eps_dual)


def _cut_intercepts(
    mw: MWDual | None,
    *,
    y_inc_out: Iterable[float],
    y_inc_ret: Iterable[float],
    ub_out: float,
    ub_ret: float,
    T: int,
    ub_val: float,
    dm_out: Mapping[int, float],
    dm_ret: Mapping[int, float],
) -> Tuple[float, float, Dict[str, float]]:
    """Directional cut intercepts, derived from alpha whenever a dual is available.

    `mw is None` only on the `finite_difference` path, which has no duals to derive
    from and is already marked as not a valid lower bound. There the intercept must
    still be imposed, and the strong-duality check is meaningless because there is
    no dual to check -- so it is skipped rather than passed vacuously.
    """
    if mw is None:
        y_inc_out = list(y_inc_out)
        y_inc_ret = list(y_inc_ret)
        a_out = float(ub_out) - sum(
            float(dm_out.get(tau, 0.0)) * float(y_inc_out[tau]) for tau in range(T)
        )
        a_ret = float(ub_ret) - sum(
            float(dm_ret.get(tau, 0.0)) * float(y_inc_ret[tau]) for tau in range(T)
        )
        return a_out, a_ret, {"intercept_source": "imposed_no_dual"}

    # Tolerance must exceed the optimal-face slack. `solve_mw_dual` admits duals
    # within `face_tol = max(1e-6, 1e-9*|ub|)` of the face, so a selected dual can
    # legitimately sit that far inside it, and the derived intercept then differs
    # from the imposed one by up to the same amount. Anything beyond an order of
    # magnitude past that slack is a real disagreement, not float noise.
    eps_dual = max(1e-5, 1e-7 * abs(float(ub_val)))
    a_out, a_ret, diag = derive_cut_intercepts(
        mw,
        y_inc_out=y_inc_out,
        y_inc_ret=y_inc_ret,
        ub_out=ub_out,
        ub_ret=ub_ret,
        T=T,
        eps_dual=eps_dual,
    )
    diag["intercept_source"] = "alpha"
    return a_out, a_ret, diag


class ProblemSubproblem(Subproblem):
    """LP assignment/waiting subproblem generating optimality cuts.

    Builds a time-expanded assignment LP for both directions (OUT, RET) with
    waiting costs and unmet-demand penalties. Extracts dual multipliers to
    form the Benders optimality cut for a minimization master:

        theta >= const - S * sum_{q,tau} [pi_OUT[tau]*yOUT[q,tau] + pi_RET[tau]*yRET[q,tau]]

    where const = sum_t alpha_OUT[t]*R_out[t] + sum_t alpha_RET[t]*R_ret[t], and
    pi_* are duals on capacity constraints (>= 0). Increasing capacity (y) reduces
    the subproblem cost, hence the negative sign.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)

    def _is_report(self) -> bool:
        return str((self.params or {}).get("log_level", "")).upper() == "REPORT"

    def _vprint(self, *args, **kwargs) -> None:
        if not self._is_report():
            print(*args, **kwargs)

    def _parse_candidate_indices(
        self, candidate: Candidate
    ) -> Tuple[set[int], set[int]]:
        qs: set[int] = set()
        ts: set[int] = set()
        for name in candidate.keys():
            if isinstance(name, str) and (
                name.startswith("yOUT[") or name.startswith("yRET[")
            ):
                inside = name[name.find("[") + 1 : name.find("]")]
                q_str, t_str = inside.split(",")
                q = int(q_str.strip())
                t = int(t_str.strip())
                qs.add(q)
                ts.add(t)
        return qs, ts

    def evaluate(self, candidate: Candidate) -> SubproblemResult:
        params = self.params or {}
        debug_early_exit = bool(params.get("debug_early_exit", False))
        debug_timing = bool(params.get("debug_timing", False))
        debug_skip_cut_generation = bool(params.get("debug_skip_cut_generation", False))
        # Parameters (with safe defaults)
        S = float(params.get("S", 1.0))
        # Resolution in minutes per slot (copied from master params via config or set here)
        slot_res = int(params.get("slot_resolution", params.get("resolution", 1)))
        time_step_min = int(params.get("time_step_minutes", 1) or 1)
        # D9: the elastic/minute-level subproblem relaxation has been removed. The
        # slot-only model (solve_subproblem + solve_mw_dual) is the only path.
        # See docs/docs_decisions.md D9 and docs/BENDERS_SPEC_v3.md §2.5.
        # Allow Wmax to be specified in minutes
        if "Wmax_minutes" in params:
            Wmax = wmax_minutes_to_slots(float(params.get("Wmax_minutes", 0)), slot_res)
        else:
            Wmax = int(params.get("Wmax_slots", params.get("Wmax", 0)))
        p_pen = float(params.get("p", 0.0))
        _eps_raw = params.get("eps_cut", None)
        eps_cut = float(_eps_raw) if _eps_raw is not None else 1e-6
        lp_solver = str(params.get("lp_solver", "cplex_direct"))
        # Optional: solver-specific options (e.g., CPLEX: {"lpmethod": 2, "threads": 0})
        solver_options = dict(params.get("solver_options", {}) or {})
        # Prefer packing demand into the first vehicle layer, then the next (LP tie-breaker)
        fill_eps = (
            0.0  # removed with the layers (D30): no per-vehicle ordering to break
        )

        # Determine T and Q from candidate if not configured
        q_idx, t_idx = self._parse_candidate_indices(candidate)
        master_q_idx = (
            sorted(q_idx) if q_idx else list(range(int(params.get("Q", 0) or 0)))
        )
        T_cand = (max(t_idx) + 1) if t_idx else int(params.get("T", 0))
        T = int(params.get("T", T_cand))

        # Helpers to read demand from files or inline and aggregate into R vectors
        # Thin delegates: the real implementations are module level so the master
        # can reuse them, D25 truncation counting included.
        def _load_doc(path: Path) -> Any:
            return load_demand_doc(path)

        def _aggregate_requests(
            container: Any, Tlen: int
        ) -> tuple[list[float], list[float]]:
            return aggregate_requests(container, Tlen, slot_res, self._vprint)

        def _ok(lhs: float | None, rhs: float | None, eps: float) -> bool:
            if lhs is None or rhs is None:
                return False
            return float(lhs) >= float(rhs) - float(eps) * (1.0 + abs(float(rhs)))

        def _cand_theta(key: str) -> float | None:
            try:
                v = candidate.get(key)
            except Exception:
                v = None
            if v is None:
                return None
            # Missing return: without it this helper yielded None for every key, so the
            # early exit below never fired and a cut was generated every iteration.
            try:
                return float(v)
            except Exception:
                return None

        def _dbg(msg: str) -> None:
            if debug_timing:
                self._vprint(msg)

        def _minutes_for(source: Any) -> dict | None:
            """Per-request arrival minutes for a demand source, or None in slot mode.

            The slot aggregation throws these away at load, so minute-level recourse has
            to read the file a second time. Returns None when the source is not a file
            path -- inline demand carries no arrival times to recover, and a minute-mode
            run against inline demand must fail loudly in solve_subproblem rather than
            quietly price a schedule against nothing.
            """
            if str(params.get("recourse_resolution", "slot")).lower() != "minute":
                return None
            if not isinstance(source, (str, Path)):
                return None
            from ..minute_pricer import load_request_minutes

            return load_request_minutes(Path(str(source)))

        def _load_demand_from_file(
            path_like: Any, Tlen: int
        ) -> tuple[list[float], list[float]]:
            p = Path(str(path_like))
            doc = _load_doc(p)
            return _aggregate_requests(doc, Tlen)

        # (legacy) aggregate_requests_to_R removed; using _aggregate_requests instead

        # Capacities induced by master decisions: C_*[tau] = S * sum_q y*_{q,tau}
        C_out = [0.0 for _ in range(T)]
        C_ret = [0.0 for _ in range(T)]
        # Vehicle counts per departure slot (per direction)
        K_out = [0 for _ in range(T)]
        K_ret = [0 for _ in range(T)]
        for name, val in candidate.items():
            if not isinstance(name, str):
                continue
            if name.startswith("yOUT["):
                inside = name[name.find("[") + 1 : name.find("]")]
                _, tau_str = inside.split(",")
                tau = int(tau_str.strip())
                if 0 <= tau < T:
                    C_out[tau] += float(val) * S
                    try:
                        if float(val) >= 0.5:
                            K_out[tau] += 1
                    except Exception:
                        pass
            elif name.startswith("yRET["):
                inside = name[name.find("[") + 1 : name.find("]")]
                _, tau_str = inside.split(",")
                tau = int(tau_str.strip())
                if 0 <= tau < T:
                    C_ret[tau] += float(val) * S
                    try:
                        if float(val) >= 0.5:
                            K_ret[tau] += 1
                    except Exception:
                        pass

        # Multi-scenario support
        scenarios: list[dict] = list(params.get("scenarios", []))
        # Allow specifying scenarios as file paths
        if not scenarios and isinstance(params.get("scenario_files"), list):
            scenarios = list(params.get("scenario_files"))
        # Normalize single-scenario lists to single-demand path
        single_scenario_override: tuple[list[float], list[float]] | None = None
        # Which demand source the single-demand path ends up using. Minute mode
        # has to re-read that exact file, because the slot aggregation discards
        # arrival minutes; reading a different one would price a schedule against
        # demand the slot run never saw.
        single_demand_source: Any = params.get("demand_file")
        if scenarios and len(scenarios) == 1:
            s0 = scenarios[0]
            if isinstance(s0, (str, Path)):
                R_out0, R_ret0 = _load_demand_from_file(s0, T)
            elif isinstance(s0, dict) and (
                "requests" in s0 or "req_matrix" in s0 or "R_out" in s0 or "R_ret" in s0
            ):
                R_out0, R_ret0 = _aggregate_requests(s0, T)
            else:
                # Best effort
                R_out0 = list(getattr(s0, "R_out", [0.0] * T))
                R_ret0 = list(getattr(s0, "R_ret", [0.0] * T))
            R_out0 = (R_out0 + [0.0] * T)[:T]
            R_ret0 = (R_ret0 + [0.0] * T)[:T]
            single_scenario_override = (R_out0, R_ret0)
            single_demand_source = s0
            scenarios = []
        # Multi-cut vs averaged cut control.
        # New flag: multi_cuts_by_scenario (True => return one cut per scenario)
        # Backward compat: if not provided, use legacy average_cuts_across_scenarios (True => single averaged cut)
        _mc = params.get("multi_cuts_by_scenario", None)
        if _mc is None:
            average_cuts: bool = bool(
                params.get("average_cuts_across_scenarios", False)
            )
            multi_cuts: bool = not average_cuts
        else:
            multi_cuts = bool(_mc)
            average_cuts = not multi_cuts
        ub_aggregation: str = str(params.get("ub_aggregation", "mean"))
        weights: list[float] | None = params.get("scenario_weights")

        # Only evaluate finite differences for time slots that appear in candidate (fewer solves)
        active_taus = sorted(t_idx) if t_idx else list(range(T))

        # Optional Magnanti–Wong selection
        _cut_mode_cfg: str = resolve_cut_mode(params)
        mw_enabled: bool = _cut_mode_cfg == "mw"
        core_point = params.get("mw_core_point") or {}
        Ybar_out = (
            list(core_point.get("Yout", [])) if isinstance(core_point, dict) else []
        )
        Ybar_ret = (
            list(core_point.get("Yret", [])) if isinstance(core_point, dict) else []
        )
        if len(Ybar_out) < T:
            Ybar_out = (Ybar_out + [0.0] * T)[:T]
        if len(Ybar_ret) < T:
            Ybar_ret = (Ybar_ret + [0.0] * T)[:T]
        # If the core point is still all zeros (common in early iters), seed it to a
        # small positive profile so MW has direction to select non-trivial duals.
        #
        # This was wrapped in `try/except Exception: pass`, and it is the one place
        # in the MW path where silence is expensive. `Ybar` IS the direction MW
        # maximises in: with an all-zero core point the objective
        # `sum((S*Ybar[tau] - C[tau]) * pi[tau])` degenerates to `-sum(C*pi)`, MW
        # selects an essentially arbitrary point of the optimal face, and the mode
        # is still reported as `mw`. Every dominance margin measured in D42 would
        # collapse toward zero and nothing would say why.
        #
        # The exception it was swallowing can only be a non-numeric entry in the
        # core point, which is a caller error worth seeing, not a condition to
        # continue through.
        core_seeded = False
        try:
            core_mass = float(sum(Ybar_out)) + float(sum(Ybar_ret))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "mw_core_point carries a non-numeric entry: "
                f"{type(exc).__name__}: {exc}. Ybar is the direction Magnanti-Wong "
                "maximises in; continuing past this would select an arbitrary dual "
                "and still label the cut 'mw'."
            ) from exc
        if core_mass == 0.0 and T > 0:
            # Seed, then PROJECT it the same way the solver projects its own core point
            # (S3). An all-ones seed puts positive Ybar on slots the master fixes to
            # zero and ignores the trip window, which is exactly the defect S3 removed
            # from `solver.py`; reintroducing it here would leave the two paths
            # disagreeing about the region, with this one reachable whenever a caller
            # supplies no core point.
            trip_seed = 1
            try:
                trip_seed = max(1, int(params.get("trip_slots") or 0))
            except (TypeError, ValueError):
                trip_seed = 1
            if trip_seed <= 1 and params.get("trip_duration_minutes"):
                try:
                    trip_seed = max(
                        1,
                        math.ceil(
                            float(params["trip_duration_minutes"])
                            / max(1.0, float(params.get("slot_resolution") or 1))
                        ),
                    )
                except (TypeError, ValueError):
                    trip_seed = 1
            Q_seed = 0
            try:
                Q_seed = int(params.get("Q") or 0)
            except (TypeError, ValueError):
                Q_seed = 0
            if Q_seed >= 1:
                Ybar_out, Ybar_ret = project_core_point(
                    [1.0] * T, [1.0] * T, Q_seed, trip_seed, 1e-3
                )
            else:
                # No fleet size in the subproblem's params: fall back to the old seed
                # rather than inventing a Q. Say so, because the selection direction is
                # then outside the master's region and the resulting cut is still valid
                # but its Pareto claim is not (16.2).
                Ybar_out = [1.0 for _ in range(T)]
                Ybar_ret = [1.0 for _ in range(T)]
                self._vprint(
                    "[MW CORE] no fleet size in the subproblem params, so the seeded "
                    "core point could not be projected into the master region; the cut "
                    "stays valid but is not Pareto-optimal with respect to a feasible "
                    "direction."
                )
            core_seeded = True
            self._vprint(
                f"[MW CORE] core point arrived all zeros; seeded over T={T} "
                f"(Q={Q_seed}, trip_slots={trip_seed}) so the selection has a direction."
            )

        def solve_mw_dual(
            T_: int,
            Wmax_slots: int,
            p_penalty: float,
            S_cap: float,
            K_out_use: list[int],
            K_ret_use: list[int],
            C_out_vec: list[float],
            C_ret_vec: list[float],
            R_out_vec: list[float],
            R_ret_vec: list[float],
            Ybar_out_vec: list[float],
            Ybar_ret_vec: list[float],
            ub_base: float,
            lp: str,
            lp_opts: dict | None = None,
        ) -> MWDual | None:
            """Solve the dual LP on the optimal face to select a Pareto-optimal dual.

            Returns the slopes `dm_d[t]` w.r.t. `sum_y_d[t]` **and** the demand duals
            `alpha_d[t]`.

            Alpha is returned because the cut intercept must be *derived* from it,
            `a_d = sum_t alpha_d[t] * R_d[t]`, rather than imposed as
            `ub_base - sum dm*y_inc` (S2). The imposed form is algebraically equal only
            when the selected dual sits exactly on the optimal face; `OptFace` below is
            deliberately an inequality with `face_tol` of slack, so the imposed intercept
            can sit up to `face_tol` ABOVE the true dual cut -- an overestimate, the one
            direction that can exclude the optimum. Deriving it also turns the tightness
            diagnostic (formal formulation 20.1) from an identity the code writes into a
            fact about the dual it selected.
            """
            md = pyo.ConcreteModel()
            md.name = "mw_dual"
            Tset = range(T_)

            # Dual variables.
            #
            # Sign convention follows Pyomo/CPLEX and the rest of this module: for a
            # '<=' constraint in a MINIMISATION, the dual is <= 0 (raising capacity
            # lowers cost). solve_subproblem reads its pi_OUT/pi_RET straight from
            # m.dual and they come out non-positive, which is why dm = +S*pi yields
            # the negative slopes the master expects. This LP must match that.
            md.a_OUT = pyo.Var(Tset)
            md.a_RET = pyo.Var(Tset)
            md.pi_OUT = pyo.Var(Tset, within=pyo.NonPositiveReals)
            md.pi_RET = pyo.Var(Tset, within=pyo.NonPositiveReals)

            # Dual feasibility: one constraint per primal variable.
            #   primal x[t,tau,k], cost (tau-t) + fill_eps*k  ->  a[t] + pi[tau,k] <= cost
            #   primal u[t],       cost p                     ->  a[t]            <= p
            def df_out_rule(m, t, tau):
                # active arc iff (t+1) <= tau <= min(T-1, t+W)
                if not ((t + 1) <= tau <= min(T_ - 1, t + Wmax_slots)):
                    return pyo.Constraint.Skip
                return m.a_OUT[t] + m.pi_OUT[tau] <= float(max(0, tau - t))

            md.DF_OUT = pyo.Constraint(
                [(t, tau) for t in Tset for tau in Tset],
                rule=lambda m, t, tau: df_out_rule(m, t, tau),
            )

            # Dual feasibility for RET
            def df_ret_rule(m, t, tau):
                if not ((t + 1) <= tau <= min(T_ - 1, t + Wmax_slots)):
                    return pyo.Constraint.Skip
                return m.a_RET[t] + m.pi_RET[tau] <= float(max(0, tau - t))

            md.DF_RET = pyo.Constraint(
                [(t, tau) for t in Tset for tau in Tset],
                rule=lambda m, t, tau: df_ret_rule(m, t, tau),
            )

            # Dual constraint from the unmet-demand variables u[t] >= 0
            md.A_OUT_CAP = pyo.Constraint(
                Tset, rule=lambda m, t: m.a_OUT[t] <= float(p_penalty)
            )
            md.A_RET_CAP = pyo.Constraint(
                Tset, rule=lambda m, t: m.a_RET[t] <= float(p_penalty)
            )

            # Optimality face equality: dual objective equals primal UB at incumbent
            cap_out_rhs = [float(C_out_vec[tau]) for tau in Tset]
            cap_ret_rhs = [float(C_ret_vec[tau]) for tau in Tset]

            def dual_obj_expr(m):
                term_dem = sum(float(R_out_vec[t]) * m.a_OUT[t] for t in Tset) + sum(
                    float(R_ret_vec[t]) * m.a_RET[t] for t in Tset
                )
                term_cap = sum(cap_out_rhs[tau] * m.pi_OUT[tau] for tau in Tset) + sum(
                    cap_ret_rhs[tau] * m.pi_RET[tau] for tau in Tset
                )
                return term_dem + term_cap

            # Restrict to the optimal face. Weak duality gives dual_obj <= ub_base for
            # every dual-feasible point, so ">= ub_base - tol" carves out the (near-)
            # optimal face. Stated as an inequality rather than "== ub_base" because a
            # float equality against a separately-computed primal optimum is brittle:
            # a few ulps of disagreement make the whole LP infeasible.
            face_tol = max(1e-6, 1e-9 * abs(float(ub_base)))
            md.OptFace = pyo.Constraint(
                expr=(dual_obj_expr(md) >= float(ub_base) - face_tol)
            )

            # Magnanti-Wong objective.
            #
            # The cut is theta >= const + sum_tau dm[tau]*Y[tau], with dm[tau] =
            # S*sum_k pi[tau,k] and const anchored so the cut is tight at the
            # incumbent: const = ub_base - sum_tau dm[tau]*y_inc[tau]. Its value at the
            # core point Ybar is therefore
            #
            #     const + sum dm*Ybar = ub_base + sum_tau dm[tau]*(Ybar[tau]-y_inc[tau])
            #
            # ub_base is fixed by OptFace, so selecting the Pareto-optimal dual means
            # maximising sum_tau dm[tau]*(Ybar[tau] - y_inc[tau]).
            #
            # The previous version maximised sum dm*Ybar, dropping the -y_inc term.
            # That term is NOT constant over the optimal face (dm varies across it), so
            # omitting it selects the wrong dual. y_inc[tau] = C[tau]/S, hence the
            # coefficient below is (S*Ybar[tau] - C[tau]).
            md.obj = pyo.Objective(
                expr=(
                    sum(
                        (
                            float(S_cap) * float(Ybar_out_vec[tau])
                            - float(C_out_vec[tau])
                        )
                        * md.pi_OUT[tau]
                        for tau in Tset
                    )
                    + sum(
                        (
                            float(S_cap) * float(Ybar_ret_vec[tau])
                            - float(C_ret_vec[tau])
                        )
                        * md.pi_RET[tau]
                        for tau in Tset
                    )
                ),
                sense=pyo.maximize,
            )

            solver = pyo.SolverFactory(lp)
            try:
                for k, v in (lp_opts or {}).items():
                    solver.options[k] = v
            except Exception:
                pass
            res = solver.solve(md, tee=False, load_solutions=False)
            term = getattr(res.solver, "termination_condition", None)
            if term not in (pyo.TerminationCondition.optimal,):
                # Report why MW failed. Without this the caller only sees None and
                # silently degrades to finite differences (see N6).
                self._vprint(
                    f"[MW FAIL] dual LP not optimal: termination={term} "
                    f"status={getattr(res.solver, 'status', None)} "
                    f"ub_base={float(ub_base):.10g}"
                )
                return None
            # HANDLER_CENSUS.md Category A: this load was wrapped in
            # `except Exception: pass` immediately before the readback below. Making
            # it loud is right on its own terms, but it is NOT what broke the
            # `Ybar_ret == 0` core point -- measured, the load succeeds and CPLEX
            # reports `optimal`. Keep the two apart.
            try:
                md.solutions.load_from(res)
            except Exception as exc:
                self._vprint(
                    f"[MW FAIL] could not load the dual solution: "
                    f"{type(exc).__name__}: {exc}. termination={term} "
                    f"status={getattr(res.solver, 'status', None)}"
                )
                return None

            # A pi with no value is a variable the backend never sent to the solver:
            # it carries a zero objective coefficient and appears in no row, which
            # happens at any tau where the candidate schedules no trip in that
            # direction. Its slope is exactly 0 -- that is arithmetic, not a guess.
            #
            # Reading it with a blanket `or 0.0` would be a guess, and the dangerous
            # kind: a failed load also leaves every pi empty, and the census names
            # the result -- "an all-zero slope vector, i.e. a cut that constrains
            # nothing, with no error". So distinguish the two. Some empty is
            # structural; all empty is a failure, and a failure returns None so the
            # caller falls back and marks the cut NOT a valid lower bound (D39).
            def _pi(v) -> tuple[float, bool]:
                raw = getattr(v, "value", None)
                return (0.0, False) if raw is None else (float(raw), True)

            dm_out, dm_ret = {}, {}
            pi_out_val, pi_ret_val = {}, {}
            alpha_out, alpha_ret = {}, {}
            n_seen = 0
            for tau in range(T_):
                vo, ok_o = _pi(md.pi_OUT[tau])
                vr, ok_r = _pi(md.pi_RET[tau])
                n_seen += int(ok_o) + int(ok_r)
                pi_out_val[tau] = vo
                pi_ret_val[tau] = vr
                dm_out[tau] = float(S_cap) * vo
                dm_ret[tau] = float(S_cap) * vr
            # Alpha is read with the same discipline, but a missing alpha is NOT
            # structural the way a missing pi is: `A_OUT_CAP`/`A_RET_CAP` put every
            # alpha in a row, so the backend always sends them. An unset alpha
            # therefore means the load did not deliver what it claimed, and the
            # intercept built from it would silently read 0 -- which is the same
            # all-zero failure the pi comment above refuses, moved to the constant.
            for t in range(T_):
                va, ok_a = _pi(md.a_OUT[t])
                vb, ok_b = _pi(md.a_RET[t])
                n_seen += int(ok_a) + int(ok_b)
                if not (ok_a and ok_b):
                    self._vprint(
                        f"[MW FAIL] demand dual alpha unset at t={t} "
                        f"(OUT set={ok_a}, RET set={ok_b}); every alpha appears in a "
                        "row, so this is a load failure, not a structural zero. "
                        "Refusing rather than deriving an intercept from it."
                    )
                    return None
                alpha_out[t] = va
                alpha_ret[t] = vb
            if n_seen == 0 and T_ > 0:
                self._vprint(
                    "[MW FAIL] the dual solution carries no values at all "
                    f"({4 * T_} multipliers, none set); refusing to build an "
                    "all-zero slope vector from it."
                )
                return None

            # Weak duality, checked on the dual that was actually selected.
            #
            # This is the runtime half of MW verification 3; the offline half is the
            # underestimation test in tests/test_solver_soundness.py. Every
            # dual-feasible point satisfies dual_obj <= ub_base, and OptFace pins it
            # from below at ub_base - face_tol, so the selected dual must land in a
            # band of width face_tol. Above the band means the dual LP does not
            # represent the dual of the primal -- a wrong row, a wrong sign, a stale
            # right-hand side -- and the cut built from it would OVERESTIMATE the
            # recourse, which is the one error that excludes the optimum. D30 was
            # that defect and it went six months unseen.
            #
            # Computed from the values read above, NOT via `pyo.value(dual_obj_expr)`.
            # That is not a style preference: `pi_RET[0]` appears in no DF row (no arc
            # can reach tau=0, since arcs need t+1 <= tau) and carries coefficient 0.0
            # in both this expression and the MW objective whenever C_ret[0] = 0, so
            # the backend never sends it and `pyo.value` raises on an uninitialized
            # var. The check was therefore REFUSING EVERY MW SOLUTION on any instance
            # with no RET capacity in slot 0 -- measured on the test suite, which
            # printed "[MW FAIL] the dual objective is unreadable after load". A guard
            # that fails closed in the wrong place is how C3 hid: MW was reported as
            # enabled and never ran. Unset multipliers contribute exactly 0.0 here,
            # which is arithmetic, not a guess -- their coefficient is 0 by the same
            # structural fact that left them unsent.
            dual_obj_val = (
                sum(float(R_out_vec[t]) * alpha_out[t] for t in range(T_))
                + sum(float(R_ret_vec[t]) * alpha_ret[t] for t in range(T_))
                + sum(cap_out_rhs[tau] * pi_out_val[tau] for tau in range(T_))
                + sum(cap_ret_rhs[tau] * pi_ret_val[tau] for tau in range(T_))
            )
            if dual_obj_val > float(ub_base) + face_tol:
                self._vprint(
                    "[MW FAIL] weak duality violated by the selected dual: "
                    f"dual_obj={dual_obj_val:.10g} > ub_base={float(ub_base):.10g} "
                    f"(tol={face_tol:.3g}). A cut from this dual would overestimate "
                    "the recourse. Refusing it; the caller falls back to the plain "
                    "capacity duals, which remain a valid lower bound (S1)."
                )
                return None

            return MWDual(
                dm_out=dm_out,
                dm_ret=dm_ret,
                # This LP's demand rows ARE slot-indexed, so the scalar is formed here
                # from its own alpha and the demand it was built with.
                intercept_out=sum(
                    float(R_out_vec[t]) * alpha_out[t] for t in range(T_)
                ),
                intercept_ret=sum(
                    float(R_ret_vec[t]) * alpha_ret[t] for t in range(T_)
                ),
                alpha_out=alpha_out,
                alpha_ret=alpha_ret,
            )

        # Finite-difference coefficient builder: for each tau, solve with +S capacity
        def coeffs_by_fdiff(
            ub_base: float,
            C_out_base: list[float],
            C_ret_base: list[float],
            K_out_base: list[int],
            K_ret_base: list[int],
            R_out_vec: list[float],
            R_ret_vec: list[float],
        ) -> tuple[
            Dict[tuple[int, int], float],
            Dict[tuple[int, int], float],
            Dict[int, float],
            Dict[int, float],
        ]:
            coeff_y_out: Dict[tuple[int, int], float] = {}
            coeff_y_ret: Dict[tuple[int, int], float] = {}
            # Marginal effects by time (per one vehicle start at tau)
            dm_out: Dict[int, float] = {}
            dm_ret: Dict[int, float] = {}

            for tau in active_taus:
                # OUT marginal in the legacy nominal-capacity LP.
                C_out_tau = C_out_base.copy()
                C_out_tau[tau] = C_out_tau[tau] + S
                K_out_tau = K_out_base.copy()
                K_out_tau[tau] = K_out_tau[tau] + 1
                _, ub_plus = solve_subproblem(
                    SPParams(
                        T=T,
                        Wmax_slots=Wmax,
                        p=p_pen,
                        lp_solver=lp_solver,
                        S=S,
                        K_out=K_out_tau,
                        K_ret=K_ret_base,
                        fill_eps=fill_eps,
                        solver_options=solver_options,
                        eps_cut=eps_cut,
                        slot_resolution=slot_res,
                        time_step_minutes=time_step_min,
                        T_minutes=params.get("T_minutes"),
                        trip_duration_minutes=params.get("trip_duration_minutes"),
                        trip_slots=params.get("trip_slots"),
                        Q=int(params.get("Q", len(q_idx) if q_idx else 0) or 0),
                        binit=params.get("binit"),
                        initial_actions=params.get("initial_actions"),
                        Emax=params.get("Emax"),
                        L=params.get("L"),
                        delta_chg=params.get("delta_chg"),
                        eps_feas=float(params.get("eps_feas", 1e-7)),
                        solve_time_limit_s=params.get("solve_time_limit_s"),
                    ),
                    C_out_tau,
                    C_ret_base,
                    R_out_vec,
                    R_ret_vec,
                )
                dm_out[tau] = float(ub_plus - ub_base)

                # RET marginal in the legacy nominal-capacity LP.
                C_ret_tau = C_ret_base.copy()
                C_ret_tau[tau] = C_ret_tau[tau] + S
                K_ret_tau = K_ret_base.copy()
                K_ret_tau[tau] = K_ret_tau[tau] + 1
                _, ub_plus_r = solve_subproblem(
                    SPParams(
                        T=T,
                        Wmax_slots=Wmax,
                        p=p_pen,
                        lp_solver=lp_solver,
                        S=S,
                        K_out=K_out_base,
                        K_ret=K_ret_tau,
                        fill_eps=fill_eps,
                        solver_options=solver_options,
                        eps_cut=eps_cut,
                        slot_resolution=slot_res,
                        time_step_minutes=time_step_min,
                        T_minutes=params.get("T_minutes"),
                        trip_duration_minutes=params.get("trip_duration_minutes"),
                        trip_slots=params.get("trip_slots"),
                        Q=int(params.get("Q", len(q_idx) if q_idx else 0) or 0),
                        binit=params.get("binit"),
                        initial_actions=params.get("initial_actions"),
                        Emax=params.get("Emax"),
                        L=params.get("L"),
                        delta_chg=params.get("delta_chg"),
                        eps_feas=float(params.get("eps_feas", 1e-7)),
                        solve_time_limit_s=params.get("solve_time_limit_s"),
                    ),
                    C_out_base,
                    C_ret_tau,
                    R_out_vec,
                    R_ret_vec,
                )
                dm_ret[tau] = float(ub_plus_r - ub_base)

            # Expand to the full master variable grid, not just sparse incumbent keys.
            for q in master_q_idx:
                for tau, v in dm_out.items():
                    if 0 <= tau < T and abs(float(v)) > 0.0:
                        coeff_y_out[(int(q), int(tau))] = float(v)
                for tau, v in dm_ret.items():
                    if 0 <= tau < T and abs(float(v)) > 0.0:
                        coeff_y_ret[(int(q), int(tau))] = float(v)

            return coeff_y_out, coeff_y_ret, dm_out, dm_ret

        # If scenarios provided, iterate; else use single-demand params
        if scenarios:
            if weights and len(weights) != len(scenarios):
                raise ValueError("scenario_weights must match number of scenarios")
            if not weights:
                weights = [1.0 / len(scenarios)] * len(scenarios)

            cuts: list[Cut] = []
            ub_vals: list[float] = []
            consts: list[float] = []
            consts_out: list[float] = []
            consts_ret: list[float] = []
            coeffs_out_list: list[Dict[tuple[int, int], float]] = []
            coeffs_ret_list: list[Dict[tuple[int, int], float]] = []
            scenario_diags: list[dict] = []
            scenario_records: list[dict] = []
            agg = {
                "objective_value": [],
                "waiting_cost_slots": [],
                "fill_eps_cost": [],
                "penalty_cost": [],
                "penalty_pax": [],
                "served_total": [],
                "total_demand": [],
            }

            for idx_s, s in enumerate(scenarios):
                if isinstance(s, (str, Path)):
                    R_out, R_ret = _load_demand_from_file(s, T)
                    scen_label = str(s)
                elif isinstance(s, dict) and (
                    "requests" in s or "req_matrix" in s or "R_out" in s or "R_ret" in s
                ):
                    R_out, R_ret = _aggregate_requests(s, T)
                    scen_label = str(s.get("name") or s.get("label") or "scenario")
                else:
                    # Best effort
                    R_out = list(getattr(s, "R_out", [0.0] * T))
                    R_ret = list(getattr(s, "R_ret", [0.0] * T))
                    scen_label = str(getattr(s, "name", "scenario"))
                R_out = (R_out + [0.0] * T)[:T]
                R_ret = (R_ret + [0.0] * T)[:T]

                # If using dual slopes, force at least one layer per time to create capacity constraints
                use_dual = _cut_mode_cfg == "dual"
                K_out_lp = (
                    [max(1, int(K_out[t])) for t in range(T)] if use_dual else K_out
                )
                K_ret_lp = (
                    [max(1, int(K_ret[t])) for t in range(T)] if use_dual else K_ret
                )
                sp_params = SPParams(
                    T=T,
                    Wmax_slots=Wmax,
                    p=p_pen,
                    lp_solver=lp_solver,
                    S=S,
                    K_out=K_out_lp,
                    K_ret=K_ret_lp,
                    fill_eps=fill_eps,
                    solver_options=solver_options,
                    eps_cut=eps_cut,
                    slot_resolution=slot_res,
                    time_step_minutes=time_step_min,
                    T_minutes=params.get("T_minutes"),
                    trip_duration_minutes=params.get("trip_duration_minutes"),
                    trip_slots=params.get("trip_slots"),
                    Q=int(params.get("Q", len(q_idx) if q_idx else 0) or 0),
                    binit=params.get("binit"),
                    initial_actions=params.get("initial_actions"),
                    Emax=params.get("Emax"),
                    L=params.get("L"),
                    delta_chg=params.get("delta_chg"),
                    eps_feas=float(params.get("eps_feas", 1e-7)),
                    debug_timing=debug_timing,
                    debug_solver_tee=bool(params.get("debug_solver_tee", False)),
                    debug_export_lp_iteration=params.get("debug_export_lp_iteration"),
                    debug_current_iteration=int(
                        params.get("debug_current_iteration", -1) or -1
                    ),
                    debug_report_dir=params.get(
                        "debug_report_dir", params.get("report_dir", "Report")
                    ),
                    debug_force_nominal_departures=bool(
                        params.get("debug_force_nominal_departures", False)
                    ),
                    debug_scenario_label=str(scen_label),
                    solve_time_limit_s=params.get("solve_time_limit_s"),
                    recourse_resolution=str(
                        params.get("recourse_resolution", "slot")
                    ),
                    Wmax_minutes=params.get("Wmax_minutes"),
                    departure_policy=str(
                        params.get("departure_policy", "midpoint")
                    ),
                    request_minutes=_minutes_for(s),
                )
                t_solve0 = time.perf_counter()
                duals, ub_val = solve_subproblem(
                    sp_params, C_out, C_ret, R_out, R_ret, candidate
                )
                t_solve1 = time.perf_counter()
                sp_solve_time = t_solve1 - t_solve0
                _dbg(
                    "[SP TIMING] iter=%s scenario=%s solve_total=%.3fs build=%.3fs solve=%.3fs extract=%.3fs post=%.3fs"
                    % (
                        str(params.get("debug_current_iteration", "-")),
                        scen_label,
                        float(sp_solve_time),
                        float(duals.get("timing_build_s", 0.0) or 0.0),
                        float(duals.get("timing_solve_s", 0.0) or 0.0),
                        float(duals.get("timing_extract_s", 0.0) or 0.0),
                        float(duals.get("timing_postprocess_s", 0.0) or 0.0),
                    )
                )
                if not bool(duals.get("is_feasible", True)):
                    scenario_diags.append(
                        {
                            "label": scen_label,
                            "T": T,
                            "R_out": [float(R_out[t]) for t in range(T)],
                            "R_ret": [float(R_ret[t]) for t in range(T)],
                            "infeasible": True,
                            "infeasibility_reason": duals.get("infeasibility_reason"),
                            "first_violation": duals.get("first_violation"),
                            "timing_sp_solve_s": sp_solve_time,
                            "timing_cutgen_s": 0.0,
                        }
                    )
                    return SubproblemResult(
                        is_feasible=False,
                        upper_bound=None,
                        diagnostics={
                            "T": T,
                            "scenarios": scenario_diags,
                            "scenario_weights": (
                                list(weights) if weights is not None else None
                            ),
                            "infeasible": True,
                            "infeasibility_reason": duals.get("infeasibility_reason"),
                            "first_violation": duals.get("first_violation"),
                            "slot_resolution": int(params.get("slot_resolution", 1)),
                            "timing_sp_solve_s": sp_solve_time,
                            "timing_cutgen_s": 0.0,
                        },
                    )
                ub_vals.append(ub_val)

                scenario_records.append(
                    {
                        "idx": int(idx_s),
                        "label": scen_label,
                        "R_out": R_out,
                        "R_ret": R_ret,
                        "duals": duals,
                        "ub_val": float(ub_val),
                        "sp_solve_time": float(sp_solve_time),
                    }
                )
                agg["objective_value"].append(
                    float(duals.get("objective_value", ub_val))
                )
                agg["waiting_cost_slots"].append(
                    float(duals.get("waiting_cost_slots", 0.0))
                )
                agg["fill_eps_cost"].append(float(duals.get("fill_eps_cost", 0.0)))
                agg["penalty_cost"].append(float(duals.get("penalty_cost", 0.0)))
                agg["penalty_pax"].append(float(duals.get("penalty_pax", 0.0)))
                agg["served_total"].append(float(duals.get("served_total", 0.0)))
                agg["total_demand"].append(float(duals.get("total_demand", 0.0)))
                agg.setdefault("timing_sp_solve_s", []).append(sp_solve_time)

            # Aggregate UB
            if ub_aggregation == "mean":
                ub_val_agg = sum(w * u for w, u in zip(weights, ub_vals))
                agg_obj = sum(w * u for w, u in zip(weights, agg["objective_value"]))
                agg_wait = sum(
                    w * u for w, u in zip(weights, agg["waiting_cost_slots"])
                )
                agg_fill = sum(w * u for w, u in zip(weights, agg["fill_eps_cost"]))
                agg_pen = sum(w * u for w, u in zip(weights, agg["penalty_cost"]))
                agg_pen_pax = sum(w * u for w, u in zip(weights, agg["penalty_pax"]))
                agg_served = sum(w * u for w, u in zip(weights, agg["served_total"]))
                agg_total = sum(w * u for w, u in zip(weights, agg["total_demand"]))
            elif ub_aggregation == "sum":
                ub_val_agg = sum(ub_vals)
                agg_obj = sum(agg["objective_value"])
                agg_wait = sum(agg["waiting_cost_slots"])
                agg_fill = sum(agg["fill_eps_cost"])
                agg_pen = sum(agg["penalty_cost"])
                agg_pen_pax = sum(agg["penalty_pax"])
                agg_served = sum(agg["served_total"])
                agg_total = sum(agg["total_demand"])
            elif ub_aggregation == "max":
                ub_val_agg = max(ub_vals)
                idx_max = int(ub_vals.index(ub_val_agg))
                agg_obj = agg["objective_value"][idx_max]
                agg_wait = agg["waiting_cost_slots"][idx_max]
                agg_fill = agg["fill_eps_cost"][idx_max]
                agg_pen = agg["penalty_cost"][idx_max]
                agg_pen_pax = agg["penalty_pax"][idx_max]
                agg_served = agg["served_total"][idx_max]
                agg_total = agg["total_demand"][idx_max]
            else:
                raise ValueError("ub_aggregation must be one of 'mean', 'sum', 'max'")

            if debug_skip_cut_generation:
                for rec in scenario_records:
                    duals = rec["duals"]
                    R_out = rec["R_out"]
                    R_ret = rec["R_ret"]
                    scenario_diags.append(
                        {
                            "label": rec["label"],
                            "T": T,
                            "R_out": [float(R_out[t]) for t in range(T)],
                            "R_ret": [float(R_ret[t]) for t in range(T)],
                            "pax_out_by_tau": list(
                                duals.get("served_out_by_tau", [0.0] * T)
                            ),
                            "pax_ret_by_tau": list(
                                duals.get("served_ret_by_tau", [0.0] * T)
                            ),
                            "pax_out_by_tau_k": list(
                                duals.get("served_out_by_tau_k", [[] for _ in range(T)])
                            ),
                            "pax_ret_by_tau_k": list(
                                duals.get("served_ret_by_tau_k", [[] for _ in range(T)])
                            ),
                            "objective_value": float(
                                duals.get("objective_value", rec["ub_val"])
                            ),
                            "waiting_cost_slots": float(
                                duals.get("waiting_cost_slots", 0.0)
                            ),
                            "fill_eps_cost": float(duals.get("fill_eps_cost", 0.0)),
                            "penalty_cost": float(duals.get("penalty_cost", 0.0)),
                            "penalty_pax": float(duals.get("penalty_pax", 0.0)),
                            "served_total": float(duals.get("served_total", 0.0)),
                            "total_demand": float(duals.get("total_demand", 0.0)),
                            "realized_departures": list(
                                duals.get("realized_departures", [])
                            ),
                            "realized_departure_min_map": dict(
                                duals.get("realized_departure_min_map", {})
                            ),
                            "refined_departure_diagnostics": list(
                                duals.get("refined_departure_diagnostics", [])
                            ),
                            "refined_departure_diagnostics_focus": list(
                                duals.get("refined_departure_diagnostics_focus", [])
                            ),
                            "effective_pre_service": list(
                                duals.get("effective_pre_service", [])
                            ),
                            "battery_trajectory": duals.get("battery_trajectory", {}),
                            "timing_sp_solve_s": rec["sp_solve_time"],
                            "timing_cutgen_s": 0.0,
                            "cut_generation_skipped": True,
                        }
                    )
                return SubproblemResult(
                    is_feasible=True,
                    cuts=[],
                    upper_bound=ub_val_agg,
                    diagnostics={
                        "T": T,
                        "scenarios": scenario_diags,
                        "scenario_weights": (
                            list(weights) if weights is not None else None
                        ),
                        "objective_value": agg_obj,
                        "waiting_cost_slots": agg_wait,
                        "fill_eps_cost": agg_fill,
                        "penalty_cost": agg_pen,
                        "penalty_pax": agg_pen_pax,
                        "served_total": agg_served,
                        "total_demand": agg_total,
                        "slot_resolution": int(params.get("slot_resolution", 1)),
                        "timing_sp_solve_s": sum(
                            float(x) for x in agg.get("timing_sp_solve_s", []) or []
                        ),
                        "timing_cutgen_s": 0.0,
                        "cut_generation_mode": "skipped_debug",
                    },
                )

            # Early-exit for scalar theta in multi-scenario runs (skip all cut generation)
            theta_val = _cand_theta("__theta")
            has_theta_s = any(
                isinstance(k, str) and k.startswith("__theta_s[")
                for k in candidate.keys()
            )
            if debug_early_exit:
                try:
                    rhs = float(ub_val_agg) - float(eps_cut) * (
                        1.0 + abs(float(ub_val_agg))
                    )
                except Exception:
                    rhs = None
                try:
                    theta_keys = [
                        k
                        for k in candidate.keys()
                        if isinstance(k, str) and k.startswith("__theta")
                    ][:20]
                except Exception:
                    theta_keys = []
                early_exit_ok = (
                    (theta_val is not None)
                    and (not has_theta_s)
                    and _ok(theta_val, float(ub_val_agg), eps_cut)
                )
                msg = (
                    "[EARLY-EXIT] scenarios=%s theta_val=%s has_theta_s=%s ub_val_agg=%s eps_cut=%s rhs=%s early_exit_ok=%s theta_keys=%s"
                    % (
                        str(len(scenarios)),
                        (f"{float(theta_val):.6g}" if theta_val is not None else "-"),
                        str(bool(has_theta_s)),
                        (f"{float(ub_val_agg):.6g}" if ub_val_agg is not None else "-"),
                        (f"{float(eps_cut):.6g}" if eps_cut is not None else "-"),
                        (f"{float(rhs):.6g}" if rhs is not None else "-"),
                        str(bool(early_exit_ok)),
                        str(theta_keys),
                    )
                )
                try:
                    import logging as _logging

                    _log = _logging.getLogger(__name__)
                    _log.setLevel(_logging.INFO)
                    if not _log.handlers:
                        _logging.basicConfig(level=_logging.INFO)
                    _log.info(msg)
                except Exception:
                    self._vprint(msg)
            if (
                (theta_val is not None)
                and (not has_theta_s)
                and _ok(theta_val, float(ub_val_agg), eps_cut)
            ):
                for rec in scenario_records:
                    duals = rec["duals"]
                    R_out = rec["R_out"]
                    R_ret = rec["R_ret"]
                    scenario_diags.append(
                        {
                            "label": rec["label"],
                            "T": T,
                            "R_out": [float(R_out[t]) for t in range(T)],
                            "R_ret": [float(R_ret[t]) for t in range(T)],
                            "pax_out_by_tau": list(
                                duals.get("served_out_by_tau", [0.0] * T)
                            ),
                            "pax_ret_by_tau": list(
                                duals.get("served_ret_by_tau", [0.0] * T)
                            ),
                            "pax_out_by_tau_k": list(
                                duals.get("served_out_by_tau_k", [[] for _ in range(T)])
                            ),
                            "pax_ret_by_tau_k": list(
                                duals.get("served_ret_by_tau_k", [[] for _ in range(T)])
                            ),
                            "objective_value": float(
                                duals.get("objective_value", rec["ub_val"])
                            ),
                            "waiting_cost_slots": float(
                                duals.get("waiting_cost_slots", 0.0)
                            ),
                            "fill_eps_cost": float(duals.get("fill_eps_cost", 0.0)),
                            "penalty_cost": float(duals.get("penalty_cost", 0.0)),
                            "penalty_pax": float(duals.get("penalty_pax", 0.0)),
                            "served_total": float(duals.get("served_total", 0.0)),
                            "total_demand": float(duals.get("total_demand", 0.0)),
                            "timing_sp_solve_s": rec["sp_solve_time"],
                            "timing_cutgen_s": 0.0,
                        }
                    )
                return SubproblemResult(
                    is_feasible=True,
                    cuts=[],
                    upper_bound=ub_val_agg,
                    diagnostics={
                        "T": T,
                        "scenarios": scenario_diags,
                        "scenario_weights": (
                            list(weights) if weights is not None else None
                        ),
                        "objective_value": agg_obj,
                        "waiting_cost_slots": agg_wait,
                        "fill_eps_cost": agg_fill,
                        "penalty_cost": agg_pen,
                        "penalty_pax": agg_pen_pax,
                        "served_total": agg_served,
                        "total_demand": agg_total,
                        "slot_resolution": int(params.get("slot_resolution", 1)),
                        "timing_sp_solve_s": sum(
                            float(x) for x in agg.get("timing_sp_solve_s", []) or []
                        ),
                        "timing_cutgen_s": 0.0,
                    },
                )

            # Phase 2: generate cuts (unless early-exit above)
            use_dual = _cut_mode_cfg == "dual"
            K_out_lp = [max(1, int(K_out[t])) for t in range(T)] if use_dual else K_out
            K_ret_lp = [max(1, int(K_ret[t])) for t in range(T)] if use_dual else K_ret
            for rec in scenario_records:
                idx_s = int(rec["idx"])
                scen_label = rec["label"]
                R_out = rec["R_out"]
                R_ret = rec["R_ret"]
                duals = rec["duals"]
                ub_val = float(rec["ub_val"])
                sp_solve_time = float(rec["sp_solve_time"])

                # Early-exit per scenario if per-scenario theta is already consistent
                # Skip any cut generation work for this scenario.
                theta_s = _cand_theta(f"__theta_s[{int(idx_s)}]")
                if (theta_s is not None) and _ok(theta_s, float(ub_val), eps_cut):
                    cutgen_time = 0.0
                    # Collect per-scenario diagnostics for reporting
                    scenario_diags.append(
                        {
                            "label": scen_label,
                            "T": T,
                            "R_out": [float(R_out[t]) for t in range(T)],
                            "R_ret": [float(R_ret[t]) for t in range(T)],
                            "pax_out_by_tau": list(
                                duals.get("served_out_by_tau", [0.0] * T)
                            ),
                            "pax_ret_by_tau": list(
                                duals.get("served_ret_by_tau", [0.0] * T)
                            ),
                            "pax_out_by_tau_k": list(
                                duals.get("served_out_by_tau_k", [[] for _ in range(T)])
                            ),
                            "pax_ret_by_tau_k": list(
                                duals.get("served_ret_by_tau_k", [[] for _ in range(T)])
                            ),
                            "objective_value": float(
                                duals.get("objective_value", ub_val)
                            ),
                            "waiting_cost_slots": float(
                                duals.get("waiting_cost_slots", 0.0)
                            ),
                            "fill_eps_cost": float(duals.get("fill_eps_cost", 0.0)),
                            "penalty_cost": float(duals.get("penalty_cost", 0.0)),
                            "penalty_pax": float(duals.get("penalty_pax", 0.0)),
                            "served_total": float(duals.get("served_total", 0.0)),
                            "total_demand": float(duals.get("total_demand", 0.0)),
                            "timing_sp_solve_s": sp_solve_time,
                            "timing_cutgen_s": cutgen_time,
                        }
                    )
                    agg.setdefault("timing_cutgen_s", []).append(cutgen_time)
                    continue

                # Build marginal slopes either from duals (fast) or finite differences (fallback)
                t_cut0 = time.perf_counter()
                cut_mode_used = "dual" if use_dual else "finite_difference"
                proxy_diag: dict[str, Any] = {}
                if mw_enabled:
                    # MW-selected dual slopes on optimal face
                    # Ensure at least one capacity layer per tau for dual π variables
                    # Layer counts no longer shape the dual: there is one pi per slot
                    # regardless. Passed through only for signature compatibility.
                    K_out_mw = list(K_out_lp)
                    K_ret_mw = list(K_ret_lp)
                    dm_pair = solve_mw_dual(
                        T,
                        Wmax,
                        p_pen,
                        S,
                        K_out_mw,
                        K_ret_mw,
                        C_out,
                        C_ret,
                        R_out,
                        R_ret,
                        Ybar_out,
                        Ybar_ret,
                        ub_val,
                        lp_solver,
                        solver_options,
                    )
                    if dm_pair is None:
                        # Fall back to the PLAIN DUAL, not to finite differences (S1).
                        #
                        # `dm = S*pi` from `solve_subproblem` is already computed and
                        # already sitting in `duals`. It is a valid lower-bounding cut --
                        # validity comes from dual feasibility (handout 77) -- merely not
                        # Pareto-optimal. Finite-difference slopes carry no such
                        # guarantee, so the previous fallback discarded a valid cut it
                        # was holding in favour of an invalid one, and then set
                        # cut_lb_valid = False, which makes solver.py drop best_lb for
                        # the WHOLE run. One MW hiccup voided the run's bound.
                        mw_sol = slopes_from_capacity_duals(duals, S, T)
                        cut_mode_used = "mw_dual_fallback"
                        self._vprint(
                            f"[SP WARN] scenario={scen_label}: solve_mw_dual returned no "
                            "solution; fell back to the plain capacity duals. The cut is "
                            "still a valid lower bound, but it is not Pareto-optimal."
                        )
                    else:
                        mw_sol = dm_pair
                        cut_mode_used = "mw"
                    dm_out, dm_ret = mw_sol.dm_out, mw_sol.dm_ret
                    c_out_map, c_ret_map = expand_slopes_to_candidate(
                        candidate, dm_out, dm_ret, T
                    )
                elif use_dual:
                    mw_sol = slopes_from_capacity_duals(duals, S, T)
                    dm_out, dm_ret = mw_sol.dm_out, mw_sol.dm_ret
                    c_out_map, c_ret_map = expand_slopes_to_candidate(
                        candidate, dm_out, dm_ret, T
                    )
                else:
                    # Finite-difference coefficients and constant per scenario.
                    # Diagnostic only: not a certified lower bound (handout 75/76).
                    mw_sol = None
                    c_out_map, c_ret_map, dm_out, dm_ret = coeffs_by_fdiff(
                        ub_val, C_out, C_ret, K_out, K_ret, R_out, R_ret
                    )
                # Validity follows from the mode that was actually used, looked up in
                # one place. Setting it by hand per branch is what let the flag and the
                # branch disagree.
                cut_lb_valid = cut_mode_carries_lower_bound(cut_mode_used)
                fallback_diag: dict[str, Any] = {}
                t_cut1 = time.perf_counter()
                cutgen_time = t_cut1 - t_cut0
                _dbg(
                    "[SP TIMING] iter=%s scenario=%s cutgen=%.3fs mode=%s proxy_obj=%s"
                    % (
                        str(params.get("debug_current_iteration", "-")),
                        scen_label,
                        float(cutgen_time),
                        cut_mode_used,
                        (
                            f"{float(proxy_diag.get('proxy_recourse_objective')):.6g}"
                            if proxy_diag.get("proxy_recourse_objective") is not None
                            else "-"
                        ),
                    )
                )

                sum_y_out = [
                    float(C_out[tau]) / S if S != 0 else 0.0 for tau in range(T)
                ]
                sum_y_ret = [
                    float(C_ret[tau]) / S if S != 0 else 0.0 for tau in range(T)
                ]
                # `solve_subproblem` splits its objective by direction exactly, so
                # ub_out + ub_ret == ub_val. Read them rather than defaulting: the
                # defaults below were dead (both keys are always present) and the
                # `ub_out = const` one would have been wrong if ever taken.
                ub_out = float(duals["ub_out"])
                ub_ret = float(duals["ub_ret"])
                const_out, const_ret, dual_diag = _cut_intercepts(
                    mw_sol,
                    y_inc_out=sum_y_out,
                    y_inc_ret=sum_y_ret,
                    ub_out=ub_out,
                    ub_ret=ub_ret,
                    T=T,
                    ub_val=ub_val,
                    dm_out=dm_out,
                    dm_ret=dm_ret,
                )
                const = const_out + const_ret
                consts.append(const)
                coeffs_out_list.append(c_out_map)
                coeffs_ret_list.append(c_ret_map)
                consts_out.append(const_out)
                consts_ret.append(const_ret)
                # Evaluate line at incumbent for diagnostics
                theta_lb_s = (
                    float(const)
                    + sum(
                        float(v)
                        * _cand_float(candidate, f"yOUT[{int(q)},{int(tau)}]", 0.0)
                        for (q, tau), v in c_out_map.items()
                    )
                    + sum(
                        float(v)
                        * _cand_float(candidate, f"yRET[{int(q)},{int(tau)}]", 0.0)
                        for (q, tau), v in c_ret_map.items()
                    )
                )
                target_val = float(ub_val)
                tight_tol = _tightness_tolerance(eps_cut, target_val, dual_diag)
                if abs(float(target_val) - float(theta_lb_s)) > tight_tol:
                    raise RuntimeError(
                        f"cut tightness failed at the incumbent (scenario {scen_label}): "
                        f"the assembled cut evaluates to {theta_lb_s:.12g} where the "
                        f"subproblem optimum is {target_val:.12g} "
                        f"(diff {target_val - theta_lb_s:.3g}, tol {tight_tol:.3g}). "
                        "The intercept itself was already checked against the duals, so "
                        "a failure here points at the slope broadcast rather than at the "
                        "constant: a coefficient on the wrong (q,tau), a slope dropped, "
                        "or a candidate key the expansion did not recognise."
                    )
                cuts.append(
                    Cut(
                        name=f"opt_cut_s_{int(idx_s)}",
                        cut_type=CutType.OPTIMALITY,
                        metadata={
                            "const": const,
                            "const_out": const_out,
                            "const_ret": const_ret,
                            "coeff_yOUT": c_out_map,
                            "theta_lb": float(theta_lb_s),
                            "coeff_yRET": c_ret_map,
                            "recourse_total": float(target_val),
                            "recourse_out": float(duals.get("ub_out", 0.0)),
                            "recourse_ret": float(duals.get("ub_ret", 0.0)),
                            "scenario_index": int(idx_s),
                            "cut_valid_lower_bound": bool(cut_lb_valid),
                        },
                    )
                )
                # Collect per-scenario diagnostics for reporting
                scenario_diags.append(
                    {
                        "label": scen_label,
                        "T": T,
                        "R_out": [float(R_out[t]) for t in range(T)],
                        "R_ret": [float(R_ret[t]) for t in range(T)],
                        "pax_out_by_tau": list(
                            duals.get("served_out_by_tau", [0.0] * T)
                        ),
                        "pax_ret_by_tau": list(
                            duals.get("served_ret_by_tau", [0.0] * T)
                        ),
                        "pax_out_by_tau_k": list(
                            duals.get("served_out_by_tau_k", [[] for _ in range(T)])
                        ),
                        "pax_ret_by_tau_k": list(
                            duals.get("served_ret_by_tau_k", [[] for _ in range(T)])
                        ),
                        "objective_value": float(duals.get("objective_value", ub_val)),
                        "waiting_cost_slots": float(
                            duals.get("waiting_cost_slots", 0.0)
                        ),
                        "fill_eps_cost": float(duals.get("fill_eps_cost", 0.0)),
                        "penalty_cost": float(duals.get("penalty_cost", 0.0)),
                        "penalty_pax": float(duals.get("penalty_pax", 0.0)),
                        "served_total": float(duals.get("served_total", 0.0)),
                        "total_demand": float(duals.get("total_demand", 0.0)),
                        "realized_departures": list(
                            duals.get("realized_departures", [])
                        ),
                        "realized_departure_min_map": dict(
                            duals.get("realized_departure_min_map", {})
                        ),
                        "refined_departure_diagnostics": list(
                            duals.get("refined_departure_diagnostics", [])
                        ),
                        "refined_departure_diagnostics_focus": list(
                            duals.get("refined_departure_diagnostics_focus", [])
                        ),
                        "effective_pre_service": list(
                            duals.get("effective_pre_service", [])
                        ),
                        "battery_trajectory": duals.get("battery_trajectory", {}),
                        "timing_sp_solve_s": sp_solve_time,
                        "timing_cutgen_s": cutgen_time,
                        "cut_generation_mode": cut_mode_used,
                        "mw_core_point_seeded": bool(core_seeded),
                        # The core point MW ACTUALLY maximised over. An all-zero Ybar gives
                        # the MW objective no direction, so it is seeded, and S3 then projects
                        # it into the master region -- either step can make this differ from
                        # what the caller passed. The Pareto claim is about THIS point;
                        # comparing dominance at the passed vector compares at a direction MW
                        # never saw. Reported because a silent substitution is how that goes
                        # unnoticed for as long as it did.
                        "mw_core_point_used": {
                            "Yout": list(Ybar_out),
                            "Yret": list(Ybar_ret),
                        },
                        "cut_generation_proxy": proxy_diag,
                        "cut_generation_fallback": fallback_diag,
                        "cut_valid_lower_bound": bool(cut_lb_valid),
                        "cut_intercept": dict(dual_diag),
                    }
                )
                agg.setdefault("timing_cutgen_s", []).append(cutgen_time)

            # Aggregate UB
            if ub_aggregation == "mean":
                ub_val_agg = sum(w * u for w, u in zip(weights, ub_vals))
                agg_obj = sum(w * u for w, u in zip(weights, agg["objective_value"]))
                agg_wait = sum(
                    w * u for w, u in zip(weights, agg["waiting_cost_slots"])
                )
                agg_fill = sum(w * u for w, u in zip(weights, agg["fill_eps_cost"]))
                agg_pen = sum(w * u for w, u in zip(weights, agg["penalty_cost"]))
                agg_pen_pax = sum(w * u for w, u in zip(weights, agg["penalty_pax"]))
                agg_served = sum(w * u for w, u in zip(weights, agg["served_total"]))
                agg_total = sum(w * u for w, u in zip(weights, agg["total_demand"]))
                agg_sp_time = sum(
                    w * u for w, u in zip(weights, agg["timing_sp_solve_s"])
                )
                agg_cut_time = sum(
                    w * u for w, u in zip(weights, agg["timing_cutgen_s"])
                )
            elif ub_aggregation == "sum":
                ub_val_agg = sum(ub_vals)
                agg_obj = sum(agg["objective_value"])
                agg_wait = sum(agg["waiting_cost_slots"])
                agg_fill = sum(agg["fill_eps_cost"])
                agg_pen = sum(agg["penalty_cost"])
                agg_pen_pax = sum(agg["penalty_pax"])
                agg_served = sum(agg["served_total"])
                agg_total = sum(agg["total_demand"])
                agg_sp_time = sum(agg["timing_sp_solve_s"])
                agg_cut_time = sum(agg["timing_cutgen_s"])
            elif ub_aggregation == "max":
                ub_val_agg = max(ub_vals)
                idx_max = int(ub_vals.index(ub_val_agg))
                agg_obj = agg["objective_value"][idx_max]
                agg_wait = agg["waiting_cost_slots"][idx_max]
                agg_fill = agg["fill_eps_cost"][idx_max]
                agg_pen = agg["penalty_cost"][idx_max]
                agg_pen_pax = agg["penalty_pax"][idx_max]
                agg_served = agg["served_total"][idx_max]
                agg_total = agg["total_demand"][idx_max]
                agg_sp_time = agg["timing_sp_solve_s"][idx_max]
                agg_cut_time = agg["timing_cutgen_s"][idx_max]
            else:
                raise ValueError("ub_aggregation must be one of 'mean', 'sum', 'max'")
            # A scenario that generated NO cut is not a scenario with an unknown cut.
            #
            # The theta early-exit skips cut generation for a scenario whose theta already
            # covers its recourse, and that scenario's diagnostics therefore carry neither
            # `cut_generation_mode` nor `cut_valid_lower_bound`. Reading the absence as
            # "unknown" made the conjunction False and dropped the run's lower bound --
            # measured in D64 on `theta_by_scen`, 2 of 96 iterations, reported as
            # `mixed(mw+unknown)` and `Bounds are heuristic only`.
            #
            # `CutValidity.NO_CUT` already names the right semantics: "no cut this
            # iteration; previously established validity is untouched" (benders/types.py).
            # So the aggregate is valid when every scenario that DID produce a cut was
            # valid, and the contributors are what the mode label reports.
            #
            # Still fail-closed where it matters: a scenario that produced a cut and said
            # nothing about it is genuinely UNKNOWN and still poisons the aggregate, and an
            # aggregate with no contributors at all is not valid.
            _contributors = [
                sd for sd in scenario_diags if "cut_valid_lower_bound" in sd
            ]
            _abstained = len(scenario_diags) - len(_contributors)
            agg_cut_lb_valid = (
                all(bool(sd["cut_valid_lower_bound"]) for sd in _contributors)
                if _contributors
                else False
            )
            # Report which cut generator produced this aggregate. A single label when
            # every contributing scenario agreed, otherwise "mixed(a+b)" -- never silently
            # blank, since a reported number must state the mode that produced it.
            # Abstaining scenarios are counted, not folded in as a mode: "no_cut(1)" says
            # one scenario contributed nothing, which is a different fact from a cut whose
            # provenance is unknown.
            _modes = sorted(
                {
                    str(sd.get("cut_generation_mode", "unknown"))
                    for sd in _contributors
                }
            )
            agg_cut_mode = (
                _modes[0]
                if len(_modes) == 1
                else (f"mixed({'+'.join(_modes)})" if _modes else "no_cut")
            )
            if _abstained:
                agg_cut_mode = f"{agg_cut_mode}+no_cut({_abstained})"

            if not multi_cuts:
                # Weighted average of constants and coefficients
                const_avg = sum(w * c for w, c in zip(weights, consts))
                const_out_avg = (
                    sum(w * c for w, c in zip(weights, consts_out))
                    if consts_out
                    else const_avg
                )
                const_ret_avg = (
                    sum(w * c for w, c in zip(weights, consts_ret))
                    if consts_ret
                    else 0.0
                )
                keys_out = set().union(*[set(d.keys()) for d in coeffs_out_list])
                keys_ret = set().union(*[set(d.keys()) for d in coeffs_ret_list])
                avg_out: Dict[tuple[int, int], float] = {}
                avg_ret: Dict[tuple[int, int], float] = {}
                for k in keys_out:
                    avg_out[k] = sum(
                        weights[i] * coeffs_out_list[i].get(k, 0.0)
                        for i in range(len(coeffs_out_list))
                    )
                for k in keys_ret:
                    avg_ret[k] = sum(
                        weights[i] * coeffs_ret_list[i].get(k, 0.0)
                        for i in range(len(coeffs_ret_list))
                    )
                cut = Cut(
                    name="opt_cut_avg",
                    cut_type=CutType.OPTIMALITY,
                    metadata={
                        "const": const_avg,
                        "const_out": const_out_avg,
                        "const_ret": const_ret_avg,
                        "coeff_yOUT": avg_out,
                        "coeff_yRET": avg_ret,
                        "recourse_total": float(ub_val_agg),
                        "recourse_out": float(
                            sum(
                                weights[i]
                                * scenario_records[i]["duals"].get("ub_out", 0.0)
                                for i in range(len(scenario_records))
                            )
                        ),
                        "recourse_ret": float(
                            sum(
                                weights[i]
                                * scenario_records[i]["duals"].get("ub_ret", 0.0)
                                for i in range(len(scenario_records))
                            )
                        ),
                        "cut_generation_mode": agg_cut_mode,
                        "cut_valid_lower_bound": bool(agg_cut_lb_valid),
                    },
                )
                return SubproblemResult(
                    is_feasible=True,
                    cut=cut,
                    upper_bound=ub_val_agg,
                    diagnostics={
                        "T": T,
                        "scenarios": scenario_diags,
                        "scenario_weights": (
                            list(weights) if weights is not None else None
                        ),
                        "objective_value": agg_obj,
                        "waiting_cost_slots": agg_wait,
                        "fill_eps_cost": agg_fill,
                        "penalty_cost": agg_pen,
                        "penalty_pax": agg_pen_pax,
                        "served_total": agg_served,
                        "total_demand": agg_total,
                        "slot_resolution": int(params.get("slot_resolution", 1)),
                        "timing_sp_solve_s": agg_sp_time,
                        "timing_cutgen_s": agg_cut_time,
                        "cut_generation_mode": agg_cut_mode,
                        "cut_valid_lower_bound": bool(agg_cut_lb_valid),
                    },
                )
            else:
                return SubproblemResult(
                    is_feasible=True,
                    cuts=cuts,
                    upper_bound=ub_val_agg,
                    diagnostics={
                        "T": T,
                        "scenarios": scenario_diags,
                        "scenario_weights": (
                            list(weights) if weights is not None else None
                        ),
                        "objective_value": agg_obj,
                        "waiting_cost_slots": agg_wait,
                        "fill_eps_cost": agg_fill,
                        "penalty_cost": agg_pen,
                        "penalty_pax": agg_pen_pax,
                        "served_total": agg_served,
                        "total_demand": agg_total,
                        "slot_resolution": int(params.get("slot_resolution", 1)),
                        "timing_sp_solve_s": agg_sp_time,
                        "timing_cutgen_s": agg_cut_time,
                        "cut_generation_mode": agg_cut_mode,
                        "cut_valid_lower_bound": bool(agg_cut_lb_valid),
                    },
                )
        else:
            # Single-demand case from params (prefer external file if given)
            if single_scenario_override is not None:
                R_out, R_ret = single_scenario_override
            elif params.get("demand_file"):
                R_out, R_ret = _load_demand_from_file(params.get("demand_file"), T)
            elif (
                ("requests" in params)
                or ("req_matrix" in params)
                or ("R_out" in params)
                or ("R_ret" in params)
            ):
                R_out, R_ret = _aggregate_requests(params, T)
            else:
                R_out = [0.0] * T
                R_ret = [0.0] * T
            if len(R_out) != T:
                R_out = (R_out + [0.0] * T)[:T]
            if len(R_ret) != T:
                R_ret = (R_ret + [0.0] * T)[:T]

            # If using dual slopes, ensure at least one layer to create capacity constraints
            use_dual = _cut_mode_cfg == "dual"
            K_out_lp = [max(1, int(K_out[t])) for t in range(T)] if use_dual else K_out
            K_ret_lp = [max(1, int(K_ret[t])) for t in range(T)] if use_dual else K_ret
            sp_params = SPParams(
                T=T,
                Wmax_slots=Wmax,
                p=p_pen,
                lp_solver=lp_solver,
                S=S,
                K_out=K_out_lp,
                K_ret=K_ret_lp,
                fill_eps=fill_eps,
                solver_options=solver_options,
                eps_cut=eps_cut,
                slot_resolution=slot_res,
                time_step_minutes=time_step_min,
                T_minutes=params.get("T_minutes"),
                trip_duration_minutes=params.get("trip_duration_minutes"),
                trip_slots=params.get("trip_slots"),
                Q=int(params.get("Q", len(q_idx) if q_idx else 0) or 0),
                binit=params.get("binit"),
                initial_actions=params.get("initial_actions"),
                Emax=params.get("Emax"),
                L=params.get("L"),
                delta_chg=params.get("delta_chg"),
                eps_feas=float(params.get("eps_feas", 1e-7)),
                debug_timing=debug_timing,
                debug_solver_tee=bool(params.get("debug_solver_tee", False)),
                debug_export_lp_iteration=params.get("debug_export_lp_iteration"),
                debug_current_iteration=int(
                    params.get("debug_current_iteration", -1) or -1
                ),
                debug_report_dir=params.get(
                    "debug_report_dir", params.get("report_dir", "Report")
                ),
                debug_force_nominal_departures=bool(
                    params.get("debug_force_nominal_departures", False)
                ),
                debug_scenario_label="single",
                solve_time_limit_s=params.get("solve_time_limit_s"),
                recourse_resolution=str(params.get("recourse_resolution", "slot")),
                Wmax_minutes=params.get("Wmax_minutes"),
                departure_policy=str(params.get("departure_policy", "midpoint")),
                request_minutes=_minutes_for(single_demand_source),
            )
            t_solve0 = time.perf_counter()
            duals, ub_val = solve_subproblem(
                sp_params, C_out, C_ret, R_out, R_ret, candidate
            )
            t_solve1 = time.perf_counter()
            sp_solve_time = t_solve1 - t_solve0
            _dbg(
                "[SP TIMING] iter=%s scenario=single solve_total=%.3fs build=%.3fs solve=%.3fs extract=%.3fs post=%.3fs"
                % (
                    str(params.get("debug_current_iteration", "-")),
                    float(sp_solve_time),
                    float(duals.get("timing_build_s", 0.0) or 0.0),
                    float(duals.get("timing_solve_s", 0.0) or 0.0),
                    float(duals.get("timing_extract_s", 0.0) or 0.0),
                    float(duals.get("timing_postprocess_s", 0.0) or 0.0),
                )
            )
            if not bool(duals.get("is_feasible", True)):
                diagnostics = {
                    "T": T,
                    "R_out": [float(R_out[t]) for t in range(T)],
                    "R_ret": [float(R_ret[t]) for t in range(T)],
                    "infeasible": True,
                    "infeasibility_reason": duals.get("infeasibility_reason"),
                    "first_violation": duals.get("first_violation"),
                    "slot_resolution": int(params.get("slot_resolution", 1)),
                    "timing_sp_solve_s": sp_solve_time,
                    "timing_cutgen_s": 0.0,
                }
                return SubproblemResult(
                    is_feasible=False, upper_bound=None, diagnostics=diagnostics
                )

            if debug_skip_cut_generation:
                diagnostics = {
                    "T": T,
                    "R_out": [float(R_out[t]) for t in range(T)],
                    "R_ret": [float(R_ret[t]) for t in range(T)],
                    "pax_out_by_tau": list(duals.get("served_out_by_tau", [0.0] * T)),
                    "pax_ret_by_tau": list(duals.get("served_ret_by_tau", [0.0] * T)),
                    "pax_out_by_tau_k": list(
                        duals.get("served_out_by_tau_k", [[] for _ in range(T)])
                    ),
                    "pax_ret_by_tau_k": list(
                        duals.get("served_ret_by_tau_k", [[] for _ in range(T)])
                    ),
                    "objective_value": float(duals.get("objective_value", ub_val)),
                    "waiting_cost_slots": float(duals.get("waiting_cost_slots", 0.0)),
                    "fill_eps_cost": float(duals.get("fill_eps_cost", 0.0)),
                    "penalty_cost": float(duals.get("penalty_cost", 0.0)),
                    "penalty_pax": float(duals.get("penalty_pax", 0.0)),
                    "served_total": float(duals.get("served_total", 0.0)),
                    "total_demand": float(duals.get("total_demand", 0.0)),
                    "realized_departures": list(duals.get("realized_departures", [])),
                    "realized_departure_min_map": dict(
                        duals.get("realized_departure_min_map", {})
                    ),
                    "refined_departure_diagnostics": list(
                        duals.get("refined_departure_diagnostics", [])
                    ),
                    "refined_departure_diagnostics_focus": list(
                        duals.get("refined_departure_diagnostics_focus", [])
                    ),
                    "effective_pre_service": list(
                        duals.get("effective_pre_service", [])
                    ),
                    "battery_trajectory": duals.get("battery_trajectory", {}),
                    "slot_resolution": int(params.get("slot_resolution", 1)),
                    "timing_sp_solve_s": sp_solve_time,
                    "timing_cutgen_s": 0.0,
                    "cut_generation_mode": "skipped_debug",
                }
                return SubproblemResult(
                    is_feasible=True, upper_bound=ub_val, diagnostics=diagnostics
                )

            # Early-exit if scalar theta is already consistent
            theta_val = _cand_theta("__theta")
            if (theta_val is not None) and _ok(theta_val, float(ub_val), eps_cut):
                diagnostics = {
                    "T": T,
                    "R_out": [float(R_out[t]) for t in range(T)],
                    "R_ret": [float(R_ret[t]) for t in range(T)],
                    "pax_out_by_tau": list(duals.get("served_out_by_tau", [0.0] * T)),
                    "pax_ret_by_tau": list(duals.get("served_ret_by_tau", [0.0] * T)),
                    "pax_out_by_tau_k": list(
                        duals.get("served_out_by_tau_k", [[] for _ in range(T)])
                    ),
                    "pax_ret_by_tau_k": list(
                        duals.get("served_ret_by_tau_k", [[] for _ in range(T)])
                    ),
                    "objective_value": float(duals.get("objective_value", ub_val)),
                    "waiting_cost_slots": float(duals.get("waiting_cost_slots", 0.0)),
                    "fill_eps_cost": float(duals.get("fill_eps_cost", 0.0)),
                    "penalty_cost": float(duals.get("penalty_cost", 0.0)),
                    "penalty_pax": float(duals.get("penalty_pax", 0.0)),
                    "served_total": float(duals.get("served_total", 0.0)),
                    "total_demand": float(duals.get("total_demand", 0.0)),
                    "realized_departures": list(duals.get("realized_departures", [])),
                    "realized_departure_min_map": dict(
                        duals.get("realized_departure_min_map", {})
                    ),
                    "refined_departure_diagnostics": list(
                        duals.get("refined_departure_diagnostics", [])
                    ),
                    "refined_departure_diagnostics_focus": list(
                        duals.get("refined_departure_diagnostics_focus", [])
                    ),
                    "effective_pre_service": list(
                        duals.get("effective_pre_service", [])
                    ),
                    "battery_trajectory": duals.get("battery_trajectory", {}),
                    "slot_resolution": int(params.get("slot_resolution", 1)),
                    "timing_sp_solve_s": sp_solve_time,
                    "timing_cutgen_s": 0.0,
                }
                return SubproblemResult(
                    is_feasible=True, upper_bound=ub_val, diagnostics=diagnostics
                )

            # Build coefficients via MW, duals (fast) or finite differences (fallback)
            t_cut0 = time.perf_counter()
            cut_mode_used = "dual" if use_dual else "finite_difference"
            proxy_diag: dict[str, Any] = {}
            if mw_enabled:
                dm_pair = solve_mw_dual(
                    T,
                    Wmax,
                    p_pen,
                    S,
                    # Ensure at least one capacity layer per tau for dual π variables
                    [max(1, int(K_out_lp[t])) for t in range(T)],
                    [max(1, int(K_ret_lp[t])) for t in range(T)],
                    C_out,
                    C_ret,
                    R_out,
                    R_ret,
                    Ybar_out,
                    Ybar_ret,
                    ub_val,
                    lp_solver,
                    solver_options,
                )
                if dm_pair is None:
                    # Fall back to the PLAIN DUAL, not to finite differences (S1).
                    # See the twin site in the multi-scenario path for the reasoning:
                    # the plain dual is valid, is already computed, and the previous
                    # fallback threw it away for slopes that void the run's bound.
                    mw_sol = slopes_from_capacity_duals(duals, S, T)
                    cut_mode_used = "mw_dual_fallback"
                    self._vprint(
                        "[SP WARN] solve_mw_dual returned no solution; fell back to the "
                        "plain capacity duals. The cut is still a valid lower bound, but "
                        "it is not Pareto-optimal."
                    )
                else:
                    mw_sol = dm_pair
                    cut_mode_used = "mw"
                dm_out, dm_ret = mw_sol.dm_out, mw_sol.dm_ret
                c_out_map, c_ret_map = expand_slopes_to_candidate(
                    candidate, dm_out, dm_ret, T
                )
            elif use_dual:
                mw_sol = slopes_from_capacity_duals(duals, S, T)
                dm_out, dm_ret = mw_sol.dm_out, mw_sol.dm_ret
                c_out_map, c_ret_map = expand_slopes_to_candidate(
                    candidate, dm_out, dm_ret, T
                )
            else:
                # Finite differences: diagnostic only, not a certified lower bound
                # (handout 75/76).
                mw_sol = None
                c_out_map, c_ret_map, dm_out, dm_ret = coeffs_by_fdiff(
                    ub_val, C_out, C_ret, K_out, K_ret, R_out, R_ret
                )
            # Validity follows from the mode used; see the twin site's note.
            cut_lb_valid = cut_mode_carries_lower_bound(cut_mode_used)
            fallback_diag: dict[str, Any] = {}

            t_cut1 = time.perf_counter()
            cutgen_time = t_cut1 - t_cut0
            _dbg(
                "[SP TIMING] iter=%s scenario=single cutgen=%.3fs mode=%s proxy_obj=%s"
                % (
                    str(params.get("debug_current_iteration", "-")),
                    float(cutgen_time),
                    cut_mode_used,
                    (
                        f"{float(proxy_diag.get('proxy_recourse_objective')):.6g}"
                        if proxy_diag.get("proxy_recourse_objective") is not None
                        else "-"
                    ),
                )
            )

            # Number of vehicles per departure time from current candidate
            sum_y_out = [float(C_out[tau]) / S if S != 0 else 0.0 for tau in range(T)]
            sum_y_ret = [float(C_ret[tau]) / S if S != 0 else 0.0 for tau in range(T)]

            # Intercepts derived from the demand duals, with strong duality checked
            # against the imposed form (S2). `solve_subproblem` splits its objective by
            # direction exactly, so ub_out + ub_ret == ub_val.
            ub_out = float(duals["ub_out"])
            ub_ret = float(duals["ub_ret"])
            const_out, const_ret, dual_diag = _cut_intercepts(
                mw_sol,
                y_inc_out=sum_y_out,
                y_inc_ret=sum_y_ret,
                ub_out=ub_out,
                ub_ret=ub_ret,
                T=T,
                ub_val=ub_val,
                dm_out=dm_out,
                dm_ret=dm_ret,
            )
            const = const_out + const_ret

            # Optional: evaluate the line at the incumbent (theta_lb) to verify tightness (≈ ub_val)
            theta_lb = (
                float(const)
                + sum(
                    float(v) * _cand_float(candidate, f"yOUT[{int(q)},{int(tau)}]", 0.0)
                    for (q, tau), v in c_out_map.items()
                )
                + sum(
                    float(v) * _cand_float(candidate, f"yRET[{int(q)},{int(tau)}]", 0.0)
                    for (q, tau), v in c_ret_map.items()
                )
            )
            target_val = float(ub_val)
            tight_tol = _tightness_tolerance(eps_cut, target_val, dual_diag)
            if abs(float(target_val) - float(theta_lb)) > tight_tol:
                raise RuntimeError(
                    "cut tightness failed at the incumbent: the assembled cut evaluates "
                    f"to {theta_lb:.12g} where the subproblem optimum is "
                    f"{target_val:.12g} (diff {target_val - theta_lb:.3g}, "
                    f"tol {tight_tol:.3g}). The intercept itself was already checked "
                    "against the duals, so a failure here points at the slope broadcast "
                    "rather than at the constant: a coefficient on the wrong (q,tau), a "
                    "slope dropped, or a candidate key the expansion did not recognise."
                )

            # Emit cut metadata
            cut = Cut(
                name="opt_cut",
                cut_type=CutType.OPTIMALITY,
                metadata={
                    "const": float(const),
                    "const_out": float(const_out),
                    "const_ret": float(const_ret),
                    "coeff_yOUT": c_out_map,
                    "coeff_yRET": c_ret_map,
                    "recourse_total": float(target_val),
                    "recourse_out": float(duals.get("ub_out", 0.0)),
                    "recourse_ret": float(duals.get("ub_ret", 0.0)),
                    # diagnostics
                    "theta_lb": float(theta_lb),
                    "cut_valid_lower_bound": bool(cut_lb_valid),
                },
            )
            # Diagnostics: demand per slot split by direction and served pax per departure slot (OUT/RET)
            diagnostics = {
                "T": T,
                "R_out": [float(R_out[t]) for t in range(T)],
                "R_ret": [float(R_ret[t]) for t in range(T)],
                "pax_out_by_tau": list(duals.get("served_out_by_tau", [0.0] * T)),
                "pax_ret_by_tau": list(duals.get("served_ret_by_tau", [0.0] * T)),
                "pax_out_by_tau_k": list(
                    duals.get("served_out_by_tau_k", [[] for _ in range(T)])
                ),
                "pax_ret_by_tau_k": list(
                    duals.get("served_ret_by_tau_k", [[] for _ in range(T)])
                ),
                "objective_value": float(duals.get("objective_value", ub_val)),
                "waiting_cost_slots": float(duals.get("waiting_cost_slots", 0.0)),
                "fill_eps_cost": float(duals.get("fill_eps_cost", 0.0)),
                "penalty_cost": float(duals.get("penalty_cost", 0.0)),
                "penalty_pax": float(duals.get("penalty_pax", 0.0)),
                "served_total": float(duals.get("served_total", 0.0)),
                "total_demand": float(duals.get("total_demand", 0.0)),
                "realized_departures": list(duals.get("realized_departures", [])),
                "realized_departure_min_map": dict(
                    duals.get("realized_departure_min_map", {})
                ),
                "refined_departure_diagnostics": list(
                    duals.get("refined_departure_diagnostics", [])
                ),
                "refined_departure_diagnostics_focus": list(
                    duals.get("refined_departure_diagnostics_focus", [])
                ),
                "effective_pre_service": list(duals.get("effective_pre_service", [])),
                "battery_trajectory": duals.get("battery_trajectory", {}),
                "slot_resolution": int(params.get("slot_resolution", 1)),
                "timing_sp_solve_s": sp_solve_time,
                "timing_cutgen_s": cutgen_time,
                "cut_generation_mode": cut_mode_used,
                "mw_core_point_seeded": bool(core_seeded),
                # The core point MW ACTUALLY maximised over. An all-zero Ybar gives
                # the MW objective no direction, so it is seeded, and S3 then projects
                # it into the master region -- either step can make this differ from
                # what the caller passed. The Pareto claim is about THIS point;
                # comparing dominance at the passed vector compares at a direction MW
                # never saw. Reported because a silent substitution is how that goes
                # unnoticed for as long as it did.
                "mw_core_point_used": {
                    "Yout": list(Ybar_out),
                    "Yret": list(Ybar_ret),
                },
                "cut_generation_proxy": proxy_diag,
                "cut_generation_fallback": fallback_diag,
                "cut_valid_lower_bound": bool(cut_lb_valid),
                "cut_intercept": dict(dual_diag),
            }
            return SubproblemResult(
                is_feasible=True, cut=cut, upper_bound=ub_val, diagnostics=diagnostics
            )


# Demand aggregation lives at module level because the master needs the same
# post-truncation R vectors to build its recourse lower bound. Duplicating the
# truncation rule there would be how the two copies drift apart -- exactly what
# happened to initial_battery and initial_actions (D23).

def wmax_minutes_to_slots(wmax_minutes: float, slot_resolution: int) -> int:
    """Largest whole number of slots whose implied wait never exceeds `wmax_minutes`.

    FLOOR, not ceil. `ceil` grants MORE waiting than the config asked for: at 30-minute
    slots, `Wmax_minutes: 45` becomes ceil(1.5) = 2 slots, and a passenger may then be
    made to wait a full 60 minutes against a stated cap of 45. The cap is a service
    promise, so rounding it up is the one direction that must not happen silently.

    Measured consequence of the old behaviour, on the shipped settings: none. At
    Wmax_minutes 60 and slot_resolution 30 the quotient is exactly 2, so ceil and floor
    agree and no result in this repository changes. The fix matters for any Wmax that is
    not a whole multiple of the slot width, and for the finer grids the multi-resolution
    work introduces.

    NOTE what this does and does not guarantee. It bounds the wait measured from the
    START of the arrival slot to the START of the departure slot. A passenger arriving
    mid-slot waits less; a departure placed mid-slot (the midpoint convention of the
    research note) adds up to half a slot, which can push the true wait past the cap
    again -- at 30-minute slots, up to 75 minutes against a stated 60. That is a
    property of the slot ABSTRACTION, not of this conversion, and it is why
    `minute_pricer` enforces the real cap in minutes when it prices a schedule.
    """
    import math as _math

    slot_resolution = int(slot_resolution)
    if slot_resolution <= 0:
        raise ValueError(
            f"slot_resolution must be positive to convert Wmax_minutes, "
            f"got {slot_resolution!r}"
        )
    slots = int(_math.floor(float(wmax_minutes) / float(slot_resolution)))
    if slots < 1:
        raise ValueError(
            f"Wmax_minutes={wmax_minutes!r} is shorter than one slot "
            f"({slot_resolution} min), so no departure can serve any demand inside the "
            "cap and every passenger would be counted as unserved. Raise Wmax_minutes "
            "to at least the slot width, or refine slot_resolution. (The previous "
            "ceil() hid this by silently rounding the cap up to one slot.)"
        )
    return slots

def load_demand_doc(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Demand file not found: {path}")
    ext = path.suffix.lower()
    if ext == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if ext in {".yaml", ".yml"}:
        if _yaml is None:
            raise RuntimeError(
                "PyYAML is required to read YAML demand files. Install with 'pip install pyyaml'."
            )
        with path.open("r", encoding="utf-8") as f:
            return _yaml.safe_load(f)
    # Fallback: try JSON
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_requests(
    container: Any,
    Tlen: int,
    slot_res: int,
    on_warning: Any = None,
) -> tuple[list[float], list[float]]:
    R_out = [0.0 for _ in range(Tlen)]
    R_ret = [0.0 for _ in range(Tlen)]
    # Requests outside the horizon used to be dropped without a trace, so
    # "Pax served: 173/224" counted a denominator that had already lost 60
    # requests and understated unmet demand -- a headline metric weighted by p.
    dropped = {"after_horizon": 0, "negative_time": 0, "array_tail": 0}
    last_minute = [0.0]

    def _slot_idx_from_minutes(tmin: float) -> int:
        # Map continuous minutes to slot index via floor:
        # [0,res)->0, [res,2res)->1, ...
        res = max(1, slot_res)
        return max(0, int(math.floor(float(tmin) / res)))

    def _report_dropped() -> None:
        total = sum(dropped.values())
        if total <= 0:
            return
        parts = [f"{k}={v}" for k, v in dropped.items() if v]
        msg = (
            f"[DEMAND] {total} request(s) discarded outside the horizon "
            f"({', '.join(parts)}); horizon is {Tlen} slots of {max(1, slot_res)} min "
            f"= {Tlen * max(1, slot_res)} min, latest request at {last_minute[0]:.0f} min. "
            "Served/total counts below exclude them."
        )
        try:
            import logging as _logging

            _logging.getLogger(__name__).warning(msg)
        except Exception:
            pass
        try:
            if on_warning is not None:
                on_warning(msg)
        except Exception:
            pass

    if container is None:
        return R_out, R_ret
    # Direct arrays
    if isinstance(container, dict) and ("R_out" in container or "R_ret" in container):
        rout = list(container.get("R_out", [0.0] * Tlen))
        rret = list(container.get("R_ret", [0.0] * Tlen))
        for arr in (rout, rret):
            if len(arr) > Tlen:
                try:
                    dropped["array_tail"] += int(sum(float(x) for x in arr[Tlen:]))
                except Exception:
                    pass
        if len(rout) != Tlen:
            rout = (rout + [0.0] * Tlen)[:Tlen]
        if len(rret) != Tlen:
            rret = (rret + [0.0] * Tlen)[:Tlen]
        _report_dropped()
        return [float(x) for x in rout], [float(x) for x in rret]
    # Pull list from mapping under 'requests' or 'req_matrix'
    if isinstance(container, dict):
        container = container.get("requests") or container.get("req_matrix") or []
    # List of dicts [{dir,time}, ...]
    if isinstance(container, list) and container and isinstance(container[0], dict):
        for r in container:
            d = r.get("dir")
            try:
                tmin = float(r.get("time", -1))
            except Exception:
                continue
            if tmin < 0:
                dropped["negative_time"] += 1
                continue
            last_minute[0] = max(last_minute[0], float(tmin))
            # Floor-based slot mapping
            t = _slot_idx_from_minutes(tmin)
            if not (0 <= t < Tlen):
                dropped["after_horizon"] += 1
                continue
            if isinstance(d, str):
                dd = d.upper()
                if dd == "OUT":
                    R_out[t] += 1.0
                elif dd == "RET":
                    R_ret[t] += 1.0
            else:
                if int(d) == 0:
                    R_out[t] += 1.0
                else:
                    R_ret[t] += 1.0
        _report_dropped()
        return R_out, R_ret
    # Matrix [[dir,time], ...]
    if isinstance(container, list):
        for row in container:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            d, tt = row[0], row[1]
            try:
                tmin = float(tt)
            except Exception:
                continue
            if tmin < 0:
                dropped["negative_time"] += 1
                continue
            last_minute[0] = max(last_minute[0], float(tmin))
            t = _slot_idx_from_minutes(tmin)
            if not (0 <= t < Tlen):
                dropped["after_horizon"] += 1
                continue
            if isinstance(d, str):
                dd = d.upper()
                if dd == "OUT":
                    R_out[t] += 1.0
                elif dd == "RET":
                    R_ret[t] += 1.0
            else:
                if int(d) == 0:
                    R_out[t] += 1.0
                else:
                    R_ret[t] += 1.0
        _report_dropped()
        return R_out, R_ret
    return R_out, R_ret


def _cand_float(candidate: Candidate | None, name: str, default: float = 0.0) -> float:
    if candidate is None:
        return default
    try:
        return float(candidate.get(name, default))
    except Exception:
        return default


def _normalize_binit(P: "SPParams") -> list[float]:
    raw = P.binit
    if raw is None:
        return [0.0] * max(0, int(P.Q))
    if isinstance(raw, (int, float)):
        return [float(raw)] * max(0, int(P.Q))
    vals = [float(x) for x in list(raw)]
    if len(vals) < int(P.Q):
        fill = vals[-1] if vals else 0.0
        vals.extend([fill] * (int(P.Q) - len(vals)))
    return vals[: int(P.Q)]


def _report_dir_from_debug(path_like: Any) -> Path:
    try:
        p = Path(str(path_like)) if path_like else Path("Report")
    except Exception:
        p = Path("Report")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _model_size_stats(model: pyo.ConcreteModel) -> dict[str, int]:
    num_vars = 0
    num_bin = 0
    for var in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        num_vars += 1
        try:
            if var.is_binary():
                num_bin += 1
        except Exception:
            pass
    num_cons = sum(
        1
        for _ in model.component_data_objects(
            pyo.Constraint, active=True, descend_into=True
        )
    )
    return {
        "model_num_variables": int(num_vars),
        "model_num_binary_variables": int(num_bin),
        "model_num_constraints": int(num_cons),
    }


def _apply_solver_time_limit(solver: Any, seconds: float | None) -> None:
    if seconds is None:
        return
    try:
        limit = float(seconds)
    except Exception:
        return
    if not math.isfinite(limit) or limit <= 0.0:
        return
    try:
        solver.options["timelimit"] = limit
    except Exception:
        pass


def _has_loaded_solution(model: pyo.ConcreteModel) -> bool:
    for var in model.component_data_objects(pyo.Var, active=True, descend_into=True):
        try:
            if var.value is not None:
                return True
        except Exception:
            continue
    return False


def _maybe_export_lp(
    model: pyo.ConcreteModel, P: "SPParams"
) -> tuple[Optional[str], float]:
    export_iter = getattr(P, "debug_export_lp_iteration", None)
    current_iter = int(getattr(P, "debug_current_iteration", -1) or -1)
    if export_iter is None or current_iter < 0 or int(export_iter) != current_iter:
        return None, 0.0
    out_dir = _report_dir_from_debug(getattr(P, "debug_report_dir", None))
    label = str(getattr(P, "debug_scenario_label", "single") or "single")
    safe_label = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label
    )
    lp_path = out_dir / f"{model.name}_iter_{current_iter}_{safe_label}.lp"
    t0 = time.perf_counter()
    model.write(str(lp_path), io_options={"symbolic_solver_labels": True})
    t1 = time.perf_counter()
    return str(lp_path), float(t1 - t0)


@dataclass
class SPParams:
    T: int
    Wmax_slots: int
    p: float
    lp_solver: str
    S: float
    K_out: list[int]
    K_ret: list[int]
    fill_eps: float = 0.0
    solver_options: dict | None = None
    eps_cut: float = 1e-8
    slot_resolution: int = 1
    time_step_minutes: int = 1
    T_minutes: int | None = None
    trip_duration_minutes: int | None = None
    trip_slots: int | None = None
    Q: int = 0
    binit: Any = None
    initial_actions: Any = None
    Emax: float | None = None
    L: float | None = None
    delta_chg: float | None = None
    eps_feas: float = 1e-7
    debug_timing: bool = False
    debug_solver_tee: bool = False
    debug_export_lp_iteration: int | None = None
    debug_current_iteration: int = -1
    debug_report_dir: str | None = None
    debug_force_nominal_departures: bool = False
    debug_scenario_label: str | None = None
    solve_time_limit_s: float | None = None
    # Multi-resolution recourse (D51). "slot" is the model this repository has always
    # had; "minute" evaluates operations on a fixed minute grid while the first stage
    # stays on slots. The capacity rows stay indexed by departure slot in both, so the
    # cut the master receives is the same object either way.
    recourse_resolution: str = "slot"
    Wmax_minutes: float | None = None
    departure_policy: str = "midpoint"
    request_minutes: dict | None = None


def solve_subproblem(
    P: SPParams,
    C_out: Iterable[float],
    C_ret: Iterable[float],
    R_out: Iterable[float],
    R_ret: Iterable[float],
    candidate: Candidate | None = None,
):
    """Replicates user's subproblem sketch and returns duals and objective.

    Returns (duals: dict[str, dict[int, float]], objective_value: float)
    """
    if str(getattr(P, "recourse_resolution", "slot")).lower() == "minute":
        # Minute-level recourse. Same dual interface (one pi per departure slot), so
        # every caller above -- cut construction, aggregation, validity, the master --
        # is unchanged. See minute_pricer.solve_minute_recourse and D51.
        from ..minute_pricer import solve_minute_recourse

        if not P.request_minutes:
            raise ValueError(
                "recourse_resolution='minute' needs per-request arrival minutes, but "
                "request_minutes is empty. The slot aggregation discards them at load, "
                "so the demand file has to be read again in minute form -- silently "
                "falling back to the slot model here would report a multi-resolution "
                "run that never happened."
            )
        if P.Wmax_minutes is None:
            raise ValueError(
                "recourse_resolution='minute' needs Wmax_minutes. Wmax_slots cannot "
                "substitute: the whole point is that a slot window is not an exact "
                "number of minutes (D51)."
            )
        return solve_minute_recourse(
            T=P.T,
            slot_resolution=P.slot_resolution,
            wmax_minutes=float(P.Wmax_minutes),
            p_slots=float(P.p),
            C_out=list(C_out),
            C_ret=list(C_ret),
            request_minutes=P.request_minutes,
            policy=str(P.departure_policy),
            lp_solver=P.lp_solver,
            solver_options=P.solver_options,
            solve_time_limit_s=P.solve_time_limit_s,
        )

    t_build0 = time.perf_counter()
    m = pyo.ConcreteModel()
    m.name = "subproblem"
    Tset = range(P.T)

    C_out = list(C_out)
    C_ret = list(C_ret)
    R_out = list(R_out)
    R_ret = list(R_ret)

    W = P.Wmax_slots

    # Define valid arcs with causality and max-wait: (t + 1) <= tau <= min(T-1, t+W)
    # Interpretation: demand aggregated in slot t (arrivals during [t*res, (t+1)*res))
    # can be served by the next slot's departure at the earliest (tau = t+1).
    # Same-slot service (tau = t) is disallowed to avoid serving passengers after the slot's departure.
    Arcs_list = [
        (t, tau) for t in Tset for tau in Tset if (t + 1) <= tau <= min(P.T - 1, t + W)
    ]
    m.Arcs = pyo.Set(initialize=Arcs_list, dimen=2, ordered=False)

    # Arcs carry no layer index.
    #
    # This model used to split each departure slot into K_d[tau] "layers", one per
    # vehicle, so that a per-layer epsilon could encourage packing the first vehicle
    # before the second. K_d[tau] comes from the master's y, so the LAYERS -- and
    # therefore the variable set AND the constraint set -- changed with y.
    #
    # Benders duality requires y to enter only through the right-hand side. When the
    # constraint matrix itself moves with y, the dual of one instance is not a
    # subgradient of the recourse across y, and NO cut generator can be valid on top
    # of it. Measured before this change: cuts forced theta to 6893 (directional),
    # 5290 (single theta) and 6087 (plain dual) at a schedule whose true recourse is
    # 4183.00. The mechanism was pi[tau] summing K layer duals with dm = S*pi, giving
    # slopes about K times too steep, which over-estimates whenever the evaluated y
    # has less capacity than the incumbent.
    #
    # The layers were redundant in capacity: K layers of min(S, S*K) = S each total
    # K*S = S*sum_q y_d[q,tau], which is exactly the aggregated right-hand side used
    # below. K = 0 gives no arcs, matching zero capacity. Only the fill_first_epsilon
    # tie-break between otherwise identical assignments is lost.
    m.ArcsOut = pyo.Set(initialize=Arcs_list, dimen=2, ordered=False)
    m.ArcsRet = pyo.Set(initialize=Arcs_list, dimen=2, ordered=False)

    m.x_OUT = pyo.Var(m.ArcsOut, within=pyo.NonNegativeReals)
    m.x_RET = pyo.Var(m.ArcsRet, within=pyo.NonNegativeReals)
    m.u_OUT = pyo.Var(Tset, within=pyo.NonNegativeReals)
    m.u_RET = pyo.Var(Tset, within=pyo.NonNegativeReals)

    def wait_cost(t: int, tau: int) -> float:
        return float(max(0, tau - t))

    m.obj = pyo.Objective(
        expr=sum(wait_cost(t, tau) * m.x_OUT[t, tau] for (t, tau) in m.ArcsOut)
        + sum(wait_cost(t, tau) * m.x_RET[t, tau] for (t, tau) in m.ArcsRet)
        + P.p * (sum(m.u_OUT[t] for t in Tset) + sum(m.u_RET[t] for t in Tset)),
        sense=pyo.minimize,
    )

    def cons_dem_OUT(m, t):
        return (
            sum(m.x_OUT[t, tau] for tau in Tset if (t, tau) in m.ArcsOut) + m.u_OUT[t]
            == R_out[t]
        )

    m.D_out = pyo.Constraint(Tset, rule=cons_dem_OUT)

    def cons_dem_RET(m, t):
        return (
            sum(m.x_RET[t, tau] for tau in Tset if (t, tau) in m.ArcsRet) + m.u_RET[t]
            == R_ret[t]
        )

    m.D_ret = pyo.Constraint(Tset, rule=cons_dem_RET)

    # One capacity row per departure slot, right-hand side linear in y:
    #   sum_t x_d[t,tau] <= C_d[tau] = S * sum_q y_d[q,tau]
    # Fixed structure, so pi_d[tau] is a genuine dual and dm = S*pi a valid subgradient.
    def cap_out_rule(m, tau):
        ts = [t for t in Tset if (t, tau) in m.ArcsOut]
        if not ts:
            return pyo.Constraint.Skip
        return sum(m.x_OUT[t, tau] for t in ts) <= float(C_out[tau])

    m.Cap_out = pyo.Constraint(Tset, rule=cap_out_rule)

    def cap_ret_rule(m, tau):
        ts = [t for t in Tset if (t, tau) in m.ArcsRet]
        if not ts:
            return pyo.Constraint.Skip
        return sum(m.x_RET[t, tau] for t in ts) <= float(C_ret[tau])

    m.Cap_ret = pyo.Constraint(Tset, rule=cap_ret_rule)

    # Dual suffix required to read duals
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    build_stats = _model_size_stats(m)
    lp_path = None
    lp_export_time = 0.0
    try:
        lp_path, lp_export_time = _maybe_export_lp(m, P)
    except Exception:
        lp_path, lp_export_time = (None, 0.0)
    t_build1 = time.perf_counter()

    solver = pyo.SolverFactory(P.lp_solver)
    # Allow tuning via options, e.g., for CPLEX: {"lpmethod": 2, "threads": 0, "parallel": 1}
    try:
        if getattr(P, "solver_options", None):
            for k, v in (P.solver_options or {}).items():
                solver.options[k] = v
    except Exception:
        pass
    _apply_solver_time_limit(solver, getattr(P, "solve_time_limit_s", None))
    if getattr(P, "debug_timing", False):
        print(
            "[SP DEBUG] iter=%s scenario=%s nominal build=%.3fs vars=%s bin=%s cons=%s lp=%s"
            % (
                str(getattr(P, "debug_current_iteration", -1)),
                str(getattr(P, "debug_scenario_label", "single")),
                float(t_build1 - t_build0),
                build_stats["model_num_variables"],
                build_stats["model_num_binary_variables"],
                build_stats["model_num_constraints"],
                lp_path or "-",
            )
        )
    t_solve0 = time.perf_counter()
    res = solver.solve(
        m, tee=bool(getattr(P, "debug_solver_tee", False)), load_solutions=False
    )
    term = getattr(res.solver, "termination_condition", None)
    if term not in (pyo.TerminationCondition.optimal,):
        # Retry with presolve off if possible
        try:
            solver.options["preind"] = 0
            solver.options["presolve"] = 0
            solver.options["reduce"] = 0
        except Exception:
            pass
        res = solver.solve(
            m, tee=bool(getattr(P, "debug_solver_tee", False)), load_solutions=False
        )
        term = getattr(res.solver, "termination_condition", None)
        if term not in (pyo.TerminationCondition.optimal,):
            try:
                out_dir = _report_dir_from_debug(getattr(P, "debug_report_dir", None))
                fail_lp_path = out_dir / "subproblem_failed.lp"
                m.write(str(fail_lp_path), io_options={"symbolic_solver_labels": True})
                print(f"[SP] Wrote LP: {fail_lp_path}")
                if candidate is not None:
                    cand_path = out_dir / "subproblem_failed_candidate.txt"
                    with cand_path.open("w", encoding="utf-8") as f:
                        for k, v in sorted(candidate.items()):
                            f.write(f"{k}={v}\n")
                    print(f"[SP] Wrote candidate: {cand_path}")
            except Exception:
                pass
            raise RuntimeError(
                f"Subproblem solve ambiguous: termination_condition={term}"
            )
    t_solve1 = time.perf_counter()
    # Load solution only after optimal termination.
    #
    # HANDLER_CENSUS.md Category A, and the worst-placed of them. Everything below
    # reads duals through `m.dual.get(..., 0.0)`, so a swallowed failure here does
    # not raise later -- it produces an all-zero slope vector, which is a cut that
    # constrains nothing, carries no error, and is still reported as valid. Every
    # ambiguous termination already raised above, so reaching this line and failing
    # to load has no benign reading.
    try:
        m.solutions.load_from(res)
    except (ValueError, AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            f"the subproblem terminated {term} but its solution could not be "
            f"loaded: {type(exc).__name__}: {exc}. The duals below would all read "
            "as 0.0 and the cut would constrain nothing while claiming to be a "
            "valid lower bound."
        ) from exc
    t_extract0 = time.perf_counter()

    alpha_OUT = {t: float(m.dual.get(m.D_out[t], 0.0)) for t in Tset}
    alpha_RET = {t: float(m.dual.get(m.D_ret[t], 0.0)) for t in Tset}
    # One dual per departure slot. This used to sum K layer duals, which is what made
    # dm = S*pi about K times too steep and the cuts invalid.
    pi_OUT = {
        tau: (float(m.dual.get(m.Cap_out[tau], 0.0)) if tau in m.Cap_out else 0.0)
        for tau in Tset
    }
    pi_RET = {
        tau: (float(m.dual.get(m.Cap_ret[tau], 0.0)) if tau in m.Cap_ret else 0.0)
        for tau in Tset
    }

    # Gather simple primal summaries
    served_out_by_tau = [0.0 for _ in Tset]
    served_ret_by_tau = [0.0 for _ in Tset]
    # Also collect per-layer (per shuttle) served counts at each departure slot
    served_out_by_tau_k = [[] for _ in Tset]
    served_ret_by_tau_k = [[] for _ in Tset]

    def _split_across_vehicles(total: float, kmax: int, seats: float) -> list[float]:
        """Assign an aggregated flow to individual vehicles, filling each in turn.

        Capacity is now one row per slot, so the model no longer says which vehicle
        carried whom -- and it never needed to: the vehicles at a slot are identical.
        This reproduces what the per-layer fill_first_epsilon used to arrange, so the
        per-shuttle report still accounts for every served passenger.
        """
        out = [0.0 for _ in range(max(0, kmax))]
        left = float(total)
        for k in range(len(out)):
            take = min(float(seats), max(0.0, left))
            out[k] = take
            left -= take
        if left > 1e-9 and out:
            out[-1] += left
        return out

    for tau in Tset:
        kmax_out = int(P.K_out[tau]) if tau < len(P.K_out) else 0
        kmax_ret = int(P.K_ret[tau]) if tau < len(P.K_ret) else 0
        total_out_tau = sum(
            float(pyo.value(m.x_OUT[t, tau])) for t in Tset if (t, tau) in m.ArcsOut
        )
        total_ret_tau = sum(
            float(pyo.value(m.x_RET[t, tau])) for t in Tset if (t, tau) in m.ArcsRet
        )
        served_out_by_tau[tau] = total_out_tau
        served_ret_by_tau[tau] = total_ret_tau
        if kmax_out > 0:
            served_out_by_tau_k[tau] = _split_across_vehicles(
                total_out_tau, kmax_out, float(P.S)
            )
        if kmax_ret > 0:
            served_ret_by_tau_k[tau] = _split_across_vehicles(
                total_ret_tau, kmax_ret, float(P.S)
            )

    # Component costs (per direction)
    try:
        out_cost_val = sum(
            wait_cost(t, tau) * float(pyo.value(m.x_OUT[t, tau]))
            for (t, tau) in m.ArcsOut
        )
        out_cost_val += float(P.p) * sum(float(pyo.value(m.u_OUT[t])) for t in Tset)
    except Exception:
        out_cost_val = 0.0
    try:
        ret_cost_val = sum(
            wait_cost(t, tau) * float(pyo.value(m.x_RET[t, tau]))
            for (t, tau) in m.ArcsRet
        )
        ret_cost_val += float(P.p) * sum(float(pyo.value(m.u_RET[t])) for t in Tset)
    except Exception:
        ret_cost_val = 0.0

    # Component costs
    wait_cost_slots = 0.0
    fill_eps_cost = 0.0
    # (contrib, t, tau, direction, x) with direction 0 = OUT, 1 = RET.
    #
    # The fourth field used to be the capacity-layer index `k`. D30 removed the
    # layers; the OUT branch was updated to a literal 0 and the RET branch was left
    # referencing `k`, which no longer exists in this scope. That is a
    # NameError waiting for its trigger, and the trigger is rare: the append only
    # runs when an individual arc contributes negatively, i.e. when the LP returns a
    # slightly negative flow. It fired at iteration 123 of the LP-only run and took
    # the whole run down -- the diagnostic built to report a numerical anomaly was
    # destroyed by the anomaly it was built to report.
    #
    # Direction is the useful thing to record here now that there are no layers:
    # when this does fire, the first question is which side it came from.
    neg_contribs: list[tuple[float, int, int, int, float]] = []
    for t, tau in m.ArcsOut:
        val = float(pyo.value(m.x_OUT[t, tau]) or 0.0)
        if val == 0.0:
            continue
        w = float(max(0, tau - t))
        wait_cost_slots += w * val
        contrib = w * val
        if contrib < -1e-9:
            neg_contribs.append((contrib, int(t), int(tau), 0, val))
    for t, tau in m.ArcsRet:
        val = float(pyo.value(m.x_RET[t, tau]) or 0.0)
        if val == 0.0:
            continue
        w = float(max(0, tau - t))
        wait_cost_slots += w * val
        contrib = w * val
        if contrib < -1e-9:
            neg_contribs.append((contrib, int(t), int(tau), 1, val))

    penalty_pax = float(
        sum(float(pyo.value(m.u_OUT[t])) for t in Tset)
        + sum(float(pyo.value(m.u_RET[t])) for t in Tset)
    )
    penalty_cost = float(P.p) * penalty_pax

    obj_val = float(pyo.value(m.obj))
    # Strong duality check
    try:
        # One capacity row per slot with right-hand side C[tau] = S*sum_q y[q,tau].
        # This used to be min(S, C[tau]) because the row was per vehicle layer.
        cap_out_rhs = [float(C_out[tau]) for tau in Tset]
        cap_ret_rhs = [float(C_ret[tau]) for tau in Tset]
        dual_obj = sum(float(R_out[t]) * alpha_OUT[t] for t in Tset) + sum(
            float(R_ret[t]) * alpha_RET[t] for t in Tset
        )
        dual_obj += sum(cap_out_rhs[tau] * pi_OUT[tau] for tau in Tset)
        dual_obj += sum(cap_ret_rhs[tau] * pi_RET[tau] for tau in Tset)
        eps_cut = float(getattr(P, "eps_cut", 1e-8)) if hasattr(P, "eps_cut") else 1e-8
        if abs(float(obj_val) - float(dual_obj)) > eps_cut * max(
            1.0, abs(float(obj_val))
        ):
            raise RuntimeError(
                f"Strong duality check failed: primal={obj_val} dual={dual_obj}"
            )
    except Exception as exc:
        try:
            out_dir = _report_dir_from_debug(getattr(P, "debug_report_dir", None))
            fail_lp_path = out_dir / "subproblem_duality_failed.lp"
            m.write(str(fail_lp_path), io_options={"symbolic_solver_labels": True})
            print(f"[SP] Wrote LP: {fail_lp_path}")
        except Exception:
            pass
        raise
    sum_components = wait_cost_slots + fill_eps_cost + penalty_cost
    if abs(sum_components - obj_val) > 1e-5:
        print(
            "[SP DIAG] Objective mismatch: obj=%.6g wait=%.6g fill_eps=%.6g penalty=%.6g sum=%.6g"
            % (obj_val, wait_cost_slots, fill_eps_cost, penalty_cost, sum_components)
        )
        assert abs(sum_components - obj_val) <= 1e-5
    if wait_cost_slots < -1e-9:
        neg_contribs.sort(key=lambda x: x[0])
        print("[SP DIAG] Negative waiting cost detected. Top negative contributions:")
        for c, t, tau, d, val in neg_contribs[:10]:
            print(
                f"  contrib={c:.6g} dir={'OUT' if d == 0 else 'RET'} "
                f"t={t} tau={tau} x={val:.6g}"
            )
        assert wait_cost_slots >= -1e-9
    if penalty_cost < -1e-9:
        print(f"[SP DIAG] Negative penalty cost detected: {penalty_cost:.6g}")
        assert penalty_cost >= -1e-9
    total_demand = float(sum(R_out) + sum(R_ret))
    served_total = float(sum(served_out_by_tau) + sum(served_ret_by_tau))
    t_extract1 = time.perf_counter()
    t_post0 = time.perf_counter()
    # Consistency check: served + unmet == total demand (within tolerance)
    if abs((served_total + penalty_pax) - total_demand) > 1e-5:
        print(
            "[SP DIAG] Demand mismatch: total=%.6g served=%.6g unmet=%.6g"
            % (total_demand, served_total, penalty_pax)
        )
        # Do not assert here; keep diagnostic only
    t_post1 = time.perf_counter()

    return (
        {
            "alpha_OUT": alpha_OUT,
            "alpha_RET": alpha_RET,
            # Demand side of the dual objective, as the scalar the cut constant needs.
            # Slot-indexed here; the minute recourse forms the same scalar over arrival
            # minutes. See MWDual for why the interface is a scalar and not a vector.
            "intercept_out": float(
                sum(alpha_OUT[t] * float(R_out[t]) for t in Tset)
            ),
            "intercept_ret": float(
                sum(alpha_RET[t] * float(R_ret[t]) for t in Tset)
            ),
            "pi_OUT": pi_OUT,
            "pi_RET": pi_RET,
            # diagnostics
            "served_out_by_tau": served_out_by_tau,
            "served_ret_by_tau": served_ret_by_tau,
            # per-layer diagnostics (per departure layer k at each tau)
            "served_out_by_tau_k": served_out_by_tau_k,
            "served_ret_by_tau_k": served_ret_by_tau_k,
            # components
            "ub_out": float(out_cost_val),
            "ub_ret": float(ret_cost_val),
            "objective_value": obj_val,
            "waiting_cost_slots": wait_cost_slots,
            "fill_eps_cost": fill_eps_cost,
            "penalty_cost": penalty_cost,
            "penalty_pax": penalty_pax,
            "served_total": served_total,
            "total_demand": total_demand,
            "timing_build_s": float(t_build1 - t_build0),
            "timing_solve_s": float(t_solve1 - t_solve0),
            "timing_extract_s": float(t_extract1 - t_extract0),
            "timing_postprocess_s": float(t_post1 - t_post0),
            "timing_lp_export_s": float(lp_export_time),
            "exported_lp_path": lp_path,
            **build_stats,
        },
        obj_val,
    )
