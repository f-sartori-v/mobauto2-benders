"""Benders cuts injected inside one CPLEX branch-and-bound tree (D44).

The loop tears down a ~19 000-node tree and rebuilds it from scratch to add a
single cut. D45 measured the consequence: with 150 LP cuts seeding the root, one
102 s master solve lifts the bound 794.62 -> ~1080 and stops at ~20% internal
gap, and eight further iterations produce no trend, because each one re-truncates
at the same place. This module is the direct attack on that: one tree, cuts added
at incumbents as the search finds them.

Two things about this file are not stylistic choices.

**A lazy cut cannot be un-added.** D39's contract is fail-closed at the level of
the reported bound: an INVALID or UNKNOWN cut is added to the master and the lower
bound is dropped afterwards. That is sound only where the loop owns the bound and
can retract it at the end. Inside a branch-and-bound tree there is nothing to
retract: a `finite_difference` slope that excludes the true optimum prunes a
subtree, CPLEX never revisits it, and no later accounting recovers what was lost.
So the rule inverts here -- only CutValidity.VALID may be injected, and every
other state aborts the solve.

*Updated by S1.* The example used to be `mw_fdiff_fallback`, the mode the
Magnanti-Wong path fell into when its auxiliary LP declined. That mode no longer
exists: MW now falls back to the plain capacity duals, which are a valid lower
bound, so an MW failure inside a tree is survivable rather than fatal to it.
`finite_difference` remains the one generator with no guarantee, and it is
diagnostic-only. The rule above is unchanged -- it just has one fewer way to fire.

**Aborting is not the same as skipping.** Returning from a LazyConstraintCallback
without calling `add` tells CPLEX the incumbent is acceptable, which is exactly
the wrong message when the subproblem just failed to price it. The deleted
pre-D30 implementation carried `# If SP fails, skip adding lazy cut` and that is
the failure class D39 closed everywhere else. `CallbackAbort` exists so the
failure leaves the callback, survives CPLEX swallowing exceptions, and is
re-raised on the outside where it can stop the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from .types import Candidate, Cut, CutValidity, SubproblemResult, classify_cut_validity

__all__ = [
    "CallbackAbort",
    "LazyCutStats",
    "cut_to_linear_form",
    "candidate_from_values",
    "vet_cut_for_injection",
    "build_variable_name_map",
    "register_lazy_callback",
]


class CallbackAbort(RuntimeError):
    """Raised inside the callback when the solve must not continue.

    CPLEX's Python callback layer will not propagate this on its own -- an
    exception raised inside a callback is reported as a solve failure with the
    original traceback lost. The registrar stores the instance and re-raises it
    after `solve()` returns, so the reason survives to the caller.
    """

    def __init__(self, reason: str, *, validity: CutValidity | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.validity = validity


@dataclass(slots=True)
class LazyCutStats:
    """What the callback did, for the manifest.

    A branch-and-cut run cannot report "N iterations" -- there is no loop. These
    counters are the replacement, and `aborted_reason` is the field that stops a
    truncated tree being read as a completed one.
    """

    invocations: int = 0
    cuts_injected: int = 0
    incumbents_accepted: int = 0
    subproblem_solves: int = 0
    validity_counts: dict[str, int] = field(default_factory=dict)
    aborted_reason: Optional[str] = None

    def note_validity(self, validity: CutValidity) -> None:
        key = validity.value
        self.validity_counts[key] = self.validity_counts.get(key, 0) + 1


def cut_to_linear_form(
    cut: Cut,
) -> tuple[float, dict[tuple[str, int, int], float]]:
    """Canonicalise a Cut into `(const, {(dir, q, t): coeff})` for `theta >= ...`.

    The master's `_add_cut` reads the same three metadata fields and then spends
    ~800 lines on filtering, aggregation, duplicate signatures and tightness
    reporting -- all of which belong to the loop, none of which a callback can
    use. This extracts only the algebra.

    Raises rather than returning an empty form when the metadata is missing. A
    cut with no coefficients is the all-zero slope vector HANDLER_CENSUS warns
    about; injected into a tree it is `theta >= const`, which is not obviously
    wrong and is unrecoverable if it is.
    """
    meta = cut.metadata if isinstance(cut.metadata, dict) else {}
    if "coeff_yOUT" not in meta and "coeff_yRET" not in meta:
        raise CallbackAbort(
            f"cut {cut.name!r} carries no coeff_yOUT/coeff_yRET metadata; "
            "its slope cannot be reconstructed and a slope-less cut must not "
            "enter the tree"
        )

    const_out = meta.get("const_out")
    const_ret = meta.get("const_ret")
    if const_out is not None and const_ret is not None:
        const = float(const_out) + float(const_ret)
    else:
        const = float(meta.get("const", 0.0) or 0.0)

    coeffs: dict[tuple[str, int, int], float] = {}
    for prefix, key in (("yOUT", "coeff_yOUT"), ("yRET", "coeff_yRET")):
        block = meta.get(key)
        if not isinstance(block, Mapping):
            continue
        for (q, t), v in block.items():
            val = float(v)
            if val == 0.0:
                continue
            coeffs[(prefix, int(q), int(t))] = coeffs.get((prefix, int(q), int(t)), 0.0) + val

    if not coeffs:
        raise CallbackAbort(
            f"cut {cut.name!r} reconstructs to zero coefficients on every y; "
            "theta >= const is not a Benders cut and cannot be retracted once "
            "it is in the tree"
        )
    return const, coeffs


def candidate_from_values(
    reader: Callable[[str, int, int], float],
    shuttles: int,
    slots: int,
) -> Candidate:
    """Build the `Candidate` the subproblem expects from callback-visible values.

    The subproblem is keyed by the string form `yOUT[q,t]`, which is the master's
    own convention; keeping it here means the callback prices exactly what the
    loop would price for the same schedule, so a branch-and-cut result stays
    comparable against `lp150_then_mip8.yaml`.
    """
    cand: Candidate = {}
    for q in range(int(shuttles)):
        for t in range(int(slots)):
            cand[f"yOUT[{q},{t}]"] = float(reader("yOUT", q, t))
            cand[f"yRET[{q},{t}]"] = float(reader("yRET", q, t))
    return cand


def vet_cut_for_injection(
    result: SubproblemResult,
    *,
    context: str,
) -> tuple[float, dict[tuple[str, int, int], float]]:
    """Return the linear form of `result`'s cut, or abort.

    This is where the inverted D39 contract lives, and it is the whole reason the
    function exists separately from the callback: the decision is testable
    without CPLEX, without a tree, and without a registered callback.

    - VALID   -> return the cut's linear form.
    - INVALID -> abort. The cut has no lower-bound guarantee (MW fallback or
                 finite differences). In the loop this is survivable because the
                 reported bound is dropped afterwards; here the cut prunes and
                 the pruning stands.
    - UNKNOWN -> abort. The producer said nothing, so nothing is known. D39's
                 rule that UNKNOWN is a report and not a verdict applies, and
                 fail-closed in a tree means refusing to continue.
    - NO_CUT  -> abort. In the loop this is benign: no cut was added, so
                 whatever the bound was certified on still holds. In a callback
                 it is not benign, because returning without adding a cut is an
                 assertion that the incumbent is acceptable -- and the
                 subproblem has just declined to price it.
    """
    validity = classify_cut_validity(result)
    if validity is not CutValidity.VALID:
        raise CallbackAbort(
            f"{context}: subproblem returned {validity.value}; a lazy cut cannot be "
            "un-added, so the solve stops rather than injecting an unguaranteed cut "
            "or silently accepting the incumbent (D39, D44)",
            validity=validity,
        )
    cut = result.cut if result.cut is not None else (result.cuts[0] if result.cuts else None)
    if cut is None:
        raise CallbackAbort(
            f"{context}: validity reported VALID with no cut attached; the two "
            "cannot both be true and the disagreement is not resolvable here",
            validity=validity,
        )
    return cut_to_linear_form(cut)


def build_variable_name_map(
    persistent_solver: object,
    model: object,
) -> tuple[dict[tuple[str, int, int], str], list[str]]:
    """Map `(dir, q, t)` and the thetas onto the solver's own column names.

    Pyomo's persistent CPLEX interface keys `_pyomo_var_to_solver_var_map` by the
    VarData object and stores a *name* (`'x37'`), not an integer index. CPLEX's
    callback API accepts names wherever it accepts indices, and names survive the
    presolve remapping that a lazy callback forces CPLEX to restrict -- so they
    are used directly rather than converted.

    Raises if any y variable is missing from the map. A silently absent column
    would produce a cut over fewer variables than the one the subproblem priced,
    which is not a weaker cut, it is a different and possibly invalid one.
    """
    vmap = getattr(persistent_solver, "_pyomo_var_to_solver_var_map", None)
    if not vmap:
        raise CallbackAbort(
            "the persistent solver exposes no _pyomo_var_to_solver_var_map; "
            "set_instance() must run before the callback is registered"
        )

    names: dict[tuple[str, int, int], str] = {}
    missing: list[str] = []
    for prefix in ("yOUT", "yRET"):
        var = getattr(model, prefix, None)
        if var is None:
            raise CallbackAbort(f"master model has no {prefix}")
        for idx in var:
            q, t = int(idx[0]), int(idx[1])
            data = var[idx]
            if data not in vmap:
                missing.append(f"{prefix}[{q},{t}]")
                continue
            names[(prefix, q, t)] = vmap[data]
    if missing:
        raise CallbackAbort(
            f"{len(missing)} y variables are absent from the solver map "
            f"(first: {missing[0]}); a cut over a subset of the priced schedule "
            "is a different cut, not a weaker one"
        )

    theta_names: list[str] = []
    for attr in ("theta", "theta_out", "theta_ret"):
        th = getattr(model, attr, None)
        if th is None:
            continue
        if th.is_indexed():
            for idx in th:
                if th[idx] in vmap:
                    theta_names.append(vmap[th[idx]])
        elif th in vmap:
            theta_names.append(vmap[th])
    if not theta_names:
        raise CallbackAbort(
            "no theta column found on the master; a Benders cut has nothing to "
            "bound and would silently become a constraint on y alone"
        )
    return names, theta_names


def register_lazy_callback(
    persistent_solver: object,
    model: object,
    cut_source: Callable[[Candidate], SubproblemResult],
    stats: LazyCutStats,
    *,
    shuttles: int,
    slots: int,
    eps_violation: float = 1e-6,
    vprint: Callable[[str], None] = lambda _msg: None,
) -> Callable[[], None]:
    """Register the lazy-constraint callback and return a re-raise hook.

    The returned callable must be invoked after `solve()`. CPLEX catches whatever
    a Python callback raises and turns it into a generic solve failure with the
    original traceback gone, so `CallbackAbort` is stored on the way out and
    re-raised outside. Without that, the one thing this module exists to
    guarantee -- that an unguaranteed cut stops the run -- would be reported as
    "CPLEX failed" with no reason attached.
    """
    from cplex import SparsePair
    from cplex.callbacks import LazyConstraintCallback

    names, theta_names = build_variable_name_map(persistent_solver, model)
    cpx = getattr(persistent_solver, "_solver_model", None)
    if cpx is None:
        raise CallbackAbort("the persistent solver holds no CPLEX model")

    y_order = [(d, q, t) for (d, q, t) in names]
    y_cols = [names[k] for k in y_order]
    held: dict[str, CallbackAbort] = {}

    class _BendersLazyCallback(LazyConstraintCallback):
        def __call__(self) -> None:  # noqa: D401 - CPLEX calls this
            if held:
                # A previous invocation already decided the solve must stop.
                # CPLEX may still call us on another thread before abort() takes
                # effect; adding nothing here would accept an incumbent, so the
                # only safe move is to abort again and add nothing else.
                self.abort()
                return
            stats.invocations += 1
            try:
                y_vals = dict(zip(y_order, self.get_values(y_cols)))
                theta_val = float(sum(self.get_values(theta_names)))

                cand = candidate_from_values(
                    lambda d, q, t: y_vals.get((d, q, t), 0.0), shuttles, slots
                )
                stats.subproblem_solves += 1
                result = cut_source(cand)
                stats.note_validity(classify_cut_validity(result))
                const, coeffs = vet_cut_for_injection(
                    result, context=f"incumbent at node {self.get_node_ID()}"
                )

                # theta >= const + sum(c*y)  <=>  theta - sum(c*y) >= const
                lhs_at_candidate = theta_val - sum(
                    c * y_vals.get(k, 0.0) for k, c in coeffs.items()
                )
                if lhs_at_candidate >= const - eps_violation:
                    # The incumbent already satisfies the cut. Accepting it here
                    # is correct and is the only place in this callback where
                    # adding nothing is the right answer.
                    stats.incumbents_accepted += 1
                    return

                ind = list(theta_names) + [names[k] for k in coeffs]
                val = [1.0] * len(theta_names) + [-float(c) for c in coeffs.values()]
                self.add(
                    constraint=SparsePair(ind=ind, val=val), sense="G", rhs=float(const)
                )
                stats.cuts_injected += 1
            except CallbackAbort as exc:
                held["abort"] = exc
                stats.aborted_reason = exc.reason
                vprint(f"[BNC ABORT] {exc.reason}")
                self.abort()
            except Exception as exc:  # noqa: BLE001 - deliberately broad, see below
                # Anything unexpected in here is still a reason to stop. The
                # alternative is returning normally, which tells CPLEX the
                # incumbent is fine -- the exact silent-accept this module was
                # written to prevent.
                wrapped = CallbackAbort(
                    f"callback raised {type(exc).__name__}: {exc}"
                )
                held["abort"] = wrapped
                stats.aborted_reason = wrapped.reason
                vprint(f"[BNC ABORT] {wrapped.reason}")
                self.abort()

    cpx.register_callback(_BendersLazyCallback)

    def raise_if_aborted() -> None:
        if "abort" in held:
            raise held["abort"]

    return raise_if_aborted
