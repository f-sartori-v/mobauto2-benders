"""A structural fingerprint of the recourse LP: everything `y` is NOT allowed to move.

WHY THIS EXISTS (B4/B5, audit items 1.5 and 1.6).

The exactness conditions are claims about the recourse's SHAPE:

  E1  the recourse depends on `y` only through the signature `Y_d[tau] = sum_q y_d[q,tau]`;
  E2  `y` enters the subproblem in the right-hand side only.

Until now the only observable the fixtures had was the DUAL vector, so E1's fibre test
asserted that two schedules sharing a signature return the same duals. That assertion is
wrong, and the audit says so. A degenerate LP has several optimal dual solutions; every
one of them is dual feasible, so every one of them yields a valid Benders cut. Requiring
them to be equal asserts a property of the *solver's pivoting* rather than a property of
the model, and it can fail on a correct implementation -- a strictly worse failure mode
than the one it was guarding, because it would send someone looking for a bug in the
formulation.

What the fibre test SHOULD assert is what actually has to hold: two schedules with the
same signature produce a byte-identical LP. Same variables, same rows, same objective
coefficients, same non-capacity right-hand sides -- and therefore the same optimal
VALUE. This module computes that object.

WHAT IS DELIBERATELY EXCLUDED. The capacity right-hand sides `C_d[tau] = S*Y_d[tau]` are
NOT part of the fingerprint. They are the one channel `y` is allowed to use, and the
whole point of E2 is that it is the only one. Including them would make the fingerprint
change with the schedule and assert nothing; excluding them is what turns "the matrix is
fixed" into a check rather than a slogan. `capacity_rhs()` returns them separately, so a
test can assert both halves: the structure is invariant AND the capacity vector is
exactly what the signature says it should be.

WHAT IT ALSO CATCHES (B5). Equation (e2) in an earlier draft defined a minute-indexed
capacity `C_d[m]` through an incidence matrix. The code never built that -- it builds
minute-indexed DEMAND rows against SLOT-indexed capacity rows, `sum_m x[d,m,tau] <=
S*Y_d[tau]` -- but nothing asserted the difference. Fingerprinting the row index sets
does: a minute-indexed capacity row set has |minutes| entries and moves with the demand
file, a slot-indexed one has T entries and does not. That interface is what lets the
minute recourse substitute for the slot recourse without touching the cut space.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence

# Rows whose right-hand side is the schedule's one legitimate channel into the LP.
# Everything else about the model is invariant and is fingerprinted.
CAPACITY_ROWS = ("Cap_out", "Cap_ret", "MinCapOut", "MinCapRet")


def _coefficients(expr) -> dict[str, float]:
    """Variable name -> linear coefficient, for one constraint or objective body.

    Read through Pyomo's standard repn rather than by differentiating: this LP is
    built but never solved, so its variables carry no values and automatic
    differentiation refuses to evaluate. The standard repn is also the form the
    solver interface itself uses, so what is fingerprinted is what would be written
    to the solver -- and it is insensitive to how the sums happened to be nested,
    which a comparison of expression strings would not be.

    Raises on a nonlinear body. The recourse is an LP; a nonlinear row here would
    mean something far more interesting than a fingerprint mismatch, and silently
    fingerprinting only its linear part would hide it.
    """
    from pyomo.repn import generate_standard_repn

    repn = generate_standard_repn(expr, quadratic=False)
    if repn.nonlinear_expr is not None or (
        getattr(repn, "quadratic_vars", None) or []
    ):
        raise ValueError(
            "the recourse LP contains a nonlinear term; it is supposed to be linear "
            "and every exactness argument in DESIGN_DD_v1 assumes so"
        )
    out: dict[str, float] = {
        var.name: float(coeff)
        for var, coeff in zip(repn.linear_vars, repn.linear_coefs)
        if float(coeff) != 0.0
    }
    const = float(repn.constant or 0.0)
    if const != 0.0:
        out["<constant>"] = const
    return out


def structural_description(model: Any) -> dict[str, Any]:
    """The parts of `model` that no schedule is allowed to change.

    Returned as a plain dict so a failing test can print the DIFFERENCE rather than
    two hashes, which is the whole reason this is not just a checksum.
    """
    import pyomo.environ as pyo

    variables = sorted(
        v.name
        for v in model.component_data_objects(pyo.Var, active=True, descend_into=True)
    )

    rows: dict[str, dict[str, Any]] = {}
    for con in model.component_objects(pyo.Constraint, active=True):
        is_capacity = con.local_name in CAPACITY_ROWS
        for idx in con:
            cdata = con[idx]
            entry: dict[str, Any] = {
                "coefficients": _coefficients(cdata.body),
                "is_capacity_row": is_capacity,
            }
            if not is_capacity:
                # The right-hand side of a non-capacity row is structure: the demand
                # rows carry R_d[t], which is data, not a decision. A schedule that
                # moved one of these would be entering the LP through a channel E2
                # forbids.
                entry["lower"] = (
                    None if cdata.lower is None else float(pyo.value(cdata.lower))
                )
                entry["upper"] = (
                    None if cdata.upper is None else float(pyo.value(cdata.upper))
                )
            rows[cdata.name] = entry

    objective = {}
    for obj in model.component_data_objects(pyo.Objective, active=True):
        objective[obj.name] = {
            "sense": int(obj.sense),
            "coefficients": _coefficients(obj.expr),
        }

    return {
        "variables": variables,
        "rows": rows,
        "objective": objective,
        "capacity_row_names": sorted(
            name for name, r in rows.items() if r["is_capacity_row"]
        ),
    }


def recourse_fingerprint(model: Any) -> str:
    """A short hash of `structural_description(model)`.

    Two recourse LPs share this hash exactly when they have the same variables, the
    same rows, the same objective, and the same non-capacity right-hand sides. They
    may still differ in capacity right-hand side -- that is `y`, and it is supposed to
    differ.
    """
    blob = json.dumps(structural_description(model), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def capacity_rhs(model: Any) -> dict[str, float]:
    """The capacity right-hand sides, by row name -- the half `y` IS allowed to move."""
    import pyomo.environ as pyo

    out: dict[str, float] = {}
    for con in model.component_objects(pyo.Constraint, active=True):
        if con.local_name not in CAPACITY_ROWS:
            continue
        for idx in con:
            cdata = con[idx]
            out[cdata.name] = (
                float("inf") if cdata.upper is None else float(pyo.value(cdata.upper))
            )
    return out


def capacity_rows_are_slot_indexed(model: Any, T: int) -> tuple[bool, str]:
    """B5. Is there one capacity row per departure SLOT, or one per minute?

    Returns ``(ok, explanation)``. This is the interface condition that carries E2
    across the resolution change: the minute recourse prices arrivals at minute
    fidelity but must still present exactly `T` capacity rows per direction, because
    those rows are what the dual `pi_d[tau]` -- and therefore the cut the master
    receives -- is indexed by. A minute-indexed capacity path would produce one dual
    per minute and no cut the master could accept without changing the cut space.
    """
    import pyomo.environ as pyo

    counts: dict[str, int] = {}
    for con in model.component_objects(pyo.Constraint, active=True):
        if con.local_name in CAPACITY_ROWS:
            counts[con.local_name] = sum(1 for _ in con)
    if not counts:
        return False, "the model has no capacity rows at all"
    bad = {k: v for k, v in counts.items() if v > int(T)}
    if bad:
        return False, (
            f"capacity rows are not slot-indexed: {bad} exceeds T={T}. A row count "
            "above T means the rows are indexed by something finer than the departure "
            "slot -- a minute-indexed capacity path -- and the dual is then not the "
            "per-slot pi the master's cut is written in."
        )
    return True, f"capacity rows per direction: {counts} (T={T})"


def slot_recourse_model_for(
    sp_params: dict, candidate_signature_out: Sequence[float],
    candidate_signature_ret: Sequence[float],
    R_out: Iterable[float], R_ret: Iterable[float],
):
    """Build the slot recourse LP straight from a signature, with no master involved.

    A convenience for the fixtures: they want to compare two schedules that share a
    signature, and the point is precisely that the LP is a function of the signature
    and nothing else -- so it is built from the signature, not from a schedule.
    """
    from .problem.subproblem_impl import SPParams, build_slot_recourse_model

    S = float(sp_params["S"])
    T = int(sp_params["T"])
    P = SPParams(
        T=T,
        Wmax_slots=int(sp_params["Wmax_slots"]),
        p=float(sp_params["p"]),
        lp_solver=str(sp_params.get("lp_solver", "cplex_direct")),
        S=S,
        K_out=[0] * T,
        K_ret=[0] * T,
        same_slot_eligibility=str(sp_params.get("same_slot_eligibility", "forbid")),
    )
    C_out = [S * float(v) for v in candidate_signature_out]
    C_ret = [S * float(v) for v in candidate_signature_ret]
    return build_slot_recourse_model(P, C_out, C_ret, list(R_out), list(R_ret))


__all__ = [
    "CAPACITY_ROWS",
    "capacity_rhs",
    "capacity_rows_are_slot_indexed",
    "recourse_fingerprint",
    "slot_recourse_model_for",
    "structural_description",
]
