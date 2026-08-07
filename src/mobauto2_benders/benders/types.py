from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional


class SolveStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"
    UNKNOWN = "UNKNOWN"


class CutType(str, Enum):
    OPTIMALITY = "OPTIMALITY"
    FEASIBILITY = "FEASIBILITY"


Candidate = Dict[str, float]


@dataclass(slots=True)
class Cut:
    """Linear cut over master variables.

    Represents: sum(coeffs[var] * x_var) + constant <= rhs  (or with sense)
    """

    name: str
    cut_type: CutType
    coeffs: Mapping[str, float] = field(default_factory=dict)
    rhs: float = 0.0
    sense: str = "<="  # one of "<=", ">=", "=="
    constant: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubproblemResult:
    is_feasible: bool
    cut: Optional[Cut] = None
    cuts: list[Cut] = field(default_factory=list)
    upper_bound: Optional[float] = None
    violation: float | None = None
    # Optional diagnostics payload for reporting/formatting
    diagnostics: dict[str, Any] = field(default_factory=dict)


class CutValidity(str, Enum):
    """Whether this iteration's cut carries a lower-bound guarantee.

    A bool cannot express this. The subproblem writes `cut_valid_lower_bound` into its
    diagnostics if and only if it generated a cut, so a reader faces four situations, and
    collapsing them onto True/False forces a default that lies in at least one of them.
    That is exactly what happened: `solver.py` defaulted the missing key to True while
    `subproblem_impl.py` defaulted the same key to False, and neither author was wrong
    locally -- the type was too small.
    """

    VALID = "valid"  # a cut was generated and it is a valid lower-bound cut
    INVALID = "invalid"  # a cut was generated without a guarantee (MW fallback, finite differences)
    NO_CUT = (
        "no_cut"  # no cut this iteration; previously established validity is untouched
    )
    UNKNOWN = "unknown"  # a cut exists but the producer said nothing -- assume nothing


def classify_cut_validity(result: "SubproblemResult") -> CutValidity:
    """Read the validity of `result` without inventing a default.

    Kept next to the enum so every consumer asks the same question the same way. The
    fail-closed policy belongs to the caller: this function reports what is known, and
    UNKNOWN is a report, not a verdict.
    """
    diag = result.diagnostics if isinstance(result.diagnostics, dict) else None
    if diag is not None and "cut_valid_lower_bound" in diag:
        return (
            CutValidity.VALID
            if bool(diag["cut_valid_lower_bound"])
            else CutValidity.INVALID
        )
    produced_a_cut = result.cut is not None or bool(result.cuts)
    if not produced_a_cut:
        # Every no-cut return path omits the key: infeasible subproblems, the theta
        # early exit, and the debug skip. Nothing new was added to the master, so
        # whatever the bound was certified on before still holds.
        return CutValidity.NO_CUT
    return CutValidity.UNKNOWN


@dataclass(slots=True)
class SolveResult:
    status: SolveStatus
    objective: Optional[float]
    candidate: Optional[Candidate]
    lower_bound: Optional[float]
    iterations: int = 0


__all__ = [
    "SolveStatus",
    "CutType",
    "Candidate",
    "Cut",
    "SubproblemResult",
    "SolveResult",
    "CutValidity",
    "classify_cut_validity",
]
