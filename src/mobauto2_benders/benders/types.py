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
]
