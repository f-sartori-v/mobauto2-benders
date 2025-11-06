from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from .types import Candidate, Cut, SolveResult, SolveStatus


class MasterProblem(ABC):
    """Abstract interface for the master problem.

    Implementations should encapsulate construction of the master model,
    managing variables of the master space and integrating added Benders cuts.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}

    def initialize(self) -> None:
        """Build or reset the master model. Called once before iteration."""

    @abstractmethod
    def solve(self) -> SolveResult:
        """Solve the master problem and return result with candidate solution.

        - `candidate` must include all master variables required by subproblems.
        - `lower_bound` should track the best known LB on the true objective.
        """

    @abstractmethod
    def add_cut(self, cut: Cut) -> None:
        """Integrate a Benders cut into the master model."""

    def best_lower_bound(self) -> Optional[float]:  # optional, can be overridden
        return None


__all__ = ["MasterProblem"]

