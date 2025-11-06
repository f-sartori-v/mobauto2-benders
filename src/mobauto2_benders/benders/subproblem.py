from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import Candidate, SubproblemResult


class Subproblem(ABC):
    """Abstract interface for the subproblem(s) evaluation.

    Given a candidate master solution, evaluate the subproblem(s) and either
    return feasibility/optimality cuts or an upper bound.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}

    @abstractmethod
    def evaluate(self, candidate: Candidate) -> SubproblemResult:
        """Evaluate subproblem(s) at candidate and return result."""


__all__ = ["Subproblem"]

