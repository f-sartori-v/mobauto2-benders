from __future__ import annotations

from typing import Any

from ..benders.subproblem import Subproblem
from ..benders.types import Candidate, Cut, CutType, SubproblemResult


class ProblemSubproblem(Subproblem):
    """Template subproblem evaluator.

    TODO: Implement subproblem(s) logic from Report/Bender decomposition.pdf.
    Given a master candidate, evaluate feasibility/optimality and return
    either a `SubproblemResult` with a `Cut` or an upper bound.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(params)
        # TODO: build any reusable structures/models

    def evaluate(self, candidate: Candidate) -> SubproblemResult:
        # TODO: build and solve your subproblem(s) using `candidate`
        # If feasible and optimal: return upper bound
        # If infeasible or violated: construct and return a Benders cut
        # Below is a placeholder raising NotImplementedError
        # Example construction of a cut (replace with your logic):
        # cut = Cut(name="sp_cut_1", cut_type=CutType.FEASIBILITY, coeffs={"x1": 1.0, "x2": -0.5}, rhs=3.0)
        # return SubproblemResult(is_feasible=False, cut=cut, violation=1.2)
        raise NotImplementedError("Implement subproblem evaluation and cut generation")

