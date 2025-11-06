"""Problem-specific implementations live here.

Replace the templates in `master_impl.py` and `subproblem_impl.py` with
models following your report (Report/Bender decomposition.pdf).
"""

from .master_impl import ProblemMaster  # noqa: F401
from .subproblem_impl import ProblemSubproblem  # noqa: F401

__all__ = ["ProblemMaster", "ProblemSubproblem"]

