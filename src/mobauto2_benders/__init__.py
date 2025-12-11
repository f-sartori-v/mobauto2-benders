"""mobauto2_benders

A lightweight, solver-agnostic framework to implement Benders decomposition
for the MobAuto2 project. The framework provides:

- Abstract interfaces for the master and subproblem models
- A generic Benders loop orchestrator with logging and termination checks
- A small CLI and YAML-based configuration

Fill in your problem-specific logic in `mobauto2_benders/problem/` by
extending the abstract base classes.
"""

from .runner import run

__all__ = [
    "__version__",
    "run",
]

__version__ = "0.1.0"
