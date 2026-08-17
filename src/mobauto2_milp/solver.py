from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Optional


class SolveStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class RunResult:
    status: SolveStatus
    iterations: int
    best_lower_bound: Optional[float]
    best_upper_bound: Optional[float]
    pax_served: Optional[float] = None
    pax_total: Optional[float] = None
    subproblem_obj: Optional[float] = None
    sp_wait_cost_slots: Optional[float] = None
    sp_fill_eps_cost: Optional[float] = None
    sp_penalty_cost: Optional[float] = None
    sp_penalty_pax: Optional[float] = None
    sp_total_demand: Optional[float] = None
    sp_slot_resolution: Optional[int] = None
    solve_started_at: Optional[datetime] = None
    solve_finished_at: Optional[datetime] = None
    elapsed_wall_s: Optional[float] = None


__all__ = ["RunResult", "SolveStatus"]
