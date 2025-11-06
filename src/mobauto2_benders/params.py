from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class Params:
    # Global
    max_iters: int = 50
    tolerance: float = 1e-4
    log_level: str = "INFO"

    # Master (MILP)
    mip_solver: str = "glpk"
    mip_solver_executable: str | None = None
    Q: int = 1
    T: int = 1
    trip_slots: int = 1
    Emax: float = 100.0
    L: float = 10.0
    delta_chg: float = 25.0
    binit: List[float] | None = None

    # Subproblem (LP)
    lp_solver: str = "glpk"
    lp_solver_executable: str | None = None
    S: float = 4.0
    Wmax_slots: int = 0
    p: float = 0.0
    R_out: List[float] | None = None
    R_ret: List[float] | None = None

    @staticmethod
    def load(path: str | Path) -> "Params":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        params = Params()
        for k, v in raw.items():
            if hasattr(params, k):
                setattr(params, k, v)
        # Defaults for arrays
        if params.binit is None:
            params.binit = [params.Emax] * params.Q
        if params.R_out is None:
            params.R_out = [0.0] * params.T
        if params.R_ret is None:
            params.R_ret = [0.0] * params.T
        # Normalize lengths
        if len(params.binit) != params.Q:
            params.binit = (params.binit + [params.Emax] * params.Q)[: params.Q]
        if len(params.R_out) != params.T:
            params.R_out = (params.R_out + [0.0] * params.T)[: params.T]
        if len(params.R_ret) != params.T:
            params.R_ret = (params.R_ret + [0.0] * params.T)[: params.T]
        return params
