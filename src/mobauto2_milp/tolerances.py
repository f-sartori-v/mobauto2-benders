from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Tolerances:
    eps_bin: float = 1e-6
    eps_feas: float = 1e-7
    eps_cut: float = 1e-8
    eps_hash: float = 1e-6


def project_binary_value(v: Any, eps_bin: float) -> int | None:
    try:
        fv = float(v)
    except Exception:
        return None
    # CPLEX can return tiny integrality residuals (for example ~1e-6 to 1e-5)
    # even for integer-feasible MIP incumbents, so keep a modest lower bound here.
    tol = max(float(eps_bin), 1e-5)
    if fv <= tol:
        return 0
    if fv >= 1.0 - tol:
        return 1
    return None


def project_candidate(
    cand: dict[str, Any],
    eps_bin: float,
    max_offenders: int = 5,
) -> tuple[dict[str, int], list[tuple[str, float]]]:
    proj: dict[str, int] = {}
    offenders: list[tuple[str, float]] = []
    for k, v in cand.items():
        pv = project_binary_value(v, eps_bin)
        if pv is None:
            try:
                offenders.append((str(k), float(v)))
            except Exception:
                offenders.append((str(k), float("nan")))
        else:
            proj[str(k)] = pv
    if offenders:
        offenders.sort(key=lambda kv: abs((kv[1] - 0.5) if kv[1] == kv[1] else 1.0), reverse=True)
        offenders = offenders[: max(0, int(max_offenders))]
    return proj, offenders
