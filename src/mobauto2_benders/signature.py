"""The master candidate's per-slot aggregate, and the fibre of master solutions over it.

The recourse is not a function of the master's per-vehicle schedule `y`. It is a
function of

    Y_d[tau] = sum_q y_d[q,tau],   d in {OUT, RET}

because `ProblemSubproblem.evaluate` reads the candidate only to build
`C_d[tau] = S * Y_d[tau]`, and `solve_subproblem` puts that vector in the right-hand
side of the capacity rows and nowhere else. Everything else about that LP -- the arc
set, the variable set, the demand rows, the objective coefficients -- is a function of
`(T, Wmax_slots, p)` alone.

Two candidates with the same signature therefore produce a byte-identical LP: the same
recourse value AND the same dual. See `docs/DESIGN_DD_v1.md` E1/E2 and D48.

This module is the one place that fact is written down as code, so that the diagnostic
in the Benders loop, the exactness tests, and the decision-diagram work all read the
signature the same way. It deliberately does NOT reimplement the parsing loop inside
`subproblem_impl`; the test suite asserts the two agree.

It lives at package top level rather than in `problem/` on purpose. `benders/solver.py`
needs it, `problem/master_impl.py` already imports `benders.solver`, and a
`benders -> problem` import at module scope closes that loop -- Python runs
`problem/__init__.py` for any submodule import, which pulls `master_impl` back in
mid-initialisation. The dependency direction `problem -> benders` is the existing
architecture (AGENTS.md); this module sits below both.
"""

from __future__ import annotations

from math import comb
from typing import Mapping, Sequence


def parse_index(name: str) -> tuple[int, int]:
    """Parse ``yOUT[q,tau]`` / ``yRET[q,tau]`` into ``(q, tau)``.

    Mirrors the parsing in `subproblem_impl`, which is the definition of record.
    """
    inside = name[name.find("[") + 1 : name.find("]")]
    q_str, t_str = inside.split(",")
    return int(q_str.strip()), int(t_str.strip())


def candidate_signature(
    candidate: Mapping[str, float], T: int
) -> tuple[list[float], list[float]]:
    """Return ``(Y_OUT, Y_RET)``, each of length ``T``.

    Values are summed as given. During the LP phase the master's `y` is fractional and
    so is the signature; that is not an error and must not be rounded away here, because
    the recourse genuinely is evaluated at the fractional aggregate (which is why cuts
    from a fractional candidate are valid -- the value function is convex).
    """
    Y_out = [0.0] * int(T)
    Y_ret = [0.0] * int(T)
    for name, val in candidate.items():
        if not isinstance(name, str):
            continue
        if name.startswith("yOUT["):
            _, tau = parse_index(name)
            if 0 <= tau < T:
                Y_out[tau] += float(val)
        elif name.startswith("yRET["):
            _, tau = parse_index(name)
            if 0 <= tau < T:
                Y_ret[tau] += float(val)
    return Y_out, Y_ret


def is_integral(
    values: Sequence[float], eps: float = 1e-6
) -> bool:
    """True when every entry is within ``eps`` of an integer."""
    return all(abs(float(v) - round(float(v))) <= eps for v in values)


def signature_key(
    Y_out: Sequence[float], Y_ret: Sequence[float], decimals: int = 6
) -> tuple:
    """A hashable key for counting distinct signatures.

    Rounded to `decimals` places. This is for the diagnostic only -- never for a
    validity decision. Two signatures that differ below the rounding are not "the same
    signature"; they are two nearby points that this counter chooses not to
    distinguish, and a counter is allowed to do that where a cut is not.
    """
    return (
        tuple(round(float(v), decimals) for v in Y_out),
        tuple(round(float(v), decimals) for v in Y_ret),
    )


def fibre_size(Y_out: Sequence[float], Y_ret: Sequence[float], Q: int) -> int | None:
    """How many distinct per-vehicle schedules `y` share this signature.

    ``prod_tau C(Q, Y_OUT[tau]) * C(Q, Y_RET[tau])`` -- the number of ways to choose
    which vehicles depart in each slot and direction. This counts assignments of a fixed
    departure profile to vehicles; it is an upper bound on the number of *feasible*
    schedules in the fibre, since the per-vehicle battery and occupancy constraints
    reject some assignments.

    Returns None for a fractional or out-of-range signature, where the count is not
    defined. A caller that wants a number regardless is asking the wrong question --
    the fibre is a set of integer points.
    """
    if not (is_integral(Y_out) and is_integral(Y_ret)):
        return None
    total = 1
    for v in list(Y_out) + list(Y_ret):
        k = int(round(float(v)))
        if k < 0 or k > int(Q):
            return None
        total *= comb(int(Q), k)
    return total
