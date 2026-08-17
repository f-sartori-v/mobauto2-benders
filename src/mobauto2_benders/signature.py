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


def departures_are_possible(T: int, trip_slots: int) -> tuple[list[bool], list[bool]]:
    """Which slots can carry a departure at all, per direction.

    Mirrors the fixings in `ProblemMaster.initialize`, which are the definition of
    record:

      * no trip may finish after the horizon, so `y_d[q,t] = 0` for `t >= T - trip_slots`;
      * an OUT must leave room for the RET that brings the vehicle home, so
        `yOUT[q,t] = 0` for `t >= T - 2*trip_slots`.

    Returned rather than recomputed at the call site because the Magnanti-Wong core
    point has to respect it: a core point with `Ybar_out[tau] > 0` at a fixed slot asks
    MW which dual best values extra capacity in a slot that can never carry a departure,
    and the answer steers the Pareto selection using a direction outside the region.
    """
    T = int(T)
    trip = max(1, int(trip_slots))
    out = [t < max(0, T - 2 * trip) for t in range(T)]
    ret = [t < max(0, T - trip) for t in range(T)]
    return out, ret


def project_core_point(
    Y_out: Sequence[float],
    Y_ret: Sequence[float],
    Q: int,
    trip_slots: int,
    eps: float,
) -> tuple[list[float], list[float]]:
    """Bring a candidate core point inside the projected master region.

    Magnanti-Wong requires a point in the relative interior of
    ``proj_Y(conv(Z))`` (formal formulation 16.2). Validity of the resulting cut does
    not depend on this -- that comes from dual feasibility -- but the *Pareto-optimality*
    claim does, and the claim is the entire reason MW is there rather than the plain
    dual.

    The previous core point was an exponential moving average clamped to the **box**
    ``[eps, Q-eps]`` per slot, which is outside the region on two counts. This enforces
    three necessary conditions on ``proj_Y``:

    1. **Zero on slots the master fixes.** See `departures_are_possible`.
    2. **The trip window.** A vehicle that starts a trip at ``u`` cannot start another
       before ``u + trip_slots``, so over any window of ``trip_slots`` consecutive slots
       the whole fleet starts at most ``Q`` trips::

           sum_{tau' in [tau, tau+trip_slots-1]} (Y_out[tau'] + Y_ret[tau']) <= Q

       Violated windows are scaled down proportionally, which keeps the profile's shape
       and cannot introduce a new violation elsewhere (every entry only decreases).
    3. **Strictly positive where a departure is possible**, floored at ``eps``, so the
       point stays interior in the coordinates that have an interior. A slot that cannot
       carry a departure is left at exactly 0: flooring it would reintroduce (1).

    These are *necessary* conditions, not a description of ``proj_Y`` -- battery and
    per-vehicle occupancy are not represented. So the result is a point in a relaxation
    of the region, which is strictly better than a box point and still not a proof of
    relative interiority. Describe it as such (16.5 option 4) rather than asserting the
    MW hypothesis outright.

    The window inequality is the same fact `problem/vehicle_dd.window_trip_caps` proves
    in stronger form for the master, where it is off by default because it bought ~1% of
    LP root for 1-2.2x the master time (D49/D50). That cost was master solve time. There
    is no master solve here, so the cheap version is used unconditionally.
    """
    T = len(list(Y_out))
    if len(list(Y_ret)) != T:
        raise ValueError(
            f"core point halves disagree in length: OUT={T}, RET={len(list(Y_ret))}"
        )
    Qf = float(Q)
    eps = float(eps)
    trip = max(1, int(trip_slots))
    ok_out, ok_ret = departures_are_possible(T, trip)

    out = [0.0 if not ok_out[t] else max(0.0, float(Y_out[t])) for t in range(T)]
    ret = [0.0 if not ok_ret[t] else max(0.0, float(Y_ret[t])) for t in range(T)]

    # Per-slot cap first: one slot alone cannot exceed the fleet.
    out = [min(v, Qf) for v in out]
    ret = [min(v, Qf) for v in ret]

    # Then the window cap. Scaling only ever decreases entries, so a single left-to-right
    # sweep cannot leave an earlier window violated -- but a later scale can tighten an
    # overlapping earlier window further, which is harmless (still feasible) and is why
    # this does not need to iterate to a fixed point.
    for t0 in range(T):
        t1 = min(T, t0 + trip)
        total = sum(out[t] + ret[t] for t in range(t0, t1))
        if total > Qf and total > 0.0:
            scale = Qf / total
            for t in range(t0, t1):
                out[t] *= scale
                ret[t] *= scale

    # Interior floor, only where a departure is possible. Capped by the window budget so
    # the floor cannot itself violate the cap: with `trip` slots in a window and both
    # directions, the largest safe uniform floor is Q / (2*trip).
    floor = min(eps, Qf / float(2 * trip)) if trip > 0 else eps
    for t in range(T):
        if ok_out[t]:
            out[t] = max(out[t], floor)
        if ok_ret[t]:
            ret[t] = max(ret[t], floor)

    return out, ret


def core_point_violations(
    Y_out: Sequence[float],
    Y_ret: Sequence[float],
    Q: int,
    trip_slots: int,
    tol: float = 1e-9,
) -> list[str]:
    """Necessary conditions of `project_core_point` that ``(Y_out, Y_ret)`` breaks.

    Returned as strings rather than a bool so a diagnostic can name the offending slot.
    Empty means the point satisfies every condition the projection enforces -- which is
    not the same as being in the relative interior of ``proj_Y``, for the reasons in
    `project_core_point`.
    """
    T = len(list(Y_out))
    trip = max(1, int(trip_slots))
    ok_out, ok_ret = departures_are_possible(T, trip)
    bad: list[str] = []
    for t in range(T):
        if not ok_out[t] and float(Y_out[t]) > tol:
            bad.append(f"Yout[{t}]={float(Y_out[t]):.6g} on a slot the master fixes to 0")
        if not ok_ret[t] and float(Y_ret[t]) > tol:
            bad.append(f"Yret[{t}]={float(Y_ret[t]):.6g} on a slot the master fixes to 0")
        if float(Y_out[t]) < -tol or float(Y_ret[t]) < -tol:
            bad.append(f"negative entry at tau={t}")
    for t0 in range(T):
        t1 = min(T, t0 + trip)
        total = sum(float(Y_out[t]) + float(Y_ret[t]) for t in range(t0, t1))
        if total > float(Q) + tol:
            bad.append(
                f"window [{t0},{t1 - 1}] starts {total:.6g} trips, above Q={Q}"
            )
    return bad


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
