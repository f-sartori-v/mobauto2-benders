"""Price a slot-level schedule at minute fidelity, to measure what aggregation costs.

THE QUESTION THIS ANSWERS. The whole multi-resolution idea rests on a premise nothing in
this repository has ever measured: that evaluating operations at slot resolution
misstates what a schedule really costs, by enough to matter. This module produces that
number. It takes a schedule the slot model produced, expands it onto a fixed minute grid,
prices it against the demand's ACTUAL arrival minutes, and reports the difference.

It is a measuring instrument, not an optimiser. There is no master here and no
decomposition -- one LP, given a schedule, answering "what does this really cost?".

WHY THE SLOT MODEL CAN BE WRONG. It aggregates demand into slots and charges every
passenger arriving anywhere in slot `t` the same wait `(tau - t)` slots to a departure in
slot `tau` (D7). A passenger arriving one minute before the end of a 30-minute slot is
charged the same as one arriving at its start, and the departure itself is placed only to
slot precision. The error is bounded by a slot width per passenger and is signed both
ways, so it can partly cancel in aggregate -- which is exactly why it has to be measured
rather than argued.

THE EXACTNESS CONDITION THIS RESPECTS (E2, D30). The minute grid is FIXED and the
schedule enters only through the right-hand side: capacity `S` at each departure minute.
No decision here changes which rows or variables exist. That is what makes this the same
construction a minute-level Benders subproblem would use, rather than a one-off script --
see DESIGN_DD_v1 section 6.

UNITS. Everything in this module is minutes and passengers:

    cost = sum over served passengers of (departure_minute - arrival_minute)
         + p_minutes * (passengers left unserved)

`p_minutes` is the resolution-independent penalty (D50). The slot model's objective is
comparable to this one only after multiplying it by `slot_resolution`, which
`slot_objective_in_minutes` does.

READ THE WAITING TERM, NOT THE TOTAL. Measured on `baseline_d9`, the objective is 93%
unmet-demand penalty and 6.8% waiting. Minute-level pricing barely moves the penalty --
it is a headcount of passengers nobody could reach in time -- so comparing TOTALS
compares a number that is mostly a term this exercise does not change, and it came out
at a misleadingly small 1.5%. The waiting term alone is wrong by 66-86%. `honest_waiting`
reports the terms separately for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

OUT = "OUT"
RET = "RET"

# Where inside its slot a departure is assumed to actually leave.
#
# CORRECTED (D76; the previous version of this comment argued `end` was the honest
# choice, and it was wrong). `tau` here is never the arrival's own slot -- it is the
# master's departure-slot INDEX, already legal only for tau >= t+1 (D7/D8). Everyone who
# boards departure tau therefore arrived during an EARLIER slot and has, by construction,
# fully arrived before slot tau even begins. `start` (D(tau) = tau*delta) is not one
# convention among three: it is a restatement of what the master's own t+1 rule already
# commits to -- the vehicle leaves the instant its slot opens, because nobody boarding it
# is still arriving. There is no passenger "not yet arrived" at that instant to protect
# against; that concern only applies to arrivals DURING slot tau itself, and the master's
# own legality rule (tau >= t+1) already excludes them from this departure by definition.
#
# `midpoint` and `end` price a DIFFERENT, slower-departing schedule -- one the master
# never committed to. Worse, because the only reachability filter this module applies is
# `0 <= dep_minute - arrival <= Wmax` (no explicit same-slot exclusion), pricing under
# `midpoint`/`end` lets a departure whose committed clock time is tau*delta claim credit
# for passengers who arrived DURING slot tau -- a same-slot pickup the master's tau>=t+1
# rule explicitly forbids for this departure. That capability is real only if the vehicle
# actually left later than tau*delta, which `start` is the only convention that does not
# assume.
#
# They remain useful, explicitly, as counterfactual sensitivity toggles -- "what if real
# dwell/boarding time pushes the departure N minutes past its committed instant" -- but
# are not candidates for "what this schedule truly costs" and are not the default.
#
# The bound-direction argument this comment used to make for `end` ("the only convention
# under which the slot recourse lower-bounds the minute recourse") does not survive
# either: D54 Section 1 measured the slot model overstating the true minute cost under
# ALL THREE conventions, `end` included, so no convention actually delivers that property
# in aggregate. It was never a reason to prefer `end` over `start`.
DeparturePolicy = Literal["start", "midpoint", "end"]


def placement_offset(policy: DeparturePolicy, slot_resolution: float) -> float:
    """Minutes from the start of a slot to the assumed departure instant."""
    delta = float(slot_resolution)
    if policy == "start":
        return 0.0
    if policy == "midpoint":
        return delta / 2.0
    if policy == "end":
        return delta
    raise ValueError(
        f"departure policy must be 'start', 'midpoint' or 'end', got {policy!r}"
    )


@dataclass(frozen=True)
class MinutePricingResult:
    total_cost: float
    waiting_minutes: float
    unserved_passengers: float
    served_passengers: float
    total_passengers: float
    departures_used: int
    policy: str

    def as_row(self) -> str:
        return (
            f"{self.policy:9s} cost={self.total_cost:12.2f} "
            f"wait_min={self.waiting_minutes:10.1f} "
            f"unserved={self.unserved_passengers:6.0f}/{self.total_passengers:.0f}"
        )


def load_request_minutes(path: str | Path) -> dict[str, list[int]]:
    """Read a demand file and return arrival minutes per direction, unaggregated.

    This is the point of the exercise: the setups carry each request's own arrival
    minute, and the slot model throws that precision away at load. Here it is kept.
    """
    import json

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML is required to read YAML demand files") from exc
        doc = yaml.safe_load(text)
    else:
        doc = json.loads(text)

    if isinstance(doc, dict):
        container = doc.get("requests") or doc.get("req_matrix") or []
    else:
        container = doc

    out: dict[str, list[int]] = {OUT: [], RET: []}
    for req in container or []:
        if isinstance(req, dict):
            direction, raw_time = req.get("dir"), req.get("time")
        elif isinstance(req, (list, tuple)) and len(req) >= 2:
            direction, raw_time = req[0], req[1]
        else:
            continue
        if raw_time is None:
            continue
        key = (
            direction.upper()
            if isinstance(direction, str)
            else (OUT if int(direction) == 0 else RET)
        )
        if key not in out:
            continue
        out[key].append(int(round(float(raw_time))))
    return out


def departure_minutes(
    departure_slots: Sequence[int], slot_resolution: int, policy: DeparturePolicy
) -> list[float]:
    """Map departure slots to the minute each departure is assumed to leave."""
    delta = float(slot_resolution)
    offset = placement_offset(policy, delta)
    return [float(tau) * delta + offset for tau in departure_slots]


def price_direction_at_minutes(
    arrival_minutes: Sequence[int],
    departure_minutes_list: Sequence[float],
    seats: float,
    wmax_minutes: float,
    p_minutes: float,
    lp_solver: str = "cplex_direct",
) -> tuple[float, float, float]:
    """Cheapest way to serve these arrivals with these departures.

    Returns ``(waiting_minutes, unserved_passengers, served_passengers)``.

    A passenger arriving at minute `m` may board a departure at minute `d` when
    ``m <= d <= m + wmax_minutes``. Boarding at exactly `m` is allowed: at minute
    granularity there is no sub-minute ordering to appeal to, and forbidding it would
    charge a full minute of wait to a passenger who walked onto a waiting vehicle.
    """
    import pyomo.environ as pyo

    if not arrival_minutes:
        return 0.0, 0.0, 0.0

    # Identical arrival minutes are pooled: the LP only needs one variable per
    # (arrival minute, departure) pair, not per passenger.
    counts: dict[int, int] = {}
    for m in arrival_minutes:
        counts[int(m)] = counts.get(int(m), 0) + 1

    arrivals = sorted(counts)
    deps = list(range(len(departure_minutes_list)))
    arcs = [
        (m, j)
        for m in arrivals
        for j in deps
        if 0.0 <= departure_minutes_list[j] - float(m) <= float(wmax_minutes)
    ]

    mdl = pyo.ConcreteModel()
    mdl.Arcs = pyo.Set(initialize=arcs, dimen=2, ordered=False)
    mdl.x = pyo.Var(mdl.Arcs, within=pyo.NonNegativeReals)
    mdl.u = pyo.Var(arrivals, within=pyo.NonNegativeReals)

    mdl.obj = pyo.Objective(
        expr=sum(
            (float(departure_minutes_list[j]) - float(m)) * mdl.x[m, j]
            for (m, j) in arcs
        )
        + float(p_minutes) * sum(mdl.u[m] for m in arrivals),
        sense=pyo.minimize,
    )

    mdl.Demand = pyo.Constraint(
        arrivals,
        rule=lambda mm, m: sum(mm.x[a, j] for (a, j) in arcs if a == m) + mm.u[m]
        == float(counts[m]),
    )
    # Capacity is the ONLY channel the schedule uses -- right-hand side alone (E2).
    mdl.Capacity = pyo.Constraint(
        deps,
        rule=lambda mm, j: (
            sum(mm.x[a, jj] for (a, jj) in arcs if jj == j) <= float(seats)
            if any(jj == j for (_a, jj) in arcs)
            else pyo.Constraint.Skip
        ),
    )

    solver = pyo.SolverFactory(lp_solver)
    res = solver.solve(mdl, tee=False, load_solutions=False)
    term = getattr(res.solver, "termination_condition", None)
    if term != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f"minute-level pricing LP did not solve to optimality: termination={term}. "
            "This LP is always feasible -- unserved demand is absorbed by u at price "
            "p_minutes -- so a non-optimal termination is a defect, not an instance."
        )
    mdl.solutions.load_from(res)

    waiting = sum(
        (float(departure_minutes_list[j]) - float(m)) * float(pyo.value(mdl.x[m, j]))
        for (m, j) in arcs
    )
    unserved = sum(float(pyo.value(mdl.u[m])) for m in arrivals)
    served = float(sum(counts.values())) - unserved
    return float(waiting), float(unserved), float(served)


def price_schedule_at_minutes(
    departures: dict[str, Sequence[int]],
    requests: dict[str, Sequence[int]],
    slot_resolution: int,
    seats: float,
    wmax_minutes: float,
    p_minutes: float,
    policy: DeparturePolicy = "start",
    lp_solver: str = "cplex_direct",
) -> MinutePricingResult:
    """Price a whole schedule (both directions) at minute fidelity.

    `policy="start"` is the default because it is the only one that prices what this
    schedule actually does (D76) -- see the `DeparturePolicy` comment above.
    `midpoint`/`end` remain available as explicit counterfactual toggles.
    """
    waiting = 0.0
    unserved = 0.0
    served = 0.0
    used = 0
    for direction in (OUT, RET):
        dep_minutes = departure_minutes(
            list(departures.get(direction, [])), slot_resolution, policy
        )
        used += len(dep_minutes)
        w, u, s = price_direction_at_minutes(
            list(requests.get(direction, [])),
            dep_minutes,
            seats,
            wmax_minutes,
            p_minutes,
            lp_solver,
        )
        waiting += w
        unserved += u
        served += s
    total_pax = served + unserved
    return MinutePricingResult(
        total_cost=waiting + float(p_minutes) * unserved,
        waiting_minutes=waiting,
        unserved_passengers=unserved,
        served_passengers=served,
        total_passengers=total_pax,
        departures_used=used,
        policy=policy,
    )


def price_schedule_given_departure_minutes(
    departures_minutes: dict[str, Sequence[float]],
    requests: dict[str, Sequence[int]],
    seats: float,
    wmax_minutes: float,
    p_minutes: float,
    lp_solver: str = "cplex_direct",
) -> MinutePricingResult:
    """Price a schedule whose departures are already minute-valued -- not a slot list.

    A4c (docs/FORWARD_PLAN_v1.md), the blocking piece for report Comparison C. The
    continuous-time CP model (S:meth-cp) places a trip's departure at ANY minute; it
    is not confined to `tau*delta + offset` for any placement policy, because it is
    not confined to a slot grid at all. `price_schedule_at_minutes` cannot price that
    schedule -- it takes departure SLOTS and applies `departure_minutes()` to them,
    which would silently reinterpret a minute value as a slot index and multiply it
    by `delta` again. This function is `price_schedule_at_minutes` with that one
    conversion removed: same LP (`price_direction_at_minutes`), same units, same
    result on a schedule that happens to be slot-aligned -- see
    `TestGivenDepartureMinutes.test_agrees_with_the_slot_path_when_slot_aligned` in
    tests/test_minute_pricer.py. Existing callers are unaffected; this is a new
    function, not a new argument on the old one, precisely so a caller cannot pass
    minute-valued departures through the slot path by forgetting a flag -- the same
    silent-unit-confusion failure mode `p` vs `p_minutes` already cost this project
    (docs/PROJECT_STATE_v6.md section 3).
    """
    waiting = 0.0
    unserved = 0.0
    served = 0.0
    used = 0
    for direction in (OUT, RET):
        dep_minutes = [float(m) for m in departures_minutes.get(direction, [])]
        used += len(dep_minutes)
        w, u, s = price_direction_at_minutes(
            list(requests.get(direction, [])),
            dep_minutes,
            seats,
            wmax_minutes,
            p_minutes,
            lp_solver,
        )
        waiting += w
        unserved += u
        served += s
    total_pax = served + unserved
    return MinutePricingResult(
        total_cost=waiting + float(p_minutes) * unserved,
        waiting_minutes=waiting,
        unserved_passengers=unserved,
        served_passengers=served,
        total_passengers=total_pax,
        departures_used=used,
        policy="given_minutes",
    )


def _offset_grid(slot_resolution: int, offsets: Sequence[float] | None) -> list[float]:
    """Candidate departure instants for a slot-`tau` departure, as minutes from `tau*delta`.

    CORRECTED (D76). `tau*delta` (offset 0, "start") is the CEILING, not one end of an
    arbitrary window: it is the instant the master's own t+1 rule already commits this
    departure to. The only genuine degree of freedom left to search is ANTICIPATION --
    could this vehicle truly have left earlier, given who actually boarded it -- so the
    default grid is every whole minute in `[-delta, 0]`, never positive.

    Moving positive (towards `tau*delta + delta`, the old `end`) is not a competing
    option to search: `tau >= t+1` (D7/D8) already means nobody boarding this departure
    is still arriving during slot tau, so waiting longer inside slot tau buys nothing and
    only adds cost -- pure noise in a relaxation, not a real choice. A caller who wants to
    price "if this departure literally happened at a later tau" already has that: it is a
    different tau in the schedule, not an offset on this one.

    `offsets`, when given explicitly, is validated against the wider symmetric range
    `[-delta, delta]` rather than clamped to the anticipate-only default -- deliberate
    counterfactual or cross-validation use (e.g. reproducing a `midpoint`/`end` fixed
    policy through this machinery) is still allowed; only the DEFAULT search space is
    restricted to what is actually a real degree of freedom.
    """
    delta = float(slot_resolution)
    if offsets is not None:
        grid = [float(o) for o in offsets]
        for o in grid:
            if not (-delta <= o <= delta):
                raise ValueError(
                    f"placement offset {o} is outside [-{delta}, {delta}]"
                )
        return sorted(set(grid))
    return [float(k) for k in range(-int(delta), 1)]


def price_direction_optimal_placement(
    arrival_minutes: Sequence[int],
    departure_slots: Sequence[int],
    slot_resolution: int,
    seats: float,
    wmax_minutes: float,
    p_minutes: float,
    offsets: Sequence[float] | None = None,
    mip_solver: str = "cplex_direct",
) -> tuple[float, float, float, list[float]]:
    """Cheapest cost when the DEPARTURE INSTANT is chosen, not assumed.

    Returns ``(waiting_minutes, unserved_passengers, served_passengers,
    chosen_minutes)``.

    THE DIFFERENCE FROM `price_direction_at_minutes`. There, the departure instant is a
    constant supplied by a `DeparturePolicy` and only the passenger assignment is
    optimised. Here the instant is a decision: each departure picks ONE offset from the
    grid, every passenger boarding that departure boards at that same instant, and the
    assignment is optimised jointly with the choice.

    WHY THIS IS A MIP AND NOT AN LP -- the fact that shapes this whole direction. With
    both the departure instant `d_t` and the assignment `x[m,t]` free, the waiting cost
    `sum_m x[m,t] * (d_t - m)` is BILINEAR. Discretising `d_t` over a grid and selecting
    with a binary `z[t,k]` linearises it, at the cost of integrality. F2
    (`solve_minute_recourse(placement_offsets=...)`, D74) avoids the binary by letting
    every passenger pick its own offset independently, which decouples the product and
    yields a RELAXATION: physically impossible (one bus, one departure time), and
    strictly cheaper. The two therefore bracket the truth:

        Q_relaxed(O)  <=  Q_optimal(O)  <=  min over policies of Q_fixed

    NOT A CUT GENERATOR, AND NOT E2. `z` enters the capacity constraint's right-hand
    side, so this model does not satisfy the E2 condition the Benders subproblem relies
    on, and it returns no duals worth the name. It exists to VALUE placement freedom at
    a given schedule, not to generate cuts from it. A cut built from this function would
    over-estimate the recourse of the fixed-placement model and be invalid there; a cut
    valid for the optimal-placement model has to come from something at or below
    `Q_optimal`, which is what the F2 relaxation provides.

    NOT YET AN UPPER BOUND ON AN IMPLEMENTABLE SCHEDULE. `departures` here is aggregated
    across the fleet -- the per-vehicle assignment is not in the schedule
    representation this module receives -- so nothing here can check that shifting a
    departure later still leaves its vehicle able to return in time for its next
    committed activity. Shifting is therefore unconstrained by vehicle feasibility,
    which can only make the answer cheaper. Read this as a LOWER bound on what optimal
    placement could achieve, not as an achievable cost. Making it an upper bound needs
    the per-vehicle schedule and the precedence chain `d_next >= d_prev + duration`.
    """
    import pyomo.environ as pyo

    if not arrival_minutes or not departure_slots:
        return 0.0, float(len(arrival_minutes)), 0.0, []

    counts: dict[int, int] = {}
    for m in arrival_minutes:
        counts[int(m)] = counts.get(int(m), 0) + 1
    arrivals = sorted(counts)

    delta = float(slot_resolution)
    grid = _offset_grid(slot_resolution, offsets)
    deps = list(range(len(departure_slots)))
    ks = list(range(len(grid)))
    dep_minute = {
        (j, k): float(departure_slots[j]) * delta + grid[k] for j in deps for k in ks
    }

    arcs = [
        (m, j, k)
        for m in arrivals
        for j in deps
        for k in ks
        if 0.0 <= dep_minute[(j, k)] - float(m) <= float(wmax_minutes)
    ]

    mdl = pyo.ConcreteModel()
    mdl.Arcs = pyo.Set(initialize=arcs, dimen=3, ordered=False)
    mdl.Sel = pyo.Set(initialize=[(j, k) for j in deps for k in ks], dimen=2, ordered=False)
    mdl.x = pyo.Var(mdl.Arcs, within=pyo.NonNegativeReals)
    mdl.u = pyo.Var(arrivals, within=pyo.NonNegativeReals)
    mdl.z = pyo.Var(mdl.Sel, within=pyo.Binary)

    mdl.obj = pyo.Objective(
        expr=sum(
            (dep_minute[(j, k)] - float(m)) * mdl.x[m, j, k] for (m, j, k) in arcs
        )
        + float(p_minutes) * sum(mdl.u[m] for m in arrivals),
        sense=pyo.minimize,
    )

    mdl.Demand = pyo.Constraint(
        arrivals,
        rule=lambda mm, m: sum(mm.x[a, j, k] for (a, j, k) in arcs if a == m) + mm.u[m]
        == float(counts[m]),
    )
    # One instant per departure. This single row is the whole difference from F2:
    # drop it and each passenger may board the same bus at a different minute.
    mdl.OneInstant = pyo.Constraint(
        deps, rule=lambda mm, j: sum(mm.z[j, k] for k in ks) == 1
    )
    # Capacity does double duty: caps the load at S, and forces x to zero on every
    # instant this departure did not select.
    mdl.Capacity = pyo.Constraint(
        mdl.Sel,
        rule=lambda mm, j, k: (
            sum(mm.x[a, jj, kk] for (a, jj, kk) in arcs if jj == j and kk == k)
            <= float(seats) * mm.z[j, k]
            if any(jj == j and kk == k for (_a, jj, kk) in arcs)
            else pyo.Constraint.Skip
        ),
    )

    solver = pyo.SolverFactory(mip_solver)
    res = solver.solve(mdl, tee=False, load_solutions=False)
    term = getattr(res.solver, "termination_condition", None)
    if term != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f"optimal-placement MIP did not solve to optimality: termination={term}. "
            "Like the pricing LP it is always feasible -- unserved demand is absorbed "
            "by u at price p_minutes -- so a non-optimal termination is a defect."
        )
    mdl.solutions.load_from(res)

    waiting = sum(
        (dep_minute[(j, k)] - float(m)) * float(pyo.value(mdl.x[m, j, k]))
        for (m, j, k) in arcs
    )
    unserved = sum(float(pyo.value(mdl.u[m])) for m in arrivals)
    served = float(sum(counts.values())) - unserved
    chosen = [
        dep_minute[(j, k)]
        for j in deps
        for k in ks
        if float(pyo.value(mdl.z[j, k])) > 0.5
    ]
    return float(waiting), float(unserved), float(served), chosen


def price_schedule_optimal_placement(
    departures: dict[str, Sequence[int]],
    requests: dict[str, Sequence[int]],
    slot_resolution: int,
    seats: float,
    wmax_minutes: float,
    p_minutes: float,
    offsets: Sequence[float] | None = None,
    mip_solver: str = "cplex_direct",
) -> MinutePricingResult:
    """Price a whole schedule with the departure instant chosen, not assumed.

    The optimal-placement counterpart to `price_schedule_at_minutes`. See
    `price_direction_optimal_placement` for what this is and, more importantly, what it
    is not: not a cut generator, and not yet an upper bound on an implementable
    schedule.
    """
    waiting = 0.0
    unserved = 0.0
    served = 0.0
    used = 0
    for direction in (OUT, RET):
        slots = list(departures.get(direction, []))
        used += len(slots)
        w, u, s, _chosen = price_direction_optimal_placement(
            list(requests.get(direction, [])),
            slots,
            slot_resolution,
            seats,
            wmax_minutes,
            p_minutes,
            offsets,
            mip_solver,
        )
        waiting += w
        unserved += u
        served += s
    return MinutePricingResult(
        total_cost=waiting + float(p_minutes) * unserved,
        waiting_minutes=waiting,
        unserved_passengers=unserved,
        served_passengers=served,
        total_passengers=served + unserved,
        departures_used=used,
        policy="optimal",
    )


def price_schedule_optimal_placement_with_chain(
    vehicle_trips: dict[int, Sequence[tuple[int, str]]],
    charging_slots: dict[int, Sequence[int]],
    requests: dict[str, Sequence[int]],
    slot_resolution: int,
    seats: float,
    wmax_minutes: float,
    p_minutes: float,
    trip_duration_minutes: float,
    offsets: Sequence[float] | None = None,
    mip_solver: str = "cplex_direct",
) -> tuple[MinutePricingResult, dict[tuple[int, int, str], float]]:
    """Optimal placement that a vehicle can actually perform. Returns (result, minutes).

    THE DIFFERENCE FROM `price_schedule_optimal_placement`. That one shifts departures
    freely, because the schedule it receives is aggregated across the fleet and carries
    no way to tell which vehicle flies which trip. Free shifting is cheaper than
    reality, so it yields a LOWER bound on optimal placement -- useful for bracketing
    the prize, useless as a bound on an implementable schedule. This one takes the
    per-vehicle schedule and enforces what a vehicle can physically do, so its answer
    IS achievable: an upper bound, and the one that may serve as theta_opt.

    THE TWO COUPLINGS, both of which you get for free from a fixed-placement model and
    have to pay for the moment placement becomes a decision:

    1. PRECEDENCE. A vehicle's trips are ordered. Delaying one delays the return, which
       delays everything after it: `d_next >= d_prev + trip_duration`. The slack that
       absorbs a delay is whatever idle time the master's own schedule already left.

    2. CHARGING. The slot model credits `delta_chg` energy for a whole CHR slot. A trip
       that runs late enough to spill into a charging slot leaves less than a slot's
       worth of charging time, so the energy the master counted on is not there. Rather
       than re-derive the battery trajectory in minutes -- which would mean rebuilding
       the master's energy model here -- this refuses the encroachment outright: a trip
       may not overlap any slot its own vehicle spends charging. That is CONSERVATIVE.
       It forbids some shifts that a proportional-charging model would allow, so the
       cost it returns is attainable but possibly not the cheapest attainable. An upper
       bound that is honestly an upper bound, in the direction that cannot mislead.

    Note what constraint 2 implies when a trip fills its slot exactly
    (`trip_duration == slot_resolution`, the baseline's own case): the only freedom left
    is anticipation (D76), so a trip immediately PRECEDED by a charging slot cannot shift
    at all, and is pinned to offset 0 -- the same instant `start` already assumes. Much of
    the freedom the free-shifting evaluator reports may be unavailable for exactly this
    reason, which is the entire point of measuring rather than assuming.

    `vehicle_trips` maps a vehicle to its trips as ordered `(slot, direction)` pairs;
    `charging_slots` maps a vehicle to the slots it charges in.
    """
    import pyomo.environ as pyo

    delta = float(slot_resolution)
    dur = float(trip_duration_minutes)
    grid = _offset_grid(slot_resolution, offsets)
    ks = list(range(len(grid)))

    trips: list[tuple[int, int, str]] = []
    for q in sorted(vehicle_trips):
        for (slot, direction) in vehicle_trips[q]:
            trips.append((int(q), int(slot), str(direction)))
    if not trips:
        total = float(sum(len(list(requests.get(d, []))) for d in (OUT, RET)))
        return (
            MinutePricingResult(
                total_cost=float(p_minutes) * total, waiting_minutes=0.0,
                unserved_passengers=total, served_passengers=0.0,
                total_passengers=total, departures_used=0, policy="optimal+chain",
            ),
            {},
        )

    dep_minute = {
        (i, k): float(trips[i][1]) * delta + grid[k]
        for i in range(len(trips))
        for k in ks
    }

    counts: dict[str, dict[int, int]] = {}
    for direction in (OUT, RET):
        c: dict[int, int] = {}
        for m in requests.get(direction, []):
            c[int(m)] = c.get(int(m), 0) + 1
        counts[direction] = c

    arcs = [
        (m, i, k)
        for i in range(len(trips))
        for k in ks
        for m in sorted(counts[trips[i][2]])
        if 0.0 <= dep_minute[(i, k)] - float(m) <= float(wmax_minutes)
    ]

    mdl = pyo.ConcreteModel()
    mdl.Arcs = pyo.Set(initialize=arcs, dimen=3, ordered=False)
    mdl.Sel = pyo.Set(
        initialize=[(i, k) for i in range(len(trips)) for k in ks], dimen=2, ordered=False
    )
    mdl.x = pyo.Var(mdl.Arcs, within=pyo.NonNegativeReals)
    mdl.z = pyo.Var(mdl.Sel, within=pyo.Binary)
    unserved_index = [
        (direction, m) for direction in (OUT, RET) for m in sorted(counts[direction])
    ]
    mdl.u = pyo.Var(unserved_index, within=pyo.NonNegativeReals)

    mdl.obj = pyo.Objective(
        expr=sum((dep_minute[(i, k)] - float(m)) * mdl.x[m, i, k] for (m, i, k) in arcs)
        + float(p_minutes) * sum(mdl.u[d, m] for (d, m) in unserved_index),
        sense=pyo.minimize,
    )

    def _demand(mm, direction, m):
        served = sum(
            mm.x[a, i, k]
            for (a, i, k) in arcs
            if a == m and trips[i][2] == direction
        )
        return served + mm.u[direction, m] == float(counts[direction][m])

    mdl.Demand = pyo.Constraint(unserved_index, rule=_demand)
    mdl.OneInstant = pyo.Constraint(
        range(len(trips)), rule=lambda mm, i: sum(mm.z[i, k] for k in ks) == 1
    )
    mdl.Capacity = pyo.Constraint(
        mdl.Sel,
        rule=lambda mm, i, k: (
            sum(mm.x[a, ii, kk] for (a, ii, kk) in arcs if ii == i and kk == k)
            <= float(seats) * mm.z[i, k]
            if any(ii == i and kk == k for (_a, ii, kk) in arcs)
            else pyo.Constraint.Skip
        ),
    )

    # (1) Precedence, per vehicle, between consecutive trips in slot order.
    by_vehicle: dict[int, list[int]] = {}
    for i, (q, slot, _d) in enumerate(trips):
        by_vehicle.setdefault(q, []).append(i)
    for q in by_vehicle:
        by_vehicle[q].sort(key=lambda i: trips[i][1])

    precedence_pairs = [
        (by_vehicle[q][n], by_vehicle[q][n + 1])
        for q in sorted(by_vehicle)
        for n in range(len(by_vehicle[q]) - 1)
    ]

    def _precedence(mm, i, j):
        d_i = sum(dep_minute[(i, k)] * mm.z[i, k] for k in ks)
        d_j = sum(dep_minute[(j, k)] * mm.z[j, k] for k in ks)
        return d_j >= d_i + dur

    if precedence_pairs:
        mdl.Precedence = pyo.Constraint(precedence_pairs, rule=_precedence)

    # (2) No trip may overlap a slot its own vehicle spends charging.
    forbidden: list[tuple[int, int]] = []
    for i, (q, slot, _d) in enumerate(trips):
        chr_slots = {int(s) for s in charging_slots.get(q, [])}
        if not chr_slots:
            continue
        for k in ks:
            start = dep_minute[(i, k)]
            end = start + dur
            first_slot = int(start // delta)
            last_slot = int((end - 1e-9) // delta)
            if any(s in chr_slots for s in range(first_slot, last_slot + 1)):
                forbidden.append((i, k))
    if forbidden:
        mdl.NoChargeEncroachment = pyo.Constraint(
            forbidden, rule=lambda mm, i, k: mm.z[i, k] == 0
        )

    solver = pyo.SolverFactory(mip_solver)
    res = solver.solve(mdl, tee=False, load_solutions=False)
    term = getattr(res.solver, "termination_condition", None)
    if term != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            "chain-constrained optimal-placement MIP did not solve to optimality: "
            f"termination={term}. Infeasible here would be a real signal, not a defect: "
            "it would mean the master handed over a slot schedule no assignment of "
            "departure instants can perform."
        )
    mdl.solutions.load_from(res)

    waiting = sum(
        (dep_minute[(i, k)] - float(m)) * float(pyo.value(mdl.x[m, i, k]))
        for (m, i, k) in arcs
    )
    unserved = sum(float(pyo.value(mdl.u[d, m])) for (d, m) in unserved_index)
    total_pax = float(sum(sum(c.values()) for c in counts.values()))
    chosen = {
        trips[i]: dep_minute[(i, k)]
        for i in range(len(trips))
        for k in ks
        if float(pyo.value(mdl.z[i, k])) > 0.5
    }
    return (
        MinutePricingResult(
            total_cost=waiting + float(p_minutes) * unserved,
            waiting_minutes=float(waiting),
            unserved_passengers=float(unserved),
            served_passengers=total_pax - float(unserved),
            total_passengers=total_pax,
            departures_used=len(trips),
            policy="optimal+chain",
        ),
        chosen,
    )




def slot_objective_in_minutes(
    slot_waiting_cost: float, slot_unserved: float, slot_resolution: int, p_slots: float
) -> float:
    """Convert a slot-level recourse objective into passenger-minutes.

    The slot objective is `wait_slots + p_slots * unserved`, with both terms in slot
    units (D7/D8). Multiplying through by the slot width puts it in the same units as
    `price_schedule_at_minutes`, with `p_minutes = p_slots * slot_resolution` (D50).
    """
    delta = float(slot_resolution)
    return delta * float(slot_waiting_cost) + delta * float(p_slots) * float(
        slot_unserved
    )


def attach_minute_recourse(
    model: Any,
    requests: dict[str, Sequence[int]] | None,
    slot_resolution: int,
    seats: float,
    wmax_minutes: float,
    p_minutes: float,
    policy: DeparturePolicy = "start",
    objective_scale: float = 1.0,
    scenarios: Sequence[tuple[dict[str, Sequence[int]], float]] | None = None,
) -> None:
    """Pin a slot master's `theta` to a MINUTE-level recourse, in place.

    This is the architecture the research idea proposes -- first stage on slots,
    operational evaluation on minutes -- expressed monolithically so it can be solved
    exactly and compared against slot valuation without a decomposition in between.

    E2 IS WHAT MAKES IT LEGITIMATE, and it is respected here by construction. The minute
    grid and the arc set are functions of `(slot_resolution, wmax_minutes, policy)`
    alone. The schedule enters in ONE place, the capacity right-hand side:

        sum_m x[m, tau]  <=  seats * Yout[tau]

    `Yout`/`Yret` are the master's own aggregation variables, so `y` never touches which
    rows or variables exist. Build it any other way -- letting the schedule decide which
    minutes are reachable, say -- and `y` enters the constraint MATRIX, at which point no
    dual is a subgradient of the recourse and no cut built on it is valid. That is D30,
    and it cost this project six months of invalid bounds.

    The master is assumed to expose `Yout`/`Yret` and either `theta_out`/`theta_ret` or
    a scalar `theta`.

    `objective_scale` multiplies the recourse before it is pinned to theta. Pass
    `1/slot_resolution` to express a minute-accurate recourse in SLOT-equivalent units,
    which is what makes the resulting objective directly comparable to a slot-valued run:
    the first-stage terms (`start_cost_epsilon`, `concurrency_penalty`) are in slot units
    and would otherwise carry `slot_resolution` times less relative weight against a
    minute-scale theta, quietly changing how ties between schedules are broken.
    """
    import pyomo.environ as pyo

    # One scenario is the one-element case of many. Generalised in place rather than
    # copied into a second function: a duplicated construction is exactly how the two
    # `cplex_log` parsers and the three `Wmax` conversions drifted apart.
    if scenarios is None:
        if requests is None:
            raise ValueError("attach_minute_recourse needs `requests` or `scenarios`")
        scen: list[tuple[dict[str, Sequence[int]], float]] = [(requests, 1.0)]
    else:
        scen = [(r, float(w)) for r, w in scenarios]
        if not scen:
            raise ValueError("`scenarios` is empty")
        total_w = sum(w for _r, w in scen)
        if total_w <= 0.0:
            raise ValueError(
                f"scenario weights must sum to a positive value, got {total_w!r}"
            )
        scen = [(r, w / total_w) for r, w in scen]

    delta = float(slot_resolution)
    offset = placement_offset(policy, delta)
    taus = list(model.T)
    dep_minute = {int(t): float(t) * delta + offset for t in taus}
    S_IDX = list(range(len(scen)))

    def _pool(reqs, direction: str) -> dict[int, int]:
        counts: dict[int, int] = {}
        for m in reqs.get(direction, []):
            counts[int(m)] = counts.get(int(m), 0) + 1
        return counts

    pools = {(sx, d): _pool(scen[sx][0], d) for sx in S_IDX for d in (OUT, RET)}
    arcs = {
        (sx, d): [
            (sx, m, int(t))
            for m in sorted(pools[(sx, d)])
            for t in taus
            if 0.0 <= dep_minute[int(t)] - float(m) <= float(wmax_minutes)
        ]
        for sx in S_IDX
        for d in (OUT, RET)
    }
    all_out = [a for sx in S_IDX for a in arcs[(sx, OUT)]]
    all_ret = [a for sx in S_IDX for a in arcs[(sx, RET)]]
    dem_out = [(sx, m) for sx in S_IDX for m in sorted(pools[(sx, OUT)])]
    dem_ret = [(sx, m) for sx in S_IDX for m in sorted(pools[(sx, RET)])]

    model.MinArcsOut = pyo.Set(initialize=all_out, dimen=3, ordered=False)
    model.MinArcsRet = pyo.Set(initialize=all_ret, dimen=3, ordered=False)
    model.MinDemOut = pyo.Set(initialize=dem_out, dimen=2, ordered=False)
    model.MinDemRet = pyo.Set(initialize=dem_ret, dimen=2, ordered=False)
    model.xm_OUT = pyo.Var(model.MinArcsOut, within=pyo.NonNegativeReals)
    model.xm_RET = pyo.Var(model.MinArcsRet, within=pyo.NonNegativeReals)
    model.um_OUT = pyo.Var(model.MinDemOut, within=pyo.NonNegativeReals)
    model.um_RET = pyo.Var(model.MinDemRet, within=pyo.NonNegativeReals)

    def _group(arclist, key):
        out: dict[tuple[int, int], list] = {}
        for a in arclist:
            out.setdefault(key(a), []).append(a)
        return out

    bm_out = _group(all_out, lambda a: (a[0], a[1]))
    bm_ret = _group(all_ret, lambda a: (a[0], a[1]))
    bt_out = _group(all_out, lambda a: (a[0], a[2]))
    bt_ret = _group(all_ret, lambda a: (a[0], a[2]))

    model.MinDemandOut = pyo.Constraint(
        model.MinDemOut,
        rule=lambda mm, sx, m: sum(mm.xm_OUT[a] for a in bm_out.get((sx, m), []))
        + mm.um_OUT[sx, m]
        == float(pools[(sx, OUT)][int(m)]),
    )
    model.MinDemandRet = pyo.Constraint(
        model.MinDemRet,
        rule=lambda mm, sx, m: sum(mm.xm_RET[a] for a in bm_ret.get((sx, m), []))
        + mm.um_RET[sx, m]
        == float(pools[(sx, RET)][int(m)]),
    )
    # Capacity is per (scenario, slot) against the SAME Yout/Yret. That shared first
    # stage is what makes this one recourse problem rather than N separate ones, and it
    # is still right-hand side only, so E2 holds per scenario.
    model.MinCapOut = pyo.Constraint(
        S_IDX,
        model.T,
        rule=lambda mm, sx, t: (
            sum(mm.xm_OUT[a] for a in bt_out.get((sx, int(t)), []))
            <= float(seats) * mm.Yout[t]
            if bt_out.get((sx, int(t)))
            else pyo.Constraint.Skip
        ),
    )
    model.MinCapRet = pyo.Constraint(
        S_IDX,
        model.T,
        rule=lambda mm, sx, t: (
            sum(mm.xm_RET[a] for a in bt_ret.get((sx, int(t)), []))
            <= float(seats) * mm.Yret[t]
            if bt_ret.get((sx, int(t)))
            else pyo.Constraint.Skip
        ),
    )

    cost_out = sum(
        scen[sx][1]
        * (
            sum(
                (dep_minute[t] - float(m)) * model.xm_OUT[sx, m, t]
                for (_sx, m, t) in arcs[(sx, OUT)]
            )
            + float(p_minutes)
            * sum(model.um_OUT[sx, m] for m in sorted(pools[(sx, OUT)]))
        )
        for sx in S_IDX
    )
    cost_ret = sum(
        scen[sx][1]
        * (
            sum(
                (dep_minute[t] - float(m)) * model.xm_RET[sx, m, t]
                for (_sx, m, t) in arcs[(sx, RET)]
            )
            + float(p_minutes)
            * sum(model.um_RET[sx, m] for m in sorted(pools[(sx, RET)]))
        )
        for sx in S_IDX
    )

    k = float(objective_scale)
    if hasattr(model, "theta_out") and hasattr(model, "theta_ret"):
        model.MinThetaOut = pyo.Constraint(expr=model.theta_out == k * cost_out)
        model.MinThetaRet = pyo.Constraint(expr=model.theta_ret == k * cost_ret)
    elif hasattr(model, "theta"):
        model.MinTheta = pyo.Constraint(expr=model.theta == k * (cost_out + cost_ret))
    else:
        raise RuntimeError(
            "master exposes no theta variable to pin the minute recourse to"
        )


@dataclass(frozen=True)
class HonestWaitingReport:
    """Slot-model waiting against minute-level truth, per departure-placement policy."""

    slot_waiting_minutes: float
    slot_unserved: float
    served_slot: float
    minute: dict[str, MinutePricingResult]

    def avg_wait_slot_estimate(self) -> float:
        return (
            self.slot_waiting_minutes / self.served_slot if self.served_slot else 0.0
        )

    def format(self) -> str:
        lines: list[str] = []
        lines.append("Waiting, as the slot model reports it vs what it would really be")
        lines.append("-" * 68)
        lines.append(
            f"  slot model : {self.slot_waiting_minutes:9.1f} pax-min  "
            f"avg {self.avg_wait_slot_estimate():5.2f} min/pax  "
            f"unserved {self.slot_unserved:.0f}"
        )
        for policy, r in self.minute.items():
            avg = r.waiting_minutes / r.served_passengers if r.served_passengers else 0.0
            over = (
                100.0 * (self.slot_waiting_minutes - r.waiting_minutes) / r.waiting_minutes
                if r.waiting_minutes
                else float("nan")
            )
            lines.append(
                f"  minute ({policy:8s}): {r.waiting_minutes:9.1f} pax-min  "
                f"avg {avg:5.2f} min/pax  unserved {r.unserved_passengers:.0f}"
                f"   slot overstates by {over:+.1f}%"
            )
        lines.append("-" * 68)
        lines.append(
            "  The objective cannot see this: on baseline_d9 waiting is 6.8% of it and "
            "the\n  unmet-demand penalty is 93.2%. Quote the waiting figure from this "
            "table, not\n  from wait_cost_slots * slot_resolution."
        )
        return "\n".join(lines)


def honest_waiting(
    departures: dict[str, Sequence[int]],
    requests: dict[str, Sequence[int]],
    slot_resolution: int,
    seats: float,
    wmax_minutes: float,
    p_minutes: float,
    slot_waiting_cost_slots: float,
    slot_unserved: float,
    served_slot: float,
    policies: Iterable[DeparturePolicy] = ("start",),
    lp_solver: str = "cplex_direct",
) -> HonestWaitingReport:
    """Report the slot model's waiting estimate beside the minute-level truth.

    CORRECTED (D76). `start` is not one candidate among several: it is what the
    schedule's own t+1 commitment deterministically does, so there is no genuine
    convention uncertainty to bracket here -- `policies` defaults to `("start",)` alone.
    Pass additional policies explicitly only for a labelled counterfactual comparison
    ("what if boarding realistically eats N extra minutes"), not as competing estimates
    of the true cost.
    """
    return HonestWaitingReport(
        slot_waiting_minutes=float(slot_waiting_cost_slots) * float(slot_resolution),
        slot_unserved=float(slot_unserved),
        served_slot=float(served_slot),
        minute={
            policy: price_schedule_at_minutes(
                departures,
                requests,
                slot_resolution,
                seats,
                wmax_minutes,
                p_minutes,
                policy=policy,
                lp_solver=lp_solver,
            )
            for policy in policies
        },
    )


def solve_minute_recourse(
    T: int,
    slot_resolution: int,
    wmax_minutes: float,
    p_slots: float,
    C_out: Sequence[float],
    C_ret: Sequence[float],
    request_minutes: dict[str, Sequence[int]],
    policy: DeparturePolicy = "start",
    lp_solver: str = "cplex_direct",
    solver_options: dict | None = None,
    solve_time_limit_s: float | None = None,
    placement_offsets: Sequence[float] | None = None,
) -> tuple[dict[str, Any], float]:
    """A minute-level recourse LP with the SAME dual interface as the slot one.

    This is what makes a decomposed multi-resolution Benders cheap rather than a rewrite.
    The recourse is evaluated on minutes, but its capacity rows stay indexed by DEPARTURE
    SLOT -- one row per `(direction, tau)` with right-hand side `C_d[tau] = S*Y_d[tau]` --
    so it yields exactly one dual per slot, exactly as `solve_subproblem` does. The cut is
    therefore the same object the master already accepts:

        theta >= const + sum_tau S * pi_d[tau] * Y_d[tau]

    and everything downstream -- `aggregate_cuts_by_tau`, `_assert_q_invariant`, the
    validity classification, the master rows -- is untouched.

    UNITS. The objective is in SLOT-EQUIVALENT units: waiting is
    `(departure_minute - arrival_minute) / slot_resolution`, penalty is `p_slots`. Scaling
    an objective scales its duals identically, so `dm = S*pi` lands in the units the
    master's theta and first-stage terms already use. Without it the recourse would
    outweigh `start_cost_epsilon` and `concurrency_penalty` by a factor of
    `slot_resolution` and quietly change how ties between schedules are broken.

    E2 HOLDS BY CONSTRUCTION: the arc set is a function of
    `(T, slot_resolution, wmax_minutes, policy, placement_offsets)` and the demand, never
    of the schedule. `y` reaches this LP only as the capacity right-hand side.

    PLACEMENT FREEDOM (F2, docs/PROJECT_STATE_v6.md section 5, DESIGN_DD_v1 section 6).
    `placement_offsets`, when given, is a fixed offset grid `O subset [-delta, delta]` -- a
    constant, chosen once at load, not a decision variable. A passenger may then board the
    departure of slot `tau` at ANY candidate minute `tau*delta + o` for `o` in `O`, not only
    at the single instant `policy` implies. This is a RELAXATION, not a refinement: two
    passengers on the SAME physical departure may be priced as boarding it at different
    candidate minutes, which cannot happen on the real vehicle. So `Q_relaxed <= Q_true`,
    and a cut derived from it is a valid lower bound on `Q_true` -- but the schedule this
    recourse prices as an UPPER bound must not be trusted as one; only the master's cut-based
    lower bound may be read from a run using this.

    CORRECTED DIRECTION (D76). `tau*delta` (offset 0) is the master's own committed
    instant (D7/D8's t+1 rule), not the low end of a free window -- see `_offset_grid`.
    A grid built for the genuine degree of freedom this recourse should explore is
    anticipate-only, `O subset [-delta, 0]`; `D74`'s F2 measurement (grid `[0, delta]`,
    bidirectional in the OLD, incorrect sense) is reopened by this change, not reaffirmed
    by it, and needs re-measuring under the corrected grid before F2 is judged again.

    The capacity row stays ONE PER SLOT regardless of `len(placement_offsets)`: an arc now
    carries an extra offset index, but capacity constraints sum over that index before
    comparing to `S*Y_d[tau]`, so the row count, right-hand side and dual object are all
    identical to the single-offset case -- E2's condition on the interface, not merely on the
    arc set. `placement_offsets=None` (the default) is exactly today's single-offset LP: the
    internal representation always carries the extra offset index, with exactly one value in
    it, so the two code paths cannot silently diverge.
    """
    import time as _time

    import pyomo.environ as pyo

    t_build0 = _time.perf_counter()
    delta = float(slot_resolution)
    offsets = (
        [float(o) for o in placement_offsets]
        if placement_offsets
        else [placement_offset(policy, delta)]
    )
    if any(o < -delta or o > delta for o in offsets):
        raise ValueError(
            f"placement_offsets must lie in [-slot_resolution, slot_resolution] = "
            f"[-{delta}, {delta}], got {offsets!r}"
        )
    taus = list(range(int(T)))
    ks = list(range(len(offsets)))
    dep_minute = {(t, k): float(t) * delta + offsets[k] for t in taus for k in ks}

    pools: dict[str, dict[int, int]] = {}
    for d in (OUT, RET):
        counts: dict[int, int] = {}
        for m in request_minutes.get(d, []):
            counts[int(m)] = counts.get(int(m), 0) + 1
        pools[d] = counts

    caps = {OUT: list(C_out), RET: list(C_ret)}
    arcs = {
        d: [
            (m, t, k)
            for m in sorted(pools[d])
            for t in taus
            for k in ks
            if 0.0 <= dep_minute[(t, k)] - float(m) <= float(wmax_minutes)
        ]
        for d in (OUT, RET)
    }

    mdl = pyo.ConcreteModel()
    mdl.name = "subproblem_minutes"
    mdl.ArcsOut = pyo.Set(initialize=arcs[OUT], dimen=3, ordered=False)
    mdl.ArcsRet = pyo.Set(initialize=arcs[RET], dimen=3, ordered=False)
    mdl.x_OUT = pyo.Var(mdl.ArcsOut, within=pyo.NonNegativeReals)
    mdl.x_RET = pyo.Var(mdl.ArcsRet, within=pyo.NonNegativeReals)
    mdl.u_OUT = pyo.Var(sorted(pools[OUT]), within=pyo.NonNegativeReals)
    mdl.u_RET = pyo.Var(sorted(pools[RET]), within=pyo.NonNegativeReals)

    def _wait(m: int, t: int, k: int) -> float:
        return (dep_minute[(t, k)] - float(m)) / delta

    mdl.obj = pyo.Objective(
        expr=sum(_wait(m, t, k) * mdl.x_OUT[m, t, k] for (m, t, k) in arcs[OUT])
        + sum(_wait(m, t, k) * mdl.x_RET[m, t, k] for (m, t, k) in arcs[RET])
        + float(p_slots)
        * (
            sum(mdl.u_OUT[m] for m in sorted(pools[OUT]))
            + sum(mdl.u_RET[m] for m in sorted(pools[RET]))
        ),
        sense=pyo.minimize,
    )

    by_minute = {
        d: {m: [a for a in arcs[d] if a[0] == m] for m in sorted(pools[d])}
        for d in (OUT, RET)
    }
    # Grouped by SLOT ONLY, summing over every offset k -- this is the one line that
    # makes F2 a relaxation of capacity-per-slot rather than a finer-grained model:
    # the row count and right-hand side below are identical to the single-offset case.
    by_tau = {
        d: {t: [a for a in arcs[d] if a[1] == t] for t in taus} for d in (OUT, RET)
    }

    mdl.D_out = pyo.Constraint(
        sorted(pools[OUT]),
        rule=lambda mm, m: sum(mm.x_OUT[a] for a in by_minute[OUT][m]) + mm.u_OUT[m]
        == float(pools[OUT][m]),
    )
    mdl.D_ret = pyo.Constraint(
        sorted(pools[RET]),
        rule=lambda mm, m: sum(mm.x_RET[a] for a in by_minute[RET][m]) + mm.u_RET[m]
        == float(pools[RET][m]),
    )
    mdl.Cap_out = pyo.Constraint(
        taus,
        rule=lambda mm, t: (
            sum(mm.x_OUT[a] for a in by_tau[OUT][t]) <= float(caps[OUT][t])
            if by_tau[OUT][t]
            else pyo.Constraint.Skip
        ),
    )
    mdl.Cap_ret = pyo.Constraint(
        taus,
        rule=lambda mm, t: (
            sum(mm.x_RET[a] for a in by_tau[RET][t]) <= float(caps[RET][t])
            if by_tau[RET][t]
            else pyo.Constraint.Skip
        ),
    )
    mdl.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    t_build1 = _time.perf_counter()

    solver = pyo.SolverFactory(lp_solver)
    for k, v in (solver_options or {}).items():
        solver.options[k] = v
    if solve_time_limit_s is not None:
        solver.options["timelimit"] = float(solve_time_limit_s)
    t_solve0 = _time.perf_counter()
    res = solver.solve(mdl, tee=False, load_solutions=False)
    t_solve1 = _time.perf_counter()
    term = getattr(res.solver, "termination_condition", None)
    if term != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f"minute-level recourse LP did not solve to optimality: termination={term}. "
            "This LP is always feasible (unmet demand is absorbed by u at price p), so a "
            "non-optimal termination is a defect, not an instance."
        )
    mdl.solutions.load_from(res)

    t_extract0 = _time.perf_counter()
    pi_OUT = {
        t: (float(mdl.dual.get(mdl.Cap_out[t], 0.0)) if t in mdl.Cap_out else 0.0)
        for t in taus
    }
    pi_RET = {
        t: (float(mdl.dual.get(mdl.Cap_ret[t], 0.0)) if t in mdl.Cap_ret else 0.0)
        for t in taus
    }

    # Demand-row duals, keyed by ARRIVAL MINUTE -- which is what this LP's demand rows
    # are indexed by, and the reason the cut interface takes a scalar rather than a
    # vector (see MWDual). The scalar is `sum_m alpha[m] * pool[m]`, the demand side of
    # the dual objective, and it is what the cut constant is derived from (S2).
    #
    # This used to return `alpha_OUT: {}` with a comment saying a minute-indexed alpha
    # "has no slot-indexed meaning to report". True of the vector, false of the SUM --
    # and once S2 started deriving the constant from alpha, the empty dict made this
    # path produce an intercept of 0 against a recourse of 280, which the strong-duality
    # check then refused. The check was right; the omission was the defect.
    alpha_OUT = {
        int(m): float(mdl.dual.get(mdl.D_out[m], 0.0)) for m in sorted(pools[OUT])
    }
    alpha_RET = {
        int(m): float(mdl.dual.get(mdl.D_ret[m], 0.0)) for m in sorted(pools[RET])
    }
    intercept_out = float(
        sum(alpha_OUT[int(m)] * float(pools[OUT][m]) for m in sorted(pools[OUT]))
    )
    intercept_ret = float(
        sum(alpha_RET[int(m)] * float(pools[RET][m]) for m in sorted(pools[RET]))
    )

    served_out = [0.0] * int(T)
    served_ret = [0.0] * int(T)
    wait_out = wait_ret = 0.0
    for (m, t, k) in arcs[OUT]:
        v = float(pyo.value(mdl.x_OUT[m, t, k]))
        served_out[t] += v
        wait_out += _wait(m, t, k) * v
    for (m, t, k) in arcs[RET]:
        v = float(pyo.value(mdl.x_RET[m, t, k]))
        served_ret[t] += v
        wait_ret += _wait(m, t, k) * v
    unmet_out = sum(float(pyo.value(mdl.u_OUT[m])) for m in sorted(pools[OUT]))
    unmet_ret = sum(float(pyo.value(mdl.u_RET[m])) for m in sorted(pools[RET]))
    t_extract1 = _time.perf_counter()

    obj_val = float(pyo.value(mdl.obj))
    duals: dict[str, Any] = {
        # Keyed by ARRIVAL MINUTE, not by slot. Diagnostics only -- never sum these
        # against a slot-indexed demand vector.
        "alpha_OUT": alpha_OUT,
        "alpha_RET": alpha_RET,
        "intercept_out": intercept_out,
        "intercept_ret": intercept_ret,
        "pi_OUT": pi_OUT,
        "pi_RET": pi_RET,
        "served_out_by_tau": served_out,
        "served_ret_by_tau": served_ret,
        "served_out_by_tau_k": [[] for _ in taus],
        "served_ret_by_tau_k": [[] for _ in taus],
        "ub_out": float(wait_out + float(p_slots) * unmet_out),
        "ub_ret": float(wait_ret + float(p_slots) * unmet_ret),
        "objective_value": obj_val,
        "waiting_cost_slots": float(wait_out + wait_ret),
        "fill_eps_cost": 0.0,
        "penalty_cost": float(p_slots) * float(unmet_out + unmet_ret),
        "penalty_pax": float(unmet_out + unmet_ret),
        "served_total": float(sum(served_out) + sum(served_ret)),
        "total_demand": float(sum(pools[OUT].values()) + sum(pools[RET].values())),
        "is_feasible": True,
        "recourse_resolution": "minute",
        "departure_policy": str(policy),
        # F2. len(offsets) == 1 and offsets[0] == placement_offset(policy, delta) is
        # today's model exactly; anything else is the placement-freedom relaxation, and
        # any run using it must not report its upper bound as the schedule's true cost.
        "placement_offsets": list(offsets),
        "timing_build_s": float(t_build1 - t_build0),
        "timing_solve_s": float(t_solve1 - t_solve0),
        "timing_extract_s": float(t_extract1 - t_extract0),
        "timing_postprocess_s": 0.0,
        "timing_lp_export_s": 0.0,
        "exported_lp_path": None,
    }
    return duals, obj_val


__all__ = [
    "HonestWaitingReport",
    "attach_minute_recourse",
    "MinutePricingResult",
    "honest_waiting",
    "departure_minutes",
    "load_request_minutes",
    "price_direction_at_minutes",
    "price_schedule_at_minutes",
    "placement_offset",
    "slot_objective_in_minutes",
    "solve_minute_recourse",
]
