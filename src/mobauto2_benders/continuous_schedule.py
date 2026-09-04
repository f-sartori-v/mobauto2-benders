"""Load a continuous-time schedule (B4, handout item, closes report Comparison C's
missing half) -- the exchange format a model that is NOT confined to a slot grid (the
CP/LBBD engine, a separate repository) uses to hand a schedule to this repository's
minute-fidelity pricing path.

D72 closed the other half: `minute_pricer.price_schedule_given_departure_minutes` already
accepts departures at arbitrary minutes (37.5 and 82.25 are in its own tests -- not
`tau*delta + offset` for any integer tau and any placement policy). What was missing was a
documented format for a continuous-time schedule to arrive in, and a loader that turns it
into what that function actually consumes: `{"OUT": [minute, ...], "RET": [minute, ...]}`,
one flat list per direction, vehicle identity discarded (pricing only cares how many seats
depart when, not which vehicle carries them -- the same convention the slot-based
`_schedule()` helpers in this project's own scripts already use).

FORMAT (schema `mobauto2_continuous_schedule`, version 1; documented in full in
`docs/BENDERS_SPEC_v4.md` section "Continuous-time schedule exchange format"):

    schema: mobauto2_continuous_schedule
    version: 1
    horizon_minutes: <float>
    seats: <float>
    vehicles:
      - id: <any>
        departures:
          OUT: [<minute>, ...]
          RET: [<minute>, ...]
        charging_plan:
          - {start_minute: <float>, end_minute: <float>}
          ...

Units are minutes from the horizon start, matching every other minute-level quantity in
this repository (D50) -- not slots, and not required to be a multiple of any slot width.
`charging_plan` is carried through for completeness (a continuous-time schedule is not
fully specified without it -- a vehicle's departures must be feasible against its own
charge state) but is NOT consumed by the pricing path below, which only prices the
passenger side; a future feasibility check against the energy model is a separate piece of
work, not this one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import yaml

from .minute_pricer import (
    DEFAULT_SAME_SLOT_ELIGIBILITY,
    OUT,
    RET,
    MinutePricingResult,
    price_schedule_given_departure_minutes,
)

SCHEMA_NAME = "mobauto2_continuous_schedule"
SCHEMA_VERSION = 1
DIRECTIONS = (OUT, RET)


@dataclass(frozen=True, slots=True)
class ChargingEvent:
    start_minute: float
    end_minute: float


@dataclass(frozen=True, slots=True)
class VehicleSchedule:
    id: str
    departures: dict[str, tuple[float, ...]]
    charging_plan: tuple[ChargingEvent, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ContinuousSchedule:
    horizon_minutes: float
    seats: float
    vehicles: tuple[VehicleSchedule, ...]


def _fail(msg: str) -> None:
    # Fail-closed: a malformed schedule must raise, never silently drop a vehicle,
    # a direction, or a departure -- a dropped departure looks identical to a
    # schedule that legitimately serves fewer passengers, and nothing downstream
    # could tell the two apart.
    raise ValueError(f"continuous schedule: {msg}")


def parse_continuous_schedule(raw: dict) -> ContinuousSchedule:
    """Validate and structure a parsed YAML/JSON document. Raises ValueError on any
    malformed input rather than silently coercing or dropping data (receitas-basicas:
    fail-closed, no swallowed exceptions)."""
    if not isinstance(raw, dict):
        _fail(f"top level must be a mapping, got {type(raw).__name__}")

    schema = raw.get("schema")
    if schema != SCHEMA_NAME:
        _fail(f"schema must be {SCHEMA_NAME!r}, got {schema!r}")
    version = raw.get("version")
    if version != SCHEMA_VERSION:
        _fail(f"version must be {SCHEMA_VERSION!r}, got {version!r}")

    if "horizon_minutes" not in raw:
        _fail("missing required key 'horizon_minutes'")
    horizon_minutes = float(raw["horizon_minutes"])
    if horizon_minutes <= 0:
        _fail(f"horizon_minutes must be positive, got {horizon_minutes}")

    if "seats" not in raw:
        _fail("missing required key 'seats'")
    seats = float(raw["seats"])
    if seats <= 0:
        _fail(f"seats must be positive, got {seats}")

    vehicles_raw = raw.get("vehicles")
    if not isinstance(vehicles_raw, list) or not vehicles_raw:
        _fail("'vehicles' must be a non-empty list")

    seen_ids: set[str] = set()
    vehicles: list[VehicleSchedule] = []
    for i, v in enumerate(vehicles_raw):
        if not isinstance(v, dict):
            _fail(f"vehicles[{i}] must be a mapping, got {type(v).__name__}")
        if "id" not in v:
            _fail(f"vehicles[{i}] missing required key 'id'")
        vid = str(v["id"])
        if vid in seen_ids:
            _fail(f"duplicate vehicle id {vid!r}")
        seen_ids.add(vid)

        dep_raw = v.get("departures")
        if not isinstance(dep_raw, dict):
            _fail(f"vehicle {vid!r}: 'departures' must be a mapping of direction -> minutes")
        unknown = set(dep_raw) - set(DIRECTIONS)
        if unknown:
            _fail(f"vehicle {vid!r}: unknown direction(s) {sorted(unknown)}, expected {DIRECTIONS}")
        departures: dict[str, tuple[float, ...]] = {}
        for d in DIRECTIONS:
            minutes = dep_raw.get(d, [])
            if not isinstance(minutes, list):
                _fail(f"vehicle {vid!r} direction {d!r}: must be a list of minutes")
            parsed: list[float] = []
            for m in minutes:
                fm = float(m)
                if not (0.0 <= fm <= horizon_minutes):
                    _fail(
                        f"vehicle {vid!r} direction {d!r}: departure at minute {fm} "
                        f"outside horizon [0, {horizon_minutes}]"
                    )
                parsed.append(fm)
            departures[d] = tuple(parsed)

        charging_raw = v.get("charging_plan", [])
        if not isinstance(charging_raw, list):
            _fail(f"vehicle {vid!r}: 'charging_plan' must be a list")
        events: list[ChargingEvent] = []
        for j, c in enumerate(charging_raw):
            if not isinstance(c, dict) or "start_minute" not in c or "end_minute" not in c:
                _fail(
                    f"vehicle {vid!r} charging_plan[{j}]: must have "
                    "'start_minute' and 'end_minute'"
                )
            start_m = float(c["start_minute"])
            end_m = float(c["end_minute"])
            if not (end_m > start_m):
                _fail(
                    f"vehicle {vid!r} charging_plan[{j}]: end_minute ({end_m}) must "
                    f"exceed start_minute ({start_m})"
                )
            events.append(ChargingEvent(start_minute=start_m, end_minute=end_m))

        vehicles.append(
            VehicleSchedule(id=vid, departures=departures, charging_plan=tuple(events))
        )

    return ContinuousSchedule(
        horizon_minutes=horizon_minutes, seats=seats, vehicles=tuple(vehicles)
    )


def load_continuous_schedule(path: str | Path) -> ContinuousSchedule:
    """Read and validate a continuous-time schedule document from a YAML file."""
    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    return parse_continuous_schedule(raw)


def to_departures_minutes(schedule: ContinuousSchedule) -> dict[str, list[float]]:
    """Flatten every vehicle's departures into the per-direction pool
    `price_schedule_given_departure_minutes` consumes. Vehicle identity is
    intentionally discarded here -- pricing only needs how many seats depart when,
    matching the slot-based `_schedule()` helpers elsewhere in this project, which
    do the same collapse over vehicles."""
    out: dict[str, list[float]] = {d: [] for d in DIRECTIONS}
    for vehicle in schedule.vehicles:
        for d in DIRECTIONS:
            out[d].extend(vehicle.departures.get(d, ()))
    return out


def price_continuous_schedule(
    schedule: ContinuousSchedule,
    requests: dict[str, Sequence[int]],
    wmax_minutes: float,
    p_minutes: float,
    lp_solver: str = "cplex_direct",
    same_slot_eligibility: str = DEFAULT_SAME_SLOT_ELIGIBILITY,
) -> MinutePricingResult:
    """Price a loaded continuous-time schedule at minute fidelity. `schedule.seats`
    is used as the per-departure capacity, matching every departure in the file --
    this repository does not (yet) support a heterogeneous fleet with different
    seat counts per vehicle.

    `same_slot_eligibility` (B6) is exposed rather than left at whatever the pricer
    defaults to, because this is Comparison C's instrument: a CP schedule priced under
    one convention against a Benders schedule priced under another would attribute the
    convention difference to the engines. The value travels on the returned
    MinutePricingResult, so a table cannot fail to state it.
    """
    departures_minutes = to_departures_minutes(schedule)
    return price_schedule_given_departure_minutes(
        departures_minutes,
        requests,
        schedule.seats,
        wmax_minutes,
        p_minutes,
        lp_solver,
        same_slot_eligibility=same_slot_eligibility,
    )
