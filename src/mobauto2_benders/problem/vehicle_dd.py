"""A relaxed decision diagram for ONE vehicle, and the window trip caps it yields.

Purpose (DESIGN_DD_v1 stage 1, D48). D46 closed by naming the remaining lever on the
master's bound as "a valid inequality in `y` alone -- the per-vehicle trip cap derived
from the battery block, which, unlike the recourse anchor D33 found inert at Q=3, does
not go slack as Q grows". This module computes that cap by dynamic programming over a
single vehicle's reachable states, and emits

    sum_{tau in [t1,t2]} ( Y_OUT[tau] + Y_RET[tau] )  <=  Q * max_trips_in_window(t1,t2)

where `Y_d[tau] = sum_q y_d[q,tau]`.

WHY IT IS VALID. `max_trips_in_window` maximises over every entry state, so it bounds
the departures of ANY vehicle in that window whatever its history before `t1`. Summing
over Q independent, identical vehicles gives the right-hand side. E3 in the design doc
is what licenses "independent": every master row is separable by vehicle except the
three couplings that are functions of the aggregate.

The inequality is in `Y` alone, so it cannot interact with the validity of the Benders
cuts -- it constrains the first stage, not the recourse.

WHICH DIRECTION THE RELAXATIONS GO. A cap that is too LARGE is merely weak; a cap that
is too SMALL cuts off feasible schedules and destroys the master's status as a
relaxation, which is the D30/spec-2.8 failure mode. So every simplification here must
make the diagram MORE permissive, never less:

  - `charge_before_idle` (master C_no_recharge_after_idle) is dropped. It restricts
    when a vehicle may charge, so dropping it can only raise the trip count.
  - The entry battery is `Emax`, the most permissive value.
  - The entry location is maximised over {Longvilliers, Massy}.
  - Charging is capped by `Emax` and allowed only at Longvilliers while not departing
    and not in transit, which is what C2d + C1a + C1c say.

Constraints that are kept because they are real and tightening the cap with them is
sound: the trip-duration spacing (C1b/C1c/C2a), `b >= 2L` before an OUT (C5), the
battery balance and its non-negativity (C4), and the horizon-end departure fixings.

The dominance rule -- carry the maximum reachable battery per (time, state, trips) --
is exact for THIS relaxed diagram, because charging here is freely choosable in any
eligible slot, so more battery is weakly better for every continuation. That argument
does NOT survive re-adding `charge_before_idle`, where charging in a slot depends on
the previous slot's charge. Stage 3's pricing DP needs the un-relaxed version and must
verify the dominance by enumeration before relying on it (D48).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor


# Vehicle location codes. 0 = Longvilliers (depot, the only place with a charger),
# 1 = Massy -- the same encoding as master_impl.
LONGVILLIERS = 0
MASSY = 1


@dataclass(frozen=True)
class VehicleParams:
    """The single-vehicle physics, in the master's own units.

    `trip_slots`, `T` in slots; `Emax`, `L`, `delta_chg` in km-equivalent, matching
    `master.build_master`. There is no minutes/slots conversion anywhere in this module
    (D7/D8): callers pass what the master already computed.
    """

    T: int
    trip_slots: int
    Emax: float
    L: float
    delta_chg: float

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T!r}")
        if self.trip_slots < 1:
            raise ValueError(f"trip_slots must be >= 1, got {self.trip_slots!r}")
        if self.L <= 0:
            raise ValueError(f"L must be positive, got {self.L!r}")
        if self.Emax < 2 * self.L:
            # Not an error in itself, but it means no OUT is ever possible (C5), and a
            # caller almost certainly mis-wired the units rather than intending it.
            raise ValueError(
                f"Emax={self.Emax!r} is below 2*L={2 * self.L!r}, so C5 forbids every "
                "OUT departure and every trip cap would be 0. Check the units."
            )
        if self.delta_chg < 0:
            raise ValueError(f"delta_chg must be >= 0, got {self.delta_chg!r}")


def departure_eligible(P: VehicleParams, t: int, direction: int) -> bool:
    """Whether the master leaves `y_d[q,t]` free, for either direction.

    Mirrors the fixings in `build_master`:
      - slot 0 is the demand accumulation bucket, never a departure slot;
      - no departure at `t >= T - trip_slots` (it could not arrive by the horizon end);
      - no OUT at `t >= T - 2*trip_slots` (an OUT must leave room for its RET).
    """
    if t <= 0 or t >= P.T:
        return False
    if t >= P.T - P.trip_slots:
        return False
    if direction == LONGVILLIERS:  # an OUT departs FROM Longvilliers
        return t < max(0, P.T - 2 * P.trip_slots)
    return True


def max_trips_from(P: VehicleParams, t1: int) -> list[int]:
    """``[max_trips_in_window(P, t1, t2) for t2 in range(t1, T)]``, in one sweep.

    The DP advances slot by slot from `t1`, so the answer for every `t2 >= t1` falls out
    of the same pass. Computing them one window at a time is the same work `T` times
    over, which matters because `window_trip_caps` asks for all `O(T^2)` of them.

    Returns a list indexed by ``t2 - t1``.
    """
    t1 = max(0, int(t1))
    if t1 >= int(P.T):
        return []

    # State key: (loc, remaining) where `remaining` is slots left before the vehicle
    # becomes available at `loc`. remaining == 0 means available now.
    # Value: {trips_so_far: max_battery}
    def _merge(dst: dict[int, float], k: int, b: float) -> None:
        cur = dst.get(k)
        if cur is None or b > cur:
            dst[k] = b

    # Entry states: maximised over location, at full battery. In transit is never
    # better -- it only delays the first departure -- so it is not seeded.
    state: dict[tuple[int, int], dict[int, float]] = {
        (LONGVILLIERS, 0): {0: float(P.Emax)},
        (MASSY, 0): {0: float(P.Emax)},
    }

    out: list[int] = []
    for t in range(t1, int(P.T)):
        nxt: dict[tuple[int, int], dict[int, float]] = {}
        for (loc, rem), by_k in state.items():
            for k, b in by_k.items():
                if rem > 0:
                    # In transit: no departure, no charging (C1c gates c on inTrip).
                    _merge(nxt.setdefault((loc, rem - 1), {}), k, b)
                    continue

                if loc == LONGVILLIERS:
                    # OUT departure: C5 requires b >= 2L, and the balance takes L.
                    if departure_eligible(P, t, LONGVILLIERS) and b >= 2.0 * P.L:
                        _merge(
                            nxt.setdefault((MASSY, P.trip_slots - 1), {}),
                            k + 1,
                            b - P.L,
                        )
                    # Stay at Longvilliers. Charging dominates idling here (it can only
                    # raise the battery and nothing in this relaxed diagram penalises
                    # it), so the idle action is subsumed rather than enumerated.
                    _merge(
                        nxt.setdefault((LONGVILLIERS, 0), {}),
                        k,
                        min(float(P.Emax), b + float(P.delta_chg)),
                    )
                else:
                    # RET departure. C5 covers OUT only; the binding requirement for a
                    # RET is the balance itself, b[t+1] = b[t] - L >= 0.
                    if departure_eligible(P, t, MASSY) and b >= P.L:
                        _merge(
                            nxt.setdefault((LONGVILLIERS, P.trip_slots - 1), {}),
                            k + 1,
                            b - P.L,
                        )
                    # Wait at Massy. No charger there (C2d: c <= atL).
                    _merge(nxt.setdefault((MASSY, 0), {}), k, b)
        state = nxt
        # After processing slot `t`, `state` holds everything reachable having made all
        # decisions in [t1, t]. So this is the answer for the window [t1, t].
        best = 0
        for by_k in state.values():
            for k in by_k:
                if k > best:
                    best = k
        out.append(int(best))
    return out


def max_trips_in_window(P: VehicleParams, t1: int, t2: int) -> int:
    """Maximum departures one vehicle can START in ``[t1, t2]``, over all entry states."""
    if int(t2) < int(t1):
        return 0
    t1 = max(0, int(t1))
    t2 = min(int(P.T) - 1, int(t2))
    if t2 < t1:
        return 0
    sweep = max_trips_from(P, t1)
    return sweep[t2 - t1]


def travel_time_bound(P: VehicleParams, t1: int, t2: int) -> int:
    """Independent upper bound from trip spacing alone.

    Departures by one vehicle are at least `trip_slots` apart, so `k` departures span
    at least `(k-1)*trip_slots + 1` slots.
    """
    W = int(t2) - int(t1) + 1
    if W <= 0:
        return 0
    return 1 + int(floor((W - 1) / P.trip_slots))


def energy_bound(P: VehicleParams, t1: int, t2: int) -> int:
    """Independent upper bound from energy alone.

    `k` departures consume `k*L`. The vehicle starts with at most `Emax` and can charge
    in at most `W - k` of the window's slots (a departure slot is never a charging slot,
    C1a), each adding at most `delta_chg`:

        k*L <= Emax + delta_chg * (W - k)   =>   k <= (Emax + delta_chg*W) / (L + delta_chg)
    """
    W = int(t2) - int(t1) + 1
    if W <= 0:
        return 0
    return int(floor((float(P.Emax) + float(P.delta_chg) * W) / (float(P.L) + float(P.delta_chg))))


def eligible_slot_count(P: VehicleParams, t1: int, t2: int) -> int:
    """Departures the window admits with no per-vehicle reasoning at all.

    C1a gives `yOUT[q,t] + yRET[q,t] <= 1`, so `Y_OUT[t] + Y_RET[t] <= Q` in any slot
    where a departure is allowed at all. A window cap is only worth emitting when it is
    strictly below this.
    """
    return sum(
        1
        for t in range(max(0, int(t1)), min(int(P.T), int(t2) + 1))
        if departure_eligible(P, t, LONGVILLIERS)
        or departure_eligible(P, t, MASSY)
    )


@dataclass(frozen=True)
class WindowCap:
    """``sum_{tau in [t1,t2]} (Y_OUT[tau] + Y_RET[tau]) <= rhs``."""

    t1: int
    t2: int
    max_trips_per_vehicle: int
    Q: int

    @property
    def rhs(self) -> int:
        return int(self.Q) * int(self.max_trips_per_vehicle)


def window_trip_caps(
    P: VehicleParams, Q: int, min_window: int = 1
) -> list[WindowCap]:
    """Every non-trivial window cap, one per ``[t1,t2]``.

    A cap is emitted only when `Q * max_trips` is strictly below the trivial bound
    `Q * eligible_slot_count`, i.e. when the per-vehicle reasoning says something that
    `Y_OUT[t] + Y_RET[t] <= Q` does not already say.

    Caps dominated by a sub-window are NOT filtered here. That is deliberate: which
    subset is worth adding to the master is a measurement (spec 2.9 records a sound
    inequality that made the master slower and its bound worse), so the caller chooses,
    and this function reports everything it can prove.
    """
    if int(Q) < 1:
        raise ValueError(f"Q must be >= 1, got {Q!r}")
    caps: list[WindowCap] = []
    for t1 in range(P.T):
        sweep = max_trips_from(P, t1)
        for t2 in range(t1 + int(min_window) - 1, P.T):
            trivial = eligible_slot_count(P, t1, t2)
            if trivial == 0:
                continue
            k = sweep[t2 - t1]
            # A defect in the DP that OVER-counts makes the cap weak; one that
            # UNDER-counts cuts off feasible schedules. Both analytic bounds are
            # independent of the DP, so a violation here is a bug in the diagram and
            # not a property of the instance -- refuse rather than emit.
            tt = travel_time_bound(P, t1, t2)
            en = energy_bound(P, t1, t2)
            if k > tt or k > en:
                raise RuntimeError(
                    f"single-vehicle DP exceeded an independent bound on window "
                    f"[{t1},{t2}]: dp={k} travel_time={tt} energy={en}. The diagram "
                    "is wrong; emitting this cap would still be valid but the "
                    "disagreement means the DP cannot be trusted for stage 3 pricing."
                )
            if k < trivial:
                caps.append(
                    WindowCap(t1=t1, t2=t2, max_trips_per_vehicle=k, Q=int(Q))
                )
    return caps
