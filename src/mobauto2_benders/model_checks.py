"""Postconditions on a built model, checkable after the fact.

WHY A CHECK AND NOT JUST A FIXING (B3). `c[q,T-1].fix(0.0)` is the right thing to do,
and it is done -- but a fixing is not a guarantee. Pyomo lets a later `set_value` change
a fixed variable's value without complaint, and what the solver receives is whatever the
variable holds at write time. So "the model forbids final-slot charging" is a claim
about the model as it stands when it is solved, and the honest way to assert it is to
look at the model as it stands.

That distinction is not academic here. The whole reason the final-slot leak survived is
that `c[q,T-1]` and `gchg[q,T-1]` appear in no surviving row and no cost term: nothing
downstream would notice a nonzero value, so nothing downstream can be relied on to
catch one.
"""

from __future__ import annotations

from typing import Any


def final_slot_energy_violations(model: Any, tol: float = 1e-9) -> list[str]:
    """B3. Vehicles charging in the last slot, as a list of readable violations.

    Empty means the model, as it stands, commits no energy in `T-1`. A non-empty list
    names the vehicle and the value, because "some vehicle charges at the end" is not
    something anyone can act on.
    """
    bad: list[str] = []
    try:
        T = len(list(model.T))
    except Exception:
        return ["model exposes no time index T"]
    if T < 2:
        return bad
    last = T - 1
    for name in ("c", "gchg"):
        var = getattr(model, name, None)
        if var is None:
            continue
        for q in model.Q:
            try:
                value = var[q, last].value
            except Exception:
                continue
            if value is None:
                continue
            if abs(float(value)) > float(tol):
                bad.append(
                    f"{name}[{q},{last}] = {float(value):.6g}: energy committed in the "
                    "final slot, where the SoC recursion has already ended. It changes "
                    "no state and enters no cost, so it makes the reported schedule's "
                    "last slot arbitrary."
                )
            if not var[q, last].is_fixed():
                bad.append(
                    f"{name}[{q},{last}] is not fixed: the final slot is free, which is "
                    "the leak itself rather than an instance of it."
                )
    return bad


def assert_no_final_slot_charging(model: Any, tol: float = 1e-9) -> None:
    """Raise if `final_slot_energy_violations` finds anything."""
    bad = final_slot_energy_violations(model, tol)
    if bad:
        raise ValueError(
            "final-slot energy leak (B3, audit item 1.7):\n  " + "\n  ".join(bad)
        )


__all__ = ["assert_no_final_slot_charging", "final_slot_energy_violations"]
