"""The one place a reported table is written, and the checks it has to pass first.

WHY THIS MODULE EXISTS. Every number this project has had to withdraw was withdrawn for
a reason that a machine could have caught at the moment the table was written:

  * 4183.24 was quoted "from an independent MILP" while no such code was in the
    repository (D50). Nothing recorded which instrument produced the row.
  * "390x slower" compared 301.4 s WITHOUT a proof of optimality against 0.8 s WITH
    one (audit item 2.2). Nothing checked that the two arms had terminated the same
    way before dividing one by the other.
  * A 228-iteration run that reached an 11.8% interval was DISCARDED because it
    contained a clock-truncated master solve (audit item 2.3). The protocol threw away
    the observation instead of labelling it.
  * "13.9-15.9 min" for one fixed schedule (audit item 3.6). Nothing required a
    waiting statistic to state its denominator.

So the checks live here, at the emitter, rather than in the head of whoever writes the
table. Three of them are refusals -- the emitter declines to write rather than writing
something that needs a caveat in prose:

  1. `manifest_required`  -- a table with no manifest id is not written.
  2. `one_manifest`       -- a table may not mix two manifest ids.
  3. `ratio_refused_on_status_mismatch` -- a time ratio between arms that terminated
     differently is not computed; the pair (time-to-proof, terminal gap) is emitted
     instead.

And one is a relabelling rather than a refusal: a censored run (B10) is KEPT, with the
count of truncated solves recorded, because discarding it loses the observation while
recording it loses nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# The manifest contract, in the order the work order states it. Order matters: the id
# is a hash over these fields IN THIS SEQUENCE, so two agents computing an id from the
# same run must list them the same way or the ids will differ for no reason.
MANIFEST_FIELDS: tuple[str, ...] = (
    "H",
    "delta",
    "Q",
    "S",
    "Emax",
    "b0",
    "c_trip",
    "rho",
    "tau_trip",
    "Wmax",
    "p_min",
    "epsilon",
    "kappa",
    "K_chg",
    "o",
    "same_slot_eligibility",
    "demand_checksum",
    "scenario_set",
    "scenario_weights",
    "objective_mode",
    "solver_name",
    "solver_version",
    "threads",
    "seed",
    "per_solve_budget_s",
    "loop_budget_s",
    "git_revision",
)


def manifest_id(fields: dict[str, Any]) -> str:
    """A short id over the manifest contract. Missing fields are recorded as missing.

    A field that is absent is hashed as the literal `null`, NOT skipped. Skipping would
    make a run that omitted `K_chg` collide with one that set it to the default, and
    the whole point of the id is that two rows sharing it were produced under the same
    conditions.
    """
    payload = [[name, fields.get(name)] for name in MANIFEST_FIELDS]
    blob = json.dumps(payload, sort_keys=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def manifest_fields_from_config(cfg, extra: dict[str, Any] | None = None) -> dict:
    """Map a loaded config onto the contract's field names.

    Written as an explicit mapping rather than a loop over `vars(cfg)` so that a
    renamed config key breaks HERE, loudly, instead of silently dropping out of the id
    and letting two different regimes share one.
    """
    sub = cfg.subproblem
    model = cfg.model
    p_min = (
        float(sub.p_minutes)
        if sub.p_minutes is not None
        else float(sub.p) * float(model.time.slot_resolution)
    )
    wmax = (
        float(sub.Wmax_minutes)
        if sub.Wmax_minutes is not None
        else float(sub.Wmax_slots) * float(model.time.slot_resolution)
    )
    binit = list(model.fleet.binit or [])
    out = {
        "H": model.time.T_minutes,
        "delta": model.time.slot_resolution,
        "Q": model.fleet.Q,
        "S": sub.S,
        "Emax": model.energy.Emax,
        "b0": binit[0] if binit else None,
        "c_trip": model.energy.L,
        "rho": None,  # filled by the caller, which knows the resolved delta_chg
        "tau_trip": model.time.trip_duration_minutes,
        "Wmax": wmax,
        "p_min": p_min,
        "epsilon": model.costs.start_cost_epsilon,
        "kappa": model.costs.concurrency_penalty,
        "K_chg": model.energy.K_chg,
        "o": list(sub.placement_offsets) if sub.placement_offsets else [0.0],
        "same_slot_eligibility": sub.same_slot_eligibility,
        "demand_checksum": None,
        "scenario_set": list(cfg.data.scenario_files or []),
        "scenario_weights": cfg.data.scenario_weights,
        "objective_mode": model.costs.objective_mode,
        "solver_name": cfg.solver.master_solver,
        "solver_version": None,
        "threads": (cfg.master.cplex_options or {}).get("CPXPARAM_Threads"),
        "seed": cfg.run.seed,
        "per_solve_budget_s": cfg.master.per_iteration_time_limit_s,
        "loop_budget_s": cfg.solver.total_time_limit_s,
        "git_revision": None,
    }
    out.update(extra or {})
    return out


def demand_checksum(paths: Iterable[str | Path]) -> str:
    """A checksum over the demand files, in the order given.

    Part of the contract because two runs quoting one manifest id must have priced the
    same passengers. A path is not enough: `setups/base.yaml` has been regenerated.
    """
    h = hashlib.sha256()
    for path in paths:
        p = Path(path)
        h.update(str(p.name).encode("utf-8"))
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"<unreadable>")
    return h.hexdigest()[:16]


@dataclass
class RunRecord:
    """One measured arm. What the emitter needs in order to refuse honestly."""

    label: str
    manifest: str
    status: str
    wall_time_s: float | None = None
    # B10. A run whose master solves stopped on the clock is KEPT and marked, not
    # discarded. Discarding it loses the observation; marking it loses nothing.
    clock_truncated_master_solves: int = 0
    lower_bound: float | None = None
    upper_bound: float | None = None
    # B10. Deterministic work, reported ALONGSIDE wall time. Wall time on a loaded
    # machine is not a property of the method; nodes and simplex iterations are, and
    # they are what makes a runtime comparison reproducible on another machine.
    nodes: int | None = None
    simplex_iterations: int | None = None
    deterministic_time: float | None = None
    notes: str = ""

    @property
    def censored(self) -> bool:
        return int(self.clock_truncated_master_solves) > 0

    @property
    def proved_optimality(self) -> bool:
        return str(self.status).upper() == "OPTIMAL" and not self.censored

    @property
    def terminal_gap(self) -> float | None:
        if self.lower_bound is None or self.upper_bound is None:
            return None
        return abs(self.upper_bound - self.lower_bound) / max(
            1.0, abs(self.upper_bound)
        )


class ManifestMismatch(ValueError):
    """Two rows in one table came from two different manifests."""


class MissingManifest(ValueError):
    """A table was written with no manifest id."""


class StatusMismatch(ValueError):
    """A ratio was asked for between arms that terminated differently."""


def time_ratio(fast: RunRecord, slow: RunRecord) -> float:
    """B11. The ratio, or a refusal.

    THE CLAIM THIS BLOCKS (audit item 2.2). "390x slower" divided 301.4 s by 0.8 s
    where the 301.4 s arm never proved optimality and the 0.8 s arm did. Those are not
    two measurements of one quantity. One is "time to reach a 27% interval and stop on
    the clock", the other is "time to prove the optimum"; their ratio is a number with
    no referent, and the direction of the error is not even known -- the truncated arm
    might have needed twice as long or a hundred times as long.

    So this refuses, and `proof_and_gap` gives the honest pair instead. It refuses on
    censoring too, not only on the status string: a run marked OPTIMAL that contains a
    clock-truncated master solve reached that status on a machine-load dependent search
    and its wall time is one draw from a distribution.
    """
    if fast.proved_optimality != slow.proved_optimality:
        raise StatusMismatch(
            f"refusing to compute a time ratio between {fast.label!r} and "
            f"{slow.label!r}: they did not terminate the same way "
            f"({fast.label}: status={fast.status} censored={fast.censored}; "
            f"{slow.label}: status={slow.status} censored={slow.censored}). "
            "A method that spends its budget without proving optimality is not "
            "'N times slower' than one that proves it -- the two numbers measure "
            "different events. Report the pair (time-to-proof, terminal gap) with "
            "`proof_and_gap` instead."
        )
    if not fast.wall_time_s or not slow.wall_time_s:
        raise ValueError("both arms need a wall time before a ratio means anything")
    return float(slow.wall_time_s) / float(fast.wall_time_s)


def proof_and_gap(record: RunRecord) -> dict[str, Any]:
    """B11. What to print when a ratio is refused.

    `time_to_proof_s` is None when the arm never proved optimality -- which is the
    fact the ratio was hiding, so it is reported as a null rather than as the wall
    clock under a different name.
    """
    return {
        "label": record.label,
        "manifest": record.manifest,
        "status": record.status,
        "censored": record.censored,
        "clock_truncated_master_solves": record.clock_truncated_master_solves,
        "time_to_proof_s": record.wall_time_s if record.proved_optimality else None,
        "wall_time_s": record.wall_time_s,
        "terminal_gap": record.terminal_gap,
        "nodes": record.nodes,
        "simplex_iterations": record.simplex_iterations,
        "deterministic_time": record.deterministic_time,
    }


@dataclass
class Table:
    """A reported table. Refuses to render until it can say what produced it."""

    title: str
    columns: Sequence[str]
    rows: list[dict[str, Any]] = field(default_factory=list)
    manifest: str | None = None
    # A table that could NOT be regenerated keeps its numbers and gains this label.
    # B9 requires the label to name the manifest and the reason, so both are required
    # together: a reason with no manifest is not a label, it is an excuse.
    legacy_label: str | None = None
    legacy_manifest: str | None = None

    def add(self, **row: Any) -> None:
        self.rows.append(dict(row))

    def render(self) -> str:
        if not self.manifest and not self.legacy_label:
            raise MissingManifest(
                f"refusing to render table {self.title!r}: it has no manifest id. "
                "Every reported figure carries the frozen record of what produced it "
                "(the shared contract with Agent CP-LBBD). A table that cannot name "
                "its manifest either needs one, or -- if it is carrying numbers that "
                "could not be regenerated -- needs `legacy_label` and "
                "`legacy_manifest` set, which renders a visible warning instead."
            )
        if self.legacy_label and not self.legacy_manifest:
            raise MissingManifest(
                f"table {self.title!r} is labelled as carrying legacy numbers but does "
                "not name the manifest they came from. The label exists so a reader "
                "can tell which regime produced the row; without the manifest it says "
                "only that someone knows it is stale."
            )
        seen = {
            str(r["manifest"]) for r in self.rows if r.get("manifest") is not None
        }
        if len(seen) > 1:
            raise ManifestMismatch(
                f"table {self.title!r} mixes {len(seen)} manifest ids: "
                f"{sorted(seen)}. No table may mix two manifests -- the rows were "
                "produced under different parameters and comparing them across the "
                "table is the error the manifest exists to prevent."
            )
        if seen and self.manifest and seen != {str(self.manifest)}:
            raise ManifestMismatch(
                f"table {self.title!r} declares manifest {self.manifest!r} but its "
                f"rows carry {sorted(seen)}"
            )

        lines: list[str] = [self.title]
        if self.legacy_label:
            lines.append(
                f"!! NOT REGENERATED: {self.legacy_label} "
                f"(numbers from manifest {self.legacy_manifest})"
            )
        if self.manifest:
            lines.append(f"manifest: {self.manifest}")
        lines.append(" | ".join(str(c) for c in self.columns))
        lines.append("-" * (3 * len(self.columns) + sum(len(str(c)) for c in self.columns)))
        for row in self.rows:
            lines.append(" | ".join(str(row.get(c, "")) for c in self.columns))
        censored = [r for r in self.rows if r.get("censored")]
        if censored:
            # B10. Named, not dropped. A censored row is evidence about the method's
            # behaviour under a budget, which is exactly what a runtime table is for.
            lines.append(
                f"censored rows: {len(censored)} of {len(self.rows)} contain at least "
                "one clock-truncated master solve. Their wall times are one draw from "
                "a machine-load dependent distribution; read the deterministic-work "
                "columns and the median trajectories, not the single number."
            )
        return "\n".join(lines)

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.render() + "\n", encoding="utf-8")
        return out


def median_trajectory(trajectories: Sequence[Sequence[float]]) -> list[float]:
    """B10. The pointwise median of several bound trajectories.

    Repetitions are summarised by their median rather than by one draw, because a
    censored run's trajectory depends on machine load. Trajectories of unequal length
    are compared only where they all have a value; truncating to the shortest is the
    honest choice, since extending the short ones would invent iterations that never
    happened.
    """
    if not trajectories:
        return []
    n = min(len(t) for t in trajectories)
    out: list[float] = []
    for i in range(n):
        column = sorted(float(t[i]) for t in trajectories)
        mid = len(column) // 2
        out.append(
            column[mid]
            if len(column) % 2
            else 0.5 * (column[mid - 1] + column[mid])
        )
    return out


def performance_profile(
    records: Sequence[RunRecord], ratios: Sequence[float] = (1.0, 2.0, 4.0, 8.0)
) -> list[tuple[float, float]]:
    """B10. Fraction of repetitions within `r` times the best wall time.

    A performance profile over repetitions instead of a single number, which is what
    the audit asks for. Censored runs are INCLUDED -- they are the runs whose behaviour
    under a budget the profile is meant to describe -- and a run with no wall time is
    excluded, since it contributes no observation.
    """
    times = [
        float(r.wall_time_s) for r in records if r.wall_time_s not in (None, 0.0)
    ]
    if not times:
        return [(float(r), 0.0) for r in ratios]
    best = min(times)
    return [
        (float(r), sum(1 for t in times if t <= float(r) * best) / len(times))
        for r in ratios
    ]


__all__ = [
    "MANIFEST_FIELDS",
    "ManifestMismatch",
    "MissingManifest",
    "RunRecord",
    "StatusMismatch",
    "Table",
    "demand_checksum",
    "manifest_fields_from_config",
    "manifest_id",
    "median_trajectory",
    "performance_profile",
    "proof_and_gap",
    "time_ratio",
]
