# Forward plan — v1

Source: the T.5.4 report's "Forward work" and "Open directions" sections (outside this
repository, in the `8_Deliver` tree) and `docs/PROJECT_STATE_v6.md` §5. This file turns those
into commit-sized items **for this repository only**. It is not a research note and asserts no
finding — every entry below is a thing to build or measure, not a result. When one produces a
number, it goes into `docs/docs_decisions.md` with its own D-number, the same as any other
measurement in this project.

Two things this file is not a substitute for: the two live bound-side levers (F2 and the level
method) are tracked in `docs/PROJECT_STATE_v6.md` §5 with their own falsifiers and are not
repeated here. And nothing on this list overrides a direction `docs/PROJECT_STATE_v6.md` §2/§5
records as **closed by measurement** — cumulative-prefix symmetry, more master seconds, the
per-iteration time limit as a tuning knob, and the rest of that list must not be reopened by
any item below.

---

## A. Actionable now — in-repo, no external dependency

Ranked by the report's own "expected value to the project" ordering, restricted to the items
that are code-and-measurement work in this repo.

### A1. Regenerate the stochastic-robustness result set at `p_minutes ≈ 56`, minute-level valuation — ✅ done, D70

Hedging costs **0.9%** in passenger-minutes (weighted average across the four scenarios),
against the withdrawn conference-era figure. See `docs_decisions.md` D70 for the full table,
the script (`scripts/stochastic_robustness.py`), and a caveat about the per-scenario oracle
that must travel with the number. Left open by D70, not part of A1's original scope: a
departure-pattern comparison to re-derive the qualitative "spreads departures" claim verbatim,
and a genuinely minute-optimal per-scenario oracle (would tighten, not overturn, the 0.9%).

**What.** Re-run the four-scenario comparison — one schedule serving all four demand scenarios
against the per-scenario deterministic optima — under the corrected penalty
(`configs/baseline_d9_p56.yaml` or equivalent), and price every resulting schedule with the
minute-level validator (report operation (i): fixed schedule, re-assign passengers at minute
resolution).

**Why first.** The report calls this "the only result that speaks directly to a decision an
operator has to make," and it is the one result set still standing on two withdrawn numbers:
the `p=1500` penalty regime and slot-only waiting valuation (`docs/PROJECT_STATE_v6.md` §3).

**Done when.** New numbers replace the flagged conference-era figures, and the qualitative
mechanism (planning under uncertainty spreads departures and trades served passengers for
waiting regularity) is checked against the new penalty rather than assumed to survive. If it
doesn't survive, that is a finding to report, not a bug to fix.

**Shape.** Extends `scripts/multiscenario_check.py`; a `data/measurements.json` entry and
(if the report gets a new table for it) a figure script under `scripts/report_figures/`.
Solver runs: cap at 120 s per run unless the converged number is the actual deliverable, and
say so before running to full convergence. Ask before starting.

### A2. Sweep `p_minutes` and `W^max` jointly; publish the frontier

**What.** A two-dimensional sweep over the penalty and the maximum waiting time, on the
baseline instance, reporting served/unserved counts and waiting at each cell.

**Why.** `p_minutes ≈ 56` is stated in the report as an assumption adopted for this work, not
an elicited operator preference. The report's own next step is "to sweep it jointly with
`W^max` and put the resulting frontier in front of whoever will operate the service."

**Done when.** The frontier is produced by a script that ships, from configs that ship — same
bar as every other figure in the report.

**Shape.** New `scripts/sweep_penalty_window.py`, patterned on
`scripts/sweep_multiresolution.py`; a figure script under `scripts/report_figures/`; a
`data/measurements.json` entry.

### A3. Validate at the official target service level (450 passengers/day, 30 trips/day)

**What.** Build an instance at the declared trial scale and run the fleet sweep against it.

**Why.** Every instance reported so far runs 300–400 requests; the project's own declared
service level (deliverable 1.4.3) has never been exercised.

**Done when.** A run exists at that scale with a recorded outcome, whichever way it goes. If
the model does not reach a certified bound in a reasonable budget at that scale, that is itself
a result and connects to Claim 2 (the decomposition is not competitive on this family) rather
than being a failed experiment to hide.

**Shape.** New instance file(s) under `setups/`, a config under `configs/`, a
`docs/docs_decisions.md` entry.

### A4. Complete the comparative protocol (report §"Comparative evaluation protocol", Comparisons A–C)

Four independently commit-sized sub-items; the report already states exactly which parts of
the protocol are unmet.

- **A4a — instrument time-to-first-feasible and the runtime split** — ✅ done, D71.
  `BendersRunResult` and the manifest now carry `time_to_first_feasible_s`,
  `total_wall_time_s`, `total_master_time_s`, `total_sp_solve_time_s`, `total_cutgen_time_s`,
  `total_cutadd_time_s`, `model_management_overhead_s`. See `docs_decisions.md` D71 for what it
  does and does not cover — it does not re-derive or refute the withdrawn 85.7% figure, and
  "time to a solution within fixed thresholds of the optimum" is still open, needing a
  reference optimum to measure against.
- **A4b — run the unrestricted continuous-time CP configuration through the common minute-level
  validator** (Comparison B). The CP side already exists and has been run; what's missing is
  routing its output through the same validator the MILP and Benders schedules go through.
- **A4c — extend the minute-level validator to accept departure minutes directly.** It
  currently prices schedules given as departure *slots*. This is the blocking piece for
  Comparison C and is named in the report as "a small extension" not yet made.
- **A4d — run the `δ × Q` factorial**, `δ ∈ {30, 15, 10} min`, `Q ∈ {2, 3, 4, 5}`, across
  smooth/peaked/sub-slot demand shapes, reporting the four effects separately (grid refinement,
  minute-level valuation, continuous departure placement, multi-scenario). Gated on A4c for the
  Comparison-C leg; the rest can run against the existing MILP/Benders/CP outputs. This is the
  largest compute item on this list — budget and ask before running the full grid, not just one
  cell.

### A5. Piecewise-linear charging; re-run the engine comparison

**What.** Replace the fixed-rate charging block with a two-phase curve (fast below a SoC
threshold, slower above roughly 80%), gated behind a config flag so existing configs keep
reproducing their current numbers unchanged.

**Why.** Already named in the assumptions table as the consequence of relaxing "linear charging
at a fixed rate": it changes the value of short top-ups and therefore where charging events get
placed.

**Done when.** Either it moves charging-event placement measurably (report it), or it doesn't
at the current parameters (report that too, the way the closed directions in Claim 2 are
reported — a negative result with a number, not a shrug).

**Shape.** A model change behind a flag in the affected engine(s); the engine comparison
(A4-adjacent) re-run so old and new stay like-for-like.

---

## B. Already tracked elsewhere — not duplicated here

- **F2 — placement freedom in the subproblem.** `docs/PROJECT_STATE_v6.md` §5. Falsifier fixed:
  refuted if cut strength at a fixed budget does not improve beyond run-to-run noise.
- **Stabilisation by a level method.** `docs/PROJECT_STATE_v6.md` §5. Falsifier fixed: refuted
  if the 1064–1090 bound band persists with no trend.

---

## C. Blocked on another team or task — do not start without that input

- **Replace synthetic demand with corridor proxies** (motorway loop-detector counts, line 91.03
  flows, the Task 1.4 origin–destination reconstruction pipeline). The pipeline is identified
  and not yet ingested; this repo can consume it once it lands but cannot produce it.
- **Couple the calibrated choice model.** Owned by the demand team. The report is explicit that
  the first step is iterative coupling (solve, read off headways/waiting, update demand,
  re-solve), not direct embedding — direct embedding is the harder route and comes later.
- **Explicit objective decomposition (`f1` waiting, `f2` unserved demand, `f3` trips) with an
  independent monolithic benchmark per term.** Stated in the report as *design for the next
  phase, not measurement* — nothing in the current results depends on it. Worth prototyping
  once A2 (policy-parameter sweeps) exists, since the two share the same policy-separation
  groundwork, but it is a design item until then, not a committed feature.

---

## D. Reading only — no code, and it gates a claim

- **Full-text review of the two Dynamic Discretization Discovery papers** before writing the
  related-work paragraph for any manuscript. The literature check that narrowed Claim 3
  (`docs_decisions.md` D69) was abstract-level only; the residual is to read both papers in full
  and confirm neither confines its refinement to the recourse the way this work's construction
  does. This is the one item the narrowed novelty claim rests on most, and the one the search
  covered least well.

---

## What this file does not do

It does not rank C or D against A — they are blocked or non-code respectively, not lower
priority. It does not allocate a D-number to anything: a D-number is earned by a measurement
landing in `docs_decisions.md`, not by being planned here. And it does not itself expire — when
an item is done, remove it from here and cite where its number now lives, the same discipline
`docs/PROJECT_STATE_v6.md` §3 applies to superseded values.
