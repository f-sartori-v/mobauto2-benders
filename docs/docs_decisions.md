# docs/decisions.md — MOB-AUTO2 Benders

One entry per decision. Append-only; never edit a past entry, add a new one that supersedes it.

---

### D1 — Charging rate frozen at ρ = 70 km-eq/h, linear
Resolves audit A1. Full charge ≈ 2.5 h (closer to the original "2 h 30" concept than the 75 km/h
figure used in `[R]`). Linear model kept for now; piecewise-linear remains future work (§9 of the
consolidated document).

### D2 — W_max is a live sensitivity parameter, not a frozen constant
Start at **60 min** for solver speed. The earlier 30↔60 flip-flopping across `[R]`/`[P]`/`[B]` vs
`[C]` was not a principled choice — it was picked per-run to make the Benders loop converge faster.
Going forward: report results *with the W_max used stated on every table*, and run the sweep
`W_max ∈ {15, 30, 45, 60}` once the decomposition is trustworthy (§C3 of the framework doc). W_max
will eventually become an input to the DCM (Task T.5.4 Challenge 1), not a free solver knob.

### D3 — Unmet-demand penalty p is a live sensitivity parameter
"Quanto vale rejeitar um passageiro?" is an open modelling question, not a constant to freeze.
Same treatment as W_max: report the p used on every table; sweep `p ∈ {30, 50, 100}` per §C3.

### D4 — ε (departure regularisation) = 0.01, unit-scaled to the waiting objective
Waiting time is measured in minutes (or slots, see D8). At ε = 0.01, **100 extra trips equal one
unit of waiting-time reduction** — i.e. ε only breaks ties between equal-waiting solutions and
never competes with the primary objective. Resolves audit N6: the value is not "small", it is 0.01,
with this stated rationale. Confirm this is the *effective* ε seen by the solver, since
`concurrency_penalty` (a separate, structural term — see D9) is a different knob and must not be
conflated with ε.

### D5 — The dominance threshold (audit N1) needs a load-dependent restatement
Original framing ("wait w costs w, rejecting costs p, so W_eff = min(W_max,p)") is too coarse. The
real trade-off: delaying a departure by w minutes to pick up one more passenger delays **every
passenger already assigned to that departure** by w minutes each. If k passengers are already
aboard, the true marginal cost of the delay is k·w, not w. In the slot-aggregate LP this is *already
captured correctly* by construction — `x[t,τ]` in the objective term `(τ−t)·x[t,τ]` aggregates every
passenger assigned to that arc, so the cost already scales with load. What D5 changes is the
**interpretation of W_eff for hand-reasoning and reporting**: there is no single load-independent
threshold; it depends on how full the departure already is. State this explicitly in the report
instead of the single-number `W_eff = min(W_max, p)` claim, and keep the (p, W_max) sweep (D2/D3) as
the empirical way to explore the trade-off rather than a closed-form threshold.

### D6 — Horizon: 10 h now, extensible to 24 h once the decomposition scales
Matches `[C]`. The slot-length buffer at the horizon boundary (why `T_minutes` in the current config
runs longer than the nominal service window) is intentional — every horizon carries one slot of
margin so early/late arrivals within a boundary slot are not artificially cut off. As slot
resolution is refined (30 → 15 → ... → 1 min), this buffer shrinks proportionally and the
mismatch disappears. Do not "fix" T_minutes without accounting for this.

### D7 — Waiting-time measurement ambiguity, resolved as a modelling note (not a bug)
Passengers arriving anywhere within slot t (e.g. 07:01 or 07:29 inside a 07:00–07:30 slot) are all
credited the same wait `(τ−t)` to departure τ. This can over- or under-state any individual
passenger's real wait by up to one slot width, symmetric in both directions — it is the standard
approximation of slot-aggregated demand, not a defect. It also explains why aggregate average-wait
figures can look inconsistent against a naive per-passenger accounting. Confirmed against the code:
`solve_subproblem`'s objective uses `wait_cost(t,τ) = max(0, τ−t)`, i.e. **slot units**, not minutes —
see D8. Once the exact-arrival-time experiment/simulation layer exists (post-deployment, real
request timestamps), *that* layer computes true per-passenger minute waits; the solver itself always
works in slots. Document both conventions side by side; do not treat the solver's slot-based
approximation as something to "correct" by converting to minutes inside the LP.

### D8 — Units inside the solver are slots throughout; minutes only in the simulation/reporting layer
Resolves the earlier concern about `(τ−t)` being ambiguously in slots vs minutes. Confirmed in code
(`solve_subproblem`, `solve_mw_dual`): `wait_cost = max(0, τ−t)`, no `× slot_res` multiplier — `p`,
`ε`, and the waiting term are all in the *same* slot-count unit. This is fully consistent, provided
`p` and `W_max` are interpreted as "slot units" and "minutes converted to slots via
`ceil(W_max_minutes / slot_res)`" respectively (which is exactly what `evaluate()` does). Real-minute
conversion (`avg_wait_min = wait_cost_slots × slot_res / pax_served`) happens only in the reporting
layer (`solver.py`, `app.py`). **`BENDERS_SPEC.md` §2.5's `(τ−t)·delta` correction from the earlier
audit round is withdrawn** — the code was already internally consistent in slot units; converting to
minutes inside the LP would have been the actual bug.

### D9 — The elastic (minute-level) subproblem relaxation is deprecated from the default path
`solve_refined_lp_relaxation_cut` / `solve_refined_subproblem` relax each vehicle's slot-quantized
departure to a continuous minute-level choice `h ∈ [lb(τ), nominal(τ)]` within its own slot, to
generate tighter cuts. **Decision: not used going forward.** Rationale (author's own assessment,
confirmed by code trace): the master's own state transitions (battery, location, `inTrip`) are
computed in whole slots, with no channel for the minute-level choice `h` to feed back into them. The
relaxation therefore "rewards" intra-slot flexibility the master can never actually realise — the
resulting cuts remain mathematically valid lower bounds (the relaxation only adds freedom, never
removes it) but are structurally loose/optimistic relative to what the rigid master can deliver, and
are a plausible contributor to slow convergence. It was an exploratory branch added under deadline
pressure and is not part of the intended design.
- **Isolation mechanism (confirmed by code trace):** `ProblemSubproblem.evaluate()` branches on
  `temporal_refinement = bool(params.get("enable_temporal_refinement", True))`. When `False`, it
  calls `solve_subproblem(...)` — the simple, slot-only, already-working model — instead of
  `solve_refined_subproblem(...)`.
- **This flag is *not* exposed in the YAML schema.** `configs/default.yaml`'s `subproblem:` section
  is validated against a fixed allow-list in `config.py` (`_check_unknown_keys`); adding
  `enable_temporal_refinement` there today raises `ValueError: Unknown key(s) in subproblem`.
- **Immediate action:** flip the Python-level default in `subproblem_impl.py::evaluate()` from
  `True` to `False`. Zero schema risk, one line.
- **Follow-up (optional, do before anyone revisits the elastic branch):** either delete the ≈800
  lines exclusive to the elastic path (`solve_refined_lp_relaxation_cut`, `solve_refined_subproblem`,
  `_proxy_cut_from_nominal_lp`, `class RefinedServiceEvent`, the minute/slot conversion helpers used
  only there), or move them to a clearly labelled `experimental/` module and add
  `enable_temporal_refinement` to the schema as an explicit, documented, off-by-default research
  flag — so a future reader knows it was a deliberate choice, not an oversight.

### D10 — C1 (silent q-aggregation of cut coefficients) is resolved once D9 is applied, not open
Traced end to end. `solve_subproblem`'s duals `pi_OUT[τ]`, `pi_RET[τ]` are aggregated **per slot
only** (summed over layers `k`, never indexed by vehicle `q`). Whichever branch consumes them
(`mw_enabled` → `solve_mw_dual`, or the plain `use_dual` branch) explicitly **broadcasts the same
per-τ value to every `(q, τ)`** before building `coeff_yOUT`/`coeff_yRET`. Master's
`aggregate_cuts_by_tau: true` (keep-first-`q`) is therefore safe on this path, because the values
are already identical across `q` by construction — there is nothing being silently dropped.
The original C1 finding is real, but was specific to `solve_refined_lp_relaxation_cut`'s
`Act_OUT[q,τ]`/`Act_RET[q,τ]` constraints, which genuinely produce **per-vehicle** duals (no
structural guarantee of equality across `q`). Once D9 removes that path from production, C1 no
longer applies.
- **Hardening kept anyway:** add an assertion in `master_impl.py`'s cut-aggregation step that
  coefficients are equal across `q` for a fixed `t` before collapsing them, so if the elastic path
  (or any future per-vehicle formulation) is ever reconnected, the code fails loudly instead of
  silently producing an invalid cut.

### D11 — θ split (per-scenario vs directional) is an empirical choice, not resolved by reading code
"Vamos para o que é mais simples, rápido e eficiente" — agreed, but this can only be settled by
running both configurations on the same instance and comparing iteration count / wall time /
final gap, not by further static analysis. Added to the roadmap as an A/B experiment, not a code
defect.

### D12 — Two independent core-point implementations exist; only one is wired in
`benders/core.py::CorePoint` (imported in `solver.py`, never instantiated as far as traced) is very
likely dead code. The actual, active Magnanti–Wong core point lives inline in
`BendersSolver.run()` (`self._mw_core_out`/`self._mw_core_ret`), an EMA of the aggregated incumbent
clipped to `[eps, Q−eps]` per slot — i.e. genuinely kept in the relative interior of the aggregate
box `[0, Q]^T`. This is a reasonable, correct implementation, better than first assessed before this
was traced. Delete `core.py::CorePoint` once confirmed unused, or wire it in if there was a reason
to prefer it — do not leave both silently coexisting.

### D13 — Runtime sign-convention verification is empirical, not further static reading
`solver.py` already contains a live runtime check —
`"[CHECK FAIL] MP total exceeds SP total at same y"` and `"[CHECK FAIL] LB exceeds UB"` — that fires
exactly when a Benders cut is invalid (θ underestimating the true recourse incorrectly, or LB/UB
crossing). Rather than continuing to hand-derive the dual sign convention from flattened,
indentation-lost source text (error-prone), the efficient path to close Gate 1's sign-convention item
is: run the (now-isolated) simple path with logging on and grep for `[CHECK FAIL]`. Absence across a
representative instance set is strong empirical evidence the sign convention is correct; presence
pinpoints exactly which iteration/candidate to hand-check.

---

## Round 2 — 2026-08-06. Correctness phase closed; focus shifts to improvement.

Context: `AUDIT_v4.md` and `BENDERS_SPEC_v4.md`. Merged as `dbc01e2`.

### D14 — The elastic subproblem path is deleted, not quarantined
D9 left the choice open between deleting the ~800 lines exclusive to the elastic path and
moving them to `experimental/`. Decision: **delete**. The block was ~1190 lines, not 800
— the estimate missed the exact-arrival plumbing, the nested helpers and the `SPParams`
fields. `subproblem_impl.py` went 3267 → 1828 lines.

Done in two separately verified steps — first the branches inside `evaluate()`, then the
orphaned definitions — with both baselines byte-identical after each. Git history is the
recovery path if the elastic idea is ever revisited.

Note for anyone reading D9 alone: the flag did more than select a subproblem model. It
also gated `mw_enabled`, `use_dual_slopes` and `cut_lb_valid`, so with it `True` the run
had no valid lower bound at all. Flipping it was a correctness fix, not a tuning choice.

### D15 — The Magnanti–Wong dual is rebuilt with the correct sign convention
`solve_mw_dual` declared `pi` non-negative. For a `<=` constraint in a minimisation the
dual is non-positive — the convention Pyomo/CPLEX return and that `dm = +S·pi` already
relied on. The wrong sign made the LP unbounded, so the function returned `None` on every
call and every cut came from the finite-difference fallback, which is not a valid lower
bound.

Two further corrections in the same function: the Pareto objective must maximise
`Σ dm·(Ȳ − y_inc)`, not `Σ dm·Ȳ` (the dropped term is not constant across the optimal
face); and the optimal face is `>= ub_base − tol`, not a float equality.

Effect: cut generation 23.90 s → 0.39 s over 10 iterations, and valid lower bounds for
the first time. See `BENDERS_SPEC_v4` §2.6 for the formulation.

### D16 — Symmetry breaking orders by total departures, never by time prefix
Prefix ordering (`cum[k][t] <= cum[k-1][t]` for every `t`) removes feasible schedules
that are not symmetric duplicates, so the master stops being a relaxation and its bound
can exceed the true optimum. Invalid **even for a homogeneous fleet**; counterexample in
`BENDERS_SPEC_v4` §2.8.

Total ordering is valid — relabel by sorting vehicles on total trips — and needs `Q-1`
rows instead of `T*(Q-1)`. AUDIT_v3's H4 framed this as redundancy plus a
heterogeneous-fleet caveat; that was too narrow, and following it as written would have
left the defect in place.

The homogeneous-fleet precondition is now enforced and raises.

### D17 — M1 is rejected on measurement
AUDIT_v3 M1 recommends stating `b[q,t] >= L·yRET[q,t]` explicitly, on the grounds that
the tighter LP relaxation is worth it because master solve time dominates. Measured: the
master phase goes 18.2 s → 49 s over 10 iterations (reproduced at 50.298 s and 50.762 s
with bit-identical results) and the bound at the same budget is **worse**, 3034.77 →
2770.97.

The constraint is sound; it does not pay for itself. Not adopted. A test asserts its
absence and a comment at the `C5` site records the numbers, so it is not re-added from
the audit text without re-measuring.

Same class of correction as M4, where the claimed performance bottleneck — per-cell
constraint construction — measures 7.6 ms against ~2.1 s per master solve, i.e. 0.04%.

### D18 — `concurrency_penalty` is kept, and must be stated on every table
It is active in the master objective via the `eOut`/`eRet` auxiliaries and is **not** part
of the originally published formulation. Magnitude on the baseline: about 1.75 of a ~4190
objective, roughly 7 penalised slots — small, but enough to select between otherwise equal
schedules.

Kept, because spreading departures rather than bunching both shuttles into one slot is
operationally defensible. The obligation is disclosure: every reported table states the
value used. The run manifest records it, so this is mechanical rather than a matter of
discipline. It is a separate knob from `ε` and must not be conflated with it (D4).

### D19 — `unused_capacity_penalty` is deleted
Schema-validated, parsed, threaded into the subproblem params dict, and read by no
model-building code. `configs/default.yaml` set it to a non-zero 0.5, so it had already
misled someone into expecting an effect.

Deleted rather than wired in: wiring it would have been inventing a model change with no
stated purpose. Results are byte-identical after removal, which is itself the proof it
was inert. Setting it is now a config error rather than a silent no-op.

### D20 — Reported lower bounds require evidence that the master is a relaxation
Two independent defects (D15, D16) each produced bounds that were not bounds, and neither
left a trace in any output. The standing check is that a reported LB never exceeds any
demonstrated feasible objective; it is asserted in the test suite against 4190.74, the
converged optimum.

Practical consequence: **every lower bound and optimality gap produced before `dbc01e2`
is void.** Upper bounds, feasible schedules and served-passenger counts were never
affected and remain usable. Any table carrying an LB or a gap must be regenerated.

### D21 — `_cand_theta` is restored, and the early exit it gates is worth nothing
`_cand_theta` had no return path for a non-`None` value, so it yielded `None` for every
key and the θ early exit — which skips cut generation when the incumbent θ already
dominates the subproblem value — never fired. AUDIT_v4 §3 item 3 listed this first among
the performance levers.

Restored (one `return float(v)`), then measured on the reference instance
`configs/mw_convergence.yaml`, byte-identical to the run behind the headline:

| | iterations | cuts | LB | UB |
|---|---|---|---|---|
| before | 42 | 38 | 4186.570873 | 4190.740015 |
| after  | 42 | 38 | 4186.570873 | 4190.740015 |

Instrumentation after the fix: `skip=None` on all 38 cut-generating iterations, i.e. the
exit still never fires. The reason is structural, not a further bug: the guard is
`θ ≥ UB(y) − ε`, and the master's θ approaches the subproblem value **from below** by
construction, so it can only hold at convergence — where the loop terminates anyway.

Kept the fix, because a helper that silently returns `None` for every input is a trap for
the next reader, and the guard is sound where it does apply. But **item 3's first lever is
closed with zero gain**: the remaining performance work is the master mipgap and the MW
core point, not this. v4's ordering of that item was wrong.

### D22 — The three stopping limits are renamed, and the master's two are wired
Three values decide when a run stops. Two of them did nothing, and the names did not
distinguish them:

| was | is | means |
|---|---|---|
| `solver.time_limit_s` | `solver.total_time_limit_s` | budget for the whole Benders loop |
| `master.solve_time_limit_s` | `master.per_iteration_time_limit_s` | ceiling on ONE master solve |
| `master.mipgap` | `master.per_iteration_mipgap` | ceiling on the master MIP gap per iteration |
| `solver.tolerance` | unchanged | relative BD gap that counts as converged |

The two master values were inert. `BendersSolver` runs a gap-tied schedule that
recomputes both every iteration from five hardcoded constants and writes them straight
into `master.params`, so the config was parsed, validated, threaded down and discarded —
the same defect class as D19. Evidence: `[SCHEDULE] master gap-tied: time_limit_s=102
mipgap=0.05` is the only distinct schedule line in all 42 iterations of the reference run
**and** in every sweep cell, including cells whose config asked for 30.

They are now the schedule's **ceilings** (`mp_gap_max`, `mp_tl_cap`) rather than being
deleted, because AUDIT_v4 §3 item 3 requires the master mipgap to be testable as a
performance lever, which was impossible from a config file. Wiring is behaviour-
preserving at the shipped values: `min(102, 120) = 102` and `gap_max 0.05 = 0.05`.

The old key names now raise at config load with the new name in the message. Silent
acceptance was the whole problem; a rename that fails loudly is the fix.

**Not behaviour-preserving, and deliberately so:** the per-iteration limit is now also
clamped by the time remaining in the run budget. Previously the budget was checked only
between iterations, so a single master solve could overshoot it without bound — measured
at 131s, 125s and >8 min against a declared 120s budget in the first sweep. The budget is
now a ceiling rather than a floor. Any "equal budget" table produced before this is
comparing cells that did not have equal budgets.

Phase guidance now carried in `configs/default.yaml`: test phase
`total_time_limit_s: 120`, `per_iteration_time_limit_s: 30`; simulation phase
`1800` and `600`. `per_iteration_mipgap` stays 0.05 in both — it is a ceiling and the
schedule tightens it to 0.001 on its own as the gap closes.

### D23 — Per-vehicle lists are [z specific, 1 shared], and both lists now obey it
A fleet of Q vehicles where z have distinct initial states and the remaining Q-z are
identical is declared as a list of length z+1 whose **last** entry is the shared value.
A homogeneous fleet is therefore a single-element list at any Q, and
`initial_battery: [150.0, 150.0, 150.0]` at Q=2 is redundant, not required.

`initial_battery` implemented this (`fill = binit[-1]`). `initial_actions`, ten lines
below it, padded with a literal `"IDL"`. A fleet declared to start charging silently
started idle from vehicle z+1 on, which feeds the energy recursion and the forced
first-action constraints. It escaped notice because every config in the repo uses `IDL`
as the shared value, making the literal coincide with the convention.

Aligned to the last-value rule and pinned by tests that assert **both lists together**,
since what failed here was the two rules diverging rather than either one alone. The
tests build with symmetry breaking off: a heterogeneous fleet is refused while it is on,
so that is the only configuration in which the convention has anything to express.

Convention documented in `BENDERS_SPEC_v4` §4 — it was undocumented, which is why it read
as a convenience fallback rather than a declared interface.

### D24 — CPLEX parameters are translated for `cplex_direct`, not skipped
`master.cplex_options` was applied only when the backend was the file-based `cplex`
plugin. Under `solver_backend: cplex_direct` — which every shipped config uses, and which
wins over `solver.master_solver` when set — every `CPXPARAM_*` key was skipped by an
explicit `continue`, to avoid an AttributeError. Third parameter in this codebase that was
schema-validated, parsed, threaded down and discarded (D19, D22).

The AttributeError was real but the fix was wrong. Pyomo's `CPLEXDirect` splits an option
key on `_` and walks `cplex.Cplex().parameters`, so `CPXPARAM_Threads` resolves
`parameters.CPXPARAM.Threads` and raises. The C name maps mechanically: drop the prefix,
lowercase, and `CPXPARAM_Preprocessing_Symmetry` becomes `preprocessing_symmetry` ->
`parameters.preprocessing.symmetry`. Now translated, and a name that cannot resolve raises
at build time.

The silence hid a second, independent mistake: **`CPXPARAM_MIP_Strategy_Symmetry` is not a
CPLEX parameter.** CPLEX calls it `preprocessing.symmetry`. Every config carried the wrong
name, so the setting would have failed even on the backend that reads these keys. Fixed in
all five configs.

Consequence for the baselines: `configs/baseline_d9.yaml` documents
`CPXPARAM_Threads: 1` with the rationale *"parallel MIP is not reproducible run to run"*.
That setting never took effect, so the frozen regression baselines were produced by
multi-threaded solves and never had the reproducibility their comment claims.

**Measured** on `configs/mw_convergence.yaml` at a 120 s budget, the two runs differing
only in `CPXPARAM_Threads` — which is itself the proof the option now reaches the solver,
since before this they would have been identical:

| threads | iters | LB | UB | served | avg wait |
|---|---|---|---|---|---|
| 1 | 23 | 4271.71 | 4496.49 | 216/300 | 40.97 min |
| auto | 23 | 3573.57 | **4190.74** | 222/300 | 39.19 min |

Neither dominates: single-threaded gives a **tighter lower bound**, auto finds a **better
feasible solution** — at 120 s it reaches the known optimum 4190.74 while single-threaded
does not. Same iteration count. No default changed on this evidence; `configs/default.yaml`
keeps `CPXPARAM_Threads: 0` and the baselines keep 1, which is now honoured and genuinely
buys reproducibility at a cost in UB quality.

### D25 — Demand outside the horizon is counted and reported
`_aggregate_requests` discarded any request mapping outside `[0, Tlen)` with a bare
`continue`. The run then printed `Pax served: 173/224` — a denominator that had already
lost 21% of demand, understating unmet demand, which is the metric weighted by `p` and the
headline number of the study.

Three drop sites, not one: the list-of-dicts path, the matrix path, and the direct-array
path, which truncates with `[:Tlen]`. All three now count what they discard, split by
cause (`after_horizon`, `negative_time`, `array_tail`), and emit a `[DEMAND]` warning
naming the horizon and the latest request time.

Warn rather than raise: `setups/demo_cont_demand.yaml` legitimately exceeds a 660-minute
horizon and raising would break it. Quantified — 284 requests, 224 counted at 660 min,
**60 discarded**; at the 24 h horizon of D6 all 284 fit, so extending the horizon resolves
this instance rather than merely moving the trap.

`setups/base.yaml` tops out at minute 598 and fits, and the structural sweep reports
`demand = 300` in every cell, so **no existing result is affected**. This was a latent
trap, and it is now loud.

### D26 — The determinism rules were already written down; two were silently violated
`configs/baseline_d9.yaml` has carried a block headed **"Determinism requirements (do not
'tune' these away)"** since before this audit. It lists four conditions: `CPXPARAM_Threads: 1`
because parallel MIP is not reproducible; a per-iteration time limit high enough that the
master **always terminates on mipgap, never on wall clock**, because "time-limited
termination depends on machine load"; `max_iterations` small and fixed; and a fixed seed.

The rules were right. Two of the four were not in force:

| requirement | status |
|---|---|
| `CPXPARAM_Threads: 1` | **violated for the repo's whole history** — the option was dropped before reaching CPLEX (D24) |
| master terminates on mipgap, not the clock | **violated by D22's remaining-budget clamp** |
| `max_iterations` fixed | held |
| seed fixed | held |

So this decision records a regression against a documented rule, not a discovery. What
follows is the evidence, and what it costs.

Task #7 set out to A/B `master.per_iteration_time_limit_s` at 15/30/60/102 s under a
120 s budget with `CPXPARAM_Threads: 1` and seed 42. It produced a clean monotone table —
and the table is worthless.

Two configs that are *functionally identical* (`per_iteration_time_limit_s` 120 vs 102,
both non-binding because the schedule computes `2 + 5/0.05 = 102`) gave 23 iterations at
LB 4271.71 and 8 iterations at LB 2274.24. Running **the same config twice** gave:

| run | iterations | LB | UB |
|---|---|---|---|
| a | 8 | 2251.15 | 4877.49 |
| b | 10 | 3487.10 | 4687.49 |

The spread *within* one config (2251 → 3487) is as large as the spread *across* the four
configs in the A/B (2274 → 3677). **The A/B measured noise.** Its apparent conclusion —
that a tighter per-iteration cap is better — is withdrawn; it is not supported.

Cause: the remaining-budget clamp added in D22. `mp_tl = min(mp_tl, total_time_limit_s -
elapsed)` makes the per-iteration ceiling a function of the wall clock. A solve that runs
slightly long because of machine load shrinks the remaining budget, which truncates the
next master solve earlier, which returns a different incumbent and bound, which changes
every iteration after it. Load jitter is amplified into result differences. Confirmed by
the schedule traces: the same nominal configuration emitted `102, 92, 75, 53, 50, 40` in
one run and `102, 97, 59, 26` in another.

This is not an argument for reverting the clamp. Without it the budget was a floor, not a
ceiling, and cells in a "fixed budget" table silently ran 131 s, 145 s and >8 min. The
tension is real and prior to either choice: **truncating a MIP by the clock makes the
result machine-dependent.** `CPXPARAM_Threads: 1` removes the nondeterminism of parallel
MIP (D24); it cannot remove the nondeterminism of the clock.

Standing rule, in two parts:

- **Experiments** — A/B comparisons and sweeps — budget by `solver.max_iterations`, with
  `solver.total_time_limit_s` set generously as a safety net that must not bind. Equal
  iteration count is a reproducible basis for comparison; equal seconds is not.
- **Simulation** — producing a deliverable number — budget by time, and state in the table
  that the run is time-truncated and therefore not bit-reproducible. A converged run
  (`status = OPTIMAL`) is exempt: it stopped on the gap, not on the clock.

Consequence for `docs/sweep/README.md`, which used a 120 s budget: the order
Q=3 < Q=4 < Q=5 by UB is **not** established and must not be quoted as a ranking. The
order-of-magnitude findings survive, because they are far larger than the spread measured
here: Q=1 is not a viable service, Q=2 is capacity-starved, and Q=3 is the largest single
step. Each UB remains a valid upper bound regardless — it is an exhibited feasible
schedule, and no amount of timing jitter makes an exhibited schedule cost less.

**Task #7 is closed by rule rather than by measurement.** The question it asked — is a
tighter `per_iteration_time_limit_s` better? — is not answerable by a four-row table,
because a cap that binds is nondeterministic by construction. Measured across three runs
of one config at a binding 15 s cap: LB 2333.29, 2153.79, 2175.87. The same config at a
non-binding 102 s cap reproduced exactly: 2422.5195186024557 twice.

So the cap is not a tuning knob at all. It must stay generous enough never to bind, which
is what `baseline_d9.yaml` already required. The remaining lever from AUDIT_v4 §3 item 3
is `per_iteration_mipgap`, which **is** adjustable without losing reproducibility, because
terminating on the gap is deterministic. That is the A/B worth running.

The run now reports what it did: `BendersRunResult.clock_truncated_master_solves` counts
master solves that stopped on `maxTimeLimit`, and a non-zero count prints a
`NOT REPRODUCIBLE` line naming the fix. Pinned by a test asserting the soundness fixture
terminates on the gap — without which every bound that suite asserts would be a sample
rather than a measurement.

### D27 — D9's evidence holds; my own change had broken the method that produced it
Task #9 opened as evidence debt: the D9 baselines proved the elastic-subproblem deletion
changed nothing by diffing before/after logs, and one of their four determinism
requirements (`CPXPARAM_Threads: 1`) was never in force (D24). Checked rather than assumed.

**The D9 evidence is sound.** Normalising timestamps, temp-file names and durations,
`baseline_d9_before.log` and `baseline_d9_after.log` are identical in every remaining
line — the sole difference is `Total solve time`. Same for the multi-scenario pair. The
archived logs are 30173 bytes each, before and after.

**Correction to what I claimed in D26.** I wrote that the baselines "never had the
reproducibility their comment claims". Too strong. CPLEX's default parallel mode is
deterministic for a fixed thread count, so those runs reproduced run-to-run on one
machine; pinning threads matters for comparability across machines and thread counts, not
for repeatability on the same one. The comment's rationale is directionally right and its
wording is stronger than the mechanism warrants. What actually protected D9 was
requirement 2 — the archived runs used `time_limit_s: 3600`, so no master solve ever
stopped on the clock.

**Two real defects surfaced, both recent, both mine or adjacent:**

1. *The schedule log line was made non-reproducible.* D22's clamp fed the wall clock into
   `mp_tl`, which is printed on every iteration, so two runs with **identical results**
   produced differing logs (`time_limit_s=94` vs `85`). That silently destroys the D9
   diff protocol: anyone re-running it would see spurious differences and have no way to
   tell them from real ones. Fixed by logging the gap-derived value (deterministic) and
   emitting a separate `[SCHEDULE] budget clamp` line only when the clamp actually bites —
   where its presence is itself the signal that the run went machine-dependent.

2. *`baseline_d9.yaml` had `total_time_limit_s` reduced from 3600 to 120*, which made the
   clamp bind on a fixture whose own header forbids clock-termination. Restored to 3600
   with a comment saying it must not bind. This does not conflict with the 120 s working
   rule: the baseline stops at 10 fixed iterations in ~40-60 s, so the limit is a safety
   net, not a budget.

Verified after both fixes: two consecutive baseline runs give
`LB 3576.684478873909, UB 4687.490045, 10 iterations`, no `NOT REPRODUCIBLE` line, and a
normalised log diff that is identical in every non-timing line. The regression method
works again.

Lesson worth keeping: a change can be correct in its effect on results and still destroy
the evidence that other work depends on. D22's clamp did not change a single bound in the
baseline — it changed what the log said, which is where the proof lived.

### D28 — Master mipgap stays at 0.05; iterations dominate it by an order of magnitude
The last open lever from AUDIT_v4 §3 item 3, and the first tuning question in this project
that could be asked on a reproducible basis (D26): unlike the time limit, `mipgap` can be
varied without losing determinism, because terminating on the gap does not read the clock.

Base `configs/baseline_d9.yaml` (the determinism fixture), 6 fixed iterations,
`per_iteration_time_limit_s: 300` so it never binds, threads 1, seed 42. Every cell
reported `clock_truncated_master_solves = 0`, so all three are reproducible and mutually
comparable:

| `per_iteration_mipgap` | master time | LB | UB |
|---|---|---|---|
| 0.05 | 22.57 s | 2003.33 | 5283.74 |
| 0.01 | 29.90 s | 2087.82 | 5283.74 |
| 0.001 | 44.45 s | 2106.65 | 5283.74 |

A tighter gap does give a better lower bound, monotonically — and the effect is **small**:
0.05 → 0.001 buys **+5.2%** of LB for roughly double the master time. The UB is untouched.

Spending the same effort on iterations instead, at the loose gap:

| iterations at mipgap 0.05 | LB | UB |
|---|---|---|
| 6 | 2003.33 | 5283.74 |
| 8 | 2422.52 | 4877.49 |
| 11 | 3576.68 | 4687.49 |

**+78.5%** of LB, and the UB improves too. Each iteration adds a cut that tightens the
master for every iteration after it; a deeper solve only refines the current node.

**The per-second trade is not measurable on this machine and is not claimed.** Timing is
unreliable at the precision required: the same deterministic master solves took 43.199 s
in one run and 20.416 s in another (2.12x) at iteration 8, and an 8-iteration run reported
110.84 s of master time against 73.90 s for an 11-iteration run whose first 8 iterations
are bit-identical work. Only the bounds reproduce; the clock does not. The conclusion does
not rest on the timings: 5.2% against 78.5% is not a margin that 2x timing noise reverses.

Default unchanged at 0.05. The knob is now real (D22) and documented as a ceiling that the
schedule tightens on its own, so a future instance where the master bound is the binding
constraint can revisit it — with `clock_truncated_master_solves = 0` checked per cell, or
the comparison means nothing.

### D29 — Recourse lower bound on theta, per prefix, on by default
The master had no link between theta and installed capacity except the Benders cuts,
which arrive one at a time. Measured consequence at T=44: the master's bound sat at
**0.22 for five consecutive iterations** while it "believed" it could serve 300 requests
with almost no departures, and the LP relaxation was 20x loose (`best_bound=52.96` against
`incumbent=1004`, 3866 nodes). The master was not slow because it was large — empty, it
solves in 0.07 s — it was slow because it was uninformed.

**The inequality.** Demand arriving in slot `t` can only be served by a departure in
`[t+1, t+W_slots]` (spec §2.5), so demand accumulated to `j` only reaches capacity
installed to `j+W_slots`:

```
served_d([0,j])  <=  S * sum_{tau <= j+W} sum_q y_d[q,tau]
```

Unserved from a subset bounds total unserved, and waiting cost is non-negative, so for
every `j` and each direction:

```
theta_d  >=  p * ( R_d_cum[j] - S * Y_d_cum[j+W] )
```

Slack automatically when the right-hand side is negative, since `theta >= 0`.

**Per prefix, not aggregate.** The aggregate version is the `j = T-1` member of this
family and lets a shuttle departing at slot 40 pay for morning demand. The prefix form is
what encodes that it cannot, and it strictly dominates. A test walks the `j=1` row and
asserts no late `yOUT[q,tau]` appears in it.

**Per direction.** The master uses `theta_out`/`theta_ret`, not `theta` —
`disaggregate_theta_by_direction` defaults True and is absent from the schema, the fifth
phantom parameter after D19/D22/D24. Bounding each direction separately is strictly
tighter than bounding their sum.

**Measured**, 8 iterations, threads 1, seed fixed, per-iteration limit 300 s:

| case | LB off | LB on | Δ LB | master off | master on | reproducible |
|---|---|---|---|---|---|---|
| T22 Q2 | 2422.52 | 3046.44 | **+25.8%** | 73.8 s | 46.0 s | yes |
| T22 Q3 | 106.25 | 201.43 | **+89.6%** | 116.9 s | 37.1 s | yes |
| T44 Q2 | 465.37 | 1854.08 | +298% | 372.4 s | 541.9 s | **no** |

On both reproducible cells it wins on **both** axes — better bound and less time. That is
what separates it from M1 (D-record: valid, 2.7x slower, worse bound, rejected). The
mechanism is visible rather than inferred: the master's first bound goes from 0.0 to
1569.09, and the sequence `0.22, 0.22, 0.22, 0.228, 87.9` becomes
`1569, 1617, 1660, 1688, 1752`.

**T=44 is not claimed.** Both cells printed `NOT REPRODUCIBLE`: the schedule computes
`2 + 5/0.05 = 102 s` per master solve regardless of the 300 s ceiling, and T=44 solves hit
it. Those two numbers are one draw (D26). Closing that needs a larger
`per_iteration_mipgap`, which is a separate experiment.

Note the node counts **rise** (0 -> 2588 on the first iteration). The gain is not that the
master became easy; it is that it started working on information instead of on nothing.

**Validity is not usefulness, and validity was checked.** A wrong derivation cuts off the
optimum, and the only thing that catches it is `LB <= 4190.74` — the same check that
exposed prefix-ordering symmetry breaking (C4). The soundness fixture now runs with the
inequality on and asserts exactly that.

**Frozen fixtures pin it off.** `baseline_d9.yaml` and `baseline_d9_multi.yaml` set
`recourse_lower_bound: false` explicitly rather than inheriting, because they exist to be
diffed against archived logs and must not move when a default changes. Verified: the
baseline log diff is identical after the flip, LB `3576.684478873909` unchanged.

**Not applied to multi-scenario runs**: the bound would need the expectation over
scenarios, which this form does not express. `_recourse_bound_data` returns None there.

### D30 — The subproblem was not a Benders subproblem; the layers are gone
Every cut this project ever produced was invalid. Found by using an independent
monolithic MILP of the same model as an oracle.

**Evidence.** Taking the MILP's optimal schedule, fixing it in the Benders master
(feasible — the master admits it) and pricing it with this codebase's own subproblem
gives 4183.24, matching the MILP to the cent at three scales (Q=2/T=22 4183.24,
Q=3/T=44 651.36, four scenarios 1674.11). So the recourse model is right. But minimising
theta over the cut set at that schedule forced:

| generator | forced theta | true recourse | excess |
|---|---|---|---|
| MW, directional theta | 6893.00 | 4183.00 | **+2710** |
| MW, single theta | 5290.00 | 4183.00 | +1107 |
| plain dual, directional | 6087.00 | 4183.00 | +1904 |

All three exclude the optimum, so it is not Magnanti–Wong and not the directional split.
Under clean conditions the reported LB reached **4492.23**, 7.4% above an optimum of
4183.24 — and it climbed with iterations, the signature of invalid cuts accumulating.

**Root cause.** `OutLayers`/`RetLayers` were indexed by `K_d[tau]`, the number of vehicles
departing at tau, which comes from the master's `y`. That made `y` change the subproblem's
**variable and constraint sets**, not just its right-hand side. Benders duality requires
`Q(y) = min{c'x : Ax >= b − By}`; when `A` itself moves with `y`, the dual of one instance
is not a subgradient of the recourse across `y`, and no cut generator can be valid on top
of it.

Mechanism: `pi[tau]` summed the K layer duals and `dm = S*pi`, giving slopes about K times
too steep, which over-estimates whenever the evaluated `y` has less capacity than the
incumbent.

**Fix.** One capacity row per `(d, tau)` with right-hand side `S * sum_q y_d[q,tau]`. The
layers were redundant in capacity — K layers of `min(S, S*K) = S` total `K*S`, exactly the
aggregated right-hand side, and `K = 0` gives no arcs either way — so the recourse is
unchanged, confirmed by the three exact matches above being re-measured after the change.

After the fix, the same diagnostic gives forced theta **4165.00 against 4183.00**: the cuts
now under-estimate, which is what valid cuts do. The reference run's LB drops from 4492.23
to 3817.84, below the optimum.

**What this costs.** `fill_first_epsilon` existed only to order vehicles within a layer, so
it is deleted (D19 precedent: an inert knob is a trap). The per-shuttle report now splits
the aggregated flow into blocks of S, which is the arrangement that epsilon produced anyway.
Bounds get weaker and correct instead of stronger and wrong, so every LB in this project's
history is void — including the "converged optimum 4190.74" headline and the D29 gains,
which are all lower-bound claims.

**Why nothing caught it.** `KNOWN_FEASIBLE_UB` was 4190.74, taken from a Benders run of this
same code: circular, and unable to detect an inflation smaller than 7.5. The cut-tightness
check validated only the combined cut, never the directional halves actually added to the
master. `cut_valid_lower_bound` reported True throughout.

The one guard that did work was the strong-duality check, which caught an inconsistency in
this very fix within seconds (`primal=5105.0 dual=5825.0`, from a stale `min(S, C)` in the
dual objective). It works because it compares two independently computed quantities. That is
the property to copy: **a check that derives its expectation from the thing it is checking
cannot fail.**

Standing consequence: `KNOWN_FEASIBLE_UB` is now 4183.24, sourced externally, and a test
asserts the subproblem prices the monolith's optimum exactly. The MILP is the oracle.

### D31 — The D9 baselines are re-recorded; the archived pair is obsolete
D30 changed the subproblem, so the logs the D9 before/after diff was validated against no
longer describe this model. Re-recorded and re-verified:

| fixture | before D30 | after D30 |
|---|---|---|
| `baseline_d9` | LB 3576.684479 / UB 4687.490045 | **LB 2891.086867 / UB 4962.990000** |
| `baseline_d9_multi` | — | **LB 879.511303 / UB 3382.240000** |

Both reproduce: two consecutive runs give a normalised log diff that is identical in every
non-timing line, and neither prints `NOT REPRODUCIBLE`. The new references are
`docs/baseline_d9_D30.log` and `docs/baseline_d9_multi_D30.log`.

The bounds moved a long way and in the expected direction: **weaker and correct** instead
of stronger and wrong. The old LB of 3576.68 was below the optimum and therefore not
visibly invalid at 10 iterations — the defect only became visible at 45, where the same
fixture family reached 4492.

`docs/baseline_d9_before.log` / `_after.log` are kept as the record of what D9 proved about
the elastic-path deletion, which still stands: that comparison was internally consistent
and concerned a different question. They are no longer a reference for current runs.

### D32 — `docs/` and `configs/` are versioned, effective at the start of the simulation phase
Both were in `.gitignore`, so a clone received the code without the reasoning or the
fixtures. That was tolerable while the repository was a single working copy. It stopped
being tolerable at `a2d9e97`, which is marked breaking and voids **every historical lower
bound** (D30): a clone got the breaking change and none of the argument for it, and could
not re-run the baselines that demonstrate it, because those configs were untracked too.

Now tracked: `docs/` (`docs_decisions.md`, `AUDIT_v3`/`v4`, `BENDERS_SPEC_v3`/`v4`,
`HANDLER_CENSUS.md`, `docs/sweep/`, and the reference run logs those documents cite by
name) and `configs/` (`default.yaml`, the three baseline fixtures, `configs/sweep/`).
115 files, ~3.6 MB, all text.

Still ignored, deliberately: `Report/` (42 MB of source PDFs and generated output, not
inputs to any run), `aux_py/`, `setups/*` except the one example, `manifests/`
(regenerated per run — D26 records provenance in the manifest, not in git), and
`cplex.log`.

The whole-directory rule is intentional. A per-file rule would need a judgement call on
every new log, and the logs are not incidental: D31 names `baseline_d9_D30.log` and
`baseline_d9_multi_D30.log` as the current references, and `docs/sweep/*.log` is the
evidence behind the sweep table. Cheap to keep, and the alternative is a citation to a
file nobody else has.

`.gitattributes` added at the same time with `*.log -text`. Reference logs are compared
line-by-line against fresh runs; letting git rewrite their line endings on checkout would
make that comparison platform-dependent, which is the same class of defect as a check
that derives its expectation from the thing it is checking (D30).

---

## Fase 1 — 2026-08-06. Is the master tractable? Results in `docs/phase1/README.md`.

### D33 — The recourse anchor extends to multi-scenario runs, and is inert at Q=3
D29 stated the anchor per prefix for a single demand vector and `_recourse_bound_data`
returned `None` for multi-scenario runs, so the four-scenario master carried no anchor at
all. The inequality holds per scenario, so weighting the scenario rows by `w_s` and
summing gives one row on the recourse term the master actually minimises:

```
sum_s w_s theta_s  >=  p * ( sum_s w_s R_s_cum[j]  -  S * Y_cum[j+W] )
```

`Y_cum` is common to every scenario because the first stage is here-and-now, so it
survives the sum unchanged. Weights are normalised before use: the averaged-cut path
normalises, and an unnormalised anchor would sit on a different scale from the cuts.

The form is valid under all three scenario configurations. Under multi-cut with a single
theta the master's theta is pushed above **every** scenario rather than their mean, and
mean <= max, so the mean form stays valid there — weaker, not wrong. Under
`theta_per_scenario` the same row is stated over `sum_s w_s theta_s`, which is the
objective's recourse term.

**Measured, and it is inert at the Fase 1 test point.** Empty master, no cuts:

| fleet | empty master with the anchor |
|---|---|
| Q=1 | 7737.62 |
| Q=2 | 1662.70 |
| **Q=3** | **0.24** |

D29's headline "first bound goes from 0.0 to 1569.09" was a **Q=2** measurement. At Q=3
over 44 slots the fleet installs far more capacity than `R_cum[j]` ever demands, so every
prefix row is slack. The inequality is correct and bounds nothing here. In the A/B it
moved the LB from 0.299 to 0.314, which is inside the baseline's own draw-to-draw spread.

Kept: it is free, it is valid, and it is worth a large amount at Q<=2 where D29 measured
it. But the multi-scenario extension is **not** what rescues the multi-scenario bound, and
the conservative decision to leave it off was not what was costing the 0.32.

Consequence for D29's record: its gains are fleet-size-dependent and were never claimed
beyond Q=2. Any future table quoting the anchor must state Q.

### D34 — LP phase in the master: works as designed, does not help here
The first iterations spend a full 30 s of branch and bound producing one cut from a cut
set that is nearly empty. `master.lp_phase` solves the master without integrality while
that is true, and switches to MIP when the LP objective stalls
(`lp_phase_stall_iters` consecutive iterations below `lp_phase_min_rel_improve`), when
`lp_phase_max_iters` is reached, or when an iteration generates no cut.

**Why the cuts are valid.** `Q(y)` is the value function of an LP parameterised by its
right-hand side, hence convex in `y`. A dual at any `y` — fractional included — supports it
everywhere. This is the same convexity D30 had to restore by de-layering capacity; before
that fix `A` itself moved with `y` and no such argument existed.

**What the phase must not do is claim an upper bound.** A fractional schedule is not
exhibitable, so its recourse cost bounds nothing. While the phase is active the loop
leaves `best_ub`, the incumbent, the reported passenger counts and the candidate-UB column
untouched, and `_collect_candidate`'s binary guard stays in force for every MIP solve,
which is where a fractional `y` would mean the schedule about to be priced as an upper
bound is not one anybody can operate. If a run ends inside the LP phase it says so
explicitly rather than printing a fractional schedule as a result.

**Measured.** Master seconds per iteration, `lp_on` cell:

```
0.9 0.3 0.4 0.4 0.4 0.6 0.6 0.5 0.3 0.3 | 30.2 30.3 30.3 30.4 ...
```

LP iterations cost 0.3–0.9 s against 30 s for MIP iterations, 50–100x, and the cell fits
19 iterations into the 300 s budget where the baseline fits 14. The mechanism works. The
LB went 0.299 -> 0.350, which is noise.

Kept and left **off by default**: it is a real reduction in cost per cut, so it is worth
having when cuts are the binding constraint. At the Fase 1 test point they are not.

### D35 — Multi-cut on a shared theta is refused: LB and UB must measure the same problem
`subproblem.multi_cuts_by_scenario: true` with `master.theta_per_scenario: false` adds one
cut per scenario against a single theta, forcing `theta >= Q_s(y)` for every `s`, i.e.
`theta >= max_s Q_s(y)`. The reported upper bound is the weighted **mean** of the same
quantities (`ub_aggregation: mean`). Since `max >= mean`, the master's optimum can exceed
the true optimum of the problem the UB measures: a bound that is not a bound, the D15/D16
failure mode with a different cause.

Reachable from config, and nothing detected it — the standing `LB <= known feasible UB`
check passes because the LB is nowhere near either. Now refused at config load, naming
both consistent alternatives: `theta_per_scenario: true` (objective
`sum_s w_s theta_s`, one cut per theta) or `multi_cuts_by_scenario: false` (weighted
average cut on one theta).

No shipped config used the combination — `default.yaml` and `baseline_d9_multi.yaml` both
average — so no recorded result is affected. This is a trap closed, not a result revised.

### D36 — Asking for the anchor without the data to build it now raises
`recourse_lower_bound: true` with no usable demand source used to return `None` and build
nothing: the run looked configured and behaved as if it were not. That is D19's inert knob
and D22's discarded config, for the third time. It now raises at parameter preparation
with the list of things it needs. This is what made D33's gap visible at all — the
multi-scenario `None` was a deliberate early return that had outlived its reason and left
no trace in any log.

### D37 — Fase 1 verdict: the master is not tractable at Q=3 / T=44 / 4 scenarios
The criterion, fixed before the runs: LB >= 785 at Q=3, T=44, four scenarios, 300 s, i.e.
50% of the monolithic MILP's LB of 1569.44. Best achieved across the four cells:
**0.350915**. Short by a factor of about 2200.

| cell | anchor | LP phase | LB | UB | iters |
|---|---|---|---|---|---|
| base | off | off | 0.299040 | 2685.86 | 14 |
| anchor | on | off | 0.313787 | 2267.36 | 11 |
| LP | off | on | 0.350393 | 2310.36 | 19 |
| both | on | on | 0.350915 | 2016.11 | 14 |

The spread between cells is the size of the baseline's own draw-to-draw spread on
identical configuration (0.299 / 0.310 / 0.312), so none of it is an effect.

**The obstacle is not the cut set.** Every MIP master solve pins its 30 s ceiling and
terminates `maxTimeLimit` at an internal gap of **0.999** — best bound 0.35 against its own
incumbent of 636. The reported Benders LB is that best bound.

**And it is not the time budget.** Re-run with 3.4x the per-solve time:

| per-solve | nodes | master best bound | gap |
|---|---|---|---|
| 30 s | ~3 960 | 0.350393 | 0.9994 |
| 102 s | ~30 700 | 0.354730 | 0.9990 |

7.7x the nodes for 1.2% of bound. The master's bound sits at zero because nothing forces
`theta` up: the D33 anchor is slack at Q=3, and 14–19 cuts cannot cover 264 binaries.
Branch and bound has nothing to lift.

**Standing consequence.** Benders as built does not produce a usable lower bound at this
size, and no amount of the two levers built in this round changes that. Upper bounds are
unaffected — each is an exhibited feasible schedule, and the best of them, 2016.11, is the
one real output of this phase. **No optimality gap may be quoted at Q>=3.**

This is not a verdict on Q<=2, where the anchor is worth 1662–7737 on the empty master and
D29 measured gains on both axes. The failure is specific to a fleet large enough to make
the capacity anchor slack. If the decomposition is to be revived at Q=3 it needs a valid
inequality that binds when capacity is ample — the binding difficulty there is timing and
energy, not capacity — and that is a modelling question, not a solver-tuning one.
