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

## Fase 1 — 2026-08-06. Is the master tractable? Verdict in D37; the five configs that
produced it ship as `configs/phase1/*.yaml` and the reading rules are in `README.md`. The
working write-up is not published (see the standing directive in the handout §6).

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

### D38 — The published repository carries what reproduces a result, not the transcript
D32 versioned `docs/` and `configs/` wholesale, on the argument that a whole-directory rule
needs no judgement call per file. That was right for the working repository and wrong for a
public one. The first push under it put ~80 run logs and 29 sweep configs into a repository
whose readers — a supervisor, a reviewer — need the final state, not the record of every
attempt.

The distinction D32 missed is between **evidence** and **the means to regenerate it**. Every
number quoted in `docs/` is produced by a config that ships here. `configs/phase1/*.yaml`
regenerates the Fase 1 A/B; `configs/baseline_d9*.yaml` regenerates the D31 fingerprints,
and those two converge on the gap, so they reproduce exactly rather than approximately.
Shipping the logs adds nothing a reader can check that re-running does not.

Untracked, and now ignored: `*.log` anywhere, `configs/sweep/`, `docs/sweep/`, `manifests/`,
`Report/`, `aux_py/`, `.idea/`, `p310.cmd`. Removed from the tree as superseded:
`AUDIT_v3.md`, `BENDERS_SPEC_v3.md`, `HANDLER_CENSUS.md`, `design.md` — the first two are
superseded by their v4, and `design.md` described the skeleton before the model existed.

**Added, and this was the real defect:** `setups/base.yaml`,
`base_plus100_out_noon.yaml`, `base_ret_peak_adv.yaml` and `base_vol20_pm60.yaml`. Every
config in the repository reads them and only `demo_cont_demand.yaml` was tracked, so a
clone could not run **anything** — including the baselines D31 calls the reference. The
whole-directory rule versioned 3.5 MB of logs and missed the four files without which none
of it runs.

`README.md` was rewritten at the same time. It documented `solver.time_limit_s` (renamed in
D22), `fill_first_epsilon` (deleted in D30) and `unused_capacity_penalty` (deleted in D19),
and its example set `multi_cuts_by_scenario: true` with `theta_per_scenario: true` — a
combination D35 now refuses. It duplicated `configs/default.yaml` and the duplicate went
stale, so it now points at that file rather than copying it.

Where an entry above cites a log by name — D31's `baseline_d9_D30.log`, the sweep tables —
read the citation as naming the run, not a file in the tree. The config that produced it is
tracked.

**History rewritten.** The two commits that carried the logs were replaced rather than
followed by a removal commit, so the public history does not contain them either. Safe
because they had been on the remote for minutes, with no PR and no other collaborator.

### D39 — Cut validity is a four-state fact, not a boolean, and guards no longer die quietly
Seven swallowed-exception sites were opened. Two of them could turn an uncertified run into
a certified-looking one; the rest degraded information that other checks consume.

**The type was too small.** `cut_valid_lower_bound` was read with a default of `True` in
`solver.py` and a default of `False` in `subproblem_impl.py` — same key, same dict, opposite
answers, and neither author wrong locally. `manifest.py` was the tell: it passes no default
and writes `null`, because a file can represent "unknown" and a `bool` cannot.

Tracing where the key is actually emitted settled the design. It appears **if and only if a
cut was generated**: the six returns that omit it are the infeasible paths, the theta early
exit and the debug skip, all with `cuts=[]`. So absence never meant "we forgot":

| state | meaning | policy |
|---|---|---|
| `VALID` | a cut was generated with a lower-bound guarantee | LB stands |
| `INVALID` | a cut was generated without one (MW fallback, finite differences) | drop the LB, warn |
| `NO_CUT` | nothing was added to the master this iteration | **leave the bound alone** — whatever certified it still does |
| `UNKNOWN` | a cut exists and nothing said whether it is valid | drop the LB, warn |

`CutValidity` and `classify_cut_validity` live in `benders/types.py` so every consumer asks
the same question the same way, and the fail-closed policy is one visible block in the loop
instead of a default re-litigated at each `.get()`. The old form also reached `True` when
the diagnostics dict merely failed to build.

Correction to a claim made while doing this: I first reported that six of nine returns omit
the key on ordinary feasible paths, implying the producers were negligent. That came from an
AST pass that only inspected the `return` expression, so dicts assembled into a variable
looked empty. The producers are consistent; the consumer was collapsing four cases into two.

**A crashed guard no longer looks like a passed guard.** The four `[CHECK FAIL]` bound
comparisons sat inside one `try: ... except Exception: pass`. A `TypeError` in the first
skipped the other three and the LB revert, and produced exactly what a clean run produces —
no `[CHECK FAIL]` line. `test_no_check_fail_lines` asserts that absence, so the test agreed.
Now `(TypeError, ValueError)` logs `[CHECK ERROR]`, says the absence of `[CHECK FAIL]` does
not mean the bounds were verified, and re-raises.

Also closed: `master_impl` fell back from the branch-and-bound bound to the **incumbent
objective** on a parse failure — an upper bound reported as a lower bound, 0.35 against 636
on the Fase 1 runs. It raises now. The theta readbacks in `_collect_theta_into` and
`_theta_snapshot` no longer swallow, because the first feeds the report that D-record showed
mixing two solutions and the second feeds the re-anchoring check that raises on its own.
Stats extraction keeps its reason rather than erasing the M3 provenance.

**Behaviour-preserving, and measured rather than asserted.** 72 tests pass; `baseline_d9`
reproduces `LB 2891.09 / UB 4962.99` and `baseline_d9_multi` reproduces
`LB 879.511 / UB 3382.24`, both matching D31 to the digit, and no new `[CHECK]` line appears.
The `INVALID` and `UNKNOWN` branches are guards that do not fire on a healthy run — which is
the right shape: unlike a skipped test, when they do fire they fail rather than pass.

Reviewed and deliberately kept: `_emit_manifest`'s broad catch, which logs the exception
object and exists so provenance cannot break a solve; and `solver.py`'s best-bound recovery,
which already fails closed with `[CHECK] MP best bound unavailable; LB not updated`.

The six rules these came from are recorded outside the repository as the `receitas-basicas`
skill; `scan_exception_handlers.py` there ranks handlers by what a reader would wrongly
believe if one fires, which is how the seven were selected out of 237.

---

## Semana 2 — 2026-08-10.

### D40 — D37 and D34 are refuted: the LP phase was measured with a 10-iteration cap
Fase 1 concluded that the master is not tractable at Q=3 / T=44 / 4 scenarios and that the
failure is **structural** — D37's words: *"no `N` and no `t_master` in reach produces a
usable bound at this size"*. D34 concluded that the master LP phase works as designed and
**does not help**, because cuts were said not to be the binding constraint.

Both conclusions came from `configs/phase1/lp_on.yaml`, which sets
`lp_phase_max_iters: 10` and `lp_phase_stall_iters: 3`.

**The measurement that settles it.** `configs/phase1/lp_only_150.yaml` — the same test
point, the same code, the LP phase given 150 iterations instead of 10 and the stall exit
disabled. The master LP objective *is* the Benders lower bound, and it is exact: an LP has
no branch and bound, so nothing is truncated and the number is not a best bound.

| iteration | LB | | iteration | LB |
|---:|---:|---|---:|---:|
| 1 | 0.000 | | 50 | 682.352 |
| 5 | 0.265 | | 80 | 773.996 |
| **10** | **0.311** | | **91** | **785.632** |
| 20 | 270.615 | | 100 | 788.145 |
| 30 | 437.575 | | 130 | 793.581 |
| 40 | 601.424 | | **150** | **794.625** |

Final: **`best_lb = 794.624549571966`, 150 iterations, no upper bound.**

**The death criterion of LB >= 785 is crossed at iteration 91**, in about 100 s of master
time. The bound rises by a factor of roughly 2500 between iteration 10 and iteration 150.

**Reproducible, and verified rather than asserted.** Two independent executions gave
`794.624549571966` both times, with identical trajectories point by point across all 150
iterations -- the first result in this project checked that way. An LP has no branch and
bound and never stops on the clock, so D26's incompatibility does not apply to it.

**Why it was never seen.** The curve is flat until about iteration 12 and only then climbs.
A 10-iteration cap samples the flattest point on it. The stall detector would have ended
the phase even earlier, because a flat objective is precisely what it is built to detect —
so the guard designed to save time was, on this instance, the thing that produced the wrong
answer.

**What is now void:**

- D37's structural verdict. The failure is the **cut budget the 30 s MIP ceiling allowed**,
  not the model. Phase 2 is not ruled out at Q>=3 on these grounds.
- D34's "does not help". The LP phase does help; it was capped before it could.
- The mechanism argued in the handout — that non-positive cut slopes let theta fall for
  fractional `y`, so no number of cuts could lift the LP bound — is wrong. That is the
  fourth time this round that reading the code produced a confident wrong answer that a
  cheap measurement overturned. **Measure, do not reason** (D30) applies again.

**What is NOT claimed.** This is not competitiveness. The monolith still solves the
instance to optimality in 39 s; this run spent ~200 s wall for a lower bound and produced
**no upper bound at all**, because a fractional schedule cannot be exhibited. **No
optimality gap may be quoted from it.** What changed is the diagnosis, not the standing.

**Quality.** All 122 cuts were generated by `mw`. Zero `[CHECK FAIL]`, `[CHECK ERROR]`,
`[MW FAIL]`, `[SP WARN]`; zero `INVALID` or `UNKNOWN` cut-validity states; cut tightness at
the candidate verified each iteration (last: `diff=2.96e-12`). No `NOT REPRODUCIBLE` line —
the LP phase never stops on the clock, so unlike every Fase 1 cell this run is reproducible
(D26).

**Consequence for the tried-and-rejected list.** Branch-and-cut with lazy constraints and
user cuts was implemented in this repository and removed on 2026-02-19; `config.py` still
refuses `use_lazy_cuts` and `lazy_cb_lp_solver` with a hard error and no decision record
says why. That removal predates the D30 de-layering (2026-08-06) by nearly six months, so
it was judged against a subproblem whose cuts were not valid lower bounds: **that verdict is
void on the same grounds as every pre-D30 bound.** With cuts now shown to lift the root
relaxation, generating them continuously at fractional nodes is the matching technique, and
revisiting it is warranted. A revival must fail loudly where the deleted implementation had
`# If SP fails, skip adding lazy cut`.

### D41 — Two defects found by running the LP phase past iteration 10
Both were reached only because D40's run went 150 iterations where every previous run
stopped at 10 or 19. Neither changes a reported number; both were destroying or hiding
information.

**A crash at iteration 123.** `solve_subproblem` built a diagnostic tuple
`(contrib, t, tau, k, x)` where `k` was the capacity-layer index. D30 removed the layers.
The OUT branch was updated to a literal `0`; the RET branch kept referencing `k`, which no
longer exists in that scope. `UnboundLocalError`, and the whole run dies.

The trigger is why it survived six months: the append runs only when an individual arc
contributes negatively, i.e. when the LP returns a slightly negative flow from solver
tolerance. **The diagnostic built to report a numerical anomaly was destroyed by the
anomaly it was built to report.** The field now records direction (0 = OUT, 1 = RET), which
is the useful fact once there are no layers. The underlying negative flow is real and
harmless at this scale: the total wait cost stayed positive and `[SP DIAG]` never printed.

**300 lines of `CPLEX Error 3003: Not a mixed-integer problem.`, two per iteration.**
`_extract_cplex_api_stats` calls `solution.MIP.get_best_objective()`,
`get_mip_relative_gap()` and `solution.progress.get_num_nodes_processed()` on every solve,
including LP-phase solves.

The guard was `hasattr(cpx.solution, "MIP")` — and that attribute exists whatever the
problem type, so **the guard tested something that is always true**. CPLEX's C library
prints the error before the Python exception reaches an `except Exception: pass`
immediately below. A guard that does not test what it claims, a swallowed exception, and
300 error lines in a healthy run, in one block. `_extract_cplex_api_stats` now takes
`lp_relax` and returns before the MIP-only queries; the caller already derived the bound
from the relaxation and recorded `best_bound_source = "lp_relaxation"`.

Two things worth recording about how this was found. The first diagnosis — that
`use_mip_start` was applying a MIP start to a relaxed model — was **wrong**. It was a real
defect, and fixing it took `[MIPSTART]` from ~250 occurrences to zero, but the 3003 lines
continued; the actual source was only found by isolating the solve in a probe. And
`_MIP_ONLY_CPLEX_OPTIONS` already existed in this file, with a comment describing error
3003 exactly, and it **was working** — options on an LP solve are down to
`{timelimit, threads}`. Half the problem had been fixed earlier and the other half
survived because the visible symptom never went away, so the earlier fix looked like it had
failed rather than like it was incomplete.

**The general lesson, and it is the same one as D30 and D40:** an error line that is always
present is not information. 300 of them per run made a real CPLEX error indistinguishable
from the wallpaper, and made a working fix look broken.

72 tests pass after both changes.

### D42 — Magnanti-Wong is verified to dominate, and the verification found a crash
Nothing checked that MW selects a *better* dual. The two existing tests assert it ran
(`test_magnanti_wong_succeeds`) and that the run labelled its cuts `mw`
(`test_cut_mode_is_reported_and_is_mw`). Both are provenance, and D30 is the proof that
provenance and quality are independent: MW, the plain dual and finite differences produced
invalid cuts identically, so a green MW label has already coexisted with cuts that excluded
the optimum. This matters more now than last week -- D40 makes branch-and-cut worth
revisiting, and that multiplies the number of cuts generated by this path.

**The invariant, and why it is not instance-dependent.** The subproblem LP is degenerate:
many duals are optimal, all give a cut tight at the incumbent, they differ away from it. MW
maximises the cut's value at a core point `Ybar` over the optimal face. The plain dual is
another point on that same face. Therefore

    cut_MW(Ybar) >= cut_dual(Ybar)   for every Ybar

holds **by construction**, and a violation is proof of a defect rather than a bad instance.
It is the check that would have caught the bug the generator's own comment records: an
earlier version maximised `sum(dm*Ybar)` and dropped the `-y_inc` term, which is not
constant over the optimal face, so it selected the wrong dual.

**Measured** on `tests/fixtures/soundness.yaml`, same candidate, one cut per generator,
margin = MW - dual at the core point:

| core point (Yout, Yret) | margin |
|---|---:|
| all zeros | 0.0000 |
| ret only (0, 1) | 0.0000 |
| all ones | 0.0002 |
| uniform (0.5, 0.3) | **21.0001** |
| out only (1, 0) | **30.0002** |

Never negative, and strictly positive at two points, so the assertion is not vacuous.
`test_mw_is_strictly_better_on_at_least_one_core_point` exists for exactly that: if MW
silently degraded to returning the solver's arbitrary dual, every margin would be zero and
a dominance-only test would still pass while checking nothing.

**Both generators must see the same LP, and that is asserted rather than assumed.**
`use_dual_slopes` is not only a generator switch -- it also floors `K` at 1, which changes
the subproblem that gets built (see the half-live note in the handout). Both configurations
therefore set it `True` and toggle only `use_magnanti_wong`, and the test compares the
recourse value across the pair to prove the face is shared. Comparing cuts off two
different LPs would have measured nothing.

**A crash found by the test, not by a run.** Building the multi-core-point comparison
produced `ValueError: No value for uninitialized VarData object pi_RET[0]` at the core
point `Ybar_ret == 0`. Cause: a multiplier that carries a zero objective coefficient and
appears in no row is never sent to the solver, so it comes back with no value -- which
happens at any `tau` where the candidate schedules no trip in that direction. The readback
called `pyo.value()` on every multiplier unconditionally.

The fix distinguishes two cases that a blanket `or 0.0` would merge, and the distinction is
the whole point. **Some** multipliers empty is structural, and their slope is exactly 0 by
arithmetic. **All** of them empty is a failed load, and the census already names the
consequence: *"an all-zero slope vector, i.e. a cut that constrains nothing, with no
error"*. All-empty now returns `None`, so the caller falls back, labels the cut
`mw_fdiff_fallback` and marks it NOT a valid lower bound (D39).

**A correction.** The first diagnosis was that the swallowed `md.solutions.load_from(res)`
-- `HANDLER_CENSUS.md` Category A -- was the cause. **It was not.** Measured, the load
succeeds and CPLEX reports `optimal`. That handler was made loud anyway, which is right on
its own terms and is one of the three Category A sites closed, but it did not explain the
crash and saying so was wrong. The cause was only found by narrowing which core point
triggered it.

76 tests pass.

### D43 — Cut validity is checked where it is observable, and the tightness test was nearly vacuous
D42 left one of the three MW checks open: strong duality on the selected dual. Writing it
found that a check already in place was weaker than it read.

**`const` is derived, so tightness is close to a tautology.** The cut constant is computed
as `const = ub - sum(dm * y_inc)` (`subproblem_impl.py`), not as `sum(alpha * R)` the way
the module docstring describes. Asserting the cut is tight at its own candidate therefore
re-derives an identity the code just imposed. It catches an index or aggregation slip
between the constant and the coefficient map, which is worth having, and nothing about
duality -- despite reading exactly like a duality check.

**What does depend on the dual being right.** `OptFace` pins the selected dual to
`dual_obj >= ub - tol`. The other side, `dual_obj <= ub`, is weak duality, and it holds
only if the dual feasible region is the true dual of the primal. A region that is too large
breaks it, which is precisely how the stale `min(S, C[tau])` in the dual objective
presented itself: `primal=5105.0 dual=5825.0`. Nothing in the code or the suite checked
that side.

**The observable consequence is the test.** A Benders optimality cut must UNDERESTIMATE the
recourse everywhere, so: build the cut at `y0`, price the true recourse at neighbouring
schedules, and require `cut(y) <= recourse(y)`.

Neighbours matter. Far-away schedules leave the cut thousands of units of slack and would
absorb a badly wrong slope; one flipped slot leaves almost none. Measured over the
single-slot perturbations, the minimum slack is **0.000000 at `yOUT[0,0]`** -- exactly
tight, so a wrong slope has nowhere to hide. Both `mw` and `dual` pass.

`test_the_underestimation_check_is_sharp` asserts that minimum slack stays under 1.0, for
the same reason `test_mw_is_strictly_better_on_at_least_one_core_point` exists: a validity
assertion with slack everywhere passes for any dual wrong by less than the slack, and would
be evidence of nothing.

**Still open, and now the only unchecked side.** There is no runtime guard that
`dual_obj <= ub_base` on every MW solve -- the test covers one candidate and twelve
neighbours, not every cut of every run. That guard is cheap and fails closed (return `None`,
fall back, mark the cut not a valid lower bound per D39). It matters more now that D40 makes
long runs worth doing.

78 tests pass.

### D44 — Branch-and-cut is feasible here, and cannot reuse D39's validity model
Scoping the revival of lazy cuts (removed 2026-02-19, verdict void per D40 because it
predates D30). Two findings, and the second is the one that matters.

**Feasibility: confirmed, with a measured cost.** The blocker the deleted code worked
around -- `lazy_cb_lp_solver` existed "to avoid nested CPLEX" -- does not bite in this
installation. Measured: a `cplex_persistent` MIP with a registered
`LazyConstraintCallback` solving a Pyomo `cplex_direct` LP inside the callback completes
and returns the right answer. CPLEX 22.1.1.0.

The cost is not zero, and it is printed rather than hidden:

    Lazy constraint callback is present.
      Disabling dual reductions (CPX_PARAM_REDUCE) in presolve.
      Disabling presolve reductions that prevent crushing forms.
      Disabling repeat represolve.
    Warning: Control callbacks may disable some MIP features.

So the master loses presolve strength and some MIP machinery in exchange for continuous cut
generation. Whether that trade is positive is a measurement, not an argument -- and this
round has been wrong four times reasoning about exactly this kind of trade.

**The blocking design constraint: a lazy cut cannot be un-added.** D39 models cut validity
as four states and handles the bad ones by dropping the lower bound afterwards -- `INVALID`
and `UNKNOWN` mean "warn and drop the LB". That works in a loop, where the bound is a
number the loop owns and can revise.

It does not transfer. A cut injected into a branch-and-bound tree becomes a constraint on
the search. If it is not a valid lower bound -- `mw_fdiff_fallback` slopes are the live
example, and D39 already marks them `INVALID` -- it can **cut off the true optimum**, and
no later bookkeeping recovers it. The run then reports an optimum that is not one, with no
marker, which is the exact failure class this whole round exists to close.

The old implementation had no notion of this: it predates D39 by six months, and its
handler was `except Exception: return` around `subproblem.evaluate` -- a silent skip. In a
loop a skipped cut costs an iteration. **In a lazy callback a skipped cut means CPLEX
accepts the incumbent as feasible and optimal**, because refusing to cut it off is exactly
how a lazy constraint says "this solution is fine".

Therefore the revival is gated on a rule the loop never needed:

    Only `CutValidity.VALID` may be injected. Anything else must abort the solve,
    not skip the cut and not drop a bound.

Two more defects in the deleted code, recorded so they are not reintroduced: candidate
values were read with `except Exception: val = 0.0`, fabricating a schedule of zeros from a
read failure; and only `cuts[0]` was used, which predates multi-scenario cuts being the
default and would silently drop every scenario but one.

### D45 — With 150 cuts the binding constraint moves from the cut set to the master solve
Two runs on top of D40's LP phase, same test point (Q=3, T=44, 4 scenarios), 1800 s.
`configs/phase1/lp150_then_mip1.yaml` (one MIP iteration) and
`lp150_then_mip8.yaml` (eight). Both differ from `lp_only_150.yaml` only in
`max_iterations` and `per_iteration_time_limit_s` (30 -> 300, so the schedule's 102 s is
no longer clipped to 30).

**The good half. Branch and bound works now.** Root relaxation with these cuts is
794.624549571966. One 102 s MIP solve lifts the best bound to **1080-1090** over ~19 000
nodes, and the internal gap falls from Fase 1's **0.9994 to about 0.20**. An upper bound
appears for the first time at Q=3.

This kills a claim `phase1/README.md` states as a property of the master: *"the LB is
limited by the master's own branch and bound being unable to lift its bound off zero in
30 s, after ~4000 nodes."* It could not lift the bound because it started from a root of
zero. From a root of 794 it lifts 36% in one solve, on the same machine, at a comparable
node rate. Combined with D40: thin cuts caused the zero root, and the zero root caused the
impotent B&B. They were never independent failures.

**The bad half. More Benders iterations buy nothing.** Master best bound over the eight MIP
iterations:

    1080.4  1090.0  1072.8  1064.8  1074.0  1087.8  1067.8  1076.3

It oscillates around ~1077 and does not trend. And the spread is not an effect: iteration
151 is the *same* iteration under the *same* configuration in both runs, and it gave
**1088.07** and **1080.36** -- 7.7 apart. Nodes explored in the same 102 s ranged 11 402 to
31 412 across iterations. That is D26 exactly: a clock-truncated MIP is machine-dependent,
and here the iteration-to-iteration spread is roughly the size of the run-to-run noise on a
single cell.

The UB behaves the same way -- 2166, 2399, 2455, 2477, 2209, 2061, 2030, 2329 -- best 2030,
no trend. Final reported pair is LB 1089.98 / UB 2030.86, a Benders gap of 46.3% against
50.4% after a single iteration. Both runs print `NOT REPRODUCIBLE`; every number here is a
single draw and no gap from them should be quoted without saying so.

**Why more iterations do not help, mechanically.** Each master solve stops on its 102 s
clock at an internal gap of ~0.20, so the reported LB is the best bound of a *truncated*
solve. Adding one cut and re-solving re-truncates at roughly the same place. The cut set is
no longer what limits the bound; **the master solve is.**

**Correction to a prediction made before the run.** I expected the per-solve budget to grow
as the Benders gap closed, since `mp_gap = min(per_iteration_mipgap, max(0.001, g_bd))`.
It did not: `g_bd` sat near 0.5 throughout, which saturates `mp_gap` at the 0.05 ceiling, so
every solve was scheduled at 102 s. The adaptive branch only engages once the Benders gap
falls below `per_iteration_mipgap`, which never happened.

**What this makes the next measurement.** Run 1 -- the long master session -- now has a real
hypothesis behind it rather than hope: if a single master solve is given far more than 102 s
with this cut set, does the bound rise past ~1077, or does it saturate there too? That is
the question D45 leaves open, and it is the one the 12 h session was designed for. Note the
lever is `per_iteration_mipgap`, not the ceiling (handout 5.2).

**One caveat on the reported LB.** `best_lb` is the maximum over iterations. Each master
best bound is individually a valid lower bound, so the maximum is valid too -- but it is the
maximum of eight noisy draws, which makes it an optimistic estimate of what one run yields.
Quote 1090 as the best observed, not as what the configuration produces.

---

## D46 — Branch-and-cut is measured, and the callback costs more than the teardown it saves

Date: 2026-08-10. Supersedes the void 2026-02 removal that D44 scoped.

D44 revived branch-and-cut on the grounds that its previous verdict was reached against the
pre-D30 subproblem -- the one whose constraint set moved with `y`, so no cut was a valid
lower bound -- and was therefore void. It is now measured on the post-D30 model, and it has
a verdict of its own.

**The design.** One CPLEX tree over the master seeded with run 2's 150 LP cuts, with Benders
cuts injected at integer incumbents by a `LazyConstraintCallback` that solves the real
subproblem inline. `configs/phase1/lp150_then_bnc.yaml`.

**The contract that does not carry over from the loop.** A lazy cut cannot be un-added. D39
is fail-closed at the level of the *reported bound*: an INVALID or UNKNOWN cut is added and
the lower bound is dropped afterwards, which is sound only where the loop owns the bound. In
a tree, a cut that excludes the true optimum prunes a subtree that is never revisited. So
the rule inverts: only `VALID` may be injected, and INVALID, UNKNOWN **and NO_CUT** abort
the solve. NO_CUT is the surprise -- benign in the loop, and in a callback the opposite,
because returning without adding a cut is how you tell CPLEX the incumbent is acceptable.

**The measurement.** Same seeded cut set, same persistent solver, same options, same stop
criterion, same budget. One variable: whether a callback is registered
(`branch_and_cut.control_no_callback`).

| run | callback | LB | UB (priced) | tree time |
|---|---|---:|---:|---:|
| `lp150_then_control` | no | **1111.05** | 2351.86 | 410 s |
| `lp150_then_bnc` | yes | 1004.22 | **1923.86** | 384 s |
| `lp150_then_mip8` (the loop, 8 iterations) | — | 1089.98 | 2030.86 | ~800 s |

Both branch-and-cut rows replicate: the control gave 1106.15 and 1111.05 on two draws, the
callback tree 1004.62 and 1004.22. The ~9.6% LB deficit is an order of magnitude outside
that spread, so it is a real effect and not a draw.

**Finding 1 — the callback's cost is what ate the bound.** Registering a lazy callback makes
CPLEX disable dual reductions, restrict presolve to crushing forms and stop repeat
represolve. That is not a footnote; it is paid on the master's own search, and here it cost
9.6% of lower bound. The first branch-and-cut run also received 600 s (see the budget defect
below) and reached 1004.62, against 1004.22 in 384 s -- the extra 216 s bought nothing, so
the callback tree saturates rather than being time-starved.

**Finding 2 — the loop's teardown is real waste, and this is not how to recover it.** The
no-callback control is one plain master solve over the 150 seeded cuts, and at 410 s it
produces a better bound than eight loop iterations produce in ~800 s (1111.05 vs 1089.98).
D45's reading that the binding constraint moved to the master solve is confirmed. But
persistence bought through a lazy callback costs more than the teardown it avoids.

**Finding 3 — the callback buys the upper bound, not the lower one.** Its incumbents are all
priced by the subproblem, so CPLEX is steered toward schedules whose theta is honest: UB
1923.86 against the control's 2351.86, an 18% improvement. Benders gaps: loop 46.3%,
callback tree 47.8%, control 52.8%. Nothing here is competitive -- the monolith still solves
this instance in 39 s.

**What this closes.** User cuts at fractional nodes (D44 step 2) are **not** the next move.
They add the same callback cost at far more nodes than the incumbent-only callback that
already lost, so the mechanism whose price is now measured would be paid more often. The
step is dropped, not deferred.

**Two defects found by the measurement, both in code written for it.**

1. *The tree received the whole run budget instead of what was left.* The driver passed
   `solver.total_time_limit_s` unchanged, so the first run spent ~150 s seeding plus 600 s of
   tree against a 600 s budget -- a 25% silent overshoot of the one number that says what a
   run costs. It gave the callback tree *more* time than intended, so it does not rescue that
   result.

2. *The master objective was reported as an upper bound.* It is `first_stage + theta`, and
   theta is bounded only by the cuts the master holds. With the callback this is defensible
   -- an accepted incumbent is one whose cut is satisfied -- but in the no-callback control
   nothing prices the incumbent, and the first control run reported an "upper bound" of
   1288.86 that is a relaxation value and can sit *below* the true optimum. A 14% Benders gap
   read off that pair would have been fiction. Both paths now price the returned schedule
   once with the subproblem and claim an upper bound only from that. The callback run is a
   check on the fix: its master objective and its priced UB agree to six digits (1923.86),
   which is what the accepted-incumbent argument predicts.

**What it points at next.** Not more time for the same master, and not user cuts. The best
configuration measured here is the plain one -- seed with LP cuts, then one long master
solve -- and the remaining lever on the bound is the formulation: the per-vehicle trip cap
derived from the battery block (handout 5.6 A), which is a valid inequality in `y` alone and,
unlike the recourse anchor D33 found inert at Q=3, does not go slack as Q grows.

Every number in this entry prints `NOT REPRODUCIBLE` and is a single draw except where two
draws are quoted. Only `lp_only_150.yaml` reproduces.

---

## D47 — Run 1 is answered by three points, and the answer is that master seconds do not buy the bound

Date: 2026-08-10.

Run 1 (handout §5.2) was designed as a bounded 12-hour session with 30-minute master
solves, to test whether a master given far more time lifts the bound past the ~1080 that a
102 s solve reaches with the 150 seeded cuts. D45 gave it that hypothesis; D46's control
path gives it a cheap way to sample the curve, because the control is one long master solve
over the seeded cuts with no callback.

Three points, all on the seeded 150-cut master:

| master time | LB |
|---:|---:|
| 102 s | ~1080 |
| 410 s | 1111.05 (two draws: 1106.15, 1111.05) |
| 1520 s | **1148.65** |

**Stated in advance, and it did not land on either.** The 1800 s config's header set the
criteria before the run: ~1130 or below means the curve is flat and the session is answered;
~1250 or above means it is still climbing and the session has a real question. The result is
1148.65, in the band between them. Recording that rather than retrofitting a threshold is
the point -- this project has three refuted conclusions from reading a number after choosing
what it should mean.

**What the curve says.** The bound is close to linear in `ln(t)`: the slope is 22.3 points
per unit of `ln t` between the first two points and 28.7 between the last two. Extrapolating
to 43 200 s (12 h) gives **~1245**. That is a projection, not a measurement, and the first
point comes from a different code path (a loop MIP iteration, with a MIP start), so the
same-path slope of 28.7 is the trustworthy one.

**The reading.** 14× more master time than the 1520 s point is projected to buy ~8% of
bound, reaching ~79% of the monolith's optimum of 1569.44 -- which the monolith itself
produces exactly, in 39 s. The 12-hour session would not change any conclusion in this
repository. It is **not** recommended as a bound-hunting exercise. What would change a
conclusion is a valid inequality in `y` alone, which is where D46 already pointed
(handout §5.6 A, the per-vehicle trip cap).

**What this does not say.** It does not say the master is at its limit, and it does not
close §5.2 as a decision the user cannot reverse -- the session is cheap to run and produces
an honest number. It says the expected value of running it is low enough that it should not
be the next thing done.

Single draw at 1520 s, `NOT REPRODUCIBLE` (D26). The 410 s point is two draws 0.4% apart.

---

## D48 — The recourse depends on `y` only through the per-slot aggregate, and that is a design, not a remark

Date: 2026-08-13. Design recorded in `DESIGN_DD_v1.md`. **Nothing here is measured yet.**

`ProblemSubproblem.evaluate` reads the master candidate only to build
`C_d[τ] = S · Σ_q y_d[q,τ]`, and `solve_subproblem` puts that vector in the right-hand side
of the capacity rows and nowhere else. The arc set, the variable set, the demand rows and
the objective coefficients are functions of `(T, Wmax_slots, p)` alone. So the recourse is a
function of the **signature** `(Y_OUT, Y_RET) ∈ Z_{≥0}^{2T}`, not of `y`: two schedules with
the same signature produce a byte-identical LP, hence the same value **and the same dual**.

At Q=3 / T=44 one signature can carry ~`3.9 × 10^8` distinct `y`.

**Half of this is already exploited, and saying otherwise would be wrong.** With
`aggregate_cuts_by_tau` the cut is written on `Yout[τ]`/`Yret[τ]` and `_assert_q_invariant`
enforces that the collapse is an identity. One cut has always bound the whole fibre. That
dates from D10 and is not a new opportunity.

**What is not exploited is two things.**

1. *The master still branches per vehicle.* `2·Q·T` binaries, and symmetry breaking orders
   vehicles only by **total** departures (`Q−1` rows). The spec §2.8 is right that the
   stronger prefix form is invalid — the answer is therefore not a stronger symmetry
   constraint but a formulation without a vehicle index. Traced constraint by constraint,
   every master row is separable by vehicle; the coupling is exactly three terms
   (`θ` cuts, `ε·Σ Z`, the concurrency penalty), and all three are functions of the
   signature alone. That is a Dantzig-Wolfe structure, and its LP relaxation optimises over
   `conv(integer points of the per-vehicle polytope)` rather than that polytope's LP
   relaxation. D40 put the bound at the LP root, so this is aimed at the measured
   bottleneck rather than at tidiness.

2. *The cut is convex where it only needs to be valid on a lattice.* The recourse is convex
   **and non-increasing** in the signature, so one LP solve proves `Q(Y) ≥ Q(Ŷ)` for every
   integer `Y ≤ Ŷ` — a whole down-set, which is the LBBD shape in Hooker §6.2 and is not
   implied by the convex envelope at fractional `Y`. The obstacle is linearising it: 88
   indicators at T=44. Spec §2.9 (M1) is the standing warning that a sound inequality can
   still make the master worse, so this is staged behind something cheaper.

**Four exactness conditions are pinned, each with the test that must fail if it stops
holding:** E1 recourse constant on the fibre (duals too, not only values); E2 `y` in the
RHS only — D30 restated as a standing condition rather than a past fix; E3 master rows
separable by vehicle, with an explicit allow-list for the three coupling terms; E4
homogeneous fleet, reusing the precondition symmetry breaking already enforces.

**The trap this closes in advance.** The CP research note proposes a minute-level
availability profile driven by the master's slot decision. Built naively that is D30 again —
`y` enters `A`, not `b`, and no cut generator on top is valid. It is safe if and only if the
minute grid is fixed and `y` scales the right-hand side through a constant 0/1 incidence
matrix, which holds for a deterministic trip duration and per scenario for travel-time
scenarios, and fails the moment trip duration becomes a decision. Note that the signature
property survives the extension unchanged, which is the actual argument for
multi-resolution: the master-side structure is indifferent to the subproblem's resolution.

**Staged, with the refutation stated in advance for each:** (0) count signature revisits in
the D40 LP run — diagnostic, free; (1) window trip-cap inequalities in `Y` alone from a
single-vehicle diagram, which is D46's own named lever, refuted if the LP root does not move
off 794.62; (2) the down-set cut, measured offline before integration; (3) the
Dantzig-Wolfe reformulation with pricing as a resource-constrained shortest path.

Stage 3 carries one claim that must be **verified by enumeration, not argued**: the pricing
DP's max-battery dominance. `charge_before_idle` makes charging in slot `t` depend on
`c[t−1]`, so the greedy policy is constrained by its own past. It looks monotone. This round
has four refuted conclusions that also looked obvious, so it gets checked against the MILP
at Q=1, T≤12 before anything relies on it.

**Standing caveat, unchanged from D46/D47.** The monolith solves the Q=3 test point exactly
in 39 s and the best measured Benders configuration reaches 1148.65 in 1520 s. None of this
is competitive until that comparison changes, and a better bound that is still an order of
magnitude behind must be reported as a result about Benders, not as a method.

---

## D49 — The window trip caps lift the LP root 6.6% and cost nothing, and the prediction that said otherwise was mine

Date: 2026-08-13. Stage 1 of `DESIGN_DD_v1`, measured. Configs:
`configs/phase1/dd_lp_caps_{off,on}.yaml` (small point) and
`configs/phase1/dd_p1_caps_{off,on}.yaml` (Fase 1 point).

**The inequality.** From the single-vehicle relaxed decision diagram
(`problem/vehicle_dd.py`), for every window `[t1,t2]`

    sum_{tau in [t1,t2]} ( Y_OUT[tau] + Y_RET[tau] )  <=  Q * max_trips(t1,t2)

`max_trips` is maximised over every entry state, so it bounds any vehicle's departures in
the window whatever preceded it; summing over Q independent vehicles gives the right-hand
side. It is in `Y` alone, so it constrains the first stage and cannot touch the validity of
the Benders cuts. Every simplification in the diagram is in the permissive direction --
`charge_before_idle` dropped, entry battery `Emax`, entry location maximised -- because a
cap that is too large is merely weak while a cap that is too small stops the master being a
relaxation, which is the 2.8 / D30 failure mode.

**Two points, one variable, `master.window_trip_caps`.**

| point | slot | Q | trip_slots | LP root off | LP root on | delta | wall off | wall on |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 30 | 2 | 1 | 2769.52 | 2781.83 | **+0.44%** | 19 s | 43 s |
| Fase 1 | 15 | 3 | 2 | 794.624549571966 | **846.936259826576** | **+6.58%** | 508 s | **465 s** |

Both arms of both pairs: zero `CHECK FAIL`, `MW FAIL`, `SP WARN`, `CHECK ERROR`, `INVALID`,
`NOT REPRODUCIBLE`. The LP phase never stops on the clock, so D26 does not apply and these
reproduce.

**The control is also a regression test, and it passed.** With caps off,
`dd_p1_caps_off.yaml` is `lp_only_150.yaml` plus one inert key, and it reproduced D40's
number to the digit -- 794.624549571966 -- with the whole trajectory matching point by
point (0.311 at 10, 601.424 at 40, 773.996 at 80). So the D48 work adds to the model
without changing it.

**The mechanism is local density, not total mass.** The caps-off LP ends at a total
departure mass of 39.2557 against a whole-horizon cap of 39; with caps on it ends at 38.6.
The horizon-wide constraint moved the total by 0.66 departures and is essentially slack.
The 6.58% came from the ~942 *window* caps redistributing where departures sit. This also
explains the small point: at `trip_slots = 1` the travel-time half of the cap is vacuous --
"departures at least one slot apart" is every slot -- leaving only the energy bound, which
the master's own LP relaxation largely already derives, since `b` is continuous and
`C4_bal` is an equality. Relative tightness of the whole-horizon cap tracks this: 26 vs a
trivial 40 (1.5x) at the small point, 39 vs 123 (3.2x) at Fase 1.

**The prediction, recorded in the config header before the run, and refuted.** It said
+1% to +5% at 2-4x the master time. The result is +6.58% at 0.92x the master time: wrong
on the magnitude and wrong on the sign of the cost. Worse, mid-run -- after seeing the
control's 39.26 against a cap of 39 -- I revised the expectation down to "well under 1%" on
the grounds that the cap was nearly slack. The observation was correct and the inference
was wrong: the gain never depended on cutting total mass. That is D30/D40/D45's pattern
again, and it happened *with the prediction already written down*. Writing it down makes
the miss legible; it does not prevent it.

**Reading, per the criteria the header set in advance.** Above +5% means the mechanism is
scale-sensitive in the direction hoped for, and the next question is **Q**, not which subset
of caps to add. The cap's right-hand side is `Q * max_trips`, so its bite scales with the
fleet.

**What does NOT change: standing.** The root goes from 50.6% to 54.0% of the monolith's
1569.44, which the monolith still produces exactly in 39 s. A 46% gap is not closed by 6.6%
of root. What this does change is the input to D47's curve -- the 1148.65-at-1520 s point
started from a root of 794.62, and whether a root of 846.94 moves it is a separate,
unrun measurement.

**Stage 0, the fibre count, is answered and it is a negative.** Across 150 iterations the
loop hands the subproblem 150 distinct signatures and repeats none. Caching priced
signatures buys nothing. At the small point only 5 of 40 candidates were integral, each
carrying a fibre of `2^24 = 16 777 216` master solutions the subproblem cannot distinguish
-- so the redundancy is real, but it lives in the master's search, not in the loop. That is
evidence for stages 1 and 3 and against a memo, which is what `DESIGN_DD_v1` predicted.

**Exactness conditions E1-E4 are now enforced rather than assumed** (`tests/`, 132 tests
green). E1 checks that the recourse is constant on the fibre **including the duals**, since
value equality alone would let two members of one fibre produce two different cuts. E2 is
D30 as a standing condition: recourse non-increasing in the signature over a strict
ascending chain, and the dual row set unchanged. E3 walks every master row and asserts it
touches at most one vehicle, with a four-entry allow-list -- the contract stage 3 needs.

**One open question, deliberately not closed.** The caps-on arm at the small point fired
`[CHECK FAIL] MP total exceeds SP total` once. The line printed both sides at `%.6g`, where
any violation under ~0.005 renders identically, so it could not be triaged; it now prints
17 digits and the excess. Measured: excess `5.48e-05`, relative `1.97e-08`, against a check
whose tolerance is **absolute 1e-6** -- about 3.6e-10 relative at that magnitude, tighter
than CPLEX's own optimality tolerance, while every comparable check in this codebase is
scale-relative (`_ok`, `_assert_q_invariant`). **The tolerance was not changed.** Loosening a
validity check on the strength of one warning is how a real defect gets tuned away.

---

## D50 — The Q hypothesis is refuted, the monolith is back in the repository, and 4190.74 is a log-parsing artifact

Date: 2026-08-13. Three results from one session, and only one of them was planned.

### 1. The window trip cap's effect does not grow with the fleet

D49's reading rule said the next question was Q. It was measured, at the Fase 1 point
(slot 15, `trip_slots` 2, 4 scenarios), one variable per pair:

| Q | LP root off | LP root on | delta | wall off | wall on |
|---:|---:|---:|---:|---:|---:|
| 3 | 794.624549571966 | 846.936259826576 | **+6.58%** | 508 s | 465 s |
| 4 | 330.003661932171 | 332.277638592403 | **+0.69%** | 411 s | 453 s |
| 5 | 313.612329376050 | 318.058777573444 | **+1.42%** | 454 s | 1019 s |

The prediction recorded in those config headers before the run was "+8% or more at Q=4 and
+9% or more at Q=5", growing with Q. The effect **falls** from Q=3 to Q=4 and is not
monotone. By the criterion those headers set in advance, **the Q hypothesis is dead and
D49's reading rule was wrong.** At Q=5 the caps also cost 2.2x the wall time for +1.42%,
which is spec 2.9's M1 trade-off appearing for real.

That is the third consecutive refuted prediction on this mechanism (D49 twice, here once).
Recording the count because the pattern, not the individual miss, is the useful signal:
every one of the three was an argument from structure that a measurement overturned.

**What the three points do track is the size of the recourse** -- 794, 330, 314. The cost
collapses once the fleet is adequate, and the cap plausibly bites only while departure
capacity is the binding resource. That is a hypothesis on three points and is labelled as
one. The measurement that separates it from a pure `trip_slots` effect is Q=2 at slot 15,
where the fleet is tightest and `trip_slots` stays at 2; the existing Q=2 point is at
slot 30 and therefore confounded. Configs: `configs/phase1/dd_q2s15_caps_{off,on}.yaml`,
prediction recorded in their headers.

**Measured, and the tightness hypothesis is refuted too.** Those headers set `>= +6.6%`
as support and `< +3%` as refutation. The result is **+1.52%** (3751.61 -> 3808.66, LP
roots at phase self-termination, 92 and 91 iterations). Q=2 at slot 15 is the tightest
fleet in the whole set and it does not beat Q=3. The full curve:

| Q | slot | trip_slots | LP root off | LP root on | delta |
|---:|---:|---:|---:|---:|---:|
| 2 | 30 | 1 | 2769.52 | 2781.83 | +0.44% |
| 2 | 15 | 2 | 3751.61 | 3808.66 | +1.52% |
| 3 | 15 | 2 | 794.62 | 846.94 | **+6.58%** |
| 4 | 15 | 2 | 330.00 | 332.28 | +0.69% |
| 5 | 15 | 2 | 313.61 | 318.06 | +1.42% |

**The honest reading: Q=3 is an outlier and nothing measured explains it.** Every other
point sits between +0.44% and +1.52%. Two mechanisms were proposed and both are dead --
fleet size, then fleet tightness -- which is four refuted predictions on this one
mechanism across D49 and D50. The useful move is to stop proposing a third and record the
curve as measured.

**Consequence for stage 1.** At ~1% of LP root for 1x to 2.2x the master time, the window
trip caps are not the lever D46 named. They are valid, cheap to derive and correctly
implemented; they are also not worth turning on by default, and `master.window_trip_caps`
stays off. Whatever is special about Q=3 is a separate question from whether the caps pay,
and the answer to the second is no.

**Method note, recorded because it cost about an hour of machine time.** These runs were
budgeted by `max_iterations` (right, per D26, for comparing two arms) with no wall-clock
bound. But the LP phase -- the only part read -- self-terminates at "no cut generated"
well before the iteration cap: at 92 of 150 here, versus consuming all 150 at Q=3. So each
run spent roughly two thirds of its wall time in a MIP tail whose numbers are discarded.
The Q=2 control took 4m23s to produce its LP root and then ran 26 more minutes. Budget
this class of experiment with `total_time_limit_s` as well, or add a flag that ends the
loop when the LP phase exits.

### 2. The monolithic MILP is in the repository again, and reproduces 4183.24

`src/mobauto2_milp/` (11 source files) plus `configs/milp/baseline_d9_monolith.yaml`.
Measured: `status=OPTIMAL best_lb=4183.24 best_ub=4183.24 gap=0.000%` in 6.7 s. The
reference that non-negotiable 8 depends on is regenerable for the first time.

**What "independent" does and does not mean, now that the code can be read.** The monolith
attaches the exact recourse LP and pins `theta` to it by *equality*, so there is no cut, no
`theta` approximation, and no D30-class defect can reach it. That is independence from the
**decomposition**. It is *not* independence from the **formulation**: `mobauto2_milp/model.py`
is a second copy of the first-stage model that `mobauto2_benders/problem/master_impl.py`
also implements, kept in sync by hand. A defect present in both copies is invisible to this
check. Non-negotiable 8 currently claims more than it delivers and should be reworded.

Constraint sets match. One difference, verified rather than assumed: `atL`, `atM` and
`inTrip` are continuous `[0,1]` in the monolith and `Binary` in the master. They are pinned
by equalities driven by binary `y` from a fixed initial state, so they are implied integral
-- measured at a maximum distance from an integer of **1.3e-15**. Safe, and it makes the MIP
easier. The four demand files are byte-identical to the repository's modulo CRLF, so the
instance is the same one.

### 3. 4190.74 is a CPLEX log-parsing artifact, and the parser is still live in both packages

The monolith run printed the solver API and the log parser side by side on the same solve:

    MP raw solver:    incumbent=4183.24   best_bound=4183.24
    MP parsed bounds: best_integer=4190.74  best_bound=4004.31

**Mechanism.** CPLEX writes the literal word `integral` in the Objective column of every
line announcing a new incumbent. `parse_cplex_log_text`'s node-table regex requires a
*number* in that column, so it matches none of them, and it does not parse the terminal
`MIP - Integer optimal solution` line either. It therefore reports the state as of the last
*periodic* progress line. In this log:

    SKIPPED: *    31    21   integral  0   4190.7400 ...
    SKIPPED: *   935   237   integral  0   4187.7400 ...
    SKIPPED: *  1752     1   integral  0   4183.2400 ...   <- the optimum
    MATCHED:     778   231  4044.4823 20   4190.7400 ...   <- what it returns

So the parser returns the incumbent from node 778 and misses every improvement after it.
**On this instance that number is 4190.74** -- the value D30 had to withdraw as the
reference optimum, and whose looseness D30 records as the reason the validity check "could
not detect a defect that inflated the bound by less than 7.5". This is a plausible origin
for it having been written down as the optimum at all.

**Severity: real defect, conservative direction.** In the Benders loop the parsed
`best_bound` feeds `best_lb` when the solver API supplies none. A branch-and-bound best
bound is monotone non-decreasing, so any earlier value is `<=` the final one, and the code
takes `max(best_lb, parsed)`. A stale parse therefore **under-reports** the lower bound and
cannot manufacture an invalid one. It widens reported gaps and can put a stale number in a
manifest. The file is byte-identical in both packages. **Not fixed here** -- it sits in the
bound path.

### 4. `p` already means different things at different resolutions

Found while scoping the minute-level reference. The waiting term is in **slots** and `p` is
in the same unit (D7/D8, deliberate and internally consistent). The consequence is that the
physical trade-off `p` encodes moves with `slot_resolution`:

| config | slot | p | one unserved passenger is worth |
|---|---:|---:|---:|
| `baseline_d9.yaml` | 30 | 50 | 1500 passenger-minutes |
| `tests/fixtures/soundness.yaml` | 30 | 50 | 1500 passenger-minutes |
| `configs/phase1/lp_only_150.yaml` | 15 | 50 | **750** passenger-minutes |

The Fase 1 test point therefore treats an unserved passenger as **half** as bad as the
baseline instance does. Every individual result stays valid -- each run is internally
consistent -- but **no objective from slot 15 is comparable to one from slot 30**, and the
multi-resolution programme (research idea RQ3) cannot compare recourse costs across
resolutions until this is fixed.

`Wmax` is already handled correctly: `Wmax_minutes` is a config key converted to slots at
load. `p` is the odd one out. The fix is to match it -- take `p_minutes` in config and
derive `p_slots = p_minutes / slot_resolution`, giving 50 at slot 30, 100 at slot 15 and
1500 at slot 1 for the same 1500 passenger-minute policy. **Not done here**: it would change
every slot-15 number in the repository, so it is a deliberate measured change, not a
silent one.

---

## D51 — Minute-level valuation changes the schedule and serves six more passengers, and the prediction that said it would not was mine again

Date: 2026-08-14. Steps 1-3 of the multi-resolution programme, plus two defects found on
the way. `minute_pricer.py`, `scripts/price_at_minutes.py`,
`scripts/minute_vs_slot_schedule.py`.

### 1. The objective is 93% a headcount of passengers nobody reached

`baseline_d9` decomposes exactly:

    4183.24  =   283.00  waiting (283 slots)               6.8%
             +  3900.00  78 unserved passengers x 50      93.2%
             +     0.24  start cost (0.01 x 24 departures)

The model is very nearly "serve as many as you can", with waiting as a tiebreak. Worth
stating because it governs how every other number here reads: at `p = 50` slot units --
1500 passenger-minutes, 25 hours -- waiting cannot compete with abandonment.

And the 78 are not a capacity shortage. 24 departures x 15 seats = **360 seats for 300
passengers**. They are unserved because no seat is inside their 60-minute window.

### 2. `Wmax` granted more waiting than the config asked for

`Wmax_slots = ceil(Wmax_minutes / slot_resolution)`, in **three** places
(`app.py`, `subproblem_impl.py`, `monolith.py`). `ceil` rounds the cap UP: at 30-minute
slots `Wmax_minutes: 45` became 2 slots, and a passenger could be made to wait 60 minutes
against a stated 45. A maximum wait is a service promise; rounding it up is the one
direction that must not happen silently.

Now `floor`, via one helper per package, and a cap shorter than one slot raises instead of
being quietly rounded to one. **No number in this repository changes**: at
`Wmax_minutes 60` with slots of 30 or 15 the quotient is exact, so ceil and floor agree,
and `baseline_d9` still gives 4183.24. Only non-multiples move (45/30: 2 -> 1).

The other half of the artifact is not fixable in the conversion. A departure placed
mid-slot -- the research note's own midpoint convention -- adds half a slot, so two slots
is 75 real minutes against a stated 60. That is a property of the slot ABSTRACTION. It is
why the minute pricer enforces the cap in real minutes, and why it finds *more* unserved
under midpoint (79) than the slot model claims (78): it refuses assignments the slot
model allowed.

### 3. The slot model reports the waiting time nearly double

Pricing the monolith's proven-optimal schedule against the demand's real arrival minutes
-- same schedule, no re-optimisation, only a truer valuation:

| valuation | waiting | avg per served pax | unserved |
|---|---:|---:|---:|
| slot model | 8490 pax-min | **38.24 min** | 78 |
| minute, departures at slot start | 4560 pax-min | **20.36 min** | 76 |
| minute, departures at slot midpoint | 5120 pax-min | **23.17 min** | 79 |

The slot model overstates waiting by **66-86%**. That is the number a paper would quote,
and it is wrong by roughly a factor of two.

**A correction to how this was first reported.** The initial write-up led with the effect
on the TOTAL objective -- 1.5% -- which is close to meaningless: 93% of the total is the
unmet-demand headcount, which minute pricing barely moves, so comparing totals compares a
number that is mostly a term the exercise does not change. The waiting term alone is the
measurement. The user caught this; the first framing was wrong.

### 4. Step 3: minute valuation changes the schedule, and the change is worth 7.2%

Same instance, solved monolithically twice to proven optimality. A is the current model
(slot master, slot recourse). B is the architecture the research idea proposes (slot
master, MINUTE recourse), with `y` entering the recourse only through the capacity
right-hand side, so E2 holds and the construction is the one a minute-level Benders
subproblem would use. B's recourse is scaled by `1/slot_resolution` so both objectives are
in the same units and the first-stage terms keep the same relative weight.

**The schedules are not the same.** OUT differs in 2 of 12 departures, RET in 4 of 12.

Both priced at minute fidelity, which is the only fair comparison -- A optimised a
valuation now known to be wrong, so it must be judged by what it really costs:

| schedule | true cost | waiting | avg wait | unserved |
|---|---:|---:|---:|---:|
| A, slot-optimised | 123 620 pax-min | 5120 | 23.17 min | **79** |
| B, minute-optimised | **114 682 pax-min** | 5182 | 22.83 min | **73** |

**B is better by 8938 passenger-minutes, 7.2%** -- and the mechanism is not shorter waits.
B's waiting is slightly WORSE (5182 vs 5120). The entire gain is **six more passengers
served**, and 6 x 1500 = 9000 accounts for all of it. The slot model mis-estimates who is
reachable inside a real 60-minute window and therefore abandons six passengers it could
have carried.

**The prediction this refutes was mine, and it was recorded one message earlier:** "since
93% of the objective is unmet-demand headcount and minute pricing barely moves it, a
minute-level subproblem is unlikely to steer the master anywhere new at this penalty. The
lever is `p`, not resolution." Both halves wrong. It steers the master, and what it buys
is precisely the headcount that was argued to be insensitive. That is the fifth refuted
prediction on this project in two days, and the fourth from reasoning about a mechanism
instead of measuring it.

**Caveat, stated rather than buried.** B optimises exactly the metric it is scored on, and
"the truth" here is the midpoint placement convention. Under the `start` convention the
comparison has not been run. The 7.2% is therefore conditional on a modelling assumption
that this project has already seen move an answer by 4% of the total.

### 5. Where waiting starts to matter, at a fixed schedule

Same schedule priced at a range of penalties. Only the weighting changes:

| `p_minutes` | `p` (slots, 30 min) | waiting share of objective |
|---:|---:|---:|
| 1500 (today) | 50 | 4.1% |
| 300 | 10 | 17.8% |
| 150 | 5 | 30.2% |
| **60** | **2** | **46.6%** |
| 15 | 0.5 | 18.2% |

Waiting peaks near `p_minutes = 60` and collapses below it -- at a low penalty the model
abandons passengers rather than making them wait, so the waiting term empties out
(5120 -> 60 pax-min at `p_minutes = 5`). The band where waiting genuinely trades against
service is roughly `p_minutes` 60-300, i.e. `p` between 2 and 10 at 30-minute slots.
Today's 50 is five to twenty-five times above it.

This does NOT say the penalty is wrong -- that is a policy question, and 25 hours per
abandoned passenger may be exactly what is intended. It says that at the current setting
the objective is a service-coverage objective with a waiting-shaped tiebreak, and any
claim about waiting time should be read that way.

### What this changes

The multi-resolution premise survives its first real test, on the evidence rather than on
the argument: a slot master with a minute-level recourse picks a different and materially
better schedule, and reports a waiting time that is not off by a factor of two. That is
step 3 answered in the affirmative. It says nothing yet about whether a *decomposed*
minute subproblem converges usefully, which is the next question and a different one.

---

## D52 — The decomposed minute recourse converges at the same rate as the slot one, and costs under 1% per iteration

Date: 2026-08-14. Answers the question D51 left open: step 3 was settled monolithically,
which says nothing about whether the same recourse works *through* the decomposition,
where it is only ever seen as a cut. Configs:
`configs/phase1/d9_recourse_{slot,minute}.yaml`.

**Why it was cheap to answer.** A minute-level recourse still has its capacity rows
indexed by DEPARTURE SLOT -- one per `(direction, tau)`, right-hand side
`C_d[tau] = S*Y_d[tau]` -- so it returns exactly one dual per slot, the same object
`solve_subproblem` already returns. The cut, `aggregate_cuts_by_tau`,
`_assert_q_invariant`, the validity classification and the master are all untouched. Only
the LP behind the dual moves to minutes. The objective is expressed in slot-equivalent
units (waiting divided by `slot_resolution`, penalty in `p_slots`) so `dm = S*pi` lands in
the units theta and the first-stage terms already use; without that the recourse would
outweigh `start_cost_epsilon` and `concurrency_penalty` by a factor of `slot_resolution`.

**E2 holds by construction** and is asserted: the arc set is a function of
`(T, slot_resolution, Wmax_minutes, departure_policy)` and the demand, never of the
schedule. Minute mode with no arrival minutes raises rather than falling back to the slot
model, which would report a multi-resolution run that never happened.

**Measured**, `baseline_d9`, plain dual slopes on both arms (Magnanti-Wong is off because
`solve_mw_dual` mirrors the SLOT primal and would be selecting duals for an LP that is no
longer the one being solved), 240 s each:

| iteration | slot LB | minute LB |
|---:|---:|---:|
| 3 | 469.3 | 84.2 |
| 8 | 2090.5 | 2127.3 |
| 12 | 2781.9 | 2467.7 |
| 14 | 2887.1 | 2538.2 |

Each arm has a different optimum -- 4183.24 for slot, 3822.97 for minute -- so the
comparable figure is progress toward its own target. At iteration 14 the slot arm is at
**69.0%** of 4183.24 and the minute arm at **66.4%** of 3822.97. **Convergence per
iteration is indistinguishable.**

**Subproblem cost: 0.117 s against 0.020 s, roughly 6x.** In absolute terms it is
irrelevant -- iterations here run 12-17 s, so the minute LP is under 1% of one. The
minute arm completed 14 iterations against 20 in the same wall clock, and that gap is
master time, not subproblem time.

**Validity.** Both arms report zero `CHECK FAIL`, `SP WARN` and `CHECK ERROR`, and each
lower bound sits under its own optimum: 2887.08 <= 4183.24 and 2538.23 <= 3822.97.

**What is NOT claimed.** Neither arm converged inside 240 s, so this is a comparison of
convergence *rates*, not a demonstration that either terminates. Both print
`NOT REPRODUCIBLE` (D26). That neither converges is a property of the master, already
established by D45/D46/D47, and is not evidence about the minute recourse. A converged
comparison needs the master problem solved, which is a different work item.

**The cross-check that makes the number trustworthy.** `solve_minute_recourse` (the
decomposition path) and `price_schedule_at_minutes` (the reporting path) are independent
implementations of the same quantity, written for different purposes. A test asserts they
agree on the same schedule, up to the `1/slot_resolution` unit factor, along with the
non-positivity of the capacity duals and monotonicity of the recourse in capacity -- P3 at
minute resolution, which is the property whose failure voided six months of bounds (D30).

**Reading.** Multi-resolution Benders is not blocked by the decomposition. The recourse
can move to minutes for under 1% of iteration cost, with unchanged cut machinery and
unchanged convergence per iteration, and D51 already showed the minute objective is worth
7.2% and six passengers on this instance. The obstacle to a usable method remains what
D45-D47 measured: the master.

---

## D53 — The multi-resolution gain is real but the placement convention decides its size, the penalty was 27x the operator's own policy, and the D47 baseline does not reproduce

Date: 2026-08-14. Three corrections and one sweep. Partially retracts D51's headline.

### 1. `p` was 27x the operator's stated policy, and it changes what the whole result means

The operator's indifference statement: delaying a shuttle already carrying 14 passengers
by 4 minutes costs `4 x 14 = 56` passenger-minutes and is worth one extra passenger
carried. So one unserved passenger is worth about **56 passenger-minutes**.

Every config in this repository states `p: 50` in SLOT units, which at 30-minute slots is
**1500 passenger-minutes** -- 27x that policy. At 1500 the model will delay 14 passengers
by 107 minutes (1500/14) to pick up one more. This is exactly the conflation that
`p_minutes` was introduced for in D50, and it survived one round of review because `50`
and `56` look like the same number.

**Consequence for D51/D52: every multi-resolution measurement so far was taken in the
regime where waiting barely matters** -- at `p_minutes = 1500` waiting is 4.1% of the
objective. The operator's regime is where waiting is roughly half of it.

`configs/baseline_d9_p56.yaml` and `configs/milp/baseline_d9_p56_monolith.yaml` ship the
policy regime. The originals are NOT edited: 4183.24, the soundness suite and every result
from D30 onward are computed against `p: 50`, and changing it in place would silently
invalidate all of them. Both regimes ship; every reported number must say which it used.
The policy-regime reference optimum is **399.9067**, proven optimal in 1 s, serving
185/300.

### 2. The 7.2% headline was placement-dependent, and under one convention it is zero

D51 reported that a minute-level recourse picks a better schedule worth 7.2%. That was
measured under the `midpoint` convention only. Under `start`:

| placement | `p_minutes` | A slot-opt | B minute-opt | gain | schedules |
|---|---:|---:|---:|---:|---|
| midpoint | 1500 | 123 620 | 114 682 | 7.2% | differ |
| **start** | **1500** | 118 560 | 118 560 | **0.00%** | **identical** |
| midpoint | 56 | 9 326 | 8 794 | 5.7% | differ |
| start | 56 | 8 596 | 8 388 | 2.4% | differ |

**At the current penalty under `start`, minute-level valuation buys nothing at all** -- the
two first stages choose character-identical schedules. The reason is mechanical: under
`start` the slot model's `(tau - t)` accounting is very nearly exact in minutes, so there
is nothing for a finer valuation to correct. Under `midpoint` every departure shifts by
half a slot and the reachability windows genuinely move.

D51's headline stands only with the convention attached. Stated without it, it was wrong.

### 3. The sweep: the gain is real, and it is large in the policy regime

Five demand shapes (`scripts/make_instances.py`), Q=2, 30-minute slots,
`p_minutes = 56`, both conventions, every solve proven optimal:

| shape | gain (start) | gain (midpoint) | extra pax served |
|---|---:|---:|---:|
| flat | 1.36% | 12.43% | 9-11 |
| commuter | 3.45% | 10.48% | 4-14 |
| bimodal | 1.32% | 8.19% | 4-14 |
| burst | 0.00% | **48.60%** | 0 |
| spiky | 8.17% | 30.90% | 11-19 |

The minute-optimised schedule is never worse and carries up to 19 more passengers. In the
policy regime the effect is an order of magnitude larger than the 7.2% first reported.

**The advance prediction was that the gain tracks how much demand structure lives below
one slot -- least on `flat`, most on `burst` and `spiky`.** Under `midpoint` that holds:
`burst` and `spiky` are far ahead at 48.6% and 30.9% against 8-12% for the rest. Under
`start` it fails on `burst`, which gives exactly 0.00%.

**And that failure is an artifact of the generator, not a property of burst demand.**
`_burst` places its windows at minutes 60, 180, 420 and 560, and the first three are exact
multiples of 30. Under `start` the departures land on the bursts, so slot reasoning is
accidentally correct. The instance set should be regenerated with deliberately
slot-misaligned windows before this row is cited; as it stands it measures the generator's
arithmetic.

### 4. The D47 baseline does not reproduce, and is inconsistent with a bound

D47 states the monolith produces the Q=3 optimum **1569.44** exactly, in **39 s**, and
reads the entire Run 1 curve against it ("~79% of the monolith's optimum").

Measured on that instance -- Q=3, slot 15, 4 scenarios -- with all cores:

    180 s -> status FEASIBLE, LB 1579.6965, UB 1674.11, gap 5.64%

**The lower bound 1579.70 exceeds the claimed optimum 1569.44.** A branch-and-bound best
bound is valid, so the optimum of this instance is at least 1579.70 and 1569.44 cannot be
it. Nor does 39 s reproduce: single-threaded it was still FEASIBLE at 300 s
(LB 1439.11 / UB 2025.11).

Until this is resolved, **every "% of the monolith's optimum" statement in D46 and D47 is
unsupported**, including D47's recommendation against the 12-hour session. The conclusion
may well survive -- the gap is large either way -- but the number it rests on does not.

There is also a standards question to settle explicitly rather than by accident: D26
requires single-threaded runs for reproducibility, and a 39 s baseline was almost certainly
multi-threaded. A baseline and the method it judges should not be held to different
standards.

### What this changes about the contribution

The multi-resolution result survives and is stronger than first reported, in the regime the
operator actually wants. But two of its three headline numbers needed correcting within a
day of being recorded, both because a modelling assumption was left implicit -- the
penalty's units, then the placement convention. Any write-up must state both explicitly
and report the range across conventions rather than a single figure.

---

## D54 — The departure-placement convention is a modelling decision, not a detail, and the D47 baseline optimum was wrong by 5.4%

Date: 2026-08-14. Closes the convention question D53 left open, and replaces the Q=3
baseline every Benders result in this repository is measured against.

### 1. `end` is the convention the aggregation implies, and it does not restore a bound

The operator's objection to `midpoint`: demand is collected over [07:00, 07:30) and the
bus is assumed to take them at 07:30, so the departure belongs at the END of its slot. A
bus leaving at 07:30 can carry everyone who arrived in that window; one leaving at 07:00
carries almost none of them.

Worked through, the three conventions differ in a way that looks decisive. A passenger
arriving at minute `m` in slot `t`, served from slot `tau`, is charged `(tau-t)*delta` by
the slot model, while the true wait is `D(tau) - m` with `m` in `[t*delta, (t+1)*delta)`:

| convention | `D(tau)` | true wait vs slot charge |
|---|---|---|
| `start` | `tau*delta` | true <= charge, slot OVERSTATES |
| `midpoint` | `tau*delta + delta/2` | straddles |
| `end` | `(tau+1)*delta` | true >= charge, slot UNDERSTATES |

That suggests `end` is the only convention under which the slot recourse lower-bounds the
minute recourse -- the direction Benders needs.

**Measured, and the aggregate does not follow the per-arc inequality.** Same schedule,
`baseline_d9` in the p56 regime:

    slot model claims          11 990 pax-min
    minute truth (start)        8 596     slot overstates by 3 394
    minute truth (midpoint)     9 326     slot overstates by 2 664
    minute truth (end)          9 806     slot overstates by 2 184

The slot model overstates under **all three**, `end` included. The per-arc inequality is
correct and does not lift, because the minute model re-optimises the ASSIGNMENT and faces
a different REACHABLE SET -- a real 60-minute cap instead of a two-slot window. Under `end`
it abandons more passengers (130 vs 115) but waits far less (2526 vs 5550 pax-min) and is
cheaper overall. Reasoning about one arc said nothing about the optimum over all of them.

**The consequence for bound claims, which matters more than the convention.** Because the
slot model overstates the cost of any given schedule, its optimum is an UPPER bound on the
minute-level optimum, not a lower one. **A slot-level Benders lower bound therefore bounds
the slot problem only and says nothing rigorous about the minute-level problem.** Any
paper reporting a slot-Benders LB against a minute-level notion of optimality would be
making a claim its own construction does not support.

### 2. The gain decomposes into two distinct errors

On `baseline_d9`, p56, midpoint:

| quantity | pax-min |
|---|---:|
| what the slot model CLAIMS its schedule costs | 11 990 |
| what that schedule REALLY costs | 9 326 |
| what the best achievable schedule costs | 8 794 |

    valuation error  +28.5%   the model misprices its own schedule
    decision error    +6.0%   the schedule it picks is worse than achievable

These are independent and should be reported separately. The valuation error is what a
reader of the model's output is misled by; the decision error is what the operator
actually loses.

### 3. The sweep across five demand shapes and three conventions

Q=2, 30-minute slots, `p_minutes = 56`, every solve proven optimal:

| shape | start | midpoint | end | extra pax served (end) |
|---|---:|---:|---:|---:|
| flat | 1.36% | 12.43% | 12.53% | 15 |
| commuter | 3.45% | 10.48% | 6.26% | 12 |
| bimodal | 1.32% | 8.19% | 14.02% | 32 |
| burst | 0.00% | 48.60% | 42.43% | 52 |
| spiky | 8.17% | 30.90% | 27.74% | 41 |

**`midpoint` and `end` agree; `start` is the outlier.** Under either of the two
operationally sensible conventions the minute-optimised schedule is 6-49% cheaper and
carries 12-52 more passengers. `start` assumes the bus departs at the instant its
collection window opens, before the passengers it is collecting have arrived, and is not
a defensible operating assumption.

D53 recorded this as "the result depends on an arbitrary choice". That reading is now
withdrawn: it depends on a choice, and one of the three options is physically implausible.
The `burst` 0.00% row under `start` remains a generator artifact (windows at minutes 60,
180, 420 are exact multiples of 30) and should not be cited until the shapes are
regenerated misaligned.

### 4. The Q=3 baseline: the optimum is 1658.86, not 1569.44, and it takes 947 s

D47 states the monolith produces the Q=3 optimum **1569.44** exactly in **39 s** and reads
the whole Run 1 curve against it. Measured, all cores, `term=optimal` with both bounds
agreeing to seven figures:

    LB 1658.8588281   UB 1658.8600   947 s

**1569.44 is 5.4% BELOW a proven optimum**, so it was never this instance's optimum. And
the monolith needs 947 s on all cores, not 39 s single-threaded.

Corrected readings: the 1520 s Benders result of 1148.65 is **69.2%** of the true optimum,
not 79%; the 12-hour projection of ~1245 becomes **75%**, not 79%. D47's qualitative
conclusion survives -- Benders remains far behind -- but it is further behind than
recorded, against a baseline 24x slower than claimed. Both directions must be restated
wherever those numbers appear.

### 5. A gap field that means different things on different runs

The 947 s run printed `gap=0.117%` while its own bounds give `(UB-LB)/UB = 0.00007%`. The
printed figure is exactly the ABSOLUTE difference (1658.86 - 1658.8588281 = 0.0011719),
yet the earlier 180 s run's `5.640%` was correctly RELATIVE. The same field therefore
carries different semantics on different runs.

It does not affect the optimum recorded above, which is pinned by `term=optimal` and the
agreeing bounds. **No gap from this reporter should be quoted until it is understood.**
Not fixed here: it sits in the reporting path of the reference instrument, and this round
has already shown what happens when a number in that path is trusted without checking
(4190.74, D50). Same family, same discipline.

---

## D55 — Aggregation error survives a 10-minute grid, and scenario averaging cuts it by six

Date: 2026-08-14. Answers falsifiers 2 and 3 of `RESEARCH_NOTE_v2.md` §9 — the two that
could have collapsed the multi-resolution result — and closes §10 items 1–3. Scripts:
`make_instances.py`, `sweep_multiresolution.py`, `multiscenario_check.py`.

### 0. The generator artifact from D54 is fixed first

`burst` windows sat at minutes 60/180/420, exact multiples of 30, so under the `start`
convention every departure landed on a burst and slot reasoning was accidentally correct.
Windows now sit at minutes congruent to 7 modulo 30, 15 **and** 10, with spike spacing 47
(coprime to all three). Zero arrivals fall on a 30-minute boundary. An instance set for a
resolution study must not be in phase with any resolution it will be studied at.

### 1. Falsifier 2 — REFUTED. The effect is not an artifact of 30-minute slots

Gain from minute-level valuation, `midpoint`, Q=2, `p_minutes = 56`, all solves proven
optimal:

| shape | slot 30 | slot 15 | slot 10 |
|---|---:|---:|---:|
| flat | 12.43% | 5.52% | 11.08% |
| commuter | 10.48% | 2.45% | 3.07% |
| bimodal | 8.19% | 4.26% | 6.07% |
| burst | 50.09% | 19.62% | 20.74% |
| spiky | 25.23% | 18.52% | 22.39% |

Under `end`, same order: 12.53/16.60/16.23, 6.26/10.07/7.84, 14.02/7.43/10.22,
46.22/44.62/37.62, 19.06/41.33/36.78.

At a 10-minute grid — finer than most operational studies use — the correction is still
worth **3–22%** under `midpoint` and **8–38%** under `end`. The finding is about
aggregation, not about 30 minutes specifically.

**A correction to a reading made from partial data.** With only the 30 and 15 columns the
trend looked like a clean halving, and that was written down as "aggregation error shrinks
with finer slots". It is **not monotone**: 15 is a dip and 10 comes back up on four of the
five shapes. Two points were not enough to name a trend, and naming one was premature.

**What refining the master does buy** is large in absolute terms. On `spiky` the
slot-optimised cost falls 9906 → 6399 → 5539 as the grid refines. A finer first stage
genuinely improves the schedule — it just does not close the valuation gap. The two are
independent levers, which is the point.

**Why the mechanism survives, stated so it can be attacked.** A 60-minute wait cap spans
six 10-minute slots, and the placement offset is still 5 minutes (`midpoint`) or 10 (`end`)
against realised waits of 15–25 minutes. The grid got finer; the quantities it distorts did
not shrink proportionally.

**Instrument note.** Under `end` the gain does not fall monotonically with resolution
because `end`'s offset **is** `delta` — refining the grid refines the offset at the same
time, so the two effects cannot be separated. `midpoint` is the cleaner instrument for a
resolution study; `end` is the better operational assumption. They answer different
questions and both are reported.

### 2. Falsifier 3 — LARGELY CONFIRMED. Scenario averaging cuts the gain by about six

One first stage serving four scenarios whose structure sits at different minutes
(commuter, bimodal, burst, spiky), slot 30, `midpoint`:

| scenario | A slot-opt | B minute-opt | gain |
|---|---:|---:|---:|
| commuter | 10 147 | 10 069 | 0.77% |
| bimodal | 9 572 | 10 183 | **−6.38%** |
| burst | 9 259 | 8 484 | 8.37% |
| spiky | 9 435 | 8 201 | 13.08% |
| **averaged** | **9 603** | **9 234** | **3.84%** |

Against 8–50% when a schedule is tailored to a single scenario, averaging brings it to
**3.84%** — roughly a sixfold attenuation. On `bimodal` the minute-optimised schedule is
actively **worse**, which is what a compromise schedule does to individual members of the
set it compromises over.

The effect is not eliminated: 3.84% is positive and comes from proven optima on both sides.
But **the honest figure for the stochastic setting this project actually targets is ~4%,
not double digits.** Every Fase 1 config is four-scenario. Single-scenario gains must not
be quoted as if they were the operational result.

### 3. Status of the falsifier list after this

- **1** (vanishes on uniform demand): not vanished — `flat` gives 5.5–12.4%, the lowest
  band but nonzero.
- **2** (artifact of 30-minute slots): **refuted**, §1.
- **3** (multi-scenario averaging): **largely holds**, §2. The result must be restated at
  ~4% for multi-scenario.
- **4** (conventions disagree on a good instance set): `midpoint` and `end` agree in sign
  and rough magnitude across five de-aligned shapes and three resolutions. `start` remains
  the outlier and remains the physically implausible one.

### 4. What is still not measured

RQ5 as the operator reframed it — **is decomposed minute-level Benders faster than the
minute-level MONOLITH?** — is the comparison that matters now, and it has not been run.
D52 compared the decomposition against a *slot* recourse; D54 corrected the *slot*-monolith
baseline. Neither is the right baseline for a minute-level method, and no timing against the
right one exists. **Nothing in this repository supports a speed claim in either direction.**

### 5. Implementation note

`attach_minute_recourse` now takes an optional `scenarios` list and treats a single scenario
as the one-element case, rather than gaining a second multi-scenario copy. Capacity rows are
per `(scenario, slot)` against the same `Yout`/`Yret`, so the shared first stage is what
makes it one recourse problem instead of N separate ones, and E2 still holds per scenario.
Written in place deliberately: a duplicated construction is exactly how the two `cplex_log`
parsers (D50) and the three `Wmax` conversions (D53) drifted apart.
