# Code audit v4 — `19_MobAuto2_Benders`

**Supersedes `AUDIT_v3.md`.** Companions: `BENDERS_SPEC_v4.md` and `docs_decisions.md`
(D1–D30).

v3 was a map of where to look. This document records what was actually found when the
code was traced, changed and **measured**, and reorients the backlog: the correctness
phase is closed, the work from here is improvement.

Severity: **C** = breaks correctness of reported results · **H** = affects numbers or
their interpretation · **M** = exposition/maintainability · **L** = cosmetic.

**Method note that matters for reading this document.** Every claim below is backed by a
run, a diff or a timing, not by reading. Five of v3's conclusions were contradicted by
measurement (§4). Where v3 said "worth doing for that reason alone", the reason was
usually wrong. Treat unmeasured claims here — there are two, marked — with the same
suspicion.

**Merged as** `dbc01e2` on `main`, eleven commits, branch
`d9-remove-elastic-subproblem`.

---

## 0. Headline

> **All lower bounds below are void (D30).** The subproblem's constraint set moved with
> the master's `y`, so no cut it produced was a valid lower bound. Every LB, gap and
> `status = OPTIMAL` in this document predates the fix and must be re-measured. Upper
> bounds, feasible schedules and served-passenger counts are unaffected: they come from
> pricing an exhibited schedule, which was always correct — verified to the cent against
> an independent monolithic MILP at three scales.
>
> The optimum of the reference instance is **4183.24**, not 4190.74.

**Superseded headline, kept so the claim is not quoted again:**

```
status = OPTIMAL      LB = 4186.570873      UB = 4190.740015
gap    = 4.169 absolute,  0.099485% relative   (tolerance 0.1%)
served = 222/300      average wait 39.19 min
42 iterations, 38 cuts, 324.75 s
```

Instance `setups/base.yaml` (300 requests), Q=2, T=22 slots of 30 min, W_max=60 min,
p=50, S=15, E_max=150, L=30, ρ=70 km-eq/h, ε=0.01, concurrency_penalty=0.25.
Provenance: `manifests/manifest_mw_convergence_20260806_041026.json`.

~~The optimum is 4190.74 to within 0.1%.~~ **False.** The optimum is at most
**4183.24**, and the 4186.57 lower bound that produced this claim was above it (D30).

**Consequence for anything already written up:** every lower bound and every optimality
gap this codebase produced before `dbc01e2` is void, for two independent reasons (C3,
C4). Upper bounds, feasible schedules and served-passenger counts were never affected —
those remain usable. Any table carrying an LB or a gap must be regenerated.

---

## 1. Correctness defects found and fixed

Three defects had to be fixed before any bound could be trusted. **Each masked the
others**: fixing MW alone produced bounds that looked plausible and were still invalid,
which is why the first "converged at 0.94%" result was reported and then withdrawn.

### C3 — `solve_mw_dual` never succeeded *(C, not in v3)*

The Magnanti–Wong dual was built with `pi` declared `NonNegativeReals`. For a `<=`
constraint in a **minimisation** the dual is **non-positive** — which is exactly what
Pyomo/CPLEX return and what `solve_subproblem` already relied on, computing
`dm = +S*pi` to obtain the negative slopes the master expects. Both dual-feasibility
inequalities and the u-variable constraint had been flipped to match the wrong sign.

The consequence was unboundedness, not infeasibility: raising `a` and `pi` together
satisfied `a + pi >= cost` while increasing the objective, so only `OptFace` could bound
them — and at the all-idle first iteration every capacity is zero, so `OptFace` did not
constrain `pi` at all. CPLEX reported `infeasibleOrUnbounded` from iteration 1.

`solve_mw_dual` therefore returned `None` on **every call** (10/10 single-scenario,
20/20 multi-scenario), and every cut came from the finite-difference fallback, which is
not a provable lower bound.

Two further defects in the same function:

- The MW objective maximised `sum(dm*Ybar)`, omitting the `-y_inc` term. The cut is
  tight at the incumbent, so its value at the core point is
  `ub_base + sum dm*(Ybar - y_inc)`; `ub_base` is pinned by `OptFace`, so the
  Pareto-optimal dual is the one maximising `sum dm*(Ybar - y_inc)`. The dropped term is
  not constant across the optimal face, so the wrong dual was being selected.
- `OptFace` was a float equality against a separately computed primal optimum, which a
  few ulps of disagreement would render infeasible. Weak duality gives
  `dual_obj <= ub_base`, so `>= ub_base - tol` carves the same face without the
  brittleness.

**Fixed** in `1b8fdb3`. Effect: cut generation over 10 iterations went **23.90 s → 0.39 s
(61×)** and total runtime 46 s → 9.5 s, because finite differences solve `2*T` extra LPs
per iteration against one LP for MW.

**How it hid for so long:** the fallback was silent (C5) and the mode label was wrong
(H7), so a log showed `mode=dual lb_valid=True` while running finite differences.

### C4 — symmetry breaking removed feasible schedules *(C, understated in v3 as H4)*

Both symmetry implementations imposed cumulative ordering at **every time prefix**,
`cum[k][t] <= cum[k-1][t]`. That is strictly stronger than ordering vehicles by total
trips and is invalid **even for a homogeneous fleet**.

Counterexample. Vehicle A makes one trip at t=0; vehicle B makes trips at t=1 and t=2.
Labelled (A,B): `cum[0]=[1,1,1]`, `cum[1]=[0,1,2]` violates at t=2. Labelled (B,A):
`cum[0]=[0,1,2]`, `cum[1]=[1,1,1]` violates at t=0. No relabelling satisfies it, so a
legitimate schedule is excluded — it is not a symmetric duplicate.

The master therefore was not a relaxation and its bound could exceed the true optimum.
Observed directly: a master `best_bound` of ~4558 alongside a known feasible solution of
4228.99.

**Found by** checking convergence after what should have been a no-op deduplication: two
mathematically equivalent models produced **disjoint** bound intervals,
`[4189.12, 4228.99]` and `[4476.07, 4515.99]`. Diffing the first-iteration LPs confirmed
the only difference was the 23 redundant rows, so a bound had to be wrong.

**Fixed** in `b0ed6bf`: ordering by **total** departures, valid for a homogeneous fleet
(relabel by sorting on total trips), needing `Q-1` rows instead of `T*(Q-1)`. This is the
constraint the deduplication step had removed as "redundant" — it is not redundant, it is
the only correct one of the two. The homogeneous-fleet precondition v3 asked for is now
enforced and raises with the offending values named.

Row count: 45 → 1 at Q=2, T=22.

### C5 / N6 — the MW fallback claimed a valid lower bound *(C, not in v3)*

`cut_lb_valid` was set to `not temporal_refinement` before the dispatch and never reset
when the MW branch fell back to finite differences. The solver then reported gaps and
optimality from cuts that cannot support them. `BENDERS_SPEC_v3` §2.6 claims the
finite-difference fallback "sets `cut_lb_valid = False`" — true only of the top-level
`else`, not of this nested one.

**Fixed** in `a423058`. Fixing this is what exposed C3.

### C2 — anti-trivial idle cut *(C in v3, resolved by deletion)*

`_build_anti_trivial_cut` added `Σ(yOUT+yRET) >= min_total_starts`, a hard constraint on
the master rather than a duality-derived cut.

v3 states it "fires from the shared `evaluate()` scaffolding regardless of which
subproblem mode is active. **Not addressed by D9** — still open." **This is incorrect.**
Both call sites were nested inside `if temporal_refinement and (not cut_lb_valid) and
_is_degenerate_cut(...)`, so the flag flip alone made it unreachable.

Confirmed empirically: zero `anti_trivial` occurrences across both baselines, including
at k=1 where the degenerate all-idle candidate does occur (`rec_total=15000`) and is
dislodged by an ordinary optimality cut (k=2, `served=181/300`). No replacement initial
θ bound was needed.

**Removed with the elastic path** in `a423058`.

### H7 / N7 — the cut generator was not reportable *(H, not in v3)*

`cut_mode_used` was assigned **before** the dispatch as
`("dual" if use_dual else "finite_difference")` and never updated by the branch that
actually ran, so MW runs were labelled `dual`. In multi-scenario runs the aggregated
diagnostics omitted the field entirely and logs read `mode=-`.

This breaks non-negotiable 7 — every reported number must state the mode that produced
it — and it is a direct enabler of C3: a log could read `mode=dual lb_valid=True` while
the code ran finite differences and the bound was invalid. It also made the D11/H6
θ-disaggregation A/B unmeasurable, since the two arms could not be distinguished in
output.

**Fixed** in `a423058`: each dispatch branch labels itself (`mw`, `mw_fdiff_fallback`,
`dual`, `finite_difference`), and the multi-scenario aggregate reports a single label
when scenarios agree or `mixed(a+b)` otherwise — never blank.

Confirmed sound and deliberately unchanged: the aggregate combines *validity*
conjunctively (`all(...)`, defaulting to false on an empty set). That is the correct
semantics and must not be "simplified".

### H8 — the report described a different solution than it quoted *(H, not in v3)*

The end-of-run report drew the timeline from the incumbent snapshot but `theta` and the
per-shuttle passenger table from the **live model**, i.e. the last master solve. Where
the two schedules disagreed, `k_out = 0` and those passengers were silently dropped from
the table.

Observed: a table summing to **126** printed directly beneath `Pax served: 222/300`, and
`theta = 4560` next to `UB_total (best): 4228.99` — a θ above the upper bound, impossible
for the solution being described.

Three causes: the candidate carried no `__theta` under directional disaggregation so
`format_solution` fell through to live values; `_map_layers_to_shuttles` read
`m.yOUT[q,tau].value`; and the single-scenario branch split passengers **equally** among
departing vehicles, a fiction that made both shuttles show identical loads.

**Fixed** in `bf504d5`, plus a cross-check that warns when the table does not sum to the
reported total — the old code printed a correct total beneath an incorrect table, which
made the table look validated.

**This is the one that would have reached the thesis**: the per-shuttle schedule is
exactly the kind of figure that gets published.

---

## 2. Items from v3 — disposition and measured result

### Executed as recommended

| v3 item | Result |
|---|---|
| **D9** flip + delete elastic path | 3267 → 1828 lines (−44%). Done in two verified steps; output **byte-identical** on both scenario branches after each. The flag also gated `mw_enabled`, `use_dual_slopes` and `cut_lb_valid`, so flipping it was a correctness fix, not a tuning choice. The elastic block was ~1190 lines, not v3's ~800 estimate — N4 missed the exact-arrival plumbing and the nested helpers |
| **C1/D10** aggregation hardening | Both invariants hold on the production path. The constant re-anchoring was provably a no-op, as `[CUT TIGHTNESS] diff=0` had already indicated; it is now a raising check. Note the check must allow for `cut_coeff_threshold` dropping small coefficients — a naive "delta must be zero" fires spuriously (up to ~0.044 at Q=2, T=22) |
| **D13** sign convention | Closed empirically: zero `[CHECK FAIL]`, `raw_pos_dm=0` across all runs. Now automated |
| **H5/D12** dead `CorePoint` | `benders/core.py` deleted entirely — it contained only `CorePoint`, imported and never instantiated |
| **M2** charge-before-idle | Gated behind `master.charge_before_idle`, default on. Put under `master:` rather than a new `policy:` section, since `use_fifo_symmetry` and `symmetry_breaking` already live there and are the same kind of canonicalisation |
| **M5** diagnostics default | `run.emit_reports`, default **false**. Previously derived as `log_level != "REPORT"` — on by default and disabled only by a log level that reads as though it would enable reports. A 10-iteration run left 20 files behind, in a Drive-synced folder. Removing the I/O alone took the baseline 22.4 s → 12.5 s |
| **N1** duplicate helper | Hoisted. **v3's mechanics were wrong**: the two copies are in mutually exclusive branches each ending in `return`, so neither shadowed the other — duplication, not dead code |
| **§4** inert parameters | `unused_capacity_penalty` deleted from all six sites and every config; results byte-identical, which *is* the proof it was inert. `concurrency_penalty` kept and documented |

### Executed, but v3's rationale was wrong

| v3 item | v3 said | Measured |
|---|---|---|
| **M4** indexed constraints | "real performance problem", "we get stuck because of these loops" | Master construction is **7.6 ms** against ~2.1 s per master solve — **0.04%**. Even a tenfold difference would be under half a percent. The cost is MIP solving as cuts accumulate. Kept as hygiene (and it merged two opposing inequalities into one equality, −42 rows), **not** as a performance fix |
| **H4** symmetry | "redundant; only valid for a homogeneous fleet" | The duplication was real but harmless; the **form** was invalid (C4). Fixing what v3 described would have left the defect in place — and the first deduplication attempt deleted the *only correct* constraint as redundant |

### Rejected on measurement

| v3 item | v3 said | Measured | Disposition |
|---|---|---|---|
| **M1** explicit `b >= L*yRET` | "tightens the LP relaxation, which is 85.7% of runtime, worth doing for that reason alone" | Master phase **18.2 s → 49 s (2.7×)**, reproduced at 50.298 s and 50.762 s with bit-identical results; bound at the same budget **worse**, LB 3034.77 → 2770.97 | **Not adopted.** Sound but does not pay for itself. A tripwire test and a comment at the `C5` site prevent re-adding it from the audit text without re-measuring |

The constraint is valid — the soundness check `LB <= 4190.74` passed with it in place. A
valid inequality cannot weaken the LP relaxation, but it can make the MIP harder and
leave a worse best-bound when the solve stops on gap or time.

### Partially executed

| v3 item | Done | Not done |
|---|---|---|
| **M3** exception handling | Bound-provenance path only: the gap computation narrowed to `(TypeError, ValueError)`, and `incumbent_source` / `best_bound_source` / `gap_source` now record `"unavailable"` rather than being silently absent | **231 bare `except Exception` handlers remain** (93 in `solver.py`, 73 in `master_impl.py`, 35 in `subproblem_impl.py`, rest elsewhere; the 165 quoted here in round 1 was never right). Narrowing them all is a large mechanical refactor where each edit can change behaviour and most merely wrap `float()` conversions. Deferred deliberately — see §5 |
| **N2** CPLEX log fixture | Five parsing tests, including that the gap is returned as a ratio not a percentage | Fixtures are **written from the format the parser expects, not captured** from a real run, because `emit_reports` is off by default. They pin the parsing contract but are a weaker drift detector |

### Not executed — these are the "improve" phase

| v3 item | Why not | Now |
|---|---|---|
| **H2/H3 → D2/D3** (p, W_max) sweep | v3 itself orders this last: "then, and only then". Until `dbc01e2` no bound was valid, so a sweep would have produced tables that had to be thrown away | **Unblocked.** §3 item 1 |
| **H6/D11** θ disaggregation A/B | Same reason. `theta_per_scenario: false`, so directional disaggregation is active and has never been compared against the alternative | **Unblocked.** §3 item 2 |
| **v1 §7** re-run matrix | Same reason | **Unblocked.** §3 item 1 |

---

## 3. Backlog — the improvement phase

> **Superseded in part by Fase 1 (D33–D37), 2026-08-06.** At Q=3 / T=44 / 4 scenarios /
> 300 s the master ends every solve at a 99.9% internal MIP gap and the Benders lower
> bound reaches 0.35 against a 785 target; 3.4x the per-solve time buys 1.2% of bound.
> No optimality gap may be quoted at Q>=3, and performance items below that assume the
> cut set is the binding constraint do not apply there. Reading rules: `README.md`,
> § Reading rules. The five configs that produced it ship as `configs/phase1/*.yaml`.

The correctness phase is closed. The ordering below is by value, and the first two items
are research output rather than code.

### 1. Run the (p, W_max) sweep and regenerate every table *(D2/D3, v1 §7)*

**Superseded in round 2.** `p = 50` and `W_max = 60` are now treated as **given inputs**,
not swept axes. The sweep was re-scoped to structural parameters — `slot_resolution` and
fleet size `Q` — and run at a fixed 120 s budget. Results and reading guidance in
`docs/sweep/README.md`.

Headline: fleet size dominates, because the objective is dominated by the unserved
penalty at `p = 50`. Q=2 is capacity-starved (3900 of its 4190.74 optimum is penalty);
Q=3 serves 290/300 and is the largest single step; Q=4 and Q=5 show clear diminishing
returns. Only the Q=1 cell converged, so the **UB column is what carries proof** — each
UB is an exhibited feasible schedule — and the LB column must not be quoted as a gap.

The 15-minute cell is not evidence that finer slots are worse; it needs a bigger budget.

Every table must still state the `(p, W_max)` pair **and** `concurrency_penalty`, which is
active in the objective and absent from the published formulation. The manifest records
all three, so this is mechanical rather than a matter of discipline.

Do not restate a single `W_eff` threshold (D5): show the sweep.

### 2. θ disaggregation A/B *(D11/H6)*

Run `theta_per_scenario` true and false on the same instance and compare iteration
count, wall time and final gap. D11 is explicit that this cannot be settled by reading.
Now that runs converge, "final gap" is a meaningful axis for the first time.

### 3. Convergence performance *(open, measured, no longer blocking)*

Converges in 42 iterations / 5.4 min. Three known inefficiencies, in the order I would
test them:

- ~~**`_cand_theta` always returns `None`**~~ **— fixed, no gain. Closed (D21).** The
  defect was real: no return path for a non-`None` value, so the early exit that skips
  cut generation never fired. Restored and measured before/after on
  `configs/mw_convergence.yaml`: **identical** — 42 iterations, 38 cuts,
  LB 4186.570873, UB 4190.740015, and `skip=None` still on every cut-generating
  iteration. The guard is `θ ≥ UB(y) − ε` and the master's θ rises toward the
  subproblem value **from below**, so it can only hold at convergence, where the loop
  stops anyway. Fix kept (a helper returning `None` for all input is a trap), but this
  was not a performance lever and should not have been listed first.

  *Correction to the claim above about an unreachable tail inside
  `_candidate_is_all_idle`: that function does not exist anywhere in the tree, and there
  is no orphaned fragment. The helper simply falls off its end.*
- **Master mipgap 0.05** caps how tight the LB can get per iteration. v3's framing of
  this (as old task #7) was drawn from runs whose LB was not a lower bound; the mechanism
  is real but must be re-measured against the corrected model.

  **Round 2 found this was not testable at all**, then tested it. The Benders loop
  overwrote both the master's mipgap and its per-iteration time limit on every iteration
  from five hardcoded constants, so `master.mipgap` in a config was parsed, validated and
  discarded — the same defect class as D19. Both are now the schedule's ceilings (D22).

  **Closed. Measured, reproducible, and the lever is weak (D28).** At 6 fixed iterations
  with the limit never binding, mipgap 0.05 → 0.001 buys **+5.2%** of LB for double the
  master time. Spending the same effort on iterations instead buys **+78.5%**. Default
  stays at 0.05.

  The per-iteration *time limit* is not a lever at all and cannot be one: a cap that binds
  is nondeterministic by construction — three runs of one config gave LB 2333.29, 2153.79,
  2175.87, while a non-binding cap reproduced to the last digit. It must stay generous,
  which `configs/baseline_d9.yaml` already required (D26).
- **MW core point decays toward the boundary**. Observed at ×0.7 per iteration
  (`0.7 → 0.49 → … → 0.0282`), consistent with an EMA at α=0.3 pulling toward an
  incumbent of 0. Magnanti–Wong needs a point in the **relative interior**; once clipped
  at `eps` it sits on the boundary and the Pareto selection degrades. **Caveat: this was
  measured while MW was silently failing**, so the core point was being fed to a function
  that never succeeded. Re-measure before acting.

### 4. Silent demand truncation *(H, not in v3)*

`_aggregate_requests` filters with `if not (0 <= t < Tlen): continue`, discarding any
request outside the horizon with no warning or count. `setups/demo_cont_demand.yaml`
declares 284 requests with times up to minute 830; at `T_minutes=660` only **224** are
counted — 60 requests, 21%, vanish. The log then reports `Pax served: 173/224`, which
looks correct and understates true unmet demand, a headline metric weighted by `p`.

Surfaced by a fixture pairing a 660-minute horizon with an 830-minute instance;
`setups/base.yaml` maxes at minute 598 and fits, so **this is a latent trap, not evidence
that existing results are wrong**. Fix: count and warn, or raise.

Relevant to D6: the horizon is to extend from 10 h to 24 h.

**Closed in round 2 (D25).** Three drop sites, not one — the list-of-dicts path, the
matrix path, and the direct-array path that truncates with `[:Tlen]`. All three now count
by cause and emit a `[DEMAND]` warning naming the horizon and the latest request minute.
Warns rather than raises, because `demo_cont_demand.yaml` legitimately exceeds a 660-minute
horizon. Confirmed numerically: 284 requests, 224 counted at 660 min, 60 discarded; at the
24 h horizon of D6 all 284 fit.

### 5. `use_dual_slopes` is inert *(N3, still open)*

The dispatch is `if mw_enabled: ... elif use_dual: ... else: fdiff`, and the MW branch
falls back to finite differences **internally**. With both `use_magnanti_wong: true` and
`use_dual_slopes: true` in `configs/default.yaml`, the plain-dual path is unreachable.

v3's N3 identified the precedence but attributed it to branch order alone; at the time
`temporal_refinement` disabled both upstream, so the ordering never even mattered. Now
that MW works, the precedence is what makes `use_dual_slopes` inert.

Per `BENDERS_SPEC_v4` §2.6 the plain-dual path is a **valid** lower bound, just not
Pareto-optimal, so it is the natural ablation baseline for the A/B in item 2. Either wire
the fallback to prefer it over finite differences, or document that it is
MW-or-nothing.

### 6. Remaining M3 hygiene

**231** `except Exception` handlers in `src/` — not 165. The figure in this document was
never accurate: the count was **225** at the round-1 merge `dbc01e2`, and round 2 added 6.
Distribution: `solver.py` 93, `master_impl.py` 73, `subproblem_impl.py` 35, rest elsewhere.

Not one problem but four, and treating them as one invites regression. Expected
categories: genuinely defensive around optional solver attributes (**keep**); swallowing
errors around required operations (**must raise or log** — this is exactly how
`solve_mw_dual` failing on every call stayed invisible); `try/except` used as control flow
where a `getattr` default is clearer; and handlers around code that cannot raise.

Census before edits. Two things now make this safe that did not exist in round 1: 49
tests, and a baseline log-diff protocol that works again (D27), so a cleanup can be proven
to change nothing the same way D9 was.

### 7. Capture a real CPLEX log for the N2 fixture

Run once with `run.emit_reports: true` and commit a captured log, replacing the synthetic
fixtures.

### 8. Inert configuration is this codebase's signature defect *(round 2)*

Not a task — a pattern worth naming, because round 2 found **three more** instances after
v3 found one, and every one of them was schema-validated, parsed, threaded down and then
discarded in silence:

| parameter | how it died | decision |
|---|---|---|
| `unused_capacity_penalty` | read by no model-building code | D19 (deleted) |
| `master.solve_time_limit_s` | overwritten every iteration by a hardcoded schedule | D22 (wired as ceiling) |
| `master.mipgap` | same schedule, same overwrite | D22 (wired as ceiling) |
| `master.cplex_options` `CPXPARAM_*` | skipped by an explicit `continue` under `cplex_direct` | D24 (translated) |

Two carried a documented rationale that was therefore false: `CPXPARAM_Threads: 1` claims
to buy reproducibility in the frozen baselines and never took effect, and
`CPXPARAM_MIP_Strategy_Symmetry` is not a CPLEX parameter name at all — the silence hid
the wrong name as well as the drop.

Two more are known and still open: `subproblem.use_dual_slopes` (item 5 above), and
`initial_actions`, which is passed to the subproblem and never read there.

**Schema validation is not evidence a parameter does anything.** The allow-list catches
typos, not dead wiring. Anything that reads a config value and then lets a later stage
overwrite or skip it must fail loudly or not accept the key. When a knob is claimed to
matter, the test is a measured A/B, not a grep that it is referenced.

---

## 4. Corrections to v3

Recorded so they are not reintroduced from the older document.

| v3 statement | Correction |
|---|---|
| C2 "fires regardless of which subproblem mode is active. Not addressed by D9 — still open" | Reachable only via the elastic path; the flag flip alone made it unreachable |
| H4 "redundant, and only valid for a homogeneous fleet (a precondition never checked)" | Understated. The **form** was invalid for homogeneous fleets too, and invalidated the master relaxation |
| M4 "author confirms this is a real performance problem" | Construction is 0.04% of master time. The perceived slowdown was MIP solving, and partly the finite-difference cut generation that MW replaced |
| M1 "worth doing for that reason alone" | 2.7× master time, worse bound. Rejected |
| N1 "the second silently shadows the first" | Mutually exclusive branches; duplication, not shadowing |
| N4 "≈800 of the file's 3266 lines" | ~1190 lines; the estimate missed the exact-arrival plumbing, the nested helpers and the `SPParams` fields |
| §2.6 "the plain finite-difference fallback sets `cut_lb_valid = False`" | Only in the top-level `else`. The MW fallback did not — that was C5 |
| H1 "`fill_first_epsilon` confirmed benign" | Still true, unchanged |
| §"85.7% of runtime in the master" | Not reproduced and probably not meaningful for the current code: before the MW fix, cut generation was 23.9 s of a 46 s run. Re-measure before quoting |

---

## 5. What is now guarded automatically

`python -m unittest discover -s tests` — **49 tests, ~10 s**, stdlib `unittest`, nothing
to install. 33 need no solver and run in under a second.

Each soundness assertion corresponds to a defect above, so the failure mode that produced
it cannot return silently:

| Test | Guards |
|---|---|
| `test_magnanti_wong_succeeds` | C3 |
| `test_lower_bound_does_not_exceed_a_known_feasible_objective` | C4 — the single inequality that exposed it |
| `test_cut_mode_is_reported_and_is_mw` | H7 |
| `test_per_shuttle_table_sums_to_served_total` | H8 |
| `test_no_check_fail_lines` | D13 |
| `test_gapped_run_is_not_reported_as_optimal` | spec §0.6 |
| `test_uses_total_ordering_not_prefix_ordering` | C4 structurally |
| `test_heterogeneous_fleet_is_refused` | the unchecked precondition |
| `test_c5_ret_not_present` | M1 being re-added from v3 |
| `test_q_varying_coefficients_are_rejected` | D10 |
| `test_unused_capacity_penalty_is_gone` | the inert-parameter trap |

Added in round 2:

| Test | Guards |
|---|---|
| `test_actions_pad_with_last_value_not_literal_idl` | D23 — the two per-vehicle lists diverging |
| `test_both_lists_pad_the_same_way` | D23 — pins them together, which is what failed |
| `test_old_master_time_limit_name_is_rejected` | D22 — a renamed key silently ignored |
| `test_old_solver_time_limit_name_is_rejected` | D22 — same, for the run budget |
| `test_schedule_ceilings_come_from_config` | D22 — the knob going inert again |
| `test_cpxparam_names_translate_to_direct_paths` | D24 — CPXPARAM dropped under `cplex_direct` |
| `test_shipped_configs_use_resolvable_names` | D24 — the wrong parameter name shipping unnoticed |
| `test_converged_reference_run_is_not_clock_truncated` | D26 — bounds becoming samples |
| `test_manifest_records_whether_the_run_is_reproducible` | D26 — the verdict not being archived |
| `test_manifest_marks_a_clock_truncated_run_as_not_reproducible` | D26 — flag following config not run |
| `test_requests_past_the_horizon_are_counted_and_warned` | D25 — silent truncation of the denominator |

And every run writes `manifests/manifest_<name>_<timestamp>.json` recording the git
commit (with a **dirty** flag), config sha256, seed, CPLEX version, `W_max`, `p`,
`concurrency_penalty`, which cut generator ran, whether its cuts support a bound, and —
added in round 2 — whether the run is bit-reproducible at all, with the count of master
solves that stopped on the clock and the CPLEX options that decide it (D26).

---

## 6. Method note

Five of v3's conclusions were wrong, and the two worst defects were not in it at all.
That is not a criticism of the exercise — v3 correctly identified *where* the fragile
code was, and every fix landed in a file it named. But its confident performance and
severity claims did not survive measurement.

Three of this round's findings came from checking a change that "should have been a
no-op":

- N6 was a two-line fix that immediately exposed C3.
- C4 surfaced because a pure deduplication produced disjoint bound intervals.
- H8 surfaced because a passenger table looked implausible.

The pattern worth carrying into the improvement phase: **a refactor that should change
nothing is a cheap probe for latent unsoundness**. Run it, then check an invariant, not
just the diff.

The corresponding failure to avoid: twice this round a symptom disappearing was read as
the problem being solved. Cut acceptance recovering after C4 was not convergence, and had
the long run not been made, "the stall is fixed" would have gone into this document
untested.
