# BENDERS_SPEC.md (v3) — Implementation contract for the MOB-AUTO2 Benders code

**Supersedes v1.** Every change below is grounded in an actual code trace of this session
(`master_impl.py`, `solver.py`, `core.py`, `subproblem_impl.py`, `config.py`, `app.py`, `cli.py`,
`types.py`, `tolerances.py`, `cplex_log.py`), not inference from configuration keys alone.
Companion documents: `docs/decisions.md` (D1–D13, the frozen/decided items this spec assumes) and
`AUDIT_v3.md` (the full findings list, ordered by required action).

**Scope decision carried through this whole document:** the elastic, minute-level subproblem
relaxation (`solve_refined_lp_relaxation_cut` / `solve_refined_subproblem`) is **out of scope**,
per D9. Every formula and constraint below describes the slot-only model
(`solve_subproblem` + `solve_mw_dual` + the master), which is the one going forward.

---

## 0. Non-negotiables (unchanged from v1)

1. No magic numbers — every parameter from config, with a stated unit (slots vs minutes vs km).
2. Model building is pure — no I/O, no solver calls inside `build_master`/`build_subproblem`.
3. Solver access through one adapter.
4. Every run reproducible: manifest with git commit, config hash, seed, solver version, W_max and p
   used (per D2/D3, these are swept — the manifest is how a table stays traceable to its parameters).
5. Every claim in the paper has a test.
6. Gapped runs never reported as optima without a marker.
7. **New:** every reported number states which subproblem mode produced it
   (`enable_temporal_refinement` should read `False` in every manifest from this point forward,
   per D9; a manifest reading `True` is from before the isolation and must be re-run before use).

---

## 1. Frozen and live parameters (Gate 0)

Two categories now, not one — this is the single most important change from v1.

### 1.1 Frozen (one value, no variants going forward)

| Symbol | Meaning | Frozen value | Source |
|---|---|---|---|
| `ρ` | charging rate | **70 km-eq/h, linear** | D1 |
| horizon | operating day | **10 h**, with a one-slot boundary buffer built in by design | D6 |
| `ε` | departure regularisation | **0.01**, in the same unit as the waiting objective (100 trips ≈ 1 unit of wait reduction) | D4 |
| units inside the solver | **slots**, throughout — `p`, `ε`, `(τ−t)` all slot-counted; minute conversion happens only in the reporting/simulation layer | D7, D8 |
| `enable_temporal_refinement` | subproblem mode | **False** (flip the Python default in `evaluate()`; not yet a schema key) | D9 |

### 1.2 Live sensitivity parameters (report the value used on every table; sweep before publication)

| Symbol | Meaning | Starting value | Sweep | Source |
|---|---|---|---|---|
| `W_max` | max admissible wait | 60 min (chosen for solver speed, not modelling reasons) | `{15, 30, 45, 60}` | D2 |
| `p` | unmet-demand penalty | to be decided — "quanto vale rejeitar um passageiro?" is open | `{30, 50, 100}` | D3 |

**On the dominance threshold (formerly "W_eff = min(W_max, p)"):** withdrawn as a single closed-form
number. The true trade-off is load-dependent — delaying a departure by `w` slots to catch one more
passenger delays every passenger *already on that departure* by `w` each, and the slot-aggregate
objective `(τ−t)·x[t,τ]` already prices this correctly since `x[t,τ]` is the full count on that arc.
Do not restate a single W_eff number in the paper; instead show the (p, W_max) sweep results
directly (D5).

### 1.3 Structural parameters (from the actual model, not swept)

| Symbol | Meaning | Value |
|---|---|---|
| `Q` | fleet size | 2 (baseline), swept 1–5 per §C3 of the framework doc |
| `S` | seats per shuttle | 15 |
| `E_max` | battery, km-equivalent | 150 |
| `c_trip` (`L`) | one-way consumption | 30 km |
| `τ_trip` | one-way duration incl. dwell | 30 min (converted to `trip_slots = ceil(trip_duration_minutes / slot_res)`) |
| `δ` (`slot_resolution`) | slot length | 30 min baseline, refine toward 1 min as the decomposition scales (D6) |

---

## 2. Mathematical specification — the model actually in production after D9

### 2.1 Sets and indices
Unchanged from v1 §2.1: `q ∈ Q`, `t, τ ∈ T`, `d ∈ {OUT, RET}`, scenarios `s ∈ S`.

### 2.2–2.4 Master
**Unchanged from v1** (§2.2–2.4: variables, objective, constraint blocks C1–C7). This part of the
original spec was already consistent with the code as traced (`master_impl.py`), modulo the
hardening items in §3 below.

### 2.5 Subproblem — corrected to match the actual layered structure

The earlier canonical spec used a single aggregate capacity `C[τ,d] = S·Σ_q y_d[q,τ]`. The real
implementation (`solve_subproblem`, `SPParams`) is a **layered** model: at each `τ`, there are
`K_out[τ]` (resp. `K_ret[τ]`) anonymous vehicle-layers `k`, each with its own capacity constraint.
This is a refinement, not a simplification — keep it.

```
Arcs[d] = { (t,τ) : t+1 <= τ <= min(T-1, t + W_slots) }     # W_slots = ceil(W_max_minutes/δ)
Layers_d[τ] = { 0, ..., K_d[τ]-1 }

min  Σ_d Σ_{(t,τ)∈Arcs[d]} Σ_{k∈Layers_d[τ]} [ (τ-t) + fill_eps·k ] · x_d[t,τ,k]
     + p · Σ_d Σ_t u_d[t]

s.t. Σ_τ Σ_k x_d[t,τ,k] + u_d[t] = R_d[t]                          ∀ d,t     (dual α_d[t], free)
     Σ_t x_d[t,τ,k]           <= min(S, C_d[τ])                    ∀ d,τ,k   (dual π_d[τ,k] >= 0)
     x, u >= 0
```

- `fill_eps` (`fill_first_epsilon`) is a genuine, negligible tie-breaker (default 1e-6) that biases
  packing into lower-numbered layers first. It is **not** a modelling distortion — confirmed by
  reading the objective construction directly (§ audit H1, corrected).
- `unused_capacity_penalty` from the config schema is **not used anywhere in the objective**.
  It is read from YAML, placed in the params dict, and never consumed by `solve_subproblem` or
  `solve_refined_*`. Either wire it in with a stated purpose, or remove it from the schema — do not
  leave it silently inert (audit H1).
- `(τ-t)` is in **slots**, matching D8. Do not introduce a `×δ` conversion inside this LP.

**Duals used for the cut:**
```
pi_OUT[τ] = Σ_k π_OUT[τ,k]        pi_RET[τ] = Σ_k π_RET[τ,k]      # per-slot only, summed over layers
```
These are **never indexed by `q`** in this model — there is no per-vehicle capacity constraint here,
only per-slot, per-layer ones. This is what makes the master's per-slot cut aggregation valid (§3).

### 2.6 Cut construction — Magnanti–Wong path (the default) and the plain-dual fallback

**Default path (`use_magnanti_wong: true`):** `solve_mw_dual` solves an auxiliary LP over the *same*
layered structure, selecting Pareto-optimal per-slot multipliers `dm_out[τ], dm_ret[τ]` on the
optimal face (`dual_obj_expr == ub_base`, where `ub_base` must come from `solve_subproblem`'s
objective — this only holds once D9 is applied; before that, mixing the refined model's objective
with the plain layered dual LP's optimality-face constraint is a latent inconsistency, moot now).

**Broadcast to per-vehicle coefficients:**
```
coeff_yOUT[(q,τ)] = dm_out[τ]      for every q      # identical across q, by construction
coeff_yRET[(q,τ)] = dm_ret[τ]      for every q
```
This broadcast is what makes `master_impl.py`'s `aggregate_cuts_by_tau: true` (keep-first-`q`)
**safe** on this path — the coefficients genuinely are equal across `q`, not silently collapsed from
unequal ones. **Add an assertion in `master_impl.py`'s `_add_cut`** verifying this equality before
aggregating, so any future change (e.g. reintroducing the elastic path, or a per-vehicle capacity
model) that breaks the invariant fails loudly instead of silently producing an invalid cut (D10).

**Fallback path (`use_magnanti_wong: false, use_dual_slopes: true`):** same broadcast, using
`solve_subproblem`'s native `pi_OUT[τ]`/`pi_RET[τ]` directly (`dm_out[τ] = S·pi_out[τ]`) instead of
the MW-selected ones. Valid lower bound, not Pareto-optimal — expect slower convergence, useful as a
baseline/ablation.

**Do not use:** the plain finite-difference fallback (`use_magnanti_wong: false,
use_dual_slopes: false`) sets `cut_lb_valid = False` in the code — its cuts are **not** provable
lower bounds, and `solver.py` disables gap/optimality reporting when this mode is active. Only use
it as a last-resort diagnostic, never for a reported result.

### 2.7 Master aggregation — validated, hardened

Master's `_add_cut` aggregates coefficients by `τ` (keeping one value per slot instead of per
`(q,τ)`) whenever `aggregate_cuts_by_tau: true` (the default). Per §2.6 this is now a **safe**
simplification on the production path, not the correctness bug identified earlier in this project's
audit history. Required hardening, not optional:

```python
# before collapsing to one value per tau:
assert all(abs(v - first_v) < eps for v in values_at_this_tau), \
    "coefficient varies across q at fixed tau — aggregation invalid, check cut-generation mode"
```

Also remove the **re-anchoring of the cut constant** (`const_adj = ub_est - contrib`) as an
unconditional step — keep it only as a **consistency check that raises** if the reconstructed
constant doesn't already match, not as a silent correction. If it doesn't match after the assertion
above passes, that's a real bug to find, not to paper over.

### 2.8 Algorithm
Unchanged from v1 §2.9, with one addition: **Gate 1's sign-convention verification is empirical, not
purely algebraic** (D13). Run the isolated (D9) model with logging and grep for `[CHECK FAIL] MP
total exceeds SP total` / `[CHECK FAIL] LB exceeds UB` across the core instance set. A clean run
across all instances is the acceptance criterion for closing this item — do not attempt to close it
by further static hand-derivation of the dual signs from source alone.

---

## 3. Repository layout and refactor plan — reduced scope after D9

The v1 refactor plan (config extraction, solver adapter, model split, sign unification, unit fix,
instrumentation) stands, with these adjustments:

1. **Step 0 (new, first):** flip `enable_temporal_refinement` default to `False` in
   `subproblem_impl.py::evaluate()`. One line. Do this before anything else in this list — every
   subsequent step's "before/after" comparison should be against the isolated model, not the mixed
   one.
2. **Step 0.5 (new):** decide the fate of the ≈800 lines exclusive to the elastic path
   (`solve_refined_lp_relaxation_cut`, `solve_refined_subproblem`, `_proxy_cut_from_nominal_lp`,
   `class RefinedServiceEvent`, the minute/slot conversion helpers used only there). Either delete,
   or move to `experimental/` with `enable_temporal_refinement` promoted to a documented,
   off-by-default schema field. Do not leave it live-but-unreferenced in the main module — it is
   the single largest source of audit surface for a part of the system that is out of scope.
3. **The "unit fix" step from v1 is withdrawn** — D8 confirms the code was already consistent in
   slot units; there is nothing to fix there. Remove that line item from the refactor plan.
4. **The "sign unification" step is now scoped to empirical verification** (D13) rather than a pure
   rewrite — add the logging-based check as a required CI/test step (§6), not just a manual review.
5. Everything else (config extraction, solver adapter, indexed constraints instead of per-cell
   `add_component` loops, narrowed exception handling, diagnostics-off default) stands as in v1.
6. **New cleanup items found this round:**
   - `subproblem_impl.py` defines `_slot_idx_from_minutes` twice (lines ~132 and ~163); the second
     silently shadows the first. Remove the dead one.
   - Two symmetry-breaking implementations coexist in `master_impl.py` (`use_fifo_symmetry` and
     `symmetry_breaking`), encoding the same ordering. Keep one.
   - `benders/core.py::CorePoint` appears unused — the active core-point logic is inline in
     `BendersSolver.run()`. Delete `CorePoint` once confirmed unreferenced, or wire it in
     deliberately (D12).
   - `cplex_log.py`'s regex-based CPLEX log parsing is a fragile last-resort bound-recovery path;
     document it as such and add a test fixture with a sample log so format drift is caught.

---

## 4. Config schema follow-ups

- `unused_capacity_penalty` is validated, parsed, and threaded through to the subproblem params dict
  in `app.py`, but never read by any model-building code. Resolve one way: wire it into the
  objective with a stated purpose, or delete it from `SubproblemSection` and `app.py`'s
  `_prepare_params`. Leaving a schema-validated, silently-inert parameter is a trap for the next
  person who sets it expecting an effect.
- `enable_temporal_refinement` is not in the schema at all (§3, Step 0.5) — decide whether it
  becomes a first-class field or the elastic code is removed entirely.

---

## 5. Test suite — additions to v1 §6

- `test_coeff_q_invariance`: for the production cut path (MW or plain-dual, both with
  `enable_temporal_refinement: false`), assert `coeff_yOUT[(q,τ)]` is identical across all `q` for
  every `τ` in the returned cut — this is the hardening from §2.7, tested directly rather than only
  asserted at runtime.
- `test_no_check_fail_in_log`: run the core instance set with logging enabled; assert no
  `[CHECK FAIL]` line appears in the solver log (D13 — the empirical Gate 1 closure).
- `test_fdiff_mode_disables_gap_reporting`: confirm that when neither MW nor dual-slopes is enabled,
  `BendersRunResult` does not claim `SolveStatus.OPTIMAL` from a gap check (guards against silently
  trusting a non-valid-lower-bound cut mode).
- `test_unused_capacity_penalty_has_no_effect` (until §4 is resolved): document current behaviour
  with a regression test, so resolving §4 is a deliberate, visible change rather than an accidental
  one.
- Everything in v1 §6 that referenced the aggregate `C[τ,d] = S·Σ_q y[q,τ]` capacity model should be
  re-read against §2.5's layered `K_d[τ]` structure — the tests still apply, the fixtures generating
  synthetic instances need the layer counts, not just aggregate demand.

---

## 6. What did not change from v1

Master constraint blocks C1–C7, the Benders algorithm skeleton (§2.9/2.8), the experiment matrix
(§5 of v1), the report framework (Parts I–III of `Final_report_framework_v2`), and Gates 2–5 all
still apply as written. This document only corrects the subproblem-side mathematics and the
parameter-freezing decisions; it does not reopen anything already settled in the master formulation.
