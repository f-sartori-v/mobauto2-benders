# BENDERS_SPEC.md (v4) — Implementation contract for the MOB-AUTO2 Benders code

**Supersedes v3.** Companions: `docs_decisions.md` (D1–D30) and `AUDIT_v4.md`.

v3 described the model as it was believed to be. This version describes the model as it
**is**, after the corrections in `AUDIT_v4` §1 were traced, applied and measured, and
after the first converged run with valid bounds. Where v3 was wrong, the correction is
marked inline so the older text is not reintroduced.

**Scope, unchanged from v3:** the elastic, minute-level subproblem relaxation is out of
scope per D9. It has now been **deleted**, not merely disabled — 3267 → 1828 lines. Every
formula below describes the slot-only model (`solve_subproblem` + `solve_mw_dual` + the
master), which is the only model.

> **All lower bounds below are void (D30).** The subproblem's constraint set moved with
> the master's `y`, so no cut it produced was a valid lower bound. Every LB, gap and
> `status = OPTIMAL` in this document predates the fix and must be re-measured. Upper
> bounds, feasible schedules and served-passenger counts are unaffected: they come from
> pricing an exhibited schedule, which was always correct — verified to the cent against
> an independent monolithic MILP at three scales.
>
> The optimum of the reference instance is **4183.24**, not 4190.74.

**Reference result** — withdrawn. The run that produced it reported
`LB 4186.570873 / UB 4190.740015 / status OPTIMAL`, but 4186.57 exceeds the true optimum
of 4183.24, so the "convergence" was against an invalid bound. A replacement reference
must be produced on the corrected model.

---

## 0. Non-negotiables

1. No magic numbers — every parameter from config, with a stated unit (slots vs minutes
   vs km).
2. Model building is pure — no I/O, no solver calls inside `build_master` /
   `build_subproblem`.
3. Solver access through one adapter.
4. Every run **traceable**: a manifest with git commit, config hash, seed, solver version,
   and the `W_max` and `p` used. **Implemented** — `src/mobauto2_benders/manifest.py`
   writes `manifests/manifest_<name>_<timestamp>.json` on every run.

   *Reworded from "every run reproducible", which is false and was never achievable.*
   A run whose master solves stop on the clock is not bit-reproducible, because the
   nodes explored in those seconds depend on machine load (§0.10, D26). The manifest
   therefore records **whether** a run was reproducible rather than asserting that it
   was: `reproducibility.bit_reproducible` and the count behind it.
5. Every claim in the paper has a test. **Implemented** — 59 tests,
   `python -m unittest discover -s tests`.
6. Gapped runs never reported as optima without a marker. **Tested**
   (`test_gapped_run_is_not_reported_as_optimal`).
7. Every reported number states which subproblem mode produced it. **Implemented** — each
   dispatch branch labels itself (`mw`, `mw_fdiff_fallback`, `dual`,
   `finite_difference`), the multi-scenario aggregate reports one label or
   `mixed(a+b)`, and the manifest records it alongside `cut_valid_lower_bound`.
8. **New.** A reported lower bound must be accompanied by evidence that the master is a
   relaxation. Two independent defects (`AUDIT_v4` C3, C4) each silently produced bounds
   that were not bounds. The standing check is
   `LB <= (any demonstrated feasible objective)`; asserted in the test suite against
   **4183.24**, sourced from an independent MILP rather than from a Benders run of this
   same code, which is what made the previous 4190.74 circular (D30).
9. **New.** `concurrency_penalty` is active in the master objective and is **not** part
   of the originally published formulation. Its value must be stated on every reported
   table. The manifest records it.
10. **New (round 2).** A wall-clock budget and a reproducible number are incompatible.
    Truncating a MIP by the clock makes the result machine-dependent: the nodes explored
    in those seconds depend on load, and every later Benders iteration inherits the
    difference. Measured — one config at a binding 15 s per-iteration cap gave
    LB 2333.29, 2153.79 and 2175.87 over three runs; the same config at a non-binding cap
    reproduced to the last digit (2422.5195186024557, twice).

    `CPXPARAM_Threads: 1` does not fix this. It removes the nondeterminism of parallel
    MIP, not that of the clock.

    Therefore:

    - **Experiments** (A/B, sweeps) are budgeted by `solver.max_iterations`, with
      `solver.total_time_limit_s` and `master.per_iteration_time_limit_s` set generously
      enough that neither binds. Equal iteration counts are a reproducible basis for
      comparison; equal seconds are not.
    - **Simulation** runs are budgeted by time, and any table quoting one must say the run
      was time-truncated. A converged run (`status = OPTIMAL`) is exempt: it stopped on
      the gap.
    - Every run reports `clock_truncated_master_solves`; non-zero prints a
      `NOT REPRODUCIBLE` line and is recorded in the manifest.

    `configs/baseline_d9.yaml` has always stated these conditions in its header. Two of
    its four were not in force — `CPXPARAM_Threads` never reached the solver (D24), and
    D22's budget clamp made the master stop on the clock (D27).

---

## 1. Frozen and live parameters (Gate 0)

### 1.1 Frozen

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| `ρ` | charging rate | **70 km-eq/h, linear** | D1 |
| horizon | operating day | **10 h**, one-slot boundary buffer by design | D6 |
| `ε` | departure regularisation | **0.01**, same unit as the waiting objective | D4 |
| units inside the solver | **slots** throughout; minutes only in reporting/simulation | D7, D8 |
| elastic subproblem | **removed from the codebase** | D9, D14 |

### 1.2 Live sensitivity parameters

| Symbol | Meaning | Starting value | Sweep | Source |
|---|---|---|---|---|
| `W_max` | max admissible wait | 60 min | `{15, 30, 45, 60}` | D2 |
| `p` | unmet-demand penalty | 50 | `{30, 50, 100}` | D3 |

**On the dominance threshold:** withdrawn as a closed-form number (D5). The trade-off is
load-dependent — delaying a departure by `w` slots delays every passenger already on it —
and the slot-aggregate objective `(τ−t)·x[t,τ]` prices this correctly since `x[t,τ]` is
the full count on that arc. Show the `(p, W_max)` sweep; do not restate a single `W_eff`.

### 1.3 Structural parameters

| Symbol | Meaning | Value |
|---|---|---|
| `Q` | fleet size | 2 baseline, swept 1–5 per §C3 |
| `S` | seats per shuttle | 15 |
| `E_max` | battery, km-equivalent | 150 |
| `c_trip` (`L`) | one-way consumption | 30 km |
| `τ_trip` | one-way duration incl. dwell | 30 min → `trip_slots = ceil(trip_duration_minutes / slot_res)` |
| `δ` (`slot_resolution`) | slot length | 30 min baseline, refine toward 1 min (D6) |
| `concurrency_penalty` | excess-departure penalty | 0.25 — see §2.4, and non-negotiable 9 |

---

## 2. Mathematical specification

### 2.1 Sets and indices

Unchanged: `q ∈ Q`, `t, τ ∈ T`, `d ∈ {OUT, RET}`, scenarios `s ∈ S`.

### 2.2–2.3 Master variables and constraint blocks

Unchanged from v1/v3 (C1–C7), with the corrections in §2.4 and §2.8.

### 2.4 Master objective

```
min  θ  +  ε · Σ_{q,t} (yOUT[q,t] + yRET[q,t])
        +  conc_pen · Σ_t (eOut[t] + eRet[t])

     eOut[t] >= Yout[t] − 1        eRet[t] >= Yret[t] − 1
```

`eOut`/`eRet` capture departures beyond the first in each slot and direction, so the term
spreads departures rather than bunching both shuttles into one slot.

**This term is not in the originally published formulation.** It is small — about 1.75 of
a ~4190 objective on the baseline, roughly 7 penalised slots — but large enough to select
between otherwise equal schedules. It is a **separate knob from `ε`** and must not be
conflated with it (D4). Kept per D18.

### 2.5 Subproblem — slot-only, **de-layered (D30, this is a model change)**

```
Arcs[d] = { (t,τ) : t+1 <= τ <= min(T-1, t + W_slots) },  W_slots = ceil(W_max_min/δ)

min  Σ_d Σ_{(t,τ)∈Arcs[d]} (τ-t) · x_d[t,τ]  +  p · Σ_d Σ_t u_d[t]

s.t. Σ_τ x_d[t,τ] + u_d[t] = R_d[t]      ∀ d,t   (dual α_d[t], free)
     Σ_t x_d[t,τ]      <= C_d[τ]         ∀ d,τ   (dual π_d[τ] <= 0)
     x, u >= 0                                   with C_d[τ] = S · Σ_q y_d[q,τ]
```

**This replaces the layered form, and the reason is not cosmetic.** Until D30 each
departure slot was split into `K_d[τ]` layers, one per vehicle, so that `fill_eps·k` could
bias packing into the first vehicle. `K_d[τ]` comes from the master's `y`, so **`y` changed
the variable and constraint sets**, not just the right-hand side.

Benders duality requires `Q(y) = min{c'x : Ax >= b − By}` — `y` in the right-hand side
alone. With `A` itself moving, the dual of one instance is not a subgradient of the recourse
across `y`, and **no cut generator can be valid on top of it**. Measured: cuts forced θ to
6893 (MW directional), 5290 (MW single) and 6087 (plain dual) at a schedule whose true
recourse is 4183.00. Mechanism: `π[τ]` summed the K layer duals and `dm = S·π`, giving
slopes about K times too steep.

The layers were **redundant in capacity**: K layers of `min(S, S·K) = S` total `K·S`, which
is exactly `C_d[τ]` above; `K = 0` gives no arcs either way. So the recourse is unchanged —
verified against an independent monolithic MILP at three scales (4183.24, 651.36, 1674.11),
matching to the cent both before and after the change.

After it, the same diagnostic gives forced θ = 4165.00 against 4183.00: cuts under-estimate,
which is what valid cuts do.

**Correction to v3.** v3 wrote `π_d[τ,k] >= 0`. For a `<=` constraint in a
**minimisation** the dual is **non-positive**, which is what Pyomo/CPLEX return and what
`dm = +S·π` relies on to produce the negative slopes the master expects. v3's sign is
what `solve_mw_dual` implemented, and it made the MW LP unbounded (`AUDIT_v4` C3).

- `fill_first_epsilon` **has been deleted** (D30). It only ordered vehicles within a
  layer, and there are no layers. The per-shuttle report reconstructs the split by filling
  S seats at a time, which is the arrangement that epsilon produced anyway.
- `unused_capacity_penalty` **has been deleted** — it was schema-validated, threaded
  through, and read by nothing (D19).
- `(τ-t)` is in **slots** (D8). Do not introduce a `×δ` conversion inside this LP.
- Demand outside the horizon is **counted and warned about** (D25). It used to be
  discarded silently.

**Duals used for the cut:**

```
pi_OUT[τ] = Σ_k π_OUT[τ,k]      pi_RET[τ] = Σ_k π_RET[τ,k]     # per-slot, summed over layers
```

Never indexed by `q`. This is what makes the master's per-slot cut aggregation valid.

### 2.6 Cut construction

#### Default path — Magnanti–Wong

`solve_mw_dual` selects a Pareto-optimal dual on the optimal face. **The formulation
below is the corrected one**; v3 described the broken version.

```
variables:  α_d[t] free,  π_d[τ,k] <= 0

dual feasibility, one constraint per primal variable:
    from x[t,τ,k]:   α_d[t] + π_d[τ,k] <= (τ-t) + fill_eps·k
    from u[t]:       α_d[t]            <= p

optimal face (weak duality gives dual_obj <= ub_base):
    Σ_t R_d[t]·α_d[t] + Σ_{τ,k} cap_d[τ,k]·π_d[τ,k]  >=  ub_base − tol

objective (Pareto selection at the core point Ȳ):
    max  Σ_τ (S·Ȳ_out[τ] − C_out[τ]) · Σ_k π_OUT[τ,k]
       + Σ_τ (S·Ȳ_ret[τ] − C_ret[τ]) · Σ_k π_RET[τ,k]
```

Three points that v3 got wrong or omitted:

1. **Sign.** `π <= 0`, and the two inequalities point as written above. v3's convention
   made the LP unbounded: raising `α` and `π` together satisfied the constraint while
   increasing the objective, bounded only by the optimal face — which at the all-idle
   first iteration has every `cap = 0` and so does not bound `π` at all.
2. **The objective must carry `− y_inc`.** The cut is tight at the incumbent, so its
   value at the core point is `ub_base + Σ dm·(Ȳ − y_inc)`; `ub_base` is fixed by the
   optimal face, so the Pareto-optimal dual maximises `Σ dm·(Ȳ − y_inc)`. Since
   `y_inc[τ] = C[τ]/S`, the coefficient is `(S·Ȳ[τ] − C[τ])`. Maximising `Σ dm·Ȳ`, as v3
   implied, selects the wrong dual: the dropped term is not constant across the optimal
   face.
3. **The optimal face is an inequality, not an equality.** A float equality against a
   separately computed primal optimum is infeasible for a few ulps of disagreement.

**Slopes and broadcast:**

```
dm_out[τ] = S · Σ_k π_OUT[τ,k]          (<= 0)
coeff_yOUT[(q,τ)] = dm_out[τ]  for every q      # identical across q by construction
```

#### Fallback paths

| Mode | Valid lower bound | Notes |
|---|---|---|
| `mw` | **yes** | default |
| `mw_fdiff_fallback` | **no** | MW returned no solution; sets `cut_lb_valid = False` and logs `[SP WARN]`. v3 wrongly stated this branch marked itself invalid — it did not, which is how C3 hid |
| `dual` (`use_magnanti_wong: false, use_dual_slopes: true`) | **yes** | `dm_out[τ] = S·pi_out[τ]` from `solve_subproblem`. Not Pareto-optimal; the natural ablation baseline. **Currently unreachable** — see `AUDIT_v4` §3.5 |
| `finite_difference` | **no** | diagnostic only; never for a reported result |

`cut_valid_lower_bound` propagates to `solver.py`, which drops `best_lb` when false. In
multi-scenario runs validity is aggregated **conjunctively** — valid only if every
scenario's cut is valid, defaulting to false on an empty set. That semantics is correct
and must not be "simplified".

### 2.7 Master aggregation — validated and enforced

`aggregate_cuts_by_tau` collapses `Σ_q coeff[q,τ]·y[q,τ]` to `coeff[τ]·Y[τ]` where
`Y[τ] = Σ_q y[q,τ]`. That is an identity **only** when the coefficients agree across `q`,
which they do on the production path by construction.

Enforced rather than assumed:

- `_add_cut` raises if a coefficient varies across `q` at fixed `τ`.
- The constant re-anchoring `const_adj = ub_est − contrib` is a **check, not a
  correction**. It must be a no-op up to the mass deliberately dropped by
  `cut_coeff_threshold`; the check subtracts that mass and raises on the residual. A
  naive "delta must be zero" fires spuriously — the threshold alone accounts for up to
  ~0.044 at Q=2, T=22.

### 2.8 Symmetry breaking — **corrected, this is a model change**

```
Σ_t (yOUT[k,t] + yRET[k,t])  <=  Σ_t (yOUT[k-1,t] + yRET[k-1,t])      k = 1..Q-1
```

Order vehicles by **total** departures. `Q-1` rows.

**Do not order by cumulative time prefix.** `cum[k][t] <= cum[k-1][t]` for every `t` is
strictly stronger, removes feasible schedules that are not symmetric duplicates, and
makes the master stop being a relaxation — so its bound, the Benders LB, can exceed the
true optimum. It is invalid **even for a homogeneous fleet**:

> Vehicle A makes one trip at t=0, vehicle B at t=1 and t=2. As (A,B):
> `cum[0]=[1,1,1]`, `cum[1]=[0,1,2]` violates at t=2. As (B,A): `cum[0]=[0,1,2]`,
> `cum[1]=[1,1,1]` violates at t=0. No relabelling works.

Total ordering is valid because any schedule can be relabelled by sorting vehicles on
total trips.

**Precondition, now enforced:** a homogeneous fleet in an identical initial state.
Differing `initial_battery` or `initial_actions` raises with the offending values named.

### 2.9 Constraints deliberately **not** added

**`b[q,t] >= L·yRET[q,t]`** (v3's M1). Sound and implied — exclusivity forces `c=0` when
`yRET=1`, hence `gchg=0`, so `b[q,t+1] = b[q,t] − L >= 0` requires `b[q,t] >= L` — but
measured, it takes the master phase from 18.2 s to 49 s over 10 iterations and yields a
**worse** bound at the same budget. v3 recommended it on the grounds that a tighter LP
relaxation is worth it because master time dominates; that reasoning does not survive
measurement. A test asserts its absence so it is not re-added from v3 without
re-measuring.

### 2.10 Algorithm

Unchanged. Gate 1's sign-convention verification is empirical (D13) and now automated:
`test_no_check_fail_lines` asserts no `[CHECK FAIL]` line appears.

---

## 3. Repository state

The v3 refactor plan is complete.

| v3 step | State |
|---|---|
| Step 0 — flip `enable_temporal_refinement` | done, then superseded by deletion |
| Step 0.5 — delete or quarantine the elastic path | **deleted** (D14). 3267 → 1828 lines |
| "unit fix" | withdrawn in v3, correctly (D8) |
| sign unification → empirical verification | done, automated |
| config extraction, solver adapter, indexed constraints, narrowed exceptions, diagnostics-off | indexed constraints and diagnostics-off done; exception narrowing partial (`AUDIT_v4` §3.6) |
| cleanup: duplicate helper, duplicate symmetry, `CorePoint`, `cplex_log` fixture | all done; the log fixture is synthetic |

Master construction measures **7.6 ms**, so the model-building cost v3 was concerned
about is 0.04% of a master solve. The cost is MIP solving as cuts accumulate.

---

## 4. Config schema

Resolved. `unused_capacity_penalty` deleted (D19). `concurrency_penalty` kept and
documented (D18). New keys: `run.emit_reports` (default false),
`master.charge_before_idle` (default true).

`enable_temporal_refinement` is not a schema key and never was; the code it gated no
longer exists.

### 4.1 The three stopping limits *(D22)*

Four values govern when a run stops. Three are time/gap limits whose names used to be
confusable, and two of those did nothing at all until they were wired.

| key | scope | test phase | simulation phase |
|---|---|---|---|
| `solver.total_time_limit_s` | budget for the **whole** Benders loop | 120 | 1800+ |
| `master.per_iteration_time_limit_s` | ceiling on **one** master MIP solve | 30 | 600 |
| `master.per_iteration_mipgap` | ceiling on the master MIP gap per iteration | 0.05 | 0.05 |
| `solver.tolerance` | relative BD gap counting as **converged** | 0.001 | 0.001 |

`per_iteration_mipgap` is the same in both phases because it is a ceiling: the gap-tied
schedule tightens it to 0.001 on its own as the Benders gap closes. Setting it low does
not buy a tighter bound, it only makes early iterations expensive.

`total_time_limit_s` is a real ceiling as of D22: it is checked between iterations **and**
clamps the master's per-iteration limit. Before that it was a floor — one master solve
could overshoot it without bound.

The old names `solver.time_limit_s`, `master.solve_time_limit_s` and `master.mipgap`
raise at config load, naming their replacement.

### 4.2 Per-vehicle list convention *(D23)*

`model.fleet.initial_battery` and `model.fleet.initial_actions` take
**[z specific vehicles…, 1 value shared by the remaining Q−z]**. The list is padded to
length Q by repeating its **last** entry.

So a homogeneous fleet is a one-element list at any Q:

```yaml
fleet:
  Q: 5
  initial_battery: [150.0]     # all five start full
  initial_actions: [IDL]       # all five start idle
```

and a fleet with one distinct vehicle is two elements:

```yaml
  initial_battery: [90.0, 150.0]   # q0 at 90, q1..q4 at 150
  initial_actions: [OUT, CHR]      # q0 departs, q1..q4 charge
```

A list longer than Q is truncated. Note that `master.symmetry_breaking` refuses a
heterogeneous fleet outright (§2.8), so the multi-element form is only usable with
symmetry breaking off.

---

## 5. Test suite

Implemented. `python -m unittest discover -s tests` — 49 tests, ~10 s, stdlib
`unittest`. 33 need no solver and run in under a second.

Beyond v3's list, the suite pins the defects v3 did not know about: MW actually
succeeding, `LB <= 4190.74`, the report summing consistently, and the cut mode being
reported. See `AUDIT_v4` §5 for the mapping.

`test_unused_capacity_penalty_has_no_effect` from v3 is **not** implemented as written —
the parameter was deleted, so the test asserts it is now rejected instead.

---

## 6. What did not change from v3

Master constraint blocks C1–C7 (modulo §2.8), the Benders algorithm skeleton, the
experiment matrix, the report framework (Parts I–III of `Final_report_framework_v2`), and
Gates 2–5. D1–D8 and D10–D13 stand as written.

---

## 7. Next phase — improvement

> **Superseded in part by Fase 1 (D33–D37), 2026-08-06.** At Q=3 / T=44 / 4 scenarios /
> 300 s the master ends every solve at a 99.9% internal MIP gap and the Benders lower
> bound reaches 0.35 against a 785 target; 3.4x the per-solve time buys 1.2% of bound.
> No optimality gap may be quoted at Q>=3, and performance items below that assume the
> cut set is the binding constraint do not apply there. Evidence and reading rules:
> `docs/phase1/README.md`.

The correctness contract above is now enforced by tests. The remaining work is research
output and performance, ordered in `AUDIT_v4` §3:

1. `(p, W_max)` sweep and full table regeneration — **every pre-`dbc01e2` LB and gap is
   void and must be regenerated**; UBs and schedules are unaffected.
2. θ disaggregation A/B (D11).
3. Convergence performance: the dead early exit in `_cand_theta`, the master mipgap
   ceiling, the MW core point decaying to the boundary.
4. Silent demand truncation, which matters more as the horizon extends to 24 h (D6).
5. `use_dual_slopes` currently unreachable.
