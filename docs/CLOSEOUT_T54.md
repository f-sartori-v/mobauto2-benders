# Close-out — Agent BENDERS, MOB-AUTO2 T.5.4

**Consolidated regime manifest id: `113e0e5675d9`** (git `8f063c7`).

    H=660  delta=30  Q=2  S=15  Emax=150  b0=150  c_trip=30  rho=70/h  tau_trip=30
    Wmax=60  p_min=56  epsilon=0.01  kappa=0.25  K_chg=null(=Q)  o=[0.0]
    same_slot_eligibility=forbid  objective_mode=weighted_sum
    demand=setups/base.yaml (checksum 44896587338a60d3)  solver=CPLEX 22.1.1  threads=1  seed=42

The legacy penalty regime (`p=50`, `p_min=1500`) is manifest `3c60024bb898` and is
quarantined, not converted.

> **The manifest id contains the git revision, as the contract specifies.** That makes
> it change on every commit, so "no table may mix two manifest ids" is stricter than it
> may look: two tables produced from different commits are formally incomparable even
> when every parameter agrees. The id above is the one to quote for this hand-back.
> `manifest_fields` is emitted beside it so a reader can see which of the 27 fields
> actually differ between two ids.

---

## 1. P0 — status and numerical delta

| Item | Status | What changed numerically |
|---|---|---|
| **B1** multi-scenario cut, one way | **landed** | No delta on shipped configs: 72 of 75 were already aggregated, 3 disaggregated, none mixed. The pairing defect was latent — it fires only when a per-scenario theta early-exit shortens the list — so nothing published is retracted. `cut_architecture` now names the two architectures and refuses a mixed setting; the reconciliation assertion runs on every aggregated insertion. |
| **B2** charger capacity | **landed** | Zero at the default: `K_chg = Q` reproduces the baseline to the last digit. At `K_chg = 1` the schedule **changes** and the objective **rises 399.907 → 405.973** (+1.52 %). The divisible row is emitted only when `K_chg < Q`, where it is not implied. Swept — see §3b. |
| **B3** final-slot energy leak | **landed** | Baseline objective **unchanged** (monolith 399.907 before and after). The leak was real but free: `c[q,T-1]`, `gchg[q,T-1]` fed no state and no cost. Both engines now fix both; the monolith previously fixed only `gchg`, so the two engines had disagreed about the last slot of every schedule they both solved. |
| **B4** exactness fixtures | **landed** | Dual-equality assertion **deleted**. Replaced by fingerprint + variable/constraint sets + objective coefficients + non-capacity RHS + optimal value. Duals are logged. E3→**M1** (master equivalence), E4→**M2** (symmetry validity). Symmetry now refused for heterogeneous SoC **and** heterogeneous initial location. |
| **B5** minute-incidence capacity path | **landed** | Nothing to delete — the incidence path existed only in prose. The assertion now runs against the slot capacity rows in **both** recourses: ≤ T capacity rows per direction, one demand row per arrival minute. |
| **B6** same-slot eligibility | **landed** | **The largest behavioural change in this hand-back.** Default is `forbid`, which under `departure_policy="start"` reproduces the master's `τ ≥ t+1` arc set at any resolution. Every minute-mode number produced before this commit was priced under `allow`. Effect measured — see §2. |
| **B7** penalty regime | **landed (Benders arm partial)** | `rejected_with_free_seat` = **32** on the p56 baseline (nonzero, as predicted). Lexicographic mode lands **on the monolith only**; the Benders engine refuses it loudly rather than running the weighted sum under its name — see §5. |
| **B8** 450/30 as two models | **landed** | 42 cells, proof status per cell. Result in §3. |

---

## 2. B6 — the convention effect, and why the δ=1 figures cannot be read as reported

Measured on ONE FIXED schedule priced twice, so nothing but the convention can move
(`scripts/same_slot_convention.py`, Part 1):

| δ | forbid | allow | effect |
|---|---:|---:|---:|
| 30 min | 6140.0 | 5899.0 | **+4.09 %** (+241 pax-min) |
| 1 min | 5645.0 | 5374.0 | **+5.04 %** (+271 pax-min) |

At δ=1 the two arms serve the *same* 257 passengers; the entire difference is waiting,
and it is almost exactly one minute per carried passenger — which is what `forbid`
mechanically adds to anyone who would have boarded at zero wait.

**The consequence for the report.** The published δ=1 figures were **1.40 %** and
**1.60 %**. The convention effect alone is **5.04 %** on the same instance. The reported
difference is therefore smaller than a confound inside it, and cannot be read as a
decomposition result. This is a result in its own right, as the work order anticipated.

**Part 2 (both arms re-optimised) did not close.** At δ=1, Q=2, 120 s per arm: `forbid`
stopped at a 61.8 % gap; `allow` exceeded CPLEX's own subprocess budget. Both are
labelled `CLOCK-TRUNCATED`, neither is quoted, and the like-for-like *optimised*
comparison at δ=1 remains open at any budget this work order could spend.

---

## 3. B8 — models (a) and (b) for the 450/30 question

`scripts/trip_cap_450.py`, 45 s per cell, 42 cells, `outputs/workorder/trip_cap_450.json`.

**Model (a) — maximise served subject to `N_trip ≤ 30`.** Every cell proven optimal
except (Q=3, H=900), which is clock-truncated and therefore a lower bound.

| Q | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H=660 | 255 | 320 | 340 | 355 | 370 | 385 | **398** |
| H=780 | 259 | 322 | 342 | 357 | 372 | 385 | **398** |
| H=900 | 259 | 322* | 342 | 357 | 372 | 385 | **398** |

The trip cap binds at exactly 30 trips from Q=3 upward; at Q=2 it does not bind (24
trips). Served saturates at **398 of 450** and does not move with the horizon.

**Model (b) — minimise `N_trip` subject to served = 450.**

| Q | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H=660 | INFEAS | INFEAS | INFEAS | INFEAS | INFEAS | INFEAS | INFEAS |
| H=780 | INFEAS | INFEAS | INFEAS | ≤52* | **42** | **40** | **38** |
| H=900 | INFEAS | INFEAS | INFEAS | ≤48* | **42** | **40** | **38** |

`*` clock-truncated: an achieved schedule, so an upper bound on the minimum.

**The answer.** The report's conclusion survives, but it is now proven rather than
inferred, and the proof says something the inference could not:

* No fleet up to Q=8 serves 450 passengers in 30 trips. The proven ceiling is **398**.
* 450 *is* reachable — from Q=6 at H=780 — and the minimum trip count to reach it is
  **38** (Q=8). The distance from 30 is 8 trips, not "impossible".
* At H=660 serving all 450 is infeasible at every Q tested, so the 660-minute horizon,
  not the fleet, is what forbids it there.

The arithmetic behind the ceiling: 30 trips × 15 seats = 450 seats exactly, so
450-in-30 requires **every trip full, in the right direction, at the right minute**.
398/450 = 88.4 % seat utilisation is what the demand's shape and `W_max = 60` actually
permit.

**Horizon necessary condition**, printed before every run and read as it should be:
`k·τ_trip + (60/ρ)·max(0, k·c_trip − b₀) ≤ H` gives 14 trips per vehicle at H=660 (17 at
780, 21 at 900), hence 14Q for the fleet. It bounds **trips, not passengers**: at Q=8 it
allows 112 trips while model (a) is capped at 30 and model (b) needs only 38. No cell's
infeasibility is explained by it.

---

## 3b. B2 — the baseline at `K_chg ∈ {1, 2, Q}`

Monolith, consolidated regime, Q=2 so `K_chg = 2` and `K_chg = Q` are the same setting.

| `K_chg` | status | objective | Δ vs default | schedule |
|---|---|---:|---:|---|
| 1 (divisible) | OPTIMAL | 405.9733 | **+6.0667** | changes |
| 1 (binary occupancy) | OPTIMAL | 405.9733 | +6.0667 | same as divisible |
| 2 = Q (default) | OPTIMAL | 399.9067 | — | baseline |

**Charger capacity binds, and it costs 1.52 %.** One charger for two vehicles forces a
different schedule and a worse objective, so `K_chg` is data the instance was previously
assuming away rather than an implementation detail — which is the audit's point.
**Extended across fleet sizes and both instances in §10.**

The divisible and indivisible forms **agree on this instance**. They are not equivalent
in general — integral occupancy forbids two vehicles sharing a charger even at `K = Q` —
but at Q=2, `K_chg=1` the divisible optimum happens to use whole slots anyway, so nothing
here distinguishes them. A site whose data say charging is not preemptible should still
set `charger_occupancy_binary` explicitly; the manifest records which form ran, and the
two must not be mixed in one table.

---

## 4. Reruns: regenerated, and still legacy

**Regenerated under `113e0e5675d9`:**

| Table | Result |
|---|---|
| p56 baseline, monolith | **399.907**, OPTIMAL, 0.32 s, proven |
| p56 baseline, Benders (10-iteration fixture) | lb 337.392 / ub 414.640, 18.6 % interval, `status=UNKNOWN` |
| `rejected_with_free_seat`, p56 baseline | **32** |
| δ=1 convention effect | §2 |
| 450/30 grid | §3 |
| waiting statistic | §5 |

**Not regenerated — numbers stand with a visible label:**

| Table | Label |
|---|---|
| Benders values around 4183.24 | manifest `3c60024bb898` (legacy p_min=1500). Not rerun: the p56 regime is a different problem, not a re-measurement of this one. |
| competitiveness rows | not rerun: needs converged Benders runs, which exceed the day's budget at every size where the comparison is interesting. |
| runtime split | not rerun: same reason, and it must be re-measured with the deterministic-work columns B10 added rather than wall time alone. |
| resolution factorial | not rerun. **Additionally invalidated by B6**: every minute-mode cell was priced under `allow` and the default is now `forbid`. |
| hedging table | renamed (B13) but **not rerun**; `--comparator minute` is implemented and unexecuted. |
| penalty frontier | not rerun. |

Every one of these is a budget shortfall, not a blocked item. I missed from the bottom,
as instructed.

---

## 5. `rejected_with_free_seat`, and the objective mode

**Baseline (p56, δ=30, W_max=60): 32 passengers.** Nonzero, as the work order required.
The mechanism is arithmetic: p = 56/30 ≈ 1.867 slot-units, so a two-slot wait costs 2 and
a rejection costs 1.867. The model prefers to leave a passenger behind rather than make
them wait an hour — **even with a free seat**. That is a coherent policy; it was never a
stated one, and nothing in any output revealed it was in force. It is now printed with
every result, in all three places that price passengers (validator, slot recourse, minute
recourse), computed from the same solved LP that produced the headline.

The acceptance test pins both directions: nonzero at p_min=56, **zero** at p_min=120,
where no admissible wait can cost more than the penalty.

**Lexicographic mode**, on the new fixture (`configs/milp/penalty_regime_*.yaml`, 21
requests, Q=1, T=6):

| mode | carried | θ |
|---|---:|---:|
| `weighted_sum` | 18 / 21 | 23.6 |
| `lexicographic` | **21 / 21** | 27.0 |

Three passengers, 12 free seats on a reachable departure, and the weighted sum leaves
them behind. Implemented as a genuine hierarchy — solve, freeze by constraint, solve —
with the freeze slack taken from the solver's own reported gap, never a constant. **Not**
the small-coefficient emulation the work order forbids.

**What blocks it on the Benders engine, stated rather than worked around.** Served demand
lives inside the recourse there and reaches the master only through `θ`. "Maximise served
demand first" would have to be imposed on a proxy the master cannot see, and the honest
ways to do that — a served-demand cut family, or a second `θ` bounding unserved demand
alone — are new formulations, not a flag. The engine therefore **refuses** the mode at
load rather than running the weighted sum under its name.

---

## 6. B12 — the 13.9–15.9 min range, reconstructed

It reconstructs, and **exactly one thing varies across its endpoints: the scenario.**

| endpoint | value | definition |
|---|---:|---|
| lower | 13.94 | scenario `base`, **direction OUT only**, minute assignment, o=0, `forbid`, denominator = carried, n=84 |
| upper | 15.90 | scenario `temporal_noise`, **direction OUT only**, minute assignment, o=0, `forbid`, denominator = carried, n=84 |

Two qualifiers were unstated: that the range is across the four demand scenarios (it was
presented as being for "one fixed schedule", which it is — the schedule is fixed and the
*day* varies), and that it is the OUT direction alone.

Across all 144 definitional choices this one schedule admits, the average spans
**4.87–21.16 min**. The published range was a narrow, unstated slice of that, so quoting
its width as if it were an uncertainty interval overstates precision and understates
spread at the same time.

**Recommended single defined number** for the consolidated regime:
`avg_wait_min=14.51 scenario=base direction=both assignment_resolution=minute
departure_offset=0 same_slot_eligibility=forbid denominator=carried n=194
exclude_unserved=True`.

---

## 7. P1 items other than reruns

| Item | Status |
|---|---|
| **B10** censoring | **landed**. `reproducibility.censored` on every manifest; `RunRecord.censored`; `median_trajectory` and `performance_profile` over repetitions; deterministic-work fields (nodes, simplex iterations, deterministic time) beside wall time. The discard rule is gone. |
| **B11** ratio refused on status mismatch | **landed**. `results_emitter.time_ratio` refuses when the arms' termination differs *or* when either is censored; `proof_and_gap` emits `(time_to_proof_s, terminal_gap)`, with `time_to_proof_s = None` for an arm that never proved optimality. |
| **B12** waiting statistics | **landed**. §6. |
| **B13** hedging comparison | **partial**. Renamed to "one stochastic schedule against scenario-specific slot-optimal schedules", with the reason printed under the table. `--comparator minute` implements the honest version (scenario optima under the *same* minute recourse that evaluates them, making EVPI defined). **Not executed** — five monolith solves, out of budget. VSS deliberately not computed: it needs the mean-value solution as a third arm. |
| **B14** MW naming | **landed**. Called *Magnanti–Wong-inspired dual selection*; `certified_relative_interior` is recorded and is never True; `mw_core_point_certification` records what was attempted and what it returned. "richer" is gone. |

**P2: not started.** Missing from the bottom, as instructed.

---

## 8. Things found that the audit did not

1. **The aggregated cut could mispair weights with scenarios.** Five parallel lists were
   appended inside a loop containing a `continue` (the per-scenario θ early-exit). A
   shortened list pairs every scenario with its predecessor's weight and silently drops
   the last one. Latent on shipped configs — the early-exit needs `theta_per_scenario`
   on, which no aggregated config sets — but it is the exact failure mode the audit's
   item 1.4 describes, one level deeper than the pseudocode.

2. **The two engines disagreed about the last slot.** The monolith fixed `gchg[q,T-1]`
   and deliberately left `c[q,T-1]` free ("Allow charging label at the last slot if
   desired"); the Benders master left both free. Every schedule the two produced for the
   same instance could differ in its final slot, in a place no objective could see and
   no comparison would report.

3. **Charger capacity, if it ever binds, retires condition M1.** `Σ_q c[q,t] ≤ K^chg`
   genuinely couples named vehicles — it is a shared physical resource, not a function of
   the signature. So a site with fewer chargers than vehicles invalidates the stage-3
   Dantzig–Wolfe reformulation's per-vehicle column. That is a fact about the site, and
   it is now recorded in the allow-list rather than discovered later. It is also why the
   row is emitted only where it binds.

4. **`p_min = 56` sits inside the waiting window, and that is what creates the regime.**
   At `W_max = 60 > p_min = 56` there exists an admissible wait that costs more than a
   rejection. The two parameters were chosen independently and their *ordering* is what
   produces the dominated-wait behaviour. `p_min > W_max` removes it entirely — which is
   what the acceptance test's p_min=120 arm demonstrates. This is a one-line policy check
   nobody was making.

5. **Shipping `cut_architecture` alongside the legacy booleans in one config is a trap.**
   The annotated reference sets `multi_cuts_by_scenario`; adding an active
   `cut_architecture` line to the same file means anyone who copies it and flips the
   boolean hits a contradiction error they did not cause. The key is documented there
   but left commented out, with the reason. Found by the config test suite the moment
   the example was edited, which is the test doing its job.

6. **The optimal-placement instrument needs the opposite eligibility default, and saying
   so is the point.** It *chooses* the departure instant, so there is no slot-aggregation
   ambiguity to guard against; imposing `forbid` there prices "choose the best instant,
   then refuse the passenger it was chosen for". It defaults to `allow`, states why in
   the code, and carries the value on every result it returns. The two *recourses* — the
   thing B6 names — agree.

---

## 10. Second pass — the items §9 left open

Everything below is new since the close-out was first written, and each run was capped
so no single cell could decide the session's budget.

### 10.1 Merged with `main`, and the merge had exactly one correct resolution

`main` moved to `5c3411c` while this branch was open, bringing D86: the minute recourse's
arc set was extracted into `_minute_recourse_geometry` so that the primal
(`solve_minute_recourse`) and the new `solve_mw_dual_minute` could not drift apart. B6
had added `same_slot_eligibility`, which *changes* that arc set, inside the primal.

Taking both sides' text — what a conflict resolution naturally produces — would have left
the primal filtering arcs by `w0` while the shared geometry did not, so the MW dual would
have been built over a strictly **larger** arc set than the primal it is restricted to:
dual variables for arcs the primal does not have, selecting on a face that is not the
primal's optimal face. Not a weaker cut, an **invalid** one, and silent — the MW path
returns `None` on weak-duality violations and falls back to the plain dual without
saying why.

The convention therefore moved *into* the geometry. `tests/test_fast_minute_geometry.py`
pins it both behaviourally (the two conventions give strictly nested arc sets; the extra
arcs are exactly the zero-wait ones; the grouped views agree with the arc list) and
structurally (both consumers call the one factory; neither rebuilds arcs itself).

### 10.2 `same_slot_eligibility` stays `forbid`, and that is the cheap choice

The question was whether flipping the default to `allow` would avoid re-running
everything. It would do the opposite. The flag enters **both** recourses, so `allow` as a
default changes the **slot** recourse from `τ ≥ t+1` to `τ ≥ t` — which moves 4183.24,
399.907, the whole soundness suite and all **94** slot-mode configs.

Under `forbid`, what changes is the **7** minute-mode configs, and those were already on
the re-run list. Two of the three tables that depend on them are regenerated below.

### 10.3 B2 closed — `K_chg` across fleet sizes and both instances

`scripts/charger_capacity_sweep.py`, 45 s per cell,
`outputs/workorder/charger_sweep.json`. **Proven cells only**: the first draft of the
trend line mixed proven and clock-truncated cells, which is worse than useless here
because the truncation is not even in a consistent direction — a truncated *reference*
makes its own row's delta look smaller, while a truncated *constrained* cell makes it
look larger.

| instance | proven `K_chg = 1` cost | `K_chg ≥ 2` |
|---|---|---|
| baseline (300 req) | Q=2: **+1.52 %**, Q=4: **+5.97 %** | **0.00 %** at every proven cell |
| target (450 req) | no proven cell at 45 s | **0.00 %** at Q=4, K∈{2,3} |

**The operational answer is the threshold, not the trend: two chargers are enough at
every fleet size tested, on both instances; one is not.** On the two proven baseline
points the cost of a single charger *grows* with the fleet (1.52 % → 5.97 %), which is
the "more vehicles competing for one charger" reading rather than the "bigger fleet has
more slack" one — but two points are two points, and Q=3 and Q=5 are excluded as
truncated.

The indivisible (`charger_occupancy_binary`) form agrees with the divisible one wherever
both are proven. That does not make them equivalent — integral occupancy forbids sharing
even at `K = Q` — it means this instance's divisible optimum happens to use whole slots.

### 10.4 B13 closed — EVPI is defined, and the impossible cell is gone

Both arms are now solved under the **same** minute recourse that prices them. The hedged
arm moved too: it was still being built under the slot recourse while its comparators
were minute-optimal, which is the same mismatch one level up.

| comparator | base | temporal_noise | return_peak_adv | midday_surge | **averaged** |
|---|---:|---:|---:|---:|---:|
| slot (as published) | **−2.4 %** | +2.1 % | +7.5 % | +1.2 % | **2.1 %** |
| minute (honest) | +0.3 % | +2.1 % | +10.3 % | +1.2 % | **3.3 %** |

The **−2.4 %** cell is the audit's own proof, reproduced: a schedule optimised for four
scenarios cannot beat a schedule optimised for `base` *on* `base` unless the comparator
was not optimal under the valuation being applied. Under the minute comparator every gap
is non-negative, as theory requires.

**EVPI = 321.0 passenger-minutes = 3.3 %** of the perfect-information cost. The published
2.1 % was 1.2 points low *and* contained a mathematically impossible cell.

VSS is still not computed: it needs the mean-value solution as a third arm.

### 10.5 P2 (audit 4.4) — cut-quality diagnostics

`scripts/cut_quality.py`, 10 signatures, half of them fractional because the LP phase
evaluates the recourse at fractional `y`.

| generator | tightness | efficacy | orthogonality | density | gen s | lower bound? |
|---|---:|---:|---:|---:|---:|---|
| `dual` | 6.8e-14 | 5.287 | 0.000 | 0.577 | 0.07 | yes |
| `mw` | 1.0e-06 | 5.287 | 0.000 | 0.584 | 0.09 | yes |
| `finite_difference` | 1.1e-14 | 8.914 | 0.192 | 0.573 | 0.87 | **NO** |

**Magnanti–Wong returns a cut parallel to the plain dual at every sampled point** —
orthogonality 0.000, identical efficacy. This is the measurement B14 asked for in place
of the word "richer", and on this instance the selection buys nothing measurable. The
generator that scores *best* on efficacy is the one with no lower-bound guarantee, which
is exactly why that column is in the table.

Normalised and combinatorial cuts are reported as **not implemented** rather than
approximated: a "normalised" row produced by rescaling a dual cut would measure the
rescaling, not the family.

### 10.6 Penalty frontier, regenerated

`scripts/sweep_penalty_window.py`, 60 s per cell, `policy=start`, `forbid`, 25 cells.
Served of 300, minute-honest:

| `W_max` \ `p_min` | 14 | 28 | 56 | 112 | 224 |
|---|---:|---:|---:|---:|---:|
| 30 | 0 | 0 | 185 | 185 | 185 |
| 45 | 0 | 0 | 190 | 197 | 197 |
| 60 | 0 | 0 | **194** | 218 | 222 |
| 90 | 0 | 0 | 193 | 227 | 231 |
| 120 | 0 | 0 | 193 | 229 | 236 |

**At `p_min` ≤ 28 the model serves nobody.** The penalty is below the cheapest available
wait, so rejecting every passenger is optimal — B7's mechanism at its limit, and a
reminder that `p_min` is a policy statement, not a tuning constant. Average wait per
served passenger at (`W_max`=60, `p_min`=56) is 14.5 min, consistent with §6.

### 10.7 Resolution factorial, regenerated — and the budget shows through

`scripts/sweep_multiresolution.py`, δ ∈ {30,15,10} × Q ∈ {2,3} × {commuter, bimodal,
spiky}, `policy=start`, 45 s per arm, 18 cells. "Gain" is how much cheaper the
minute-optimised schedule is, both arms priced at minute fidelity.

| δ | gains observed |
|---|---|
| 30 | 0.00 %, 0.00 %, 0.72 %, 1.17 %, **3.08 %**, **3.62 %** |
| 15 | 0.00 %, 0.01 %, 0.13 %, 0.18 %, 0.73 %, 0.98 % |
| 10 | **−1.35 %**, **−0.09 %**, 0.00 %, 0.00 %, 0.18 %, 1.08 % |

**Read the negative cells, not past them.** A negative gain is impossible if both arms
were solved to optimality — the minute-optimal arm minimises exactly the quantity being
measured, so it cannot lose to the slot-optimal one. Two negative cells at δ=10 are
therefore the 45 s cap showing itself, not a finding about resolution.

What the table does support: the multi-resolution gain is **largest at the coarsest
grid** and collapses below 1 % by δ=15, which is the direction the mechanism predicts —
a finer first-stage grid leaves the minute recourse less to correct. What it does **not**
support is any claim at δ=10, where the budget is too small to resolve an effect of that
size. Re-running δ=10 needs a per-cell budget this session did not have.

Two cells at δ=15 also carry a *negative* `served+` (−2, −4): the minute-optimised
schedule is cheaper while carrying fewer passengers. That is the penalty regime again
(§5), not an anomaly — at `p_min = 56` shedding a passenger who would wait an hour is a
cost reduction.

---

## 9. What is still open

* The like-for-like **optimised** δ=1 comparison (B6 Part 2). Neither arm closes at any
  budget either session could spend; the convention effect in §2 is the measurement
  that stands. **Still open.**
* From §4's second list: the **penalty frontier** (§10.6) and the **resolution
  factorial** (§10.7) are regenerated. The **competitiveness rows** and the **runtime
  split** are **still open** — both need converged Benders runs, and B10's deterministic-
  work columns mean they must be re-measured rather than re-formatted.
* B13's honest EVPI: **done** (§10.4). **VSS still open** — it needs the mean-value
  solution as a third arm.
* P2: the **cut-quality grid is done** (§10.5). The **LP-strength grid**, the
  **temporal-semantics factorial** beyond B6's first cell, and **demand generalisation**
  (30 independent days × 3 volumes with a train/validation/test split) are **still
  open**.
* The δ=10 row of §10.7 needs a larger per-cell budget before it can be read at all.
* `K^chg` at fleet sizes other than Q=2, and on the 450-passenger instance. The Q=2
  sweep is in §3b; whether the 1.52 % cost grows or shrinks with the fleet is unmeasured.
