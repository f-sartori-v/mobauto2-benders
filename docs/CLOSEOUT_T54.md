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

5. **The optimal-placement instrument needs the opposite eligibility default, and saying
   so is the point.** It *chooses* the departure instant, so there is no slot-aggregation
   ambiguity to guard against; imposing `forbid` there prices "choose the best instant,
   then refuse the passenger it was chosen for". It defaults to `allow`, states why in
   the code, and carries the value on every result it returns. The two *recourses* — the
   thing B6 names — agree.

---

## 9. What is still open

* The like-for-like **optimised** δ=1 comparison (B6 Part 2). Neither arm closes at any
  budget this work order could spend; the convention effect in §2 is the measurement
  that stands.
* Every table in §4's second list.
* B13's honest EVPI: implemented, unexecuted.
* All of P2.
* `K^chg` at fleet sizes other than Q=2, and on the 450-passenger instance. The Q=2
  sweep is in §3b; whether the 1.52 % cost grows or shrinks with the fleet is unmeasured.
