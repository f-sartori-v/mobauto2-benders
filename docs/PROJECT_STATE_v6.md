# Project state — v6

**Read this first, and read it instead of the older documents.** Everything below is what is
still true on `main` as of D69. It exists because the project's failure mode is not wrong
work, it is **stale documents that are still readable**: two outside reports were written
six weeks apart, and both quoted numbers this repository had already withdrawn, because
nothing said in one place which numbers were dead.

Version 6, not 5: `AUDIT_v5.md`, `BENDERS_SPEC_v5.md` and `docs_decisions_v5.md` were
allocated on `week2-lp-only-measurement`, a branch abandoned on 2026-08-10 and never merged
(D67 §1, D68 §1). Reusing v5 would recreate exactly the collision this document exists to
close.

---

## 1. Reading rule

| Document | Status | What it is for |
|---|---|---|
| **this file** | current | Where the project stands, what is dead, how to run it |
| `FORWARD_PLAN_v1.md` | current | Commit-sized backlog derived from the report's forward-work and open-directions sections. Nothing in it is measured; a D-number is earned when an item lands in `docs_decisions.md`, not by appearing there |
| `docs_decisions.md` | current, authoritative | The register, D1–D69, one contiguous allocation. **The only file whose D-numbers need no remapping** |
| `BENDERS_SPEC_v4.md` | current | The model as implemented, and the eight non-negotiables |
| `RESEARCH_NOTE_v2.md` | current | The multi-resolution contribution (claim 3), measured |
| `DESIGN_DD_v1.md` | current for E1–E4 | Exactness conditions and the structural argument. **Design only — nothing in it is measured** |
| `AUDIT_v4.md` | current | Defect record C2–C5, H7, H8, M1–M5, N1–N7 |
| `REPORT_REVIEW_v1.md` | historical | What report v1 got wrong. Superseded as an errata by report v2 and by §3 below |
| `AUDIT_v3.md`, `BENDERS_SPEC_v3.md` | **superseded** | Kept only as a record of what must not be reintroduced |
| `AUDIT_v5.md`, `BENDERS_SPEC_v5.md`, `docs_decisions_v5.md`, `RESULT.md` | **unmerged, abandoned** | On `week2-lp-only-measurement`. Not in this tree. Do not cite (D68 §1) |
| `HANDOUT.md`, `BENDERS_CORRECTION_PLAN.md`, `HANDLER_CENSUS.md`, `SESSION_TOOLING.md` | **not in this repository** | They exist outside it. Their D-numbers need §4's remapping |

One rule covers the whole table: **a number is quotable only if this file, or an entry in
`docs_decisions.md` at D51 or later, still asserts it.**

---

## 2. Where the project stands

Three claims. Keeping them separate is what makes the strong one credible; the project's
contribution is the third, and it needs the first but not the second.

### Claim 1 — the decomposition is exact and validated ✅ **closed**

- Recourse matches an independent monolithic MILP to the cent at three scales (**4183.24**,
  651.36, 1674.11), before and after the D30 de-layering — which is what proves the
  de-layering did not change the model.
- **Phase 5 (D62) is done**: `|z*_Benders − z*_monolith| ≤ ε`, two instances × two cut
  modes, both sides proving optimality rather than stopping on the clock. This was the
  missing rung — every earlier exactness claim rested on an *inequality*, which catches the
  D30 class and nothing else.
- Since D67 the gate also passes on a **second solver** (HiGHS), and **4183.24 regenerates
  under it** to two decimals. Agreement across two independent implementations *and* two
  independent solvers rules out a CPLEX-specific artefact.
- Exactness conditions **E1–E4 are tests, not assumptions**.
- **263 passed, 9 skipped** (D71 added 4 for the runtime-split instrumentation below). The 9
  are CPLEX-specific machinery (the branch-and-cut callback, `CPXPARAM_*` name resolution); no
  formulation invariant is among them.

*Caveats that must travel with claim 1:* every `mw`-labelled number written before D61
predates the Magnanti–Wong guard fix; D42's dominance margins are withdrawn and unreplaced
(D65); the θ`[ω,d]` cell and the anchor A/B in D64 are unmeasured.

### Claim 2 — the decomposition is not competitive on this family ✅ **closed, by three refutations**

| Measurement | Decomposition | Direct solve |
|---|---|---|
| Q=3, minute recourse (D56) | LB 219.74 / UB 299.37, 27% gap, **301.4 s** | minute monolith **293.37, proven optimal, 0.8 s** |
| Dantzig–Wolfe root, Q=3–5 (D57/D58) | 61–75% of optimum, 50–208 s | CPLEX's own root: **95–96%, 9–15 s** (D60) |
| Q=3 slot model | **1148.65**, a 46% internal gap, 1520 s — **69.2% of the optimum** | **optimal 1658.86 in 947 s** (D50) |

**Which bound fails is the decisive detail.** The upper bound is 299.37 against a true
optimum of 293.37 — a schedule within **2%**, found almost immediately and unprovable. **The
lower bound is what is stuck**, at 74.9%. So as a *heuristic* the decomposition is fast and
good; as a *proof system* on this family it is not competitive with solving the model
directly. And any future work on the bound must attack the **relaxation**: another family of
cuts into the same master is aimed at the wrong object.

The sharpest form, and the most publishable methodological observation here: **the bound
lives at the fractional LP root, and cuts do not move it because CPLEX's own root cuts had
already moved it 96% of the way** before any of this project's machinery was invited. D40
found "nothing moved the bound" — it had already been moved.

The "hard band" fallback was tested rather than assumed: the band exists at Q=5–6, and the
method's advantage inside it is measured at **zero** (D59).

### Claim 3 — the contribution is evaluation fidelity 🟡 **measured; the novelty claim is narrowed, not clear (D69)**

*Slot-aggregated vehicle scheduling models misprice their own schedules and choose worse
ones, because a slot is too coarse to say who is reachable inside a service-time promise. A
minute-level operational recourse corrects both, at under 1% of iteration time, needing no
new cut machinery, and leaving the decomposition and its bounds intact.*

Valuation error and decision error are independent and must be reported separately: the
first misleads whoever reads the output, the second is what the operator loses.

| Quantity | Passenger-minutes |
|---|---:|
| what the slot model **claims** its schedule costs | 11 990 |
| what that schedule **really** costs | 9 326 |
| what the **best achievable** schedule costs | 8 794 |

Valuation error **+28.5%**; decision error **+6.0%**. On the waiting term alone the
valuation error is **66–86%** (38.24 min/pax reported against a true 20.4–23.2); the
objective hides it because at the default penalty it is 93% unmet-demand headcount and only
6.8% waiting.

Across five de-aligned demand shapes and three grids, decision error is 3–22% under
`midpoint` and 8–38% under `end`, and **does not vanish as the grid refines** — grid
refinement and minute-level valuation are independent levers, and the trend is **not
monotone**, so no trend may be read off two columns.

**Quote 3.84% as the operational figure.** One schedule serving four scenarios gains 3.84%,
against 8–50% tailored to one, and is actively worse on one member of the set. Single-scenario
gains are not the operational result.

**It survives the decomposition** (D52): a minute recourse keeping its capacity rows indexed
by departure slot returns exactly one dual per slot — the same object the slot subproblem
returns. No projection machinery. Convergence per iteration is indistinguishable (69.0% vs
66.4% at iteration 14), and the subproblem is **under 1% of an iteration**.

**Departure placement is a first-class modelling decision** and moves the measured gain
between 0% and 49% (D54). `end` is what the demand aggregation implies; `start` assumes the
bus leaves before the passengers it is collecting have arrived and is not defensible.

**The scoping constraint.** The slot model **overstates** cost under all three conventions,
so the slot optimum is an **upper** bound on the minute-level optimum. Therefore *a
slot-level Benders lower bound bounds the slot problem and says nothing rigorous about the
minute-level problem.* Reporting one against minute-level optimality would be a claim the
construction does not support.

**The literature check is done (D69), and it removes the headline claim.** Discretisation
error in vehicle scheduling is not only well-trodden, it has an **exact** treatment in this
exact family: Dynamic Discretization Discovery for the multi-depot vehicle scheduling problem
with trip shifting, whose premise — letting departure times deviate a few minutes from the
timetable so new trip combinations become feasible — is ours almost verbatim. So *"slot
aggregation misprices its own schedules and nobody notices"* is answerable with one citation
and **must not be written**. Separately, the valuation/decision split is routine practice in
energy-systems time-series aggregation, and aggregation error bounds date to Zipkin (1980);
neither is ours to introduce.

**What survives is a combination, and it is narrower:** (i) here the aggregation
**over-approximates**, where DDD's coarse model is a *relaxation* — so the refinement
machinery does not transfer, and that same property is why a slot-level bound says nothing
about the minute-level problem; (ii) the fine resolution is confined to the **recourse**, and
is dual-compatible with a coarse master that never changes, where DDD refines the network and
therefore the master; (iii) grid refinement does not substitute for it and is not monotone
(D55), and the placement convention moves the gain 0–49% (D54). Item (ii) is load-bearing and
is the one D69's search covers least well.

**Residual, and it no longer blocks:** read both DDD papers in full before writing the
related-work paragraph, and confirm neither confines refinement to the recourse.

### The stochastic-robustness result set is regenerated (D70)

A related but separate question, outside this repository until now: what does it cost to run
**one** schedule against four demand scenarios instead of solving each scenario's own
deterministic optimum? A conference-era answer to that question exists and is void — it
predates both the `p_minutes` correction and minute-level valuation. **D70 regenerates it:
hedging costs 0.9% in passenger-minutes**, at `p_minutes=56`, minute fidelity, weighted equally
across `base`, `temporal_noise`, `return_peak_advanced` and `midday_surge`. A caveat travels
with it — the per-scenario "oracle" is itself only slot-optimal, not minute-optimal, so on two
of the four scenarios the hedged schedule outperforms its own scenario's oracle once both are
priced honestly. See D70 for the full table and `scripts/stochastic_robustness.py`.

---

## 3. The withdrawn register — every dead number in one place

This table is the point of the document. If a figure appears in the left column anywhere —
in `docs/`, in a draft, in a slide — it is dead, whatever surrounds it.

| Dead | Live | Withdrawn by |
|---|---|---|
| monolith **1569.44 in 39 s** | **1658.86 in 947 s** | D50 |
| LP root **794.6245** (`794.624549571966`) | **794.7795573706986** — the old figure was a *truncated* iterate that stopped on `iteration budget (150)` while still generating cuts, not a root | D64 |
| reference optimum **4190.74** | **4183.24** — 4190.74 was a log-parsing artefact of a Benders run of this same code, so the guard was circular | D50 |
| **"best bound 0.35"**, internal gap 99.9%, "the failure is structural" | Measured with 14–19 cuts because `lp_phase_max_iters: 10` sampled the flattest point of the curve. Root is 794.78; CPLEX's own root reaches 95–96% | D40, D45, D60 |
| **46% of the monolith's optimum** and every "% of optimum" against 1569.44 | 1148.65 is **69.2%** of 1658.86. The 46% figure is the master's *internal* LB/UB gap, not a distance to the optimum — the two were conflated | D54 |
| **98 tests** / 49 / 169 / 196 / 233 / 248 / 268 (259 passed, 9 skipped) | **272** (263 passed, 9 skipped) | D67, D68, D71 |
| Magnanti–Wong **dominance margins** (D42), "uniform ~21, out_only ~30" | **Withdrawn and unreplaced.** MW had been silently declining whenever no RET capacity existed in slot 0, while reporting its mode as `mw` throughout | D61, D65 |
| master ≈ **85.7% of runtime**, "so buy master seconds" | Not reproduced and not meaningful for current code. The lever it motivated (M1) was **measured and rejected**: master phase 18.2 s → 49 s, bound *worse* | AUDIT_v4 §4 |
| `except Exception` count **165** | **231** (225 + 6). Only **3** have a correctness argument | HANDLER_CENSUS (outside this repo) |
| `mw_fdiff_fallback` | Replaced by **`mw_dual_fallback`**. The old behaviour discarded a valid cut it was already holding in favour of finite-difference slopes carrying no guarantee, then voided the run's bound | S1 |
| `W_slots = ceil(Wmax/δ)` | **`floor`**. With `ceil`, asking for 45 min at δ=30 **granted 60** | S6 |
| `p: 50` quoted as a policy | **`p` is in slot units.** `p: 50` is 1500 passenger-minutes at δ=30 and 750 at δ=15 — two objectives live at once, neither comparable. The operator's indifference gives **`p_minutes ≈ 56`**; configs were running **27×** that | D53 |
| "the project is closed" (`docs/RESULT.md`) | **Open.** RESULT.md lives in one commit, `34a0eeb`, which is not an ancestor of `main`; `main` ran twelve more days past it | D67 §1 |

Two reading rules, both learned expensively:

- **Reproducible and truncated are not exclusive.** A bound can reproduce to the last digit
  and still be a truncated iterate. Check the log for the budget line before calling a
  number a root.
- **A guard whose expectation comes from the thing it guards proves nothing.** 4190.74 was a
  Benders run of this same code; the cut-tightness test re-derived an identity the code had
  imposed.

---

## 4. D-number remapping

`docs_decisions.md` is **not** renumbered — a register that renumbers itself invalidates
every citation ever made against it. Documents outside this repository allocated D48–D50
independently, so a D-number taken from one of them must be remapped first:

| Cited as | Means, outside this repo | Means **here** |
|---|---|---|
| D48 | per-vehicle trip cap implied by the LP relaxation | **D68 §3** (ported). This register's D48 is the signature/fibre design |
| D49 | battery block to the subproblem gains nothing | **D68 §4** (ported). This register's D49 is the window trip caps |
| D50 | `use_dual_slopes` split into two keys | **D68 §2** (superseded — `main` keys the model switch off the resolved `cut_mode` instead). This register's D50 is the Q refutation and the monolith's return |

**Action for report v2:** delete §0.3 and the "no manuscript may cite a bare D-number"
consequence with it. The premise is false for this register. Remap the three numbers above
and cite bare D-numbers against `docs_decisions.md` normally.

**Action for report v2 §6:** it lists as unverified eleven commit hashes and four documents.
Resolved: six hashes verify with matching dates and subjects (`01b39e8`, `11768ea`,
`4257a7d`, `e00c1bc`, `fab1bfb`, `13fcf90`); **five are not valid objects in any ref here**
(`dbc01e2`, `1b8fdb3`, `b0ed6bf`, `a423058`, `bf504d5`) and must be dropped or re-sourced.
The four documents are genuinely absent from this repository — including
`BENDERS_CORRECTION_PLAN.md`, which v2 ranks first and declares governing.

---

## 5. What is open

Two levers, and one reading task (D69 §8: the two DDD papers in full). Everything else on the bound is closed **by
measurement** and must not be re-attempted: cumulative-prefix symmetry; `b >= L·yRET` (M1);
branch-and-Benders-cut for the *lower* bound; tightening `per_iteration_mipgap`; the
per-iteration time limit as a tuning knob; Dantzig–Wolfe on the vehicle index; window trip
caps in the master; more master seconds; the down-set recourse cut; the per-vehicle trip cap
(D68 §3); moving the battery block down (D68 §4).

**F2 — placement freedom in the subproblem, Design 3 only.** Fix an offset grid `O ⊂ [0,δ)`
once at load, so candidate departure minutes are **constants**. One capacity row per slot,
same count, same right-hand side, same dual as today, with `Y` in `b` only — **E2 holds by
construction** and the cut machinery is untouched. It is a *relaxation* (two passengers on
one physical departure may board at different candidate minutes), so `Q_relaxed ≤ Q_true`
and a cut from it is a valid lower bound on `Q_true`; the **upper bound must not come from
the relaxed model**. Two designs to refuse: a continuous offset as a variable (second-stage
makes the recourse non-LP; first-stage puts a decision in `A`, which is D30 verbatim), and
pre-enumerated minutes with binary selection (makes the second stage a MILP).
*Prerequisite S3, the core-point projection, is done (D63).* **Not implemented.**

**Stabilisation, level method.** The one lever whose diagnosis matches the observed symptom:
Benders iterations on a 150-cut master oscillate in **1064–1090 with no trend**, a spread the
size of run-to-run noise, at a 46% internal gap. Prefer a level method to a trust region —
the target `L = LB + λ(UB − LB)` needs only bounds the loop already tracks, where a trust
region needs a norm on binary schedules. **Run it under the Phase-5 gate**, so a
stabilisation bug cannot be mistaken for a convergence result. **Not implemented.**

**Falsifiers, fixed in advance.** F2 is refuted if cut strength at a fixed budget does not
improve beyond noise. Stabilisation is refuted if the 1064–1090 band persists with no trend.
And **any** bound work is refuted *as a method claim* if it stays an order of magnitude
behind a monolith that solves the test point exactly in 947 s — a better Benders bound that
is still far behind the monolith is a result about Benders, and must be reported as one.

Smaller open items: the θ`[ω,d]` cell and the anchor A/B (D64) — the anchor A/B has **no
valid measurement**, 0.299 vs 0.314 is void and unreplaced; D42's margins need re-measuring
after the D61 fix; folding F1–F8 into the formal formulation, which is outside this repo and
currently disagrees with the code, **the code being the more correct of the two**.

---

## 6. Running it locally

### 6.1 Install

Python **3.10 or newer** (3.11 is what the current numbers were produced on).

```bash
git clone https://github.com/f-sartori-v/mobauto2-benders.git
cd mobauto2-benders

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e .                    # pyomo + pyyaml
pip install pytest                  # the suite runs under pytest or unittest
```

### 6.2 Pick a solver

**Either backend runs the whole test suite.** This is new in D67 — every gate used to name
CPLEX literally, so a machine without a licence skipped 63 tests and still printed green.

| | CPLEX | HiGHS |
|---|---|---|
| Install | IBM ILOG CPLEX Optimization Studio, then `python setup.py install` from its `python/` directory. The free Community Edition caps model size and **will not run the Q=3 instances** | `pip install highspy` |
| Licence | Academic (free, via IBM Academic Initiative) or commercial | None |
| Runs the suite | yes | **yes** |
| Runs branch-and-cut | yes | no — needs a lazy-constraint callback, which HiGHS has no interface for |
| Reproduces the archived timings | **yes — only CPLEX does** | no |

Which one you need depends on what you are doing:

- **Checking the work is sound, or changing code** → HiGHS is enough, and is one `pip`
  command.
- **Producing any number that goes in a table beside an existing one** → **CPLEX, and only
  CPLEX.** HiGHS is a different instrument; a timing from it is not comparable with anything
  in `docs/`, and mixing them in one table is the same error as quoting `p: 50` without its
  unit.

Verify the backend is visible:

```bash
python -c "import pyomo.environ as pyo; print('cplex_direct', pyo.SolverFactory('cplex_direct').available(False)); print('appsi_highs ', pyo.SolverFactory('appsi_highs').available(False))"
```

### 6.3 Run the tests — do this first

```bash
python -m pytest tests/ -q
```

Expect **263 passed, 9 skipped** on HiGHS in about a minute; the 9 skips are branch-and-cut
and `CPXPARAM_*` name resolution. On CPLEX, all 272 run.

CPLEX is preferred automatically when installed, so a licensed machine measures exactly what
it measured before. To force one:

```bash
MOBAUTO2_TEST_SOLVER=appsi_highs python -m pytest tests/ -q
```

A pin that is not installed **raises** rather than skipping — naming a solver and quietly
getting a different one is how a result gets attributed to the wrong instrument.

**Read the skip lines, not just the count.** A skip in this suite is not a pass; several of
the invariants switch *themselves* off when their precondition is missing, which is by
design and is also how 63 of them went unrun for months.

### 6.4 Run the solver

```bash
mobauto2-benders --config configs/baseline_d9.yaml run     # or: python -m mobauto2_benders.cli
mobauto2-benders --config configs/baseline_d9.yaml info    # show the resolved configuration
mobauto2-benders --config configs/baseline_d9.yaml validate
```

**Every shipped config names a CPLEX plugin.** On a machine without a licence, add
`--solver`, which repoints the master, the subproblem and the seeding LP phase together:

```bash
mobauto2-benders --config configs/phase5/tiny_mw.yaml run --solver appsi_highs
```

All three move together on purpose — a master on one backend and a subproblem on another is
a configuration nobody has measured. The run manifest records which backend actually ran and
its version, so an overridden run says so in its own provenance instead of looking like the
config it started from. `--solver cplex_persistent` is refused: branch-and-cut builds its own
persistent solver for the tree, and repointing the master's would change the cuts the tree
starts from.

`configs/default.example.yaml` is the annotated reference and the source of truth for what
every key means — including which keys change a reported bound. Copy it rather than editing
`configs/default.yaml`, which is the live experiment file.

The monolith, which is the reference instrument rather than an alternative solver:

```bash
mobauto2-milp --config configs/milp/baseline_d9_monolith.yaml run
```

Useful starting points:

| Config | What it is |
|---|---|
| `configs/baseline_d9.yaml` | The frozen regression baseline. Its fingerprint must not move |
| `configs/baseline_d9_p56.yaml` | The same instance at the **operator's actual penalty**, `p_minutes ≈ 56` |
| `configs/phase5/tiny_*.yaml` | The exactness gate: tiny, fast, and the thing to run after touching cut generation |
| `configs/milp/phase5_tiny*_monolith.yaml` | The other side of that gate |
| `configs/phase1/lp_only_150.yaml` | The converged LP root, 794.7795573706986 |

### 6.5 Before you quote any number it produces

Six conditions travel with every reported figure, and all six are in the run manifest
(`manifests/`), so the check is mechanical rather than a matter of memory:

1. **`p_minutes`** — `p` in minutes, never bare. `p: 50` is not a policy.
2. **`concurrency_penalty`** — active in the objective, absent from the published formulation.
3. **The cut budget** — the largest single source of wrong conclusions in this project.
4. **The subproblem mode**, and whether its cuts support a bound at all.
5. **`clock_truncated_master_solves`** — a wall-clock budget and a reproducible number are
   incompatible. Three runs of one config at a binding 15 s cap gave LB 2333.29 / 2153.79 /
   2175.87; the same config at a non-binding cap reproduced to the last digit, twice.
   `CPXPARAM_Threads: 1` does **not** fix this: it removes the nondeterminism of parallel
   MIP, not that of the clock.
6. **The departure placement convention** — it moves the measured gain between 0% and 49%.

Budget experiments by `max_iterations`, not by seconds. Budget simulation runs by time and
say so.

### 6.6 If you are extending the model

`DESIGN_DD_v1.md` §4 states E1–E4 as tests. **E2 is the dangerous one**, and the minute-level
extension is where it gets broken: if the master's `y` decides *which minutes* a vehicle is
available, and availability decides which rows the subproblem contains, then `y` has entered
`A`, not `b`. That is D30 verbatim, it cost six months of invalid lower bounds, and it left
no trace in any output — every run reported healthy provenance throughout. The extension is
valid **if and only if** the minute grid is fixed and `y` scales the right-hand side:

```
C_d[m] = S * sum_q sum_tau a_d[tau,m] * y_d[q,tau]        for a CONSTANT 0/1 matrix a
```

With deterministic trip duration `a` is constant; with travel-time scenarios it is constant
*per scenario*, which the multi-scenario path already handles. **It breaks the moment trip
duration becomes a decision variable.**

One method note worth keeping: three of the correctness findings came from checking a change
that *should have been a no-op*. N6 was a two-line fix that immediately exposed C3; C4
surfaced because a pure deduplication produced **disjoint** bound intervals; H8 surfaced
because a passenger table looked implausible. **A refactor that should change nothing is a
cheap probe for latent unsoundness — run it, then check an invariant, not just the diff.**
The matching failure to avoid: twice, a symptom disappearing was read as the problem being
solved.

---

## 7. What this document does not establish

- **No new measurement is in it.** Every figure is quoted from the decision-log entry that
  recorded it. §2's claim-1 additions (Phase 5 and 4183.24 under HiGHS) are soundness
  results, not performance results.
- **No competitiveness number moves.** HiGHS is not CPLEX; every timing in `docs/` stands on
  the CPLEX run that produced it.
- **The literature check (D69) is a search-level check, not a systematic review.** Abstracts
  were read, not full texts; no bibliographic database was queried; and the one item it
  covers least well — a fine recourse dual-compatible with a coarse master — is the one the
  narrowed novelty claim rests on. Absence of evidence there is weak evidence.
- **Neither live lever in §5 is implemented**, so neither falsifier has been exercised.
- **Scope limits stand as limits, not as an argument to keep going:** one instance size, four
  scenarios, a single monolith baseline, and scenario scaling never tested.
