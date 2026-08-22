# Review of the T.5.4 Scientific Report, against the repository it describes

**Status of the report under review:** substantially sound in framing, **not publishable as
written**. Six of its headline numbers are withdrawn by this repository's own decision log,
four of its high-severity findings are cited under identifiers the audit does not use, two
of its four Stage-0 blockers specify machinery that does not exist in `src/`, and its
top-priority recommendation would reverse a change that was measured and kept.

**Why this happened, and it is not the report's fault.** The report states in its provenance
note that `docs/docs_decisions.md`, `docs/BENDERS_SPEC_v4.md` and the design handouts could
not be fetched. It was written against `README.md` plus a task brief. The decision log is
198 KB and 3384 lines and runs to **D65**; the report's evidence stops at **D50**. Fifteen
decisions, three of which close a whole stage of work and one of which withdraws the
report's single most-quoted number, were invisible to it.

This document is the revision that access produces. It is organised as the report's own
sections so the two can be read side by side.

**Method.** Every correction below names the file and decision that produces it. Where the
repository and the report disagree, the repository governs and the conflict is stated.
Where the repository is *itself* inconsistent, that is recorded as a defect against the
repository, not against the report — four such sites were found and fixed in this change.

**One limit on this review, stated first.** CPLEX and Pyomo are not installed in the
environment this review ran in, so **no run was executed and nothing here is a new
measurement.** Every number is quoted from a decision-log entry that records one. Claims
that would need a run to settle are marked `NOT RE-MEASURED` rather than asserted.

---

## 1. Corrections register — numbers the report publishes that are void

| # | Report states | Repository establishes | Source |
|---|---|---|---|
| C1 | LP root relaxation = **794.6245**, "reproducible across three runs" | **Not a root.** The LP phase stopped on `iteration budget (150)` with a cut still being generated (`cuts=1`, `rel_improve=1.11e-06`). The converged root is **794.7795573706986**, reached on `no cut generated`. Both reproduce; only the second is a root | **D64** |
| C2 | Monolithic MILP reference = **1569.44**, proven optimal in **39 s** | **1658.86 in 947 s.** Every "% of the monolith's optimum" computed against the old pair is wrong | **D50** |
| C3 | "the best Benders run reaches a **46% Benders gap**" | Void — it is `1 − 1148.65/1569.44` on the withdrawn pair. Against the real optimum the decomposition reaches **69.2%** of it after 1520 s | **D50, D54** |
| C4 | Test suite = **98 tests, ≈50 s** | **248 tests, 57 s** | **D65** |
| C5 | Magnanti–Wong Pareto selection "is applied", cited as a strength of the implementation | MW's dominance margins of ~21 and ~30 are **withdrawn**. MW was *silently failing* on the fixture that measured them — its weak-duality check evaluated a Pyomo expression containing a variable the backend never sends, so it raised, and **the cut labelled `mw` was the finite-difference fallback**. Corrected margins are `[0, 0, 0, 0, 3.86e-05]`: MW does select (slopes differ by up to `S=15`), and **the selection is worth ≈0 on the only fixture the repository has** | **D65**, and D61 |
| C6 | commit `a2d9e97` for D30 — `[brief, UNVERIFIED]` | **Verified.** `a2d9e97 fix(subproblem)!: de-layer capacity so Benders duality actually holds` | `git log` |
| C7 | `subproblem_impl.py` 3267 → 1828 lines — "not independently verifiable" | **Verified** (AUDIT_v4 §2, D9). But the file is now **3088 lines**: the minute recourse (D51–D55) restored about 70% of the deletion. The report's framing of this as a settled simplification is stale | AUDIT_v4 §2, `wc -l` |
| C8 | "master ≈85.7% of runtime" — `[UNVERIFIED]` | Stronger than unverified: that figure appears in the audit only as a **quotation from v3 attached to a recommendation that was measured and rejected** (item M1 — the proposed constraint made the master phase 18.2 s → 49 s and left the bound *worse*). It should not be repeated even as a caveat | AUDIT_v4 §2 |

**C1 and C2 are load-bearing.** The report's §3.1 results table, its §3.2 "negative result,
stated honestly", and its Stage-1 premise that the root relaxation "sits near zero" all rest
on one or both. They must be regenerated before anything is published.

**A defect in this repository, found while checking C2 and fixed in this change.** `README.md`
corrected 1569.44/39 s to 1658.86/947 s in a block quote, and then **restated the withdrawn
"39 s" three paragraphs later** in its live competitiveness verdict. The report read the
second sentence and reproduced it faithfully. Three further sites carried the same withdrawn
figure or a stale test count; all four are corrected here. This is the repository's own
reading rule — *"kept so the claim is not quoted again"* — failing at exactly the point where
an outside reader picks the claim up.

---

## 2. The N-series does not exist

The report elevates **N1, N11, N12 and N14** to "four high-severity audit findings [that]
must be closed before publication", and builds its entire Stage 0 on them.

`docs/AUDIT_v4.md` contains **no N11, N12 or N14.** Its **N1** is a duplicate-helper finding,
already closed — *"Hoisted. v3's mechanics were wrong: the two copies are in mutually
exclusive branches each ending in `return`, so neither shadowed the other — duplication, not
dead code."* The only other `N1` in the tree is D5, "the dominance threshold (audit N1)",
a different and also-closed item.

The four *topics* the report attaches to those labels are real and worth work. The
identifiers are not the repository's, and a reader who takes the report to the audit to check
them will find nothing. **Renumber them or state them as the report's own findings.** They
are handled on their merits in §5 below, under names rather than numbers.

---

## 3. Three of the four Stage-0 blockers describe code that does not exist

`grep` across `src/` for `rolling`, `roll_horizon`, `commitment`, `carried`, `B8`, `B9`,
soft/hard capacity toggling, policy vectors, and cut-pool flushing returns **nothing but
prose in comments.** Concretely:

- **There is no rolling horizon.** The horizon is frozen at **10 h, single-shot**, with a
  one-slot boundary buffer (`BENDERS_SPEC_v4` §1.1, D6). The report's §2.1 rolling geometry —
  24 h master, 6 h blocks advancing +6 h, breaks at 03:00/09:00/15:00/21:00 — is design intent
  from the brief. It is not implemented, and the report presents it as the engine's geometry.
- **There is no B1–B9 architecture and no commitment carrying.** Nothing in `src/` reads or
  writes a pickup commitment.
- **There is no policy-toggle family.** `W_max` and `p` are plain scalars validated in
  `config.py`; there is no hard/soft capacity switch and no cut pool keyed by policy.

The consequences for the report's Stage 0 are direct:

**"N11" (per-roll-under-carried-commitments scoping) has nothing to scope.** With no rolling
horizon, the qualifier is vacuous — and worse, printing it would *assert* a rolling structure
the code does not have. There **is** a real scoping caveat that the report should carry
instead, and the repository states it twice: the monolith benchmark is independent of the
**decomposition** (it pins `theta` to the exact recourse by equality, so the whole D30 defect
class cannot reach it) but **not of the formulation** — `mobauto2_milp/model.py` is a second,
hand-synced copy of the first stage. *A defect present in both copies is invisible to every
check in this repository.* That is the sentence every optimality claim needs, and the report
does not have it.

**"N14" (flush the cut pool across soft-constraint toggles) is not merely unbuilt — it is
contraindicated.** D30's fix was to *de-layer capacity* so that `y` enters the recourse
**through the right-hand side only**; `RESEARCH_NOTE_v2` §6 states the condition and records
that violating it *"cost this project six months of invalid bounds"*, and four tests now
enforce it. A runtime hard/soft capacity toggle changes which rows exist as a function of
configuration, which is the adjacent failure mode. If policy toggles are wanted, the design
question to answer first is how to keep them out of the recourse's constraint *structure* —
not how to flush a pool afterwards.

---

## 4. What the report could not see: D51–D65

The report's evidence ends at D50. Everything below post-dates it, and three items reverse
recommendations the report makes.

### 4.1 The bound is stuck; the schedule is not (D56)

On `baseline_d9` at Q=2 — *the smallest instance in the project* — against the **minute-level
monolith**, i.e. the same model solved without a decomposition in between:

| arm | objective | status | wall |
|---|---:|---|---:|
| monolith | **293.37** | proven optimal | **0.8 s** |
| Benders | LB 219.74 / UB 299.37 | 27% gap, 34 iterations | 301.4 s |

390× slower. But the decisive observation is *which* number is bad: **the decomposition finds
a schedule within 2% of optimal and cannot prove it.** As a heuristic it is quick and good;
as a proof system it is not competitive. Every previous speed comparison in the project used a
baseline that was not this model.

This reframes the report's §3.2 wholesale. "Mathematically valid cuts + a non-converging
master" is right, but the actionable form is narrower: **only an attack on the relaxation can
work.** Anything that buys upper bound is buying the quantity that is already fine.

### 4.2 Stage 2 closed — the down-set cut is valid and dominated (D56 §3)

The logic-based down-set cut proposed in `DESIGN_DD_v1.md` §3.2 was worked out, found to admit
a cheaper encoding than designed, shown valid — **and dominated anyway.** Recorded as *do not
build*.

### 4.3 Stage 3 closed — CPLEX's own root cuts beat the reformulation everywhere (D60)

Root against root, T=44, four scenarios, `p_minutes = 56`:

| Q | compact LP | DW root | **CPLEX root** | optimum | DW as % | **CPLEX as %** |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 248.98 | 281.69 | **415.19** | 431.54 | 65.3% | **96.2%** |
| 4 | 173.17 | 183.11 | **289.37** | 302.05 | 60.6% | **95.8%** |
| 5 | 173.15 | 173.15 | **219.55** | 231.25\* | 74.9% | **94.9%** |

\* incumbent; the Q=5 monolith stopped on the clock.

**CPLEX reaches 95–96% of the optimum at its root in 9–15 s.** Dantzig-Wolfe reaches 61–75% in
50–208 s. And the DW lift **collapses to +0.0% at Q=5** — largest where the monolith is
fastest, zero where the monolith fails, *"the opposite of the shape a method needs."*

This is the most important thing the report is missing, and §6 explains why it changes the
Stage-1 recommendation rather than merely adding to it.

### 4.4 A target regime exists (D59)

| Q | objective | bound | status | wall |
|---:|---:|---:|---|---:|
| 3 | 431.5433 | 431.5433 | optimal | 181.0 s |
| 4 | 302.0467 | 302.0464 | optimal | 314.8 s |
| **5** | 231.2500 | **226.0703** | **hit the 1200 s cap** | 1208.7 s |

**At Q=5 the monolith does not close** — 2.24% relative gap on the clock. The report's closing
recommendation reads *"If the monolith continues to win at all instance sizes the operation
will ever require (Q ≤ 5, single corridor), ship the monolith."* The repository has already
tested the boundary of that condition and **found it false at Q=5**, which is inside the range
the report names. The caveat D59 attaches: `Q` grows while `T`, demand and horizon stay fixed,
so this measures difficulty in `Q` for fixed demand — horizon and demand volume are untested.

### 4.5 Exactness is proven at small scale (D62)

The repository had only ever asserted `LB ≤ (known feasible objective)`, which catches the D30
class and nothing else — *"It is satisfied by a decomposition that converges to the wrong
place."* D62 adds the gate that was missing, `|z*_Benders − z*_extensive| ≤ eps`, on two Q=1
instances (one slack, one where capacity binds), under **both** cut modes:

| cell | monolith | Benders `dual` | Benders `mw` |
|---|---:|---:|---:|
| slack | 12.02 | 12.019999999999989 | 12.019998999999995 |
| tight | 36.086666666666667 | 36.086666666666666 | 36.08666566666666 |

All six proved optimality. The report's validation table lists "Optimality certification —
**Open**" with no mention that the decomposition is now *proven exact against the extensive
form* where both can be solved. That is a materially different status, and it is the correct
answer to "is the decomposition converging to the right place".

D62 also states its own limit precisely, and the report should adopt the wording: it
establishes that the **decomposition** is exact with respect to the formulation. It does not
establish that the **formulation** is right.

### 4.6 The penalty used in every published table is 27× the operator's policy (D53)

`p = 50` at 30-minute slots is **1500 passenger-minutes**; the operator's stated policy is
about **56**. At 1500 the model *"will delay 14 passengers by an hour rather than leave one
behind"*, and waiting is **4.1% of the objective** — a regime where waiting barely matters.
`p_minutes` is the resolution-independent form and exists precisely for this.

The report's §4.1 proposes a 6×5 sweep of `p ∈ {10,25,50,75,100,150}` × `W_max ∈
{30,45,60,90,120}` **with p = 50 as centre**. Every one of those thirty cells sits in the
wrong policy regime, and `p` in slot units is not comparable across the resolutions the same
report wants to vary. The sweep as specified would produce thirty numbers about a policy the
operator did not ask for.

### 4.7 Departure placement is a first-class modelling decision (D54, RESEARCH_NOTE_v2 §3)

The convention for placing a departure within its slot (`start` / `midpoint` / `end`) moves
the measured multi-resolution gain from **0% to 49%**, and the D47 baseline optimum was wrong
by 5.4% because of it. The report does not mention placement at all. It belongs in §1.3 as a
parameter, not in a footnote.

### 4.8 The honest multi-scenario figure is ~4%, not 8–50% (D55)

Single-scenario minute-level gains of 8–50% attenuate **sixfold** to **3.84%** when one
schedule must serve four scenarios — and on one scenario the minute-optimised schedule is
*worse*. Since every Fase 1 config is four-scenario, *"single-scenario gains must not be
quoted as the operational result."*

---

## 5. Stage 0, reformulated

The report's Stage 0 has four items. **One is already closed and its recommendation is
backwards; two specify unbuilt machinery; one is aimed at the wrong policy regime.** Restated
against what is in the tree:

### S0.1 — Granularity ("N12"). **CLOSED. The report's recommendation is the wrong option.**

The report calls this *"the top priority … the same defect class that loosened the deleted
elastic path"*, and recommends **Option B**: aggregate demand to slot resolution, generate
cuts natively there, and use the minute subproblem only as an upper-bound evaluator.

The repository answered this in D52 and `RESEARCH_NOTE_v2` §5, under the heading **"Cut
projection is free"**:

> If the minute recourse keeps its capacity rows indexed by **departure slot** — capacity
> `S * Y_d[tau]` at each slot, demand rows on minutes — it produces exactly **one dual per
> slot natively**. The cut is the same object the slot subproblem already produces […] and
> cut construction, aggregation, the `q`-invariance check, validity classification and the
> master rows are all untouched.

So the report's Option A is implemented, needs **no projection machinery at all**, and costs
**under 1% per iteration** with unchanged convergence per iteration (D52). The dichotomy the
report poses — either project and risk loose cuts, or aggregate and lose exactness — is false;
the third door was taken.

And **Option B is the thing that was measured as harmful.** Slot aggregation misprices its own
schedules by **28.5% on the objective and 66–86% on waiting** (D53–D55), and it survives a
10-minute grid, so it is not an artefact of coarse slots. Adopting Option B now would
reintroduce a measured error to solve a problem that no longer exists.

**What is genuinely open here** is not validity but **units**: the minute recourse must be
scaled by `1/slot_resolution` or the recourse outweighs the departure regularisation and the
concurrency penalty by exactly that factor, *"and quietly changes how ties between schedules
are broken."* Three units defects were fixed on this path already (commit `e00e7d9`). That is
the thing to test, and it is a test, not a redesign.

**Action:** delete the Option A/B decision from Stage 0. Replace it with a citation of D52 and
`RESEARCH_NOTE_v2` §§5–6, and carry §6's four enforced exactness conditions as the validity
argument.

### S0.2 — The (p, W_max) sweep ("N1"). **OPEN, but re-scoped, and not as specified.**

Two things are wrong with the sweep as the report designs it.

*It is centred on the wrong policy regime.* Per §4.6, re-centre on **`p_minutes ≈ 56`** and
state the penalty in passenger-minutes throughout. `p` in slot units is not comparable across
resolutions and must not label an axis in a study that also varies resolution.

*The repository deliberately re-scoped it, and the report should engage with that rather than
reinstate the old plan.* AUDIT_v4 §3.1 records `p` and `W_max` as **given inputs**, with the
sweep moved to the structural parameters `slot_resolution` and `Q`. The finding was that fleet
size dominates because the objective is dominated by the unserved penalty — which is itself an
artefact of the 27× penalty, so the re-scoping decision is **due for re-examination in the
corrected regime.** That is a legitimate reason to revive a sensitivity sweep; "the report
recommends it" is not.

The report's **acceptance criteria are the strongest part of its Stage 0 and should be kept
verbatim** — monotonicity of unserved demand in `p` and in `W_max`, the rejection threshold
reported as a 2-D contour rather than a point (which is also D5's finding, arrived at
independently), and no cell reporting a lower bound above the monolith's optimum.

**Add two axes the report omits:** `departure_policy` (§4.7 — worth 0–49%) and the number of
scenarios (§4.8 — worth a factor of six).

**Add one budgeting rule, non-negotiable 10:** a wall-clock budget and a reproducible number
are incompatible. Budget the sweep by `solver.max_iterations` with the time limits set so
neither binds. A 30-cell grid budgeted in seconds produces 30 single draws.

### S0.3 — Optimality-claim scoping ("N11"). **REPLACED.**

Per §3: there is no rolling horizon, so the per-roll-under-carried-commitments qualifier
asserts a structure the code does not have. Replace it with the formulation-independence
caveat from `README.md` and D62 §5, which is the caveat that is actually true:

> Every bound is checked against a monolithic MILP that pins `theta` to the exact recourse by
> equality, so it is independent of the **decomposition** — the D30 defect class cannot reach
> it. It is **not** independent of the **formulation**: `mobauto2_milp/model.py` is a second
> copy of the first stage, kept in sync by hand, and a defect present in both copies is
> invisible to this check.

The report's proposed test-suite assertion — that no report emits an unqualified "optimal"
string — is **already non-negotiable 6** and already tested
(`test_gapped_run_is_not_reported_as_optimal`). Cite it rather than propose it.

### S0.4 — Cut-pool flushing across policy toggles ("N14"). **WITHDRAWN as a Stage-0 blocker.**

Per §3: the toggles do not exist, and building the capacity one is contraindicated by the D30
exactness condition. This cannot block publication of results produced by code that has no
policy toggles.

If the policy-sweep capability is wanted as a product feature, the correct Stage-0 artefact is
**a design note answering one question** — how a hard/soft capacity switch can change the
recourse's *cost structure* without changing which *rows exist* — reviewed against
`RESEARCH_NOTE_v2` §6 before any code. Flushing is the easy half; the exactness condition is
the hard half, and the report addresses only the easy half.

### S0.5 — NEW, and this one really is a blocker

**Regenerate every table that quotes 794.62 or 1569.44/39 s.** Per C1–C3 these are void, and
they appear in the report's §3.1 results table, §3.2 verdict, §4.6 validation table and
Recommendations. This is mechanical — the configs that produce them ship in the repository —
but nothing else in Stage 0 matters until it is done.

---

## 6. Stage 1, reformulated

The report's Stage 1 rests on a premise it inherited from the withdrawn numbers: *"a root
relaxation sitting near zero"*, from the 0.35 figures that D40/D45 retired. **The root is not
near zero.** In the corrected instance family, CPLEX's own root reaches **95–96% of the
optimum in 9–15 seconds** (D60). Everything in Stage 1 has to clear that bar, and two of the
report's three items do not.

### S1.1 — Partial Benders / scenario retention. **KEEP, with the bar restated.**

Still the right first structural intervention, and Crainic et al. (2021) is the right
reference. But the target is not "lift a root from zero" — it is **"beat, or usefully
complement, a 96% root that a commercial solver produces for free in 9–15 s."** That is a much
harder bar than the report sets, and it should be stated *before* the experiment so the result
can be negative in a readable way. D60 is the precedent: the same bar closed Stage 3.

**Run it at Q=5, not Q=3.** Per D59 that is where the monolith stops closing, and per D60 the
DW lift went to **+0.0%** exactly there. A relaxation-strengthening method that helps only
where the direct solve already wins has learned nothing. The falsifier: if partial Benders
also collapses at Q=5, the class of "strengthen the relaxation by reformulation" is finished
for this problem and the honest report says so.

### S1.2 — Papadakos core-point updating. **DEFER. Premature.**

The report recommends adopting Papadakos (2008) *"if not already present"*. Two findings
post-dating the report say wait:

- **D63** found the MW core point was **outside the master's region** and fixed it. The
  measurement basis for anything MW-related is 5 days old.
- **D65** established that MW's *selection* buys **≈0** on the only fixture the repository
  has — margins `[0, 0, 0, 0, 3.86e-05]`, because the optimal face is flat there — while also
  establishing that *"the repository has never had a measurement showing otherwise, since the
  only one it had was measuring the fallback."*

Accelerating a selection rule whose value has never been measured on a real instance is
optimising an unknown. **The prerequisite is one measurement:** MW versus plain `dual` on
`baseline_d9` or the Q=3 stress instance, LP-only so it reproduces. `use_dual_slopes` is the
natural ablation baseline and AUDIT_v4 §3.5 records that it is currently unreachable by
dispatch precedence — wiring it is a precondition, and D62 already made it reachable inside
the suite. If MW is worth ~0 at scale too, Papadakos is worth ~0 of ~0.

### S1.3 — Branch-and-cut as the default. **REVERSE.**

The report recommends adopting branch-and-cut *"since D46 shows it buys an 18% better upper
bound."* D46 also shows what it costs, and the report quotes that too without joining the two:
registering the lazy callback disables dual reductions, restricts presolve to crushing forms
and stops repeat represolve, **worth 9.6% of lower bound.**

Join them with D56: **the upper bound is already within 2% of optimal and the lower bound is
what is stuck.** Branch-and-cut therefore trades 9.6% of the scarce quantity for 18% of the
abundant one. On the evidence in the repository that is the wrong direction, and it should be
recorded as measured-and-rejected in the same register as M1.

### S1.4 — KEEP: per-objective monolith benchmarks

The report's §4.4 — build the monolith separately for `f_1`, `f_2`, `f_3` plus the composite —
is a good proposal, is not in the repository, and does not conflict with anything measured.
Keep it. It also gives §4.8's scenario-attenuation finding a clean place to live, since `f_2`
is where the sixfold attenuation acts.

Attach the C7 caveat: the monolith is a hand-synced second copy of the first stage, so three
more monoliths are three more copies to keep in sync. Non-negotiable 8 should be extended to
say which copy is authoritative.

### S1.5 — NEW: state the target regime in the paper

D59's Q=5 result is the project's best claim to relevance and the report does not have it. The
frontier the report wants to map in §4.4 is **already partly mapped**, and it has a boundary
inside the operational range the report itself names. Publish it: *"the direct solve stops
closing at Q=5"* is a stronger motivation for decomposition than any of the literature
benchmarking in §3.3.

---

## 7. Stage 2 — the CP-master bet, re-scoped

The report's attribution caution is **correct and should be kept**: a CP master with an LP
recourse inverts the pairing of Elçi & Hooker (2022), so the design is original and must not
be attributed to them. `DESIGN_DD_v1.md` exists in the repository and gives the staged plan.

Two things change with access:

1. **Stages 2 and 3 of that plan are already closed, negatively** (§4.2, §4.3). The report
   treats the CP/DD direction as unexplored. The down-set cut is valid and dominated; the
   Dantzig-Wolfe root loses to CPLEX's default cuts by 5–12×. Whatever is proposed next must
   be argued *against those two results*, and `DESIGN_DD_v1.md` §7 already lists what would
   falsify the whole design.
2. **The gate the report proposes is right and should be tightened.** It says: gate adoption on
   beating both the current engine and the monolith at Q=3. Per D59, Q=3 is where the monolith
   is comfortable (181 s, optimal). **Gate at Q=5**, against the monolith's own 226.07 bound
   after 1200 s — which is precisely the sharp, falsifiable comparison D59 §3 sets up and
   states *before* running, so it can be wrong.

---

## 8. What this review does NOT establish

- **Nothing here was re-measured.** CPLEX and Pyomo are absent from this environment; no
  config was run. Every number is quoted from the decision log entry that recorded it, and
  inherits that entry's conditions — including D26 (any MIP-phase number is one draw) and the
  reading rule that a bound must state its cut budget and whether its LP phase converged or was
  truncated.
- **The corrections in §1 are corrections of provenance, not independent verification.** C1 and
  C2 say the repository withdrew those numbers; they do not independently confirm the
  replacements. `NOT RE-MEASURED`.
- **§3's absence claims rest on `grep` over `src/`**, plus the frozen-horizon entry in
  `BENDERS_SPEC_v4` §1.1. A rolling horizon implemented under vocabulary I did not search for
  would not have been found — though D6 freezing the horizon at 10 h makes that unlikely.
- **Nothing is said about whether the formulation is right.** D62's limit is this review's
  limit: the master and the monolith are two hand-synced copies of one first stage, and a
  defect in both is invisible to every check described here, including the Phase 5 gate.
- **The report's Section 1 (operational context) and Section 3.3 (literature benchmarking)
  were not audited.** They cite external sources this review did not fetch. The one
  literature-adjacent correction made here — that Stage 1's premise of a near-zero root is
  withdrawn — is about this repository's numbers, not about Rahmaniani et al.
- **Section 4.5 (B1–B9 validation order) is untouched** beyond noting the blocks do not exist
  in `src/`. Whether the proposed order is right for an architecture that has not been built
  is not a question the repository can answer.
