# Multi-Resolution Vehicle Scheduling — research note v2

**Supersedes `cp_assisted_benders_research_idea.md` (v1).** Every claim below is either
measured and cited to a decision entry, or explicitly marked as untested. Where v1 made an
assumption that measurement contradicted, the correction is marked inline so the older
text is not reintroduced.

The headline change from v1: **the contribution is evaluation fidelity, not solver speed.**
v1 was organised around a CP layer reducing first-stage complexity and improving runtime.
Nothing measured supports that framing, and a good deal contradicts it. What the
measurements do support is a different and more defensible claim, stated in §1.

---

## 1. Hypothesis (replaces v1 §13)

> Slot-aggregated vehicle scheduling models **misprice their own schedules** and
> **choose worse schedules than they could**, because a slot is too coarse to say who is
> reachable inside a service-time promise. A minute-level operational recourse corrects
> both. It costs under 1% of iteration time, requires no new cut machinery, and leaves the
> Benders decomposition and its bounds intact.

Magnitudes, so the claim is not read larger than it is: the valuation error is ~28% on the
objective and 66–86% on reported waiting. The decision error is **3–22% single-scenario at
grids from 30 down to 10 minutes**, and **~4% when one schedule must serve four scenarios**
(D55). The four-scenario figure is the operational one.

v1's hypothesis was that CP compression plus minute-level evaluation would yield "a smaller
and more semantically meaningful decomposition". Smaller and more meaningful are not
measurable as stated, and the CP layer does not exist. The version above is measurable and
has been measured.

---

## 2. What is measured

All on `baseline_d9` (Q=2, T=22, 30-minute slots) unless stated, in the **policy penalty
regime** of §4, with every solve proven optimal.

### 2.1 The valuation error (D51, D53)

| quantity | passenger-minutes |
|---|---:|
| what the slot model **claims** its schedule costs | 11 990 |
| what that schedule **really** costs | 9 326 |
| what the **best achievable** schedule costs | 8 794 |

    valuation error  +28.5%    the model misprices its own schedule
    decision error    +6.0%    the schedule it picks is worse than achievable

These are independent errors and must be reported separately. The first misleads whoever
reads the model's output; the second is what the operator loses.

On the waiting term alone the valuation error is far larger: the slot model reports an
average wait of **38.24 min/pax** where the truth is **20.4–23.2** — wrong by 66–86%
(D51). The objective hides this because at the default penalty it is 93% unmet-demand
headcount and only 6.8% waiting.

### 2.2 The decision error across demand shapes and grids (D54, D55)

Five generated shapes on the **de-aligned** instance set — windows congruent to 7 modulo
30, 15 and 10, spike spacing coprime with all three, so no grid can fall into phase with
the demand. Q=2, `p_minutes = 56`, every solve proven optimal. Gain from minute-level
valuation, both defensible placement conventions (§3):

| shape | 30 min | 15 min | 10 min | 30 min | 15 min | 10 min |
|---|---:|---:|---:|---:|---:|---:|
| | *midpoint* | *midpoint* | *midpoint* | *end* | *end* | *end* |
| flat | 12.43% | 5.52% | 11.08% | 12.53% | 16.60% | 16.23% |
| commuter | 10.48% | 2.45% | 3.07% | 6.26% | 10.07% | 7.84% |
| bimodal | 8.19% | 4.26% | 6.07% | 14.02% | 7.43% | 10.22% |
| burst | 50.09% | 19.62% | 20.74% | 46.22% | 44.62% | 37.62% |
| spiky | 25.23% | 18.52% | 22.39% | 19.06% | 41.33% | 36.78% |

**The gain does not vanish as the grid refines** — 3–22% under `midpoint` and 8–38% under
`end` even at 10 minutes. It is also not monotone in resolution, so no trend should be read
off two columns (D55 records that mistake being made and corrected).

Refining the master *does* cut absolute cost substantially — `spiky` slot-optimised falls
9906 → 6399 → 5539 — and does **not** close the valuation gap. Grid refinement and
minute-level valuation are independent levers.

**Multi-scenario attenuates this sharply.** One schedule serving four scenarios gains
**3.84%**, against 8–50% tailored to a single one, and is actively worse on one member of
the set. §9 falsifier 3. Quote 3.84% as the operational figure.

The `start` convention is omitted here. It gives 1.3–8.2% and is the reading under which a
bus departs before the passengers it is collecting have arrived (§3); on the earlier
slot-aligned instance set it also produced a 0.00% cell that measured the generator rather
than the demand.

### 2.3 It survives the decomposition (D52)

A minute-level recourse keeps its capacity rows indexed by **departure slot**, so it
returns exactly one dual per slot — the same object the slot subproblem returns.

| | slot recourse | minute recourse |
|---|---:|---:|
| progress toward own optimum at iteration 14 | 69.0% | 66.4% |
| subproblem solve time | 0.020 s | 0.117 s |

Convergence per iteration is indistinguishable. The subproblem is ~6× slower in relative
terms and **under 1% of an iteration** in absolute terms.

### 2.4 It does not survive the comparison with a monolith (D56)

The same model — slot first stage, minute recourse — solved two ways, both
single-threaded, Q=2, T=22, `p_minutes = 56`:

| | objective | status | wall |
|---|---:|---|---:|
| minute **monolith** | 293.37 | proven optimal | **0.8 s** |
| minute **Benders** | LB 219.74 / UB 299.37 | 27% gap, 34 iterations | 301.4 s |

**390× slower on the smallest instance in the project, and it does not close.**

The decisive detail is *which* bound fails. The decomposition's upper bound is 299.37
against a true optimum of 293.37 — it finds a schedule within **2%** almost immediately
and cannot prove it. The lower bound is what is stuck, at 74.9%. This is D40's finding
from a second direction: the bound lives at the fractional LP root, and adding cuts does
not move it.

Two consequences. As a **heuristic** the decomposition is fast and good; as a **proof
system** on this instance it is not competitive with just solving the model. And any
future work aimed at the bound must attack the **relaxation** — another family of cuts
into the same master is attacking the wrong object.

---

## 3. Departure placement is a first-class modelling decision (corrects v1 §4)

v1 §4 picks the midpoint in one line, with no discussion. It is the single most
consequential choice in the design: it moves the measured gain between **0% and 49%**.

A slot says *a bus departs in [07:00, 07:30)*. It does not say when. Three readings:

| convention | departure instant | per-arc property |
|---|---|---|
| `start` | 07:00 | slot charge ≥ true wait — overstates |
| `midpoint` | 07:15 | straddles |
| `end` | 07:30 | slot charge ≤ true wait — understates |

**`end` is the reading the demand aggregation implies.** Arrivals are collected over
[07:00, 07:30); a bus leaving at 07:30 can carry all of them, one leaving at 07:00 carries
almost none. `start` assumes the bus departs before the passengers it is collecting have
arrived, and is not a defensible operating assumption — which is why the `start` column
above should be read as a lower envelope, not as a competing estimate.

**A correction worth recording.** The per-arc table suggests `end` makes the slot recourse
a lower bound on the minute recourse, which is the direction Benders needs. Measured, it
does not: the slot model overstates under all three conventions, because the minute model
re-optimises the assignment and faces a different reachable set. The per-arc inequality is
correct and does not lift to the optimum (D54).

**Consequence, and it constrains what may be claimed.** Since the slot model overstates the
cost of any given schedule, its optimum is an **upper** bound on the minute-level optimum.
A slot-level Benders lower bound therefore bounds the **slot** problem and says nothing
rigorous about the minute-level problem. Reporting a slot-Benders LB against minute-level
optimality would be a claim the construction does not support.

---

## 4. The penalty is a policy parameter and must be stated in minutes (new)

`p` is expressed in **slot units**, so a bare `p` silently encodes a different policy at
every resolution: `p: 50` is 1500 passenger-minutes at 30-minute slots and 750 at 15-minute
slots. Both were live in this repository simultaneously, and no objective from one was
comparable with the other (D50).

The operator's stated indifference: delaying a shuttle carrying 14 passengers by 4 minutes
(56 passenger-minutes) is worth one extra passenger carried. So **`p_minutes ≈ 56`**. The
configs were running 1500 — 27× that, a setting under which the model would delay 14
passengers by 107 minutes to collect one more (D53).

This is not a detail of calibration. At 1500 waiting is 4.1% of the objective and there is
almost nothing for resolution to change; at 56 it is roughly half, and resolution decides
who gets carried. **Every result in §2 is regime-dependent and states its regime.**
Configs must use `p_minutes`.

---

## 5. Cut projection is free (corrects v1 §7)

v1 §7 is the note's central technical proposal: minute-level duals `pi_t` must be
aggregated into activity-level coefficients `beta_a = sum_t pi_t * Delta_{a,t}` so the
master does not need one coefficient per minute.

**No such machinery is required.** If the minute recourse keeps its capacity rows indexed
by departure slot — capacity `S * Y_d[tau]` at each slot, demand rows on minutes — it
produces exactly one dual per slot natively. The cut is the same object the slot
subproblem already produces:

    theta >= const + sum_tau S * pi_d[tau] * Y_d[tau]

and cut construction, aggregation, the `q`-invariance check, validity classification and
the master rows are all untouched (D52). v1 anticipated a harder problem than exists.

What does need care is **units**: the minute recourse must be scaled by `1/slot_resolution`
so `dm = S*pi` lands in the units theta and the first-stage terms already use. Without it
the recourse outweighs the departure regularisation and concurrency penalty by a factor of
`slot_resolution` and quietly changes how ties between schedules are broken.

---

## 6. The exactness condition, restated as an enforced contract (extends v1 §6)

v1 §6 warns that the CP layer must not silently shrink the first-stage space. The sharper
and more immediate condition, learned the expensive way, concerns the **subproblem**:

> `y` must enter the recourse through the right-hand side only. The minute grid and arc
> set must be functions of `(T, slot_resolution, Wmax_minutes, departure_policy)` and the
> demand — never of the schedule.

Build the availability profile so that the schedule decides *which rows exist* and the dual
is no longer a subgradient of the recourse; no cut built on it is valid. That defect cost
this project six months of invalid bounds (D30). v1 §3 describes the minute-level
availability profile in exactly the shape that invites it.

Four conditions are now enforced by tests rather than assumed: the recourse depends on `y`
only through the per-slot aggregate (**including its duals**); `y` enters the right-hand
side only; every master row is separable by vehicle; the fleet is homogeneous. See
`DESIGN_DD_v1.md` and D48.

---

## 7. Research questions, with current status

| RQ (v1 numbering) | status |
|---|---|
| RQ3 — does minute-level evaluation improve recourse accuracy? | **Answered, strongly.** §2.1 |
| RQ4 — can minute duals be projected without weakening convergence? | **Answered, yes** — and no projection is needed. §5, §2.3 |
| RQ5 — do the projected cuts converge *faster*? | **No, and against the right baseline it is not close.** The minute monolith proves optimality in 0.8 s; the decomposition is at a 27% gap after 301 s — 390× (D56). §2.4 |
| RQ7 — sensitivity to travel-time treatment | **Partly.** Placement is this question in miniature and moves the answer 0%→49%. §3 |
| RQ1, RQ2, RQ6, RQ8 | **Untouched.** All four concern the CP layer, which does not exist |

---

## 8. Success criteria (replaces v1 §14)

v1's eight criteria are mostly about CP reducing complexity and improving runtime. Those
are the framing that has repeatedly failed here: measured against a monolithic MILP, the
Benders decomposition reaches **69.2%** of the true Q=3 optimum after 1520 s, while the
monolith produces that optimum (**1658.86**) in 947 s (D54). Speed is not the contribution.

Revised criteria:

1. **Measured.** Slot aggregation misprices its own schedules — 28.5% on the objective,
   66–86% on waiting.
2. **Measured.** It chooses schedules 6–49% worse than achievable, carrying up to 52 fewer
   passengers, under both defensible placement conventions.
3. **Measured.** A minute-level recourse corrects both at under 1% of iteration cost with
   unchanged cut machinery and unchanged convergence per iteration.
4. **Enforced.** Benders validity is preserved — the exactness conditions of §6 are tests.
5. **Open.** Generalisation: more instances, larger fleets, finer grids, and demand shapes
   regenerated without slot alignment.
6. **Open.** Whether the effect survives multi-scenario recourse, which none of the above
   tests.
7. **Not claimed.** Runtime or convergence improvement. There is none.
8. **Not attempted.** Anything requiring the CP layer.

---

## 9. Falsifiers, and what happened when they were tested (D55)

1. **Uniform demand.** Not falsified. `flat` gives 5.5–12.4% — the lowest band, nonzero.
2. **An artifact of 30-minute slots.** **Refuted.** At a 10-minute grid the correction is
   still worth 3–22% (`midpoint`) or 8–38% (`end`). Refining the master cuts absolute cost
   substantially (`spiky`: 9906 → 6399 → 5539) and does **not** close the valuation gap —
   the two are independent levers. The trend is also **not monotone**: a reading of "it
   halves", taken from the 30 and 15 columns alone, did not survive the 10 column.
3. **Multi-scenario averaging.** **Largely holds.** One schedule serving four scenarios
   gains **3.84%**, against 8–50% tailored to one — a sixfold attenuation, and on one
   scenario the minute-optimised schedule is actively worse. **The honest figure for the
   stochastic setting this project targets is ~4%.** Every Fase 1 config is four-scenario;
   single-scenario gains must not be quoted as the operational result.
4. **Conventions disagree on a good instance set.** Not falsified. `midpoint` and `end`
   agree in sign and rough magnitude across five de-aligned shapes and three resolutions.
   `start` remains the outlier and remains physically implausible.

**Answered, and it went against the decomposition (D56).** Measured against the right
baseline — the minute-level **monolith**, the same model solved without a decomposition in
between — the monolith proves optimality in **0.8 s** while the decomposition sits at a
**27% gap after 301 s**. See §2.4. Nothing in this note claims speed, and now nothing
could.

---

## 10. Immediate next steps

1. Regenerate the demand shapes with deliberately slot-misaligned structure, and re-run
   §2.2. The current `burst` row measures the generator.
2. Sweep master resolution (30 / 15 / 10 minutes) at fixed `p_minutes`, to separate
   "aggregation is lossy" from "30 minutes is too coarse".
3. Multi-scenario check.
4. Literature check on the specific framing — aggregate first stage, fine-grained
   dual-compatible recourse, decision-level cost quantified — before investing further.
   Discretisation error in scheduling is well-trodden; this combination may not be.
