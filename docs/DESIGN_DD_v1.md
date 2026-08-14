# DESIGN_DD_v1.md — A decision-diagram master for the MOB-AUTO2 Benders

**Status: design only. Nothing here is measured.** Companions: `BENDERS_SPEC_v4.md`
(the model as it is), `docs_decisions.md` (D1–D48), `AUDIT_v4.md`.

This document exists to pin the exactness conditions *before* code, because every
correctness failure this project has recorded — D30, D24, D27, D39 — was a condition that
held when written and stopped holding when something else moved, with nothing asserting
it. The design below rests on four such conditions. Each is stated as a proposition, with
the code that currently satisfies it and the test that must fail if it stops.

The design is also written against a hostile baseline, deliberately. D46 and D47 measured
the current method at the Q=3 test point: best configuration reaches LB 1148.65 in 1520 s
of master time, against a monolithic MILP that produces the exact optimum 1569.44 in 39 s.
Nothing proposed here is worth building unless it is measured against those two numbers.

---

## 1. The structural fact

### 1.1 Statement

> **P1.** The recourse is not a function of the master's per-vehicle schedule `y`. It is a
> function of the per-slot, per-direction *aggregate*
>
> ```
> Y_d[τ] = Σ_q y_d[q,τ] ∈ {0, 1, …, Q},    d ∈ {OUT, RET},  τ ∈ {0,…,T−1}
> ```
>
> We call `(Y_OUT, Y_RET) ∈ Z_{≥0}^{2T}` the **signature** of `y`.

### 1.2 Why it holds, in the code

`ProblemSubproblem.evaluate` reads the candidate exactly twice: once to collect the index
sets, and once to build the capacity vectors
([subproblem_impl.py:151-175](../src/mobauto2_benders/problem/subproblem_impl.py#L151)):

```python
C_out[tau] += float(val) * S     # for every yOUT[q,tau] in the candidate
C_ret[tau] += float(val) * S     # for every yRET[q,tau] in the candidate
```

`C_out`/`C_ret` are then the *only* channel by which `y` reaches the LP. In
`solve_subproblem` they appear in one place — the right-hand side of the capacity rows
([subproblem_impl.py:2248-2262](../src/mobauto2_benders/problem/subproblem_impl.py#L2248)):

```python
sum(m.x_OUT[t, tau] for t in ts) <= float(C_out[tau])
```

The arc set `Arcs_list` depends on `T` and `Wmax_slots` only; the variable set
(`x_OUT`, `x_RET`, `u_OUT`, `u_RET`), the demand rows and the objective coefficients
`(τ−t)` and `p` do not mention `y` at all. `K_out`/`K_ret` are still computed and still
travel through `SPParams`, but after the D30 de-layering they shape nothing — the comment
at [subproblem_impl.py:2191](../src/mobauto2_benders/problem/subproblem_impl.py#L2191)
records why they were removed and what they cost.

Therefore two master solutions with the same signature produce a byte-identical LP, hence
the same recourse value and the same dual — **not merely the same value**.

### 1.3 Two consequences, both used below

> **P2 (convexity).** `Q(·)` is the value function of an LP whose parameter enters only in
> the right-hand side. It is therefore convex and piecewise linear in `(C_OUT, C_RET)`, and
> hence in the signature.

This is what makes the existing Benders cut a valid global underestimator, and it is the
property D30 destroyed and restored.

> **P3 (monotonicity).** `Q(·)` is non-increasing in the signature, componentwise.

Raising `C_d[τ]` relaxes a `≤` row, so the feasible set grows and the minimum cannot rise.
The subproblem is never infeasible — `u_d[t]` absorbs any demand at price `p` — so the
value function is finite everywhere on `Z_{≥0}^{2T}` and the statement has no exceptions.
P3 is why the cut slopes are `≤ 0` and why `dm = S·π` with `π ≤ 0` is the sign the master
expects (spec §2.5).

### 1.4 The size of the fibre

At the Q=3 / T=44 test point, the number of distinct `y` mapping to one signature is

```
Π_τ  C(3, Y_OUT[τ]) · C(3, Y_RET[τ])
```

For a schedule with, say, 18 departures spread one-per-slot this is `3^18 ≈ 3.9 × 10^8`.
Those are `3.9 × 10^8` master solutions that the subproblem cannot tell apart.

**This is the user's observation, and it is correct.** The design question is which part of
it is already banked and which is not.

---

## 2. What is already banked

`aggregate_cuts_by_tau` (default `true`) collapses the cut from `(q,τ)` onto `τ`
([master_impl.py:1963-2003](../src/mobauto2_benders/problem/master_impl.py#L1963)), writing
it against the master's own aggregation variables `m.Yout[t]` / `m.Yret[t]`
([master_impl.py:291](../src/mobauto2_benders/problem/master_impl.py#L291),
[master_impl.py:371-375](../src/mobauto2_benders/problem/master_impl.py#L371)). The
collapse is an identity only if the coefficients agree across `q`, and `_assert_q_invariant`
raises if they do not.

**So a single Benders cut already binds the whole fibre.** One subproblem solve already
constrains all `3.9 × 10^8` of those master solutions simultaneously. No redesign is needed
to obtain that, and any proposal that claims it as a new benefit is claiming something the
repository has had since D10.

What is *not* banked is everything in §3.

---

## 3. What is not banked

### 3.1 The master still branches per vehicle

The recourse cannot distinguish members of a fibre; the master's branch-and-bound still
enumerates them. The decision variables are `2·Q·T` binaries
([master_impl.py:288-289](../src/mobauto2_benders/problem/master_impl.py#L288)) and CPLEX
branches on those, not on `Yout`/`Yret`.

Symmetry breaking removes only a sliver of this. It orders vehicles by **total** departures
— `Q−1` rows
([master_impl.py:556-563](../src/mobauto2_benders/problem/master_impl.py#L556)) — which
quotients the `Q!` relabellings but leaves every distinct *assignment* of a fixed departure
profile to vehicles in play. The spec (§2.8) explains why the stronger prefix form is
invalid, and that reasoning stands: the weak form is the only *valid* one in this
formulation. The right conclusion is not a stronger symmetry constraint; it is a
formulation where the vehicle index does not exist.

D45 measured the consequence: ~19 000 nodes at an internal gap of 0.20 after 102 s. D46
measured that the binding constraint has moved from the cut set to the master solve.

### 3.2 The recourse cut is convex where it only needs to be valid on a lattice

By P2 the linear cut is the tightest *convex* underestimator, and Benders will recover the
full convex envelope in the limit. But `Y` is integer, and P3 gives information that no
hyperplane carries:

> for every integer `Y ≤ Ŷ` componentwise: `Q(Y) ≥ Q(Ŷ)`

one LP solve, a statement about an entire **down-set of the lattice**. This is the
logic-based Benders shape in Hooker §6.2 — a cut derived from a proof of a property, valid
over a class, rather than a subgradient valid in a neighbourhood. It is not implied by the
convex envelope at fractional `Y`, and D40 established that the bound in this problem lives
precisely at the fractional LP root.

The obstacle is expressing it linearly. A down-set indicator needs a big-M or an auxiliary
binary per `(d,τ)` — 88 of them at T=44. **Whether it pays is a measurement.** This
repository has been wrong four times reasoning about exactly this class of trade (D40's
own note says so), so §5 stages it behind something cheaper.

---

## 4. The exactness conditions

These are the contract. Each is a proposition, the code that satisfies it today, and the
test that must fail if it stops being satisfied.

### E1 — The recourse depends on `y` only through the signature

*Needed by:* every stage. It is the whole design.

*Holds because:* §1.2.

*Breaks if:* anyone reintroduces a per-vehicle term in the subproblem — a per-vehicle
capacity, a vehicle-dependent cost, an occupancy-dependent arc set. The D30 layers were
exactly this and cost six months of invalid bounds.

*Test:* `test_recourse_is_constant_on_the_fibre` — build two candidates with an identical
signature and a different per-vehicle assignment, assert the recourse values are equal to
the cent **and** the dual vectors `pi_OUT`/`pi_RET` agree. Value equality alone is too weak:
two different duals give two different cuts, and the aggregation in §2 would then be
selecting one arbitrarily.

### E2 — `y` enters the subproblem in the right-hand side only

*Needed by:* the validity of every cut, present and future. This is D30 restated as a
standing condition rather than a historical fix.

*Holds because:* the arc set, variable set and objective of `solve_subproblem` are
functions of `(T, Wmax_slots, p)` alone.

*Breaks if:* the minute-level extension is built the way the research note describes it.
See §6 — this is the single most dangerous item in the plan.

*Test:* `test_subproblem_structure_is_independent_of_y` — build the LP at two candidates
with different signatures, assert equal numbers of variables and constraints and an
identical constraint-index set. Cheap, and it fails loudly at exactly the moment someone
makes the availability profile decide which rows exist.

### E3 — Every master constraint is separable by vehicle except through the signature

*Needed by:* the reformulation in §5 stage 3 (Dantzig-Wolfe). Without it the per-vehicle
column is not well defined.

*Holds because:* traced constraint by constraint in `build_master`. Everything below is
indexed `(q,t)` with no term mentioning another vehicle:

| block | rows | reference |
|---|---|---|
| C1a exclusivity | `yOUT[q,t] + yRET[q,t] + c[q,t] ≤ 1` | [:365](../src/mobauto2_benders/problem/master_impl.py#L365) |
| C1b `inTrip` definition | `inTrip[q,t] = Σ_u (yOUT[q,u]+yRET[q,u])` | [:385](../src/mobauto2_benders/problem/master_impl.py#L385) |
| C1c travel exclusivity | `… ≤ 1 − inTrip[q,t]` | [:397](../src/mobauto2_benders/problem/master_impl.py#L397) |
| C2a occupancy recursions | `atL[q,t] = atL[q,t−1] − yOUT[q,t−1] + yRET[q,t−trip]` | [:427](../src/mobauto2_benders/problem/master_impl.py#L427) |
| C2b/c/d gating | `yOUT ≤ atL`, `yRET ≤ atM`, `c ≤ atL` | [:446](../src/mobauto2_benders/problem/master_impl.py#L446) |
| charge-before-idle | `c[q,t] ≤ yOUT[q,t−1] + c[q,t−1] + 1 − atL[q,t−1]` | [:476](../src/mobauto2_benders/problem/master_impl.py#L476) |
| C4 battery balance / link / cap | `b[q,t+1] = b[q,t] − L·(…) + gchg[q,t]` etc. | [:568](../src/mobauto2_benders/problem/master_impl.py#L568) |
| C5 departure charge | `b[q,t] ≥ 2L·yOUT[q,t]` | [:599](../src/mobauto2_benders/problem/master_impl.py#L599) |
| initial/final fixings | `atL[q,0]`, `atL[q,T−1]`, `yOUT[q,0]`, horizon-end zeros | [:413](../src/mobauto2_benders/problem/master_impl.py#L413), [:487](../src/mobauto2_benders/problem/master_impl.py#L487) |

And the coupling is **exactly three terms, all functions of the signature alone**:

| coupling | form | reference |
|---|---|---|
| the Benders cuts | `θ ≥ const + Σ_τ dm_d[τ]·Y_d[τ]` | §2 |
| departure regularisation | `ε · Σ_{q,t}(yOUT+yRET) = ε · Σ_τ (Y_OUT[τ]+Y_RET[τ])` | [:355](../src/mobauto2_benders/problem/master_impl.py#L355) |
| concurrency penalty | `e_d[τ] ≥ Y_d[τ] − 1` | [:336](../src/mobauto2_benders/problem/master_impl.py#L336) |

plus symmetry breaking ([:556](../src/mobauto2_benders/problem/master_impl.py#L556)), which
is a cross-vehicle row — and which the reformulation **deletes** rather than translates,
because it exists only to suppress the symmetry the reformulation removes outright.

*Breaks if:* a genuinely coupling resource is added. The realistic one is a shared charger:
`Σ_q c[q,t] ≤ n_chargers`. Note that this is *also* a function of an aggregate, so it does
not break the design — it adds one coupling row per slot and the column generation absorbs
it. What would break it is a constraint coupling two *named* vehicles.

*Test:* `test_master_rows_are_vehicle_separable` — walk the constructed Pyomo model, and
for every constraint collect the set of `q` indices appearing in its body. Assert every row
touches at most one `q`, with an explicit allow-list for the coupling rows named above.
This is the test that turns E3 from an observation into a contract.

### E4 — The fleet is homogeneous

*Needed by:* the single shared column pool in stage 3. With heterogeneous vehicles the
design still works but needs one pool per type, and the aggregate `Y` splits accordingly.

*Holds because:* `symmetry_breaking` already refuses a heterogeneous fleet outright, naming
the offending values
([master_impl.py:520-537](../src/mobauto2_benders/problem/master_impl.py#L520)).

*Test:* reuse the existing precondition. Stage 3 must raise the same way rather than
silently pooling distinguishable vehicles.

---

## 5. The staged plan

Ordered by (value of the answer) ÷ (cost of being wrong). Every stage states what it claims
and what would refute it.

### Stage 0 — Measure the fibre (measurement 1)

*Cost:* one logging change, no model change.

*Question:* is the Benders loop paying for the fibre redundancy at all? Log the signature at
every iteration of the D40 150-iteration LP run and count distinct signatures against
distinct `y`.

*Reading:* if the loop revisits signatures, that is wasted subproblem work recoverable by a
memo, immediately and for free. If it does not, the redundancy is costing search rather than
solves, which points at stages 1 and 3 rather than at caching.

*This is not a bound experiment.* It is diagnostic and cheap, and it constrains what the
later stages are allowed to claim.

### Stage 1 — Window trip-cap inequalities from a single-vehicle diagram

*Cost:* a self-contained DP module, no change to the Benders loop.

D46 closed with the statement that the remaining lever on the bound is *"a valid inequality
in `y` alone… the per-vehicle trip cap derived from the battery block, which, unlike the
recourse anchor D33 found inert at Q=3, does not go slack as Q grows."* Stage 1 is that
inequality, derived by a decision diagram rather than by hand.

Build the diagram for **one** vehicle: layers `τ = 0..T−1`, node state
`(location ∈ {L,M}, in-transit remaining, battery)`, arcs `{IDL, CHR, OUT, RET}` gated by
C1a/C1c/C2b/C2d/C5 and the battery recursion. Then, by DP on that diagram, compute for every
window `[t1,t2]`

```
maxTrips(t1,t2) = max departures one vehicle can START in [t1,t2],
                  maximised over every entry state
```

and emit

```
Σ_{τ ∈ [t1,t2]} ( Y_OUT[τ] + Y_RET[τ] )  ≤  Q · maxTrips(t1,t2)
```

**Validity argument, stated so it can be attacked.** `maxTrips` maximises over all entry
states, so it bounds the departures of *any* vehicle in that window regardless of its
history. Summing over `Q` independent vehicles gives the right-hand side. The inequality is
in `Y` alone, so it is orthogonal to the recourse cuts and cannot interact with their
validity. It scales with `Q`, so it does not go slack as the fleet grows — which is the
property D33's recourse anchor lacked.

**Two independent bounds worth computing separately, because a discrepancy is a bug:**

- *travel-time bound*: with `trip_slots = 2` at the test point, one vehicle's departures are
  ≥ `trip_slots` apart, so `maxTrips ≤ ceil(W / trip_slots)` for a window of `W` slots.
- *energy bound*: each departure costs `L = 30`; charging adds at most
  `delta_chg = 17.5` per slot and only at Longvilliers while not in transit. With
  `Emax = 150` this caps trips independently of the travel-time argument.

The DP result must be `≤` both. If it exceeds either, the diagram is wrong, and that check
costs nothing.

*Refuted if:* the inequalities do not move the LP root above D40's 794.62 on the seeded
cut set. That is a direct, reproducible measurement — the LP phase never stops on the clock
(D26 does not apply), so it is one of the few numbers in this repository that reproduces to
the digit.

### Stage 2 — The down-set recourse cut

*Cost:* moderate; needs a linearisation study before any integration.

Per §3.2. Build it offline first, exactly as D43 built the underestimation test: construct
the cut at `y0`, price the true recourse over sampled points of the down-set, and confirm
`cut(Y) ≤ Q(Y)` with the minimum slack reported. **Do not integrate before measuring how
much LP-root bound the big-M encoding actually buys**, because a big-M at 88 indicators is
a plausible way to make the master worse while looking stronger on paper — which is exactly
what M1 did in spec §2.9.

*Note the interaction with stage 1:* both act on the LP root. Measure them separately
before measuring them together, or neither number means anything.

### Stage 3 — Dantzig-Wolfe over per-vehicle trajectories

*Cost:* high. This is the reformulation, not a cut.

By E3 and E4 the master is `Q` identical, independent per-vehicle polytopes coupled only
through the signature. Reformulate: let `J` be the set of feasible single-vehicle
trajectories, `λ_j ∈ Z_{≥0}` the number of vehicles flying trajectory `j`. Then

```
min   θ  +  ε·Σ_τ (Y_OUT[τ] + Y_RET[τ])  +  κ·Σ_τ (e_OUT[τ] + e_RET[τ])
s.t.  Y_d[τ] = Σ_j λ_j · y_d^j[τ]                      (aggregation)
      Σ_j λ_j = Q                                       (fleet size)
      e_d[τ] ≥ Y_d[τ] − 1
      θ ≥ const_i + Σ_τ dm_{d,i}[τ]·Y_d[τ]              (existing cuts, unchanged)
      λ ∈ Z_{≥0}^{|J|}
```

**The vehicle index is gone.** Symmetry is not broken, it does not exist; the
symmetry-breaking rows are deleted rather than weakened.

**Why this is the bound attack and not just a tidiness argument.** The LP relaxation of the
reformulation optimises over `conv(integer points of the per-vehicle polytope)`, whereas the
current master's LP relaxation optimises over the per-vehicle polytope's own LP relaxation.
The former is contained in the latter, generally strictly. D40 established that the bound in
this problem lives at the LP root. This is the standard reason branch-and-price beats a
compact formulation on exactly this problem shape, and it is a claim that a single
measurement settles.

**`J` cannot be enumerated.** At the test point a trajectory is an alternating OUT/RET
sequence over 44 slots with OUT forbidden from `t ≥ 40` and all departures forbidden from
`t ≥ 42`; the count is of order `C(40,10) ≈ 8 × 10^8`. So this is column generation, and the
pricing problem is a resource-constrained shortest path on the stage-1 diagram with arc
weights `−σ_d[τ]` from the aggregation-row duals, and battery as the resource.

**One dominance rule needs a proof, not an assertion.** The natural DP carries the *maximum
reachable battery* at each node, on the argument that more battery is weakly better for
completing any suffix. That argument is clean for C5 and the balance rows, but
`charge_before_idle` makes charging in slot `t` depend on `c[t−1]`, so the greedy-max-charge
policy is constrained by its own past. It is still monotone — charging more earlier only
relaxes the later bound — but *"still monotone"* is precisely the kind of sentence this
project has had refuted four times. **Verify by enumeration against the MILP on a small
instance (Q=1, T≤12) before relying on it.**

*Refuted if:* the reformulated LP root does not exceed 794.62 by a margin larger than the
measurement noise, or if pricing cost per iteration exceeds the master-solve time it saves.

---

## 6. The trap in the minute-level extension

`cp_assisted_benders_research_idea.md` §3 proposes a minute-level availability profile
`A_{q,t}` driven by the master's slot decision, with the subproblem evaluating operational
consequences at minute resolution. **Built naively, that is D30 again.**

If the master's `y` decides *which minutes* a vehicle is available, and the availability
decides which rows or which variables the subproblem contains, then `y` has entered `A`, not
`b`. The dual of one instance is then not a subgradient of the recourse across `y`, and — in
the spec's words — no cut generator can be valid on top of it. The measured cost last time:
cuts forced `θ` to 6893, 5290 and 6087 at a schedule whose true recourse is 4183.00.

The extension is valid **if and only if** the minute grid is fixed and `y` scales the
right-hand side:

```
C_d[m] = S · Σ_q Σ_τ a_d[τ, m] · y_d[q,τ]      for a CONSTANT 0/1 matrix a
```

With a deterministic trip duration, `a_d[τ,m] = 1` iff minute `m` falls in the service
window induced by a departure in slot `τ`, which is a constant. With travel-time scenarios
it is a constant *per scenario*, so the multi-scenario path already handles it. It breaks the
moment trip duration becomes a decision variable.

Note that P1 survives the extension unchanged: `C_d[m]` is still a function of the signature
alone, so the whole of §4 and §5 carries over to a minute-level subproblem without
modification. **That is the actual argument for multi-resolution here** — not that minutes
are more realistic, but that the master-side structure is indifferent to the subproblem's
resolution, so the resolution can be raised without touching the decomposition.

The research note's §5 classification is correct as written: with the boundary at
`MILP master ↔ LP subproblem` this remains classical Benders, and the diagram/CP layer is
auxiliary to the master. Stage 3 does not change that — Dantzig-Wolfe reformulates the
master, it does not move the Benders boundary.

---

## 7. What would falsify the whole design

Stated in advance, per D47's practice of fixing the reading before the run:

1. **E1 fails** — some per-vehicle dependence is found in the subproblem. Then the fibre is
   not a fibre and §2 onward collapses. Stage 0's test settles this on day one.
2. **Stage 1 does not lift the LP root** above 794.62. Then the "valid inequality in `y`
   alone" lever D46 identified is not a lever, and stages 2 and 3 lose their motivating
   evidence.
3. **The reformulated root is not tighter.** The DW argument in stage 3 is standard but not
   automatic; if the per-vehicle polytope is already integral, `conv` adds nothing and the
   whole reformulation buys only the symmetry removal.
4. **Pricing dominates.** Even a much tighter root loses if each column-generation iteration
   costs more than the master solve it replaces.

And the standing one, from D46/D47: **none of this is competitive unless it changes the
comparison against a monolith that solves the test point exactly in 39 s.** A better Benders
bound that is still an order of magnitude behind the monolith is a result about Benders, and
must be reported as that rather than as a method.
