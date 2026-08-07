# Fase 1 — is the master tractable?

The question: can the master be solved fast enough for Benders to have a chance? The
master is ~99% of run time and Benders solves it dozens of times against the monolith's
once, so if each solve costs tens of seconds, `N * t_master` never beats `t_monolith`.

Test point, fixed for every cell: **Q=3, T=44 (660 min / 15-min slots), 4 scenarios,
300 s total budget**, `p = 50`, `W_max = 60 min`, `concurrency_penalty = 0.25`,
`per_iteration_time_limit_s = 30`, `per_iteration_mipgap = 0.05`. Configs in
`configs/phase1/`.

**Death criterion, set before the runs:** the LB must reach **>= 785**, i.e. 50% of the
monolithic MILP's LB of 1569.44.

---

## 1. The baseline was re-measured first

D30 changed the subproblem, so the pre-D30 `LB = 0.32` described a model that no longer
exists and a cut set that was invalid. Re-recorded with the current code:

| run | LB | UB | iterations | note |
|---|---|---|---|---|
| `base_off_a` (first) | 0.311663 | 2686.86 | 14 | |
| `base_off_b` | 0.310428 | 2812.86 | 15 | overlapped with a test run |
| `base_off_a` (A/B batch) | 0.299040 | 2685.86 | 14 | the cell used below |

Three draws of the LB: 0.3117, 0.3104, 0.2990 — a spread of about 4%, and every run
prints `NOT REPRODUCIBLE` because master solves stop on the clock rather than the gap.
The expectation that valid cuts would make the corrected baseline **worse** than the old
0.32 did not materialise: it was already this low, and the correction moved it by less
than the run-to-run noise. The old number was not misleading about the magnitude.

## 2. The A/B

One draw per cell, run sequentially on an otherwise idle machine, same code for all four.

| cell | anchor (D33) | LP phase (D34) | LB | UB | iters | cuts |
|---|---|---|---|---|---|---|
| `base_off` | off | off | 0.299040 | 2685.86 | 14 | 14 |
| `anchor_on` | **on** | off | 0.313787 | 2267.36 | 11 | 11 |
| `lp_on` | off | **on** | 0.350393 | 2310.36 | 19 | 19 |
| `both_on` | **on** | **on** | **0.350915** | 2016.11 | 14 | 14 |

**Best LB reached: 0.35, against a target of 785.** The criterion fails by a factor of
about 2200. The UB column moves in the right direction and the LB column does not move at
all on the scale that matters.

Do not read the LB spread across these cells as an effect. It is 0.30–0.35 against a
baseline draw-to-draw spread of 0.299–0.312 on the *same* configuration: the differences
between cells are the same size as the noise.

## 3. Why the bound does not move, and what actually broke

Per-iteration master solve time, in seconds:

```
base_off    0.2 0.5 1.0 0.9 4.6 31.3 31.6 31.5 30.2 30.2 30.2 30.4 30.7 21.0
anchor_on   1.5 2.9 5.2 30.4 33.3 32.7 31.1 34.7 31.7 30.5 30.7
lp_on       0.9 0.3 0.4 0.4 0.4 0.6 0.6 0.5 0.3 0.3 | 30.2 30.3 30.3 30.4 30.4 30.4 30.4 31.3 30.4
both_on     0.3 1.1 1.0 1.3 0.9 | 30.6 30.6 30.3 30.7 30.4 31.7 30.4 31.4 26.9
```

Every MIP iteration pins the 30 s ceiling. Not "takes about 30 s" — pins it, and
terminates `maxTimeLimit / aborted`. The gap it terminates at, on the last iterations of
`lp_on`:

```
nodes=3748 best_bound=0.347384 incumbent=502.361 gap=0.9993
nodes=3959 best_bound=0.350393 incumbent=636.361 gap=0.9994
```

**The master ends each solve with a 99.9% internal MIP gap.** Its incumbent is ~600 and
its best bound is 0.35. The reported Benders LB is that best bound. So the LB is not
limited by the cut set being thin — it is limited by the master's own branch and bound
being unable to lift its bound off zero in 30 s, after ~4000 nodes.

That reframes the two items:

- The **anchor** is valid and correctly stated across scenarios, and at this test point it
  is *slack*. Solving the empty master with no cuts: Q=1 gives 7737.62, Q=2 gives 1662.70,
  **Q=3 gives 0.24**. D29's headline 1569.09 was a Q=2 measurement. With three vehicles
  over 44 slots the fleet installs far more capacity than `R_cum[j]` ever demands, so
  every prefix row is slack and the inequality bounds nothing. It costs nothing and buys
  nothing here.
- The **LP phase** does exactly what it was designed to do. LP iterations cost 0.3–0.9 s
  against 30 s for MIP iterations, a 50–100x reduction, and `lp_on` fits 19 iterations
  into the budget where `base_off` fits 14. It converts master time into cuts at a much
  better rate. It does not help, because cuts are not the binding constraint.

## 4. What this says about the Fase 1 question

Within a 300 s budget the master is not solved — it is truncated, every time, at a gap of
99.9%. `N * t_master` does not lose to `t_monolith` by a tunable margin; the master is not
producing a usable bound at all at this size.

The obvious objection is that 300 s total with a 30 s per-iteration ceiling starves the
master, so `configs/phase1/master_headroom.yaml` re-ran `lp_on` inside a 900 s budget with
the ceiling raised to 300 s. The gap-tied schedule caps the solve at `2 + 5/0.05 = 102 s` regardless (D29
recorded the same thing), so what it actually tests is **102 s per solve, 3.4x the A/B**:

| per-solve budget | nodes explored | master best bound | internal gap |
|---|---|---|---|
| 30 s | ~3 960 | 0.350393 | 0.9994 |
| **102 s** | **~30 700** | **0.354730** | **0.9990** |

**7.7x the nodes and 3.4x the time buy 1.2% of bound**, and the run ends at LB 0.35473 —
indistinguishable from the 300 s cells. The master is not time-starved. Its bound sits at
zero because nothing in the model forces `theta` up: the anchor is slack at Q=3 (§3), and
14–19 cuts cannot cover a space of 264 binaries. Branch and bound then has nothing to
work with, and more of it changes nothing.

**Verdict: the criterion fails, and it fails structurally rather than by a tunable
margin.** `N * t_master` does not merely lose to `t_monolith` — no `N` and no `t_master`
in reach produces a usable bound at this size. The two items were built correctly and
both were aimed at a constraint that is not binding.

What this does not say: nothing here is evidence against Benders at Q<=2, where the
anchor is worth 1662–7737 on the empty master and D29 measured real gains. The failure is
specific to a fleet large enough to make the capacity anchor slack.

## 5. Reading rules for anything quoted from here

- Every cell is **one draw** and prints `NOT REPRODUCIBLE`. Differences smaller than ~15%
  in the LB are noise (D26).
- Each UB is an exhibited feasible schedule and is real. No LB here is within a factor of
  1000 of useful, so **no gap from this table may be quoted**.
- The LP phase claims no upper bound while it is active: a fractional schedule is not
  exhibitable. The UB column for `lp_on`/`both_on` comes only from MIP-phase iterations.
- State `(p, W_max) = (50, 60)` and `concurrency_penalty = 0.25` on any table derived from
  these runs (D18). The manifest records all three.
