# Structural sweep — fixed 120 s budget

`p = 50` and `W_max = 60` are **given inputs**, not swept. The axes are structural:
time discretisation and fleet size. Instance `setups/base.yaml`, 300 requests,
660-minute horizon, S=15, E_max=150, ε=0.01, concurrency_penalty=0.25.

Every cell ran under `solver.total_time_limit_s: 120` with
`master.per_iteration_time_limit_s: 30` (D22). This is the first sweep in which those
two knobs did anything: previously the Benders loop overwrote them from hardcoded
constants, and the budget was checked only between iterations, so cells silently ran
131 s, 145 s and >8 min against a declared 120 s.

## Results

| cell | slot_res | T | Q | status | iters | LB | UB | gap % | served | wait min | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `res30_q1` | 30 | 22 | 1 | **OPTIMAL** | 63 | 9089.5689 | 9098.1200 | 0.094 | 121/300 | 36.69 | 78.4 |
| `res30_q2` | 30 | 22 | 2 | UNKNOWN | 7 | 2904.2685 | 4651.9900 | 37.569 | 213/300 | 42.39 | 140.2 |
| `res30_q3` | 30 | 22 | 3 | UNKNOWN | 30 | 804.9097 | 898.8601 | 10.452 | 290/300 | 40.97 | 122.9 |
| `res30_q4` | 30 | 22 | 4 | UNKNOWN | 24 | 3.6629 | 697.3600 | 99.475 | 294/300 | 40.41 | 121.7 |
| `res30_q5` | 30 | 22 | 5 | UNKNOWN | 16 | 32.4711 | 519.7501 | 93.753 | 297/300 | 36.77 | 121.7 |
| `res15_q2` | 15 | 44 | 2 | UNKNOWN | 8 | 52.9611 | 5046.2400 | 98.950 | 207/300 | 28.70 | 144.5 |

## How to read this

> **Reproducibility caveat (D26).** These cells were budgeted in seconds, and a
> time-truncated run of this model is **not** bit-reproducible: the same config run twice
> gave 8 iterations at LB 2251 and 10 at LB 3487. Treat every LB and iteration count here
> as one draw from a noisy distribution, not a measurement. The **UB column is unaffected**
> — each UB is the cost of an exhibited feasible schedule, and jitter cannot make an
> exhibited schedule cheaper. The `res30_q1` row converged on the gap rather than the
> clock and is exempt.
>
> The order Q=3 < Q=4 < Q=5 by UB is therefore **not established**. The order-of-magnitude
> findings survive, being far larger than the spread above.


**Only `res30_q1` converged.** Every other cell reports a bracket, not an optimum. The
UB column is the one that carries proof: each UB is the cost of an exhibited feasible
schedule, so "Q=3 achieves at most 898.86" is a statement that holds regardless of the
budget. The LB column is nearly worthless at 120 s for Q≥2 and **must not** be quoted as
an optimality gap.

Consistency check, since the objective is dominated by the unserved-passenger penalty:

| cell | unserved | 50 × unserved | UB | remainder (wait + starts + concurrency) |
|---|---|---|---|---|
| `res30_q1` | 179 | 8950 | 9098.12 | 148.12 |
| `res30_q2` | 87 | 4350 | 4651.99 | 301.99 |
| `res30_q3` | 10 | 500 | 898.86 | 398.86 |
| `res30_q4` | 6 | 300 | 697.36 | 397.36 |
| `res30_q5` | 3 | 150 | 519.75 | 369.75 |

`demand = 300` in every cell, so no request fell outside the horizon (backlog item 4 is
not biting this instance).

## What the fleet axis says

Cost is dominated by unmet demand at `p = 50`, so fleet size is the dominant lever:

- **Q=1 is not a viable service.** 121/300 served; the run converges quickly precisely
  because there is little to decide.
- **Q=2 is capacity-starved.** Its true optimum is 4190.74 (converged separately, 42
  iterations, 324 s), of which 3900 is penalty for 78 unserved passengers.
- **Q=3 nearly saturates demand** — 290/300 served, and the objective falls by roughly
  3300 against Q=2's optimum. This is the largest single step in the sweep.
- **Q=4 and Q=5 show clear diminishing returns**: 294 and 297 served, and the remainder
  column stops falling, i.e. extra shuttles now add start and concurrency cost for very
  few extra passengers.

The ranking Q=3 < Q=4 < Q=5 by UB is suggestive but **not** established — those UBs come
from unconverged runs, and a longer budget could reorder them. Q=3's advantage over Q=2
is large enough to survive that caveat.

## What the discretisation axis says

`res15_q2` doubles T to 44 and is much harder: 8 iterations, LB essentially zero, and a
UB (5046.24) worse than the 30-minute cell reached in the same budget. Average wait
improves (28.70 vs 42.39 min), which is expected — finer slots let the model place
departures closer to demand — but at 120 s the run cannot exploit it.

**This cell does not show that 15-minute slots are worse.** It shows they need a bigger
budget. Re-run at simulation-phase settings before drawing any conclusion.

## Caveat on comparing with the first attempt

An earlier sweep, before D22, produced a better bracket for `res30_q2`
(LB 3826.99 / UB 4190.74). That run used 102 s master solves and overshot its declared
budget. Fewer but longer master solves gave a **better lower bound** than the 7 shorter
ones here — the LB comes from the master's best bound, so master solve length feeds it
directly.

That is a lead, not a result: the two runs did not have equal budgets, which is precisely
the defect D22 fixed. `master.per_iteration_time_limit_s` is now a real, testable knob,
and 30 s looks too tight for this instance. Worth an A/B at 30 / 60 / 102 s under an
honest budget.
