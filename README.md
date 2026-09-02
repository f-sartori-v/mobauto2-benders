# mobauto2-benders

Benders decomposition for the MobAuto² shuttle fleet sizing and scheduling problem, with
the audit that established what its bounds do and do not certify.

The master decides the here-and-now schedule — which shuttle departs OUT or RET in which
slot, and when it charges. The subproblem is an LP that assigns passenger demand to those
departures and prices waiting time plus a penalty `p` per unserved passenger.

## Headline result

At **Q=3, T=44 slots (660 min / 15 min), four demand scenarios**, the decomposition
produces a usable lower bound once it is given enough cuts, and is still not competitive.

The earlier headline here — *"does not produce a usable lower bound … internal gap 99.9%,
best bound 0.35"* — was measured with 14–19 cuts, because the master's LP phase was capped
at 10 iterations. It is **withdrawn** (D40, D45).

With the cap removed, 150 LP iterations at ~0.8 s each:

| | lower bound |
|---|---:|
| LP root relaxation, converged | **794.78** (reproducible) |
| after one 102 s MIP solve | ~1080 (single draw) |
| after one 410 s MIP solve | 1111.05 (two draws) |
| after one 1520 s MIP solve | **1148.65** (single draw) |
| monolithic MILP, for reference | 1658.86 in 947 s |

> **Two corrections to this table, both from re-measurement.**
>
> **The LP root is 794.7795573706986, not 794.624549571966** (D64). The older figure was
> never the LP root: at that commit the LP phase stopped on `iteration budget (150)` while
> still generating cuts (`cuts=1`, `rel_improve=1.11e-06`), so it was a *truncated* bound
> that the iteration cap happened to catch. After D61 the phase stops on
> `no cut generated` — it converges — at 794.7796. Both numbers are reproducible; only the
> second is a root. Bisected across four commits: D61 (S1+S2) is what changed it, and
> D63 (S3+S4) left it unchanged to 8 significant figures.
>
> **The monolith reference is 1658.86 in 947 s, not 1569.44 in 39 s** (D50). Every
> "% of the monolith's optimum" computed against the old pair is wrong.

Those three MIP points are close to linear in `ln(t)`: **14× more master time is projected
to buy about 8% of bound** (D47). Master seconds are not what the lower bound is short of.

The master's internal gap falls from **0.9994 to about 0.20**, and an upper bound appears
for the first time at this size. So the earlier reading — that the master's branch and
bound cannot lift its own bound — described a symptom of a root relaxation sitting at zero,
not a property of the master.

**It is still not competitive.** The monolith solves the same instance to optimality in
**947 s at 1658.86** (D50); the run above spent ~1000 s to reach a lower bound of 1148.65,
which is **69.2% of that optimum** (D54), and it comes from a clock-truncated run, so it is
a single draw. The "39 s" and the 46% gap that used to stand here were computed against the
withdrawn 1569.44 pair and are void — see the correction above the table.

Adding Benders iterations on top does not move the bound — it oscillates in 1064–1090 with
no trend, a spread the size of the run-to-run noise. The binding constraint has moved from
the cut set to **the master solve being truncated at ~20% internal gap**.

**Upper bounds were never in doubt**: each is an exhibited feasible schedule. Nothing here
is a verdict on Q≤2, where the capacity anchor is worth 1662–7737 on the empty master.

Reading rules for the numbers below: [§ Reading rules](#reading-rules).
Decision log: [`docs/docs_decisions.md`](docs/docs_decisions.md).
Formulation: [`docs/BENDERS_SPEC_v4.md`](docs/BENDERS_SPEC_v4.md).

> **Start at [`docs/PROJECT_STATE_v6.md`](docs/PROJECT_STATE_v6.md), not here.** It is the
> current state in one file: where the three claims stand, **every withdrawn number in one
> table**, the D-number remapping for citations taken from outside this repository, and how
> to install and run the project locally with or without a CPLEX licence.
>
> This README is still accurate, but it is long and it is arranged historically, so a figure
> on this page can be a live claim or a withdrawn one depending on which paragraph it is in.
> Two outside reports were written from it six weeks apart and both quoted numbers this
> repository had already withdrawn (D66, D67 §1). The v6 table exists so that cannot happen
> again; `docs/REPORT_REVIEW_v1.md` records the first of those two in detail.

## Requirements

- Python 3.10
- `pyomo`, `pyyaml`
- CPLEX with Python bindings installed into the same environment. The configs use
  `cplex_direct` for both the master and the subproblem.

```bash
pip install -e .
cp configs/default.example.yaml configs/default.yaml
```

The second line is needed once. `configs/default.yaml` is not tracked — it is the live
experiment file and is edited constantly, so its edits would otherwise land in every
commit. `configs/default.example.yaml` is the tracked copy. The named configs under
`configs/phase1/` and the two `baseline_d9*` files are tracked and run without it.

## Reproducing the results

Every number quoted in `docs/` comes from a config in this repository. Run any of them:

```bash
python -m mobauto2_benders --config configs/phase1/base_off.yaml run
```

### The cut-budget runs (these are the current result)

```bash
python -m mobauto2_benders --config configs/phase1/lp_only_150.yaml run
```

| config | what it does | LB | UB |
|---|---|---:|---:|
| `lp_only_150.yaml` | 150 LP iterations, no MIP phase | **794.7795573706986** | none claimed |
| `lp150_then_mip1.yaml` | the same, then one MIP iteration | ~1080 | ~2170 |
| `lp150_then_mip8.yaml` | the same, then eight | 1089.98 | 2030.86 |
| `lp150_then_control.yaml` | the same, then ONE long master solve (410 s) | 1111.05 | 2351.86 |
| `lp150_then_control_1800.yaml` | the same at a 1800 s budget (1520 s of solve) | **1148.65** | 2349.61 |
| `lp150_then_bnc.yaml` | the same, then one branch-and-cut tree | 1004.22 | **1923.86** |

The last three answer a question the first three cannot: the loop rebuilds a ~19 000-node tree
every iteration to add one cut, so is the teardown the problem? Partly — one plain solve over
the same 150 cuts beats eight loop iterations in half the time. But recovering it with a
CPLEX lazy callback costs more than it saves: registering one disables dual reductions,
restricts presolve to crushing forms and stops repeat represolve, and that is worth 9.6% of
lower bound here. Branch-and-cut buys the *upper* bound instead (18% better, because every
incumbent it accepts has been priced by the subproblem). Details and the two defects the
measurement found: D46.

`lp_only_150.yaml` is **reproducible**: two independent executions on the current code gave
`794.7795573706986` to the last digit, and the superseded `794.624549571966` reproduced three
times on the code that produced it. An LP has no branch and bound and never stops on the
clock, so D26 does not apply to it. It is the only number in this README that is not a single
draw.

**It is now a converged LP root rather than a truncated bound**, and that distinction is the
whole of D64. The phase used to stop on `iteration budget (150)` with a cut still being added
every iteration; it now stops on `no cut generated`. The mechanism that fits the evidence:
the cut intercept used to be *imposed* as `Q(y) − Σ dm·y_inc` rather than derived from the
duals, which let it sit up to `face_tol` above the true dual cut — so a cut could look
marginally violated when it was not, and the phase kept adding near-duplicates without ever
terminating. Deriving the intercept from `α` (D61) removed the manufactured violations. That
is the explanation consistent with every number measured, not one isolated by a separate
experiment.

The other five print `NOT REPRODUCIBLE` — their solves stop on the clock, and the same
iteration under the same configuration gave 1088.07 in one run and 1080.36 in another. The
two branch-and-cut rows were each drawn twice and held to within 0.4%
(control 1106.15 / 1111.05, tree 1004.62 / 1004.22), which is why the gap between them is
readable as an effect rather than as noise.

### The Phase 1 A/B (4 cells, 300 s each) — superseded

Kept because the configs reproduce it and because the reason it misleads is worth seeing.
Every cell here ran with the LP phase capped at 10 iterations, which samples the flattest
point of a curve that only starts climbing around iteration 12.

| config | anchor | LP phase | LB obtained | UB obtained |
|---|---|---|---|---|
| `configs/phase1/base_off.yaml` | off | off | 0.299040 | 2685.86 |
| `configs/phase1/anchor_on.yaml` | on | off | 0.313787 | 2267.36 |
| `configs/phase1/lp_on.yaml` | off | on | 0.350393 | 2310.36 |
| `configs/phase1/both_on.yaml` | on | on | 0.350915 | 2016.11 |

`configs/phase1/master_headroom.yaml` is the 900 s diagnostic that gives the master 102 s
per solve. Its conclusion — that the bound is not time-starved — held for the cut set it
was run with, and does not carry over to the 150-cut master.

### Reading rules

These conditions travel with any number lifted from the tables above. They are not caveats
added after the fact; each one exists because a number was once quoted without it.

- **Anything from a run with a MIP phase is one draw**, and prints `NOT REPRODUCIBLE`
  because master solves stop on the clock rather than the gap (D26). `lp_only_150.yaml` is
  the exception and the only one: it is pure LP, so it reproduces exactly.
- **Say which cut budget produced a number.** The single largest source of wrong
  conclusions in this project was quoting a bound measured at 10–19 cuts as if it described
  the method. 794.78 is a converged-LP number; 0.35 is a 14-cut number; they are not
  comparable and neither supersedes the other without the budget stated.
- **Say whether the LP phase converged or was truncated**, and read the reason it printed —
  `no cut generated` is a root, `iteration budget (N)` is a bound that the cap happened to
  catch. This rule exists because `794.624549571966` was published as "LP root relaxation,
  150 cuts" for months while its own log said `iteration budget (150)` with a cut still
  being added every iteration (D64). Reproducible and truncated are not exclusive: that
  number reproduced three times and was still not a root.
- **Each UB is real** — an exhibited feasible schedule. A Benders gap may now be quoted at
  Q=3, but only from the MIP-phase runs, only as a single draw, and never from the Phase 1
  table, whose LB is not within a factor of 1000 of useful.
- **The LP phase claims no upper bound while it is active**: a fractional schedule cannot be
  exhibited. The UB for `lp_on` and `both_on` comes only from MIP-phase iterations.
- **The master's objective is not an upper bound.** It is `first_stage + theta`, and theta is
  bounded only by the cuts the master currently holds, so it can sit *below* the true
  optimum. Every UB in this file comes from pricing an exhibited schedule with the
  subproblem. This rule exists because the branch-and-cut driver briefly reported the master
  objective as a UB, which would have produced a 14% Benders gap that does not exist (D46).
- **State `(p, W_max) = (50, 60)` and `concurrency_penalty = 0.25`** on any table derived
  from these runs (D18). `concurrency_penalty` is active in the objective and is not part of
  the published formulation. The manifest records all three.
- **Every reported number states which subproblem mode produced it** — `mw`,
  `mw_fdiff_fallback`, `dual`, `finite_difference`, or `mixed(a+b)` for a multi-scenario
  aggregate.

### The regression baselines

```bash
python -m mobauto2_benders --config configs/baseline_d9.yaml run
python -m mobauto2_benders --config configs/baseline_d9_multi.yaml run
```

These two converge on the gap and **are** reproducible: `LB 2891.086867 / UB 4962.990000`
and `LB 879.511303 / UB 3382.240000`. They are computational fingerprints of the post-D30
implementation, not results about the method. Use them to check that a change to the code
did not move behaviour unintentionally.

### Tests

```bash
python -m unittest discover -s tests
```

276 tests, about a minute (D68, D71, D72). They cover cut soundness invariants, the Phase 5
exactness gate against the monolith (D62), Magnanti–Wong provenance
and fallback, symmetry validity, the conditions under which a lower bound may be reported,
the recourse anchor, the LP phase, configuration combinations that are refused, the
runtime-split instrumentation (D71), and the minute-departure validator (D72).

**A CPLEX licence is no longer needed to run them.** Every gate used to name `cplex`
literally, so a checkout without one skipped 63 tests — including E1/E2, cut
underestimation, the Phase-5 gate and both bound-validity invariants — and still printed
green. `pip install highspy` runs all of them; CPLEX is still preferred when present, so a
licensed checkout measures exactly what it measured before. `MOBAUTO2_TEST_SOLVER` pins a
backend, and raises rather than skipping if it is absent. What stays CPLEX-only is 9 tests
of CPLEX-specific machinery — the branch-and-cut callback and `CPXPARAM_*` name resolution
— and no formulation invariant is among them (D67).

## Configuration

`configs/default.example.yaml` is the annotated reference and is the source of truth — it
carries the meaning of every key inline, including which of them changes a reported bound.
This README deliberately does not duplicate it, because the duplicate went stale. Copy it
to `configs/default.yaml` to get a working local config; that copy is not tracked.

Three limits are easy to confuse:

| key | meaning |
|---|---|
| `solver.total_time_limit_s` | wall clock for the whole Benders loop |
| `solver.tolerance` | relative Benders gap that counts as converged |
| `master.per_iteration_time_limit_s` | ceiling on ONE master solve |
| `master.per_iteration_mipgap` | ceiling on the master's internal gap per iteration |

Any table derived from a run must state `(p, W_max)` **and** `concurrency_penalty`, which
is active in the objective and absent from the published formulation. The run manifest
records all three.

Other commands:

```bash
python -m mobauto2_benders --config configs/default.yaml validate
python -m mobauto2_benders --config configs/default.yaml info
```

## Layout

```
src/mobauto2_benders/
  benders/      decomposition loop, cut filtering, shared types
  problem/      MobAuto2 master and subproblem models
  app.py        parameter assembly    config.py  YAML schema (v2)
src/mobauto2_milp/
                the monolithic MILP -- the benchmark reference, see below
configs/        run configurations, including the Phase 1 cells
  milp/         configs for the monolith
setups/         demand scenarios -- the inputs every config reads
tests/          no network, CPLEX required for the soundness and monolith fixtures
scripts/        sweep driver and a diagnostics smoke check
docs/           formulation, audit, decision log, Phase 1 evidence
```

## Benchmarking against the monolithic MILP

Every bound this project reports is checked against a feasible objective produced by a
*monolithic* MILP rather than by a Benders run of this same code — spec non-negotiable 8.
On the `baseline_d9` instance that objective is **4183.24**.

```bash
mobauto2-milp --config configs/milp/baseline_d9_monolith.yaml run
```

It solves to proven optimality in about 7 seconds and prints
`status=OPTIMAL best_lb=4183.24 best_ub=4183.24`.

**Why it ships here.** For months this code was not in the repository — `aux_py/` was
empty, and 4183.24 appeared only as prose in the docs and a hardcoded constant in two test
files. The check could not be regenerated and its independence could not be inspected
(D50). `tests/test_monolith_reference.py` now fails if the package goes missing, if the
number changes, if the solve stops on the clock instead of the gap, or if the monolith's
config drifts away from the instance `configs/baseline_d9.yaml` describes.

**What "independent" means, precisely.** The monolith attaches the exact recourse LP and
pins `theta` to it by *equality*, so there is no cut and no `theta` approximation — the
whole D30 defect class cannot reach it. That is independence from the **decomposition**.
It is not independence from the **formulation**: `mobauto2_milp/model.py` is a second copy
of the first-stage model that `mobauto2_benders/problem/master_impl.py` also implements,
kept in sync by hand. A defect present in both copies is invisible to this check. Cite the
number that way.

## What is not in this repository, and why

Run logs, sweep transcripts, run manifests and superseded document versions are not
tracked. Everything needed to regenerate them is: each result is produced by a config that
ships here. A published repository should carry what reproduces a result, not the
transcript of every attempt (D38).

`Report/` holds the source PDFs and is not an input to any run.
