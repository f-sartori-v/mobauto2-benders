# mobauto2-benders

Benders decomposition for the MobAuto² shuttle fleet sizing and scheduling problem, with
the audit that established what its bounds do and do not certify.

The master decides the here-and-now schedule — which shuttle departs OUT or RET in which
slot, and when it charges. The subproblem is an LP that assigns passenger demand to those
departures and prices waiting time plus a penalty `p` per unserved passenger.

## Headline result

At **Q=3, T=44 slots (660 min / 15 min), four demand scenarios, 300 s**, the decomposition
does not produce a usable lower bound. Every master MIP solve terminates on its time
ceiling at an internal gap of **99.9%** — best bound 0.35 against its own incumbent of 636
— and giving it 3.4× the time and 7.7× the nodes raises the bound by 1.2%. The monolithic
MILP solves the same instance to optimality in 39 s.

**Upper bounds are unaffected**: each is an exhibited feasible schedule. **No optimality
gap may be quoted at Q≥3.** This is not a verdict on Q≤2, where the capacity anchor is
worth 1662–7737 on the empty master.

Evidence and reading rules: [`docs/phase1/README.md`](docs/phase1/README.md).
Decision log: [`docs/docs_decisions.md`](docs/docs_decisions.md).
Formulation: [`docs/BENDERS_SPEC_v4.md`](docs/BENDERS_SPEC_v4.md).

## Requirements

- Python 3.10
- `pyomo`, `pyyaml`
- CPLEX with Python bindings installed into the same environment. The configs use
  `cplex_direct` for both the master and the subproblem.

```bash
pip install -e .
```

## Reproducing the results

Every number quoted in `docs/` comes from a config in this repository. Run any of them:

```bash
python -m mobauto2_benders --config configs/phase1/base_off.yaml run
```

### The Phase 1 A/B (4 cells, 300 s each)

| config | anchor | LP phase | LB obtained | UB obtained |
|---|---|---|---|---|
| `configs/phase1/base_off.yaml` | off | off | 0.299040 | 2685.86 |
| `configs/phase1/anchor_on.yaml` | on | off | 0.313787 | 2267.36 |
| `configs/phase1/lp_on.yaml` | off | on | 0.350393 | 2310.36 |
| `configs/phase1/both_on.yaml` | on | on | 0.350915 | 2016.11 |

`configs/phase1/master_headroom.yaml` is the 900 s diagnostic that gives the master 102 s
per solve to show the bound is not time-starved.

**These runs are budget-truncated and print `NOT REPRODUCIBLE`** when master solves stop on
the clock rather than the gap. Your numbers will differ in the last digits and in the UB;
the LB stays in the 0.29–0.36 band. Differences under ~15% in the LB are machine noise, not
an effect.

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

72 tests, about 8 seconds. They cover cut soundness invariants, Magnanti–Wong provenance
and fallback, symmetry validity, the conditions under which a lower bound may be reported,
the recourse anchor, the LP phase, and configuration combinations that are refused.

## Configuration

`configs/default.yaml` is the annotated reference and is the source of truth — it carries
the meaning of every key inline, including which of them changes a reported bound. This
README deliberately does not duplicate it, because the duplicate went stale.

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
configs/        run configurations, including the Phase 1 cells
setups/         demand scenarios -- the inputs every config reads
tests/          72 tests, no network, CPLEX required for the soundness fixture
scripts/        sweep driver and a diagnostics smoke check
docs/           formulation, audit, decision log, Phase 1 evidence
```

## What is not in this repository, and why

Run logs, sweep transcripts, run manifests and superseded document versions are not
tracked. Everything needed to regenerate them is: each result is produced by a config that
ships here. A published repository should carry what reproduces a result, not the
transcript of every attempt (D38).

`Report/` holds the source PDFs and is not an input to any run.
