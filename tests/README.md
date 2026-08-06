# Tests

Run everything:

```
python -m unittest discover -s tests -v
```

Run only the fast tests (no solver, no CPLEX licence needed):

```
python -m unittest discover -s tests -p "test_fast_*.py" -v
```

Stdlib `unittest` rather than pytest, so there is nothing to install.

## Layout

| File | Needs CPLEX | What it protects |
|---|---|---|
| `test_fast_config.py` | no | Config schema: unknown keys rejected, defaults that matter |
| `test_fast_model.py`  | no | Master model structure: symmetry form, cut aggregation guard |
| `test_fast_cplex_log.py` | no | Bound recovery by regex over CPLEX log text (audit N2) |
| `test_solver_soundness.py` | **yes** | End-to-end invariants that caught real bugs |

The fast tests build Pyomo models but never call a solver, so they run in
about a second and can be run on every edit. The soundness tests run a short
Benders loop and take a couple of minutes.

## Why these particular assertions

Each soundness test corresponds to a defect that was live in this codebase and
was found by hand. They exist so the next regression is caught automatically.

- **MW actually succeeds.** `solve_mw_dual` returned `None` on every call for an
  unknown length of time, so every cut came from the finite-difference fallback,
  which is not a valid lower bound. Nothing failed loudly; the run just reported
  bounds it had no right to.
- **LB never exceeds a known feasible objective.** Prefix-ordering symmetry
  breaking removed feasible schedules, so the master was not a relaxation and its
  bound could sit above a solution we had already exhibited. This single
  inequality is what exposed it.
- **The per-shuttle table sums to the served total.** The end-of-run report drew
  the schedule from one solution and the passenger counts from another, printing
  a table summing to 126 directly beneath "Pax served: 222/300".
- **Gap reporting is disabled when cuts are not valid lower bounds.** The
  Magnanti-Wong fallback claimed `cut_valid_lower_bound = True` while emitting
  finite-difference cuts.
