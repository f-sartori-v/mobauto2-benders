# Benders Framework Design

This skeleton provides a clean separation between the Benders loop and the
problem-specific modeling. Map the formulations from `Report/Bender decomposition.pdf`
into the `problem/` package by implementing:

- `ProblemMaster` (src/mobauto2_benders/problem/master_impl.py)
- `ProblemSubproblem` (src/mobauto2_benders/problem/subproblem_impl.py)

Key modules:

- `benders/types.py` — shared enums and dataclasses for candidates, cuts, results
- `benders/master.py` — abstract master interface
- `benders/subproblem.py` — abstract subproblem interface
- `benders/solver.py` — the orchestrator loop (termination: time, iterations, gap)
- `config.py` — TOML-based configuration (uses stdlib `tomllib` on Python 3.11+)
- `logging_config.py` — basic structured logging setup
- `cli.py` — `mobauto2-benders` command to run/validate/info

Implementation hints:

- Keep master variables limited to those needed in cuts; represent cuts as
  linear inequalities over master variables.
- On each iteration: solve master → evaluate subproblem(s) → add feasibility or
  optimality cuts → repeat until gap within tolerance.
- Use the `metadata` on `Cut` to stash any auxiliary values you need.

Configuration (configs/default.toml):

```
[run]
max_iterations = 100
tolerance = 1e-4
time_limit_s = 600
log_level = "INFO"
seed = 42

[master]
impl = "to_fill"
[master.params]

[subproblem]
impl = "to_fill"
[subproblem.params]
```

Next steps:

1. Translate the master model and subproblem(s) from your report into the
   templates in `problem/`.
2. Choose and integrate a solver stack (e.g., Pyomo + CBC/CPLEX/Gurobi), then
   implement `solve()` and `add_cut()` in the master, and `evaluate()` in the
   subproblem.
3. Run `mobauto2-benders run --config configs/default.toml`.

