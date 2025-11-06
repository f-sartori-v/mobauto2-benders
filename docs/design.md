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
- `config.py` — YAML-based configuration
- `logging_config.py` — basic structured logging setup
- `cli.py` — `mobauto2-benders` command to run/validate/info
  - Subcommands:
    - `run` — main orchestrator using structured YAML (`configs/default.yaml`)

Implementation hints:

- Keep master variables limited to those needed in cuts; represent cuts as
  linear inequalities over master variables.
- On each iteration: solve master → evaluate subproblem(s) → add feasibility or
  optimality cuts → repeat until gap within tolerance.
- Use the `metadata` on `Cut` to stash any auxiliary values you need.

Benders cut for this project:

- Master exposes variables `yOUT[q,t]`, `yRET[q,t]`, and scalar `theta` with
  objective `min theta`.
- Add cuts in the form: `theta >= const + Σ coeff_yOUT[q,t]*yOUT[q,t] + Σ coeff_yRET[q,t]*yRET[q,t]`.
- Create a `Cut` with `metadata` keys:
  - `const`: float
  - `coeff_yOUT`: dict[(q,t) -> float]
  - `coeff_yRET`: dict[(q,t) -> float]
  The master translates these to a Pyomo `Constraint` when `add_cut()` is called.

Subproblem (assignment + waiting LP):

- Variables: `x_OUT[t,tau]`, `x_RET[t,tau]` (served demand flows), `u_OUT[t]`, `u_RET[t]` (unserved demand) — all nonnegative.
- Objective: minimize waiting cost `Σ (tau - t)^+ * x + p * Σ u`.
- Constraints:
  - Demand conservation: `Σ_{tau in [t, t+W]} x[t,tau] + u[t] = R[t]` for both directions.
  - Departure capacity: `Σ_{t: t ≤ tau ≤ t+W} x[t,tau] ≤ C[tau]` for both directions.
- Capacity from master: `C_out[tau] = S * Σ_q yOUT[q,tau]`, `C_ret[tau] = S * Σ_q yRET[q,tau]`.
- Dual multipliers: `alpha_OUT[t], alpha_RET[t]` for demand constraints (free), `pi_OUT[tau], pi_RET[tau]` for capacity (≥ 0).
- Optimality cut: `const = Σ alpha_OUT[t] R_out[t] + Σ alpha_RET[t] R_ret[t]` and coefficients `S * pi_*[tau]` spread over all `q` at each `tau`.

Scenarios and aggregation:

- Provide scenarios via a YAML list under `subproblem.params.scenarios` with `R_out`, `R_ret`.
- `average_cuts_across_scenarios = true` produces one averaged cut (weighted by `scenario_weights` if provided).
- Otherwise, one cut is generated per scenario and all are added to the master.
- `ub_aggregation`: how to combine per-scenario UB values (`mean`|`sum`|`max`). Defaults to `mean`.

Configuration (YAML)

YAML example (configs/default.yaml):

```
run:
  max_iterations: 100
  tolerance: 0.0001
  time_limit_s: 600
  log_level: INFO
  seed: 42

master:
  impl: pyomo
  params:
    solver: cplex_direct
    Q: 2
    T: 8
    trip_slots: 2
    Emax: 100.0
    L: 10.0
    delta_chg: 25.0
    binit: [60.0, 80.0]

subproblem:
  impl: pyomo_lp
  params:
    lp_solver: cplex_direct
    S: 4.0
    T: 8
    Wmax_slots: 2
    p: 1000.0
    R_out: [0, 1, 0, 2, 0, 0, 0, 0]
    R_ret: [0, 0, 0, 0, 0, 2, 0, 0]
```

Next steps:

1. Translate the master model and subproblem(s) from your report into the
   templates in `problem/`.
2. Choose and integrate a solver stack (e.g., Pyomo + CBC/CPLEX/Gurobi), then
   implement `solve()` and `add_cut()` in the master, and `evaluate()` in the
   subproblem.
3. Run `mobauto2-benders run --config configs/default.yaml`.
