# Code audit v3 — `19_MobAuto2_Benders`, consolidated

Supersedes `AUDIT_2026-08-05.md`. Companion: `docs/decisions.md` (D1–D13) and `BENDERS_SPEC_v3.md`.
Severity unchanged: **C** = breaks correctness of reported results · **H** = affects numbers/their
interpretation · **M** = exposition/maintainability · **L** = cosmetic.

**Files read this round, in full or by targeted trace:** `master_impl.py` (full), `core.py` (full),
`solver.py` (full), `subproblem_impl.py` (full function map + deep trace of the production-path
functions: `evaluate()` dispatch, `solve_subproblem`, `solve_mw_dual`, plus enough of the elastic
path — `Act_OUT`/`Act_RET`, `solve_refined_lp_relaxation_cut` — to confirm the deprecation
rationale), `config.py` (full), `app.py` (full), `cli.py` (full), `types.py`, `tolerances.py`,
`cplex_log.py` (full, both small). **Not read:** `aux_py/`, `scripts/`, `setups/`, `tests/`,
`pyproject.toml` details. Recommend a pass over `tests/` next, to see what's already covered before
writing the §5 additions from the spec.

---

## RESOLVED this round (were open or misdiagnosed in earlier passes)

### C1 (was: silent q-aggregation of cut coefficients) — resolved conditional on D9
Traced end to end. On the production path (`solve_subproblem` → `solve_mw_dual` or plain
`use_dual`), the duals `pi_OUT[τ]`/`pi_RET[τ]` are aggregated per-slot only, never per-vehicle, and
are explicitly broadcast to every `q` before reaching `coeff_yOUT`/`coeff_yRET`. Master's
`aggregate_cuts_by_tau: true` is safe here. The real defect was isolated to
`solve_refined_lp_relaxation_cut`'s `Act_OUT[q,τ]`/`Act_RET[q,τ]` constraints, which are genuinely
per-vehicle with no equality guarantee across `q` — that whole path is deprecated per D9.
**Required hardening kept regardless:** assertion in `master_impl._add_cut` that coefficients are
equal across `q` before aggregating (spec §2.7), so a future regression fails loudly.

### H1 (unused_capacity_penalty / fill_first_epsilon) — corrected
- `unused_capacity_penalty`: confirmed dead. It is declared in `SubproblemSection` (config.py),
  validated, and threaded into the subproblem params dict in `app.py::_prepare_params`
  (`sp["unused_capacity_penalty"] = ...`) — but grepping all of `subproblem_impl.py` for this key
  returns zero hits in any objective or constraint. It is schema-validated and silently inert. Not
  a modelling distortion (it does nothing), but a trap: someone will eventually set it expecting an
  effect. Resolve per spec §4.
- `fill_first_epsilon`: confirmed benign. Appears exactly once per relevant term, weighted by a
  within-slot layer index `k`, default 1e-6 — a legitimate, negligible packing tie-breaker, not a
  distortion of the waiting-cost objective.
- `concurrency_penalty` (master, not subproblem): still real and still in the master objective via
  `eOut`/`eRet` auxiliaries. Not addressed this round — recommend the same "confirm it's wanted, or
  remove" treatment as `unused_capacity_penalty`, since it wasn't in the original published
  formulation either.

### H5 (Magnanti–Wong core point) — corrected, was too harsh
`benders/core.py::CorePoint` (plain EMA of binary incumbents, no interior guarantee) is very likely
**dead code** — imported in `solver.py` but not instantiated anywhere traced. The actual, active
mechanism is inline in `BendersSolver.run()`: an EMA of the aggregated incumbent, clipped every
iteration to `[eps, Q−eps]` per time slot. That *is* a legitimate interior point of the aggregate box
`[0,Q]^T` the MW theory needs, contrary to the first-pass assessment (which read only `core.py` and
extrapolated). Action: delete `CorePoint` once confirmed unreferenced (or wire it in if there's a
reason to prefer it over the inline version — don't leave both).

### "Unit mismatch" concern from the original spec — withdrawn
The original spec's §2.5 proposed `(τ−t)·δ` (converting to minutes inside the LP). Traced code shows
`(τ−t)` is used bare, in slots, consistently across `solve_subproblem`, `solve_mw_dual`, and the
reporting layer's minute conversion (`avg_wait_min = wait_slots × slot_res / pax_served`) happens
strictly downstream, in `solver.py`/`app.py`. The code was already correct; the spec's proposed "fix"
would have been the actual bug. See D7/D8.

---

## NEW — decided by the author this round (see docs/decisions.md, not re-litigated here)

- ρ = 70 km-eq/h, linear (D1)
- W_max and p are live sensitivity parameters, not constants to freeze (D2, D3)
- ε = 0.01 with a stated unit rationale (D4)
- The single-number dominance threshold (`W_eff = min(W_max,p)`) is withdrawn in favour of the
  load-dependent explanation already implicit in the `(τ−t)·x[t,τ]` objective (D5)
- Horizon: 10 h now, 24 h later; the slot-boundary buffer is intentional (D6)
- **The elastic/minute-level subproblem relaxation is deprecated from the default path** (D9) —
  this is the largest architectural decision of the session and reframes several earlier findings
  (below).

---

## OPEN — action required

### C2 — the "anti-trivial idle" fallback cut is still not a valid Benders cut *(C, unchanged)*
`_build_anti_trivial_cut` in `subproblem_impl.py` (confirmed still present, called from `evaluate()`
in the degenerate-cut-after-fallback path) adds `Σ(yOUT+yRET) >= min_total_starts` — a hard
constraint on the master, not a cut derived from LP duality. It removes feasible master solutions,
so the master stops being a valid relaxation. Diagnoses a real problem (a degenerate/all-idle
incumbent that ordinary cuts can't dislodge); the fix is a valid initial lower bound on θ (e.g. one
subproblem evaluation at a heuristic ŷ before the first master solve — see original C2 writeup),
not a forced-departure constraint. **Not addressed by D9** — this fires from the shared
`evaluate()` scaffolding regardless of which subproblem mode is active. Still open.

### D9 execution — two mechanical steps not yet done
1. Flip `enable_temporal_refinement` default `True → False` in `subproblem_impl.py::evaluate()`.
   One line, zero schema risk (confirmed: the flag isn't in the YAML schema at all, so no config
   file needs to change).
2. Decide the fate of the ≈800 lines exclusive to the elastic path (function map below) — delete or
   quarantine to `experimental/`. Until this happens, the audit surface stays inflated and there's a
   real risk of accidental reactivation (e.g. someone flips the flag back not realising why it was
   set False).

### H2/H3 (W_max, horizon) — no longer "inconsistencies to fix", now "sweep to run"
Reframed by D2/D3/D6. Not an open defect; open **work item**: run the sweep, report every table with
its (p, W_max) pair stated, re-run the headline robustness table last per the original re-run matrix
priority (`BENDERS_SPEC.md` v1 §7 — unchanged).

### H4 — duplicate symmetry-breaking constraints *(H, unchanged, confirmed again)*
`use_fifo_symmetry` and `symmetry_breaking` in `master_impl.py` encode the same cumulative ordering
in both directions (`≤` and `≥` against the previous vehicle), both default `true` in
`configs/default.yaml`. Redundant, and — per the earlier finding — only valid for a homogeneous
fleet with identical initial state (a precondition never checked). Keep one; add the precondition
guard.

### H6 — θ disaggregation (per-scenario vs directional) *(H, unchanged — empirical, not a defect)*
Confirmed in `master_impl.py`: `theta_per_scenario: true` (default) disables directional
(`theta_out`/`theta_ret`) disaggregation. This is a real either/or in the code, not a bug. Per D11,
resolve by running both on the same instance and comparing — not resolvable by further reading.

### M1 — explicit return-leg energy constraint *(M, unchanged, still worth adding)*
`b[q,t] >= L·yRET[q,t]` is implied (via exclusivity forcing `c=0` when `yRET=1`, hence `gchg=0`,
hence `b[q,t] >= 0` requires `b[q,t] >= L` before the trip) but not explicit. Adding it tightens the
master's LP relaxation, which is 85.7% of runtime per the published computational study — worth
doing for that reason alone, independent of correctness (no infeasible schedule is currently
reachable).

### M2 — charge-before-idle policy, now understood as intentional canonicalisation
Confirmed purpose from the author: with e.g. 6 idle slots at the depot (5 needed to charge, 1 truly
idle), the policy picks the canonical ordering (charge first, then idle) instead of leaving the
solver free to interleave equivalent orderings. This is a legitimate readability/canonicalisation
device for reporting, not a physical constraint and not a bug. Recommend: keep it, but gate it behind
a named config flag (`policy.charge_before_idle`, default on) rather than hard-coding it, so it's
visible as a deliberate choice in the config diff whenever it's touched.

### M3 — exception handling still too broad *(M, unchanged)*
`master_impl.py` and `solver.py` both contain many bare `except Exception: pass`/fallback blocks
around solver calls, stats extraction, and log parsing. Confirmed again this round while tracing the
bound-provenance chain (`solver_results → cplex_api → cplex_log → computed`). Narrow to specific
exceptions; on failure, record `bound_source: "unavailable"` rather than silently substituting.

### M4 — per-cell `add_component` loops instead of indexed constraints *(M, unchanged)*
Confirmed again in `master_impl.py` (`C1b_intrip_*`, `C2a_locL/M_*`, `C4_bal_*`, `C4_chg*`,
`C_no_recharge_*`, all symmetry blocks, every cut). Author confirms this is a real performance
problem ("percebi que logo a gente trava, por causa desses loops"). Convert to indexed
`pyo.Constraint(m.Q, m.T, rule=...)`. Re-measure the master/subproblem/cutgen time split after —
the published 85.7%/14.2% breakdown may partly be model *construction*, not solving.

### M5 — diagnostics/LP-dump on by default *(M, unchanged)*
`emit_reports` defaults true; every master solve writes a symbolic LP + solver log. Author confirms:
default off, explicit flag to turn on for debugging. One default flip in `app.py`'s `_prepare_params`
(`mp["emit_reports"] = ...`) or the config default.

---

## NEW this round

### N1 — `subproblem_impl.py` defines `_slot_idx_from_minutes` twice *(L)*
Lines ~132 and ~163, both inside `evaluate()`'s scope. The second silently shadows the first
(dead code, not a runtime bug, but worth removing — same class of issue as the earlier
`Σ := {OUT,RET,REC,IDL,IDM,OUT}` duplicate-OUT typo found in `[R]`).

### N2 — `cplex_log.py`'s bound recovery is regex-over-log-text, and is a real fallback path *(M)*
Used whenever Pyomo/the solver interface doesn't expose `best_bound` directly — confirmed this is
reached in practice (it's third in the fallback chain: `solver_results → cplex_api → cplex_log →
computed`). Regex against CPLEX's free-text log output is inherently version-fragile. Not urgent, but
add a test fixture (a captured sample log) so a CPLEX version bump that changes log formatting is
caught by CI instead of silently degrading `best_bound` to `None` → falling through to the
`"computed"` source, which itself depends on having both `incumbent` and `best_bound` — a chain of
three fallbacks that could plausibly all miss at once with no loud failure.

### N3 — `evaluate()`'s branch order means `mw_enabled` silently overrides `use_dual_slopes` *(M)*
Confirmed: the dispatch is `if refined_lp_relaxation: ... elif nominal_lp_proxy/proxy_dual: ...
elif mw_enabled: ... elif use_dual: ... else: finite_difference`. Since `mw_enabled` defaults `true`,
setting `use_dual_slopes: true` without also setting `use_magnanti_wong: false` has **no effect** —
the MW branch is taken first regardless. Not a bug (MW is the better choice when available), but an
undocumented precedence rule that will confuse anyone trying to A/B test `use_dual_slopes` (relevant
directly to closing D11/H6). Document this precedence explicitly wherever `use_dual_slopes` is
mentioned (README, config comments).

### N4 — function map of `subproblem_impl.py`, production vs deprecated
For the eventual deletion/quarantine step (§ D9 follow-up):

**Production path (keep):** `ProblemSubproblem.__init__/_is_report/_vprint/_parse_candidate_indices`,
`evaluate()` (the dispatcher itself — needs light editing to drop the elastic branches, not
deletion), `_load_doc`, `_aggregate_requests`, `_ok`, `_cand_theta`, `_dbg`,
`_candidate_is_all_idle`, `_load_demand_from_file`, `solve_mw_dual` + its nested rules, `coeffs_by_fdiff`
(fallback, keep for the diagnostic-only fdiff mode), `_is_degenerate_cut`, `_build_anti_trivial_cut`
(pending C2 fix), `_model_size_stats`, `_apply_solver_time_limit`, `_has_loaded_solution`,
`_maybe_export_lp`, `SPParams`, `solve_subproblem` + its nested rules.

**Elastic-path only (candidate for deletion/quarantine):** `_load_exact_arrivals_from_file`,
`_pick_vehicle_for_tau`, `_expand_time_slopes`, `_slot_exposure`, `_top_probe_slots`,
`_restricted_temporal_fdiff`, `_proxy_cut_from_nominal_lp`, `class RefinedServiceEvent`,
`_charge_profile`, `_service_slot_window_lb_min/ub_min`, `_slot_nominal_departure_min`,
`_demand_release_min`, `_wait_slots_from_minute`, `_slot_index_from_minute`,
`_extract_exact_arrival_minutes`, `_eligible_arrivals_by_slot_and_cutoff`,
`_direction_eligible_arrivals_by_minute`, `_validate_realized_departure_minutes`,
`_build_service_events`, `_charge_minutes_between`, `_first_violation`,
`solve_refined_lp_relaxation_cut`, `solve_refined_subproblem` (includes its own `_precedence_rule` —
worth noting this function models inter-event sequencing that the LP-relaxation-for-cuts function
does *not*, i.e. even within the elastic path, the model used to generate cuts and the model used to
evaluate the "true" recourse are structurally different — a second-order reason D9 is the right call,
independent of the master-linkage argument).

Rough size: the elastic-only block is ≈800 of the file's 3266 lines (~25%), plus a proportional share
of `evaluate()`'s branching logic that can also be deleted once D9 is final.

### N5 — `app.py::_prepare_params` is the single place all config → params mapping happens *(map, not a defect)*
Useful to know for the refactor: every `sp[...]`/`mp[...]` key set here is exactly the params dict
`ProblemMaster`/`ProblemSubproblem` receive. Confirmed no other place mutates these before they reach
the model builders. This is a good, single-source-of-truth mapping point — the config-extraction
step in `BENDERS_SPEC.md` §3/§4 should extend this function rather than bypass it.

---

## Immediate order of work (supersedes v1/v2's ordering)

1. **D9 mechanical steps** — flip the default, decide delete-vs-quarantine for the elastic block.
   Nothing else in this list matters much until this lands, since it changes what "the model" even
   means for every subsequent item.
2. **C2** — replace the anti-trivial constraint with a valid initial θ bound.
3. **Hardening from C1/D10** — add the q-invariance assertion in `master_impl._add_cut`, remove the
   unconditional constant re-anchoring (keep only as a failing check).
4. **D13** — run with logging, grep for `[CHECK FAIL]`, close the sign-convention question
   empirically.
5. **H4** — collapse duplicate symmetry constraints; add the homogeneous-fleet precondition guard.
6. **M4** — indexed constraints instead of per-cell loops; re-measure the runtime split after.
7. **M3, M5, N1, N2** — hygiene pass.
8. **§4 of the spec** — resolve `unused_capacity_penalty` and `concurrency_penalty` (wire in or
   remove, don't leave inert-but-validated).
9. Then, and only then: the parameter sweeps (D2/D3), the θ-disaggregation A/B (D11), and the full
   re-run matrix from `BENDERS_SPEC.md` v1 §7.
