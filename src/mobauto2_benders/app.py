from __future__ import annotations

import math
from pathlib import Path
# Imported under an alias on purpose: `_prepare_params` binds a local named
# `time` to cfg.model.time, so a plain `import time` would read as available in
# this module and be shadowed exactly where someone would reach for it.
from time import monotonic as _monotonic

from .config import DEFAULT_CONFIG_PATH, load_config, resolve_energy_params
from .logging_config import setup_logging
from .benders.solver import BendersSolver, BendersRunResult


def import_problem_impl():
    """Import default problem-specific implementations.

    Expects classes `ProblemMaster` and `ProblemSubproblem` in
    `mobauto2_benders.problem.master_impl` and `.subproblem_impl`.
    """
    try:
        from .problem.master_impl import ProblemMaster  # type: ignore
        from .problem.subproblem_impl import ProblemSubproblem  # type: ignore

        return ProblemMaster, ProblemSubproblem
    except Exception as exc:  # noqa: BLE001 - provide friendly message
        raise SystemExit(
            "Problem-specific implementations not found.\n"
            "Create classes `ProblemMaster` and `ProblemSubproblem` under:\n"
            "  src/mobauto2_benders/problem/master_impl.py\n"
            "  src/mobauto2_benders/problem/subproblem_impl.py\n"
            "Each should extend the abstract base classes in\n"
            "  src/mobauto2_benders/benders/master.py and subproblem.py\n"
            f"\nOriginal import error: {exc}"
        )


def _apply_solver_backend_override(cfg, backend: str | None) -> None:
    """Point all three solver fields at one backend.

    Every shipped config names a CPLEX plugin, so without this a checkout with no
    licence can run the test suite (D67) and still not run the solver at all --
    the one thing the configs are for. `--solver appsi_highs` closes that gap.

    All three fields move together on purpose. A master on one backend and a
    subproblem on another is a configuration nobody has measured, and the seeding
    LP phase would not be the one the archived numbers came from. The manifest
    records `solver.master` and `solver.subproblem`, so an overridden run says so
    in its own provenance rather than looking like the config it started from.

    `cplex_persistent` is refused here for the same reason `config.py` refuses it
    as `master.solver_backend`: branch-and-cut builds its own persistent solver for
    the tree, and repointing the master's would make the cuts the tree starts from
    different from the ones run 2 seeds it with.
    """
    if backend is None:
        return
    backend = str(backend).strip()
    if not backend:
        return
    if backend.lower() == "cplex_persistent":
        raise ValueError(
            "--solver cplex_persistent is not accepted. Branch-and-cut creates its "
            "own persistent solver for the tree (master.branch_and_cut); the "
            "master's own backend stays a non-persistent one so the seeding LP "
            "phase reproduces run 2 exactly."
        )
    cfg.solver.master_solver = backend
    cfg.solver.subproblem_solver = backend
    cfg.master.solver_backend = backend
    if backend != "cplex_direct":
        # CPXPARAM_* keys are mapped by name for cplex_direct and mean nothing to
        # another backend. Dropping them is honest; passing them through would let
        # a run claim options it never applied -- the signature defect (D64 (b)).
        cfg.master.cplex_options = {}


def _apply_run_overrides(cfg, overrides: dict | None) -> None:
    if not overrides:
        return
    _apply_solver_backend_override(cfg, overrides.get("solver_backend"))
    run_overrides = overrides.get("run") if isinstance(overrides, dict) else None
    if not isinstance(run_overrides, dict):
        return
    for key, value in run_overrides.items():
        if hasattr(cfg.run, key):
            try:
                setattr(cfg.run, key, value)
            except Exception:
                pass


def _set_if_not_none(target: dict, key: str, value) -> None:
    if value is not None:
        target[key] = value


def _scenario_containers(cfg) -> list | None:
    """The scenario list in the same form the subproblem consumes it.

    `scenarios` (inline dicts) wins over `scenario_files` (paths), matching
    ProblemSubproblem.evaluate. Returns None when the run is single-scenario.
    """
    if isinstance(cfg.data.scenarios, list) and cfg.data.scenarios:
        return list(cfg.data.scenarios)
    if isinstance(cfg.data.scenario_files, list) and cfg.data.scenario_files:
        return list(cfg.data.scenario_files)
    return None


def _demand_vectors(
    container, T: int, slot_resolution: int
) -> tuple[list[float], list[float]]:
    from .problem.subproblem_impl import aggregate_requests, load_demand_doc

    if isinstance(container, (str, Path)):
        container = load_demand_doc(Path(str(container)))
    return aggregate_requests(container, T, int(slot_resolution))


def _recourse_bound_data(cfg, slot_resolution: int) -> dict | None:
    """Data the master needs for the recourse lower bound on theta.

    Model building must stay pure (spec non-negotiable 2), so the demand vectors
    are aggregated here and passed in rather than read inside build_master. They
    must be the SAME post-truncation vectors the subproblem uses: a total larger
    than the subproblem's would demand more unserved passengers than can actually
    occur, and the inequality would cut off the optimum instead of bounding it.
    That is why aggregation is shared rather than reimplemented.

    Multi-scenario runs use the weighted mean demand across scenarios. The
    inequality holds per scenario, so summing the scenario rows with weights
    w_s >= 0 summing to 1 gives

        sum_s w_s * theta_s  >=  p * ( sum_s w_s * R_s_cum[j]  -  S * Y_cum[j+W] )

    and `sum_s w_s * theta_s` is exactly the recourse term the master minimises
    (build_master weights theta_s the same way, and the averaged-cut path weights
    the single theta's cut the same way). Y_cum is common to every scenario --
    the first stage is here-and-now -- so it leaves the sum untouched.

    Note the direction of the resulting slack: the mean is at most the max, so
    this stays valid even under the multi-cut/single-theta combination where the
    master's theta is forced above every scenario individually. Weaker there, not
    wrong.
    """
    T_minutes = cfg.model.time.T_minutes
    T = cfg.model.time.T
    if T is None:
        if T_minutes is None:
            return None
        T = max(1, int(T_minutes) // max(1, int(slot_resolution)))
    T = int(T)

    scen = _scenario_containers(cfg)
    if scen:
        n = len(scen)
        wts = cfg.data.scenario_weights
        if not isinstance(wts, list) or len(wts) != n:
            wts = [1.0 / float(n)] * n
        wts = [float(w) for w in wts]
        total_w = sum(wts)
        if total_w <= 0.0:
            return None
        # Normalise. The derivation above needs the weights to sum to 1; the
        # subproblem's averaged cut normalises the same way, so an unnormalised
        # list would put the anchor and the cuts on different scales.
        wts = [w / total_w for w in wts]

        R_out = [0.0] * T
        R_ret = [0.0] * T
        # Also kept per scenario, for the scenario-direction anchor (S4). There the
        # proxy is theta_out_s[s], bounded by scenario s's OWN unserved cost:
        #
        #   theta_out_s[s] >= p * ( R^s_out_cum[j] - S * Y_out_cum[j+W] )
        #
        # which is valid for the same reason the aggregated row is, and strictly
        # tighter -- bounding a weighted sum by the mean's implied cost is weaker than
        # bounding each term by its own, by Jensen. The coarser shapes still read the
        # mean; both live in the same payload so the master picks by shape rather than
        # the caller having to know which it will build.
        R_out_by_scenario: list[list[float]] = []
        R_ret_by_scenario: list[list[float]] = []
        for w, container in zip(wts, scen):
            r_o, r_r = _demand_vectors(container, T, int(slot_resolution))
            R_out_by_scenario.append([float(r_o[t]) for t in range(T)])
            R_ret_by_scenario.append([float(r_r[t]) for t in range(T)])
            for t in range(T):
                R_out[t] += w * float(r_o[t])
                R_ret[t] += w * float(r_r[t])
    else:
        container: object | None = None
        if cfg.data.R_out is not None or cfg.data.R_ret is not None:
            container = {"R_out": cfg.data.R_out or [], "R_ret": cfg.data.R_ret or []}
        elif cfg.data.demand_file:
            container = Path(str(cfg.data.demand_file))
        if container is None:
            return None
        R_out, R_ret = _demand_vectors(container, T, int(slot_resolution))

    Wmax_slots = cfg.subproblem.Wmax_slots
    if Wmax_slots is None:
        if cfg.subproblem.Wmax_minutes is None:
            return None
        from .problem.subproblem_impl import wmax_minutes_to_slots

        Wmax_slots = wmax_minutes_to_slots(
            float(cfg.subproblem.Wmax_minutes), int(slot_resolution)
        )
    payload = {
        "R_out": [float(x) for x in R_out],
        "R_ret": [float(x) for x in R_ret],
        "p": float(cfg.subproblem.p),
        "S": float(cfg.subproblem.S),
        "W_slots": int(Wmax_slots),
        "num_scenarios": int(len(scen)) if scen else 1,
    }
    if scen:
        payload["R_out_by_scenario"] = R_out_by_scenario
        payload["R_ret_by_scenario"] = R_ret_by_scenario
    return payload


def _energy_params_for_resolution(cfg, slot_resolution: int) -> dict[str, float | int]:
    names = {
        "slot_resolution": slot_resolution,
        "T_minutes": cfg.model.time.T_minutes,
        "T": cfg.model.time.T,
        "trip_duration_minutes": cfg.model.time.trip_duration_minutes,
        "trip_duration": cfg.model.time.trip_duration,
        "trip_slots": cfg.model.time.trip_slots,
    }
    return resolve_energy_params(cfg.model.energy, names)


def _prepare_params(cfg, overrides: dict | None) -> tuple[dict, dict]:
    mp: dict[str, float | int | str | list | bool] = {}
    sp: dict[str, float | int | str | list | bool] = {}

    time = cfg.model.time
    fleet = cfg.model.fleet
    costs = cfg.model.costs

    _set_if_not_none(mp, "T_minutes", time.T_minutes)
    _set_if_not_none(mp, "T", time.T)
    mp["slot_resolution"] = int(time.slot_resolution)
    _set_if_not_none(mp, "trip_duration_minutes", time.trip_duration_minutes)
    _set_if_not_none(mp, "trip_duration", time.trip_duration)
    _set_if_not_none(mp, "trip_slots", time.trip_slots)

    mp["Q"] = int(fleet.Q)
    _set_if_not_none(mp, "binit", fleet.binit)
    _set_if_not_none(mp, "initial_actions", fleet.initial_actions)

    mp.update(_energy_params_for_resolution(cfg, int(time.slot_resolution)))

    _set_if_not_none(mp, "start_cost_epsilon", costs.start_cost_epsilon)
    _set_if_not_none(mp, "concurrency_penalty", costs.concurrency_penalty)

    mp["use_fifo_symmetry"] = bool(cfg.master.use_fifo_symmetry)
    mp["symmetry_breaking"] = bool(cfg.master.symmetry_breaking)
    mp["eps_bin"] = float(cfg.tolerances.eps_bin)
    mp["eps_cut"] = float(cfg.tolerances.eps_cut)
    mp["use_mip_start"] = bool(cfg.master.use_mip_start)
    # These are the starting values for the master's per-iteration controls. The
    # Benders loop's gap-tied schedule overwrites both every iteration, bounded by
    # the same two config values (see BendersSolver: mp_gap_max, mp_tl_cap).
    if cfg.master.per_iteration_time_limit_s is not None:
        mp["solve_time_limit_s"] = int(cfg.master.per_iteration_time_limit_s)
    if cfg.master.per_iteration_mipgap is not None:
        mp["mipgap"] = float(cfg.master.per_iteration_mipgap)
    if cfg.master.cplex_options:
        mp["cplex_options"] = dict(cfg.master.cplex_options)
    if cfg.master.solver_backend:
        mp["solver_backend"] = str(cfg.master.solver_backend)
    bnc = cfg.master.branch_and_cut
    mp["bnc_enabled"] = bool(bnc.enabled)
    mp["bnc_lazy_cuts"] = bool(bnc.lazy_cuts)
    mp["bnc_user_cuts"] = bool(bnc.user_cuts)
    mp["bnc_callback_lp_solver"] = str(bnc.callback_lp_solver)
    mp["bnc_seed_from_lp_phase"] = bool(bnc.seed_from_lp_phase)
    mp["aggregate_cuts_by_tau"] = bool(cfg.master.aggregate_cuts_by_tau)
    mp["cut_coeff_threshold"] = float(cfg.master.cut_coeff_threshold)
    mp["theta_per_scenario"] = bool(cfg.master.theta_per_scenario)
    # Resolve the direction split so the pre-S4 behaviour is the default in BOTH
    # branches, and the new scenario-direction shape is opted into rather than arriving
    # as a side effect.
    #
    # Before S4 the master computed `disagg_dir = False if theta_per_scenario else True`,
    # i.e. the direction split was forced OFF whenever per-scenario thetas were on --
    # which is exactly why the formulation's recommended shape (12) was inexpressible.
    # Leaving `theta_by_direction` unset reproduces that, so no existing config changes
    # shape on this commit. Setting it true alongside `theta_per_scenario: true` is what
    # asks for the 2*|Omega| shape.
    if cfg.master.theta_by_direction is None:
        mp["disaggregate_theta_by_direction"] = not bool(cfg.master.theta_per_scenario)
    else:
        mp["disaggregate_theta_by_direction"] = bool(cfg.master.theta_by_direction)
    mp["write_lp_after_cut"] = bool(cfg.master.write_lp_after_cut)
    mp["window_trip_caps"] = bool(cfg.master.window_trip_caps)
    mp["charge_before_idle"] = bool(cfg.master.charge_before_idle)
    # Carried on the master params so the manifest can record which side of the
    # A/B produced a number. The master itself reads only `lp_relaxation`, which
    # the Benders loop sets per iteration.
    mp["lp_phase"] = bool(cfg.master.lp_phase)
    mp["lp_phase_max_iters"] = int(cfg.master.lp_phase_max_iters)
    mp["lp_phase_stall_iters"] = int(cfg.master.lp_phase_stall_iters)
    mp["lp_phase_min_rel_improve"] = float(cfg.master.lp_phase_min_rel_improve)
    mp["recourse_lower_bound"] = bool(cfg.master.recourse_lower_bound)
    if cfg.master.recourse_lower_bound:
        _rlb = _recourse_bound_data(cfg, int(time.slot_resolution))
        if _rlb is None:
            # Asking for the anchor and silently not getting it is the D19/D22
            # defect: the run looks configured and behaves as if it were not.
            # Multi-scenario used to land here on purpose; it no longer does.
            raise ValueError(
                "master.recourse_lower_bound is true but the bound data cannot be "
                "built. It needs a demand source (data.demand_file, data.R_out/"
                "R_ret, data.scenarios or data.scenario_files), a horizon "
                "(model.time.T or T_minutes) and a waiting window "
                "(subproblem.Wmax_slots or Wmax_minutes). Set it to false to run "
                "without the anchor."
            )
        mp["recourse_bound_data"] = _rlb

    mp["solver"] = cfg.solver.master_solver
    mp["solver_tee"] = bool(cfg.solver.solver_tee)
    mp["log_level"] = str(cfg.run.log_level)
    # Off by default (M5). Writing a symbolic LP plus a solver log on every master
    # solve is a debugging aid, not something a normal run should produce -- a
    # 10-iteration run left 20 files behind. Previously this was derived from
    # log_level != "REPORT", i.e. on unless you set a log level that reads as if it
    # would enable reports rather than disable them.
    mp["emit_reports"] = bool(cfg.run.emit_reports)

    sp["lp_solver"] = cfg.solver.subproblem_solver
    sp["multi_cuts_by_scenario"] = bool(cfg.subproblem.multi_cuts_by_scenario)
    # The resolved generator (S1b). config.py collapses cut_mode and the legacy
    # boolean pair into exactly one of mw / dual / finite_difference, so the
    # dispatch reads one value instead of guessing a precedence between two flags.
    sp["cut_mode"] = str(cfg.subproblem.cut_mode)
    sp["acknowledge_no_lower_bound"] = bool(
        cfg.subproblem.acknowledge_no_lower_bound
    )
    sp["use_magnanti_wong"] = bool(cfg.subproblem.use_magnanti_wong)
    sp["mw_core_alpha"] = float(cfg.subproblem.mw_core_alpha)
    sp["mw_core_eps"] = float(getattr(cfg.subproblem, "mw_core_eps", 1e-3))
    sp["use_dual_slopes"] = bool(cfg.subproblem.use_dual_slopes)
    sp["S"] = cfg.subproblem.S
    _set_if_not_none(sp, "Wmax_minutes", cfg.subproblem.Wmax_minutes)
    _set_if_not_none(sp, "Wmax_slots", cfg.subproblem.Wmax_slots)
    sp["p"] = cfg.subproblem.p
    sp["recourse_resolution"] = cfg.subproblem.recourse_resolution
    sp["departure_policy"] = cfg.subproblem.departure_policy
    sp["degenerate_cut_probe_top_k"] = int(cfg.subproblem.degenerate_cut_probe_top_k)
    _set_if_not_none(
        sp,
        "degenerate_cut_probe_top_k_out",
        cfg.subproblem.degenerate_cut_probe_top_k_out,
    )
    _set_if_not_none(
        sp,
        "degenerate_cut_probe_top_k_ret",
        cfg.subproblem.degenerate_cut_probe_top_k_ret,
    )
    sp["degenerate_cut_zero_tol"] = float(cfg.subproblem.degenerate_cut_zero_tol)
    # tolerances
    sp["eps_cut"] = float(cfg.tolerances.eps_cut)

    _set_if_not_none(sp, "demand_file", cfg.data.demand_file)
    _set_if_not_none(
        sp,
        "scenario_files",
        cfg.data.scenario_files if cfg.data.scenario_files else None,
    )
    _set_if_not_none(sp, "scenario_weights", cfg.data.scenario_weights)
    _set_if_not_none(sp, "R_out", cfg.data.R_out)
    _set_if_not_none(sp, "R_ret", cfg.data.R_ret)
    _set_if_not_none(sp, "scenarios", cfg.data.scenarios)

    sp["slot_resolution"] = int(time.slot_resolution)
    sp["T_minutes"] = int(time.T_minutes) if time.T_minutes is not None else None
    _set_if_not_none(sp, "T", time.T)
    _set_if_not_none(sp, "trip_duration_minutes", time.trip_duration_minutes)
    _set_if_not_none(sp, "trip_duration", time.trip_duration)
    _set_if_not_none(sp, "trip_slots", time.trip_slots)
    sp["Q"] = int(fleet.Q)
    _set_if_not_none(sp, "binit", fleet.binit)
    _set_if_not_none(sp, "initial_actions", fleet.initial_actions)
    sp.update(_energy_params_for_resolution(cfg, int(time.slot_resolution)))
    sp["eps_feas"] = float(cfg.tolerances.eps_feas)
    sp["log_level"] = str(cfg.run.log_level)

    if overrides:
        mp.update(
            (overrides.get("master_params") or {})
            if isinstance(overrides, dict)
            else {}
        )
        sp.update(
            (overrides.get("subproblem_params") or {})
            if isinstance(overrides, dict)
            else {}
        )

    # Propagate slot_resolution from master to subproblem if not explicitly set
    if "slot_resolution" not in sp and "slot_resolution" in mp:
        sp["slot_resolution"] = mp["slot_resolution"]

    # If multi-cuts by scenario is enabled and scenarios present, propagate scenario count/weights to master
    try:
        multi_cuts = bool(sp.get("multi_cuts_by_scenario", False))
    except Exception:
        multi_cuts = False
    scen_list = []
    try:
        if isinstance(sp.get("scenarios"), list) and sp.get("scenarios"):
            scen_list = list(sp.get("scenarios"))
        elif isinstance(sp.get("scenario_files"), list) and sp.get("scenario_files"):
            scen_list = list(sp.get("scenario_files"))
    except Exception:
        scen_list = []
    if multi_cuts and scen_list:
        S = len(scen_list)
        mp.setdefault("theta_per_scenario", True)
        mp["num_scenarios"] = S
        # Pass weights if provided; else default uniform weights summing to 1
        wts = sp.get("scenario_weights")
        if not isinstance(wts, list) or len(wts) != S:
            wts = [1.0 / float(S) for _ in range(S)]
        mp["scenario_weights"] = wts

    return mp, sp


def _build_solver(cfg, mp: dict, sp: dict):
    ProblemMaster, ProblemSubproblem = import_problem_impl()
    master = ProblemMaster(mp)
    sub = ProblemSubproblem(sp)
    solver = BendersSolver(master, sub, cfg)
    return solver, master, sub


def _print_cfg(cfg, mp: dict, sp: dict) -> None:
    print("Run configuration:")
    print(
        f"  solver: iterations={cfg.solver.max_iterations} tol={cfg.solver.tolerance} "
        f"total_time_limit_s={cfg.solver.total_time_limit_s} seed={cfg.run.seed}"
    )
    T_minutes = mp.get("T_minutes")
    slot_res = mp.get("slot_resolution", 1)
    trip_dur_min = mp.get("trip_duration_minutes", mp.get("trip_duration"))
    if T_minutes is not None:
        try:
            T_slots = int(int(T_minutes) // int(slot_res or 1))
        except Exception:
            T_slots = mp.get("T", "-")
    else:
        T_slots = mp.get("T", "-")
    trip_slots = mp.get("trip_slots")
    print(
        "  master: solver=%s Q=%s T_minutes=%s slot_res=%s (slots=%s) trip_dur_min=%s Emax=%s L=%s eps=%s conc_pen=%s delta_chg=%s"
        % (
            mp.get("solver", "-"),
            mp.get("Q", "-"),
            T_minutes if T_minutes is not None else mp.get("T", "-"),
            slot_res,
            T_slots,
            trip_dur_min if trip_dur_min is not None else trip_slots,
            mp.get("Emax", "-"),
            mp.get("L", "-"),
            mp.get("start_cost_epsilon", "-"),
            mp.get("concurrency_penalty", "-"),
            mp.get("delta_chg", "-"),
        )
    )
    try:
        _T = int(T_slots) if isinstance(T_slots, int) else int(mp.get("T"))
        import math

        if trip_dur_min is not None:
            _res = int(slot_res or 1)
            _ts = int(math.ceil(float(trip_dur_min) / max(1, _res)))
        else:
            _ts = int(mp.get("trip_slots"))
        if _ts >= _T:
            print(
                "  NOTE: trip duration (in slots) >= horizon; starts limited to t=0 and may prevent serving demand."
            )
    except Exception:
        pass
    print(
        "  subproblem: solver=%s S=%s Wmax=%s p=%s (slot_res=%s)"
        % (
            sp.get("lp_solver", "-"),
            sp.get("S", "-"),
            sp.get("Wmax_minutes", sp.get("Wmax_slots", sp.get("Wmax", "-"))),
            sp.get("p", "-"),
            sp.get("slot_resolution", mp.get("slot_resolution", "-")),
        )
    )
    if "demand_file" in sp:
        print(f"  demand_file: {sp.get('demand_file')}")
    if "scenario_files" in sp:
        print(f"  scenario_files: {sp.get('scenario_files')}")
    if "R_out" in sp:
        print(f"  R_out: {sp.get('R_out')} (inline)")
    if "R_ret" in sp:
        print(f"  R_ret: {sp.get('R_ret')} (inline)")


def _maybe_print_summary(result: BendersRunResult, sp: dict) -> None:
    try:
        if result.pax_served is not None and result.pax_total is not None:
            print(f"Pax served: {result.pax_served:.0f}/{result.pax_total:.0f}")
        # Use subproblem diagnostics for consistent decomposition
        if result.subproblem_obj is not None:
            wait_slots = float(result.sp_wait_cost_slots or 0.0)
            fill_eps = float(result.sp_fill_eps_cost or 0.0)
            pen_cost = float(result.sp_penalty_cost or 0.0)
            pen_pax = float(result.sp_penalty_pax or 0.0)
            total_dem = float(result.sp_total_demand or 0.0)
            slot_res = int(result.sp_slot_resolution or 1)
            sum_components = wait_slots + fill_eps + pen_cost
            if abs(sum_components - float(result.subproblem_obj)) > 1e-5:
                print(
                    "[DIAG] Subproblem objective mismatch: obj=%.6g wait=%.6g fill_eps=%.6g penalty=%.6g sum=%.6g"
                    % (
                        float(result.subproblem_obj),
                        wait_slots,
                        fill_eps,
                        pen_cost,
                        sum_components,
                    )
                )
            if wait_slots < -1e-9:
                print(f"[DIAG] Negative waiting cost detected: {wait_slots:.6g}")
            if pen_cost < -1e-9:
                print(f"[DIAG] Negative penalty cost detected: {pen_cost:.6g}")
            wait_per_pax_min = None
            if result.pax_served and result.pax_served > 0:
                wait_per_pax_min = (wait_slots * float(slot_res)) / float(
                    result.pax_served
                )
            print(
                "Subproblem (last): obj=%.6g wait_slots=%.6g fill_eps=%.6g penalty_cost=%.6g penalty_pax=%.6g total_demand=%.6g"
                % (
                    float(result.subproblem_obj),
                    wait_slots,
                    fill_eps,
                    pen_cost,
                    pen_pax,
                    total_dem,
                )
            )
            if wait_per_pax_min is not None:
                print(f"Avg wait (min): {wait_per_pax_min:.6g}")
        if result.best_upper_bound is not None:
            print(f"UB_total (best): {float(result.best_upper_bound):.6g}")
    except Exception:
        pass
    print(
        f"\nResult: status={result.status} iterations={result.iterations} "
        f"best_lb={result.best_lower_bound} best_ub={result.best_upper_bound}"
    )
    # A run whose master solves stopped on the clock is not bit-reproducible, and
    # must not be quoted as if it were. Measured: a binding per-iteration limit moved
    # the LB 8% between two runs of one config; a non-binding one reproduced exactly.
    truncated = getattr(result, "clock_truncated_master_solves", None)
    if truncated:
        print(
            f"NOT REPRODUCIBLE: {truncated} master solve(s) stopped on the clock, not "
            "the gap. Bounds from this run vary with machine load. For a comparable "
            "number, raise master.per_iteration_time_limit_s and solver.total_time_limit_s "
            "until the master always terminates on the gap (configs/baseline_d9.yaml)."
        )


def _map_candidate_to_warm_start(
    cand: dict[str, float],
    res_old: int,
    res_new: int,
    mp: dict,
) -> dict[tuple[str, int, int], float]:
    # Compute T_new and trip_slots at new resolution to avoid proposing invalid starts
    import math

    T_minutes = mp.get("T_minutes")
    if T_minutes is not None:
        T_new = int(int(T_minutes) // max(1, int(res_new)))
    else:
        T_new = int(mp.get("T", 0))
    trip_min = mp.get("trip_duration_minutes", mp.get("trip_duration"))
    if trip_min is not None:
        trip_slots_new = int(math.ceil(float(trip_min) / max(1, int(res_new))))
    else:
        trip_slots_new = int(mp.get("trip_slots", 0))

    def _map_t(t_old: int) -> int:
        # Map by minutes, rounding to nearest slot at new resolution
        minutes = int(t_old) * int(res_old)
        return int(round(minutes / float(max(1, int(res_new)))))

    starts: dict[tuple[str, int, int], float] = {}
    for k, v in (cand or {}).items():
        if not isinstance(k, str) or float(v or 0.0) < 0.5:
            continue
        if k.startswith("yOUT[") or k.startswith("yRET["):
            inside = k[k.find("[") + 1 : k.find("]")]
            q_str, t_str = inside.split(",")
            try:
                q = int(q_str.strip())
                t_old = int(t_str.strip())
            except Exception:
                continue
            t_new = _map_t(t_old)
            if not (0 <= t_new < T_new):
                continue
            # Respect last-start feasibility windows at new resolution
            if t_new > (T_new - trip_slots_new - 1):
                continue
            typ = "yOUT" if k.startswith("yOUT[") else "yRET"
            starts[(typ, q, t_new)] = 1.0
    return starts


def _run_single(
    cfg,
    mp: dict,
    sp: dict,
    emit_cli_output: bool,
    warm_start: dict | None = None,
    emit_summary: bool = True,
):
    solver, master, sub = _build_solver(cfg, mp, sp)
    if warm_start:
        try:
            master.set_warm_start(warm_start)
        except Exception:
            pass
    if emit_cli_output:
        _print_cfg(cfg, mp, sp)
    _t_seed_start = _monotonic()
    result = solver.run()
    if cfg.master.branch_and_cut.enabled:
        result = _run_branch_and_cut(
            cfg,
            master,
            sub,
            result,
            emit_cli_output,
            seeding_elapsed_s=_monotonic() - _t_seed_start,
        )
    if emit_cli_output and emit_summary:
        _maybe_print_summary(result, sp)
    return result, master


def _run_branch_and_cut(
    cfg, master, sub, seed_result, emit_cli_output: bool, *, seeding_elapsed_s: float
):
    """Solve the seeded master once inside a single tree, and report it as such.

    `solver.run()` above stopped at the end of the LP phase -- the config refuses
    any other iteration budget when seeding is on -- so the master now holds
    exactly the cut set that run 2 produced, and the tree starts from run 2's
    root. That is the comparison D45 asks for: same cuts, one tree instead of a
    loop that rebuilds one per iteration.

    The seeding phase is pure LP and claims no upper bound (a fractional schedule
    cannot be exhibited), so the seed result's bound is a lower bound only. What
    the tree returns replaces it wholesale rather than being merged with it: two
    bounds from two different solves, silently maxed, is how a number stops being
    attributable to a run.
    """
    from dataclasses import replace

    bnc = cfg.master.branch_and_cut
    if not (bnc.lazy_cuts or bnc.control_no_callback):
        # Reachable only via user_cuts, which the schema refuses today. Left as a
        # raise rather than an `if` that quietly solves an ordinary master.
        raise NotImplementedError(
            "branch_and_cut with lazy_cuts=false has no generator wired yet"
        )

    # solver.total_time_limit_s is documented as the budget for the WHOLE run.
    # Passing it to the tree unchanged made the first measured run take
    # 150 s of seeding + 600 s of tree against a 600 s budget -- a 25% overshoot
    # of the one number a reader uses to decide what a run costs. The tree gets
    # what is left, and is told so.
    remaining = float(cfg.solver.total_time_limit_s) - float(seeding_elapsed_s)
    if remaining <= 0.0:
        raise RuntimeError(
            f"the seeding LP phase used {seeding_elapsed_s:.0f} s of a "
            f"{cfg.solver.total_time_limit_s} s budget, leaving nothing for the "
            "tree. Raise solver.total_time_limit_s rather than reporting a tree "
            "that never ran."
        )
    label = (
        "CONTROL, no callback registered"
        if bnc.control_no_callback
        else "one tree, lazy cuts at incumbents"
    )
    if emit_cli_output:
        print(
            f"\n=== Branch-and-cut ({label}) over {master.cuts_count()} seeded cuts, "
            f"{remaining:.0f} s left of a {cfg.solver.total_time_limit_s} s budget ==="
        )

    bnc_result, stats = master.solve_branch_and_cut(
        sub.evaluate,
        time_limit_s=remaining,
        mipgap=cfg.master.per_iteration_mipgap,
        register_callback=not bnc.control_no_callback,
    )

    # The master's own objective is NOT an upper bound, and reporting it as one
    # was wrong. It is first_stage + theta, and theta is bounded only by the cuts
    # the master holds. With the lazy callback the final incumbent has been
    # priced -- an accepted incumbent is one whose cut is satisfied, so theta is
    # at or above its true recourse -- but in the no-callback control nothing
    # prices it, and the first control run duly reported an "upper bound" of
    # 1288.86 that is a relaxation value and can sit BELOW the true optimum. A
    # 14% gap read off that pair would have been fiction.
    #
    # So the schedule is priced here, once, for both paths. That is the same
    # thing the loop does and the same thing that makes an upper bound mean
    # something: an exhibited feasible schedule with its recourse evaluated.
    upper_bound = None
    priced_recourse = None
    cand = bnc_result.candidate
    if cand is not None:
        sres = sub.evaluate(cand)
        if sres.upper_bound is not None:
            priced_recourse = float(sres.upper_bound)
            upper_bound = float(master.first_stage_cost(cand)) + priced_recourse

    if emit_cli_output:
        if upper_bound is None:
            print(
                "[BNC] no upper bound claimed: the tree returned no schedule the "
                "subproblem could price."
            )
        else:
            print(
                f"[BNC] UB from the exhibited schedule: "
                f"first_stage + recourse = {upper_bound:.6g} "
                f"(recourse {priced_recourse:.6g}); "
                f"master objective was {bnc_result.objective:.6g}, which is "
                "first_stage + theta and is NOT an upper bound"
            )
        print(
            f"[BNC] callback invocations={stats.invocations} "
            f"cuts_injected={stats.cuts_injected} "
            f"incumbents_accepted={stats.incumbents_accepted} "
            f"validity={stats.validity_counts}"
        )
        print(
            f"[BNC] status={bnc_result.status.value} "
            f"LB={bnc_result.lower_bound} UB={upper_bound}"
        )
        # There is no iteration count in a tree. Saying so is cheaper than having
        # a reader assume the number below came from one.
        print(
            "[BNC] NOT REPRODUCIBLE: the tree stops on the clock, and a lazy "
            "callback makes CPLEX's node order thread-dependent (D26)."
        )

    return replace(
        seed_result,
        status=bnc_result.status,
        best_lower_bound=bnc_result.lower_bound,
        best_upper_bound=upper_bound,
        subproblem_obj=priced_recourse,
        # The loop's iteration counter does not describe a tree. Reporting the
        # seeding phase's count here would read as "N Benders iterations", which
        # is not what produced these bounds.
        iterations=0,
        clock_truncated_master_solves=1,
    )


def _emit_manifest(cfg, config_path, result, master, emit_cli_output: bool) -> None:
    """Write the run manifest (spec §0.4). Never fails a run.

    A manifest that cannot be written must not take a completed solve with it,
    but the failure has to be visible -- silently skipping provenance is how the
    invalid-lower-bound problem became hard to scope after the fact.
    """
    from .manifest import build_manifest, write_manifest

    try:
        diag = {
            "cut_generation_mode": getattr(result, "cut_generation_mode", None),
            "cut_valid_lower_bound": getattr(result, "cut_valid_lower_bound", None),
        }
        repo_root = Path(__file__).resolve().parents[2]
        out_dir = (
            Path(cfg.run.report_dir)
            if cfg.run.report_dir
            else (repo_root / "manifests")
        )
        manifest = build_manifest(
            cfg, Path(config_path) if config_path else None, result, repo_root, diag
        )
        path = write_manifest(manifest, out_dir, cfg.run.name)
        if emit_cli_output:
            print(f"\nManifest: {path}")
    except Exception as exc:  # noqa: BLE001 - provenance must not break a solve
        print(f"\n[WARN] could not write run manifest: {exc!r}")


def run(
    config_path: str | Path | None = None, overrides: dict | None = None
) -> BendersRunResult:
    """Run the Benders solver with a single canonical execution path.

    Parameters are taken from configs/default.yaml by default.
    """
    cfg = load_config(config_path)
    _apply_run_overrides(cfg, overrides)
    setup_logging(cfg.run.log_level)
    report_mode = str(cfg.run.log_level).upper() == "REPORT"

    mp_base, sp_base = _prepare_params(cfg, overrides)
    emit_cli_output = bool(overrides.get("emit_cli_output")) if overrides else False

    multi_res = overrides.get("multi_res") if isinstance(overrides, dict) else None
    if multi_res:
        seq = [int(x) for x in multi_res if str(x).strip()]
        if not seq:
            raise ValueError("No valid resolutions provided for multi-res run.")
        prev_cand: dict[str, float] | None = None
        prev_res: int | None = None
        last_result: BendersRunResult | None = None
        for i, res in enumerate(seq, start=1):
            mp = dict(mp_base)
            sp = dict(sp_base)
            try:
                prev_slot_res = int(mp.get("slot_resolution", res))
            except Exception:
                prev_slot_res = int(res)
            mp["slot_resolution"] = int(res)
            sp["slot_resolution"] = int(res)
            mp.update(_energy_params_for_resolution(cfg, int(res)))
            # The demand vectors and W_slots are resolution-dependent, so they must be
            # rebuilt here rather than inherited from mp_base's base resolution.
            if mp.get("recourse_lower_bound"):
                _rlb = _recourse_bound_data(cfg, int(res))
                if _rlb is not None:
                    mp["recourse_bound_data"] = _rlb
                else:
                    mp.pop("recourse_bound_data", None)
            if cfg.model.energy.delta_chg is not None and not isinstance(
                cfg.model.energy.delta_chg, str
            ):
                try:
                    if "delta_chg" in mp:
                        mp["delta_chg"] = float(mp["delta_chg"]) * (
                            float(res) / max(1.0, float(prev_slot_res))
                        )
                except Exception:
                    pass
            if emit_cli_output:
                if not report_mode:
                    print(
                        f"\n=== Multi-res stage {i}/{len(seq)}: slot_resolution={res} ==="
                    )
            warm_start = None
            if prev_cand is not None and prev_res is not None:
                warm_start = _map_candidate_to_warm_start(
                    prev_cand, prev_res, int(res), mp
                )
                if emit_cli_output and warm_start:
                    if not report_mode:
                        print(f"Applied warm start with {len(warm_start)} start(s).")
            result, master = _run_single(
                cfg,
                mp,
                sp,
                emit_cli_output,
                warm_start=warm_start,
                emit_summary=False,
            )
            last_result = result
            if emit_cli_output:
                if not report_mode:
                    print(
                        f"Stage {i} result: status={result.status} iters={result.iterations} "
                        f"LB={result.best_lower_bound} UB={result.best_upper_bound}"
                    )
            try:
                prev_cand = getattr(master, "_collect_candidate")()
                prev_res = int(res)
            except Exception:
                prev_cand = None
                prev_res = None
        if last_result is None:
            raise ValueError("Multi-res run produced no results.")
        return last_result

    result, _master = _run_single(cfg, mp_base, sp_base, emit_cli_output)
    _emit_manifest(cfg, config_path, result, _master, emit_cli_output)
    return result


__all__ = ["DEFAULT_CONFIG_PATH", "import_problem_impl", "run"]
