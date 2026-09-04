from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import ast
import operator as _op
import warnings

try:
    import yaml as _yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _yaml = None


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


# ---- Schema sections ----


@dataclass(slots=True)
class SchemaSection:
    name: str
    version: int


@dataclass(slots=True)
class RunSection:
    name: str | None = None
    log_level: str = "INFO"
    log_file: str | None = None
    report_dir: str | None = None
    seed: int | None = None
    # Write a symbolic LP and a solver log for every master solve. Debugging aid,
    # off by default: it produces two files per iteration (M5).
    emit_reports: bool = False


@dataclass(slots=True)
class DataSection:
    demand_file: str | None = None
    scenario_files: list[str] = field(default_factory=list)
    scenario_weights: list[float] | None = None
    R_out: list[float] | None = None
    R_ret: list[float] | None = None
    scenarios: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class TimeSection:
    T_minutes: int | None = None
    T: int | None = None
    slot_resolution: int = 1
    trip_duration_minutes: int | None = None
    trip_duration: int | None = None
    trip_slots: int | None = None


@dataclass(slots=True)
class FleetSection:
    Q: int
    binit: list[float] | None = None
    initial_actions: list[str] | None = None


@dataclass(slots=True)
class EnergySection:
    Emax: float
    L: float
    delta_chg: float | int | str | None = None
    # B2 (audit 1.8). How many vehicles may draw from the depot's chargers in one
    # slot. `None` means Q -- every vehicle can charge at once, which is the
    # assumption every archived result was produced under, and at which the row is
    # implied by c in [0,1] so those results reproduce exactly. State a number to
    # make the site's real charger count part of the instance rather than an
    # accident of the formulation.
    K_chg: int | None = None
    # Whether a charger is held for a whole slot (True) or is preemptible within it
    # (False, the divisible default). These are different physical claims and give
    # different schedules; the manifest records which one a run used.
    charger_occupancy_binary: bool = False


@dataclass(slots=True)
class CostSection:
    start_cost_epsilon: float = 0.0
    concurrency_penalty: float = 0.0
    # B7.2/B7.3 (audit item 1.3). "weighted_sum" is the DEFAULT, so every archived
    # result reproduces and epsilon/kappa keep the meaning they have always had --
    # policy weights, not tie-breakers; the audit is explicit that kappa=0.25 is large
    # enough to select schedules.
    #
    # "lexicographic" states the other policy: maximise served demand first, then
    # minimise waiting, then the epsilon and kappa terms. Under the weighted sum at
    # p_min=56 and delta=30 a two-slot wait is strictly worse than a rejection, so
    # some admissible waits are dominated by leaving the passenger behind. That is a
    # real policy choice and it must be selectable rather than implied by arithmetic.
    objective_mode: str = "weighted_sum"


@dataclass(slots=True)
class ModelSection:
    time: TimeSection
    fleet: FleetSection
    energy: EnergySection
    costs: CostSection


@dataclass(slots=True)
class BranchAndCutSection:
    """Branch-and-cut: cuts injected inside one master tree instead of a loop (D44).

    Every default here is off. The previous implementation was deleted in 2026-02
    without a decision record, and it was judged against the pre-D30 subproblem
    whose constraint set moved with `y` -- so its verdict is void on the same
    grounds as every pre-D30 lower bound, and nothing about it is inherited as a
    default.

    The one contract that does NOT carry over from the Benders loop: **a lazy cut
    cannot be un-added.** D39 is fail-closed at the level of the reported bound --
    an INVALID or UNKNOWN cut is added and the lower bound is dropped afterwards,
    which works only where the loop owns the bound. Inside a branch-and-bound tree
    a cut that excludes the true optimum cannot be retracted and nothing recovers
    it. So the rule inverts: only CutValidity.VALID may be injected, and anything
    else aborts the solve. It must never merely skip the cut -- in a callback,
    skipping is how you tell CPLEX the incumbent is acceptable.
    """

    # Off by default. Turning this on changes the master's solve path, not a
    # heuristic: the backend must be cplex_persistent, which is the only Pyomo
    # CPLEX interface here that exposes register_callback.
    enabled: bool = False
    # Solver for the subproblem LP solved *inside* the callback. The deleted
    # implementation carried this key to "avoid nested CPLEX"; nesting is measured
    # to work on this installation, so the key stays as a lever rather than a
    # workaround.
    callback_lp_solver: str = "cplex_direct"
    # Lazy constraints at integer incumbents. This reproduces the current loop's
    # semantics inside one tree, so a run is comparable against lp150_then_mip8.
    lazy_cuts: bool = True
    # User cuts at fractional nodes. The part with no equivalent in the loop, and
    # deliberately a separate step: enabling both at once makes an unexpected
    # result uninterpretable.
    user_cuts: bool = False
    # Keep the LP phase as a warm-up that seeds the tree. The 150 LP cuts are what
    # makes the root strong (D40/D45), so leaving this off asks a different
    # question -- whether the tree can build its own root -- and the two must not
    # be confused for one another.
    seed_from_lp_phase: bool = True
    # The control for the branch-and-cut measurement: same seeded master, same
    # persistent solver, same budget, same options -- and no callback registered.
    # It exists because the first comparison available (a 102 s loop solve at
    # ~1080 against a 600 s tree at 1004.6) differed in TWO things at once, the
    # callback and the clock, and a two-variable comparison decides nothing.
    # Requires lazy_cuts and user_cuts off, so the file cannot claim to generate
    # cuts and not generate them.
    control_no_callback: bool = False


@dataclass(slots=True)
class MasterSection:
    use_fifo_symmetry: bool = False
    symmetry_breaking: bool = False
    use_mip_start: bool = False
    # Ceiling on ONE master MIP solve, per Benders iteration. Not the run budget:
    # that is solver.total_time_limit_s. The gap-tied schedule in the Benders loop
    # picks a per-iteration limit and this value caps it.
    per_iteration_time_limit_s: int | None = None
    # Ceiling on the master MIP gap, per Benders iteration. Not the convergence
    # criterion: that is solver.tolerance. The schedule tightens the gap on its own
    # as the Benders gap closes; this is the loosest it is allowed to be.
    per_iteration_mipgap: float | None = None
    cplex_options: dict[str, Any] = field(default_factory=dict)
    solver_backend: str = "cplex_direct"
    aggregate_cuts_by_tau: bool = True
    cut_coeff_threshold: float = 0.0
    theta_per_scenario: bool = False
    # Direction split of the recourse proxy. Was hardcoded True and read through
    # `self._p("disaggregate_theta_by_direction", ...)` from a key no config could set --
    # the inert-configuration pattern (AUDIT_v4 3.8), except inverted: not a knob that did
    # nothing, but a behaviour with no knob. Exposed because S4 makes the combination
    # (per-scenario AND per-direction) meaningful, and that combination must be OPTED
    # INTO rather than arriving as a side effect of a default.
    #
    # The four shapes and how to select them:
    #
    #   theta_per_scenario  theta_by_direction   shape          proxies
    #   false               false                single         1
    #   false               true   (default)     by_direction   2
    #   true                false                by_scenario    |Omega|
    #   true                true                 by_scen_dir    2*|Omega|   <-- S4
    #
    # Defaults reproduce the pre-S4 behaviour exactly: `theta_per_scenario: false` gave
    # the directional pair, and `true` gave the per-scenario set with the direction split
    # forced OFF. So `theta_by_direction` defaults to true, and a per-scenario config that
    # does not mention it keeps its old shape -- see the resolution in app.py.
    theta_by_direction: bool | None = None
    write_lp_after_cut: bool = False
    # Window trip caps from the single-vehicle decision diagram (D48, stage 1):
    #   sum_{tau in [t1,t2]} (Yout+Yret)[tau] <= Q * max_trips(t1,t2)
    # A valid inequality in Y alone. OFF by default: spec 2.9 (M1) records a sound,
    # implied inequality that made the master 2.7x slower and its bound worse, so
    # this is an opt-in whose effect is measured rather than a default.
    window_trip_caps: bool = False
    # Canonical ordering: charge before idling at the depot (M2). On by default.
    charge_before_idle: bool = True
    # Valid inequality anchoring theta to installed capacity per prefix of the
    # horizon. ON by default since D29: measured at equal iterations it improves
    # the lower bound 26-90% AND cuts master time 38-68% on the reproducible
    # cells. Unlike M1, it is not a trade-off.
    recourse_lower_bound: bool = True
    # LP phase: solve the master without integrality while the cut set is still
    # poor. Cuts from a fractional y are valid (the recourse value function is
    # convex in y), but its subproblem cost is NOT an upper bound, so the loop
    # claims none while the phase is on. Off by default; earn it by measurement.
    lp_phase: bool = False
    # Hard ceiling on LP-phase iterations.
    lp_phase_max_iters: int = 10
    # Consecutive iterations of sub-threshold LP objective improvement that end
    # the phase. 0 disables the stall test and leaves only the iteration ceiling.
    lp_phase_stall_iters: int = 3
    # Relative improvement in the LP master objective that counts as progress.
    lp_phase_min_rel_improve: float = 0.005
    branch_and_cut: BranchAndCutSection = field(default_factory=BranchAndCutSection)


@dataclass(slots=True)
class SubproblemSection:
    # B1 (audit 1.4). Which multi-scenario cut architecture a run uses. Two are
    # implemented and they are NOT interchangeable:
    #
    #   "aggregated"    one theta, one cut carrying pi_bar = sum_s w_s pi_hat_s and
    #                   a = sum_s w_s sum_t alpha_hat_{s,t} R_{s,t}. What the report
    #                   states, and the default.
    #   "disaggregated" theta_{s,d} and one cut per (scenario, direction). The
    #                   stronger formulation -- it never aggregates away a scenario's
    #                   own subgradient -- and the one B9 wants for the strength
    #                   comparison.
    #
    # Resolved from the two legacy booleans below when it is not stated, so every
    # existing config keeps its architecture. Stating it AND contradicting it with the
    # booleans is an error rather than a precedence rule: the two ways of asking for an
    # architecture disagreeing silently is exactly the shape of defect this key exists
    # to remove.
    cut_architecture: str | None = None
    multi_cuts_by_scenario: bool = True
    # Which generator builds the cut. One key with three values, replacing two booleans
    # that could not express three mutually exclusive modes (S1b).
    #
    #   mw                 Magnanti-Wong Pareto-optimal dual on the optimal face
    #   dual               plain capacity duals -- valid, not Pareto-optimal. The Level
    #                      A/B ablation baseline the formal formulation asks for
    #   finite_difference  perturbation estimates. NOT a certified lower bound
    #                      (handout 75/76); requires acknowledge_no_lower_bound
    #
    # The old form was `use_magnanti_wong` + `use_dual_slopes`, dispatched as
    # `if mw: ... elif dual: ... else: fdiff`. With both true -- which every shipped
    # config sets -- the `dual` branch was UNREACHABLE, so the ablation baseline could not
    # be run at all (AUDIT_v4 3.5). Two booleans cannot carry three exclusive states
    # without one of them becoming dead, which is what happened.
    #
    # `None` means "derive from the legacy booleans", so no existing config changes
    # behaviour. Setting both forms is refused rather than silently resolved.
    cut_mode: str | None = None
    # Required to run `finite_difference` (S7). That mode produces cuts with no
    # lower-bound guarantee, and the runtime already drops `best_lb` when it is used --
    # but only after the run has spent its budget. Refusing at load turns an hour of
    # wasted solve into an immediate error.
    acknowledge_no_lower_bound: bool = False
    use_magnanti_wong: bool = False
    mw_core_alpha: float = 0.3
    # B14 (audit item 2.8). What certification is attempted for the Magnanti-Wong
    # core point. "necessary_conditions" checks the conditions
    # signature.project_core_point enforces on a relaxation of the projected region;
    # "none" skips it. NEITHER establishes relative interiority of conv(Y) -- no
    # method here does -- so the cut is described as Magnanti-Wong-INSPIRED either
    # way. What this key changes is whether the run can say it looked.
    mw_core_point_certification: str = "necessary_conditions"
    mw_core_eps: float = 1e-3
    use_dual_slopes: bool = False
    S: float = 0.0
    Wmax_minutes: int | None = None
    Wmax_slots: int | None = None
    # `p` is ALWAYS in slot units by the time it reaches here -- it is the coefficient
    # the LP uses, and the waiting term it trades against is `(tau - t)` in slots
    # (D7/D8). `p_minutes` is the resolution-independent way to state the same policy
    # and is converted once, at load: p = p_minutes / slot_resolution.
    #
    # Why the second form exists (D50). Stating `p` directly makes the physical
    # trade-off move with the grid: p=50 at 30-minute slots means one unserved
    # passenger is worth 50 slots of waiting = 1500 passenger-minutes, while the same
    # p=50 at 15-minute slots means 750. The repository already contained both --
    # baseline_d9 at 1500 and the Fase 1 point at 750 -- so no objective from one
    # resolution was comparable with the other. `Wmax` never had this problem because
    # `Wmax_minutes` was always the stated form; `p` was the odd one out.
    p: float = 0.0
    # The input, kept for the manifest so a run can say which form it was given.
    # None means `p` was stated directly in slot units.
    p_minutes: float | None = None
    # Multi-resolution recourse (D51). "slot" is the model this repository has always
    # had. "minute" evaluates operations on a fixed minute grid while the first stage
    # stays on slots -- the architecture the CP research note proposes. The capacity
    # rows stay indexed by departure slot in both, so the cut the master receives is
    # the same object and every downstream check is unchanged.
    recourse_resolution: str = "slot"
    # Where inside its slot a departure is assumed to leave, in minute mode. "start" is
    # the only convention that prices what the schedule's own t+1 commitment actually
    # does (D76) -- see minute_pricer.py's `DeparturePolicy` comment. "midpoint"/"end"
    # remain available as explicit counterfactuals, not as competing estimates.
    departure_policy: str = "start"
    # B6 (audit 2.4). Whether a passenger may board a departure leaving in their own
    # arrival slot/minute. "forbid" is the tau >= t+1 rule the slot recourse has always
    # enforced and, under departure_policy="start", it reproduces that arc set exactly
    # at ANY resolution -- which is what makes a minute run and a slot run comparable.
    # The minute recourse used to hard-code the opposite; see
    # minute_pricer.SameSlotEligibility. Every table must state which one it used.
    same_slot_eligibility: str = "forbid"
    # F2 (docs/PROJECT_STATE_v6.md section 5, D76): a fixed offset grid O subset
    # [-delta, delta], chosen once at load, letting the minute recourse treat a departure
    # as reachable at any minute in O rather than only at departure_policy's single
    # instant. A RELAXATION -- Q_relaxed <= Q_true -- so a run using it may report a
    # valid LOWER bound and must not report its upper bound as the schedule's true cost.
    # None (default) is today's single-offset model exactly. Only meaningful under
    # recourse_resolution == "minute". tau*delta (offset 0) is the ceiling, not the
    # floor: it is the master's own committed instant, so a genuine grid is
    # anticipate-only (O subset [-delta, 0]); positive offsets remain accepted for
    # deliberate counterfactual use but do not represent a real degree of freedom.
    placement_offsets: list[float] | None = None
    degenerate_cut_probe_top_k: int = 6
    degenerate_cut_probe_top_k_out: int | None = None
    degenerate_cut_probe_top_k_ret: int | None = None
    degenerate_cut_zero_tol: float = 1e-9


@dataclass(slots=True)
class SolverSection:
    max_iterations: int
    tolerance: float
    # Wall-clock budget for the whole Benders loop. Checked between iterations, so
    # a run overshoots by at most one master solve (see master.per_iteration_time_limit_s).
    total_time_limit_s: int
    stall_max_no_improve_iters: int = 0
    stall_min_abs_improve: float = 0.0
    stall_min_rel_improve: float = 0.0
    master_solver: str = "cplex"
    subproblem_solver: str = "cplex_direct"
    solver_tee: bool = False


@dataclass(slots=True)
class RootConfig:
    schema: SchemaSection
    run: RunSection
    data: DataSection
    model: ModelSection
    master: MasterSection
    subproblem: SubproblemSection
    solver: SolverSection
    tolerances: "TolerancesSection"


@dataclass(slots=True)
class TolerancesSection:
    eps_bin: float = 1e-6
    eps_feas: float = 1e-7
    eps_cut: float = 1e-8
    eps_hash: float = 1e-6


# ---- Expression evaluation (energy params only) ----


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _eval_expr(expr: str, names: Mapping[str, Any]) -> float | int:
    """Safely evaluate a simple arithmetic expression with provided names."""
    node = ast.parse(expr, mode="eval")

    bin_ops = {
        ast.Add: _op.add,
        ast.Sub: _op.sub,
        ast.Mult: _op.mul,
        ast.Div: _op.truediv,
        ast.FloorDiv: _op.floordiv,
        ast.Mod: _op.mod,
        ast.Pow: _op.pow,
    }
    unary_ops = {ast.UAdd: _op.pos, ast.USub: _op.neg}

    def _eval(n: ast.AST) -> float | int:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("non-numeric constant in expression")
        num_node = getattr(ast, "Num", None)  # Python < 3.12 compatibility
        if num_node is not None and isinstance(
            n, num_node
        ):  # pragma: no cover - old ASTs
            return n.n  # type: ignore[attr-defined]
        if isinstance(n, ast.Name):
            if n.id not in names:
                raise NameError(f"unknown name '{n.id}' in expression")
            v = names[n.id]
            if _is_number(v):
                return v  # type: ignore[return-value]
            try:
                return float(v) if (isinstance(v, str) and v.strip()) else v  # type: ignore[return-value]
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"name '{n.id}' is not numeric: {v}") from exc
        if isinstance(n, ast.BinOp):
            if type(n.op) not in bin_ops:
                raise ValueError("operator not allowed in expression")
            return bin_ops[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp):
            if type(n.op) not in unary_ops:
                raise ValueError("unary operator not allowed in expression")
            return unary_ops[type(n.op)](_eval(n.operand))
        if isinstance(n, (ast.Tuple, ast.List)) and len(getattr(n, "elts", [])) == 1:
            return _eval(n.elts[0])  # type: ignore[index]
        raise ValueError("unsupported syntax in expression")

    return _eval(node)


def _looks_like_expr(value: str) -> bool:
    s = value.strip()
    return any(ch in s for ch in "+-*/()")


def resolve_energy_params(
    energy: EnergySection, names: Mapping[str, Any]
) -> dict[str, Any]:
    out = {
        "Emax": energy.Emax,
        "L": energy.L,
    }
    if energy.delta_chg is not None:
        if isinstance(energy.delta_chg, str) and _looks_like_expr(energy.delta_chg):
            out["delta_chg"] = _eval_expr(energy.delta_chg, names)
        else:
            out["delta_chg"] = energy.delta_chg
    return out


# ---- Validation helpers ----


def _as_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    return value


def _validated_objective_mode(raw) -> str:
    """B7.2. One of two named modes, refused rather than defaulted on a typo.

    A misspelled mode silently falling back to the weighted sum is exactly the shape
    of defect the manifest exists to prevent: the run would report an objective mode
    it did not use.
    """
    if raw is None:
        return "weighted_sum"
    mode = str(raw).strip().lower()
    if mode not in {"weighted_sum", "lexicographic"}:
        raise ValueError(
            "model.costs.objective_mode must be 'weighted_sum' or 'lexicographic', "
            f"got {raw!r}"
        )
    return mode


def _validated_core_certification(raw) -> str:
    """B14. One of two named modes, refused on a typo rather than defaulted.

    A misspelling falling back to "necessary_conditions" would make a run report a
    certification attempt it did not make -- the same class of defect as a manifest
    naming an objective mode the solve never used.
    """
    if raw is None:
        return "necessary_conditions"
    mode = str(raw).strip().lower()
    if mode not in {"none", "necessary_conditions"}:
        raise ValueError(
            "subproblem.mw_core_point_certification must be 'none' or "
            f"'necessary_conditions', got {raw!r}"
        )
    return mode


def _check_unknown_keys(data: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(k for k in data.keys() if k not in allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) in {where}: {', '.join(unknown)}")


def _require_keys(data: Mapping[str, Any], required: set[str], where: str) -> None:
    missing = sorted(k for k in required if k not in data)
    if missing:
        raise ValueError(f"Missing required key(s) in {where}: {', '.join(missing)}")


def _ensure_int(value: Any, where: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{where} must be an int")
    try:
        return int(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{where} must be an int") from exc


def _ensure_float(value: Any, where: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{where} must be a float")
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{where} must be a float") from exc


def _ensure_bool(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{where} must be a bool")


def _ensure_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")
    return value


def _ensure_str_list(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{where} must be a list of strings")
    return list(value)


def _ensure_num_list(value: Any, where: str) -> list[float]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a list of numbers")
    out: list[float] = []
    for v in value:
        out.append(_ensure_float(v, where))
    return out


def _ensure_num_or_expr(value: Any, where: str) -> float | int | str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip()
        if _looks_like_expr(s):
            return s
        try:
            return float(s)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"{where} must be numeric or an arithmetic expression"
            ) from exc
    raise ValueError(f"{where} must be numeric or an arithmetic expression")


def _ensure_mapping(value: Any, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    return dict(value)


def _disallow_expr(value: Any, where: str) -> Any:
    if isinstance(value, str) and _looks_like_expr(value):
        raise ValueError(f"{where} cannot be an expression; provide a numeric value")
    return value


# ---- Load / parse ----


def _load_yaml(path: Path) -> dict[str, Any]:
    if _yaml is None:
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. Install with 'pip install pyyaml'."
        )
    with path.open("r", encoding="utf-8") as f:
        data = _yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML document must be a mapping")
        return data


def upgrade_config_v1_to_v2(old: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade a v1 config dict to the v2 schema.

    Emits warnings describing deprecated keys and their mappings.
    """
    warnings_list: list[str] = []

    run = _as_mapping(old.get("run", {}), "run")
    master = _as_mapping(old.get("master", {}), "master")
    sub = _as_mapping(old.get("subproblem", {}), "subproblem")
    master_params = _as_mapping(master.get("params", {}), "master.params")
    sub_params = _as_mapping(sub.get("params", {}), "subproblem.params")

    # Branch-and-cut was removed in 2026-02 and revived under D44. A v1 config that
    # still carries the old keys predates the revival by a year and predates D30 by
    # six months, so its settings were tuned against a subproblem whose constraint
    # set moved with y. Upgrading them silently would carry that tuning forward as
    # if it meant something. Refuse, and make the user re-state the intent in v2.
    lazy_key = "use_lazy_cuts"
    lazy_cb_key = "lazy_cb_lp_solver"
    if lazy_key in master_params:
        raise ValueError(
            f"master.params.{lazy_key} is a pre-D30 key and does not upgrade. "
            "Branch-and-cut is available again in the v2 schema as "
            "master.branch_and_cut; set it there deliberately (D44)."
        )
    if lazy_cb_key in master_params:
        raise ValueError(
            f"master.params.{lazy_cb_key} is a pre-D30 key and does not upgrade. "
            "The v2 equivalent is master.branch_and_cut.callback_lp_solver (D44)."
        )
    if str(master_params.get("solver", "")).lower() == "cplex_persistent":
        raise ValueError(
            "master.params.solver=cplex_persistent does not upgrade. In v2 the "
            "backend is master.solver_backend, and cplex_persistent is only "
            "accepted there when master.branch_and_cut.enabled is true (D44)."
        )

    def _note(msg: str) -> None:
        warnings_list.append(msg)

    new: dict[str, Any] = {
        "schema": {"name": "mobauto2_benders_config", "version": 2},
        "run": {
            "name": None,
            "log_level": run.get("log_level", "INFO"),
            "log_file": None,
            "report_dir": None,
            "seed": run.get("seed"),
        },
        "data": {
            "demand_file": sub_params.get("demand_file"),
            "scenario_files": sub_params.get("scenario_files") or [],
            "scenario_weights": sub_params.get("scenario_weights"),
            "R_out": sub_params.get("R_out"),
            "R_ret": sub_params.get("R_ret"),
        },
        "model": {
            "time": {
                "T_minutes": master_params.get("T_minutes"),
                "T": master_params.get("T"),
                "slot_resolution": master_params.get("slot_resolution", 1),
                "trip_duration_minutes": master_params.get("trip_duration_minutes"),
                "trip_duration": master_params.get("trip_duration"),
                "trip_slots": master_params.get("trip_slots"),
            },
            "fleet": {
                "Q": master_params.get("Q"),
                "binit": master_params.get("binit"),
                "initial_actions": master_params.get("initial_actions"),
            },
            "energy": {
                "Emax": master_params.get("Emax"),
                "L": master_params.get("L"),
                "delta_chg": master_params.get("delta_chg"),
            },
            "costs": {
                "start_cost_epsilon": master_params.get("start_cost_epsilon", 0.0),
                "concurrency_penalty": master_params.get("concurrency_penalty", 0.0),
            },
        },
        "master": {
            "use_fifo_symmetry": master_params.get("use_fifo_symmetry", False),
            "symmetry_breaking": master_params.get("symmetry_breaking", False),
            "use_mip_start": master_params.get("use_mip_start", False),
            "per_iteration_time_limit_s": master_params.get("solve_time_limit_s"),
            "per_iteration_mipgap": master_params.get("mipgap"),
            "cplex_options": master_params.get("cplex_options", {}),
            "solver_backend": master_params.get("solver_backend", "cplex_direct"),
            "aggregate_cuts_by_tau": master_params.get("aggregate_cuts_by_tau", True),
            "cut_coeff_threshold": master_params.get("cut_coeff_threshold", 0.0),
            "theta_per_scenario": master_params.get("theta_per_scenario", False),
            "theta_by_direction": master_params.get("theta_by_direction"),
            "write_lp_after_cut": master_params.get("write_lp_after_cut", False),
            "window_trip_caps": master_params.get("window_trip_caps", False),
            # Both change the bound a run reports, so a table quoting one has to be
            # able to say which side of the A/B it came from (D18's obligation).
            "recourse_lower_bound": master_params.get("recourse_lower_bound", True),
            "lp_phase": master_params.get("lp_phase", False),
            "lp_phase_max_iters": master_params.get("lp_phase_max_iters"),
            "lp_phase_stall_iters": master_params.get("lp_phase_stall_iters"),
            "lp_phase_min_rel_improve": master_params.get("lp_phase_min_rel_improve"),
        },
        "subproblem": {
            "multi_cuts_by_scenario": sub_params.get("multi_cuts_by_scenario", True),
            "cut_mode": sub_params.get("cut_mode"),
            "acknowledge_no_lower_bound": sub_params.get(
                "acknowledge_no_lower_bound", False
            ),
            "use_magnanti_wong": sub_params.get("use_magnanti_wong", False),
            "mw_core_alpha": sub_params.get("mw_core_alpha", 0.3),
            "mw_core_eps": sub_params.get("mw_core_eps", 1e-3),
            "use_dual_slopes": sub_params.get("use_dual_slopes", False),
            "S": sub_params.get("S"),
            "Wmax_minutes": sub_params.get("Wmax_minutes"),
            "Wmax_slots": sub_params.get("Wmax_slots"),
            "p": sub_params.get("p"),
            "recourse_resolution": sub_params.get("recourse_resolution", "slot"),
            "departure_policy": sub_params.get("departure_policy", "start"),
            "placement_offsets": sub_params.get("placement_offsets"),
            "degenerate_cut_probe_top_k": sub_params.get(
                "degenerate_cut_probe_top_k", 6
            ),
            "degenerate_cut_probe_top_k_out": sub_params.get(
                "degenerate_cut_probe_top_k_out"
            ),
            "degenerate_cut_probe_top_k_ret": sub_params.get(
                "degenerate_cut_probe_top_k_ret"
            ),
            "degenerate_cut_zero_tol": sub_params.get("degenerate_cut_zero_tol", 1e-9),
        },
        "solver": {
            "max_iterations": run.get("max_iterations", 100),
            "tolerance": run.get("tolerance", 1e-4),
            "total_time_limit_s": run.get("time_limit_s", 600),
            "stall_max_no_improve_iters": run.get("stall_max_no_improve_iters", 0),
            "stall_min_abs_improve": run.get("stall_min_abs_improve", 0.0),
            "stall_min_rel_improve": run.get("stall_min_rel_improve", 0.0),
            "master_solver": master_params.get("solver", "cplex"),
            "subproblem_solver": sub_params.get("lp_solver", "cplex_direct"),
            "solver_tee": master_params.get("solver_tee", False),
        },
    }

    _note(
        "Deprecated v1 key 'run.*' mapped to 'run.*' (logging) and 'solver.*' (iterations/tolerance/time limits)."
    )
    _note(f"v1 run keys: {sorted(run.keys())}")
    _note(
        "Deprecated v1 key 'master.params.*' mapped into 'model.*', 'master.*', and 'solver.*'."
    )
    _note(f"v1 master.params keys: {sorted(master_params.keys())}")
    _note(
        "Deprecated v1 key 'subproblem.params.*' mapped into 'data.*', 'subproblem.*', and 'solver.*'."
    )
    _note(f"v1 subproblem.params keys: {sorted(sub_params.keys())}")

    if warnings_list:
        warnings.warn(
            "Loaded v1 config; please upgrade to schema version 2.\n"
            + "\n".join(warnings_list),
            stacklevel=2,
        )

    return new


def _parse_v2(raw: Mapping[str, Any]) -> RootConfig:
    data = _as_mapping(raw, "config")
    _check_unknown_keys(
        data,
        {
            "schema",
            "run",
            "data",
            "model",
            "master",
            "subproblem",
            "solver",
            "tolerances",
        },
        "config",
    )
    _require_keys(
        data,
        {"schema", "run", "data", "model", "master", "subproblem", "solver"},
        "config",
    )

    schema_raw = _as_mapping(data.get("schema"), "schema")
    _check_unknown_keys(schema_raw, {"name", "version"}, "schema")
    _require_keys(schema_raw, {"name", "version"}, "schema")
    schema = SchemaSection(
        name=_ensure_str(schema_raw.get("name"), "schema.name"),
        version=_ensure_int(schema_raw.get("version"), "schema.version"),
    )
    if schema.name != "mobauto2_benders_config" or schema.version != 2:
        raise ValueError(
            "Unsupported schema version; expected mobauto2_benders_config v2"
        )

    run_raw = _as_mapping(data.get("run"), "run")
    _check_unknown_keys(
        run_raw,
        {"name", "log_level", "log_file", "report_dir", "seed", "emit_reports"},
        "run",
    )
    run_name = run_raw.get("name")
    run = RunSection(
        name=(_ensure_str(run_name, "run.name") if run_name is not None else None),
        log_level=str(run_raw.get("log_level", "INFO")),
        log_file=run_raw.get("log_file"),
        report_dir=run_raw.get("report_dir"),
        seed=(
            _ensure_int(run_raw.get("seed"), "run.seed")
            if "seed" in run_raw and run_raw.get("seed") is not None
            else None
        ),
        emit_reports=_ensure_bool(
            run_raw.get("emit_reports", False), "run.emit_reports"
        ),
    )

    data_raw = _as_mapping(data.get("data"), "data")
    _check_unknown_keys(
        data_raw,
        {
            "demand_file",
            "scenario_files",
            "scenario_weights",
            "R_out",
            "R_ret",
            "scenarios",
        },
        "data",
    )
    demand_file_val = data_raw.get("demand_file")
    data_section = DataSection(
        demand_file=(
            _ensure_str(demand_file_val, "data.demand_file")
            if demand_file_val is not None
            else None
        ),
        scenario_files=_ensure_str_list(
            data_raw.get("scenario_files"), "data.scenario_files"
        ),
        scenario_weights=(
            _ensure_num_list(data_raw.get("scenario_weights"), "data.scenario_weights")
            if data_raw.get("scenario_weights") is not None
            else None
        ),
        R_out=(
            _ensure_num_list(data_raw.get("R_out"), "data.R_out")
            if data_raw.get("R_out") is not None
            else None
        ),
        R_ret=(
            _ensure_num_list(data_raw.get("R_ret"), "data.R_ret")
            if data_raw.get("R_ret") is not None
            else None
        ),
        scenarios=(
            data_raw.get("scenarios")
            if isinstance(data_raw.get("scenarios"), list)
            else None
        ),
    )

    model_raw = _as_mapping(data.get("model"), "model")
    _check_unknown_keys(model_raw, {"time", "fleet", "energy", "costs"}, "model")
    _require_keys(model_raw, {"time", "fleet", "energy", "costs"}, "model")

    time_raw = _as_mapping(model_raw.get("time"), "model.time")
    _check_unknown_keys(
        time_raw,
        {
            "T_minutes",
            "T",
            "slot_resolution",
            "trip_duration_minutes",
            "trip_duration",
            "trip_slots",
        },
        "model.time",
    )
    _require_keys(time_raw, {"slot_resolution"}, "model.time")
    if "T_minutes" not in time_raw and "T" not in time_raw:
        raise ValueError("model.time must include T_minutes or T")
    time_section = TimeSection(
        T_minutes=(
            _ensure_int(
                _disallow_expr(time_raw.get("T_minutes"), "model.time.T_minutes"),
                "model.time.T_minutes",
            )
            if time_raw.get("T_minutes") is not None
            else None
        ),
        T=(
            _ensure_int(
                _disallow_expr(time_raw.get("T"), "model.time.T"), "model.time.T"
            )
            if time_raw.get("T") is not None
            else None
        ),
        slot_resolution=_ensure_int(
            _disallow_expr(
                time_raw.get("slot_resolution"), "model.time.slot_resolution"
            ),
            "model.time.slot_resolution",
        ),
        trip_duration_minutes=(
            _ensure_int(
                _disallow_expr(
                    time_raw.get("trip_duration_minutes"),
                    "model.time.trip_duration_minutes",
                ),
                "model.time.trip_duration_minutes",
            )
            if time_raw.get("trip_duration_minutes") is not None
            else None
        ),
        trip_duration=(
            _ensure_int(
                _disallow_expr(
                    time_raw.get("trip_duration"), "model.time.trip_duration"
                ),
                "model.time.trip_duration",
            )
            if time_raw.get("trip_duration") is not None
            else None
        ),
        trip_slots=(
            _ensure_int(
                _disallow_expr(time_raw.get("trip_slots"), "model.time.trip_slots"),
                "model.time.trip_slots",
            )
            if time_raw.get("trip_slots") is not None
            else None
        ),
    )

    fleet_raw = _as_mapping(model_raw.get("fleet"), "model.fleet")
    _check_unknown_keys(
        fleet_raw, {"Q", "binit", "initial_battery", "initial_actions"}, "model.fleet"
    )
    _require_keys(fleet_raw, {"Q"}, "model.fleet")
    binit_raw = fleet_raw.get("initial_battery", fleet_raw.get("binit"))
    fleet_section = FleetSection(
        Q=_ensure_int(
            _disallow_expr(fleet_raw.get("Q"), "model.fleet.Q"), "model.fleet.Q"
        ),
        binit=(
            _ensure_num_list(binit_raw, "model.fleet.initial_battery")
            if binit_raw is not None
            else None
        ),
        initial_actions=(
            _ensure_str_list(
                fleet_raw.get("initial_actions"), "model.fleet.initial_actions"
            )
            if fleet_raw.get("initial_actions") is not None
            else None
        ),
    )

    energy_raw = _as_mapping(model_raw.get("energy"), "model.energy")
    _check_unknown_keys(
        energy_raw,
        {"Emax", "L", "delta_chg", "K_chg", "charger_occupancy_binary"},
        "model.energy",
    )
    _require_keys(energy_raw, {"Emax", "L"}, "model.energy")
    energy_section = EnergySection(
        Emax=_ensure_float(
            _disallow_expr(energy_raw.get("Emax"), "model.energy.Emax"),
            "model.energy.Emax",
        ),
        L=_ensure_float(
            _disallow_expr(energy_raw.get("L"), "model.energy.L"), "model.energy.L"
        ),
        delta_chg=(
            _ensure_num_or_expr(energy_raw.get("delta_chg"), "model.energy.delta_chg")
            if energy_raw.get("delta_chg") is not None
            else None
        ),
        K_chg=(
            int(
                _ensure_float(
                    _disallow_expr(energy_raw.get("K_chg"), "model.energy.K_chg"),
                    "model.energy.K_chg",
                )
            )
            if energy_raw.get("K_chg") is not None
            else None
        ),
        charger_occupancy_binary=bool(
            energy_raw.get("charger_occupancy_binary", False)
        ),
    )

    costs_raw = _as_mapping(model_raw.get("costs"), "model.costs")
    _check_unknown_keys(
        costs_raw,
        {"start_cost_epsilon", "concurrency_penalty", "objective_mode"},
        "model.costs",
    )
    cost_section = CostSection(
        start_cost_epsilon=(
            _ensure_float(
                _disallow_expr(
                    costs_raw.get("start_cost_epsilon"),
                    "model.costs.start_cost_epsilon",
                ),
                "model.costs.start_cost_epsilon",
            )
            if costs_raw.get("start_cost_epsilon") is not None
            else 0.0
        ),
        concurrency_penalty=(
            _ensure_float(
                _disallow_expr(
                    costs_raw.get("concurrency_penalty"),
                    "model.costs.concurrency_penalty",
                ),
                "model.costs.concurrency_penalty",
            )
            if costs_raw.get("concurrency_penalty") is not None
            else 0.0
        ),
        objective_mode=_validated_objective_mode(costs_raw.get("objective_mode")),
    )

    master_raw = _as_mapping(data.get("master"), "master")
    if "use_lazy_cuts" in master_raw:
        raise ValueError(
            "master.use_lazy_cuts is the pre-D30 flag and is not read. Branch-and-cut "
            "is configured under master.branch_and_cut (D44)."
        )
    # Renamed for clarity: both were confusable with the run-level budget and the
    # convergence tolerance, and both were silently overwritten before they were wired.
    if "solve_time_limit_s" in master_raw:
        raise ValueError(
            "master.solve_time_limit_s renamed to master.per_iteration_time_limit_s "
            "(ceiling on one master solve; the run budget is solver.total_time_limit_s)"
        )
    if "mipgap" in master_raw:
        raise ValueError(
            "master.mipgap renamed to master.per_iteration_mipgap "
            "(ceiling on the master MIP gap; the convergence criterion is solver.tolerance)"
        )
    _check_unknown_keys(
        master_raw,
        {
            "use_fifo_symmetry",
            "symmetry_breaking",
            "use_mip_start",
            "per_iteration_time_limit_s",
            "per_iteration_mipgap",
            "cplex_options",
            "solver_backend",
            "aggregate_cuts_by_tau",
            "cut_coeff_threshold",
            "theta_per_scenario",
            "theta_by_direction",
            "write_lp_after_cut",
            "window_trip_caps",
            "charge_before_idle",
            "recourse_lower_bound",
            "lp_phase",
            "lp_phase_max_iters",
            "lp_phase_stall_iters",
            "lp_phase_min_rel_improve",
            "branch_and_cut",
        },
        "master",
    )
    bnc_raw = _as_mapping(master_raw.get("branch_and_cut", {}), "master.branch_and_cut")
    _check_unknown_keys(
        bnc_raw,
        {
            "enabled",
            "callback_lp_solver",
            "lazy_cuts",
            "user_cuts",
            "seed_from_lp_phase",
            "control_no_callback",
        },
        "master.branch_and_cut",
    )
    bnc_section = BranchAndCutSection(
        enabled=_ensure_bool(
            bnc_raw.get("enabled", False), "master.branch_and_cut.enabled"
        ),
        callback_lp_solver=_ensure_str(
            bnc_raw.get("callback_lp_solver", "cplex_direct"),
            "master.branch_and_cut.callback_lp_solver",
        ),
        lazy_cuts=_ensure_bool(
            bnc_raw.get("lazy_cuts", True), "master.branch_and_cut.lazy_cuts"
        ),
        user_cuts=_ensure_bool(
            bnc_raw.get("user_cuts", False), "master.branch_and_cut.user_cuts"
        ),
        seed_from_lp_phase=_ensure_bool(
            bnc_raw.get("seed_from_lp_phase", True),
            "master.branch_and_cut.seed_from_lp_phase",
        ),
        control_no_callback=_ensure_bool(
            bnc_raw.get("control_no_callback", False),
            "master.branch_and_cut.control_no_callback",
        ),
    )
    backend_raw = str(master_raw.get("solver_backend", "cplex_direct")).lower()
    # The tree builds its own cplex_persistent solver from the same model, so
    # master.solver_backend is NOT the branch-and-cut switch and must stay on
    # cplex_direct: it is the backend the seeding LP phase runs on, and changing
    # it would make the cuts the tree starts from different from run 2's.
    if backend_raw == "cplex_persistent":
        raise ValueError(
            "master.solver_backend=cplex_persistent is not accepted. Branch-and-cut "
            "creates its own persistent solver for the tree (master.branch_and_cut); "
            "the master's own backend stays cplex_direct so the seeding LP phase "
            "reproduces run 2 exactly."
        )
    if bnc_section.enabled:
        if bnc_section.control_no_callback:
            if bnc_section.lazy_cuts or bnc_section.user_cuts:
                raise ValueError(
                    "master.branch_and_cut.control_no_callback is the no-callback "
                    "control and requires lazy_cuts=false and user_cuts=false; "
                    "otherwise the file names generators it will not register."
                )
        elif not (bnc_section.lazy_cuts or bnc_section.user_cuts):
            raise ValueError(
                "master.branch_and_cut.enabled with neither lazy_cuts nor user_cuts "
                "and control_no_callback=false registers no callback and generates "
                "nothing. If a no-callback control is what you want, say so with "
                "control_no_callback=true."
            )
        if bnc_section.user_cuts:
            raise ValueError(
                "master.branch_and_cut.user_cuts is not implemented yet (D44 step 2). "
                "Lazy constraints at integer incumbents come first, because they "
                "reproduce the loop's semantics inside one tree and make a "
                "regression against lp150_then_mip8.yaml legible."
            )
    master_section = MasterSection(
        use_fifo_symmetry=_ensure_bool(
            master_raw.get("use_fifo_symmetry", False), "master.use_fifo_symmetry"
        ),
        symmetry_breaking=_ensure_bool(
            master_raw.get("symmetry_breaking", False), "master.symmetry_breaking"
        ),
        use_mip_start=_ensure_bool(
            master_raw.get("use_mip_start", False), "master.use_mip_start"
        ),
        per_iteration_time_limit_s=(
            _ensure_int(
                _disallow_expr(
                    master_raw.get("per_iteration_time_limit_s"),
                    "master.per_iteration_time_limit_s",
                ),
                "master.per_iteration_time_limit_s",
            )
            if master_raw.get("per_iteration_time_limit_s") is not None
            else None
        ),
        per_iteration_mipgap=(
            _ensure_float(
                _disallow_expr(
                    master_raw.get("per_iteration_mipgap"),
                    "master.per_iteration_mipgap",
                ),
                "master.per_iteration_mipgap",
            )
            if master_raw.get("per_iteration_mipgap") is not None
            else None
        ),
        cplex_options=_ensure_mapping(
            master_raw.get("cplex_options"), "master.cplex_options"
        ),
        solver_backend=_ensure_str(
            master_raw.get("solver_backend", "cplex_direct"), "master.solver_backend"
        ),
        aggregate_cuts_by_tau=_ensure_bool(
            master_raw.get("aggregate_cuts_by_tau", True),
            "master.aggregate_cuts_by_tau",
        ),
        cut_coeff_threshold=_ensure_float(
            _disallow_expr(
                master_raw.get("cut_coeff_threshold", 0.0), "master.cut_coeff_threshold"
            ),
            "master.cut_coeff_threshold",
        ),
        theta_per_scenario=_ensure_bool(
            master_raw.get("theta_per_scenario", False), "master.theta_per_scenario"
        ),
        theta_by_direction=(
            None
            if master_raw.get("theta_by_direction") is None
            else _ensure_bool(
                master_raw.get("theta_by_direction"), "master.theta_by_direction"
            )
        ),
        write_lp_after_cut=_ensure_bool(
            master_raw.get("write_lp_after_cut", False), "master.write_lp_after_cut"
        ),
        window_trip_caps=_ensure_bool(
            master_raw.get("window_trip_caps", False), "master.window_trip_caps"
        ),
        charge_before_idle=_ensure_bool(
            master_raw.get("charge_before_idle", True), "master.charge_before_idle"
        ),
        recourse_lower_bound=_ensure_bool(
            master_raw.get("recourse_lower_bound", True), "master.recourse_lower_bound"
        ),
        lp_phase=_ensure_bool(master_raw.get("lp_phase", False), "master.lp_phase"),
        lp_phase_max_iters=_ensure_int(
            _disallow_expr(
                master_raw.get("lp_phase_max_iters", 10), "master.lp_phase_max_iters"
            ),
            "master.lp_phase_max_iters",
        ),
        lp_phase_stall_iters=_ensure_int(
            _disallow_expr(
                master_raw.get("lp_phase_stall_iters", 3), "master.lp_phase_stall_iters"
            ),
            "master.lp_phase_stall_iters",
        ),
        lp_phase_min_rel_improve=_ensure_float(
            _disallow_expr(
                master_raw.get("lp_phase_min_rel_improve", 0.005),
                "master.lp_phase_min_rel_improve",
            ),
            "master.lp_phase_min_rel_improve",
        ),
        branch_and_cut=bnc_section,
    )
    if bnc_section.enabled and bnc_section.seed_from_lp_phase and not master_section.lp_phase:
        raise ValueError(
            "master.branch_and_cut.seed_from_lp_phase=true needs master.lp_phase=true. "
            "The strong root measured in D40/D45 comes from the LP phase's cuts; "
            "without it the tree starts from a root of ~0, which is a different "
            "experiment and must be asked for explicitly (seed_from_lp_phase=false)."
        )

    sub_raw = _as_mapping(data.get("subproblem"), "subproblem")
    _check_unknown_keys(
        sub_raw,
        {
            "cut_architecture",
            "multi_cuts_by_scenario",
            "cut_mode",
            "acknowledge_no_lower_bound",
            "use_magnanti_wong",
            "mw_core_alpha",
            "mw_core_point_certification",
            "mw_core_eps",
            "use_dual_slopes",
            "S",
            "Wmax_minutes",
            "Wmax_slots",
            "p",
            "p_minutes",
            "recourse_resolution",
            "departure_policy",
            "same_slot_eligibility",
            "placement_offsets",
            "degenerate_cut_probe_top_k",
            "degenerate_cut_probe_top_k_out",
            "degenerate_cut_probe_top_k_ret",
            "degenerate_cut_zero_tol",
        },
        "subproblem",
    )
    _require_keys(sub_raw, {"S"}, "subproblem")
    if "Wmax_minutes" not in sub_raw and "Wmax_slots" not in sub_raw:
        raise ValueError("subproblem must include Wmax_minutes or Wmax_slots")

    # Resolve the unmet-demand penalty to slot units, once, here (D50).
    #
    # Exactly one of the two forms, and both being present is an error rather than a
    # precedence rule. A precedence rule would let a config state two different
    # policies and silently honour one -- and the whole reason this key exists is that
    # `p` already meant two different things in two places without anything saying so.
    _has_p = "p" in sub_raw
    _has_p_minutes = "p_minutes" in sub_raw
    if _has_p and _has_p_minutes:
        raise ValueError(
            "subproblem sets both p and p_minutes; they state the same policy in "
            "different units and only one may be given. p is in slot units and moves "
            "with model.time.slot_resolution; p_minutes does not. Prefer p_minutes."
        )
    if not _has_p and not _has_p_minutes:
        raise ValueError("subproblem must include p or p_minutes")
    _recourse_resolution = str(
        sub_raw.get("recourse_resolution", "slot")
    ).strip().lower()
    if _recourse_resolution not in {"slot", "minute"}:
        raise ValueError(
            f"subproblem.recourse_resolution must be 'slot' or 'minute', got "
            f"{_recourse_resolution!r}"
        )
    _departure_policy = str(
        sub_raw.get("departure_policy", "start")
    ).strip().lower()
    if _departure_policy not in {"start", "midpoint", "end"}:
        raise ValueError(
            f"subproblem.departure_policy must be 'start', 'midpoint' or 'end', got "
            f"{_departure_policy!r}"
        )
    _same_slot_eligibility = str(
        sub_raw.get("same_slot_eligibility", "forbid")
    ).strip().lower()
    if _same_slot_eligibility not in {"forbid", "allow"}:
        raise ValueError(
            f"subproblem.same_slot_eligibility must be 'forbid' or 'allow', got "
            f"{_same_slot_eligibility!r}"
        )
    if _recourse_resolution == "minute" and "Wmax_minutes" not in sub_raw:
        raise ValueError(
            "subproblem.recourse_resolution='minute' requires Wmax_minutes. Wmax_slots "
            "cannot substitute: a slot window is not an exact number of minutes, which "
            "is the entire point of evaluating at minute resolution (D51)."
        )
    # F2 (docs/PROJECT_STATE_v6.md section 5). Only meaningful when the recourse can
    # already see minutes -- the offset grid refines WHICH minute within a slot a
    # departure is reachable at, and the slot recourse has no minute axis to refine.
    _placement_offsets_raw = sub_raw.get("placement_offsets")
    _placement_offsets: list[float] | None = None
    if _placement_offsets_raw is not None:
        if _recourse_resolution != "minute":
            raise ValueError(
                "subproblem.placement_offsets is only meaningful under "
                "recourse_resolution='minute'; got "
                f"recourse_resolution={_recourse_resolution!r}"
            )
        if not isinstance(_placement_offsets_raw, list) or not _placement_offsets_raw:
            raise ValueError(
                "subproblem.placement_offsets must be a non-empty list of minute "
                f"offsets, got {_placement_offsets_raw!r}"
            )
        try:
            _placement_offsets = [float(o) for o in _placement_offsets_raw]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "subproblem.placement_offsets must be numeric, got "
                f"{_placement_offsets_raw!r}"
            ) from exc
        # D76: negative offsets are legitimate here (anticipation -- a departure leaving
        # earlier than its slot's own tau*delta), so no sign restriction at this layer.
        # The exact bound, [-slot_resolution, slot_resolution], is enforced downstream by
        # minute_pricer.py::_offset_grid / solve_minute_recourse, which have delta in
        # scope; this layer only needs the values to be numeric, already checked above.
    # Resolve the cut generator (S1b).
    #
    # `cut_mode` is the one key. The legacy pair `use_magnanti_wong` / `use_dual_slopes`
    # is still accepted so no shipped config changes behaviour, and is resolved with the
    # dispatch's own precedence: mw wins, then dual, then finite differences. That
    # precedence is exactly what made `dual` unreachable -- every shipped config sets
    # BOTH booleans true -- so the enum exists to let it be selected at all.
    _CUT_MODES = ("mw", "dual", "finite_difference")
    _cut_mode_raw = sub_raw.get("cut_mode")
    _legacy_mw = "use_magnanti_wong" in sub_raw
    _legacy_dual = "use_dual_slopes" in sub_raw
    if _cut_mode_raw is not None and (_legacy_mw or _legacy_dual):
        raise ValueError(
            "subproblem.cut_mode is set alongside use_magnanti_wong/use_dual_slopes. "
            "They state the same choice in two forms and resolving both would mean "
            "picking a winner silently. Use cut_mode alone: "
            f"one of {', '.join(_CUT_MODES)}."
        )
    if _cut_mode_raw is None:
        _resolved_cut_mode = (
            "mw"
            if bool(sub_raw.get("use_magnanti_wong", False))
            else ("dual" if bool(sub_raw.get("use_dual_slopes", False)) else "finite_difference")
        )
    else:
        _resolved_cut_mode = str(_cut_mode_raw).strip().lower()
        if _resolved_cut_mode not in _CUT_MODES:
            raise ValueError(
                f"subproblem.cut_mode must be one of {', '.join(_CUT_MODES)}, got "
                f"{_cut_mode_raw!r}"
            )

    # finite_difference produces cuts with NO lower-bound guarantee (S7).
    #
    # The runtime already drops `best_lb` when this mode is used, which is correct and
    # tested -- but only after the run has spent its budget. A run that cannot produce a
    # bound should fail before it spends an hour not producing one. Requiring an explicit
    # acknowledgement also stops the mode being reached by ACCIDENT, which is how it was
    # reachable before: it is the `else` branch of the legacy booleans, so a config that
    # simply omitted both landed on it.
    if _resolved_cut_mode == "finite_difference" and not bool(
        sub_raw.get("acknowledge_no_lower_bound", False)
    ):
        raise ValueError(
            "subproblem cut generation resolves to 'finite_difference', which produces "
            "cuts with no lower-bound guarantee: a perturbation estimate bounds the "
            "recourse from ABOVE, so a cut built from it can exclude the optimum "
            "(handout 75/76). The run would complete and then report no lower bound. "
            "Set subproblem.cut_mode to 'mw' or 'dual', or set "
            "subproblem.acknowledge_no_lower_bound: true to run it as a diagnostic. "
            "Note this mode is also the fall-through of the legacy "
            "use_magnanti_wong/use_dual_slopes pair, so omitting both selects it."
        )

    # Magnanti-Wong on the minute recourse (B2, handout item, 2026-09-02).
    #
    # Until this port, `use_magnanti_wong: true` with `recourse_resolution: 'minute'`
    # was refused at load: `solve_mw_dual` built only the dual of the SLOT primal
    # (`alpha[t] + pi[tau] <= (tau-t)` over slot arcs), which is a different LP from
    # the minute recourse's own (`alpha[m] + pi[tau] <= (dep_minute-m)/delta` over
    # minute arcs, one demand row per arrival minute, D51) -- a dual feasible for one
    # carries no weak-duality relation to the other, and a cut built from it can
    # OVERESTIMATE, D30's exact failure mode. `solve_mw_dual_minute`
    # (`minute_pricer.py`) is the minute recourse's OWN dual, built over its own
    # arcs, so the combination is no longer refused. See
    # `docs/BENDERS_SPEC_v4.md` section 2.6/2.11 and docs_decisions.md's B2 entry.
    if _has_p_minutes:
        _p_minutes_val = _ensure_float(
            _disallow_expr(sub_raw.get("p_minutes"), "subproblem.p_minutes"),
            "subproblem.p_minutes",
        )
        _slot_res = int(time_section.slot_resolution)
        if _slot_res <= 0:
            raise ValueError(
                f"model.time.slot_resolution must be positive to convert p_minutes, "
                f"got {_slot_res!r}"
            )
        # No rounding. p is a cost coefficient, not an index bound like Wmax, so
        # ceil()-ing it here would silently change the policy it encodes.
        _p_slots_val = float(_p_minutes_val) / float(_slot_res)
    else:
        _p_minutes_val = None
        _p_slots_val = _ensure_float(
            _disallow_expr(sub_raw.get("p"), "subproblem.p"), "subproblem.p"
        )
    sub_section = SubproblemSection(
        cut_architecture=(
            str(sub_raw.get("cut_architecture")).strip().lower()
            if sub_raw.get("cut_architecture") is not None
            else None
        ),
        multi_cuts_by_scenario=_ensure_bool(
            sub_raw.get("multi_cuts_by_scenario", True),
            "subproblem.multi_cuts_by_scenario",
        ),
        cut_mode=_resolved_cut_mode,
        acknowledge_no_lower_bound=_ensure_bool(
            sub_raw.get("acknowledge_no_lower_bound", False),
            "subproblem.acknowledge_no_lower_bound",
        ),
        use_magnanti_wong=_ensure_bool(
            sub_raw.get("use_magnanti_wong", False), "subproblem.use_magnanti_wong"
        ),
        mw_core_point_certification=_validated_core_certification(
            sub_raw.get("mw_core_point_certification")
        ),
        mw_core_alpha=_ensure_float(
            _disallow_expr(
                sub_raw.get("mw_core_alpha", 0.3), "subproblem.mw_core_alpha"
            ),
            "subproblem.mw_core_alpha",
        ),
        mw_core_eps=_ensure_float(
            _disallow_expr(sub_raw.get("mw_core_eps", 1e-3), "subproblem.mw_core_eps"),
            "subproblem.mw_core_eps",
        ),
        use_dual_slopes=_ensure_bool(
            sub_raw.get("use_dual_slopes", False), "subproblem.use_dual_slopes"
        ),
        S=_ensure_float(
            _disallow_expr(sub_raw.get("S"), "subproblem.S"), "subproblem.S"
        ),
        Wmax_minutes=(
            _ensure_int(
                _disallow_expr(sub_raw.get("Wmax_minutes"), "subproblem.Wmax_minutes"),
                "subproblem.Wmax_minutes",
            )
            if sub_raw.get("Wmax_minutes") is not None
            else None
        ),
        Wmax_slots=(
            _ensure_int(
                _disallow_expr(sub_raw.get("Wmax_slots"), "subproblem.Wmax_slots"),
                "subproblem.Wmax_slots",
            )
            if sub_raw.get("Wmax_slots") is not None
            else None
        ),
        p=_p_slots_val,
        p_minutes=_p_minutes_val,
        recourse_resolution=_recourse_resolution,
        departure_policy=_departure_policy,
        same_slot_eligibility=_same_slot_eligibility,
        placement_offsets=_placement_offsets,
        degenerate_cut_probe_top_k=_ensure_int(
            _disallow_expr(
                sub_raw.get("degenerate_cut_probe_top_k", 6),
                "subproblem.degenerate_cut_probe_top_k",
            ),
            "subproblem.degenerate_cut_probe_top_k",
        ),
        degenerate_cut_probe_top_k_out=(
            _ensure_int(
                _disallow_expr(
                    sub_raw.get("degenerate_cut_probe_top_k_out"),
                    "subproblem.degenerate_cut_probe_top_k_out",
                ),
                "subproblem.degenerate_cut_probe_top_k_out",
            )
            if sub_raw.get("degenerate_cut_probe_top_k_out") is not None
            else None
        ),
        degenerate_cut_probe_top_k_ret=(
            _ensure_int(
                _disallow_expr(
                    sub_raw.get("degenerate_cut_probe_top_k_ret"),
                    "subproblem.degenerate_cut_probe_top_k_ret",
                ),
                "subproblem.degenerate_cut_probe_top_k_ret",
            )
            if sub_raw.get("degenerate_cut_probe_top_k_ret") is not None
            else None
        ),
        degenerate_cut_zero_tol=_ensure_float(
            _disallow_expr(
                sub_raw.get("degenerate_cut_zero_tol", 1e-9),
                "subproblem.degenerate_cut_zero_tol",
            ),
            "subproblem.degenerate_cut_zero_tol",
        ),
    )

    solver_raw = _as_mapping(data.get("solver"), "solver")
    if "time_limit_s" in solver_raw:
        raise ValueError(
            "solver.time_limit_s renamed to solver.total_time_limit_s "
            "(budget for the whole Benders loop; the per-iteration master ceiling is "
            "master.per_iteration_time_limit_s)"
        )
    _check_unknown_keys(
        solver_raw,
        {
            "max_iterations",
            "tolerance",
            "total_time_limit_s",
            "stall_max_no_improve_iters",
            "stall_min_abs_improve",
            "stall_min_rel_improve",
            "master_solver",
            "subproblem_solver",
            "solver_tee",
        },
        "solver",
    )
    _require_keys(
        solver_raw,
        {
            "max_iterations",
            "tolerance",
            "total_time_limit_s",
            "master_solver",
            "subproblem_solver",
        },
        "solver",
    )
    solver_section = SolverSection(
        max_iterations=_ensure_int(
            _disallow_expr(solver_raw.get("max_iterations"), "solver.max_iterations"),
            "solver.max_iterations",
        ),
        tolerance=_ensure_float(
            _disallow_expr(solver_raw.get("tolerance"), "solver.tolerance"),
            "solver.tolerance",
        ),
        total_time_limit_s=_ensure_int(
            _disallow_expr(
                solver_raw.get("total_time_limit_s"), "solver.total_time_limit_s"
            ),
            "solver.total_time_limit_s",
        ),
        stall_max_no_improve_iters=_ensure_int(
            _disallow_expr(
                solver_raw.get("stall_max_no_improve_iters", 0),
                "solver.stall_max_no_improve_iters",
            ),
            "solver.stall_max_no_improve_iters",
        ),
        stall_min_abs_improve=_ensure_float(
            _disallow_expr(
                solver_raw.get("stall_min_abs_improve", 0.0),
                "solver.stall_min_abs_improve",
            ),
            "solver.stall_min_abs_improve",
        ),
        stall_min_rel_improve=_ensure_float(
            _disallow_expr(
                solver_raw.get("stall_min_rel_improve", 0.0),
                "solver.stall_min_rel_improve",
            ),
            "solver.stall_min_rel_improve",
        ),
        master_solver=_ensure_str(
            solver_raw.get("master_solver"), "solver.master_solver"
        ),
        subproblem_solver=_ensure_str(
            solver_raw.get("subproblem_solver"), "solver.subproblem_solver"
        ),
        solver_tee=_ensure_bool(
            solver_raw.get("solver_tee", False), "solver.solver_tee"
        ),
    )
    if solver_section.master_solver.lower() == "cplex_persistent":
        # Deliberately still refused even under D44. Two keys can name a backend and
        # master.solver_backend is the one master_impl reads first; accepting the
        # backend here too would let a config set them to different values and give
        # no way to tell which one the run used.
        raise ValueError(
            "solver.master_solver=cplex_persistent is not the branch-and-cut switch. "
            "Set master.solver_backend=cplex_persistent together with "
            "master.branch_and_cut.enabled=true, and leave solver.master_solver=cplex."
        )
    if bnc_section.enabled and bnc_section.seed_from_lp_phase:
        # The seeding run must end when the LP phase ends. If max_iterations is
        # larger, the loop starts MIP iterations before the tree ever runs, and
        # the result would be branch-and-cut on top of an unrecorded number of
        # loop iterations -- exactly the "say which cut budget produced a number"
        # failure the README's reading rules exist for.
        if solver_section.max_iterations != master_section.lp_phase_max_iters:
            raise ValueError(
                "with master.branch_and_cut.seed_from_lp_phase, "
                f"solver.max_iterations ({solver_section.max_iterations}) must equal "
                f"master.lp_phase_max_iters ({master_section.lp_phase_max_iters}); "
                "otherwise the loop runs MIP iterations before the tree and the "
                "tree's cut budget is not the one the config names."
            )

    tol_raw = _as_mapping(data.get("tolerances", {}), "tolerances")
    _check_unknown_keys(
        tol_raw, {"eps_bin", "eps_feas", "eps_cut", "eps_hash"}, "tolerances"
    )
    tol_section = TolerancesSection(
        eps_bin=_ensure_float(
            _disallow_expr(tol_raw.get("eps_bin", 1e-6), "tolerances.eps_bin"),
            "tolerances.eps_bin",
        ),
        eps_feas=_ensure_float(
            _disallow_expr(tol_raw.get("eps_feas", 1e-7), "tolerances.eps_feas"),
            "tolerances.eps_feas",
        ),
        eps_cut=_ensure_float(
            _disallow_expr(tol_raw.get("eps_cut", 1e-8), "tolerances.eps_cut"),
            "tolerances.eps_cut",
        ),
        eps_hash=_ensure_float(
            _disallow_expr(tol_raw.get("eps_hash", 1e-6), "tolerances.eps_hash"),
            "tolerances.eps_hash",
        ),
    )

    # A multi-scenario run has to put its lower bound and its upper bound on the
    # same problem. One cut per scenario against a SINGLE theta forces
    # theta >= Q_s(y) for every s, i.e. theta >= max_s Q_s(y), while the reported
    # upper bound is the weighted mean of the same quantities. Since max >= mean,
    # the master's optimum can then exceed the true optimum of the problem the UB
    # measures: a bound that is not a bound, the D15/D16 failure mode again.
    #
    # Either give each scenario its own theta (master.theta_per_scenario: true,
    # objective sum_s w_s theta_s) or average the cuts
    # (subproblem.multi_cuts_by_scenario: false). Both are consistent with a
    # mean-aggregated UB.
    _has_scenarios = bool(data_section.scenario_files or data_section.scenarios)
    if (
        _has_scenarios
        and sub_section.multi_cuts_by_scenario
        and not master_section.theta_per_scenario
    ):
        raise ValueError(
            "subproblem.multi_cuts_by_scenario is true with master.theta_per_scenario "
            "false on a multi-scenario run. One cut per scenario on a shared theta "
            "bounds max_s Q_s(y) while the reported UB is the weighted mean, so the "
            "lower bound would not bound the problem being measured. Set "
            "master.theta_per_scenario: true, or "
            "subproblem.multi_cuts_by_scenario: false."
        )

    # ---------------------------------------------------------------- B1
    # Resolve `subproblem.cut_architecture` into the two booleans the engines read,
    # or derive it from them when it was not stated.
    #
    # The architecture used to be expressible only as a PAIR of booleans in two
    # different config sections, with no name for either combination and nothing
    # recording which one produced a table. The audit's item 1.4 is downstream of
    # that: a run could not say what its cuts were, so the report described one
    # architecture while the code ran another.
    #
    # Precedence is deliberately absent. Stating the architecture and then
    # contradicting it with a boolean is an error, not something resolved quietly in
    # favour of one of them.
    _ARCHITECTURES = {
        "aggregated": (False, False),
        "disaggregated": (True, True),
    }
    _arch = sub_section.cut_architecture
    _legacy_pair = (
        bool(sub_section.multi_cuts_by_scenario),
        bool(master_section.theta_per_scenario),
    )
    _explicit_legacy = {
        k
        for k, present in (
            ("subproblem.multi_cuts_by_scenario", "multi_cuts_by_scenario" in sub_raw),
            ("master.theta_per_scenario", "theta_per_scenario" in master_raw),
        )
        if present
    }
    if _arch is not None:
        if _arch not in _ARCHITECTURES:
            raise ValueError(
                "subproblem.cut_architecture must be 'aggregated' or 'disaggregated', "
                f"got {_arch!r}"
            )
        _want = _ARCHITECTURES[_arch]
        if _explicit_legacy and _legacy_pair != _want:
            raise ValueError(
                f"subproblem.cut_architecture: {_arch} contradicts "
                + ", ".join(sorted(_explicit_legacy))
                + f". The architecture implies (multi_cuts_by_scenario, "
                f"theta_per_scenario) = {_want}, the config states {_legacy_pair}. "
                "Drop the booleans and keep the architecture, or drop the "
                "architecture and keep the booleans -- not both."
            )
        sub_section.multi_cuts_by_scenario, master_section.theta_per_scenario = _want
    elif not _has_scenarios:
        # A single-demand run has nothing to aggregate over, so the two architectures
        # are the same object and the booleans below say nothing about it. Naming it
        # here keeps every manifest carrying the field, without inventing a
        # distinction the run does not have.
        sub_section.cut_architecture = "single_scenario"
    else:
        # No architecture stated. Derive it, so the manifest can still name one.
        for name, pair in _ARCHITECTURES.items():
            if _legacy_pair == pair:
                sub_section.cut_architecture = name
                break
        else:
            raise ValueError(
                "subproblem.multi_cuts_by_scenario="
                f"{_legacy_pair[0]} with master.theta_per_scenario={_legacy_pair[1]} "
                "is neither the aggregated nor the disaggregated architecture. One cut "
                "per scenario needs one theta per scenario, and a single theta needs a "
                "single aggregated cut; the two halves of a mixed setting bound "
                "different quantities. State subproblem.cut_architecture as "
                "'aggregated' or 'disaggregated' instead."
            )

    model_section = ModelSection(
        time=time_section,
        fleet=fleet_section,
        energy=energy_section,
        costs=cost_section,
    )

    return RootConfig(
        schema=schema,
        run=run,
        data=data_section,
        model=model_section,
        master=master_section,
        subproblem=sub_section,
        solver=solver_section,
        tolerances=tol_section,
    )


def load_config(path: str | Path | None) -> RootConfig:
    """Load configuration from YAML and return the v2 config dataclasses."""
    cfg_path = DEFAULT_CONFIG_PATH if path is None else Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    raw = _load_yaml(cfg_path)
    if "schema" not in raw:
        raw = upgrade_config_v1_to_v2(raw)
    else:
        schema_raw = raw.get("schema")
        if not isinstance(schema_raw, dict) or schema_raw.get("version") != 2:
            raw = upgrade_config_v1_to_v2(raw)
    return _parse_v2(raw)


__all__ = [
    "SchemaSection",
    "RunSection",
    "DataSection",
    "TimeSection",
    "FleetSection",
    "EnergySection",
    "CostSection",
    "ModelSection",
    "MasterSection",
    "SubproblemSection",
    "SolverSection",
    "RootConfig",
    "TolerancesSection",
    "DEFAULT_CONFIG_PATH",
    "load_config",
    "resolve_energy_params",
    "upgrade_config_v1_to_v2",
]
