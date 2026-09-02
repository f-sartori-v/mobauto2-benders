"""Stage 3, step 2: the Dantzig-Wolfe root by COLUMN GENERATION, where enumeration is closed.

    python scripts/stage3_column_generation.py --validate
    python scripts/stage3_column_generation.py --config configs/milp/<bigger>.yaml

D57 measured the DW root exactly at Q=2/T=22 by enumerating all 87 863 columns, and it
moved: 216.35 -> 242.49 against a proven optimum of 293.37. Enumeration is closed past
that instance (133 957 148 patterns at the Q=3 point, D56), which is where this script
takes over.

THE PRICING PROBLEM IS THE MASTER WITH Q=1, AND THAT IS THE WHOLE DESIGN OF THIS SCRIPT.
`DESIGN_DD_v1.md` stage 3 routes pricing through a resource-constrained shortest path on
the stage-1 diagram, which needs a battery dominance rule that the design itself flags as
unproven -- `charge_before_idle` makes charging at `t` depend on `c[t-1]`, so "carry the
maximum reachable battery" is a claim about continuations, not an obvious fact.

Building the pricing problem as `MobautoMilpModel` with one vehicle sidesteps that
entirely:

  * its feasible set IS the master's single-vehicle feasible set, by construction rather
    than by a re-implementation that has to be kept in sync -- and a hand-written second
    copy of a model is exactly how the two `cplex_log` parsers (D50) and the three `Wmax`
    conversions (D53) drifted apart;
  * no dominance rule is involved, so nothing here depends on an unproved lemma;
  * at T=44 a single-vehicle MILP is tiny.

It is slower per call than a DP would be. That is the right trade for a measurement whose
purpose is to decide whether the DP is worth writing at all.

VALIDATION BEFORE USE. `--validate` runs at Q=2/T=22 and requires the root to match the
enumerated answer 242.4891 to 1e-3. A column generation loop with a dual sign error, a
missing constant, or a pricing model that is subtly more permissive than the master will
still converge -- to the wrong number, confidently. The enumerated instance is the only
place that error is detectable, so it is checked there before the loop is pointed anywhere
it cannot be.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

VALIDATE_CONFIG = "configs/milp/baseline_d9_p56_monolith.yaml"
# D57, all 159 768 columns present. This constant was 242.4891 for one afternoon, from an
# enumeration that silently dropped every pattern whose last trip arrives exactly at T-1.
# THIS CHECK IS WHAT CAUGHT IT: column generation converged BELOW the "known" root, which
# is impossible for a restricted master and can only mean the pool was a subset. A pool
# missing columns raises the root, so the defect flattered the result and the integer
# pool check (lambda integer -> the monolith optimum) passed straight through it -- that
# check only proves the optimum is IN the pool, never that the pool is complete.
ENUMERATED_ROOT = 233.1067
ENUMERATED_TOL = 1e-3


def _pricing_model(mp: dict, T: int):
    """The master's single-vehicle feasible set, as a model whose objective we rewrite."""
    import pyomo.environ as pyo
    from mobauto2_milp.model import MobautoMilpModel

    mp1 = dict(mp)
    mp1["Q"] = 1
    mp1["binit"] = list(mp["binit"])[:1]
    mp1["initial_actions"] = list(mp["initial_actions"])[:1]
    # Symmetry breaking over one vehicle is vacuous, and leaving it on risks a canonical
    # ordering row quietly forbidding a column the Q-vehicle master would accept.
    mp1["use_fifo_symmetry"] = False
    mp1["symmetry_breaking"] = False

    pm = MobautoMilpModel(mp1)
    pm.initialize()
    m = pm.m
    m.obj.deactivate()
    # theta is split by direction by default and per-scenario under another flag, so pin
    # whichever exists. Missing one entirely would leave a free variable out of the
    # pricing objective, which is silent and would corrupt every reduced cost.
    pinned = 0
    for nm in ("theta", "theta_out", "theta_ret"):
        v = getattr(m, nm, None)
        if v is not None:
            v.fix(0.0)
            pinned += 1
    if hasattr(m, "theta_s"):
        for s in m.Scenarios:
            m.theta_s[s].fix(0.0)
            pinned += 1
    if pinned == 0:
        raise RuntimeError("no theta variable found on the pricing model to pin")
    m.price_coef_out = pyo.Param(m.T, initialize=0.0, mutable=True)
    m.price_coef_ret = pyo.Param(m.T, initialize=0.0, mutable=True)
    m.price_obj = pyo.Objective(
        expr=sum(
            m.price_coef_out[t] * m.yOUT[0, t] + m.price_coef_ret[t] * m.yRET[0, t]
            for t in m.T
        ),
        sense=pyo.minimize,
    )
    return pm


def _extract_column(m, T: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    import pyomo.environ as pyo

    o = tuple(t for t in range(T) if float(pyo.value(m.yOUT[0, t])) > 0.5)
    r = tuple(t for t in range(T) if float(pyo.value(m.yRET[0, t])) > 0.5)
    return o, r


def _scenarios(cfg, load_request_minutes):
    """Every scenario the config lists, with equal weights unless it says otherwise.

    Silently pricing scenario 0 alone would be a different problem from the one the
    config describes, and would report a root for it as if it were the root for this
    one -- D55 measured that single-scenario gains are several times the four-scenario
    figure, so the two are not interchangeable.
    """
    files = list(cfg.data.scenario_files)
    if not files:
        raise SystemExit("config lists no scenario_files")
    weights = getattr(cfg.data, "scenario_weights", None)
    if weights:
        if len(weights) != len(files):
            raise SystemExit(
                f"{len(files)} scenario files but {len(weights)} weights"
            )
        ws = [float(w) for w in weights]
    else:
        ws = [1.0] * len(files)
    return [(load_request_minutes(f), w) for f, w in zip(files, ws)]


def solve_root(config: str, p_minutes: float, policy: str, max_iters: int,
               verbose: bool = True, time_limit_s: float | None = None,
               ) -> tuple[float, int, float]:
    """Returns (root, columns_used, seconds)."""
    import pyomo.environ as pyo
    from mobauto2_milp.config import load_config
    from mobauto2_milp.app import _prepare_params
    from mobauto2_benders.minute_pricer import attach_minute_recourse, load_request_minutes

    cfg = load_config(config)
    mp, _sp = _prepare_params(cfg, {})
    delta = int(cfg.model.time.slot_resolution)
    T = int(cfg.model.time.T_minutes) // delta
    Q = int(cfg.model.fleet.Q)
    S = float(cfg.service.S)
    wmax = float(cfg.service.Wmax_minutes)
    scen = _scenarios(cfg, load_request_minutes)
    eps_start = float(mp.get("start_cost_epsilon", 0.0) or 0.0)
    conc_pen = float(mp.get("concurrency_penalty", 0.0) or 0.0)

    binit = list(mp["binit"])[:Q]
    actions = list(mp["initial_actions"])[:Q]
    if len(set(binit)) != 1 or len(set(actions)) != 1:
        raise SystemExit(
            f"fleet is not homogeneous (E4): binit={binit} actions={actions}; "
            "one column pool cannot serve every vehicle"
        )

    pool: list[tuple[tuple[int, ...], tuple[int, ...]]] = [((), ())]  # the idle column
    pm = _pricing_model(mp, T)
    price_opt = pyo.SolverFactory("cplex_direct")
    rmp_opt = pyo.SolverFactory("cplex_direct")

    t_start = time.perf_counter()
    root = float("nan")
    for it in range(1, max_iters + 1):
        m = pyo.ConcreteModel()
        m.T = pyo.RangeSet(0, T - 1)
        m.Jset = pyo.RangeSet(0, len(pool) - 1)
        m.lam = pyo.Var(m.Jset, within=pyo.NonNegativeReals)
        m.Yout = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, Q))
        m.Yret = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, Q))
        m.eOut = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.eRet = pyo.Var(m.T, within=pyo.NonNegativeReals)
        m.theta = pyo.Var(within=pyo.NonNegativeReals)

        c_out = {t: [] for t in range(T)}
        c_ret = {t: [] for t in range(T)}
        for j, (o, r) in enumerate(pool):
            for t in o:
                c_out[t].append(j)
            for t in r:
                c_ret[t].append(j)

        m.fleet = pyo.Constraint(expr=sum(m.lam[j] for j in m.Jset) == Q)
        m.agg_out = pyo.Constraint(
            m.T, rule=lambda mm, t: mm.Yout[t] - sum(mm.lam[j] for j in c_out[t]) == 0
        )
        m.agg_ret = pyo.Constraint(
            m.T, rule=lambda mm, t: mm.Yret[t] - sum(mm.lam[j] for j in c_ret[t]) == 0
        )
        m.conc_out = pyo.Constraint(m.T, rule=lambda mm, t: mm.eOut[t] >= mm.Yout[t] - 1)
        m.conc_ret = pyo.Constraint(m.T, rule=lambda mm, t: mm.eRet[t] >= mm.Yret[t] - 1)
        m.obj = pyo.Objective(
            expr=m.theta
            + eps_start * sum(m.Yout[t] + m.Yret[t] for t in m.T)
            + conc_pen * sum(m.eOut[t] + m.eRet[t] for t in m.T),
            sense=pyo.minimize,
        )
        attach_minute_recourse(
            m, None, delta, S, wmax, p_minutes,
            policy=policy, objective_scale=1.0 / float(delta), scenarios=scen,
        )
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        rmp_opt.solve(m, tee=False)
        root = float(pyo.value(m.obj))

        pi_out = {t: float(m.dual.get(m.agg_out[t], 0.0)) for t in range(T)}
        pi_ret = {t: float(m.dual.get(m.agg_ret[t], 0.0)) for t in range(T)}
        pi_fleet = float(m.dual.get(m.fleet, 0.0))

        # lambda_j has no objective cost of its own; it appears in agg rows with
        # coefficient -y^j and in the fleet row with coefficient 1. So
        #   reduced cost = sum_t pi_out[t]*y^j_out[t] + sum_t pi_ret[t]*y^j_ret[t] - pi_fleet
        for t in range(T):
            pm.m.price_coef_out[t] = pi_out[t]
            pm.m.price_coef_ret[t] = pi_ret[t]
        price_opt.solve(pm.m, tee=False)
        rc = float(pyo.value(pm.m.price_obj)) - pi_fleet
        col = _extract_column(pm.m, T)

        if verbose:
            print(f"  it {it:3d}  root {root:12.4f}  |pool| {len(pool):5d}  "
                  f"min rc {rc:+12.6f}")

        if rc >= -1e-7:
            break
        if time_limit_s is not None and time.perf_counter() - t_start > time_limit_s:
            print(f"  TIME LIMIT {time_limit_s:.0f}s with rc={rc:.3e} still negative. "
                  "The value above is an upper bound on the root, NOT the root.")
            break
        if col in pool:
            # The pricer returned a column already priced in. Continuing would loop
            # forever reporting progress; stopping silently would report a root that is
            # not one. Neither is acceptable, so say so.
            print(f"  STALLED: pricer returned an existing column with rc={rc:.3e}. "
                  "The root below is NOT proved optimal.")
            break
        pool.append(col)
    else:
        print(f"  hit max_iters={max_iters} with rc still negative; root NOT proved.")

    return root, len(pool), time.perf_counter() - t_start


def monolith(config: str, p_minutes: float, policy: str, time_limit_s: float,
             ) -> tuple[float, float, float, str]:
    """The optimum of the SAME model both roots relax: binaries intact, theta pinned.

    This is the denominator. Without it a lift is a number with no scale -- D58 could
    report +32.71 at Q=3 and not what fraction of the gap that is, because the only
    optimum on record for that instance (1658.86, D54) is at `p = 750` passenger-minutes
    and belongs to a different regime (D50, D53). Returns (objective, bound, seconds,
    status), and the caller must not read the objective as an optimum unless the status
    says so.
    """
    import pyomo.environ as pyo
    from mobauto2_milp.config import load_config
    from mobauto2_milp.app import _prepare_params
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_benders.minute_pricer import attach_minute_recourse, load_request_minutes

    cfg = load_config(config)
    mp, _sp = _prepare_params(cfg, {})
    mp = dict(mp)
    mp["solve_time_limit_s"] = float(time_limit_s)
    delta = int(cfg.model.time.slot_resolution)
    pm = MobautoMilpModel(mp)
    pm.initialize()
    attach_minute_recourse(
        pm.m, None, delta, float(cfg.service.S), float(cfg.service.Wmax_minutes),
        p_minutes, policy=policy, objective_scale=1.0 / float(delta),
        scenarios=_scenarios(cfg, load_request_minutes),
    )
    opt = pyo.SolverFactory("cplex_direct")
    opt.options["mip_tolerances_mipgap"] = 1e-6
    # The cap has to reach the SOLVER. Putting it in `mp` only reaches
    # `MobautoMilpModel.solve()`, which this function deliberately does not call -- so
    # for one run the limit was accepted, stored, and silently inert, and a Q=4 solve
    # would have run until something else stopped it.
    opt.options["timelimit"] = float(time_limit_s)
    t0 = time.perf_counter()
    res = opt.solve(pm.m, tee=False)
    secs = time.perf_counter() - t0
    obj = float(pyo.value(pm.m.obj))
    try:
        bound = float(res.problem.lower_bound)
    except Exception:
        bound = float("nan")
    return obj, bound, secs, str(res.solver.termination_condition)


def cplex_root_bound(config: str, p_minutes: float, policy: str, time_limit_s: float,
                     ) -> tuple[float, float]:
    """CPLEX's OWN root bound: node limit 0, so presolve and root cuts run and nothing else.

    This is the comparison the D59 question should have asked. "Does the DW root exceed
    the monolith's bound after 1200 s?" pits a root against twenty minutes of tree search
    and is unfair in the direction that makes the reformulation look bad, exactly as the
    earlier slot-monolith comparisons were unfair in the direction that made it look good.

    A root is comparable to a root. If the DW root beats what CPLEX reaches at ITS root
    after its own cuts, a branch-and-price tree starts ahead of a branch-and-cut tree, and
    that is a claim about relaxations that survives. If it does not, the reformulation is
    weaker than cuts CPLEX already applies for free.
    """
    import pyomo.environ as pyo
    from mobauto2_milp.config import load_config
    from mobauto2_milp.app import _prepare_params
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_benders.minute_pricer import attach_minute_recourse, load_request_minutes

    cfg = load_config(config)
    mp, _sp = _prepare_params(cfg, {})
    delta = int(cfg.model.time.slot_resolution)
    pm = MobautoMilpModel(dict(mp))
    pm.initialize()
    attach_minute_recourse(
        pm.m, None, delta, float(cfg.service.S), float(cfg.service.Wmax_minutes),
        p_minutes, policy=policy, objective_scale=1.0 / float(delta),
        scenarios=_scenarios(cfg, load_request_minutes),
    )
    opt = pyo.SolverFactory("cplex_direct")
    opt.options["mip_limits_nodes"] = 0
    opt.options["timelimit"] = float(time_limit_s)
    t0 = time.perf_counter()
    # `load_solutions=False` is required, not tidiness: at node limit 0 CPLEX may finish
    # with no incumbent at all, and pyomo raises "bad status: error" trying to load a
    # solution that does not exist. The bound is what is wanted here and it survives on
    # the results object either way.
    res = opt.solve(pm.m, tee=False, load_solutions=False)
    secs = time.perf_counter() - t0
    try:
        bound = float(res.problem.lower_bound)
    except Exception:
        bound = float("nan")
    return bound, secs


def compact_root(config: str, p_minutes: float, policy: str) -> tuple[float, float]:
    """The control: the master's own formulation with its binaries relaxed.

    Solved through pyomo directly, not through `MobautoMilpModel.solve()`, which refuses
    a non-binary solution -- the right guard for its job and exactly wrong here, where a
    fractional answer is the measurement.
    """
    import pyomo.environ as pyo
    from mobauto2_milp.config import load_config
    from mobauto2_milp.app import _prepare_params
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_benders.minute_pricer import attach_minute_recourse, load_request_minutes

    cfg = load_config(config)
    mp, _sp = _prepare_params(cfg, {})
    delta = int(cfg.model.time.slot_resolution)
    pmaster = MobautoMilpModel(dict(mp))
    pmaster.initialize()
    attach_minute_recourse(
        pmaster.m, None, delta, float(cfg.service.S), float(cfg.service.Wmax_minutes),
        p_minutes, policy=policy, objective_scale=1.0 / float(delta),
        scenarios=_scenarios(cfg, load_request_minutes),
    )
    for v in pmaster.m.component_data_objects(pyo.Var, active=True, descend_into=True):
        if not v.fixed and v.domain is pyo.Binary:
            v.domain = pyo.NonNegativeReals
            v.setlb(0.0)
            v.setub(1.0)
    t0 = time.perf_counter()
    pyo.SolverFactory("cplex_direct").solve(pmaster.m, tee=False)
    return float(pyo.value(pmaster.m.obj)), time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=VALIDATE_CONFIG)
    ap.add_argument("--policy", default="start")
    ap.add_argument("--p-minutes", type=float, default=56.0)
    ap.add_argument("--max-iters", type=int, default=2000)
    ap.add_argument("--time-limit", type=float, default=None,
                    help="Wall-clock cap on the CG loop. On expiry the value reported "
                         "is an upper bound on the root, not the root.")
    ap.add_argument("--compact", action="store_true",
                    help="Also solve the compact LP root, the control the lift is "
                         "measured against.")
    ap.add_argument("--validate", action="store_true",
                    help="Run at Q=2/T=22 and require the enumerated root.")
    ap.add_argument("--monolith", type=float, default=None, metavar="SECONDS",
                    help="Also solve the same model with binaries intact, to get the "
                         "optimum the lift is a fraction of. Argument is the time cap.")
    ap.add_argument("--monolith-only", action="store_true",
                    help="Solve only the monolith and stop.")
    ap.add_argument("--cplex-root", type=float, default=None, metavar="SECONDS",
                    help="Also report CPLEX's own root bound (node limit 0). The "
                         "root-vs-root comparison; see cplex_root_bound.")
    args = ap.parse_args()

    if args.monolith_only:
        if args.monolith is None:
            args.monolith = 1800.0
        obj, bound, secs, status = monolith(
            args.config, args.p_minutes, args.policy, args.monolith
        )
        proven = status.lower() in ("optimal", "globallyoptimal")
        print(f"\nmonolith  objective {obj:12.4f}   bound {bound:12.4f}   "
              f"{status}   {secs:.1f}s")
        if not proven:
            print("NOT PROVEN OPTIMAL -- this is an incumbent, and the bound beside it is")
            print("what the solve actually established. Do not quote it as an optimum.")
        return 0 if proven else 1

    print("=" * 78)
    print("Stage 3 -- DW root by column generation (pricing = the master at Q=1)")
    print(f"  {args.config}   policy={args.policy}   p_minutes={args.p_minutes:.0f}")
    print("=" * 78)

    if args.cplex_root is not None:
        b, t = cplex_root_bound(args.config, args.p_minutes, args.policy, args.cplex_root)
        print(f"\nCPLEX root bound (node limit 0, its own cuts)  {b:12.4f}   {t:.1f}s")
        if not args.compact:
            return 0

    z_compact = None
    if args.compact:
        z_compact, t_c = compact_root(args.config, args.p_minutes, args.policy)
        print(f"\nA  compact LP root   {z_compact:12.4f}   ({t_c:.1f}s)")

    root, ncols, secs = solve_root(
        args.config, args.p_minutes, args.policy, args.max_iters,
        time_limit_s=args.time_limit,
    )
    print("-" * 78)
    print(f"B  DW root by CG     {root:12.4f}   columns generated {ncols}   {secs:.1f}s")
    if z_compact is not None:
        if root < z_compact - 1e-6:
            print("\nB BELOW A. Impossible for two relaxations of the same problem:")
            print("the pricing feasible set is more permissive than the master's, or a")
            print("dual sign is wrong. Do not read these numbers.")
            return 3
        print(f"\nlift                 {root - z_compact:+12.4f}")

    if args.validate or args.config == VALIDATE_CONFIG:
        ok = abs(root - ENUMERATED_ROOT) <= ENUMERATED_TOL
        print(f"\nvalidation vs enumerated root {ENUMERATED_ROOT:.4f} (D57): "
              f"{'OK' if ok else 'MISMATCH'}  (delta {root - ENUMERATED_ROOT:+.6f})")
        if not ok:
            print("A column generation loop converges confidently to the wrong number")
            print("when a dual sign, a constant, or the pricing feasible set is wrong.")
            print("Do not point this at an instance that cannot be enumerated.")
            return 1
        print(f"\n{ncols} generated columns reproduce what 159 768 enumerated ones give.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
