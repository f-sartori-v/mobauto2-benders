"""Stage 3, step 1: does the Dantzig-Wolfe reformulation move the LP root?

    python scripts/stage3_dw_root.py

This is the one question stage 3 exists to ask, and at the small instance it can be
answered EXACTLY -- every column enumerated, no column generation, no pricing problem,
and no dependence on the battery dominance rule `DESIGN_DD_v1.md` flags as unproven
(`scripts/stage3_size_enumeration.py`: 262 144 patterns before the battery filter).

THE CLAIM BEING TESTED, from the design:

    the LP relaxation of the reformulation optimises over conv(integer points of the
    per-vehicle polytope), whereas the current master's LP relaxation optimises over the
    per-vehicle polytope's own LP relaxation

so the reformulated root must be at least as high, and the question is whether it is
higher by enough to matter. D56 is why this is the only staged item worth building: the
decomposition already finds schedules within 2% of optimal and fails only at the bound,
and the bound lives at the fractional LP root (D40).

NO BENDERS CUTS ARE INVOLVED. Both arms pin `theta` to the exact minute recourse, so both
are relaxations of the SAME mixed-integer problem -- the one the monolith solved to
293.37 in D56 -- and the only difference between them is how the first stage is
described. Comparing two roots that carry different cut sets would measure the cuts.

    A  compact   the master's own formulation, binaries relaxed to [0,1]
    B  DW        lambda_j >= 0 over enumerated per-vehicle patterns, sum lambda_j = Q

BATTERY FEASIBILITY IS DECIDED BY GREEDY MAX-CHARGE, and that is exact for FEASIBILITY:
raising `c[t]` raises `b[t+1]`, which only helps C5 and `b >= 0`, and it appears with a
POSITIVE sign in the `charge_before_idle` bound on `c[t+1]`, so it only relaxes the next
slot too. The one thing it consumes is headroom `Emax - b`, which caps later charging
rather than violating anything. Note this argument settles feasibility only -- it is NOT
the optimisation dominance rule a pricing DP would need, which is a different claim about
continuations and remains unproven.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

MONOLITH_CONFIG = "configs/milp/baseline_d9_p56_monolith.yaml"
AT_L, AT_M = 0, 1


def enumerate_patterns(T: int, trip_slots: int, start_at_massy: bool):
    """Yield (out_slots, ret_slots) admitted by the location dynamics and fixings.

    Mirrors `stage3_size_enumeration.count_patterns` exactly; that script's histogram
    matching C(19, 2k) term by term is the check that this walk is the right one.
    """
    out: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    o_acc: list[int] = []
    r_acc: list[int] = []

    def walk(t: int, loc: int, arrive_at: int, dest: int) -> None:
        # Land first -- see the note in stage3_size_enumeration.count_patterns. Testing
        # the terminal condition before the arrival drops every pattern whose last trip
        # arrives exactly at T-1, and a pool missing columns raises the root.
        if arrive_at == t:
            loc, arrive_at, dest = dest, -1, -1
        if t == T - 1:
            if arrive_at < 0 and loc == AT_L:
                out.append((tuple(o_acc), tuple(r_acc)))
            return
        if arrive_at >= 0:
            walk(t + 1, loc, arrive_at, dest)
            return

        walk(t + 1, loc, -1, -1)  # idle

        if t >= 1 and t < T - trip_slots:
            if loc == AT_L and t < max(0, T - 2 * trip_slots):
                o_acc.append(t)
                walk(t + 1, AT_L, t + trip_slots, AT_M)
                o_acc.pop()
            elif loc == AT_M:
                r_acc.append(t)
                walk(t + 1, AT_M, t + trip_slots, AT_L)
                r_acc.pop()

    walk(0, AT_M if start_at_massy else AT_L, -1, -1)
    return out


def battery_feasible(
    out_slots, ret_slots, T: int, trip_slots: int, binit: float,
    Emax: float, L: float, delta_chg: float, start_at_massy: bool,
) -> bool:
    """Greedy max-charge forward pass. See the module docstring for why greedy is exact."""
    dep = {}
    for t in out_slots:
        dep[t] = "OUT"
    for t in ret_slots:
        dep[t] = "RET"

    # Location by slot, from the same recursion the enumerator used.
    atL = [False] * T
    loc = AT_M if start_at_massy else AT_L
    arrive_at, dest = -1, -1
    for t in range(T):
        if arrive_at == t:
            loc, arrive_at, dest = dest, -1, -1
        atL[t] = (arrive_at < 0) and (loc == AT_L)
        if t in dep:
            dest = AT_M if dep[t] == "OUT" else AT_L
            arrive_at = t + trip_slots

    b = float(binit)
    c_prev = 0.0
    atL_prev = False
    out_prev = 0.0
    for t in range(T):
        if t in dep and dep[t] == "OUT" and b < 2.0 * L - 1e-9:
            return False  # C5

        c_t = 0.0
        if t < T - 1 and atL[t] and t not in dep:
            allowed = 1.0
            if t >= 1 and atL_prev:
                # charge_before_idle: c[t] <= yOUT[t-1] + c[t-1] + 1 - atL[t-1]
                allowed = min(allowed, out_prev + c_prev)
            if allowed > 0.0 and delta_chg > 0.0:
                c_t = max(0.0, min(allowed, (Emax - b) / delta_chg))

        if t < T - 1:
            b = b - (L if t in dep else 0.0) + delta_chg * c_t
            if b < -1e-9:
                return False
            b = min(b, Emax)

        c_prev = c_t
        atL_prev = atL[t]
        out_prev = 1.0 if (t in dep and dep[t] == "OUT") else 0.0

    return True


def _relax_binaries(m) -> int:
    import pyomo.environ as pyo

    n = 0
    for v in m.component_data_objects(pyo.Var, active=True, descend_into=True):
        if v.fixed:
            continue
        if v.domain is pyo.Binary:
            v.domain = pyo.NonNegativeReals
            v.setlb(0.0)
            v.setub(1.0)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="midpoint")
    ap.add_argument("--p-minutes", type=float, default=56.0)
    args = ap.parse_args()

    import pyomo.environ as pyo
    from mobauto2_milp.config import load_config
    from mobauto2_milp.app import _prepare_params
    from mobauto2_milp.model import MobautoMilpModel
    from mobauto2_benders.minute_pricer import attach_minute_recourse, load_request_minutes

    cfg = load_config(MONOLITH_CONFIG)
    mp, _sp = _prepare_params(cfg, {})
    delta = int(cfg.model.time.slot_resolution)
    T = int(cfg.model.time.T_minutes) // delta
    trip_slots = int(cfg.model.time.trip_duration_minutes) // delta
    Q = int(cfg.model.fleet.Q)
    S = float(cfg.service.S)
    wmax = float(cfg.service.Wmax_minutes)
    requests = load_request_minutes(list(cfg.data.scenario_files)[0])
    scale = 1.0 / float(delta)

    Emax = float(mp["Emax"])
    L = float(mp["L"])
    delta_chg = float(mp["delta_chg"])
    binit = list(mp["binit"])
    actions = list(mp["initial_actions"])[:Q]
    eps_start = float(mp.get("start_cost_epsilon", 0.0) or 0.0)
    conc_pen = float(mp.get("concurrency_penalty", 0.0) or 0.0)

    print("=" * 78)
    print("Stage 3 -- does the Dantzig-Wolfe reformulation move the LP root?")
    print(f"  T={T}  trip_slots={trip_slots}  Q={Q}  Emax={Emax:.0f}  L={L:.0f}  "
          f"delta_chg={delta_chg:.1f}  binit={binit[:Q]}")
    print("=" * 78)

    if len(set(binit[:Q])) != 1 or len(set(actions)) != 1:
        print("ABORT: the fleet is not homogeneous (E4), so one column pool cannot")
        print(f"       serve every vehicle. binit={binit[:Q]} actions={actions}")
        return 2

    t0 = time.perf_counter()
    pats = enumerate_patterns(T, trip_slots, start_at_massy=(actions[0] == "RET"))
    t_enum = time.perf_counter() - t0
    J = [
        p for p in pats
        if battery_feasible(p[0], p[1], T, trip_slots, binit[0], Emax, L, delta_chg,
                            start_at_massy=(actions[0] == "RET"))
    ]
    t_filt = time.perf_counter() - t0
    print(f"\npatterns (location only): {len(pats):,}   [{t_enum:.1f}s]")
    print(f"columns after battery filter |J|: {len(J):,}   "
          f"({100.0 * len(J) / max(len(pats), 1):.1f}% survive)   [{t_filt:.1f}s]")

    # ---------- arm A: compact LP root ----------
    master_a = MobautoMilpModel(dict(mp))
    master_a.initialize()
    attach_minute_recourse(
        master_a.m, requests, delta, S, wmax, args.p_minutes,
        policy=args.policy, objective_scale=scale,
    )
    n_relaxed = _relax_binaries(master_a.m)
    # Solved through pyomo directly, NOT through `MobautoMilpModel.solve()`: that path
    # refuses a non-binary solution before returning one, which is the right guard for
    # its own job (a fractional schedule must never reach the subproblem as if it were a
    # schedule) and exactly wrong here, where a fractional answer is the measurement.
    t0 = time.perf_counter()
    opt_a = pyo.SolverFactory("cplex_direct")
    res_a = opt_a.solve(master_a.m, tee=False)
    t_a = time.perf_counter() - t0
    z_compact = float(pyo.value(master_a.m.obj))
    print(f"\nA  compact LP root   {z_compact:10.4f}   ({n_relaxed} binaries relaxed, "
          f"{t_a:.1f}s, {res_a.solver.termination_condition})")

    # ---------- arm B: DW LP root ----------
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T - 1)
    m.Jset = pyo.RangeSet(0, len(J) - 1)
    m.lam = pyo.Var(m.Jset, within=pyo.NonNegativeReals)
    m.Yout = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, Q))
    m.Yret = pyo.Var(m.T, within=pyo.NonNegativeReals, bounds=(0, Q))
    m.eOut = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.eRet = pyo.Var(m.T, within=pyo.NonNegativeReals)
    m.theta = pyo.Var(within=pyo.NonNegativeReals)

    cols_out: dict[int, list[int]] = {t: [] for t in range(T)}
    cols_ret: dict[int, list[int]] = {t: [] for t in range(T)}
    for j, (o, r) in enumerate(J):
        for t in o:
            cols_out[t].append(j)
        for t in r:
            cols_ret[t].append(j)

    m.fleet = pyo.Constraint(expr=sum(m.lam[j] for j in m.Jset) == Q)
    m.agg_out = pyo.Constraint(
        m.T, rule=lambda mm, t: mm.Yout[t] == sum(mm.lam[j] for j in cols_out[t])
    )
    m.agg_ret = pyo.Constraint(
        m.T, rule=lambda mm, t: mm.Yret[t] == sum(mm.lam[j] for j in cols_ret[t])
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
        m, requests, delta, S, wmax, args.p_minutes,
        policy=args.policy, objective_scale=scale,
    )

    t0 = time.perf_counter()
    opt = pyo.SolverFactory("cplex_direct")
    res_b = opt.solve(m, tee=False)
    t_b = time.perf_counter() - t0
    z_dw = float(pyo.value(m.obj))
    print(f"B  DW LP root        {z_dw:10.4f}   ({len(J):,} columns, {t_b:.1f}s, "
          f"{res_b.solver.termination_condition})")

    # ---------- validate the column pool against an independent expectation ----------
    # A root comparison is worthless if the pool is wrong, and a pool that is missing
    # columns produces a HIGHER root, i.e. it fails in the flattering direction. Forcing
    # lambda integer must reproduce the monolith's proven optimum exactly: same model,
    # same instance, expectation from D56 and not from anything computed here.
    optimum = 293.37  # D56, proven optimal in 0.8 s
    for j in m.Jset:
        m.lam[j].domain = pyo.NonNegativeIntegers
    t0 = time.perf_counter()
    res_i = opt.solve(m, tee=False)
    t_i = time.perf_counter() - t0
    z_int = float(pyo.value(m.obj))
    ok = abs(z_int - optimum) <= 0.01
    print(f"\npool check  DW with lambda integer -> {z_int:10.4f}   "
          f"(monolith {optimum:.2f})   {'OK' if ok else 'MISMATCH'}   [{t_i:.1f}s]")
    if not ok:
        print("ABORT: the enumerated pool does not reproduce the monolith's optimum, so")
        print("       it is not a description of the same feasible set. A pool missing")
        print("       columns raises the root -- the error flatters the result. Fix the")
        print("       enumeration before reading the roots above.")
        return 4

    # ---------- read it ----------
    print("\n" + "-" * 78)
    print(f"true optimum (D56)   {optimum:10.4f}")
    print(f"A closes             {100.0 * z_compact / optimum:6.1f}% of it")
    print(f"B closes             {100.0 * z_dw / optimum:6.1f}% of it")
    if z_dw < z_compact - 1e-6:
        print("\nB BELOW A. That cannot happen if both are relaxations of the same")
        print("problem -- the column pool is missing feasible patterns, or the two")
        print("first stages are not the same model. Do not read the numbers; fix this.")
        return 3
    lift = z_dw - z_compact
    print(f"\nlift                 {lift:+10.4f}  "
          f"({100.0 * lift / max(optimum - z_compact, 1e-9):.1f}% of the gap A leaves)")
    print("-" * 78)
    if lift < 1e-6:
        print("VERDICT: the reformulation does NOT move the root. Stage 3 is dead for")
        print("         the same reason stage 2 was, and column generation at the Q=3")
        print("         point would be building on nothing. Record it.")
    else:
        print("VERDICT: the root moves. Whether that is worth a pricing problem at the")
        print("         Q=3 point depends on how much of the gap it closes -- and the")
        print("         dominance rule still has to be proved before any DP relies on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
