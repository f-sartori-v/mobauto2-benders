"""Stage 2, step 1: is the down-set cut valid, and is it tighter than the Benders cut?

    python scripts/stage2_downset_probe.py

OFFLINE ONLY. Nothing here touches the master or the Benders loop. `DESIGN_DD_v1.md`
stage 2 is explicit that the cut must be built and priced against the true recourse
before any integration, because this repository has four times reasoned its way into a
cut that looked stronger on paper and made the master worse (spec 2.9, D40).

THE PROPERTY (design section 3.2, P3). The recourse sees the schedule only as the
capacity right-hand side `C_d[tau] = S * Y_d[tau]`, so raising `Y` can only enlarge the
recourse's feasible set. Hence for integer signatures

    Y <= Yhat componentwise   =>   Q(Y) >= Q(Yhat)

One LP solve at `Yhat` is therefore a statement about an entire DOWN-SET of the lattice,
which is the logic-based Benders shape in Hooker section 6.2: a cut valid over a class
because of a proof, not over a neighbourhood because of a subgradient.

THE ENCODING, AND WHY IT MAY BE CHEAPER THAN THE DESIGN FEARED. `DESIGN_DD_v1.md`
assumed a down-set indicator needs an auxiliary BINARY per `(d, tau)`. It does not. With
`v_d[tau] >= Y_d[tau] - Yhat_d[tau]`, `v >= 0` continuous, and `M = Q(Yhat)`:

    theta >= Q(Yhat) - M * sum_{d,tau} v_d[tau]

is valid for every INTEGER `Y`. Inside the down-set every `v` can be 0 and the cut reads
`theta >= Q(Yhat)`, which P3 justifies. Outside it, some component exceeds `Yhat` by at
least 1 (integrality), so `sum v >= 1`, the right-hand side falls to `<= 0`, and the row
is slack because `theta >= 0`. Continuous auxiliaries, no binaries, no big-M tuning --
`M = Q(Yhat)` is exactly large enough.

That argument is why this script exists rather than a paragraph asserting it. The claim
is checked numerically at sampled integer points on both sides of the boundary.

WHAT IS BEING COMPARED. At the same anchor, the classical Benders optimality cut is

    theta >= Q(Yhat) + sum_{d,tau} S * pi_d[tau] * (Y_d[tau] - Yhat_d[tau])

Both are valid. The question is where each dominates: the Benders cut is the tightest
convex underestimator and is strong near `Yhat` in every direction; the down-set cut is
flat at `Q(Yhat)` over the whole down-set and worthless outside it. If the down-set cut
never exceeds the Benders cut at any sampled point, stage 2 is dead and should be
recorded as dead rather than integrated.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

CONFIG = "configs/phase1/rq5_benders_minute_p56.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors", type=int, default=3)
    ap.add_argument("--samples", type=int, default=40, help="Points per anchor, per side.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from mobauto2_benders.config import load_config
    from mobauto2_benders.minute_pricer import load_request_minutes, solve_minute_recourse

    cfg = load_config(CONFIG)
    T = int(cfg.model.time.T_minutes) // int(cfg.model.time.slot_resolution)
    delta = int(cfg.model.time.slot_resolution)
    Q = int(cfg.model.fleet.Q)
    S = float(cfg.subproblem.S)
    wmax = float(cfg.subproblem.Wmax_minutes)
    p_slots = float(cfg.subproblem.p)
    policy = str(cfg.subproblem.departure_policy)
    requests = load_request_minutes(cfg.data.demand_file)

    rng = random.Random(args.seed)

    def recourse(Y_out, Y_ret):
        """Q(Y) and its duals. Y enters ONLY as the capacity right-hand side."""
        duals, obj = solve_minute_recourse(
            T, delta, wmax, p_slots,
            [S * float(v) for v in Y_out],
            [S * float(v) for v in Y_ret],
            requests, policy=policy,
        )
        return float(obj), duals

    print("=" * 78)
    print("Stage 2 probe -- down-set cut validity and tightness")
    print(f"  T={T} slots of {delta}min   Q={Q}   S={S:.0f}   Wmax={wmax:.0f}min   "
          f"policy={policy}   p_slots={p_slots:.4f}")
    print("=" * 78)

    worst_p3 = None          # most negative Q(Y) - Q(Yhat) over the down-set
    worst_downset_slack = None   # most negative Q(Y) - cut(Y), any Y
    tighter = 0
    compared = 0

    for a in range(args.anchors):
        # Anchors are drawn over the whole lattice, not around one schedule: a cut form
        # that only behaves near a good solution is not a cut form.
        Yhat_out = [rng.randint(0, Q) for _ in range(T)]
        Yhat_ret = [rng.randint(0, Q) for _ in range(T)]
        q_hat, duals = recourse(Yhat_out, Yhat_ret)
        M = q_hat

        # `solve_minute_recourse` returns one dual per DEPARTURE SLOT under these exact
        # keys -- that shape is the whole reason a minute recourse drops into the
        # existing cut unchanged. Fail loudly if it ever stops being so, rather than
        # silently comparing against an all-zero Benders cut that always loses.
        if "pi_OUT" not in duals or "pi_RET" not in duals:
            raise KeyError(
                f"expected per-slot duals pi_OUT/pi_RET, got keys {sorted(duals)}"
            )
        pi_out = duals["pi_OUT"]
        pi_ret = duals["pi_RET"]

        def benders_cut(Y_out, Y_ret) -> float:
            val = q_hat
            for t in range(T):
                val += S * float(pi_out.get(t, 0.0)) * (Y_out[t] - Yhat_out[t])
                val += S * float(pi_ret.get(t, 0.0)) * (Y_ret[t] - Yhat_ret[t])
            return val

        def downset_cut(Y_out, Y_ret) -> float:
            excess = sum(max(0, Y_out[t] - Yhat_out[t]) for t in range(T))
            excess += sum(max(0, Y_ret[t] - Yhat_ret[t]) for t in range(T))
            return q_hat - M * excess

        print(f"\nanchor {a}:  Q(Yhat) = {q_hat:.4f}   "
              f"sum Yhat = {sum(Yhat_out) + sum(Yhat_ret)}")

        for side in ("down-set", "outside"):
            for _ in range(args.samples):
                if side == "down-set":
                    Y_out = [rng.randint(0, v) for v in Yhat_out]
                    Y_ret = [rng.randint(0, v) for v in Yhat_ret]
                else:
                    Y_out = [rng.randint(0, Q) for _ in range(T)]
                    Y_ret = [rng.randint(0, Q) for _ in range(T)]
                    if all(Y_out[t] <= Yhat_out[t] for t in range(T)) and all(
                        Y_ret[t] <= Yhat_ret[t] for t in range(T)
                    ):
                        continue  # landed inside; counted on the other pass

                q_y, _ = recourse(Y_out, Y_ret)

                if side == "down-set":
                    gap_p3 = q_y - q_hat
                    if worst_p3 is None or gap_p3 < worst_p3:
                        worst_p3 = gap_p3

                d_val = downset_cut(Y_out, Y_ret)
                b_val = benders_cut(Y_out, Y_ret)
                slack = q_y - d_val
                if worst_downset_slack is None or slack < worst_downset_slack:
                    worst_downset_slack = slack
                compared += 1
                if d_val > b_val + 1e-9:
                    tighter += 1

    print("\n" + "-" * 78)
    print(f"P3 monotonicity   min over sampled down-set points of Q(Y) - Q(Yhat): "
          f"{worst_p3:+.6f}")
    print("                  must be >= 0. A negative value refutes the whole design")
    print("                  section 3.2 and stage 2 stops here.")
    print(f"cut validity      min over ALL sampled points of Q(Y) - cut(Y): "
          f"{worst_downset_slack:+.6f}")
    print("                  must be >= 0, or the cut removes feasible integer points.")
    print(f"tightness         down-set cut beat the Benders cut at "
          f"{tighter}/{compared} sampled points "
          f"({100.0 * tighter / max(compared, 1):.1f}%)")
    print("-" * 78)
    if worst_p3 is not None and worst_p3 < -1e-6:
        print("VERDICT: P3 REFUTED. Do not integrate. Record it.")
        return 1
    if worst_downset_slack is not None and worst_downset_slack < -1e-6:
        print("VERDICT: the cut is INVALID as encoded. Do not integrate. Record it.")
        return 1
    if tighter == 0:
        print("VERDICT: valid but never tighter on this sample. Stage 2 buys nothing")
        print("         here; record it as dead rather than integrating it.")
        return 0
    print("VERDICT: valid, and tighter somewhere. NOT yet a reason to integrate --")
    print("         the next measurement is how much LP root the encoding actually buys,")
    print("         measured SEPARATELY from stage 1 (both act on the root).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
