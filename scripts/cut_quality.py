"""P2 (audit 4.4). Cut generators compared on one common set of signatures.

    python scripts/cut_quality.py [--config configs/baseline_d9_p56.yaml]
                                  [--samples 12] [--out FILE.json]

WHAT IS COMPARED, AND WHAT IS NOT. The audit asks for plain dual, Magnanti-Wong,
multi-cut, normalised and combinatorial cuts. Three of those exist in this repository and
two do not:

    dual               plain capacity duals                     exists
    mw                 Magnanti-Wong-inspired selection (B14)   exists
    finite_difference  finite-difference slopes                 exists, NOT a lower bound
    multi-cut          one cut per (scenario, direction)         exists as an ARCHITECTURE
                                                                 (subproblem.cut_architecture),
                                                                 not as a generator
    normalised         --                                        NOT IMPLEMENTED
    combinatorial      --                                        NOT IMPLEMENTED

The two missing families are reported as missing rather than approximated. A row labelled
"normalised" produced by rescaling a dual cut would measure the rescaling, not the family.

`finite_difference` is included deliberately even though it carries no lower-bound
guarantee: the whole reason it is still in the codebase is as the ablation baseline, and
a quality table that omits it cannot show what the guarantee costs.

THE METRICS, and what each one is for.

    violation      how much the cut cuts off at the evaluation point,
                   `cut(Y) - theta_lb(Y)`. Larger is better, but it is not
                   scale-free -- a cut with big coefficients scores big.
    efficacy       violation / ||a||_2, the Euclidean distance from the point to the
                   cut hyperplane. This is the scale-free version and is the one to
                   read when the generators disagree about magnitude.
    orthogonality  1 - max |cos| against the other generators' cuts at the same
                   point. Near 0 means the generators are producing the same
                   hyperplane and the choice between them is cosmetic.
    density        fraction of nonzero coefficients. A denser cut costs more per
                   master solve for the same bound.
    tightness      |cut(Y_gen) - Q(Y_gen)| at the point the cut was GENERATED at.
                   Must be ~0 for any valid Benders cut; printed because a nonzero
                   value here invalidates every other column in the row.
    gen_seconds    wall time to produce the cut.

SIGNATURES. Both integral (incumbent-like) and FRACTIONAL points are sampled, because the
LP phase evaluates the recourse at fractional `y` and a generator that behaves well only
at integer points would be judged on the wrong half of the run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

GENERATORS = ("dual", "mw", "finite_difference")
NOT_IMPLEMENTED = {
    "normalised": "no normalised-cut family exists in this repository",
    "combinatorial": "no combinatorial-cut family exists in this repository",
}


def _candidate(Y_out, Y_ret, Q: int) -> dict:
    """A per-vehicle schedule realising a signature, fractional entries included.

    A fractional `Y_d[tau] = 1.4` is spread as 1.0 on vehicle 0 and 0.4 on vehicle 1 --
    which is what the master's LP relaxation actually hands over, and what the recourse
    reads through `sum_q y_d[q,tau]`.
    """
    horizon = len(Y_out)
    cand: dict[str, float] = {}
    for q in range(Q):
        for t in range(horizon):
            cand[f"yOUT[{q},{t}]"] = 0.0
            cand[f"yRET[{q},{t}]"] = 0.0
    for key, vec in (("yOUT", Y_out), ("yRET", Y_ret)):
        for t in range(horizon):
            left = float(vec[t])
            for q in range(Q):
                take = min(1.0, max(0.0, left))
                cand[f"{key}[{q},{t}]"] = take
                left -= take
    return cand


def _coeff_vector(meta: dict, T: int, Q: int) -> list[float]:
    """The cut's slopes as one dense vector, so norms and angles are comparable."""
    vec = [0.0] * (2 * Q * T)
    for i, key in enumerate(("coeff_yOUT", "coeff_yRET")):
        for (q, tau), v in dict(meta.get(key) or {}).items():
            idx = i * Q * T + int(q) * T + int(tau)
            if 0 <= idx < len(vec):
                vec[idx] = float(v)
    return vec


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(_dot(a, a))


def _cut_at(meta: dict, cand: dict) -> float:
    total = float(meta["const"])
    for key, prefix in (("coeff_yOUT", "yOUT"), ("coeff_yRET", "yRET")):
        for (q, tau), v in dict(meta.get(key) or {}).items():
            total += float(v) * cand.get(f"{prefix}[{int(q)},{int(tau)}]", 0.0)
    return total


def _evaluate(params: dict, cand: dict):
    from mobauto2_benders.problem.subproblem_impl import ProblemSubproblem

    return ProblemSubproblem(dict(params)).evaluate(dict(cand))


def _sample_signatures(T: int, Q: int, trip_slots: int, n: int, seed: int):
    """Half integral, half fractional, all inside the master's own fixings."""
    from mobauto2_benders.signature import departures_are_possible

    rng = random.Random(seed)
    ok_out, ok_ret = departures_are_possible(T, trip_slots)
    out_slots = [t for t in range(T) if ok_out[t]]
    ret_slots = [t for t in range(T) if ok_ret[t]]
    samples = []
    for i in range(n):
        integral = i < n // 2
        Y_out = [0.0] * T
        Y_ret = [0.0] * T
        # Roughly one departure per trip window, so the signature is not absurd.
        for slots, vec in ((out_slots, Y_out), (ret_slots, Y_ret)):
            for t in slots:
                if rng.random() < 0.35:
                    v = rng.randint(1, Q) if integral else round(rng.uniform(0.2, Q), 3)
                    vec[t] = float(v)
        samples.append(("integral" if integral else "fractional", Y_out, Y_ret))
    return samples


def main() -> int:
    from mobauto2_benders.app import _prepare_params
    from mobauto2_benders.config import load_config

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/baseline_d9_p56.yaml")
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mp, sp = _prepare_params(cfg, {})
    sp = dict(sp)
    T = int(mp.get("T") or (int(mp["T_minutes"]) // int(mp["slot_resolution"])))
    sp["T"] = T
    Q = int(sp.get("Q", 2))
    trip_slots = max(
        1,
        int(round(float(cfg.model.time.trip_duration_minutes)
                  / float(cfg.model.time.slot_resolution))),
    )
    sp["mw_core_point"] = {"Yout": [0.5] * T, "Yret": [0.3] * T}

    print("P2 (audit 4.4) -- cut generators on a common set of signatures")
    print("=" * 88)
    print(f"config {args.config}  T={T} Q={Q} p={sp['p']:.4f} S={sp['S']:g}")
    print(f"generators compared: {', '.join(GENERATORS)}")
    for name, why in NOT_IMPLEMENTED.items():
        print(f"  NOT COMPARED -- {name}: {why}")
    print()

    samples = _sample_signatures(T, Q, trip_slots, args.samples, args.seed)
    rows: list[dict] = []

    header = (
        f"{'point':>10s} {'kind':>10s} | {'gen':>17s} | {'tight':>9s} | "
        f"{'violation':>10s} | {'efficacy':>9s} | {'orth':>6s} | {'dens':>5s} | "
        f"{'gen_s':>6s}"
    )
    print(header)
    print("-" * len(header))

    for idx, (kind, Y_out, Y_ret) in enumerate(samples):
        cand = _candidate(Y_out, Y_ret, Q)
        # The true recourse at the generation point, once, shared by every generator.
        base = dict(sp)
        base["cut_mode"] = "dual"
        try:
            true_val = float(_evaluate(base, cand).upper_bound)
        except Exception as exc:  # noqa: BLE001
            print(f"{idx:>10d} {kind:>10s} | recourse failed: {exc}")
            continue

        vectors: dict[str, list[float]] = {}
        metas: dict[str, dict] = {}
        timings: dict[str, float] = {}
        for gen in GENERATORS:
            params = dict(sp)
            params["cut_mode"] = gen
            params["use_magnanti_wong"] = gen == "mw"
            params["use_dual_slopes"] = gen in ("dual", "mw")
            if gen == "finite_difference":
                # It carries no lower-bound guarantee and the loop refuses to run it
                # without the acknowledgement. Included on purpose as the ablation.
                params["acknowledge_no_lower_bound"] = True
            t0 = time.perf_counter()
            try:
                res = _evaluate(params, cand)
            except Exception as exc:  # noqa: BLE001
                print(f"{idx:>10d} {kind:>10s} | {gen:>17s} | FAILED: {exc}")
                continue
            timings[gen] = time.perf_counter() - t0
            metas[gen] = (res.cuts or [res.cut])[0].metadata
            vectors[gen] = _coeff_vector(metas[gen], T, Q)

        for gen, meta in metas.items():
            vec = vectors[gen]
            nrm = _norm(vec)
            at_gen = _cut_at(meta, cand)
            tight = abs(at_gen - true_val)
            violation = at_gen  # against theta = 0, the master's initial relaxation
            efficacy = violation / nrm if nrm > 1e-12 else float("nan")
            others = [v for g, v in vectors.items() if g != gen and _norm(v) > 1e-12]
            if others and nrm > 1e-12:
                cos = max(abs(_dot(vec, o) / (nrm * _norm(o))) for o in others)
                orth = 1.0 - cos
            else:
                orth = float("nan")
            density = sum(1 for v in vec if abs(v) > 1e-12) / max(1, len(vec))
            rows.append(
                dict(point=idx, kind=kind, generator=gen,
                     recourse=true_val, cut_at_generation=at_gen, tightness=tight,
                     violation=violation, norm=nrm, efficacy=efficacy,
                     orthogonality=orth, density=density,
                     gen_seconds=timings.get(gen),
                     valid_lower_bound=bool(meta.get("cut_valid_lower_bound")))
            )
            print(
                f"{idx:>10d} {kind:>10s} | {gen:>17s} | {tight:9.2e} | "
                f"{violation:10.2f} | {efficacy:9.3f} | {orth:6.3f} | "
                f"{density:5.2f} | {timings.get(gen, float('nan')):6.2f}"
            )

    print()
    print("SUMMARY (means over sampled points, per generator)")
    print("-" * 88)
    print(
        f"{'generator':>17s} | {'tightness':>10s} | {'efficacy':>9s} | {'orth':>6s} | "
        f"{'density':>7s} | {'gen_s':>6s} | lower bound?"
    )
    summary = {}
    for gen in GENERATORS:
        sel = [r for r in rows if r["generator"] == gen]
        if not sel:
            continue

        def _mean(key):
            vals = [r[key] for r in sel if r[key] == r[key]]
            return sum(vals) / len(vals) if vals else float("nan")

        summary[gen] = {
            "n": len(sel),
            "tightness": _mean("tightness"),
            "efficacy": _mean("efficacy"),
            "orthogonality": _mean("orthogonality"),
            "density": _mean("density"),
            "gen_seconds": _mean("gen_seconds"),
            "valid_lower_bound": all(r["valid_lower_bound"] for r in sel),
        }
        s = summary[gen]
        print(
            f"{gen:>17s} | {s['tightness']:10.2e} | {s['efficacy']:9.3f} | "
            f"{s['orthogonality']:6.3f} | {s['density']:7.3f} | "
            f"{s['gen_seconds']:6.2f} | "
            + ("yes" if s["valid_lower_bound"] else "NO")
        )

    print()
    print("HOW TO READ THIS")
    print("  tightness must be ~0 for a Benders cut. A row with a nonzero tightness is")
    print("  not a weaker cut, it is a wrong one, and its other columns mean nothing.")
    print("  efficacy is the scale-free comparison; violation is not, so a generator")
    print("  with larger coefficients wins on violation without cutting off more.")
    print("  orth near 0 means the generators are returning the same hyperplane, and")
    print("  the choice between them is cosmetic on this instance.")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"rows": rows, "summary": summary,
                        "not_implemented": NOT_IMPLEMENTED}, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
