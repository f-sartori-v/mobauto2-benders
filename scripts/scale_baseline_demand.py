"""Scale the baseline demand to the trial's declared service level: 450 passengers/day.

    python scripts/scale_baseline_demand.py [--out setups/base_scale450.yaml] [--n 450]

THE QUESTION (forward-plan A3, docs/FORWARD_PLAN_v1.md). The trial's a-priori safety
evaluation (deliverable 1.4.3, report Section intro-corridor) declares a service level of
"up to 450 passengers per day over up to 30 trips per day." Every instance reported in this
project so far runs 300-400 requests (setups/base.yaml and its perturbations) -- the
project's own declared target scale has never been exercised.

METHOD. Bootstrap resampling, not a new generative shape: draw arrival minutes WITH
REPLACEMENT from setups/base.yaml's own 150-per-direction empirical distribution, seeded,
until each direction reaches n/2. This is deliberate -- it reproduces the exact documented
profile (the 08:00 OUT peak, the 16:00 RET peak, the midday valley, report Section
res-instances) at the new volume, rather than asking a reader to trust that a re-parameterised
Gaussian mixture matches it. It is not scripts/make_instances.py's shape sweep (which varies
TEMPORAL STRUCTURE at fixed volume, for a different question) -- this varies VOLUME at fixed
structure, for this one.

Deterministic: a fixed seed, so the instance regenerates byte-identically.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "setups" / "base.yaml"
SEED = 450


def _load_source() -> dict[str, list[int]]:
    """Minimal YAML reading, matching scripts/report_figures/_style.py's load_requests --
    the setup files are a flat list, so this avoids a PyYAML dependency for a one-off
    generation script."""
    out: dict[str, list[int]] = {"OUT": [], "RET": []}
    direction: str | None = None
    for raw in SOURCE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("- dir:"):
            direction = line.split(":", 1)[1].strip()
        elif line.startswith("time:") and direction is not None:
            out[direction].append(int(line.split(":", 1)[1].strip()))
            direction = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "setups" / "base_scale450.yaml"))
    ap.add_argument("--n", type=int, default=450, help="Total requests; split evenly by direction.")
    args = ap.parse_args()

    if args.n % 2 != 0:
        raise ValueError(f"--n must be even to split evenly OUT/RET, got {args.n}")
    per_direction = args.n // 2

    source = _load_source()
    if not source["OUT"] or not source["RET"]:
        raise RuntimeError(f"{SOURCE} did not parse -- check its format has not changed")

    rng = random.Random(SEED)
    out_times = sorted(rng.choices(source["OUT"], k=per_direction))
    ret_times = sorted(rng.choices(source["RET"], k=per_direction))

    lines = [f"n: {args.n}", "requests:"]
    # Interleave by time so the file reads like base.yaml's own chronological layout,
    # rather than all OUT followed by all RET.
    combined = sorted(
        [("OUT", t) for t in out_times] + [("RET", t) for t in ret_times],
        key=lambda r: r[1],
    )
    for d, t in combined:
        lines.append(f"- dir: {d}")
        lines.append(f"  time: {t}")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}: {args.n} requests ({per_direction} OUT, {per_direction} RET), "
          f"resampled from {SOURCE.name}, seed={SEED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
