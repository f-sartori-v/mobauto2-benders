"""The five generated demand shapes used in the resolution study.

    python scripts/report_figures/fig_demand_shapes.py [--slot 30]

The shapes differ in one dimension only: how much of the demand lives at a finer
grain than one slot. Total volume, fleet size and horizon are held fixed by
scripts/make_instances.py, so a difference between panels is a difference in
temporal structure and nothing else.

Each panel shows arrivals per minute-bin, with the slot boundaries drawn behind
them: a shape whose mass sits well inside a slot is one a slot average describes
badly, and those are the shapes where minute-level pricing is expected to buy
the most.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _style import (  # noqa: E402
    NEUTRAL,
    OUT_COLOUR,
    RET_COLOUR,
    TEXT_WIDTH_IN,
    apply_style,
    default_outdir,
    load_requests,
    repo_root,
    save,
    value_grid,
)

import matplotlib.pyplot as plt  # noqa: E402

ORDER = ("flat", "commuter", "bimodal", "burst", "spiky")
BLURB = {
    "flat": "arrivals uniform over the horizon; sub-slot position close to uniform",
    "commuter": "one broad OUT peak, one broad RET peak",
    "bimodal": "two narrower peaks per direction",
    "burst": "four windows of 12 min, each far narrower than a 30 min slot",
    "spiky": "many very short bursts, spaced 47 min, coprime with every grid used",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="setups/generated")
    ap.add_argument("--slot", type=int, default=30)
    ap.add_argument("--bin", type=int, default=5, help="width of a histogram bin, minutes")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--stem", default="demand_shapes")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_absolute():
        root = repo_root() / root

    apply_style()
    fig, axes = plt.subplots(len(ORDER), 1, figsize=(TEXT_WIDTH_IN, 5.9), sharex=True)
    fig.subplots_adjust(hspace=0.75)

    horizon = 0
    for shape in ORDER:
        rows = load_requests(root / f"{shape}.yaml")
        horizon = max(horizon, max(t for _, t in rows) + 1)

    for ax, shape in zip(axes, ORDER):
        rows = load_requests(root / f"{shape}.yaml")
        n_bins = -(-horizon // args.bin)
        out = Counter(t // args.bin for d, t in rows if d == "OUT")
        ret = Counter(t // args.bin for d, t in rows if d == "RET")
        x = [b * args.bin for b in range(n_bins)]
        ax.bar(x, [out.get(b, 0) for b in range(n_bins)], width=args.bin * 0.9,
               color=OUT_COLOUR, label="OUT", zorder=2)
        ax.bar(x, [-ret.get(b, 0) for b in range(n_bins)], width=args.bin * 0.9,
               color=RET_COLOUR, label="RET", zorder=2)
        for boundary in range(0, horizon + args.slot, args.slot):
            ax.axvline(boundary, color="#DCDCDC", lw=0.5, zorder=0)
        ax.axhline(0, color=NEUTRAL, lw=0.6, zorder=3)
        ax.set_title(f"{shape} — {BLURB[shape]}", loc="left", fontsize=7.5,
                     color="#555555", pad=3)
        ax.set_yticks([])
        ax.grid(False)
        for side in ("left", "top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    axes[0].legend(loc="lower right", ncol=2, fontsize=7.5,
                   bbox_to_anchor=(1.0, 1.02))
    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].set_xlabel("arrival minute")
    axes[-1].set_xlim(-5, horizon + 5)

    outdir = Path(args.outdir) if args.outdir else default_outdir()
    save(fig, outdir, args.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
