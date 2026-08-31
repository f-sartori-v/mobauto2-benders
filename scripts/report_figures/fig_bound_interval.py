"""Where the decomposition fails: the bound interval on the Q = 3 instance.

    python scripts/report_figures/fig_bound_interval.py

The decomposition finds a schedule within 2 % of the optimum almost immediately
and cannot certify it. Plotting the two bounds against the certified optimum
shows which side of the interval is responsible: the picture is asymmetric, and
the whole of the remaining gap is on the lower side.

Values come from data/measurements.json, which names the decision entry that
recorded each of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _style import (  # noqa: E402
    ACCENT,
    LIGHT,
    NEUTRAL,
    OUT_COLOUR,
    RET_COLOUR,
    TEXT_WIDTH_IN,
    apply_style,
    default_outdir,
    save,
    value_grid,
)

import matplotlib.pyplot as plt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(Path(__file__).parent / "data" / "measurements.json"))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--stem", default="bound_interval")
    args = ap.parse_args()

    d = json.loads(Path(args.data).read_text(encoding="utf-8"))["bound_interval"]
    lb, ub, opt = d["lower_bound"], d["upper_bound"], d["optimum"]

    apply_style()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 1.95))

    # Two segments, split at the certified optimum: the left one is the part of
    # the interval the decomposition never closes, the right one is how far its
    # best schedule sits above the optimum.
    ax.barh([0], [opt - lb], left=[lb], height=0.30, color=ACCENT, alpha=0.30,
            edgecolor=ACCENT, linewidth=0.8, zorder=2)
    ax.barh([0], [ub - opt], left=[opt], height=0.30, color=OUT_COLOUR, alpha=0.35,
            edgecolor=OUT_COLOUR, linewidth=0.8, zorder=2)
    ax.plot([lb], [0], marker="|", ms=15, mew=2.0, color=ACCENT, zorder=3)
    ax.plot([ub], [0], marker="|", ms=15, mew=2.0, color=OUT_COLOUR, zorder=3)
    ax.axvline(opt, color=NEUTRAL, lw=1.1, ls=(0, (4, 2)), zorder=4)

    ax.annotate(f"lower bound {lb:.2f}", (lb, 0), xytext=(0, -14),
                textcoords="offset points", ha="center", va="top",
                color=ACCENT, fontsize=7.5)
    ax.annotate(f"{100 * (opt - lb) / opt:.1f} % of the optimum, never closed",
                ((lb + opt) / 2, 0), xytext=(0, 16), textcoords="offset points",
                ha="center", va="bottom", color=ACCENT, fontsize=7.5)
    ax.annotate(f"upper bound {ub:.2f}\n{100 * (ub - opt) / opt:.1f} % above",
                (ub, 0), xytext=(6, -6), textcoords="offset points",
                ha="left", va="top", color=OUT_COLOUR, fontsize=7.5)
    ax.annotate(f"optimum {opt:.2f}", (opt, 0), xytext=(-5, 31),
                textcoords="offset points", ha="right", va="bottom",
                color=NEUTRAL, fontsize=7.5)

    ax.set_yticks([])
    ax.set_ylim(-0.7, 0.85)
    ax.set_xlim(lb - 14, ub + 34)
    ax.set_xticks([220, 240, 260, 280, 300])
    ax.set_xlabel("objective, passenger-minutes")
    ax.spines["left"].set_visible(False)
    value_grid(ax, axis="x")
    ax.set_title(
        f"{d['instance']}: {d['iterations']} iterations in "
        f"{d['decomposition_seconds']:.0f} s,\nagainst {d['monolith_seconds']:.1f} s "
        f"to proven optimality without the decomposition",
        loc="left", color="#444444", fontsize=7.8,
    )

    outdir = Path(args.outdir) if args.outdir else default_outdir()
    save(fig, outdir, args.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
