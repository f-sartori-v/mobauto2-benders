"""What minute-level pricing buys, by demand shape and by master resolution.

    python scripts/report_figures/fig_multiresolution_gain.py

The quantity plotted is a decision gain, and it is a comparison between two
schedules priced the same way. For one instance and one placement convention:

  A  solve the first stage with a slot recourse, then price the schedule it
     chose at the arrival minute;
  B  solve the same first stage with a minute recourse, then price the schedule
     it chose at the arrival minute;

     gain = (cost of A - cost of B) / cost of A.

Pricing both at minute fidelity is what makes the comparison fair to either: it
isolates the schedule that was chosen from the yardstick used to report it, so
none of the reporting error of Figure 3 leaks into this number.

The dashed reference is the same measurement when one schedule has to serve four
scenarios, which is the regime the project targets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _style import (  # noqa: E402
    ACCENT,
    NEUTRAL,
    TEXT_WIDTH_IN,
    apply_style,
    default_outdir,
    save,
    value_grid,
)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ORDER = ("flat", "commuter", "bimodal", "burst", "spiky")
SHADES = ("#7C9CC9", "#42679D", "#27406B")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(Path(__file__).parent / "data" / "measurements.json"))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--stem", default="multiresolution_gain")
    args = ap.parse_args()

    d = json.loads(Path(args.data).read_text(encoding="utf-8"))["multiresolution_gain"]
    resolutions = d["resolutions"]

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.7), sharey=True)

    # `start` (o=0) is the committed departure instant (D76) -- the only panel that
    # prices what the master's schedule actually costs. `midpoint` is kept beside it
    # as a labelled counterfactual ("what if this departure left mid-slot instead"),
    # never presented as a second measurement of the same thing.
    for ax, policy in zip(axes, ("start", "midpoint")):
        table = d[policy]
        width = 0.26
        for j, res in enumerate(resolutions):
            xs = [i + (j - 1) * width for i in range(len(ORDER))]
            ys = [table[shape][j] for shape in ORDER]
            ax.bar(xs, ys, width=width, color=SHADES[j], zorder=2,
                   label=f"{res} min master" if policy == "midpoint" else None)
        if policy != "start":
            ax.axhline(d["multi_scenario_gain"], color=ACCENT, lw=1.0,
                       ls=(0, (4, 2)), zorder=3)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(ORDER, rotation=30, ha="right")
        title = "start (committed instant)" if policy == "start" else "midpoint (counterfactual)"
        ax.set_title(title, loc="left", fontsize=8.5, color="#444444")
        value_grid(ax)
        ax.set_ylim(0, 55)

    axes[0].set_ylabel("decision gain, per cent")
    reference = Line2D([], [], color=ACCENT, lw=1.0, ls=(0, (4, 2)),
                       label=f"{d['multi_scenario_gain']:.2f} %, one schedule "
                             f"serving four scenarios")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles + [reference], labels + [reference.get_label()],
               loc="lower center", ncol=4, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.16))

    outdir = Path(args.outdir) if args.outdir else default_outdir()
    save(fig, outdir, args.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
