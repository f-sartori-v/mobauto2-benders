"""What slot aggregation costs, split into a valuation error and a decision error.

    python scripts/report_figures/fig_valuation_decomposition.py

Three costs on one axis, all in passenger-minutes:

  1. what the slot model says its own schedule costs;
  2. what that same schedule costs when its passengers are priced at the minute
     they actually arrive;
  3. what the best schedule available under that pricing costs.

The distance between (1) and (2) is a reporting error: it misleads whoever reads
the model's output and costs the operator nothing. The distance between (2) and
(3) is the schedule the operator does not get. The two are separate quantities
and the figure keeps them apart rather than summing them.
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
    ap.add_argument("--stem", default="valuation_decomposition")
    args = ap.parse_args()

    d = json.loads(Path(args.data).read_text(encoding="utf-8"))["valuation_decomposition"]
    claimed = d["claimed_cost_of_slot_schedule"]
    true_slot = d["true_cost_of_slot_schedule"]
    best = d["true_cost_of_best_schedule"]

    labels = [
        "cost the slot model\nreports for its schedule",
        "cost of that same schedule,\npriced at the arrival minute",
        "cost of the best schedule\navailable under that pricing",
    ]
    values = [claimed, true_slot, best]
    colours = [LIGHT, OUT_COLOUR, ACCENT]

    apply_style()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.6))
    y = [2, 1, 0]
    ax.barh(y, values, height=0.46, color=colours, alpha=0.85, zorder=2)
    for yy, v in zip(y, values):
        ax.annotate(f"{v:,.0f}".replace(",", "\u2009"), (v, yy), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"cost, {d['unit']}")
    ax.set_xlim(0, claimed * 1.34)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    value_grid(ax, axis="x")

    # The two errors, drawn as brackets between the bars they separate.
    def bracket(x_from: float, x_to: float, y_at: float, text: str, colour: str) -> None:
        ax.annotate("", xy=(x_to, y_at), xytext=(x_from, y_at),
                    arrowprops=dict(arrowstyle="<->", color=colour, lw=1.0))
        ax.annotate(text, ((x_from + x_to) / 2, y_at), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    color=colour, fontsize=7.5)

    # Guides, so the two gaps are readable against a common baseline.
    for xv in (claimed, true_slot, best):
        ax.plot([xv, xv], [-0.35, 2.55], color="#B0B0B0", lw=0.6,
                ls=(0, (2, 2)), zorder=1)

    bracket(true_slot, claimed, 2.45,
            f"reporting error  +{d['reporting_error_pct']:.1f} %", NEUTRAL)
    ax.annotate("", xy=(true_slot, 1.45), xytext=(best, 1.45),
                arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.0))
    ax.annotate(f"decision error  +{d['decision_error_pct']:.1f} %",
                (true_slot, 1.45), xytext=(26, 10), textcoords="offset points",
                ha="left", va="bottom", color=ACCENT, fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.7,
                                connectionstyle="arc3,rad=0.0"))

    ax.set_ylim(-0.45, 2.95)
    ax.set_title(d["instance"], loc="left", color="#444444", fontsize=7.8)

    outdir = Path(args.outdir) if args.outdir else default_outdir()
    save(fig, outdir, args.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
