"""The p_minutes x Wmax frontier, and the cliff a sweep alone found.

    python scripts/report_figures/fig_penalty_window.py

Left panel: share of demand served, against the penalty, one line per Wmax.
Right panel: average wait among served passengers, same axes. Both priced at
minute fidelity (D51), because a slot-reported figure here would carry the same
28.5%/66-86% error the rest of this report corrects for.

The point of the figure is the left panel's cliff: at p_minutes 14 and 28 -- below
the operator's stated indifference of ~56 -- the model's cost-minimising choice is
to run NO service at all, whatever Wmax is. Service is not gradual in the penalty;
it turns on somewhere between 28 and 35 (a refinement sweep at Wmax=60, marked with
the dashed band). The operator's own value sits just above that threshold, not
comfortably inside the region where more service is worth it.
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

SHADES = ("#C9D6E8", "#8FABCF", "#5A82B5", "#2F5A93", "#173257")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(Path(__file__).parent / "data" / "measurements.json"))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--stem", default="penalty_window_frontier")
    args = ap.parse_args()

    d = json.loads(Path(args.data).read_text(encoding="utf-8"))["penalty_window_frontier"]
    p_grid = d["p_minutes_grid"]
    wmax_grid = d["wmax_grid"]
    total_pax = 300.0

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.75), sharex=True)

    for j, wmax in enumerate(wmax_grid):
        key = str(int(wmax))
        served_pct = [100.0 * v / total_pax for v in d["served"][key]]
        axes[0].plot(p_grid, served_pct, marker="o", ms=3.0, color=SHADES[j],
                     label=f"Wmax={wmax:.0f} min")
        axes[1].plot(p_grid, d["avg_wait_min"][key], marker="o", ms=3.0,
                     color=SHADES[j])

    z = d["zero_service_threshold"]
    axes[0].axvspan(z["p_minutes"][0], z["p_minutes"][1], color=ACCENT, alpha=0.15,
                     zorder=1)
    axes[0].annotate(
        "zero service\nbelow here", (31.0, 45.0), fontsize=7.0, color=ACCENT,
        ha="center", va="bottom",
    )
    axes[0].axvline(56.0, color=NEUTRAL, lw=1.0, ls=(0, (4, 2)), zorder=2)
    axes[0].annotate("operator's\np_min ~ 56", (56.0, 45.0), xytext=(5, 0),
                      textcoords="offset points", fontsize=7.0, color=NEUTRAL,
                      ha="left", va="bottom")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(p_grid)
        ax.set_xticklabels([f"{p:.0f}" for p in p_grid])
        ax.set_xlabel("p_minutes")
        value_grid(ax)

    axes[0].set_ylabel("served, per cent of demand")
    axes[0].set_ylim(-4, 100)
    axes[1].set_ylabel("average wait among served, minutes")
    policy = d.get("instance", "").split("policy = ")[-1].split(",")[0] or "start"
    axes[1].set_title(
        f"minute-honest pricing throughout, policy = {policy}", loc="right",
        fontsize=7.5, color="#444444",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7.0,
               bbox_to_anchor=(0.5, -0.14))

    outdir = Path(args.outdir) if args.outdir else default_outdir()
    save(fig, outdir, args.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
