"""Baseline demand profile of the instance family (report Figure 1).

    python scripts/report_figures/fig_demand_profile.py [--setup setups/base.yaml]
                                                        [--slot 30] [--start 07:00]

Reads the arrival minutes from the setup file and aggregates them at the slot
width the tactical model uses, which is exactly what the model does at load. The
figure therefore shows the demand as the slot models see it, not a redrawing of
it.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _style import (  # noqa: E402
    LIGHT,
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


def _labels(n_slots: int, slot: int, start: str) -> list[str]:
    h0, m0 = (int(x) for x in start.split(":"))
    base = h0 * 60 + m0
    out = []
    for s in range(n_slots):
        t = base + s * slot
        out.append(f"{t // 60:02d}:{t % 60:02d}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--setup", default="setups/base.yaml")
    ap.add_argument("--slot", type=int, default=30)
    ap.add_argument("--start", default="07:00")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--stem", default="demand_profile")
    args = ap.parse_args()

    path = Path(args.setup)
    if not path.is_absolute():
        path = repo_root() / path
    requests = load_requests(path)
    if not requests:
        raise SystemExit(f"no requests parsed from {path}")

    counts: Counter = Counter((d, t // args.slot) for d, t in requests)
    n_slots = max(s for _, s in counts) + 1
    out = [counts.get(("OUT", s), 0) for s in range(n_slots)]
    ret = [counts.get(("RET", s), 0) for s in range(n_slots)]

    apply_style()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.5))
    x = range(n_slots)
    w = 0.42
    ax.bar([i - w / 2 for i in x], out, width=w, color=OUT_COLOUR,
           label=r"OUT  (Longvilliers $\rightarrow$ Massy)")
    ax.bar([i + w / 2 for i in x], ret, width=w, color=RET_COLOUR,
           label=r"RET  (Massy $\rightarrow$ Longvilliers)")

    labels = _labels(n_slots, args.slot, args.start)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("bookings in slot")
    ax.set_xlabel(f"departure slot, {args.slot} min")
    ax.set_xlim(-0.8, n_slots - 0.2)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    value_grid(ax)

    # The two directional peaks are what forces repositioning legs, so they are
    # named on the figure rather than left to the caption.
    peak_out = max(range(n_slots), key=lambda s: out[s])
    peak_ret = max(range(n_slots), key=lambda s: ret[s])
    ax.annotate(f"{out[peak_out]}", (peak_out - w / 2, out[peak_out]),
                textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=7, color=OUT_COLOUR, clip_on=False)
    ax.annotate(f"{ret[peak_ret]}", (peak_ret + w / 2, ret[peak_ret]),
                textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=7, color=RET_COLOUR, clip_on=False)
    ax.axvspan(-0.8, 4.5, color=LIGHT, alpha=0.30, lw=0, zorder=0)
    ax.axvspan(15.5, n_slots - 0.2, color=LIGHT, alpha=0.30, lw=0, zorder=0)
    ax.set_ylim(0, max(max(out), max(ret)) * 1.18)

    total = sum(out) + sum(ret)
    ax.set_title(f"R = {total} requests, {sum(out)} OUT and {sum(ret)} RET, "
                 f"source {Path(args.setup).name}", loc="left", color="#444444")

    outdir = Path(args.outdir) if args.outdir else default_outdir()
    save(fig, outdir, args.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
