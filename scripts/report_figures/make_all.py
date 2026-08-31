"""Regenerate every figure the T.5.4 report prints.

    python scripts/report_figures/make_all.py [--outdir outputs/figures]

Then copy the PDFs into the report tree's img/ folder. The report includes them
by stem, so a regenerated figure replaces the printed one with no edit to the
LaTeX source.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = (
    "fig_demand_profile.py",
    "fig_demand_shapes.py",
    "fig_bound_interval.py",
    "fig_valuation_decomposition.py",
    "fig_multiresolution_gain.py",
    "fig_penalty_window.py",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    failed: list[str] = []
    for name in SCRIPTS:
        cmd = [sys.executable, str(HERE / name)]
        if args.outdir:
            cmd += ["--outdir", args.outdir]
        print(f"--- {name}")
        if subprocess.call(cmd) != 0:
            failed.append(name)
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
