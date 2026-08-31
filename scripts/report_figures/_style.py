"""Shared plotting style for the T.5.4 report figures.

Every figure the report prints is produced by one of the scripts in this folder,
from a file that ships in this repository. Nothing is drawn by hand and nothing
is transcribed from a log: a figure whose data cannot be regenerated is prose.

Output is vector PDF at the text width of the report (about 5.9 in inside the
1-inch margins of an A4 article), so the figures sit at 100 % scale and the
label sizes below are the sizes the reader sees.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The report's own section colour and a second hue that survives greyscale
# printing and the common forms of colour vision deficiency.
OUT_COLOUR = "#42679D"     # Longvilliers -> Massy
RET_COLOUR = "#D98C3F"     # Massy -> Longvilliers
ACCENT = "#8C2F39"
NEUTRAL = "#5A5A5A"
LIGHT = "#C9C9C9"

TEXT_WIDTH_IN = 5.9


def apply_style() -> None:
    """Set the report look. Grid on the value axis only is applied per-axes."""
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E3E3E3",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
        }
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_outdir() -> Path:
    return repo_root() / "outputs" / "figures"


def save(fig, outdir: Path, stem: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{stem}.pdf"
    fig.savefig(path)
    fig.savefig(outdir / f"{stem}.png")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def load_requests(path: Path) -> list[tuple[str, int]]:
    """Read a setup file as (direction, arrival minute) pairs.

    The setup files are flat YAML lists, so they are parsed here without a YAML
    dependency: the figure scripts must run in a checkout that has matplotlib
    and nothing else.
    """
    rows: list[tuple[str, int]] = []
    direction: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("- dir:"):
            direction = line.split(":", 1)[1].strip()
        elif line.startswith("time:") and direction is not None:
            rows.append((direction, int(line.split(":", 1)[1].strip())))
            direction = None
    return rows


def value_grid(ax, axis: str = "y") -> None:
    """Grid on the value axis only; category axes carry no rules."""
    ax.grid(True, axis=axis)
    ax.grid(False, axis="x" if axis == "y" else "y")
