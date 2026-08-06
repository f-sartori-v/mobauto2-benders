"""Fixed-budget parameter sweep for the corrected Benders decomposition.

`p` and `W_max` are treated as given inputs (50, 60) and are not swept. The axes
here are structural: time discretisation and fleet size.

Every cell runs under the same wall-clock budget rather than to convergence.
This is only meaningful because the cuts are now valid lower bounds (D20): a
truncated run still yields a valid LB and a feasible UB, i.e. an honest bracket
around the optimum. Cells that converge are marked OPTIMAL; the rest report the
bracket they reached at the cutoff.

Usage:
    python scripts/sweep.py                 # all axes
    python scripts/sweep.py --axis slots    # one axis
    python scripts/sweep.py --budget 120
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "configs" / "mw_convergence.yaml"
CONFIG_DIR = ROOT / "configs" / "sweep"
LOG_DIR = ROOT / "docs" / "sweep"

# Given inputs, held fixed across every cell.
FIXED_P = 50.0
FIXED_WMAX = 60

# Ceiling on one master solve. Well under the total budget, or a single master
# solve eats the cell and it reports a one-iteration bracket. This only started
# having an effect once the Benders loop stopped overwriting it from hardcoded
# constants -- the first sweep ran with an inert 30 and every cell used 102s.
MASTER_ITER_LIMIT_S = 30

RESULT_RE = re.compile(
    r"Result:\s*status=(?P<status>\S+)\s+iterations=(?P<iters>\d+)\s+"
    r"best_lb=(?P<lb>\S+)\s+best_ub=(?P<ub>\S+)"
)
SERVED_RE = re.compile(r"Pax served:\s*(?P<served>\d+)/(?P<total>\d+)")
WAIT_RE = re.compile(r"Avg wait \(min\):\s*(?P<wait>\S+)")
MANIFEST_RE = re.compile(r"Manifest:\s*(?P<path>.+)")


def cells(axis: str) -> list[dict]:
    """Sweep cells. Q=2 / 30 min is the reference point and is shared by both axes."""
    out: list[dict] = []
    if axis in ("slots", "all"):
        for res in (30, 15):
            out.append({"tag": f"res{res}_q2", "slot_resolution": res, "Q": 2})
    if axis in ("fleet", "all"):
        for q in (1, 2, 3, 4, 5):
            if axis == "all" and q == 2:
                continue  # already covered by res30_q2
            out.append({"tag": f"res30_q{q}", "slot_resolution": 30, "Q": q})
    return out


def build_config(cell: dict, budget: int) -> Path:
    doc = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    doc = copy.deepcopy(doc)

    q = int(cell["Q"])
    doc["run"]["name"] = f"sweep_{cell['tag']}"
    doc["model"]["time"]["slot_resolution"] = int(cell["slot_resolution"])
    doc["model"]["fleet"]["Q"] = q
    # Per-vehicle lists are [z specific vehicles..., 1 value shared by the rest],
    # so a homogeneous fleet is a single value regardless of Q.
    doc["model"]["fleet"]["initial_battery"] = [150.0]
    doc["model"]["fleet"]["initial_actions"] = ["IDL"]

    doc["subproblem"]["p"] = FIXED_P
    doc["subproblem"]["Wmax_minutes"] = FIXED_WMAX

    doc["solver"]["total_time_limit_s"] = int(budget)
    doc["master"]["per_iteration_time_limit_s"] = min(MASTER_ITER_LIMIT_S, int(budget))

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{cell['tag']}.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def run_cell(cell: dict, budget: int) -> dict:
    cfg = build_config(cell, budget)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{cell['tag']}.log"

    started = time.time()
    # Stream to the log rather than buffering until the process returns: the first
    # sweep lost a cell that was killed mid-solve and left no evidence at all of
    # how far it had got.
    deadline = started + budget * 6 + 300
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [sys.executable, "-m", "mobauto2_benders", "--config", str(cfg), "run"],
            cwd=str(ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            proc.wait(timeout=max(1.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
    elapsed = time.time() - started
    text = log_path.read_text(encoding="utf-8", errors="replace")

    row: dict = dict(cell)
    row["wall_s"] = round(elapsed, 1)
    row["exit"] = proc.returncode
    row["T_slots"] = 660 // int(cell["slot_resolution"])

    m = RESULT_RE.search(text)
    if m:
        row["status"] = m.group("status").split(".")[-1]
        row["iterations"] = int(m.group("iters"))
        row["lb"] = float(m.group("lb"))
        row["ub"] = float(m.group("ub"))
        if row["ub"] != 0:
            row["gap_pct"] = 100.0 * (row["ub"] - row["lb"]) / abs(row["ub"])
    else:
        row["status"] = "NO_RESULT"

    served = None
    for served in SERVED_RE.finditer(text):
        pass  # the final occurrence is the run total
    if served:
        row["served"] = int(served.group("served"))
        row["demand"] = int(served.group("total"))

    wait = None
    for wait in WAIT_RE.finditer(text):
        pass
    if wait:
        row["avg_wait_min"] = float(wait.group("wait"))

    man = MANIFEST_RE.search(text)
    if man:
        row["manifest"] = Path(man.group("path").strip()).name

    return row


def render_table(rows: list[dict]) -> str:
    head = (
        "| cell | slot_res | T | Q | status | iters | LB | UB | gap % | served | wait min | wall s |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = ""
    for r in rows:
        def g(k, fmt="{}"):
            v = r.get(k)
            return fmt.format(v) if v is not None else "-"

        body += (
            f"| `{r['tag']}` | {r['slot_resolution']} | {r.get('T_slots', '-')} | {r['Q']} "
            f"| {r.get('status', '-')} | {g('iterations')} | {g('lb', '{:.4f}')} "
            f"| {g('ub', '{:.4f}')} | {g('gap_pct', '{:.3f}')} | {g('served')} "
            f"| {g('avg_wait_min', '{:.2f}')} | {r.get('wall_s', '-')} |\n"
        )
    return head + body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis", choices=["slots", "fleet", "all"], default="all")
    ap.add_argument("--budget", type=int, default=120, help="Per-cell seconds")
    args = ap.parse_args()

    plan = cells(args.axis)
    print(f"{len(plan)} cells, {args.budget}s each, p={FIXED_P} Wmax={FIXED_WMAX}\n")

    rows: list[dict] = []
    for i, cell in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {cell['tag']} ... ", end="", flush=True)
        try:
            row = run_cell(cell, args.budget)
        except subprocess.TimeoutExpired:
            row = dict(cell, status="HARD_TIMEOUT")
        rows.append(row)
        print(f"{row.get('status')} lb={row.get('lb')} ub={row.get('ub')}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    table = render_table(rows)
    (LOG_DIR / "results.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"Wrote {LOG_DIR / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
