#!/usr/bin/env python3
"""
Generate a random demand file for the subproblem.

Inputs:
  - T: horizon length (number of time slots)
  - n: total number of requests (n = a + b)
  - a: number of OUT requests
  - b: number of RET requests (optional; defaults to n - a)

Requests are spread uniformly at random over [0, T-1].
Output is a YAML file (if PyYAML installed) or JSON fallback, with a single key:
  requests: [ {dir: OUT|RET, time: int}, ... ]

Examples:
  python setups/gen_random_demand.py -T 300 -n 445 -a 200 -o setups/scenario_445.yaml
  python setups/gen_random_demand.py -T 96 -n 500 -a 250 --format json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

try:
    import yaml as _yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _yaml = None


def build_requests(T: int, a: int, b: int, seed: int | None = None) -> List[Dict[str, Any]]:
    if seed is not None:
        random.seed(seed)
    reqs: List[Dict[str, Any]] = []
    for _ in range(a):
        reqs.append({"dir": "OUT", "time": random.randint(0, T - 1)})
    for _ in range(b):
        reqs.append({"dir": "RET", "time": random.randint(0, T - 1)})
    random.shuffle(reqs)
    return reqs


def write_output(path: Path, data: Dict[str, Any], fmt: str | None = None) -> None:
    fmt = (fmt or ("yaml" if _yaml is not None else "json")).lower()
    path = path.with_suffix(".yaml" if fmt == "yaml" else ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "yaml":
        if _yaml is None:
            raise RuntimeError("PyYAML not installed; cannot write YAML. Install 'pyyaml' or use --format json.")
        with path.open("w", encoding="utf-8") as f:
            _yaml.safe_dump(data, f, sort_keys=False)
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    print(f"Wrote demand to: {path}")


def default_out_path(T: int, n: int, a: int, b: int, fmt: str | None) -> Path:
    suffix = ".yaml" if (fmt or ("yaml" if _yaml is not None else "json")).lower() == "yaml" else ".json"
    return Path(f"setups/random_demand_T{T}_n{n}_a{a}_b{b}{suffix}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate random demand requests over a time horizon")
    p.add_argument("-T", "--horizon", type=int, required=True, help="Number of time slots (T >= 1)")
    p.add_argument("-n", "--total", type=int, required=True, help="Total number of requests n = a + b")
    p.add_argument("-a", "--out", type=int, required=True, help="Number of OUT requests (a)")
    p.add_argument("-b", "--ret", type=int, default=None, help="Number of RET requests (b). Defaults to n - a")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output file path (.yaml or .json). Default auto-named under setups/")
    p.add_argument("--format", choices=["yaml", "json"], default=None, help="Force output format")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    T: int = args.horizon
    n: int = args.total
    a: int = args.out
    b: int = args.ret if args.ret is not None else (n - a)
    if T <= 0:
        raise SystemExit("T must be >= 1")
    if a < 0 or b < 0 or a + b != n:
        raise SystemExit("Invalid counts: require a >= 0, b >= 0, and a + b = n")

    reqs = build_requests(T, a, b, seed=args.seed)
    data = {"n": n, "requests": reqs}

    out_path = args.output if args.output is not None else default_out_path(T, n, a, b, args.format)
    write_output(out_path, data, fmt=args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

