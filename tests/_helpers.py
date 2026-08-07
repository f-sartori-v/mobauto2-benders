"""Shared helpers. Keeps the src/ layout importable without installing."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONFIGS = REPO_ROOT / "configs"


def load_cfg(name: str):
    from mobauto2_benders.config import load_config

    return load_config(CONFIGS / name)


def master_params(name: str = "baseline_d9.yaml") -> dict:
    """Master params exactly as app.py would assemble them."""
    from mobauto2_benders.app import _prepare_params

    mp, _sp = _prepare_params(load_cfg(name), {})
    mp = dict(mp)
    # _prepare_params leaves T implicit when the config gives T_minutes; the
    # solver derives it. Do the same here so the model can be built standalone.
    if mp.get("T") is None:
        t_min = int(mp.get("T_minutes") or 0)
        res = int(mp.get("slot_resolution") or 1)
        mp["T"] = max(1, t_min // max(1, res))
    return mp


def build_master(params: dict):
    from mobauto2_benders.problem.master_impl import ProblemMaster

    pm = ProblemMaster(params)
    pm.initialize()
    return pm


def constraint_names(model) -> set[str]:
    import pyomo.environ as pyo

    return {c.name for c in model.component_objects(pyo.Constraint)}
