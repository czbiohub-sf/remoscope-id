"""Cell-class configuration for embedding-MIL bag construction.

Analogous to ``class_config`` (which maps experiments -> disease label), a ``cell_config``
maps individual cells -> a cell class via regionprops gates, and attaches a per-class
sampling ``weight``. It is the policy that decides:

- which cell populations exist (e.g. WBC / RBC / PLT, or a custom split),
- how a cell is assigned to one (a ``where`` expression over regionprops columns,
  first-match priority, ``"*"`` = catch-all),
- how strongly each class is enriched when composing a bag (``weight``),
- the strict per-class RAM cap (``pool_cap``, default 200k distinct cells/class/experiment).

Example YAML::

    pool_cap: 200000
    cells:
      - {name: wbc,      where: "(area > 2300) | (intensity_mean > 140)", weight: 20}
      - {name: platelet, where: "(area < 500) & (intensity_mean > 140)",  weight: 5}
      - {name: rbc,      where: "*",                                       weight: 1}

Bag composition uses ``weight_c * true_count_c`` (the *true*, uncapped per-experiment class
counts), so cross-sample composition differences survive while rare classes are enriched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

DEFAULT_POOL_CAP = 200_000
CATCH_ALL = "*"


@dataclass
class CellClass:
    name: str
    where: str
    weight: float


@dataclass
class CellConfig:
    classes: list[CellClass]
    pool_cap: int

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.classes]

    @property
    def weights(self) -> np.ndarray:
        return np.array([c.weight for c in self.classes], dtype=np.float64)


def load_cell_config(path: str) -> CellConfig:
    """Parse + validate a cell_config YAML."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not raw or "cells" not in raw or not raw["cells"]:
        raise ValueError(f"cell_config {path!r} must define a non-empty 'cells' list")
    classes = []
    for i, c in enumerate(raw["cells"]):
        for k in ("name", "where", "weight"):
            if k not in c:
                raise ValueError(f"cell_config class {i} missing required '{k}'")
        classes.append(CellClass(name=str(c["name"]), where=str(c["where"]),
                                  weight=float(c["weight"])))
    names = [c.name for c in classes]
    if len(set(names)) != len(names):
        raise ValueError(f"cell_config has duplicate class names: {names}")
    # Catch-all, if present, must be last (first-match priority).
    catch = [i for i, c in enumerate(classes) if c.where.strip() == CATCH_ALL]
    if catch and catch != [len(classes) - 1]:
        raise ValueError("a '*' catch-all class must be the LAST entry in cell_config")
    pool_cap = int(raw.get("pool_cap", DEFAULT_POOL_CAP))
    if pool_cap <= 0:
        raise ValueError(f"pool_cap must be positive, got {pool_cap}")
    return CellConfig(classes=classes, pool_cap=pool_cap)


def classify_cells(regionprops: pd.DataFrame, cfg: CellConfig) -> np.ndarray:
    """Assign each regionprops row to a class index (first match wins).

    Returns an int array of class indices, ``-1`` for cells matching no class (only
    possible when there is no ``"*"`` catch-all).
    """
    n = len(regionprops)
    out = np.full(n, -1, dtype=np.int64)
    for idx, c in enumerate(cfg.classes):
        unassigned = out == -1
        if not unassigned.any():
            break
        if c.where.strip() == CATCH_ALL:
            mask = unassigned
        else:
            try:
                m = regionprops.eval(c.where)
            except Exception as e:  # noqa: BLE001 - surface a clear config error
                raise ValueError(
                    f"cell_config class '{c.name}' where-expression failed: {c.where!r} ({e}). "
                    f"Available columns: {list(regionprops.columns)}") from e
            m = np.asarray(m, dtype=bool)
            mask = unassigned & m
        out[mask] = idx
    return out
