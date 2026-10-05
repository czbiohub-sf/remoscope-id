"""Enumerate the experiment_ids each model's class_config considers in-scope,
and how they partition into the model's classes.

Mirrors the scope-computation portion of
``RemoIDDataModule._prepare_data_class_config`` / ``_prepare_data_manifest``
so that a cascade can auto-infer its tree from the same filters training used.
Only set operations are needed here — no DataFrame construction, no zarr checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from remo_id.data.class_config import parse_class_config, ClassConfigResult
from remo_id.data.manifest import assign_manifest_classes, is_manifest_class_config


@dataclass
class ModelScope:
    model_id: str
    class_names: list[str]
    scope: set[str]
    class_experiments: dict[str, set[str]]
    class_config_result: ClassConfigResult


def compute_model_scopes(models) -> dict[str, ModelScope]:
    """For each model, return the set of in-scope experiment_ids and the
    per-class partitioning of those ids, computed from the model's
    class_config against the live database.

    Manifest-backed class_configs are resolved from their CSVs. Database-backed
    ones share one database connection and pre-fetch the experiment table once
    (each ``_resolve_filter`` call otherwise re-issues queries).
    """
    database = experiment_table = None

    out: dict[str, ModelScope] = {}
    for model in models:
        cfg = parse_class_config(model.class_config)
        if is_manifest_class_config(cfg.class_definitions):
            class_experiments, scope = _compute_manifest_scope(cfg)
        else:
            if database is None:
                from remo_id.data.datamodule import open_database
                database = open_database()
                experiment_table = database.experiment.get()
            class_experiments, scope = _compute_one_scope(
                database, experiment_table, cfg
            )
        out[model.id] = ModelScope(
            model_id=model.id,
            class_names=cfg.class_names,
            scope=scope,
            class_experiments=class_experiments,
            class_config_result=cfg,
        )
    return out


def _compute_manifest_scope(cfg: ClassConfigResult) -> tuple[dict[str, set[str]], set[str]]:
    """(class_experiments, scope) for a manifest-backed class_config."""
    if len(cfg.class_names) < 2:
        raise ValueError("Cascade requires class_configs with >=2 explicitly declared classes")
    default_class_idx = (
        cfg.class_names.index(cfg.default_class) if cfg.default_class in cfg.class_names else None
    )
    df, class_names, _ = assign_manifest_classes(
        cfg.class_definitions, cfg.exclude_filters, cfg.include_filters, default_class_idx,
    )
    class_experiments = {
        name: set(df.loc[df["class_label"] == i, "experiment_id"].astype(str))
        for i, name in enumerate(class_names)
    }
    return class_experiments, set(df["experiment_id"].astype(str))


def _compute_one_scope(
    database, experiment_table, cfg: ClassConfigResult
) -> tuple[dict[str, set[str]], set[str]]:
    """Return (class_experiments, scope) for a parsed class_config.

    Reproduces the per-class assignment logic of
    ``_prepare_data_class_config`` without touching filesystem/zarr.
    """
    from remoscope_database.class_config import (
        _resolve_filter,
        _get_all_study_experiment_ids,
        _get_non_null_experiment_ids,
    )

    class_names = cfg.class_names
    if len(class_names) < 2:
        raise ValueError(
            f"Cascade requires class_configs with ≥2 explicitly declared classes; "
            f"got single-class binary with class_names={class_names}. "
            f"Single-class binary has an implicit 'negative' class that can't be "
            f"named in route rules."
        )
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    default_class_idx: Optional[int] = (
        class_names.index(cfg.default_class) if cfg.default_class in class_names else None
    )

    included_ids: Optional[set[str]] = None
    if cfg.include_filters:
        included_ids = set()
        for incl in cfg.include_filters:
            included_ids |= _resolve_filter(database, incl, experiment_table)

    excluded_ids: set[str] = set()
    for excl in cfg.exclude_filters:
        excluded_ids |= _resolve_filter(database, excl, experiment_table)

    class_assignment: dict[str, int] = {}
    all_referenced_ids: set[str] = set()

    for class_def in cfg.class_definitions:
        class_idx = name_to_idx[class_def["name"]]
        matched_ids = _resolve_filter(database, class_def)

        if "study_id" in class_def:
            sid = class_def["study_id"]
            all_study_ids = _get_all_study_experiment_ids(database, sid)
            all_referenced_ids |= all_study_ids
            if len(class_names) == 1 and "column" in class_def:
                non_null_ids = _get_non_null_experiment_ids(database, sid, class_def["column"])
                excluded_ids |= (all_study_ids - non_null_ids)

        if included_ids is not None:
            matched_ids &= included_ids
        matched_ids -= excluded_ids

        for eid in matched_ids:
            if eid in class_assignment and class_assignment[eid] != class_idx:
                prev_name = class_names[class_assignment[eid]]
                raise ValueError(
                    f"Experiment {eid} matches both '{prev_name}' and '{class_def['name']}'"
                )
            class_assignment[eid] = class_idx

    for neg_sid in cfg.negative_study_ids:
        neg_ids = _get_all_study_experiment_ids(database, neg_sid)
        all_referenced_ids |= neg_ids
        for eid in neg_ids:
            if included_ids is not None and eid not in included_ids:
                continue
            if eid not in excluded_ids and eid not in class_assignment:
                class_assignment[eid] = 0

    if default_class_idx is not None:
        for eid in all_referenced_ids:
            if included_ids is not None and eid not in included_ids:
                continue
            if eid not in class_assignment and eid not in excluded_ids:
                class_assignment[eid] = default_class_idx

    class_experiments: dict[str, set[str]] = {name: set() for name in class_names}
    for eid, idx in class_assignment.items():
        class_experiments[class_names[idx]].add(eid)

    scope = set(class_assignment.keys())
    return class_experiments, scope
