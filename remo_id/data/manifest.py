"""CSV manifests: train and evaluate without a database.

A manifest lists one experiment per row:

==============  ========  =====================================================
column          required  meaning
==============  ========  =====================================================
experiment_id   yes       unique integer id
path            yes       experiment directory holding the ``.zip`` image stack
                          (see :mod:`remo_id.data.zarr_io`); relative paths are
                          resolved against the manifest's directory
num_images      yes       number of frames in the stack
label           yes       class name; the column name is configurable
                          (``label_column``)
sample_id       no        experiments sharing a sample_id (e.g. technical
                          replicates of one blood sample) always stay in the same
                          train/val split; empty = independent sample
study_id, site  no        free-text metadata carried into predictions
(any other)     no        kept as read, e.g. ``scope`` or clinical variables
==============  ========  =====================================================

Only empty fields are missing values (text such as ``None`` stays text).

A manifest is turned into an ordinary class config whose entries reference
the manifest instead of a database study::

    classes:
      - {name: negative, manifest: /data/manifest.csv, column: label, value: negative}
      - {name: positive, manifest: /data/manifest.csv, column: label, value: positive}

Class index = list order. Two classes -> binary (the second class is the
positive class), three or more -> multiclass. This YAML is what training saves
as ``classes.yaml`` next to the checkpoint, so every command that accepts
``--class_config`` also works for manifest-trained models.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("experiment_id", "path", "num_images")
OPTIONAL_COLUMNS = ("sample_id", "study_id", "site")


def load_manifest(path: str, label_column: str = "label") -> pd.DataFrame:
    """Read and validate a manifest CSV.

    Returns a DataFrame with ``experiment_id`` (str of an int), absolute
    ``path``, int ``num_images``, the label column (str, NaN when empty), the
    optional ``sample_id``/``study_id``/``site`` columns (``None`` when absent or
    empty) and then every other column as read. Only empty fields are missing.
    """
    text_cols = {c: "string" for c in (label_column, *OPTIONAL_COLUMNS)}
    df = pd.read_csv(path, dtype={"path": "string", **text_cols},
                     keep_default_na=False, na_values=[""])
    missing = [c for c in (*REQUIRED_COLUMNS, label_column) if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest {path} is missing required column(s): {missing}")
    if len(df) == 0:
        raise ValueError(f"Manifest {path} has no rows")

    exp_ids = pd.to_numeric(df["experiment_id"], errors="coerce")
    bad = df.loc[exp_ids.isna() | (exp_ids % 1 != 0), "experiment_id"].tolist()
    if bad:
        raise ValueError(f"Manifest {path}: experiment_id must be an integer; got {bad[:5]}")
    df["experiment_id"] = exp_ids.astype(np.int64).astype(str)
    dupes = df.loc[df["experiment_id"].duplicated(), "experiment_id"].tolist()
    if dupes:
        raise ValueError(f"Manifest {path}: duplicate experiment_id {dupes[:5]}")

    n_img = pd.to_numeric(df["num_images"], errors="coerce")
    bad = df.loc[n_img.isna() | (n_img <= 0) | (n_img % 1 != 0), "experiment_id"].tolist()
    if bad:
        raise ValueError(f"Manifest {path}: num_images must be a positive integer (experiment_id {bad[:5]})")
    df["num_images"] = n_img.astype(np.int64)

    if df["path"].isna().any():
        raise ValueError(f"Manifest {path}: empty path for experiment_id "
                         f"{df.loc[df['path'].isna(), 'experiment_id'].tolist()[:5]}")
    base = os.path.dirname(os.path.abspath(path))
    df["path"] = [p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))
                  for p in df["path"].astype(str)]

    df[label_column] = df[label_column].str.strip().astype(object).where(df[label_column].notna(), np.nan)
    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(object).where(df[col].notna(), None)
        else:
            df[col] = None
    columns = [*REQUIRED_COLUMNS, label_column, *OPTIONAL_COLUMNS]
    return df[columns + [c for c in df.columns if c not in columns]].reset_index(drop=True)


def manifest_classes(df: pd.DataFrame, label_column: str = "label", classes=None,
                     require_present: bool = True) -> list[str]:
    """Class names in index order: ``classes`` if given, otherwise the sorted
    unique labels (numeric order when all labels are numbers). With
    ``require_present`` every listed class must occur in the manifest (training);
    inference on a test set may lack some classes."""
    present = sorted(df[label_column].dropna().unique().tolist())
    if classes is not None:
        classes = [str(c).strip() for c in classes]
        if len(set(classes)) != len(classes):
            raise ValueError(f"Duplicate class names in classes={classes}")
        unknown = [c for c in classes if c not in present]
        if unknown and require_present:
            raise ValueError(f"classes {unknown} do not occur in manifest column '{label_column}' "
                             f"(labels present: {present})")
        return classes
    numeric = pd.to_numeric(pd.Series(present), errors="coerce")
    if len(present) and numeric.notna().all():
        present = [p for _, p in sorted(zip(numeric.tolist(), present))]
    return present


def manifest_class_config(path: str, label_column: str = "label", classes=None,
                          require_present: bool = True) -> dict:
    """Build the class-config dict for a classification manifest (see module doc)."""
    df = load_manifest(path, label_column)
    names = manifest_classes(df, label_column, classes, require_present)
    if len(names) < 2:
        raise ValueError(f"Manifest {path} needs at least two classes in '{label_column}'; found {names}")
    abs_path = os.path.abspath(path)
    return {"classes": [{"name": n, "manifest": abs_path, "column": label_column, "value": n}
                        for n in names]}


def is_manifest_class_config(class_definitions) -> bool:
    """True if the class definitions reference manifests (no database needed)."""
    return (bool(class_definitions) and isinstance(class_definitions, list)
            and all(isinstance(d, dict) and "manifest" in d for d in class_definitions))


def assign_manifest_classes(class_definitions: list, exclude_filters=None, include_filters=None,
                            default_class_idx: int | None = None):
    """Resolve manifest-backed class definitions to labelled experiments.

    Returns ``(df, class_names, unassigned_paths)`` where ``df`` holds the manifest
    columns plus an int ``class_label`` (class index; for a single class:
    1 = matched, 0 = any other labelled experiment) and ``unassigned_paths`` are
    experiments whose label is empty or matches no class.
    """
    if include_filters:
        raise ValueError("'include' filters are not supported for manifest class configs")
    excluded = set()
    for f in exclude_filters or []:
        if set(f) != {"experiment_ids"}:
            raise ValueError(f"Manifest class configs only support 'exclude: - experiment_ids: [...]'; got {f}")
        excluded.update(str(e) for e in f["experiment_ids"])

    class_names = list(dict.fromkeys(d["name"] for d in class_definitions))
    frames = {}
    for d in class_definitions:
        key = (d["manifest"], d.get("column", "label"))
        if key not in frames:
            frames[key] = load_manifest(*key)

    labels = {}     # experiment_id -> class index
    for d in class_definitions:
        key = (d["manifest"], d.get("column", "label"))
        df = frames[key]
        idx = class_names.index(d["name"])
        for eid in df.loc[df[key[1]] == str(d["value"]).strip(), "experiment_id"]:
            if labels.get(eid, idx) != idx:
                raise ValueError(f"Experiment {eid} matches classes '{class_names[labels[eid]]}' "
                                 f"and '{d['name']}'")
            labels[eid] = idx

    single = len(class_names) == 1
    out, seen, unassigned = [], set(), set()
    for (_, label_column), df in frames.items():
        for _, row in df.iterrows():
            eid = row["experiment_id"]
            if eid in seen or eid in excluded:
                continue
            seen.add(eid)
            has_label = pd.notna(row[label_column])
            if eid in labels:
                cls = 1 if single else labels[eid]
            elif single and has_label:
                cls = 0
            elif default_class_idx is not None and has_label:
                cls = default_class_idx
            else:
                unassigned.add(row["path"])
                continue
            rec = {c: row[c] for c in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)}
            rec["class_label"] = int(cls)
            out.append(rec)
    df = pd.DataFrame(out, columns=[*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS, "class_label"])
    return df, class_names, unassigned
