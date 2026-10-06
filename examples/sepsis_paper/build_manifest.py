#!/usr/bin/env python3
"""Export the study's manifest from the Remoscope database.

This is the only script in this folder that reads the database
(``remoscope_database``); every other script, and every remo-id command, can read
the manifest it writes instead (``--manifest manifest.csv``). The manifest carries
pseudonymous patient IDs and is distributed with the data release, never with this
repository.

One row per experiment of every study named in the class config, healthy donors
included:

    experiment_id, sample_id, study_id, scope, path, num_images, label, <clinical columns>

``study_id`` is the study as named in the class config; ``scope``, ``path`` and
``num_images`` come from the experiment table and ``sample_id`` (the patient) from
the study table. ``label`` is the class name the class config assigns (exclusions
applied), empty for every other experiment, so the file works as a remo-id manifest
with ``--classes`` in class-config order. A study that backs a single class with a
``column``/``value`` filter (the healthy donors) keeps only the rows passing that
filter; a study split over several classes (sepsis / ICU control) keeps every row,
including the patients without an adjudicated label (the indeterminate patients).
The clinical columns are the study-table columns that the clinical spec
(``--clinical_spec``) or the class config name; database nulls are written as empty
fields. Rows keep the order of the experiment-table query of the class-config datamodule,
which fixes the per-experiment bag seeds of the LOO predictions.

    python examples/sepsis_paper/build_manifest.py \\
        --class_config examples/sepsis_paper/configs/sepsis_icu_healthy.yaml --output manifest.csv
"""
import argparse

import pandas as pd
import yaml

from common import DEFAULT_CLINICAL_SPEC, spec_columns

IDS = ["experiment_id", "sample_id", "study_id", "scope", "path", "num_images", "label"]


def build_manifest(database, class_config, spec_path):
    from remo_id.inference.context import load_class_config
    from remoscope_database.class_config import resolve_class_assignment

    defs, names, _, _, exclude, include, default_idx = load_class_config(class_config)
    with open(spec_path) as f:
        spec = yaml.safe_load(f)
    wanted = spec_columns(spec) + [d["column"] for d in defs if d.get("column")]
    labels = resolve_class_assignment(database, defs, exclude_filters=exclude,
                                      include_filters=include,
                                      default_class_idx=default_idx)["class_assignment"]
    by_study = {}
    for d in defs:
        if d.get("study_id"):
            by_study.setdefault(str(d["study_id"]), []).append(d)
    frames = []
    for study, sdefs in by_study.items():
        df = database.studies[study].get().to_pandas()
        col, val = sdefs[0].get("column"), sdefs[0].get("value")
        if len(sdefs) == 1 and col and val is not None:
            df = df[df[col].astype(str).str.strip().str.lower() == str(val).strip().lower()]
        df["experiment_id"] = df["experiment_id"].astype(str)
        df["study_id"] = study
        frames.append(df[["experiment_id", "sample_id", "study_id"]
                         + [c for c in df.columns if c in wanted and c not in IDS]])
        print(f"  {study}: {len(df)} experiments")
    rows = pd.concat(frames, ignore_index=True)

    # The same experiment-table query as the class-config datamodule, so rows keep its
    # order (which fixes the per-experiment bag seeds of the LOO predictions).
    et = database.experiment.get()
    exp = (et.select(["experiment_id", "scope", "path", "num_images"])
           .filter(et["experiment_id"].isin(list(rows["experiment_id"]))).execute())
    exp["experiment_id"] = exp["experiment_id"].astype(str)
    out = exp.merge(
        rows, on="experiment_id", how="left", validate="one_to_one")
    out["label"] = [names[labels[e]] if e in labels else None for e in out["experiment_id"]]
    return out[IDS + [c for c in out.columns if c not in IDS]]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--class_config", required=True, help="class config YAML")
    p.add_argument("--output", required=True, help="manifest CSV")
    p.add_argument("--clinical_spec", default=DEFAULT_CLINICAL_SPEC,
                   help="clinical spec YAML (default configs/clinical_tables.yaml)")
    args = p.parse_args()

    from remoscope_database.database.database_manager import DatabaseManager
    df = build_manifest(DatabaseManager(), args.class_config, args.clinical_spec)
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} experiments x {df.shape[1]} columns "
          f"({df['label'].notna().sum()} labelled) -> {args.output}")
    print(df.groupby(["study_id", "scope", "label"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()
