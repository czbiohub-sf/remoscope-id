#!/usr/bin/env python3
"""Export the cohort table and the clinical input table from the database.

This is the only script in this folder that reads the clinical database
(``remoscope_database``); every other script reads the two tables it writes.
Both carry pseudonymous patient IDs and are distributed with the controlled-access
data release, never with this repository.

cohort.csv (``--output``): one row per LOO experiment with its patient and scope

    experiment_id, sample_id, study_id, scope

The LOO experiments are the held-out experiments of every fold in
``{cv_dir}/cv_splits.yaml``; study and scope come from the experiment table and
sample_id (the patient) from each study table. make_paper_results.py (--cohort)
uses it for every patient-level output and the Fig 1 sample panel; umap_images.py
uses the scope column to select the UMAP experiments.

clinical.csv (``--clinical_output``, needs ``--class_config``): one row per
experiment of every study named in the class config, with

    experiment_id, sample_id, study_id, path, num_images, <clinical columns>

``study_id`` is the study as named in the class config; ``path``/``num_images``
come from the experiment table; the clinical columns are the study-table columns
that the clinical-table spec (``--clinical_spec``) or the class config name.
A study that backs a single class with a ``column``/``value`` filter (the healthy
donors) keeps only the rows passing that filter; a study split over several
classes (sepsis / ICU control) keeps every row, including the patients without
an adjudicated label (the indeterminate table). make_paper_results.py
(--clinical_table) and infer_indeterminate.py (--clinical_table) read it.

    python examples/sepsis_paper/build_cohort_table.py --cv_dir <LOO_DIR> \\
        --output cohort.csv --class_config examples/sepsis_paper/configs/sepsis_icu_healthy.yaml \\
        --clinical_output clinical.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import yaml

from common import DEFAULT_CLINICAL_SPEC


def loo_experiments(cv_dir):
    with open(os.path.join(cv_dir, "cv_splits.yaml")) as f:
        splits = yaml.safe_load(f)
    return sorted({str(e) for folds in splits["repeats"] for fold in folds
                   for e in fold.get("val_experiment_ids", [])})


def db_sample_map(database, study_ids):
    """experiment_id -> sample_id from the given study tables."""
    sample_map = {}
    for sid in sorted(set(np.asarray(study_ids).astype(str)) - {""}):
        try:
            stbl = database.get_table_from_string(sid).get()
            col_names = [c.get_name() if hasattr(c, "get_name") else str(c)
                         for c in stbl.columns]
            if "sample_id" not in col_names:
                continue
            df = stbl.select(["experiment_id", "sample_id"]).execute()
            df["experiment_id"] = df["experiment_id"].astype(str)
            for e, s in zip(df["experiment_id"], df["sample_id"]):
                if s is not None:
                    sample_map.setdefault(e, str(s))
        except Exception as exc:  # noqa: BLE001
            print(f"  (sample_id lookup failed for study {sid}: {exc})")
    return sample_map


def spec_columns(spec, class_definitions):
    """Every study-table column the clinical-table spec or the class config names."""
    cols = []

    def add(*names):
        cols.extend(n for n in names if n and n not in cols)

    pre = spec["preprocess"]
    add(*pre.get("missing_codes", {}), pre.get("age", {}).get("column"),
        *pre.get("boolean_columns", []), *pre.get("sparse_flags", []))
    for terms in pre.get("derived_scores", {}).values():
        add(*(c for term in terms for c, _, _ in term))
    add(*spec["diagnosis"].values())
    t1 = spec["table1"]
    add(t1["mortality"]["column"], t1["microbiology"]["column"], t1["pathogen_column"],
        t1["age_column"], *t1["sex_columns"])
    t3 = spec["table3"]
    add(*(r["column"] for r in t3["rows"]),
        *(r["column"] for r in t3["manuscript"].get("insert_rows", [])),
        *t3.get("healthy_column_map", {}).values(),
        t3["race"]["column"], t3["race"]["ethnicity_column"])
    add(*spec["variable_labels"])
    for entries in spec["correlation_groups"].values():
        for e in entries:
            add(*(e if isinstance(e, list) else [e]))
    add(*spec["discrimination"]["variables"])
    add(*(d.get("column") for d in class_definitions))
    return cols


def clinical_table(database, class_config, spec_path):
    from remo_id.inference.context import load_class_config

    class_definitions = load_class_config(class_config)[0]
    with open(spec_path) as f:
        spec = yaml.safe_load(f)
    wanted = set(spec_columns(spec, class_definitions))
    single = set(spec.get("single_precision_columns", []))
    exp = database.experiment.get().to_pandas()
    exp["experiment_id"] = exp["experiment_id"].astype(str)
    by_study = {}
    for d in class_definitions:
        if d.get("study_id"):
            by_study.setdefault(str(d["study_id"]), []).append(d)
    frames = []
    for study, defs in by_study.items():
        df = database.studies[study].get().to_pandas()
        col, val = defs[0].get("column"), defs[0].get("value")
        if len(defs) == 1 and col and val is not None:
            df = df[df[col].astype(str).str.strip().str.lower() == str(val).strip().lower()]
        df["experiment_id"] = df["experiment_id"].astype(str)
        df["study_id"] = study
        df = df.merge(exp[["experiment_id", "path", "num_images"]], on="experiment_id",
                      how="left", validate="one_to_one")
        ids = ["experiment_id", "sample_id", "study_id", "path", "num_images"]
        df = df[[c for c in ids if c in df.columns]
                + [c for c in df.columns if c in wanted and c not in ids]]
        f32 = [c for c in df.columns[df.dtypes == np.float32] if c not in single]
        if f32:
            print(f"  note: float32 columns not in single_precision_columns: {f32}")
        frames.append(df)
        print(f"  {study}: {len(df)} experiments, {df.shape[1]} columns")
    return pd.concat(frames, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cv_dir", required=True, help="LOO CV folder")
    p.add_argument("--output", required=True, help="cohort table CSV")
    p.add_argument("--class_config", default=None,
                   help="class config YAML (needed for --clinical_output)")
    p.add_argument("--clinical_output", default=None, help="clinical input table CSV")
    p.add_argument("--clinical_spec", default=DEFAULT_CLINICAL_SPEC,
                   help="clinical-table spec YAML (default configs/clinical_tables.yaml)")
    args = p.parse_args()
    if args.clinical_output and not args.class_config:
        p.error("--clinical_output needs --class_config")

    from remoscope_database.database.database_manager import DatabaseManager
    database = DatabaseManager()
    eids = loo_experiments(args.cv_dir)
    exp = database.experiment.get().to_pandas()
    exp["experiment_id"] = exp["experiment_id"].astype(str)
    exp = exp[exp["experiment_id"].isin(eids)][["experiment_id", "study_id", "scope"]]
    missing = sorted(set(eids) - set(exp["experiment_id"]))
    if missing:
        raise SystemExit(f"{len(missing)} LOO experiments not in the experiment table")

    samples = db_sample_map(database, exp["study_id"])
    exp["sample_id"] = exp["experiment_id"].map(samples).fillna("")
    df = exp[["experiment_id", "sample_id", "study_id", "scope"]].sort_values("experiment_id")
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} experiments ({df['sample_id'].replace('', pd.NA).nunique()} "
          f"patients, {int((df['sample_id'] == '').sum())} without sample_id) -> {args.output}")
    print(df.groupby(["study_id", "scope"]).size().to_string())

    if args.clinical_output:
        clin = clinical_table(database, args.class_config, args.clinical_spec)
        clin.to_csv(args.clinical_output, index=False)
        print(f"wrote {len(clin)} experiments x {clin.shape[1]} columns -> "
              f"{args.clinical_output}")


if __name__ == "__main__":
    main()
