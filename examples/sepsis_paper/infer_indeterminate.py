"""Ensemble-predict the indeterminate sepsis samples with a LOO fold set.

The 3-class sepsis/ICU-ctrl/healthy resnet50-mil LOO run excluded the
``sepsis_positive = None`` patients (null labels) from every fold, so all N fold
checkpoints are unbiased predictors for them. This script points EVERY fold at the
SAME indeterminate experiment IDs at one fixed epoch, then averages the softmax
across folds -- the "on average" LOO ensemble prediction.

Mirrors ``remo_id.workflows.loo_predict.predict_epoch`` but with a fixed external
predict set instead of each fold's held-out validation set.

Indeterminate set: by default derived from the manifest (``--manifest``) as the
experiments whose ``label`` is empty *and* which carry a free-text clinical
diagnosis (``patient_notes``). Patients enrolled but not yet adjudicated have
neither a label nor notes and are therefore excluded.

Repeats and variation
---------------------
``--repeats N`` predicts each experiment N times with independent bags and keeps
every repeat separately, so prediction spread can be decomposed into

  * ``sd_fold``   -- between-fold (ensemble member) disagreement
  * ``sd_repeat`` -- within-fold bag-resampling noise
  * ``sd_exp``    -- (patient level) between-replicate-experiment spread

Each repeat is a single-bag call seeded so that repeat ``r`` draws exactly the bag
that ``n_bags=N`` would have drawn as its ``r``-th bag. The reported probability is
therefore the mean of per-bag softmax outputs (``n_bags=N`` instead averages bag
logits before the softmax -- a marginally different estimator).

The images are loaded from the manifest's paths (no database). Output feeds the
indeterminate_patients / indeterminate_experiments tables of make_paper_results.py.

Usage:
    # one fold (SLURM array element) -> writes a shard
    python examples/sepsis_paper/infer_indeterminate.py \
        --cv_dir <cv_dir> --manifest manifest.csv --epoch 50 \
        --bag_size 500 --repeats 5 --fold_index $SLURM_ARRAY_TASK_ID

    # merge all shards -> {cv_dir}/analysis/indeterminate/ensemble_probs_epoch50.npz
    python examples/sepsis_paper/infer_indeterminate.py \
        --cv_dir <cv_dir> --manifest manifest.csv --epoch 50 \
        --bag_size 500 --repeats 5 --aggregate
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from remo_id.data.class_config import class_config_from_dict
from remo_id.data.manifest import manifest_class_config
from remo_id.inference.core import (
    extract_checkpoint_metadata,
    load_checkpoint,
    set_inference_seeds,
)
from remo_id.inference.mil import _predict_experiments
from remo_id.data import RemoIDDataModule
from remo_id.models import get_augmentation_backbone
from remo_id.workflows.loo_report import _find_epoch_checkpoint

from common import Manifest, resolve_classes

# _predict_experiments derives its per-bag seed as
#   bag_seed = seed + 1009 * experiment_index + 7919 * bag_index
# so stepping the base seed by 7919 per repeat reproduces the bag sequence of a
# single n_bags=N call, one bag at a time.
_BAG_SEED_STRIDE = 7919


def fold_dirs_for(cv_dir):
    dirs = sorted(
        glob.glob(os.path.join(cv_dir, "repeat*_fold*")),
        key=lambda p: int(p.split("fold")[-1]),
    )
    if not dirs:
        raise RuntimeError(f"No repeat*_fold* dirs in {cv_dir}")
    return dirs


def resolve_indeterminate_ids(manifest, explicit=None):
    """Experiments with an empty label but a recorded clinical diagnosis."""
    if explicit:
        return [str(e) for e in explicit]
    t = manifest.table
    null_label = t[t["label"].isna()]
    if "patient_notes" not in t.columns or t["patient_notes"].isna().all():
        raise SystemExit("the manifest has no patient_notes; pass --experiment_ids")
    has_dx = null_label["patient_notes"].notna()
    keep = null_label[has_dx]

    dropped = null_label[~has_dx]
    if len(dropped):
        print(f"Excluding {len(dropped)} null-label experiment(s) with no clinical "
              f"diagnosis (unadjudicated or outside the classes)")
    ids = sorted(str(e) for e in keep["experiment_id"])
    print(f"Indeterminate set from the manifest: {len(ids)} experiments / "
          f"{keep['sample_id'].nunique()} patients")
    return ids


def build_predict_datamodule(manifest, req_ids, class_names, output_dir, backbone_name):
    """Loader for the explicit indeterminate IDs.

    Their manifest rows have no label, so a manifest datamodule would skip them:
    write them to ``{output_dir}/indeterminate_manifest.csv`` labelled with the
    first class (the label is meaningless here; the outputs are the 3-class softmax).
    """
    t = manifest.table
    rows = t[t["experiment_id"].isin(req_ids)].copy()
    if rows.empty:
        raise RuntimeError(f"None of {req_ids} are in the manifest.")
    missing = sorted(set(req_ids) - set(rows["experiment_id"]))
    if missing:
        print(f"WARNING: requested IDs not in the manifest: {missing}")
    rows["label"] = class_names[0]
    path = os.path.join(output_dir, "indeterminate_manifest.csv")
    rows[["experiment_id", "sample_id", "study_id", "path", "num_images", "label"]].to_csv(
        path, index=False)
    cfg = class_config_from_dict(manifest_class_config(path, classes=class_names,
                                                       require_present=False))
    ds = RemoIDDataModule(
        class_definitions=cfg.class_definitions,
        batch_size=1,
        backbone_name=backbone_name,
        train=False,
        task_type=cfg.task_type,
        num_classes=cfg.num_classes,
        background_correction="none",
    )
    df = ds.data.copy()
    df["experiment_id"] = df["experiment_id"].astype(str)
    # Sorted so every shard shares one experiment ordering.
    df = df.sort_values("experiment_id").reset_index(drop=True)
    print(f"Resolved {len(df)} experiments")
    return ds, df


def predict_one_fold(fold_dir, args, ds, df, class_names, num_classes, task_type,
                     default_model_name, device):
    """(n_repeats, n_exp, n_classes) softmax probs for one fold, or None."""
    ckpt_path = _find_epoch_checkpoint(fold_dir, args.epoch)
    if ckpt_path is None:
        print(f"  {os.path.basename(fold_dir)}: no epoch={args.epoch} checkpoint")
        return None, None

    model_name = extract_checkpoint_metadata(ckpt_path).get(
        "model_name", default_model_name
    )
    model, device_used, _ = load_checkpoint(
        ckpt_path, model_name, task_type=task_type, num_classes=num_classes,
        n_experiments=1, device=device,
    )

    exp_ids = list(df["experiment_id"])
    out = np.full((args.repeats, len(exp_ids), num_classes), np.nan)
    for r in range(args.repeats):
        # One bag per call, seeded to match bag r of an n_bags=repeats run.
        seed_r = args.seed + _BAG_SEED_STRIDE * r
        set_inference_seeds(seed_r)
        results, _, _, _, _, _ = _predict_experiments(
            model=model, ds=ds, df=df,
            database_column=ds.database_column,
            task_type=task_type, num_classes=num_classes,
            bag_size=args.bag_size, seed=seed_r, device=device_used,
            pred_threshold=0.5, class_names=class_names, n_bags=1,
        )
        for res in results:
            if res.y_probs_all is None or np.any(np.isnan(res.y_probs_all)):
                continue
            out[r, exp_ids.index(res.experiment_id)] = np.asarray(
                res.y_probs_all, dtype=float
            )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mean_p = np.nanmean(out, axis=0)
    preds = {
        e: class_names[int(np.argmax(mean_p[i]))]
        for i, e in enumerate(exp_ids)
        if not np.all(np.isnan(mean_p[i]))
    }
    print(f"  {os.path.basename(fold_dir)}: {os.path.basename(ckpt_path)} -> {preds}")
    return out, ckpt_path


def _summarise(arr, class_names, repeat_axis_arr=None):
    """Collapse (n_folds, n_draws, n_classes) -> mean probs + spread components.

    ``repeat_axis_arr``: (n_folds, n_repeats, ..., n_classes) used for the pure
    bag-resampling SD. Pass it whenever ``arr`` pooled extra draws (e.g. a
    patient's replicate experiments) into axis 1, so replicate spread is averaged
    over rather than counted as bag noise. Defaults to ``arr``.
    """
    per_fold = np.nanmean(arr, axis=1)                    # (n_folds, n_classes)
    valid = per_fold[~np.isnan(per_fold).any(axis=1)]
    mean_p = valid.mean(axis=0)

    # Between-fold spread of the per-fold mean probability.
    sd_fold = valid.std(axis=0)
    # Within-fold bag-resampling spread, averaged over folds (and replicates).
    src = arr if repeat_axis_arr is None else repeat_axis_arr
    with np.errstate(invalid="ignore"):
        per = np.nanstd(src, axis=1)
        sd_repeat = np.nanmean(per, axis=tuple(range(per.ndim - 1)))
    votes = np.bincount(valid.argmax(axis=1), minlength=len(class_names)) / len(valid)
    return mean_p, sd_fold, sd_repeat, votes, len(valid)


def _spread_columns(row, class_names, mean_p, sd_fold, sd_repeat, votes):
    for ci, cn in enumerate(class_names):
        row[f"P({cn})"] = round(float(mean_p[ci]), 4)
    for ci, cn in enumerate(class_names):
        row[f"sd_fold({cn})"] = round(float(sd_fold[ci]), 4)
    for ci, cn in enumerate(class_names):
        row[f"sd_repeat({cn})"] = round(float(sd_repeat[ci]), 4)
    for ci, cn in enumerate(class_names):
        row[f"vote_frac({cn})"] = round(float(votes[ci]), 4)
    return row


def aggregate(args, output_dir, class_names, num_classes):
    shard_dir = os.path.join(output_dir, "shards")
    shards = sorted(glob.glob(os.path.join(shard_dir, f"*_epoch{args.epoch}.npz")))
    if not shards:
        raise RuntimeError(f"No shards in {shard_dir}")

    expected = len(fold_dirs_for(args.cv_dir))
    if len(shards) != expected:
        print(f"WARNING: {len(shards)} shards but {expected} folds -- incomplete run")

    fold_labels, blocks, exp_ids, sample_ids = [], [], None, None
    for s in shards:
        z = np.load(s, allow_pickle=False)
        ids = [str(e) for e in z["experiment_ids"]]
        if exp_ids is None:
            exp_ids, sample_ids = ids, [str(x) for x in z["sample_ids"]]
        elif ids != exp_ids:
            raise RuntimeError(f"Shard {s} experiment order differs from the first shard")
        blocks.append(z["probs"])
        fold_labels.append(str(z["fold_label"]))

    # (n_folds, n_repeats, n_exp, n_classes)
    arr = np.stack(blocks, axis=0)
    print(
        f"Aggregated {arr.shape[0]} folds x {arr.shape[1]} repeats x "
        f"{arr.shape[2]} experiments"
    )

    npz_path = os.path.join(output_dir, f"ensemble_probs_epoch{args.epoch}.npz")
    np.savez(
        npz_path,
        probs=arr,
        fold_labels=np.array(fold_labels, dtype=str),
        experiment_ids=np.array(exp_ids, dtype=str),
        sample_ids=np.array(sample_ids, dtype=str),
        class_names=np.array(class_names, dtype=str),
        epoch=np.array(args.epoch),
        bag_size=np.array(args.bag_size),
        repeats=np.array(args.repeats),
    )

    # ---- experiment level ----
    exp_rows = []
    for ei, e in enumerate(exp_ids):
        mean_p, sd_fold, sd_repeat, votes, n_valid = _summarise(
            arr[:, :, ei, :], class_names
        )
        row = {"experiment_id": e, "sample_id": sample_ids[ei], "n_folds": n_valid,
               "ensemble_pred": class_names[int(mean_p.argmax())]}
        exp_rows.append(_spread_columns(row, class_names, mean_p, sd_fold, sd_repeat, votes))
    exp_df = pd.DataFrame(exp_rows)
    exp_csv = os.path.join(output_dir, f"indeterminate_predictions_epoch{args.epoch}.csv")
    exp_df.to_csv(exp_csv, index=False)

    # ---- patient level: pool every experiment of a patient ----
    pat_rows = []
    for sid in sorted(set(sample_ids)):
        cols = [i for i, s in enumerate(sample_ids) if s == sid]
        mean_p, sd_fold, sd_repeat, votes, n_valid = _summarise(
            arr[:, :, cols, :].reshape(arr.shape[0], -1, arr.shape[3]), class_names,
            repeat_axis_arr=arr[:, :, cols, :],
        )
        row = {"sample_id": sid, "n_experiments": len(cols), "n_folds": n_valid,
               "ensemble_pred": class_names[int(mean_p.argmax())]}
        row = _spread_columns(row, class_names, mean_p, sd_fold, sd_repeat, votes)
        # Between-replicate spread: std across this patient's experiment means.
        per_exp = np.stack([
            np.nanmean(arr[:, :, i, :].reshape(-1, arr.shape[3]), axis=0) for i in cols
        ])
        for ci, cn in enumerate(class_names):
            row[f"sd_exp({cn})"] = (
                round(float(per_exp[:, ci].std(ddof=0)), 4) if len(cols) > 1 else 0.0
            )
        pat_rows.append(row)
    pat_df = pd.DataFrame(pat_rows)
    pat_csv = os.path.join(
        output_dir, f"indeterminate_predictions_patient_epoch{args.epoch}.csv"
    )
    pat_df.to_csv(pat_csv, index=False)

    print("\n" + "=" * 78)
    print(f"INDETERMINATE ENSEMBLE  (epoch {args.epoch}, {arr.shape[0]} folds x "
          f"{args.repeats} repeats, bag {args.bag_size})")
    print("=" * 78)
    with pd.option_context("display.width", 220, "display.max_columns", 60):
        print("\n-- experiment level --")
        print(exp_df.to_string(index=False))
        print("\n-- patient level --")
        print(pat_df.to_string(index=False))
    print(f"\nSaved: {exp_csv}")
    print(f"Saved: {pat_csv}")
    print(f"Saved: {npz_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cv_dir", required=True)
    ap.add_argument("--manifest", required=True,
                    help="manifest CSV (indeterminate set and image paths)")
    ap.add_argument("--classes", default=None,
                    help="Comma-separated class names in index order (default: the "
                         "model's, from the fold classes.yaml; must match it)")
    ap.add_argument("--epoch", type=int, default=50,
                    help="Fixed epoch checkpoint used across all folds")
    ap.add_argument("--experiment_ids", nargs="+", default=None,
                    help="Explicit indeterminate experiment IDs (default: empty label "
                         "and a recorded diagnosis in the manifest)")
    ap.add_argument("--bag_size", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=5,
                    help="Independent bags per experiment per fold")
    ap.add_argument("--seed", type=int, default=403)
    ap.add_argument("--fold_index", type=int, default=None,
                    help="Predict only this fold (SLURM array element) and write a shard")
    ap.add_argument("--aggregate", action="store_true",
                    help="Merge existing shards into the summary tables and exit")
    ap.add_argument("--output_dir", default=None,
                    help="Default: {cv_dir}/analysis/indeterminate/")
    args = ap.parse_args()

    output_dir = args.output_dir or os.path.join(args.cv_dir, "analysis", "indeterminate")
    os.makedirs(output_dir, exist_ok=True)

    class_names = resolve_classes(args.cv_dir, args.classes)
    num_classes = len(class_names)
    task_type = "multiclass" if num_classes > 2 else "binary"

    if args.aggregate:
        aggregate(args, output_dir, class_names, num_classes)
        return

    manifest = Manifest(args.manifest)
    req_ids = resolve_indeterminate_ids(manifest, args.experiment_ids)

    fold_dirs = fold_dirs_for(args.cv_dir)
    if args.fold_index is not None:
        if args.fold_index >= len(fold_dirs):
            raise SystemExit(
                f"fold_index {args.fold_index} out of range ({len(fold_dirs)} folds)"
            )
        fold_dirs = [fold_dirs[args.fold_index]]

    default_model_name = "resnet18-mil"
    for c in glob.glob(os.path.join(fold_dirs[0], "*.ckpt")):
        try:
            default_model_name = extract_checkpoint_metadata(c).get(
                "model_name", default_model_name
            )
            break
        except Exception:  # noqa: BLE001
            continue
    backbone_name = get_augmentation_backbone(default_model_name)
    print(f"Model: {default_model_name} (backbone {backbone_name}) | classes {class_names}")
    print(f"epoch={args.epoch} bag_size={args.bag_size} repeats={args.repeats} | "
          f"{len(fold_dirs)} fold(s)")

    ds, df = build_predict_datamodule(manifest, req_ids, class_names, output_dir,
                                      backbone_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    shard_dir = os.path.join(output_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    for fd in fold_dirs:
        fold_label = os.path.basename(fd)
        probs, _ckpt = predict_one_fold(
            fd, args, ds, df, class_names, num_classes, task_type,
            default_model_name, device,
        )
        if probs is None:
            continue
        np.savez(
            os.path.join(shard_dir, f"{fold_label}_epoch{args.epoch}.npz"),
            probs=probs,                                  # (n_repeats, n_exp, n_classes)
            fold_label=np.array(fold_label),
            experiment_ids=np.array(list(df["experiment_id"]), dtype=str),
            sample_ids=np.array([str(s) for s in df["sample_id"]], dtype=str),
            class_names=np.array(class_names, dtype=str),
            epoch=np.array(args.epoch),
            bag_size=np.array(args.bag_size),
        )

    print(f"Shards written to {shard_dir}")
    if args.fold_index is None:
        aggregate(args, output_dir, class_names, num_classes)


if __name__ == "__main__":
    main()
