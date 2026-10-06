"""LOO CV comprehensive report workflow.

Evaluates all folds at the same epoch, sweeps epochs to find the best one,
generates performance tables, figures, error analysis, and assembles a report.
"""

import glob as glob_module
import json
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from remo_id.inference.core import (
    load_checkpoint,
    extract_checkpoint_metadata,
    set_inference_seeds,
)
from remo_id.inference.mil import _predict_experiments
from remo_id.training.cross_validation import load_cv_splits
from remo_id.workflows.cv import _deduplicate_pooled_predictions
from remo_id.analysis.optimal_thresholds import (
    find_youden_thresholds,
    find_optimal_thresholds,
    build_sensitivity_targets,
    dual_operating_points_table,
)
from remo_id.analysis.pairwise import run_pairwise
from remo_id.analysis.error_analysis import identify_misclassifications_from_arrays
from remo_id.analysis.performance_table import (
    compute_full_metrics_at_thresholds,
    compute_argmax_metrics_table,
    compute_fold_f1_std,
)
from remo_id.reports import style
from remo_id.reports.figures import (
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrix,
    plot_calibration,
    plot_prob_distributions,
    plot_threshold_sensitivity,
    plot_kfold_variability,
    plot_dual_operating_points,
    plot_bag_size_sweep,
    plot_bag_variance,
    plot_error_confidence,
    plot_error_by_study,
    plot_confusion_errors,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _find_common_epochs(cv_dir: str, include_best_last: bool = False) -> list[int]:
    """Scan fold dirs for epoch=*.ckpt files, return epochs present in ALL folds.

    By default only returns numbered epoch checkpoints. When include_best_last=True
    (or when no epoch checkpoints exist), appends sentinels:
        -2: best.ckpt
        -1: last.ckpt

    This prevents the epoch sweep from selecting best.ckpt, which uses a
    different epoch per fold and breaks the LOO same-epoch requirement.
    """
    fold_dirs = sorted(glob_module.glob(os.path.join(cv_dir, "repeat*_fold*")))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found in {cv_dir}")

    per_fold_epochs = []
    for fd in fold_dirs:
        ckpts = glob_module.glob(os.path.join(fd, "epoch=*.ckpt"))
        epochs = set()
        for c in ckpts:
            base = os.path.basename(c)
            # Handle: epoch=N.ckpt, epoch=N-step=M.ckpt, epoch=epoch=N.ckpt
            try:
                # Take the last "epoch=" segment to handle double-prefix
                parts = base.split("epoch=")
                last_part = parts[-1]  # e.g. "14.ckpt" or "14-step=100.ckpt"
                epoch_num = last_part.split("-")[0].split(".")[0]
                epochs.add(int(epoch_num))
            except (IndexError, ValueError):
                continue
        per_fold_epochs.append(epochs)

    common = per_fold_epochs[0]
    for s in per_fold_epochs[1:]:
        common = common & s

    result = sorted(common) if common else []

    # Only append best/last when explicitly requested or as fallback
    if include_best_last or not result:
        all_have_best = all(
            os.path.exists(os.path.join(fd, "best.ckpt")) for fd in fold_dirs
        )
        all_have_last = all(
            os.path.exists(os.path.join(fd, "last.ckpt")) for fd in fold_dirs
        )
        if all_have_best:
            result.append(-2)
        if all_have_last:
            result.append(-1)

    if not result:
        return []

    if not common:
        names = []
        if -2 in result:
            names.append("best.ckpt")
        if -1 in result:
            names.append("last.ckpt")
        print(f"No common epoch checkpoints found. Available: {', '.join(names)}")

    return result


def _find_epoch_checkpoint(fold_dir: str, epoch: int) -> Optional[str]:
    """Find the checkpoint file for a specific epoch in a fold directory.

    ``epoch=N`` is the weights after N training epochs (``epoch=N.ckpt``). When no
    such file exists, ``last.ckpt`` is used if it was written after exactly N
    epochs (runs trained without ``--save_epochs`` only keep ``last.ckpt``).

    Special values:
        epoch=-1: use last.ckpt
        epoch=-2: use best.ckpt
    """
    if epoch == -1:
        last_path = os.path.join(fold_dir, "last.ckpt")
        return last_path if os.path.exists(last_path) else None
    if epoch == -2:
        best_path = os.path.join(fold_dir, "best.ckpt")
        return best_path if os.path.exists(best_path) else None

    # Try multiple patterns to handle epoch=N.ckpt and epoch=epoch=N.ckpt
    for pattern in [
        os.path.join(fold_dir, f"epoch={epoch}.ckpt"),
        os.path.join(fold_dir, f"epoch={epoch}-*.ckpt"),
        os.path.join(fold_dir, f"epoch=epoch={epoch}.ckpt"),
        os.path.join(fold_dir, f"epoch=epoch={epoch}-*.ckpt"),
    ]:
        matches = glob_module.glob(pattern)
        if matches:
            return matches[0]

    # Broad search fallback
    all_epoch_ckpts = glob_module.glob(os.path.join(fold_dir, "epoch=*.ckpt"))
    for m in all_epoch_ckpts:
        base = os.path.basename(m)
        try:
            parts = base.split("epoch=")
            epoch_num = int(parts[-1].split("-")[0].split(".")[0])
            if epoch_num == epoch:
                return m
        except (IndexError, ValueError):
            continue

    last_path = os.path.join(fold_dir, "last.ckpt")
    if epoch > 0 and os.path.exists(last_path) and _checkpoint_epochs_trained(last_path) == epoch:
        return last_path
    return None


def _checkpoint_epochs_trained(ckpt_path: str) -> Optional[int]:
    """Number of completed training epochs stored in a Lightning checkpoint.

    Lightning stores the 0-indexed ``epoch`` of the epoch in progress, so a
    checkpoint saved at the end of epoch N (``epoch=N.ckpt``) stores ``N - 1``.
    The file is memory-mapped, so only the small ``epoch`` entry is materialized.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    return int(epoch) + 1 if epoch is not None else None


def _find_int_in_nested(obj, key):
    """Recursively search a nested dict/list for `key`, returning its int value.

    Handles the wandb config.yaml shape where each entry is {key: {value: X}}.
    """
    if isinstance(obj, dict):
        if key in obj:
            v = obj[key]
            v = v.get("value") if isinstance(v, dict) and "value" in v else v
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        for vv in obj.values():
            r = _find_int_in_nested(vv, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for vv in obj:
            r = _find_int_in_nested(vv, key)
            if r is not None:
                return r
    return None


def _extract_training_bag_size(cv_dir: str) -> int:
    """Resolve the bag size the folds were TRAINED at.

    MIL checkpoints do NOT store ``batch_size`` (it's a data/training arg, not a
    model hyperparameter), so reading it from the checkpoint silently falls back
    to the default 200 and yields a wrong bag (e.g. 200/3=66 for a model trained
    at batch_size 1500). We therefore read the real value from the fold's wandb
    training config first:
      1. wandb config.yaml ``samples_per_experiment`` (the actual training bag);
      2. wandb config.yaml ``batch_size // n_experiments``;
      3. checkpoint hparams ``batch_size // n_experiments`` (only if batch_size
         was actually stored);
      4. last-resort default 200/8 with a loud warning.
    Pass ``--bag_size`` to override.
    """
    import yaml
    fold_dirs = sorted(glob_module.glob(os.path.join(cv_dir, "repeat*_fold*")))

    # 1+2. wandb training config (most reliable)
    for cfg in sorted(glob_module.glob(
            os.path.join(cv_dir, "repeat*_fold*", "wandb", "*", "files", "config.yaml"))):
        try:
            d = yaml.safe_load(open(cfg))
        except Exception:
            continue
        spe = _find_int_in_nested(d, "samples_per_experiment")
        if spe:
            print(f"Training bag size {spe} (from wandb samples_per_experiment)")
            return spe
        bsz = _find_int_in_nested(d, "batch_size")
        nexp = _find_int_in_nested(d, "n_experiments")
        if bsz and nexp:
            print(f"Training bag size {bsz // nexp} (from wandb batch_size//n_experiments)")
            return bsz // nexp

    # 3. checkpoint hparams — only trust if batch_size is actually present
    for fd in fold_dirs:
        for c in glob_module.glob(os.path.join(fd, "*.ckpt")):
            try:
                ckpt = torch.load(c, map_location="cpu")
                hp = ckpt.get("hyper_parameters", {})
                if hp.get("batch_size") and hp.get("n_experiments"):
                    bs = hp["batch_size"] // hp["n_experiments"]
                    print(f"Training bag size {bs} (from checkpoint hparams)")
                    return bs
            except Exception:
                continue

    print("WARNING: could not determine training bag size from wandb config or "
          "checkpoint hparams; defaulting to 200/8=25. Pass --bag_size explicitly.")
    return 200 // 8


def _get_fold_val_paths(cv_dir: str, data_module) -> dict:
    """Load cv_splits.yaml and return {fold_label: val_paths_set}."""
    splits_path = os.path.join(cv_dir, "cv_splits.yaml")
    all_repeats = load_cv_splits(splits_path, data_module)
    result = {}
    for repeat_idx, folds in enumerate(all_repeats):
        for fold_idx, (train_paths, val_paths) in enumerate(folds):
            label = f"repeat{repeat_idx}_fold{fold_idx}"
            result[label] = val_paths
    return result


def apply_thresholds_to_predictions(y_probs, class_names, thresholds):
    """Pick class with highest margin above its threshold.

    margin_c = p_c - threshold_c

    Among classes above their threshold (margin > 0), pick highest margin.
    If none exceed, pick highest margin (least negative = closest to positive).
    """
    y_probs = np.asarray(y_probs)
    thr_arr = np.array([thresholds[c] for c in class_names])
    margins = y_probs - thr_arr  # (N, C)
    y_pred = margins.argmax(axis=1)
    return y_pred


# ── per-fold prediction ─────────────────────────────────────────────────

def _predict_fold_at_epoch(
    fold_dir: str,
    epoch: int,
    data_module,
    class_names: list,
    bag_size: int,
    n_bags: int,
    seed: int,
    experiment_paths: set = None,
) -> Optional[dict]:
    """Load epoch checkpoint from fold_dir, predict specified experiments.

    If experiment_paths is None, uses the val paths from cv_splits.
    Returns dict with y_true, y_probs, y_pred, experiment_ids, study_ids.
    """
    ckpt_path = _find_epoch_checkpoint(fold_dir, epoch)
    if ckpt_path is None:
        print(f"Warning: no epoch={epoch} checkpoint in {fold_dir}")
        return None

    task_type = data_module.task_type
    num_classes = data_module.num_classes

    ckpt_meta = extract_checkpoint_metadata(ckpt_path)
    model_name = ckpt_meta.get("model_name", "resnet18-mil")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_inference_seeds(seed)

    model, device, _ = load_checkpoint(
        ckpt_path, model_name,
        task_type=task_type, num_classes=num_classes,
        n_experiments=1, device=device,
    )

    # Build a filtered data module for the target experiments
    if experiment_paths is not None:
        all_paths = set(data_module.data["path"].values)
        other_paths = all_paths - experiment_paths
        _, target_module = data_module.split_from_paths(other_paths, experiment_paths)
    else:
        # Default: use all data
        target_module = data_module

    df = target_module.data.copy()
    df["experiment_id"] = df["experiment_id"].astype(str)

    results, yt, yp, ypred, _, _ = _predict_experiments(
        model=model, ds=target_module, df=df,
        database_column=target_module.database_column,
        task_type=task_type, num_classes=num_classes,
        bag_size=bag_size, seed=seed, device=device,
        pred_threshold=0.5,
        class_names=class_names,
        n_bags=n_bags,
    )

    valid = [r for r in results if not np.isnan(r.y_prob) and not r.has_null_label]
    if not valid:
        return None

    fold_yt = np.array([r.y_true for r in valid], dtype=int)
    fold_yp = np.array([r.y_probs_all for r in valid], dtype=float)
    fold_ypred = np.array([r.y_pred for r in valid], dtype=int)
    fold_exp_ids = np.array([r.experiment_id for r in valid])
    fold_study_ids = np.array([getattr(r, "study_id", "") for r in valid])

    # Clean up GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fold_label = os.path.basename(fold_dir)
    parts = fold_label.replace("repeat", "").replace("fold", " ").split("_")
    try:
        repeat_idx = int(parts[0])
        fold_idx = int(parts[1].strip())
    except (ValueError, IndexError):
        repeat_idx, fold_idx = 0, 0

    return {
        "repeat": repeat_idx,
        "fold": fold_idx,
        "y_true": fold_yt,
        "y_probs": fold_yp,
        "y_pred": fold_ypred,
        "experiment_ids": fold_exp_ids,
        "study_ids": fold_study_ids,
    }


# ── epoch sweep ──────────────────────────────────────────────────────────

def sweep_epochs(
    cv_dir: str,
    data_module,
    class_names: list,
    training_bag_size: int,
    n_bags: int,
    seed: int,
) -> tuple:
    """Sweep all common epochs, predict held-out folds, pick best by pooled macro F1.

    Returns
    -------
    epoch_numbers : sorted list of epochs evaluated
    metrics_per_epoch : {epoch: {"macro_f1": float, "macro_auc": float, "per_class_f1": dict}}
    best_epoch_predictions : {"pooled": (y_true, y_probs, y_pred, exp_ids, study_ids),
                              "per_fold": list}
    """
    epoch_numbers = _find_common_epochs(cv_dir)
    if not epoch_numbers:
        raise ValueError(f"No common epoch checkpoints found across all folds in {cv_dir}")

    fold_dirs = sorted(glob_module.glob(os.path.join(cv_dir, "repeat*_fold*")))
    fold_val_paths = _get_fold_val_paths(cv_dir, data_module)

    print(f"Found {len(epoch_numbers)} common epochs: {epoch_numbers}")
    print(f"Sweeping {len(fold_dirs)} folds x {len(epoch_numbers)} epochs...")

    metrics_per_epoch = {}
    best_f1 = -1.0
    best_epoch = epoch_numbers[0]
    best_predictions = None

    for epoch in epoch_numbers:
        print(f"\n--- Epoch {epoch} ---")
        per_fold = []

        for fd in fold_dirs:
            fold_label = os.path.basename(fd)
            val_paths = fold_val_paths.get(fold_label, set())
            if not val_paths:
                print(f"  Warning: no val paths for {fold_label}, skipping")
                continue

            result = _predict_fold_at_epoch(
                fold_dir=fd, epoch=epoch,
                data_module=data_module, class_names=class_names,
                bag_size=training_bag_size, n_bags=n_bags, seed=seed,
                experiment_paths=val_paths,
            )
            if result is not None:
                per_fold.append(result)

        if not per_fold:
            print(f"  No valid predictions at epoch {epoch}")
            continue

        # Pool predictions and compute metrics
        y_true, y_probs, y_pred, exp_ids, study_ids, _sample_ids = _deduplicate_pooled_predictions(per_fold)

        # Compute Youden-J thresholds for this epoch
        youden = find_youden_thresholds(y_true, y_probs, class_names)
        thresholds = {r["class_name"]: r["optimal_threshold"] for r in youden}

        # Apply thresholds for F1 computation
        y_pred_thr = apply_thresholds_to_predictions(y_probs, class_names, thresholds)

        # Per-class OvR F1 at Youden-J thresholds
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize

        per_class_f1 = {}
        for c, name in enumerate(class_names):
            y_bin = (y_true == c).astype(int)
            y_sc = y_probs[:, c]
            thr = thresholds[name]
            pred = (y_sc >= thr).astype(int)
            tp = int(((pred == 1) & (y_bin == 1)).sum())
            fp = int(((pred == 1) & (y_bin == 0)).sum())
            fn = int(((pred == 0) & (y_bin == 1)).sum())
            ppv = tp / max(1, tp + fp)
            sens = tp / max(1, tp + fn)
            f1 = 2 * ppv * sens / max(1e-12, ppv + sens)
            per_class_f1[name] = f1

        macro_f1 = np.mean(list(per_class_f1.values()))

        # Argmax metrics
        y_pred_argmax = y_probs.argmax(axis=1)
        argmax_per_class_f1 = {}
        for c, name in enumerate(class_names):
            tp = int(((y_pred_argmax == c) & (y_true == c)).sum())
            fp = int(((y_pred_argmax == c) & (y_true != c)).sum())
            fn = int(((y_pred_argmax != c) & (y_true == c)).sum())
            ppv = tp / max(1, tp + fp)
            sens = tp / max(1, tp + fn)
            f1 = 2 * ppv * sens / max(1e-12, ppv + sens)
            argmax_per_class_f1[name] = f1
        argmax_macro_f1 = np.mean(list(argmax_per_class_f1.values()))
        argmax_accuracy = float((y_pred_argmax == y_true).mean())

        # Macro AUC
        try:
            yb = label_binarize(y_true, classes=list(range(len(class_names))))
            if len(class_names) == 2:
                yb = np.column_stack([1 - yb, yb])
            macro_auc = float(roc_auc_score(yb, y_probs, average="macro", multi_class="ovr"))
        except Exception:
            macro_auc = float("nan")

        metrics_per_epoch[epoch] = {
            "macro_f1": float(macro_f1),
            "macro_auc": float(macro_auc),
            "per_class_f1": per_class_f1,
            "argmax_macro_f1": float(argmax_macro_f1),
            "argmax_accuracy": float(argmax_accuracy),
            "argmax_per_class_f1": argmax_per_class_f1,
        }

        print(f"  macro F1={macro_f1:.4f}, argmax F1={argmax_macro_f1:.4f}, macro AUC={macro_auc:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            best_predictions = {
                "pooled": (y_true, y_probs, y_pred_thr, exp_ids, study_ids),
                "per_fold": per_fold,
            }

    print(f"\nBest epoch: {best_epoch} (macro F1={best_f1:.4f})")
    return epoch_numbers, metrics_per_epoch, best_predictions, best_epoch


# ── error consistency ────────────────────────────────────────────────────

def compute_error_consistency(
    cv_dir: str,
    data_module,
    class_names: list,
    best_epoch: int,
    training_bag_size: int,
    seeds: list = None,
) -> pd.DataFrame:
    """Predict ALL experiments from every fold checkpoint x multiple seeds.

    Returns DataFrame with experiment_id, study_id, true_class, n_runs,
    n_correct, n_errors, error_rate, most_common_prediction.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1011]

    fold_dirs = sorted(glob_module.glob(os.path.join(cv_dir, "repeat*_fold*")))
    all_paths = set(data_module.data["path"].values)

    # Collect all predictions: {experiment_id: [(true_class, pred_class), ...]}
    exp_preds = defaultdict(list)
    exp_meta = {}  # experiment_id -> {study_id, true_class}

    detail_rows = []

    for fd in fold_dirs:
        fold_label = os.path.basename(fd)
        for s in seeds:
            print(f"  Error consistency: {fold_label}, seed={s}")
            result = _predict_fold_at_epoch(
                fold_dir=fd, epoch=best_epoch,
                data_module=data_module, class_names=class_names,
                bag_size=training_bag_size, n_bags=1, seed=s,
                experiment_paths=all_paths,
            )
            if result is None:
                continue

            for i, eid in enumerate(result["experiment_ids"]):
                true_cls = int(result["y_true"][i])
                pred_cls = int(result["y_pred"][i])
                sid = result["study_ids"][i] if "study_ids" in result else ""
                exp_preds[eid].append((true_cls, pred_cls))
                exp_meta[eid] = {
                    "study_id": sid,
                    "true_class": class_names[true_cls] if true_cls < len(class_names) else str(true_cls),
                    "true_class_idx": true_cls,
                }
                detail_rows.append({
                    "experiment_id": eid,
                    "fold": fold_label,
                    "seed": s,
                    "true_class": class_names[true_cls] if true_cls < len(class_names) else str(true_cls),
                    "predicted_class": class_names[pred_cls] if pred_cls < len(class_names) else str(pred_cls),
                    "correct": true_cls == pred_cls,
                })

    # Aggregate per experiment
    rows = []
    for eid, preds in exp_preds.items():
        meta = exp_meta[eid]
        true_idx = meta["true_class_idx"]
        n_runs = len(preds)
        n_correct = sum(1 for _, p in preds if p == true_idx)
        n_errors = n_runs - n_correct

        # Most common prediction
        from collections import Counter
        pred_counts = Counter(p for _, p in preds)
        most_common_idx = pred_counts.most_common(1)[0][0]
        most_common_name = class_names[most_common_idx] if most_common_idx < len(class_names) else str(most_common_idx)

        rows.append({
            "experiment_id": eid,
            "study_id": meta["study_id"],
            "true_class": meta["true_class"],
            "n_runs": n_runs,
            "n_correct": n_correct,
            "n_errors": n_errors,
            "error_rate": n_errors / max(1, n_runs),
            "most_common_prediction": most_common_name,
        })

    consistency_df = pd.DataFrame(rows).sort_values("error_rate", ascending=False)
    detail_df = pd.DataFrame(detail_rows)

    return consistency_df, detail_df


# ── bag size sweep ───────────────────────────────────────────────────────

def run_cv_bag_size_sweep(
    cv_dir: str,
    data_module,
    class_names: list,
    best_epoch: int,
    bag_sizes: list,
    seeds: list = None,
) -> tuple:
    """At best epoch, for each (bag_size, seed), predict all folds and compute macro AUC.

    Returns (bag_sizes, aucs_by_seed) compatible with plot_bag_size_sweep.
    """
    if seeds is None:
        seeds = [42, 123, 456]

    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    fold_dirs = sorted(glob_module.glob(os.path.join(cv_dir, "repeat*_fold*")))
    fold_val_paths = _get_fold_val_paths(cv_dir, data_module)
    task_type = data_module.task_type
    num_classes = data_module.num_classes

    def _pool_auc(per_fold):
        if not per_fold:
            return float("nan")
        y_true, y_probs, _, _, _, _ = _deduplicate_pooled_predictions(per_fold)
        try:
            yb = label_binarize(y_true, classes=list(range(len(class_names))))
            if len(class_names) == 2:
                yb = np.column_stack([1 - yb, yb])
            return float(roc_auc_score(yb, y_probs, average="macro", multi_class="ovr"))
        except Exception:
            return float("nan")

    epoch_label = {-2: "best", -1: "last"}.get(best_epoch, str(best_epoch))
    emb_dir = os.path.join(cv_dir, "embeddings", f"epoch_{epoch_label}")

    # --- Fast path: reuse cached per-image embeddings (no GPU re-encode) ---
    if os.path.isdir(emb_dir):
        print(f"  Bag-size sweep: reusing cached embeddings at {emb_dir} (no re-encode)")
        from remo_id.inference.mil import pick_global_indices_for_experiment
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # per_fold_lists[si][bi] -> list of per-fold result dicts
        per_fold_lists = [[[] for _ in bag_sizes] for _ in seeds]

        for fd in fold_dirs:
            fold_label = os.path.basename(fd)
            val_paths = fold_val_paths.get(fold_label, set())
            if not val_paths:
                continue
            ckpt_path = _find_epoch_checkpoint(fd, best_epoch)
            if ckpt_path is None:
                continue
            mn = extract_checkpoint_metadata(ckpt_path).get("model_name", "resnet18-mil")
            model, dev_used, _ = load_checkpoint(
                ckpt_path, mn, task_type=task_type, num_classes=num_classes,
                n_experiments=1, device=device,
            )
            model.eval()
            all_paths = set(data_module.data["path"].values)
            _, tm = data_module.split_from_paths(all_paths - val_paths, val_paths)
            df = tm.data.copy()
            df["experiment_id"] = df["experiment_id"].astype(str)
            parts = fold_label.replace("repeat", "").replace("fold", " ").split("_")
            try:
                repeat_idx = int(parts[0]); fold_idx = int(parts[1].strip())
            except (ValueError, IndexError):
                repeat_idx, fold_idx = 0, 0

            # preload this fold's cached embeddings once
            exps = []
            for i, row in df.iterrows():
                exp_id = str(row["experiment_id"]); exp_path = str(row["path"])
                if "class_label" not in row or bool(row.get("_has_null_label", False)):
                    continue
                if exp_path not in tm.experiment_ranges:
                    continue
                ef = os.path.join(emb_dir, f"{exp_id}.embeddings.npz")
                if not os.path.exists(ef):
                    continue
                feats = torch.from_numpy(
                    np.load(ef, allow_pickle=True)["embeddings"].astype(np.float32))
                start, _ = tm.experiment_ranges[exp_path]
                exps.append((i, exp_id, exp_path, int(row["class_label"]),
                             str(row.get("study_id", "")), start, feats))

            for si, s in enumerate(seeds):
                for bi, bs in enumerate(bag_sizes):
                    fy_t, fy_p, fy_pred, feid, fsid = [], [], [], [], []
                    for (i, exp_id, exp_path, y_true, study_id, start, feats) in exps:
                        bag_seed = int(s + 1009 * i)
                        set_inference_seeds(bag_seed)
                        idxs = pick_global_indices_for_experiment(
                            tm.experiment_ranges, exp_path, bs, bag_seed,
                            sample_mode="random", sample_percent=100.0)
                        if not idxs:
                            continue
                        local = [g - start for g in idxs]
                        with torch.no_grad():
                            logits, _ = model._aggregate(feats[local].unsqueeze(0).to(dev_used))
                        if task_type == "binary":
                            p = float(torch.sigmoid(logits[0]).item())
                            fy_p.append([1.0 - p, p]); fy_pred.append(int(p >= 0.5))
                        elif task_type == "multiclass":
                            l0 = logits[0] if logits.dim() > 1 else logits
                            pt = torch.softmax(l0, dim=-1)
                            fy_p.append(pt.cpu().tolist()); fy_pred.append(int(pt.argmax().item()))
                        else:
                            v = float(logits[0].item()); fy_p.append([v]); fy_pred.append(v)
                        fy_t.append(y_true); feid.append(exp_id); fsid.append(study_id)
                    if fy_t:
                        per_fold_lists[si][bi].append({
                            "repeat": repeat_idx, "fold": fold_idx,
                            "y_true": np.array(fy_t, dtype=int),
                            "y_probs": np.array(fy_p, dtype=float),
                            "y_pred": np.array(fy_pred, dtype=int),
                            "experiment_ids": np.array(feid),
                            "study_ids": np.array(fsid),
                        })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        aucs_by_seed = [[_pool_auc(per_fold_lists[si][bi]) for bi in range(len(bag_sizes))]
                        for si in range(len(seeds))]
        return bag_sizes, aucs_by_seed

    # --- Fallback: GPU re-encode per (bag_size, seed, fold) ---
    print(f"  Bag-size sweep: no cached embeddings at {emb_dir}; re-encoding on GPU")
    aucs_by_seed = []
    for s in seeds:
        seed_aucs = []
        for bs in bag_sizes:
            print(f"  Bag size sweep: bag_size={bs}, seed={s}")
            per_fold = []
            for fd in fold_dirs:
                fold_label = os.path.basename(fd)
                val_paths = fold_val_paths.get(fold_label, set())
                if not val_paths:
                    continue
                result = _predict_fold_at_epoch(
                    fold_dir=fd, epoch=best_epoch,
                    data_module=data_module, class_names=class_names,
                    bag_size=bs, n_bags=1, seed=s,
                    experiment_paths=val_paths,
                )
                if result is not None:
                    per_fold.append(result)
            seed_aucs.append(_pool_auc(per_fold))
        aucs_by_seed.append(seed_aucs)

    return bag_sizes, aucs_by_seed



# ── epoch performance plot ───────────────────────────────────────────────

def plot_epoch_performance(
    epoch_numbers: list,
    metrics_per_epoch: dict,
    best_epoch: int,
    class_names: list,
) -> plt.Figure:
    """Line plot: x=epoch, y=metric. Lines for macro F1, macro AUC, per-class F1."""
    style.apply()
    fig, ax = style.figure(width="double", aspect=0.6)
    colors = style.class_colors_muted(class_names)

    epochs = sorted(metrics_per_epoch.keys())
    macro_f1s = [metrics_per_epoch[e]["macro_f1"] for e in epochs]
    macro_aucs = [metrics_per_epoch[e]["macro_auc"] for e in epochs]

    ax.plot(epochs, macro_f1s, "o-", color=style.ACCENT, lw=1.8, ms=3, label="Macro F1 (Youden)", zorder=5)
    ax.plot(epochs, macro_aucs, "s--", color=style.FG, lw=1.0, ms=2.5, label="Macro AUC", zorder=4)

    # Argmax macro F1
    argmax_f1s = [metrics_per_epoch[e].get("argmax_macro_f1", float("nan")) for e in epochs]
    if not all(np.isnan(v) for v in argmax_f1s):
        ax.plot(epochs, argmax_f1s, "^-", color=style.ACCENT, lw=1.2, ms=3,
                alpha=0.6, label="Macro F1 (Argmax)", zorder=4)

    for c, name in enumerate(class_names):
        class_f1s = [metrics_per_epoch[e]["per_class_f1"].get(name, 0) for e in epochs]
        ax.plot(epochs, class_f1s, "-", color=colors[c], lw=0.7, alpha=0.6, label=f"F1 {name}")

    ax.axvline(best_epoch, color=style.MUTE_FAINT, lw=1.0, ls=(0, (3, 2)),
               label=f"Best epoch ({best_epoch})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric value")
    ax.set_title("Performance vs Epoch (LOO CV)")
    ax.legend(loc="lower right", fontsize=6)
    ax.set_ylim(0, 1.05)
    style.trim_spines(ax)
    fig.tight_layout()
    return fig


# ── methods text generation ──────────────────────────────────────────────

def _compute_pooled_metrics(y_true, y_probs, class_names) -> dict:
    """Pooled macro F1 (Youden-J), macro AUC and argmax metrics from pooled predictions.

    Populates per-epoch metrics for the forced-epoch / precomputed paths where no sweep ran
    (otherwise best_epoch.json + the report header would show macro_f1 = 0.0000)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    youden = find_youden_thresholds(y_true, y_probs, class_names)
    thresholds = {r["class_name"]: r["optimal_threshold"] for r in youden}
    y_pred_argmax = y_probs.argmax(axis=1)
    per_class_f1, argmax_per_class_f1 = {}, {}
    for c, name in enumerate(class_names):
        y_bin = (y_true == c).astype(int)
        pred = (y_probs[:, c] >= thresholds[name]).astype(int)
        tp = int(((pred == 1) & (y_bin == 1)).sum()); fp = int(((pred == 1) & (y_bin == 0)).sum())
        fn = int(((pred == 0) & (y_bin == 1)).sum())
        ppv = tp / max(1, tp + fp); sens = tp / max(1, tp + fn)
        per_class_f1[name] = 2 * ppv * sens / max(1e-12, ppv + sens)
        atp = int(((y_pred_argmax == c) & (y_true == c)).sum())
        afp = int(((y_pred_argmax == c) & (y_true != c)).sum())
        afn = int(((y_pred_argmax != c) & (y_true == c)).sum())
        appv = atp / max(1, atp + afp); asens = atp / max(1, atp + afn)
        argmax_per_class_f1[name] = 2 * appv * asens / max(1e-12, appv + asens)
    try:
        yb = label_binarize(y_true, classes=list(range(len(class_names))))
        if len(class_names) == 2:
            yb = np.column_stack([1 - yb, yb])
        macro_auc = float(roc_auc_score(yb, y_probs, average="macro", multi_class="ovr"))
    except Exception:
        macro_auc = float("nan")
    return {
        "macro_f1": float(np.mean(list(per_class_f1.values()))),
        "macro_auc": macro_auc,
        "per_class_f1": per_class_f1,
        "argmax_macro_f1": float(np.mean(list(argmax_per_class_f1.values()))),
        "argmax_accuracy": float((y_pred_argmax == y_true).mean()),
        "argmax_per_class_f1": argmax_per_class_f1,
    }


def generate_report_text(
    cv_dir: str,
    class_config: str,
    class_names: list,
    num_classes: int,
    best_epoch: int,
    training_bag_size: int,
    n_bags: int = 5,
) -> dict:
    """Read checkpoint hparams and generate methods text sections."""
    # Find any checkpoint to extract hparams
    fold_dirs = sorted(glob_module.glob(os.path.join(cv_dir, "repeat*_fold*")))
    hp = {}
    for fd in fold_dirs:
        ckpts = glob_module.glob(os.path.join(fd, "*.ckpt"))
        for c in ckpts:
            try:
                ckpt = torch.load(c, map_location="cpu")
                hp = ckpt.get("hyper_parameters", {})
                if hp:
                    break
            except Exception:
                continue
        if hp:
            break

    model_name = hp.get("model_name", "unknown")
    is_embedding = model_name == "embedding-mil" or "embedding_dim" in hp
    n_experiments = hp.get("n_experiments", 8)
    max_epochs = hp.get("max_epochs", 100)
    lr = hp.get("lr", 1e-4)
    lr_schedule = hp.get("lr_schedule", "cosine")
    warmup_epochs = hp.get("warmup_epochs", 5)
    weight_decay = hp.get("weight_decay", 0.01)
    dropout = hp.get("dropout", 0.3)
    attn_dim = hp.get("attn_dim", 128)
    num_heads = hp.get("num_heads", 1)
    pooling = hp.get("pooling", "attention")
    num_batches = hp.get("num_batches", 100)
    n_folds = len(fold_dirs)

    # Per-bag unit and effective batch differ: embedding-mil bags are CELLS (cells_per_bag is
    # not stored in hparams — it is the resolved training bag size), image models bag IMAGES.
    if is_embedding:
        unit = "cells"
        per_bag = training_bag_size
        batch_size = per_bag * n_experiments
    else:
        unit = "images"
        batch_size = hp.get("batch_size", 200)
        per_bag = batch_size // n_experiments if n_experiments else batch_size

    sections = {}

    arch = [
        f"**Model**: {model_name}",
        f"- Pooling: {pooling} (attn_dim={attn_dim}, num_heads={num_heads})",
        f"- Dropout: {dropout}",
    ]
    if is_embedding:
        arch.append(f"- Input: precomputed {hp.get('embedding_dim', '?')}-d SSL cell embeddings "
                    f"(frozen embedder — no backbone fine-tuning)")
        if hp.get("projection_dim"):
            arch.append(f"- Projection dim: {hp['projection_dim']}")
    else:
        arch.append(f"- Fine-tuned layers: {hp.get('finetune_layers', 'layer3,layer4')}")
    arch.append(f"- Classes: {num_classes} ({', '.join(class_names)})")
    sections["model_architecture"] = "\n".join(arch) + "\n"

    sections["training_configuration"] = (
        f"- LOO cross-validation: {n_folds} folds (1 per sample_id)\n"
        f"- max_epochs={max_epochs}, {unit}_per_bag={per_bag}, n_experiments={n_experiments} "
        f"({batch_size} {unit}/batch)\n"
        f"- {unit.capitalize()} per epoch: {batch_size * num_batches} ({num_batches} batches x {batch_size})\n"
        f"- Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})\n"
        f"- LR schedule: {lr_schedule} (warmup={warmup_epochs} epochs)\n"
        f"- Best epoch selected as **epoch {best_epoch}** by pooled macro F1 across all folds\n"
    )

    sections["stats_methodology"] = (
        f"- Fixed-epoch selection: pooled macro F1 across all {n_folds} held-out folds\n"
        f"- Youden-J thresholds optimized on pooled held-out predictions\n"
        f"- Bootstrap 95% CIs (1000 iterations)\n"
        f"- Per-fold F1 std reported for fold variability\n"
    )

    bag_line = (f"- cells_per_bag = {training_bag_size} (training-matched weighted cell-class composition)"
                if is_embedding else
                f"- bag_size = batch_size / n_experiments = {training_bag_size}")
    sections["inference_methodology"] = (
        f"{bag_line}\n"
        f"- n_bags = {n_bags} (averaged per experiment)\n"
        f"- Each experiment predicted only when held out (LOO)\n"
    )

    return sections


# ── report assembly ──────────────────────────────────────────────────────

def _assemble_report_markdown(
    output_dir: str,
    text_sections: dict,
    best_epoch: int,
    best_f1: float,
    class_names: list,
    pairwise_pairs: list = None,
) -> str:
    """Assemble report.md with all sections."""
    lines = ["# LOO CV Comprehensive Report\n"]

    lines.append(f"**Best epoch**: {best_epoch} (pooled macro F1 = {best_f1:.4f})\n")
    lines.append(f"**Classes**: {', '.join(class_names)}\n")

    lines.append("\n## Methods\n")
    for key in ["model_architecture", "training_configuration", "inference_methodology", "stats_methodology"]:
        if key in text_sections:
            lines.append(f"### {key.replace('_', ' ').title()}\n")
            lines.append(text_sections[key])
            lines.append("")

    lines.append("\n## Performance vs Epoch\n")
    lines.append("![Epoch Performance](figures/epoch_performance.png)\n")

    lines.append("\n## Performance at Youden-J Thresholds\n")
    lines.append("See `table_youden.csv`\n")

    lines.append("\n## Performance at Sensitivity-Optimized Thresholds\n")
    lines.append("See `table_sensitivity.csv`\n")

    lines.append("\n## Performance at Argmax\n")
    lines.append("See `table_argmax.csv`\n")

    lines.append("\n## Figures\n")
    figure_names = [
        ("roc_curves", "ROC Curves"),
        ("pr_curves", "PR Curves"),
        ("confusion_youden", "Confusion Matrix (Youden-J)"),
        ("confusion_sensitivity", "Confusion Matrix (Sensitivity-Optimized)"),
        ("confusion_argmax", "Confusion Matrix (Argmax)"),
        ("threshold_sensitivity", "Threshold Sensitivity"),
        ("prob_distributions", "Probability Distributions"),
        ("calibration", "Calibration"),
        ("kfold_variability", "K-Fold Variability"),
        ("dual_operating_points", "Dual Operating Points"),
        ("error_confidence", "Error Confidence"),
        ("error_by_study", "Error by Study"),
        ("error_transitions", "Error Transitions"),
        ("bag_variance", "Prediction Stability vs Bag Size"),
        ("per_image_probabilities", "Per-Image Class Predictions"),
    ]
    for fname, title in figure_names:
        path = os.path.join("figures", f"{fname}.png")
        lines.append(f"### {title}\n")
        lines.append(f"![{title}]({path})\n")

    if pairwise_pairs:
        lines.append("\n## Pairwise Analysis\n")
        for pair in pairwise_pairs:
            pos, neg = pair.split(":")
            subdir = f"pairwise_{pos}_vs_{neg}"
            lines.append(f"### {pos} vs {neg}\n")
            lines.append(f"See `{subdir}/`\n")

    lines.append("\n## Error Consistency\n")
    lines.append("See `error_analysis/error_consistency.csv`\n")

    return "\n".join(lines)


# ── main orchestrator ────────────────────────────────────────────────────

def run_loo_report(
    cv_dir: str,
    class_config: str,
    output: str = None,
    seed: int = 403,
    bootstrap_n: int = 1000,
    n_bags: int = 5,
    pairwise_pairs: list = None,
    min_sensitivity: float = 0.95,
    sensitivity_targets_str: str = None,
    bag_sizes_str: str = None,
    forced_epoch: int = None,
    bag_size_override: int = None,
    error_consistency: bool = False,
    no_regenerate: bool = False,
    target_class: str = None,
) -> str:
    """Top-level LOO report orchestrator.

    ``target_class`` (a class name) is highlighted in the dual-operating-point
    ROC figure; the figure is skipped when it is not given.

    When ``no_regenerate`` is set, the report will REFUSE to re-encode/predict on
    GPU (the forced-epoch-missing and no-precomputed branches) and raise instead.
    This guarantees it only ever reads existing ``predictions/epoch_*.npz`` and can
    never silently regenerate them at the wrong bag size.

    Returns path to the output directory.
    """
    from remo_id.inference.context import load_class_config
    from remo_id.data import RemoIDDataModule
    from remo_id.models import get_augmentation_backbone

    # 1. Load class config
    class_definitions, class_names, num_classes, task_type, exclude_filters, include_filters, default_class_idx = (
        load_class_config(class_config)
    )

    # 2. Scan fold dirs, find common epochs, extract training_bag_size
    training_bag_size = _extract_training_bag_size(cv_dir)
    if bag_size_override is not None:
        print(f"Training bag size: {training_bag_size}, overridden to: {bag_size_override}")
        training_bag_size = bag_size_override
    else:
        print(f"Training bag size: {training_bag_size}")

    # Set up output directory
    if output is None:
        output = os.path.join(cv_dir, "analysis", "loo_report")
    os.makedirs(output, exist_ok=True)
    fig_dir = os.path.join(output, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    error_dir = os.path.join(output, "error_analysis")
    os.makedirs(error_dir, exist_ok=True)

    # Extract model name from a checkpoint for data module setup
    fold_dirs = sorted(glob_module.glob(os.path.join(cv_dir, "repeat*_fold*")))
    model_name = "resnet18-mil"
    for fd in fold_dirs:
        ckpts = glob_module.glob(os.path.join(fd, "*.ckpt"))
        for c in ckpts:
            try:
                meta = extract_checkpoint_metadata(c)
                model_name = meta.get("model_name", "resnet18-mil")
                break
            except Exception:
                continue
        break

    backbone_name = get_augmentation_backbone(model_name)

    # Build data module (full dataset, unsplit)
    data_module = RemoIDDataModule(
        database_column="class_label",
        database_label=None,
        study_id=None,
        threshold=None,
        threshold_mode="==",
        negative_study_ids=[],
        negative_experiment_ids=[],
        batch_size=1,
        backbone_name=backbone_name,
        train=False,
        task_type=task_type,
        num_classes=num_classes,
        class_definitions=class_definitions,
        exclude_filters=exclude_filters,
        include_filters=include_filters,
        default_class_idx=default_class_idx,
        skip_null_labels=True,
        background_correction='none',
    )

    # 3. EPOCH SWEEP — try pre-computed predictions first
    from remo_id.workflows.loo_predict import load_epoch_predictions
    precomputed = load_epoch_predictions(cv_dir)
    if precomputed:
        print(f"Found pre-computed predictions for {len(precomputed)} epochs: {sorted(precomputed.keys())}")

    if precomputed and forced_epoch is None:
        # Use pre-computed predictions for epoch sweep
        # Exclude best/last sentinels (-2, -1) — LOO requires same epoch across folds
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize

        epoch_numbers = sorted(k for k in precomputed.keys() if k >= 0)
        metrics_per_epoch = {}
        best_f1 = -1.0
        best_epoch = epoch_numbers[0]
        best_predictions = None

        for epoch in epoch_numbers:
            pdata = precomputed[epoch]
            y_true = pdata["y_true"]
            y_probs = pdata["y_probs"]
            exp_ids = pdata["experiment_ids"]
            study_ids = pdata["study_ids"]
            folds = pdata["folds"]
            repeats = pdata["repeats"]

            # Compute Youden thresholds and F1
            youden = find_youden_thresholds(y_true, y_probs, class_names)
            thresholds = {r["class_name"]: r["optimal_threshold"] for r in youden}
            y_pred_thr = apply_thresholds_to_predictions(y_probs, class_names, thresholds)

            per_class_f1 = {}
            for c, name in enumerate(class_names):
                y_bin = (y_true == c).astype(int)
                y_sc = y_probs[:, c]
                thr = thresholds[name]
                pred = (y_sc >= thr).astype(int)
                tp = int(((pred == 1) & (y_bin == 1)).sum())
                fp = int(((pred == 1) & (y_bin == 0)).sum())
                fn = int(((pred == 0) & (y_bin == 1)).sum())
                ppv = tp / max(1, tp + fp)
                sens = tp / max(1, tp + fn)
                f1 = 2 * ppv * sens / max(1e-12, ppv + sens)
                per_class_f1[name] = f1

            macro_f1 = np.mean(list(per_class_f1.values()))

            # Argmax metrics
            y_pred_argmax = y_probs.argmax(axis=1)
            argmax_per_class_f1 = {}
            for c, name in enumerate(class_names):
                tp = int(((y_pred_argmax == c) & (y_true == c)).sum())
                fp = int(((y_pred_argmax == c) & (y_true != c)).sum())
                fn = int(((y_pred_argmax != c) & (y_true == c)).sum())
                ppv = tp / max(1, tp + fp)
                sens = tp / max(1, tp + fn)
                f1 = 2 * ppv * sens / max(1e-12, ppv + sens)
                argmax_per_class_f1[name] = f1
            argmax_macro_f1 = np.mean(list(argmax_per_class_f1.values()))
            argmax_accuracy = float((y_pred_argmax == y_true).mean())

            try:
                yb = label_binarize(y_true, classes=list(range(len(class_names))))
                if len(class_names) == 2:
                    yb = np.column_stack([1 - yb, yb])
                macro_auc = float(roc_auc_score(yb, y_probs, average="macro", multi_class="ovr"))
            except Exception:
                macro_auc = float("nan")

            metrics_per_epoch[epoch] = {
                "macro_f1": float(macro_f1),
                "macro_auc": float(macro_auc),
                "per_class_f1": per_class_f1,
                "argmax_macro_f1": float(argmax_macro_f1),
                "argmax_accuracy": float(argmax_accuracy),
                "argmax_per_class_f1": argmax_per_class_f1,
            }
            print(f"  Epoch {epoch}: macro F1={macro_f1:.4f}, argmax F1={argmax_macro_f1:.4f}, macro AUC={macro_auc:.4f}")

            # Reconstruct per_fold structure
            per_fold = []
            for fold_idx in sorted(set(folds)):
                mask = folds == fold_idx
                repeat_idx = repeats[mask][0] if mask.any() else 0
                per_fold.append({
                    "repeat": int(repeat_idx),
                    "fold": int(fold_idx),
                    "y_true": y_true[mask],
                    "y_probs": y_probs[mask],
                    "y_pred": y_pred_thr[mask],
                    "experiment_ids": exp_ids[mask],
                    "study_ids": study_ids[mask],
                })

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_epoch = epoch
                best_predictions = {
                    "pooled": (y_true, y_probs, y_pred_thr, exp_ids, study_ids),
                    "per_fold": per_fold,
                }

        print(f"\nBest epoch: {best_epoch} (macro F1={best_f1:.4f})")

    elif precomputed and forced_epoch is not None and forced_epoch in precomputed:
        # Use pre-computed predictions for forced epoch
        pdata = precomputed[forced_epoch]
        y_true = pdata["y_true"]
        y_probs = pdata["y_probs"]
        exp_ids = pdata["experiment_ids"]
        study_ids = pdata["study_ids"]
        folds = pdata["folds"]
        repeats = pdata["repeats"]

        youden = find_youden_thresholds(y_true, y_probs, class_names)
        thresholds = {r["class_name"]: r["optimal_threshold"] for r in youden}
        y_pred = apply_thresholds_to_predictions(y_probs, class_names, thresholds)

        per_fold = []
        for fold_idx in sorted(set(folds)):
            mask = folds == fold_idx
            repeat_idx = repeats[mask][0] if mask.any() else 0
            per_fold.append({
                "repeat": int(repeat_idx),
                "fold": int(fold_idx),
                "y_true": y_true[mask],
                "y_probs": y_probs[mask],
                "y_pred": y_pred[mask],
                "experiment_ids": exp_ids[mask],
                "study_ids": study_ids[mask],
            })

        best_epoch = forced_epoch
        epoch_numbers = [forced_epoch]
        metrics_per_epoch = {}
        best_predictions = {
            "pooled": (y_true, y_probs, y_pred, exp_ids, study_ids),
            "per_fold": per_fold,
        }

    elif forced_epoch is not None:
        if no_regenerate:
            raise RuntimeError(
                f"--no_regenerate set but epoch {forced_epoch} has no precomputed "
                f"predictions in {cv_dir}/predictions/. Refusing to re-encode on GPU "
                f"(would regenerate at bag size {training_bag_size}). Run "
                f"`workflow loo-aggregate --epoch {forced_epoch}` (or loo-predict) first.")
        print(f"Using forced epoch {forced_epoch} (skipping sweep)")
        # Predict all folds at the forced epoch
        fold_val_paths = _get_fold_val_paths(cv_dir, data_module)
        per_fold = []
        for fd in fold_dirs:
            fold_label = os.path.basename(fd)
            val_paths = fold_val_paths.get(fold_label, set())
            if not val_paths:
                continue
            result = _predict_fold_at_epoch(
                fold_dir=fd, epoch=forced_epoch,
                data_module=data_module, class_names=class_names,
                bag_size=training_bag_size, n_bags=n_bags, seed=seed,
                experiment_paths=val_paths,
            )
            if result is not None:
                per_fold.append(result)

        if not per_fold:
            raise ValueError(f"No valid predictions at epoch {forced_epoch}")

        y_true, y_probs, y_pred, exp_ids, study_ids, _sample_ids = _deduplicate_pooled_predictions(per_fold)

        # Compute Youden thresholds to get y_pred
        youden = find_youden_thresholds(y_true, y_probs, class_names)
        thresholds = {r["class_name"]: r["optimal_threshold"] for r in youden}
        y_pred = apply_thresholds_to_predictions(y_probs, class_names, thresholds)

        best_epoch = forced_epoch
        epoch_numbers = [forced_epoch]
        metrics_per_epoch = {}
        best_predictions = {
            "pooled": (y_true, y_probs, y_pred, exp_ids, study_ids),
            "per_fold": per_fold,
        }
    else:
        if no_regenerate:
            raise RuntimeError(
                f"--no_regenerate set but no precomputed predictions found in "
                f"{cv_dir}/predictions/. Refusing to submit a GPU prediction job. "
                f"Run the embed+aggregate pipeline (workflow loo-aggregate per epoch) first.")
        # No pre-computed predictions — submit parallel prediction job instead of slow sequential sweep
        from remo_id.workflows.loo_predict import submit_predict_array
        print("\nNo pre-computed predictions found. Submitting parallel prediction job...")
        print("(Each epoch gets its own GPU — much faster than sequential sweep)\n")
        job_id = submit_predict_array(
            cv_dir=cv_dir,
            class_config=class_config,
            seed=seed,
            n_bags=n_bags,
            bag_size=bag_size_override,
        )
        if job_id:
            print(f"\nPrediction job submitted: {job_id}")
            print(f"Re-run this report after predictions complete:")
            print(f"  remo-id workflow loo-report --cv_dir {cv_dir} --class_config {class_config}")
        else:
            print("Failed to submit prediction job. Falling back to sequential sweep...")
            epoch_numbers, metrics_per_epoch, best_predictions, best_epoch = sweep_epochs(
                cv_dir, data_module, class_names, training_bag_size, n_bags, seed,
            )
        return output

    y_true, y_probs, y_pred, exp_ids, study_ids = best_predictions["pooled"]
    per_fold = best_predictions["per_fold"]

    # Populate pooled metrics for the chosen epoch when no sweep ran (forced/precomputed paths),
    # so best_epoch.json + the report header report the real macro F1/AUC instead of 0.0.
    if best_epoch not in metrics_per_epoch:
        metrics_per_epoch[best_epoch] = _compute_pooled_metrics(y_true, y_probs, class_names)

    # 4. Save best epoch info
    best_f1 = metrics_per_epoch.get(best_epoch, {}).get("macro_f1", 0.0) if metrics_per_epoch else 0.0
    # Find best argmax epoch
    best_argmax_epoch = best_epoch
    best_argmax_f1 = 0.0
    if metrics_per_epoch:
        for e, m in metrics_per_epoch.items():
            af1 = m.get("argmax_macro_f1", 0.0)
            if af1 > best_argmax_f1:
                best_argmax_f1 = af1
                best_argmax_epoch = e
    best_epoch_info = {
        "epoch": best_epoch,
        "macro_f1": best_f1,
        "macro_auc": metrics_per_epoch.get(best_epoch, {}).get("macro_auc", 0.0) if metrics_per_epoch else 0.0,
        "best_argmax_epoch": best_argmax_epoch,
        "argmax_macro_f1": best_argmax_f1,
        "training_bag_size": training_bag_size,
        "n_folds": len(fold_dirs),
        "n_bags": n_bags,
    }
    with open(os.path.join(output, "best_epoch.json"), "w") as f:
        json.dump(best_epoch_info, f, indent=2)
    if best_argmax_epoch != best_epoch:
        print(f"Note: best argmax epoch ({best_argmax_epoch}, F1={best_argmax_f1:.4f}) differs from best Youden epoch ({best_epoch})")

    # 5. Plot performance vs epoch (if we did a sweep)
    if len(metrics_per_epoch) > 1:
        fig_epoch = plot_epoch_performance(epoch_numbers, metrics_per_epoch, best_epoch, class_names)
        fig_epoch.savefig(os.path.join(fig_dir, "epoch_performance.png"), dpi=200, bbox_inches="tight")
        plt.close(fig_epoch)

    # Save epoch sweep CSV
    if metrics_per_epoch:
        epoch_rows = []
        for e in sorted(metrics_per_epoch.keys()):
            m = metrics_per_epoch[e]
            row = {"epoch": e, "macro_f1": m["macro_f1"], "macro_auc": m["macro_auc"],
                   "argmax_macro_f1": m.get("argmax_macro_f1", float("nan")),
                   "argmax_accuracy": m.get("argmax_accuracy", float("nan"))}
            for name, f1 in m["per_class_f1"].items():
                row[f"f1_{name}"] = f1
            for name, f1 in m.get("argmax_per_class_f1", {}).items():
                row[f"argmax_f1_{name}"] = f1
            epoch_rows.append(row)
        pd.DataFrame(epoch_rows).to_csv(os.path.join(output, "epoch_sweep.csv"), index=False)

    # ── Phase 3: Analysis at best epoch ──────────────────────────────────

    # 6. Compute thresholds
    youden_results = find_youden_thresholds(y_true, y_probs, class_names)
    youden_thresholds = {r["class_name"]: r["optimal_threshold"] for r in youden_results}

    sensitivity_targets = build_sensitivity_targets(class_names, min_sensitivity, sensitivity_targets_str)
    sens_results = find_optimal_thresholds(y_true, y_probs, class_names, sensitivity_targets)
    sens_thresholds = {r["class_name"]: r["optimal_threshold"] for r in sens_results}

    # Apply thresholds for predictions
    y_pred_youden = apply_thresholds_to_predictions(y_probs, class_names, youden_thresholds)
    y_pred_sens = apply_thresholds_to_predictions(y_probs, class_names, sens_thresholds)

    # 7. Performance tables
    print("\nComputing performance tables...")
    table_youden = compute_full_metrics_at_thresholds(
        y_true, y_probs, class_names, youden_thresholds,
        bootstrap_n=bootstrap_n, seed=seed,
    )
    table_youden.to_csv(os.path.join(output, "table_youden.csv"), index=False)
    print("Youden-J table:")
    print(table_youden.to_string(index=False))

    table_sens = compute_full_metrics_at_thresholds(
        y_true, y_probs, class_names, sens_thresholds,
        bootstrap_n=bootstrap_n, seed=seed,
    )
    table_sens.to_csv(os.path.join(output, "table_sensitivity.csv"), index=False)
    print("\nSensitivity-optimized table:")
    print(table_sens.to_string(index=False))

    # Argmax table
    table_argmax = compute_argmax_metrics_table(
        y_true, y_probs, class_names,
        bootstrap_n=bootstrap_n, seed=seed,
    )
    table_argmax.to_csv(os.path.join(output, "table_argmax.csv"), index=False)
    print("\nArgmax table:")
    print(table_argmax.to_string(index=False))

    # Dual operating points table
    dual_df = dual_operating_points_table(
        y_true, y_probs, class_names, sensitivity_targets,
        output_csv=os.path.join(output, "dual_operating_points.csv"),
    )

    # Per-fold metrics
    fold_f1_std = compute_fold_f1_std(per_fold, class_names, youden_thresholds)
    fold_metrics_rows = []
    for fold_data in per_fold:
        yt = np.asarray(fold_data["y_true"])
        yp = np.asarray(fold_data["y_probs"])
        row = {"fold": f"repeat{fold_data['repeat']}_fold{fold_data['fold']}"}
        for c, name in enumerate(class_names):
            y_bin = (yt == c).astype(int)
            y_sc = yp[:, c]
            thr = youden_thresholds[name]
            pred = (y_sc >= thr).astype(int)
            tp = int(((pred == 1) & (y_bin == 1)).sum())
            fp = int(((pred == 1) & (y_bin == 0)).sum())
            fn = int(((pred == 0) & (y_bin == 1)).sum())
            ppv = tp / max(1, tp + fp)
            sens = tp / max(1, tp + fn)
            f1 = 2 * ppv * sens / max(1e-12, ppv + sens)
            row[f"f1_{name}"] = f1
        fold_f1s = [row[f"f1_{name}"] for name in class_names]
        row["macro_f1"] = np.mean(fold_f1s)
        fold_metrics_rows.append(row)
    pd.DataFrame(fold_metrics_rows).to_csv(os.path.join(output, "per_fold_metrics.csv"), index=False)

    # 8. Figures
    print("\nGenerating figures...")

    def _save_fig(fig, name):
        fig.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    _save_fig(plot_roc_curves(y_true, y_probs, class_names, bootstrap_n=bootstrap_n), "roc_curves")
    _save_fig(plot_pr_curves(y_true, y_probs, class_names, bootstrap_n=bootstrap_n), "pr_curves")
    _save_fig(plot_confusion_matrix(y_true, y_pred_youden, class_names, title="Confusion Matrix (Youden-J)"), "confusion_youden")
    _save_fig(plot_confusion_matrix(y_true, y_pred_sens, class_names, title="Confusion Matrix (Sensitivity-Optimized)"), "confusion_sensitivity")
    y_pred_argmax = y_probs.argmax(axis=1)
    _save_fig(plot_confusion_matrix(y_true, y_pred_argmax, class_names, title="Confusion Matrix (Argmax)"), "confusion_argmax")
    _save_fig(plot_threshold_sensitivity(y_true, y_probs, class_names), "threshold_sensitivity")
    _save_fig(plot_prob_distributions(y_true, y_probs, class_names), "prob_distributions")
    _save_fig(plot_calibration(y_true, y_probs, class_names), "calibration")

    # K-fold variability from per_fold data
    from remo_id.workflows.cv import aggregate_cv_results
    cv_agg = aggregate_cv_results(per_fold, (y_true, y_probs, y_pred_youden), class_names, bootstrap_n=bootstrap_n, task_type=task_type)
    _save_fig(plot_kfold_variability(cv_agg["per_fold_metrics"], class_names), "kfold_variability")

    # Dual operating points
    if target_class is not None:
        _save_fig(plot_dual_operating_points(y_true, y_probs, class_names, youden_results,
                                             sens_results, target_class=target_class),
                  "dual_operating_points")
    else:
        print("  Skipping dual_operating_points figure (pass --target_class to draw it)")

    # Error analysis figures from held-out predictions
    _save_fig(plot_error_confidence(y_true, y_pred_youden, y_probs, class_names), "error_confidence")

    error_df = identify_misclassifications_from_arrays(
        y_true, y_pred_youden, y_probs, class_names,
        experiment_ids=exp_ids, study_ids=study_ids,
    )
    error_df.to_csv(os.path.join(error_dir, "misclassifications.csv"), index=False)

    _save_fig(plot_error_by_study(error_df, class_names), "error_by_study")
    _save_fig(plot_confusion_errors(error_df, class_names), "error_transitions")

    # 9. Pairwise analysis
    if pairwise_pairs:
        for pair in pairwise_pairs:
            pos, neg = pair.split(":")
            pair_dir = os.path.join(output, f"pairwise_{pos}_vs_{neg}")
            try:
                run_pairwise(
                    y_true, y_probs, class_names,
                    class_positive=pos, class_negative=neg,
                    output_dir=pair_dir, bootstrap_n=bootstrap_n,
                )
            except Exception as e:
                print(f"Warning: pairwise {pos} vs {neg} failed: {e}")

    # 10. Error consistency (Phase 2) — GPU-intensive, off by default
    if error_consistency:
        print("\nComputing error consistency...")
        try:
            consistency_df, detail_df = compute_error_consistency(
                cv_dir, data_module, class_names, best_epoch, training_bag_size,
            )
            consistency_df.to_csv(os.path.join(error_dir, "error_consistency.csv"), index=False)
            detail_df.to_csv(os.path.join(error_dir, "error_consistency_detail.csv"), index=False)
            print(f"Error consistency: {len(consistency_df)} experiments, "
                  f"{(consistency_df['error_rate'] > 0).sum()} with errors")
        except Exception as e:
            print(f"Warning: error consistency failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nSkipping error consistency (use --error_consistency to enable)")

    # 11. Bag size sweep (optional)
    if bag_sizes_str:
        bag_sizes = [int(x.strip()) for x in bag_sizes_str.split(",")]
        print(f"\nBag size sweep: {bag_sizes}")
        bs_list, aucs = run_cv_bag_size_sweep(
            cv_dir, data_module, class_names, best_epoch, bag_sizes,
        )
        _save_fig(plot_bag_size_sweep(bs_list, aucs), "bag_size_sweep")

    # 12. Bag variance analysis (prediction stability vs bag size)
    from remo_id.workflows.loo_predict import load_bag_variance_results, submit_bag_variance_array
    variance_probs = load_bag_variance_results(cv_dir)
    if variance_probs is not None:
        print("\nBag variance: loading pre-computed results...")
        bag_sizes = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
        _save_fig(plot_bag_variance(bag_sizes, variance_probs, class_names), "bag_variance")
        print("  Saved bag_variance.png")
    else:
        print("\nNo pre-computed bag variance results found. Submitting parallel job...")
        print("(Each fold gets its own GPU — re-run this report after it completes)")
        submit_bag_variance_array(
            cv_dir, class_config, epoch=best_epoch, seed=seed,
        )

    # 13. Per-image probability heatmap
    per_img_dir = os.path.join(cv_dir, "per_image_predictions")
    per_img_csvs = glob_module.glob(os.path.join(per_img_dir, "*.csv")) if os.path.exists(per_img_dir) else []
    if per_img_csvs:
        print(f"\nPer-image predictions: loading {len(per_img_csvs)} experiments...")
        experiment_probs = []
        for csv_path in per_img_csvs:
            df_img = pd.read_csv(csv_path)
            prob_cols = [c for c in df_img.columns if c.startswith("prob_")]
            if not prob_cols:
                continue
            experiment_probs.append({
                "experiment_id": str(df_img["experiment_id"].iloc[0]),
                "y_true": int(df_img["y_true"].iloc[0]),
                "probs": df_img[prob_cols].values,
            })
        if experiment_probs:
            from remo_id.reports.figures import plot_per_image_probabilities
            _save_fig(plot_per_image_probabilities(experiment_probs, class_names),
                      "per_image_probabilities")
            print(f"  Saved per_image_probabilities.png ({len(experiment_probs)} experiments)")
    else:
        print("\nNo per-image predictions found. Submit with:")
        print(f"  remo-id workflow loo-per-image (or use submit_per_image_array)")
        from remo_id.workflows.loo_predict import submit_per_image_array
        submit_per_image_array(cv_dir, class_config, epoch=best_epoch, seed=seed)

    # 14. Generate methods text + assemble the single report.md (Methods is a section within it,
    # so we no longer emit a redundant standalone methods.md).
    text_sections = generate_report_text(
        cv_dir, class_config, class_names, num_classes, best_epoch, training_bag_size,
        n_bags=n_bags,
    )
    report_md = _assemble_report_markdown(
        output, text_sections, best_epoch, best_f1, class_names,
        pairwise_pairs=pairwise_pairs,
    )
    with open(os.path.join(output, "report.md"), "w") as f:
        f.write(report_md)

    print(f"\nLOO report complete: {output}")
    return output
