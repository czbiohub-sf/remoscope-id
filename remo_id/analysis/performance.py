"""File-based performance analysis.

Reads saved predictions from disk (no GPU needed) and computes metrics,
ROC/PR curves, confusion matrix, calibration data.

Usage:
    remo-id analysis performance --checkpoint best.ckpt
    # or programmatically:
    from remo_id.analysis.performance import run_performance
    run_performance(checkpoint_path, output_dir=...)
"""

import os
from typing import Optional

import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from remo_id.inference.results import load_predictions
from remo_id.analysis.results import save_metrics, save_curves, analysis_output_dir
from remo_id.analysis.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
)


def run_performance(
    checkpoint_path: str,
    output: Optional[str] = None,
    split: str = "val",
    bootstrap_n: int = 1000,
    pred_threshold: float = 0.5,
):
    """Compute performance metrics from saved predictions.

    Reads ``{checkpoint}/predictions.npz`` and produces:
        - metrics.json (accuracy, F1, AUC, etc. with bootstrap CIs)
        - roc_curves.npz (fpr/tpr per class)
        - pr_curves.npz (precision/recall per class)
        - confusion_matrix.npz

    Args:
        checkpoint_path: Path to model checkpoint.
        output: Output directory (default: ``{ckpt_dir}/analysis/performance/``).
        split: Which split to evaluate ("train", "val", or "all").
        bootstrap_n: Number of bootstrap iterations for CIs.
        pred_threshold: Decision threshold for binary classification.
    """
    df, meta = load_predictions(checkpoint_path)
    task_type = meta.get("task_type", "binary") if meta else "binary"
    num_classes = meta.get("num_classes", 2) if meta else 2
    class_names = meta.get("class_names") if meta else None

    out = output or analysis_output_dir(checkpoint_path, "performance")
    os.makedirs(out, exist_ok=True)

    # Filter by split
    if split != "all" and "split" in df.columns:
        mask = df["split"] == split
        if mask.sum() == 0:
            print(f"Warning: no experiments found for split='{split}', using all")
        else:
            df = df[mask].copy()

    # Exclude null labels
    if "has_null_label" in df.columns:
        n_null = df["has_null_label"].sum()
        if n_null > 0:
            print(f"Excluding {n_null} experiments with null labels")
            df = df[~df["has_null_label"]].copy()

    # Remove NaN predictions
    df = df.dropna(subset=["y_prob"])

    if len(df) == 0:
        raise RuntimeError("No valid predictions to analyze")

    print(f"Analyzing {len(df)} experiments (split={split}, task={task_type})")

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    y_prob = df["y_prob"].to_numpy()

    # Detect per-class probability columns
    prob_cols = sorted([c for c in df.columns if c.startswith("y_probs_class_")])
    y_probs = df[prob_cols].to_numpy() if prob_cols else None

    if task_type == "binary":
        _run_binary(y_true, y_pred, y_prob, y_probs, out, bootstrap_n, pred_threshold, class_names)
    elif task_type == "multiclass":
        _run_multiclass(y_true, y_pred, y_probs, out, bootstrap_n, num_classes, class_names)

    print(f"Saved performance analysis to {out}")


def _run_binary(y_true, y_pred, y_prob, y_probs, out, bootstrap_n, threshold, class_names):
    y_true_int = y_true.astype(int)
    y_pred_int = y_pred.astype(int)

    metrics = compute_binary_metrics(y_true_int, y_pred_int, y_prob)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_int, y_prob)
    metrics["auc"] = float(auc(fpr, tpr))

    # PR curve
    prec_arr, rec_arr, _ = precision_recall_curve(y_true_int, y_prob)
    metrics["pr_auc"] = float(auc(rec_arr, prec_arr))

    # Bootstrap CIs
    if bootstrap_n > 0 and len(y_true) >= 10:
        ci = _bootstrap_binary_ci(y_true_int, y_prob, bootstrap_n, threshold)
        metrics["bootstrap_ci"] = ci

    save_metrics(os.path.join(out, "metrics.json"), metrics)
    save_curves(os.path.join(out, "roc_curves.npz"), fpr=fpr, tpr=tpr, thresholds=thresholds)
    save_curves(os.path.join(out, "pr_curves.npz"), precision=prec_arr, recall=rec_arr)
    save_curves(os.path.join(out, "confusion_matrix.npz"),
                cm=np.array([[metrics["tn"], metrics["fp"]],
                             [metrics["fn"], metrics["tp"]]]))

    print(f"  Accuracy={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}")


def _run_multiclass(y_true, y_pred, y_probs, out, bootstrap_n, num_classes, class_names):
    y_true_int = y_true.astype(int)
    y_pred_int = y_pred.astype(int)

    metrics = compute_multiclass_metrics(y_true_int, y_pred_int, num_classes, y_probs=y_probs)

    # Per-class ROC curves
    if y_probs is not None and y_probs.shape[1] == num_classes:
        roc_data = {}
        for i in range(num_classes):
            binary_true = (y_true_int == i).astype(int)
            if binary_true.sum() > 0 and binary_true.sum() < len(binary_true):
                fpr_i, tpr_i, _ = roc_curve(binary_true, y_probs[:, i])
                name = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
                roc_data[f"fpr_{name}"] = fpr_i
                roc_data[f"tpr_{name}"] = tpr_i
                auc_i = float(auc(fpr_i, tpr_i))
                metrics[f"auc_{name}"] = auc_i
        if roc_data:
            save_curves(os.path.join(out, "roc_curves.npz"), **roc_data)

    save_metrics(os.path.join(out, "metrics.json"), metrics)
    save_curves(os.path.join(out, "confusion_matrix.npz"), cm=metrics["confusion_matrix"])

    print(f"  Accuracy={metrics['accuracy']:.4f}  F1_macro={metrics['f1_macro']:.4f}")


def _bootstrap_binary_ci(y_true, y_prob, n, threshold, alpha=0.05):
    """Bootstrap confidence intervals for binary metrics."""
    rng = np.random.RandomState(42)
    aucs, accs, f1s = [], [], []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        aucs.append(float(auc(fpr, tpr)))
        yp_bin = (yp >= threshold).astype(int)
        metrics = compute_binary_metrics(yt, yp_bin)
        accs.append(metrics["accuracy"])
        f1s.append(metrics["f1"])

    lo, hi = alpha / 2, 1 - alpha / 2
    return {
        "auc": [float(np.percentile(aucs, lo * 100)), float(np.percentile(aucs, hi * 100))] if aucs else None,
        "accuracy": [float(np.percentile(accs, lo * 100)), float(np.percentile(accs, hi * 100))] if accs else None,
        "f1": [float(np.percentile(f1s, lo * 100)), float(np.percentile(f1s, hi * 100))] if f1s else None,
    }
