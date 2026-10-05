"""Cross-sweep comparison of CV configurations (one summary table row per config).

Walks a list of CV directories, loads pooled held-out predictions and per-fold
metrics from each, and emits a single CSV row per configuration with macro AUC,
macro/weighted F1, error rate, and per-fold F1 std — the columns needed to
contrast configurations (e.g., 3-class vs binary, one microscope vs all).
"""

import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score,
)
from sklearn.preprocessing import label_binarize

from remo_id.workflows.cv import load_cached_cv_predictions, aggregate_cv_results


def _bootstrap_ci_multiclass(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    y_pred: np.ndarray,
    fn,
    n: int = 1000,
    seed: int = 0,
) -> tuple:
    """Bootstrap 95% CI for a scalar metric that needs (y_true, y_probs, y_pred).

    Resamples experiments with replacement; skips draws that lose a class.
    """
    rng = np.random.default_rng(seed)
    point = fn(y_true, y_probs, y_pred)
    samples = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, yp, ypr = y_true[idx], y_probs[idx], y_pred[idx]
        try:
            samples.append(fn(yt, yp, ypr))
        except Exception:
            continue
    if not samples:
        return float(point), float("nan"), float("nan")
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    return float(point), lo, hi


def _macro_auc(y_true, y_probs, y_pred, num_classes, task_type):
    if task_type == "binary":
        return float(roc_auc_score(y_true, y_probs[:, 1]))
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if num_classes == 2:
        y_bin = np.column_stack([1 - y_bin, y_bin])
    return float(roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr"))


def _macro_f1(y_true, y_probs, y_pred, task_type):
    average = "binary" if task_type == "binary" else "macro"
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def _weighted_f1(y_true, y_probs, y_pred, task_type):
    average = "binary" if task_type == "binary" else "weighted"
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def summarize_cv_dir(
    cv_dir: str,
    class_names: list,
    task_type: str,
    label: Optional[str] = None,
    bootstrap_n: int = 1000,
    use_last: bool = False,
) -> dict:
    """Load pooled predictions for one cv_dir and compute its summary-table row."""
    num_classes = len(class_names)
    pooled, per_fold, missing = load_cached_cv_predictions(
        cv_dir, data_module=None, use_last=use_last,
    )
    if missing:
        raise FileNotFoundError(
            f"{cv_dir}: {len(missing)} folds missing cached predictions "
            f"({', '.join(missing[:3])}{'...' if len(missing) > 3 else ''})"
        )
    y_true, y_probs, y_pred = pooled[0], pooled[1], pooled[2]

    auc_pt, auc_lo, auc_hi = _bootstrap_ci_multiclass(
        y_true, y_probs, y_pred,
        lambda yt, yp, ypr: _macro_auc(yt, yp, ypr, num_classes, task_type),
        n=bootstrap_n,
    )
    mf1_pt, mf1_lo, mf1_hi = _bootstrap_ci_multiclass(
        y_true, y_probs, y_pred,
        lambda yt, yp, ypr: _macro_f1(yt, yp, ypr, task_type),
        n=bootstrap_n,
    )
    wf1_pt, wf1_lo, wf1_hi = _bootstrap_ci_multiclass(
        y_true, y_probs, y_pred,
        lambda yt, yp, ypr: _weighted_f1(yt, yp, ypr, task_type),
        n=bootstrap_n,
    )

    err_rate = float((y_pred != y_true).mean())

    agg = aggregate_cv_results(
        per_fold, pooled, class_names, bootstrap_n=0, task_type=task_type,
    )
    fold_f1s = np.asarray(agg["per_fold_metrics"]["F1"], dtype=float)
    fold_f1_std = float(np.nanstd(fold_f1s))

    def fmt(p, lo, hi):
        return f"{p:.3f} [{lo:.3f}\u2013{hi:.3f}]"

    return {
        "config": label or os.path.basename(cv_dir.rstrip("/")),
        "task": task_type,
        "n_classes": num_classes,
        "n_experiments": int(len(y_true)),
        "n_folds": len(per_fold),
        "macro_auc": fmt(auc_pt, auc_lo, auc_hi),
        "macro_f1": fmt(mf1_pt, mf1_lo, mf1_hi),
        "weighted_f1": fmt(wf1_pt, wf1_lo, wf1_hi),
        "error_rate": f"{err_rate:.3f}",
        "fold_f1_std": f"{fold_f1_std:.3f}",
    }


def compare_configurations(
    configs: Iterable[dict],
    output_path: str,
    bootstrap_n: int = 1000,
) -> pd.DataFrame:
    """Build the summary table from a list of config descriptors.

    Each config dict must have keys: ``cv_dir``, ``class_names``, ``task_type``.
    Optional: ``label`` (human-readable name) and ``use_last`` (bool).
    """
    rows = []
    for cfg in configs:
        row = summarize_cv_dir(
            cv_dir=cfg["cv_dir"],
            class_names=cfg["class_names"],
            task_type=cfg["task_type"],
            label=cfg.get("label"),
            bootstrap_n=bootstrap_n,
            use_last=cfg.get("use_last", False),
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved configuration comparison: {output_path}")
    print(df.to_string(index=False))
    return df
