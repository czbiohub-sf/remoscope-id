"""Pairwise (one-vs-one) discrimination between two classes from multiclass predictions.

Filters pooled predictions to two class indices, renormalizes probabilities
across the pair, and reports standard binary metrics with bootstrap CIs —
the appropriate analysis when the clinically consequential question is a
specific pair (e.g., disease vs a matched control class), not one-vs-rest.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score

from remo_id.analysis.metrics import compute_binary_metrics
from remo_id.analysis.optimal_thresholds import find_optimal_thresholds
from remo_id.analysis.results import save_metrics, save_curves
from remo_id.reports.figures.classification import (
    plot_roc_curves, plot_confusion_matrix,
)


def filter_pairwise(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list,
    class_positive: str,
    class_negative: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Filter multiclass predictions to a single class pair and renormalize.

    Returns
    -------
    y_true_bin : (M,) array of 0/1 labels (1 = class_positive, 0 = class_negative).
    y_probs_bin : (M, 2) array of renormalized probabilities [P(neg), P(pos)].
    mask : (N,) boolean selecting the M rows kept from the original arrays.
    pos_idx, neg_idx : original class indices.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    if class_positive not in class_names:
        raise ValueError(f"class_positive='{class_positive}' not in class_names={class_names}")
    if class_negative not in class_names:
        raise ValueError(f"class_negative='{class_negative}' not in class_names={class_names}")
    if class_positive == class_negative:
        raise ValueError("class_positive and class_negative must differ")

    pos_idx = class_names.index(class_positive)
    neg_idx = class_names.index(class_negative)

    mask = (y_true == pos_idx) | (y_true == neg_idx)
    y_true_sub = y_true[mask]
    y_probs_sub = y_probs[mask]

    y_true_bin = (y_true_sub == pos_idx).astype(int)

    p_pair = y_probs_sub[:, [neg_idx, pos_idx]]
    denom = p_pair.sum(axis=1, keepdims=True).clip(min=1e-12)
    y_probs_bin = p_pair / denom

    return y_true_bin, y_probs_bin, mask, pos_idx, neg_idx


def _binary_metrics_at_youden(y_bin: np.ndarray, y_sc: np.ndarray) -> dict:
    """AUC, AP, and sens/spec/ppv/npv/f1 at the Youden-J threshold for one resample."""
    auc_val = float(roc_auc_score(y_bin, y_sc))
    ap = float(average_precision_score(y_bin, y_sc))
    fpr, tpr, thr = roc_curve(y_bin, y_sc)
    best_thr = float(thr[int(np.argmax(tpr - fpr))])

    yp = (y_sc >= best_thr).astype(int)
    tp = int(((yp == 1) & (y_bin == 1)).sum())
    fp = int(((yp == 1) & (y_bin == 0)).sum())
    fn = int(((yp == 0) & (y_bin == 1)).sum())
    tn = int(((yp == 0) & (y_bin == 0)).sum())

    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    ppv = tp / max(1, tp + fp)
    npv = tn / max(1, tn + fn)
    f1 = 2 * ppv * sens / max(1e-12, ppv + sens)

    return {
        "auc": auc_val, "ap": ap, "threshold": best_thr,
        "sensitivity": sens, "specificity": spec,
        "ppv": ppv, "npv": npv, "f1": f1,
    }


def pairwise_metrics_table(
    y_true_bin: np.ndarray,
    y_probs_bin: np.ndarray,
    class_positive: str,
    class_negative: str,
    bootstrap_n: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Single-row pairwise metrics table: AUC, AP, Sens, Spec, PPV, NPV, F1 at Youden-J with 95% CIs.

    Bootstraps all metrics in a single resampling loop so roc_curve and the
    Youden-J threshold are computed once per draw.
    """
    y_sc = y_probs_bin[:, 1]
    point = _binary_metrics_at_youden(y_true_bin, y_sc)

    metric_keys = ["auc", "ap", "sensitivity", "specificity", "ppv", "npv", "f1"]
    samples = {k: np.empty(bootstrap_n) for k in metric_keys}
    rng = np.random.default_rng(seed)
    valid = np.ones(bootstrap_n, dtype=bool)
    for i in range(bootstrap_n):
        idx = rng.integers(0, len(y_true_bin), size=len(y_true_bin))
        yt, ys = y_true_bin[idx], y_sc[idx]
        if len(np.unique(yt)) < 2:
            valid[i] = False
            continue
        try:
            m = _binary_metrics_at_youden(yt, ys)
        except Exception:
            valid[i] = False
            continue
        for k in metric_keys:
            samples[k][i] = m[k]

    def ci(arr):
        a = arr[valid]
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    def fmt(p, lo, hi):
        return f"{p:.3f} [{lo:.3f}\u2013{hi:.3f}]"

    row = {
        "pair": f"{class_positive} vs {class_negative}",
        "n_positive": int(y_true_bin.sum()),
        "n_negative": int((1 - y_true_bin).sum()),
        "best_threshold": f"{point['threshold']:.3f}",
    }
    for k in metric_keys:
        lo, hi = ci(samples[k])
        row[k] = fmt(point[k], lo, hi)
    return pd.DataFrame([row])


def run_pairwise(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list,
    class_positive: str,
    class_negative: str,
    output_dir: str,
    bootstrap_n: int = 1000,
    sensitivity_target: float = 0.95,
) -> dict:
    """End-to-end pairwise analysis: metrics, ROC fig, confusion matrix fig.

    Writes into ``output_dir``:
      - metrics_table.csv  — pairwise metrics (single row)
      - metrics.json       — scalar metrics for downstream use
      - optimal_thresholds.json — Youden + sensitivity-target thresholds
      - roc_curve.png      — pairwise ROC with bootstrap CI via plot_roc_curves
      - confusion_matrix.png — 2x2 CM at Youden-J threshold
      - pairwise_curves.npz — fpr/tpr arrays
    """
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)

    y_true_bin, y_probs_bin, mask, pos_idx, neg_idx = filter_pairwise(
        y_true, y_probs, class_names, class_positive, class_negative,
    )
    n_kept = int(mask.sum())
    n_pos = int(y_true_bin.sum())
    n_neg = n_kept - n_pos

    print(f"Pairwise: kept {n_kept}/{len(y_true)} experiments "
          f"({n_pos} {class_positive}, {n_neg} {class_negative})")

    if n_pos < 2 or n_neg < 2:
        raise RuntimeError(
            f"Too few samples in one class (pos={n_pos}, neg={n_neg}) for pairwise analysis"
        )

    table = pairwise_metrics_table(
        y_true_bin, y_probs_bin, class_positive, class_negative, bootstrap_n=bootstrap_n,
    )
    table.to_csv(os.path.join(output_dir, "metrics_table.csv"), index=False)
    print(table.to_string(index=False))

    y_sc = y_probs_bin[:, 1]
    fpr, tpr, thr = roc_curve(y_true_bin, y_sc)
    youden_thr = float(thr[int(np.argmax(tpr - fpr))])
    y_pred_bin = (y_sc >= youden_thr).astype(int)
    bin_metrics = compute_binary_metrics(y_true_bin, y_pred_bin, y_sc)
    bin_metrics["auc"] = float(auc(fpr, tpr))
    bin_metrics["n_positive"] = n_pos
    bin_metrics["n_negative"] = n_neg
    bin_metrics["class_positive"] = class_positive
    bin_metrics["class_negative"] = class_negative
    bin_metrics["youden_threshold"] = youden_thr

    thr_results = find_optimal_thresholds(
        y_true_bin, y_probs_bin,
        class_names=[class_negative, class_positive],
        sensitivity_targets={class_negative: 0.0, class_positive: sensitivity_target},
    )
    bin_metrics["sensitivity_target"] = sensitivity_target
    bin_metrics["sensitivity_target_result"] = next(
        (r for r in thr_results if r["class_name"] == class_positive), None,
    )

    save_metrics(os.path.join(output_dir, "metrics.json"), bin_metrics)
    save_metrics(os.path.join(output_dir, "optimal_thresholds.json"), thr_results)
    save_curves(os.path.join(output_dir, "pairwise_curves.npz"),
                fpr=fpr, tpr=tpr, thresholds=thr)

    pair_class_names = [class_negative, class_positive]

    fig_roc = plot_roc_curves(
        y_true_bin, y_probs_bin, pair_class_names,
        bootstrap_n=bootstrap_n, task_type="binary",
    )
    fig_roc.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close(fig_roc)

    fig_cm = plot_confusion_matrix(y_true_bin, y_pred_bin, pair_class_names)
    fig_cm.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close(fig_cm)

    print(f"Saved pairwise analysis to {output_dir}")
    return {
        "metrics": bin_metrics,
        "table": table,
        "y_true_bin": y_true_bin,
        "y_probs_bin": y_probs_bin,
        "thresholds": thr_results,
    }
