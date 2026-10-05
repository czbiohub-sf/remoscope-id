"""Detailed performance tables with bootstrap CIs at given operating thresholds.

Computes per-class OvR metrics at externally-provided thresholds (unlike
metrics_table_with_ci which always uses Youden-J). All metrics in a single
bootstrap loop per class for efficiency.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _ovr_metrics_at_threshold(y_bin, y_sc, threshold):
    """Compute all metrics for one OvR class at a fixed threshold."""
    yp = (y_sc >= threshold).astype(int)
    tp = int(((yp == 1) & (y_bin == 1)).sum())
    fp = int(((yp == 1) & (y_bin == 0)).sum())
    fn = int(((yp == 0) & (y_bin == 1)).sum())
    tn = int(((yp == 0) & (y_bin == 0)).sum())

    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    ppv = tp / max(1, tp + fp)
    npv = tn / max(1, tn + fn)
    f1 = 2 * ppv * sens / max(1e-12, ppv + sens)
    acc = (tp + tn) / max(1, tp + fp + fn + tn)

    try:
        auc_val = float(roc_auc_score(y_bin, y_sc))
    except Exception:
        auc_val = float("nan")

    return {
        "auc": auc_val,
        "f1": f1,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
    }


def _fmt(point, lo, hi):
    return f"{point:.3f} [{lo:.3f}\u2013{hi:.3f}]"


def compute_full_metrics_at_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list,
    thresholds: dict,
    bootstrap_n: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-class OvR metrics at given thresholds with bootstrap 95% CIs.

    Parameters
    ----------
    y_true : (N,) int array of class indices
    y_probs : (N, C) float array of probabilities
    class_names : list of class name strings
    thresholds : dict mapping class_name -> float threshold
    bootstrap_n : number of bootstrap iterations
    seed : random seed for bootstrap

    Returns
    -------
    DataFrame with columns: class, threshold, AUC, F1, Accuracy,
    Sensitivity, Specificity, PPV, NPV, Support.
    All metrics formatted as "X.XXX [X.XXX-X.XXX]".
    Final row is macro average.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    rng = np.random.default_rng(seed)
    metric_keys = ["auc", "f1", "accuracy", "sensitivity", "specificity", "ppv", "npv"]

    rows = []
    per_class_points = {}

    for c, name in enumerate(class_names):
        y_bin = (y_true == c).astype(int)
        y_sc = y_probs[:, c]
        thr = thresholds[name]

        # Point estimate
        point = _ovr_metrics_at_threshold(y_bin, y_sc, thr)
        per_class_points[name] = point

        # Bootstrap
        samples = {k: np.empty(bootstrap_n) for k in metric_keys}
        valid = np.ones(bootstrap_n, dtype=bool)
        for i in range(bootstrap_n):
            idx = rng.integers(0, len(y_true), size=len(y_true))
            yb, ys = y_bin[idx], y_sc[idx]
            if len(np.unique(yb)) < 2:
                valid[i] = False
                continue
            try:
                m = _ovr_metrics_at_threshold(yb, ys, thr)
            except Exception:
                valid[i] = False
                continue
            for k in metric_keys:
                samples[k][i] = m[k]

        def ci(k):
            a = samples[k][valid]
            if len(a) == 0:
                return 0.0, 0.0
            return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

        row = {"class": name, "threshold": f"{thr:.3f}", "support": int(y_bin.sum())}
        for k in metric_keys:
            lo, hi = ci(k)
            row[k] = _fmt(point[k], lo, hi)
        rows.append(row)

    # Macro row: mean of per-class point estimates, CI from bootstrap of class means
    macro_samples = {k: np.empty(bootstrap_n) for k in metric_keys}
    macro_valid = np.ones(bootstrap_n, dtype=bool)
    for i in range(bootstrap_n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        class_vals = {k: [] for k in metric_keys}
        all_ok = True
        for c, name in enumerate(class_names):
            yb = (y_true[idx] == c).astype(int)
            ys = y_probs[idx, c]
            if len(np.unique(yb)) < 2:
                all_ok = False
                break
            try:
                m = _ovr_metrics_at_threshold(yb, ys, thresholds[name])
            except Exception:
                all_ok = False
                break
            for k in metric_keys:
                class_vals[k].append(m[k])
        if not all_ok:
            macro_valid[i] = False
            continue
        for k in metric_keys:
            macro_samples[k][i] = np.mean(class_vals[k])

    macro_point = {k: np.mean([per_class_points[n][k] for n in class_names]) for k in metric_keys}
    macro_row = {"class": "macro", "threshold": "", "support": len(y_true)}
    for k in metric_keys:
        a = macro_samples[k][macro_valid]
        if len(a) > 0:
            lo, hi = float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
        else:
            lo, hi = 0.0, 0.0
        macro_row[k] = _fmt(macro_point[k], lo, hi)
    rows.append(macro_row)

    df = pd.DataFrame(rows)
    # Rename columns to presentation format
    df = df.rename(columns={
        "auc": "AUC",
        "f1": "F1",
        "accuracy": "Accuracy",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "ppv": "PPV",
        "npv": "NPV",
        "class": "Class",
        "threshold": "Threshold",
        "support": "Support",
    })
    return df


def _argmax_class_metrics(y_true, y_pred, class_idx):
    """Compute per-class metrics from multiclass argmax predictions."""
    tp = int(((y_pred == class_idx) & (y_true == class_idx)).sum())
    fp = int(((y_pred == class_idx) & (y_true != class_idx)).sum())
    fn = int(((y_pred != class_idx) & (y_true == class_idx)).sum())
    tn = int(((y_pred != class_idx) & (y_true != class_idx)).sum())

    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    ppv = tp / max(1, tp + fp)
    npv = tn / max(1, tn + fn)
    f1 = 2 * ppv * sens / max(1e-12, ppv + sens)
    acc = (tp + tn) / max(1, tp + fp + fn + tn)

    return {
        "f1": f1,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
    }


def compute_argmax_metrics_table(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list,
    bootstrap_n: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-class metrics using argmax predictions with bootstrap 95% CIs.

    Unlike threshold-based metrics, this uses argmax(y_probs) as the
    prediction for each sample. AUC is still computed from probabilities.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    y_pred = y_probs.argmax(axis=1)
    rng = np.random.default_rng(seed)
    metric_keys = ["auc", "f1", "accuracy", "sensitivity", "specificity", "ppv", "npv"]

    rows = []
    per_class_points = {}

    for c, name in enumerate(class_names):
        y_bin = (y_true == c).astype(int)
        y_sc = y_probs[:, c]

        point = _argmax_class_metrics(y_true, y_pred, c)
        try:
            point["auc"] = float(roc_auc_score(y_bin, y_sc))
        except Exception:
            point["auc"] = float("nan")
        per_class_points[name] = point

        # Bootstrap
        samples = {k: np.empty(bootstrap_n) for k in metric_keys}
        valid = np.ones(bootstrap_n, dtype=bool)
        for i in range(bootstrap_n):
            idx = rng.integers(0, len(y_true), size=len(y_true))
            yt_b, yp_b = y_true[idx], y_probs[idx]
            ypred_b = yp_b.argmax(axis=1)
            ybin_b = (yt_b == c).astype(int)
            if len(np.unique(ybin_b)) < 2:
                valid[i] = False
                continue
            m = _argmax_class_metrics(yt_b, ypred_b, c)
            try:
                m["auc"] = float(roc_auc_score(ybin_b, yp_b[:, c]))
            except Exception:
                valid[i] = False
                continue
            for k in metric_keys:
                samples[k][i] = m[k]

        def ci(k):
            a = samples[k][valid]
            if len(a) == 0:
                return 0.0, 0.0
            return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

        row = {"class": name, "threshold": "argmax", "support": int(y_bin.sum())}
        for k in metric_keys:
            lo, hi = ci(k)
            row[k] = _fmt(point[k], lo, hi)
        rows.append(row)

    # Macro row
    macro_samples = {k: np.empty(bootstrap_n) for k in metric_keys}
    macro_valid = np.ones(bootstrap_n, dtype=bool)
    for i in range(bootstrap_n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt_b, yp_b = y_true[idx], y_probs[idx]
        ypred_b = yp_b.argmax(axis=1)
        class_vals = {k: [] for k in metric_keys}
        all_ok = True
        for c, name in enumerate(class_names):
            ybin_b = (yt_b == c).astype(int)
            if len(np.unique(ybin_b)) < 2:
                all_ok = False
                break
            m = _argmax_class_metrics(yt_b, ypred_b, c)
            try:
                m["auc"] = float(roc_auc_score(ybin_b, yp_b[:, c]))
            except Exception:
                all_ok = False
                break
            for k in metric_keys:
                class_vals[k].append(m[k])
        if not all_ok:
            macro_valid[i] = False
            continue
        for k in metric_keys:
            macro_samples[k][i] = np.mean(class_vals[k])

    macro_point = {k: np.mean([per_class_points[n][k] for n in class_names]) for k in metric_keys}
    macro_row = {"class": "macro", "threshold": "", "support": len(y_true)}
    for k in metric_keys:
        a = macro_samples[k][macro_valid]
        if len(a) > 0:
            lo, hi = float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
        else:
            lo, hi = 0.0, 0.0
        macro_row[k] = _fmt(macro_point[k], lo, hi)
    rows.append(macro_row)

    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "auc": "AUC", "f1": "F1", "accuracy": "Accuracy",
        "sensitivity": "Sensitivity", "specificity": "Specificity",
        "ppv": "PPV", "npv": "NPV", "class": "Class",
        "threshold": "Threshold", "support": "Support",
    })
    return df


def compute_fold_f1_std(
    per_fold: list,
    class_names: list,
    thresholds: dict,
) -> dict:
    """Per-fold OvR F1 at given thresholds; return std across folds.

    Parameters
    ----------
    per_fold : list of dicts with keys: y_true, y_probs
    class_names : list of class name strings
    thresholds : dict mapping class_name -> float threshold

    Returns
    -------
    Dict mapping class_name -> std, plus "macro" -> macro_std.
    """
    per_class_f1s = {name: [] for name in class_names}
    macro_f1s = []

    for fold_data in per_fold:
        yt = np.asarray(fold_data["y_true"])
        yp = np.asarray(fold_data["y_probs"])

        fold_f1s = []
        for c, name in enumerate(class_names):
            y_bin = (yt == c).astype(int)
            y_sc = yp[:, c]
            thr = thresholds[name]

            pred = (y_sc >= thr).astype(int)
            tp = int(((pred == 1) & (y_bin == 1)).sum())
            fp = int(((pred == 1) & (y_bin == 0)).sum())
            fn = int(((pred == 0) & (y_bin == 1)).sum())

            ppv = tp / max(1, tp + fp)
            sens = tp / max(1, tp + fn)
            f1 = 2 * ppv * sens / max(1e-12, ppv + sens)

            per_class_f1s[name].append(f1)
            fold_f1s.append(f1)

        macro_f1s.append(np.mean(fold_f1s))

    result = {}
    for name in class_names:
        vals = np.array(per_class_f1s[name])
        result[name] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    result["macro"] = float(np.array(macro_f1s).std(ddof=1)) if len(macro_f1s) > 1 else 0.0
    return result
