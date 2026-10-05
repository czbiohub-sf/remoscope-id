"""Find optimal decision thresholds per class given minimum sensitivity constraints.

For clinical diagnostics where different classes have different costs of
misclassification (e.g., missing a disease is worse than a false alarm).
"""

import os
import numpy as np
from sklearn.metrics import roc_curve


def build_sensitivity_targets(
    class_names: list,
    min_sensitivity: float,
    sensitivity_targets_str: str | None = None,
) -> dict:
    """Build a dict mapping class name → target sensitivity.

    Parameters
    ----------
    class_names : list
        All class names.
    min_sensitivity : float
        Default minimum sensitivity for all classes.
    sensitivity_targets_str : str or None
        Optional comma-separated overrides, e.g. "disease:0.99,healthy:0.90".
    """
    targets = {name: min_sensitivity for name in class_names}
    if sensitivity_targets_str:
        for pair in sensitivity_targets_str.split(","):
            pair = pair.strip()
            if ":" not in pair:
                raise ValueError(f"Invalid sensitivity target '{pair}', expected 'name:value'")
            name, val = pair.rsplit(":", 1)
            name = name.strip()
            if name not in targets:
                raise ValueError(f"Unknown class '{name}' in sensitivity targets. "
                                 f"Available: {class_names}")
            targets[name] = float(val.strip())
    return targets


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list,
    sensitivity_targets: dict,
) -> list[dict]:
    """Find the optimal threshold per class (one-vs-rest).

    Algorithm per class:
    1. Sweep thresholds from 0.01 to 0.99
    2. Compute sensitivity and specificity at each
    3. Among thresholds where sensitivity >= target, pick the one maximizing specificity
    4. If none meet the constraint, pick the threshold with maximum sensitivity and warn

    Returns list of dicts with per-class results.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []

    for c, name in enumerate(class_names):
        y_bin = (y_true == c).astype(int)
        y_sc = y_probs[:, c]
        target = sensitivity_targets[name]
        n_pos = int(y_bin.sum())
        n_neg = int((1 - y_bin).sum())

        sensitivities = np.empty(len(thresholds))
        specificities = np.empty(len(thresholds))
        precisions = np.empty(len(thresholds))
        f1s = np.empty(len(thresholds))
        npvs = np.empty(len(thresholds))

        for i, thr in enumerate(thresholds):
            yp = (y_sc >= thr).astype(int)
            tp = int(((yp == 1) & (y_bin == 1)).sum())
            fp = int(((yp == 1) & (y_bin == 0)).sum())
            fn = int(((yp == 0) & (y_bin == 1)).sum())
            tn = int(((yp == 0) & (y_bin == 0)).sum())

            sens = tp / max(1, tp + fn)
            spec = tn / max(1, tn + fp)
            prec = tp / max(1, tp + fp)
            f1 = 2 * prec * sens / max(1e-12, prec + sens)
            npv = tn / max(1, tn + fn)

            sensitivities[i] = sens
            specificities[i] = spec
            precisions[i] = prec
            f1s[i] = f1
            npvs[i] = npv

        # Find thresholds meeting sensitivity constraint
        meets_target = sensitivities >= target
        warning = None

        if meets_target.any():
            # Among those meeting the target, maximize specificity
            candidates = np.where(meets_target)[0]
            best_idx = candidates[int(np.argmax(specificities[candidates]))]
        else:
            # No threshold meets the constraint — pick max sensitivity
            best_idx = int(np.argmax(sensitivities))
            warning = (f"No threshold achieves sensitivity >= {target:.2f} for '{name}'. "
                       f"Best achievable: {sensitivities[best_idx]:.3f}")

        result = {
            "class_name": name,
            "target_sensitivity": target,
            "optimal_threshold": float(thresholds[best_idx]),
            "achieved_sensitivity": float(sensitivities[best_idx]),
            "achieved_specificity": float(specificities[best_idx]),
            "achieved_precision": float(precisions[best_idx]),
            "achieved_f1": float(f1s[best_idx]),
            "achieved_npv": float(npvs[best_idx]),
            "n_positive": n_pos,
            "n_negative": n_neg,
        }
        if warning:
            result["warning"] = warning
        results.append(result)

    return results


def find_youden_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list,
) -> list[dict]:
    """Per-class Youden-J optimal threshold (maximises sensitivity + specificity - 1).

    Returns the same dict schema as :func:`find_optimal_thresholds` so the two
    outputs can be concatenated into one operating-point table.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    results = []

    for c, name in enumerate(class_names):
        y_bin = (y_true == c).astype(int)
        y_sc = y_probs[:, c]
        n_pos = int(y_bin.sum())
        n_neg = int(len(y_bin) - n_pos)

        fpr, tpr, thr = roc_curve(y_bin, y_sc)
        j_idx = int(np.argmax(tpr - fpr))
        best_thr = float(thr[j_idx]) if np.isfinite(thr[j_idx]) else 1.0

        yp = (y_sc >= best_thr).astype(int)
        tp = int(((yp == 1) & (y_bin == 1)).sum())
        fp = int(((yp == 1) & (y_bin == 0)).sum())
        fn = int(((yp == 0) & (y_bin == 1)).sum())
        tn = int(((yp == 0) & (y_bin == 0)).sum())

        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        prec = tp / max(1, tp + fp)
        f1 = 2 * prec * sens / max(1e-12, prec + sens)
        npv = tn / max(1, tn + fn)

        results.append({
            "class_name": name,
            "rule": "youden_j",
            "optimal_threshold": best_thr,
            "achieved_sensitivity": float(sens),
            "achieved_specificity": float(spec),
            "achieved_precision": float(prec),
            "achieved_f1": float(f1),
            "achieved_npv": float(npv),
            "n_positive": n_pos,
            "n_negative": n_neg,
        })

    return results


def dual_operating_points_table(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list,
    sensitivity_targets: dict,
    output_csv: str | None = None,
):
    """Operating-point table: per-class Youden-J row + high-sensitivity rule-out row.

    Columns: class, rule, threshold, sensitivity, specificity, PPV, NPV, F1.
    Writes a long-format CSV with two rows per class.
    """
    import pandas as pd

    youden = find_youden_thresholds(y_true, y_probs, class_names)
    highsens = find_optimal_thresholds(y_true, y_probs, class_names, sensitivity_targets)

    def _row(r: dict) -> dict:
        tgt = r.get("target_sensitivity")
        rule = "youden_j" if tgt is None else f"sens_ge_{tgt:.2f}"
        return {
            "class": r["class_name"],
            "rule": rule,
            "target_sensitivity": tgt,
            "threshold": r["optimal_threshold"],
            "sensitivity": r["achieved_sensitivity"],
            "specificity": r["achieved_specificity"],
            "ppv": r["achieved_precision"],
            "npv": r["achieved_npv"],
            "f1": r["achieved_f1"],
            "n_positive": r["n_positive"],
            "n_negative": r["n_negative"],
        }

    df = pd.DataFrame([_row(r) for r in youden] + [_row(r) for r in highsens])
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved dual operating-points table: {output_csv}")
    return df


def find_thresholds_per_fold(
    per_fold: list,
    class_names: list,
    rule: str = "sens_constrained",
    sensitivity_targets: dict | None = None,
):
    """For each CV fold, pick the operating threshold on that fold's held-out preds.

    Parameters
    ----------
    per_fold : list of dicts
        Each entry has keys: ``y_true``, ``y_probs``, ``repeat``, ``fold``
        (as produced by :func:`remo_id.workflows.cv.collect_cv_predictions_by_fold`).
    rule : {"sens_constrained", "youden_j"}
    sensitivity_targets : dict, required when rule == "sens_constrained"

    Returns
    -------
    DataFrame with one row per (fold, class): threshold + achieved sens/spec/ppv/npv/f1.
    """
    import pandas as pd

    if rule == "sens_constrained" and sensitivity_targets is None:
        raise ValueError("sensitivity_targets required when rule='sens_constrained'")

    rows = []
    for fold_data in per_fold:
        fold_label = f"repeat{fold_data.get('repeat', 0)}_fold{fold_data.get('fold', 0)}"
        y_true = np.asarray(fold_data["y_true"])
        y_probs = np.asarray(fold_data["y_probs"])

        if rule == "sens_constrained":
            per_class = find_optimal_thresholds(
                y_true, y_probs, class_names, sensitivity_targets,
            )
        elif rule == "youden_j":
            per_class = find_youden_thresholds(y_true, y_probs, class_names)
        else:
            raise ValueError(f"unknown rule: {rule}")

        for r in per_class:
            rows.append({
                "fold": fold_label,
                "class": r["class_name"],
                "rule": rule,
                "threshold": r["optimal_threshold"],
                "sensitivity": r["achieved_sensitivity"],
                "specificity": r["achieved_specificity"],
                "ppv": r["achieved_precision"],
                "npv": r["achieved_npv"],
                "f1": r["achieved_f1"],
                "n_positive": r["n_positive"],
                "n_negative": r["n_negative"],
            })
    return pd.DataFrame(rows)


def summarize_per_fold_thresholds(per_fold_df):
    """Per-class median/IQR summary of threshold, sensitivity, specificity."""
    import pandas as pd

    def _stats(s):
        return pd.Series({
            "median": float(s.median()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        })

    summary_rows = []
    for (cls, rule), grp in per_fold_df.groupby(["class", "rule"]):
        row = {"class": cls, "rule": rule, "n_folds": len(grp)}
        for col, prefix in [
            ("threshold", "threshold"),
            ("sensitivity", "sens"),
            ("specificity", "spec"),
            ("f1", "f1"),
        ]:
            s = _stats(grp[col])
            for k, v in s.items():
                row[f"{prefix}_{k}"] = v
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def pooled_metrics_with_per_fold_thresholds(
    per_fold: list,
    class_names: list,
    per_fold_df,
) -> "pd.DataFrame":
    """Apply each fold's threshold to its own held-out preds, then pool.

    Mimics deployment: each fold picks a threshold from its (training-independent)
    held-out data; the pooled sens/spec across all held-out experiments is what
    you'd actually observe if you deployed the median/fold-specific threshold.

    Returns DataFrame with one row per class: pooled tp/fp/fn/tn + sens/spec.
    """
    import pandas as pd

    thr_by_fold_class = {
        (r["fold"], r["class"]): float(r["threshold"])
        for _, r in per_fold_df.iterrows()
    }

    rows = []
    for c, name in enumerate(class_names):
        tp = fp = fn = tn = 0
        for fold_data in per_fold:
            fold_label = f"repeat{fold_data.get('repeat', 0)}_fold{fold_data.get('fold', 0)}"
            thr = thr_by_fold_class.get((fold_label, name))
            if thr is None:
                continue
            y_true = np.asarray(fold_data["y_true"])
            y_probs = np.asarray(fold_data["y_probs"])
            y_bin = (y_true == c).astype(int)
            y_sc = y_probs[:, c]
            yp = (y_sc >= thr).astype(int)
            tp += int(((yp == 1) & (y_bin == 1)).sum())
            fp += int(((yp == 1) & (y_bin == 0)).sum())
            fn += int(((yp == 0) & (y_bin == 1)).sum())
            tn += int(((yp == 0) & (y_bin == 0)).sum())

        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        ppv = tp / max(1, tp + fp)
        npv = tn / max(1, tn + fn)
        f1 = 2 * ppv * sens / max(1e-12, ppv + sens)
        rows.append({
            "class": name,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "f1": f1,
        })
    return pd.DataFrame(rows)
