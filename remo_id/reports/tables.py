"""Table formatting (LaTeX + CSV).

Functions produce DataFrames with formatted strings (e.g. "0.95 [0.92-0.97]").
"""

import numpy as np
import pandas as pd


def _argmax_metric_fn(c, metric):
    """Closure for bootstrapping argmax-based metrics for class c.

    Works with _bootstrap_ci(y_true, y_probs, fn) where y_probs is 2-D.
    """
    def fn(y_true, y_probs):
        y_bin = (y_true == c).astype(int)
        y_pred_c = (y_probs.argmax(axis=1) == c).astype(int)
        tp = int(((y_pred_c == 1) & (y_bin == 1)).sum())
        fp = int(((y_pred_c == 1) & (y_bin == 0)).sum())
        fn_count = int(((y_pred_c == 0) & (y_bin == 1)).sum())
        tn = int(((y_pred_c == 0) & (y_bin == 0)).sum())
        if metric == "sensitivity":
            return tp / max(1, tp + fn_count)
        elif metric == "specificity":
            return tn / max(1, tn + fp)
        else:  # f1
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn_count)
            return 2 * prec * rec / max(1e-12, prec + rec)
    return fn


def format_performance_table(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list,
    bootstrap_n: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-class + macro metrics with 95% bootstrap CIs.

    Args:
        y_true: Ground truth class indices [N]
        y_probs: Probability matrix [N, C]
        class_names: Class name strings
        bootstrap_n: Bootstrap iterations
        seed: Random seed

    Returns:
        DataFrame with formatted metric strings
    """
    from remo_id.reports.figures._bootstrap import _bootstrap_ci
    from sklearn.metrics import roc_auc_score, average_precision_score

    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    n_classes = len(class_names)

    def _fmt(point, lo, hi):
        return f"{point:.3f} [{lo:.3f}\u2013{hi:.3f}]"

    rows = []

    for c, name in enumerate(class_names):
        y_bin = (y_true == c).astype(int)
        y_sc = y_probs[:, c]
        # AUC
        auc_pt, auc_lo, auc_hi = _bootstrap_ci(y_bin, y_sc, roc_auc_score, n=bootstrap_n, seed=seed)

        # AP
        ap_pt, ap_lo, ap_hi = _bootstrap_ci(y_bin, y_sc, average_precision_score, n=bootstrap_n, seed=seed)

        # Bootstrap CIs for sensitivity, specificity, F1 (argmax-based)
        sens_pt, sens_lo, sens_hi = _bootstrap_ci(
            y_true, y_probs, _argmax_metric_fn(c, "sensitivity"), n=bootstrap_n, seed=seed)
        spec_pt, spec_lo, spec_hi = _bootstrap_ci(
            y_true, y_probs, _argmax_metric_fn(c, "specificity"), n=bootstrap_n, seed=seed)
        f1_pt, f1_lo, f1_hi = _bootstrap_ci(
            y_true, y_probs, _argmax_metric_fn(c, "f1"), n=bootstrap_n, seed=seed)

        # Brier
        brier = float(np.mean((y_sc - y_bin) ** 2))

        rows.append({
            "Class": name,
            "N": int(y_bin.sum()),
            "AUC": _fmt(auc_pt, auc_lo, auc_hi),
            "AP": _fmt(ap_pt, ap_lo, ap_hi),
            "Sensitivity": _fmt(sens_pt, sens_lo, sens_hi),
            "Specificity": _fmt(spec_pt, spec_lo, spec_hi),
            "F1": _fmt(f1_pt, f1_lo, f1_hi),
            "Brier": f"{brier:.4f}",
        })

    # Macro row with bootstrap CI (joint resampling captures covariance)
    def _macro_auc_fn(y_true_b, y_probs_b):
        aucs = []
        for c_i in range(n_classes):
            y_bin_i = (y_true_b == c_i).astype(int)
            if len(np.unique(y_bin_i)) < 2:
                continue
            aucs.append(roc_auc_score(y_bin_i, y_probs_b[:, c_i]))
        return np.mean(aucs) if aucs else np.nan

    macro_pt, macro_lo, macro_hi = _bootstrap_ci(
        y_true, y_probs, _macro_auc_fn, n=bootstrap_n, seed=seed)

    rows.append({
        "Class": "Macro",
        "N": len(y_true),
        "AUC": _fmt(macro_pt, macro_lo, macro_hi),
        "AP": "",
        "Sensitivity": "",
        "Specificity": "",
        "F1": "",
        "Brier": "",
    })

    return pd.DataFrame(rows)


def format_baseline_table(
    our_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    model_name: str = "Our model",
) -> pd.DataFrame:
    """Combine our model results with baseline results into one table."""
    our_row = {
        "model": model_name,
        "accuracy": our_results.get("accuracy", np.nan),
        "balanced_accuracy": our_results.get("balanced_accuracy", np.nan),
        "f1_macro": our_results.get("f1_macro", np.nan),
        "auc_macro": our_results.get("auc_macro", np.nan),
    }

    combined = pd.concat([
        pd.DataFrame([our_row]),
        baseline_results[["model", "accuracy", "balanced_accuracy", "f1_macro", "auc_macro"]],
    ], ignore_index=True)

    return combined


def format_dataset_table(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Format dataset breakdown table."""
    formatted = stats_df[["class", "n_experiments", "n_images"]].copy()
    formatted.columns = ["Class", "Experiments", "Images"]

    # Add totals row
    totals = pd.DataFrame([{
        "Class": "Total",
        "Experiments": formatted["Experiments"].sum(),
        "Images": formatted["Images"].sum(),
    }])
    formatted = pd.concat([formatted, totals], ignore_index=True)

    return formatted


def to_latex(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Convert DataFrame to LaTeX table string."""
    latex = df.to_latex(index=False, escape=False)

    if caption or label:
        lines = latex.split("\n")
        # Insert caption and label after \\begin{tabular}
        for i, line in enumerate(lines):
            if "\\begin{tabular}" in line:
                lines.insert(i, f"\\caption{{{caption}}}")
                lines.insert(i + 1, f"\\label{{{label}}}")
                break

        # Wrap in table environment
        lines.insert(0, "\\begin{table}[htbp]")
        lines.insert(1, "\\centering")
        lines.append("\\end{table}")
        latex = "\n".join(lines)

    return latex
