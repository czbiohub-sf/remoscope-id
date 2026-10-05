"""Classification figure generation: ROC, PR, confusion matrix, calibration, etc.

Multiclass conventions
----------------------
y_true  : 1-D int array  [N]          class indices
y_probs : 2-D float array [N, C]      softmax probability vectors
y_pred  : 1-D int array  [N]          argmax predictions (where needed)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve

from remo_id.reports import style
from remo_id.reports.figures._bootstrap import (
    _bootstrap_ci, _bootstrap_roc_band, _bootstrap_pr_band,
)


# ---------------------------------------------------------------------------
# Youden-J metric helpers (re-compute threshold per bootstrap resample)
# ---------------------------------------------------------------------------

def _youden_f1(y_bin, y_sc):
    """F1 score at Youden-J optimal threshold."""
    fpr, tpr, thr = roc_curve(y_bin, y_sc)
    best_thr = float(thr[int(np.argmax(tpr - fpr))])
    yp = (y_sc >= best_thr).astype(int)
    tp = int(((yp == 1) & (y_bin == 1)).sum())
    fp = int(((yp == 1) & (y_bin == 0)).sum())
    fn = int(((yp == 0) & (y_bin == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return 2 * prec * rec / max(1e-12, prec + rec)


def _youden_precision(y_bin, y_sc):
    """Precision at Youden-J optimal threshold."""
    fpr, tpr, thr = roc_curve(y_bin, y_sc)
    best_thr = float(thr[int(np.argmax(tpr - fpr))])
    yp = (y_sc >= best_thr).astype(int)
    tp = int(((yp == 1) & (y_bin == 1)).sum())
    fp = int(((yp == 1) & (y_bin == 0)).sum())
    return tp / max(1, tp + fp)


def _youden_recall(y_bin, y_sc):
    """Recall at Youden-J optimal threshold."""
    fpr, tpr, thr = roc_curve(y_bin, y_sc)
    best_thr = float(thr[int(np.argmax(tpr - fpr))])
    yp = (y_sc >= best_thr).astype(int)
    tp = int(((yp == 1) & (y_bin == 1)).sum())
    fn = int(((yp == 0) & (y_bin == 1)).sum())
    return tp / max(1, tp + fn)


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_true,
    y_probs,
    class_names: list,
    *,
    bootstrap_n: int = 1000,
    task_type: str = "multiclass",
) -> plt.Figure:
    """One-vs-rest ROC curves with bootstrap 95 % CI shading.

    For binary tasks, plots a single curve for the positive class.
    """
    style.apply()
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    n_classes = len(class_names)
    colors = style.class_colors_muted(class_names)

    fig, ax = style.figure(aspect=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Chance reference — faint dot-dash
    ax.plot([0, 1], [0, 1], color=style.MUTE_FAINT, lw=0.5, ls=(0, (1, 1.5)), zorder=1)

    def _draw(y_bin, y_sc, color, label):
        fpr, tpr, _ = roc_curve(y_bin, y_sc)
        point, lo, hi = _bootstrap_ci(y_bin, y_sc, roc_auc_score, n=bootstrap_n)
        fpr_g, _, tpr_lo, tpr_hi = _bootstrap_roc_band(y_bin, y_sc, n=bootstrap_n)
        ax.fill_between(fpr_g, tpr_lo, tpr_hi, color=color, alpha=0.18, linewidth=0)
        full_label = f"{label}  AUC={point:.3f} [{lo:.3f}\u2013{hi:.3f}]"
        ax.plot(fpr, tpr, color=color, lw=1.8, label=full_label)

    if task_type == "binary":
        curve_color = colors[1] if len(colors) > 1 else style.ACCENT
        _draw(y_true, y_probs[:, 1], curve_color, class_names[-1])
    else:
        for c in range(n_classes):
            y_bin = (y_true == c).astype(int)
            _draw(y_bin, y_probs[:, c], colors[c], class_names[c])

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curve" if task_type == "binary" else "ROC Curves (OvR)")
    ax.legend(loc="lower right")
    style.range_frame(ax)
    style.trim_spines(ax)
    return fig


# ---------------------------------------------------------------------------
# Precision-Recall curves
# ---------------------------------------------------------------------------

def plot_pr_curves(
    y_true,
    y_probs,
    class_names: list,
    *,
    bootstrap_n: int = 1000,
    task_type: str = "multiclass",
) -> plt.Figure:
    """One-vs-rest precision-recall curves with bootstrap AP CI.

    For binary tasks, plots a single curve for the positive class.
    """
    style.apply()
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    n_classes = len(class_names)
    colors = style.class_colors_muted(class_names)

    fig, ax = style.figure(aspect=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def _draw(y_bin, y_sc, color, label):
        prec, rec, _ = precision_recall_curve(y_bin, y_sc)
        point, lo, hi = _bootstrap_ci(y_bin, y_sc, average_precision_score, n=bootstrap_n)
        rec_g, _, prec_lo, prec_hi = _bootstrap_pr_band(y_bin, y_sc, n=bootstrap_n)
        ax.fill_between(rec_g, prec_lo, prec_hi, color=color, alpha=0.18, linewidth=0)
        full_label = f"{label}  AP={point:.3f} [{lo:.3f}\u2013{hi:.3f}]"
        ax.plot(rec, prec, color=color, lw=1.8, label=full_label)
        prevalence = y_bin.mean()
        ax.axhline(prevalence, color=color, lw=0.4, ls=(0, (1, 2)), alpha=0.5)

    if task_type == "binary":
        curve_color = colors[1] if len(colors) > 1 else style.ACCENT
        _draw(y_true, y_probs[:, 1], curve_color, class_names[-1])
    else:
        for c in range(n_classes):
            y_bin = (y_true == c).astype(int)
            _draw(y_bin, y_probs[:, c], colors[c], class_names[c])

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve" if task_type == "binary" else "PR Curves (OvR)")
    ax.legend(loc="lower left")
    style.range_frame(ax)
    style.trim_spines(ax)
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list,
    title: str = None,
) -> plt.Figure:
    """Confusion matrix with count annotations and row-normalised colour."""
    style.apply()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(class_names)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = style.figure(width="single", aspect=1.0)
    im = ax.imshow(cm_norm, cmap="Greys", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    style.thin_colorbar(cbar)
    style.strip_imshow_frame(ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion Matrix")

    for i in range(n):
        for j in range(n):
            text_color = "white" if cm_norm[i, j] > 0.55 else style.FG
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", color=text_color)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _expected_calibration_error(y_bin, y_sc, n_bins=10):
    """Compute expected calibration error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_sc >= lo) & (y_sc < hi)
        if mask.sum() == 0:
            continue
        frac = y_bin[mask].mean()
        conf = y_sc[mask].mean()
        ece += mask.sum() * abs(frac - conf)
    return ece / len(y_bin) if len(y_bin) > 0 else 0.0


def plot_calibration(
    y_true,
    y_probs,
    class_names: list,
    *,
    task_type: str = "multiclass",
) -> plt.Figure:
    """Reliability diagram (calibration) per class (OvR) with Brier + ECE annotations.

    For binary tasks, plots a single calibration curve for the positive class.
    """
    style.apply()
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    n_classes = len(class_names)
    colors = style.class_colors_muted(class_names)

    fig, ax = style.figure(aspect=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Perfect calibration reference — faint dot-dash (unlabeled; conventional)
    ax.plot([0, 1], [0, 1], color=style.MUTE_FAINT, lw=0.5,
            ls=(0, (1, 1.5)), zorder=1)

    if task_type == "binary":
        y_sc = y_probs[:, 1]
        pos_name = class_names[-1]
        try:
            frac_pos, mean_pred = calibration_curve(y_true, y_sc, n_bins=10, strategy="quantile")
            brier = float(np.mean((y_sc - y_true) ** 2))
            ece = _expected_calibration_error(y_true, y_sc)
            label = f"{pos_name}  Brier={brier:.3f}  ECE={ece:.3f}"
            curve_color = colors[1] if len(colors) > 1 else style.ACCENT
            ax.plot(mean_pred, frac_pos, "s-", color=curve_color, lw=0.8, ms=3, label=label)
        except Exception:
            pass
    else:
        for c in range(n_classes):
            y_bin = (y_true == c).astype(int)
            y_sc = y_probs[:, c]
            try:
                frac_pos, mean_pred = calibration_curve(y_bin, y_sc, n_bins=10, strategy="quantile")
                brier = float(np.mean((y_sc - y_bin) ** 2))
                ece = _expected_calibration_error(y_bin, y_sc)
                label = f"{class_names[c]}  Brier={brier:.3f}  ECE={ece:.3f}"
                ax.plot(mean_pred, frac_pos, "s-", color=colors[c], lw=0.8, ms=3, label=label)
            except Exception:
                pass

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration")
    ax.legend()
    style.range_frame(ax)
    style.trim_spines(ax)
    return fig


# ---------------------------------------------------------------------------
# Probability distributions
# ---------------------------------------------------------------------------

def plot_prob_distributions(
    y_true,
    y_probs,
    class_names: list,
    *,
    task_type: str = "multiclass",
    show_points: bool = False,
    experiment_ids=None,
) -> plt.Figure:
    """Violin plots of predicted probability per true class (OvR).

    For binary tasks, plots a single panel showing P(positive) per true class.

    show_points overlays one jittered dot per experiment on each violin
    (multiclass only). With experiment_ids, argmax-misclassified experiments
    are additionally ring-marked and labeled with their ID in the panel of
    their TRUE class (where their miss shows as a low P(true class)).
    """
    style.apply()
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    n_classes = len(class_names)
    colors_all = style.class_colors_muted(class_names)

    if task_type == "binary":
        fig, ax = style.figure(width="single", aspect=0.9)
        y_sc = y_probs[:, 1]
        pos_name = class_names[-1]
        groups = [y_sc[y_true == k] for k in range(n_classes)]
        non_empty = [(k, g) for k, g in enumerate(groups) if len(g) > 0]
        if non_empty:
            positions = [k for k, _ in non_empty]
            data = [g for _, g in non_empty]
            colors = [colors_all[k] for k, _ in non_empty]
            style.tufte_violin(ax, data, positions=positions, colors=colors)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=30, ha="right")
        ax.set_ylabel(f"P({pos_name})")
        ax.set_ylim(0, 1)
        ax.set_title(f"P({pos_name}) by True Class")
        style.trim_spines(ax)
        fig.tight_layout()
        return fig

    fig, axes = style.small_multiples_grid(
        n_classes, width="double", panel_aspect=0.7, max_cols=2,
        sharex=True, sharey=True,
    )
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.13, 0.13, size=len(y_true))
    miss = y_probs.argmax(axis=1) != y_true
    for c, ax in enumerate(axes):
        y_sc = y_probs[:, c]
        groups = [y_sc[y_true == k] for k in range(n_classes)]
        non_empty = [(k, g) for k, g in enumerate(groups) if len(g) > 0]
        if non_empty:
            positions = [k for k, _ in non_empty]
            data = [g for _, g in non_empty]
            colors = [colors_all[k] for k, _ in non_empty]
            style.tufte_violin(ax, data, positions=positions, colors=colors)
        if show_points:
            for k in range(n_classes):
                sel = y_true == k
                ax.scatter(k + jitter[sel], y_sc[sel], s=6,
                           color=colors_all[k], edgecolors="white",
                           linewidths=0.4, alpha=0.85, zorder=5)
            # Ring + label argmax misses in their TRUE class's panel
            sel = miss & (y_true == c)
            ax.scatter(c + jitter[sel], y_sc[sel], s=22, facecolors="none",
                       edgecolors=style.FG, linewidths=0.7, zorder=6)
            if experiment_ids is not None:
                ids = np.asarray(experiment_ids)
                pts = sorted(zip(c + jitter[sel], y_sc[sel], ids[sel]),
                             key=lambda p: p[1])
                # Stack labels upward so co-located misses (e.g. several at
                # P≈0) don't overprint each other.
                for i, (x_j, y_j, eid) in enumerate(pts):
                    ax.annotate(str(eid), (x_j, y_j), textcoords="offset points",
                                xytext=(5, 3 + i * 6), fontsize=4.5,
                                color=style.MUTE, zorder=7,
                                arrowprops=dict(arrowstyle="-", lw=0.3,
                                                color=style.MUTE_LIGHT)
                                if i else None)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"P({class_names[c]}) by True Class")
        style.trim_spines(ax)

    fig.supylabel("Predicted probability")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Metrics table with CI
# ---------------------------------------------------------------------------

def metrics_table_with_ci(
    y_true,
    y_probs,
    class_names: list,
    n: int = 1000,
    *,
    task_type: str = "multiclass",
) -> pd.DataFrame:
    """DataFrame with AUC, AP, precision/recall/F1 at Youden threshold, 95 % CIs.

    For binary tasks, returns a single row for the positive class using
    standard binary metrics (not OvR binarization).
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    rows = []

    if task_type == "binary":
        # Single row for positive class using y_true directly (0/1)
        y_sc = y_probs[:, 1]
        pos_name = class_names[-1]

        auc_pt, auc_lo, auc_hi = _bootstrap_ci(y_true, y_sc, roc_auc_score, n=n)
        ap_pt, ap_lo, ap_hi = _bootstrap_ci(y_true, y_sc, average_precision_score, n=n)

        fpr_c, tpr_c, thr_c = roc_curve(y_true, y_sc)
        j_idx = int(np.argmax(tpr_c - fpr_c))
        best_thr = float(thr_c[j_idx])

        prec_pt, prec_lo, prec_hi = _bootstrap_ci(y_true, y_sc, _youden_precision, n=n)
        rec_pt, rec_lo, rec_hi = _bootstrap_ci(y_true, y_sc, _youden_recall, n=n)
        f1_pt, f1_lo, f1_hi = _bootstrap_ci(y_true, y_sc, _youden_f1, n=n)

        rows.append({
            "class":           pos_name,
            "n_pos":           int(y_true.sum()),
            "auc":             f"{auc_pt:.3f} [{auc_lo:.3f}\u2013{auc_hi:.3f}]",
            "ap":              f"{ap_pt:.3f} [{ap_lo:.3f}\u2013{ap_hi:.3f}]",
            "best_threshold":  f"{best_thr:.3f}",
            "precision@best":  f"{prec_pt:.3f} [{prec_lo:.3f}\u2013{prec_hi:.3f}]",
            "recall@best":     f"{rec_pt:.3f} [{rec_lo:.3f}\u2013{rec_hi:.3f}]",
            "f1@best":         f"{f1_pt:.3f} [{f1_lo:.3f}\u2013{f1_hi:.3f}]",
        })
        return pd.DataFrame(rows)

    for c, name in enumerate(class_names):
        y_bin = (y_true == c).astype(int)
        y_sc = y_probs[:, c]

        auc_pt, auc_lo, auc_hi = _bootstrap_ci(y_bin, y_sc, roc_auc_score, n=n)
        ap_pt, ap_lo, ap_hi = _bootstrap_ci(y_bin, y_sc, average_precision_score, n=n)

        # Youden-J optimal threshold (point estimate for display)
        fpr_c, tpr_c, thr_c = roc_curve(y_bin, y_sc)
        j_idx = int(np.argmax(tpr_c - fpr_c))
        best_thr = float(thr_c[j_idx])

        # Bootstrap CIs for F1, precision, recall at Youden threshold
        prec_pt, prec_lo, prec_hi = _bootstrap_ci(y_bin, y_sc, _youden_precision, n=n)
        rec_pt, rec_lo, rec_hi = _bootstrap_ci(y_bin, y_sc, _youden_recall, n=n)
        f1_pt, f1_lo, f1_hi = _bootstrap_ci(y_bin, y_sc, _youden_f1, n=n)

        rows.append({
            "class":           name,
            "n_pos":           int(y_bin.sum()),
            "auc":             f"{auc_pt:.3f} [{auc_lo:.3f}\u2013{auc_hi:.3f}]",
            "ap":              f"{ap_pt:.3f} [{ap_lo:.3f}\u2013{ap_hi:.3f}]",
            "best_threshold":  f"{best_thr:.3f}",
            "precision@best":  f"{prec_pt:.3f} [{prec_lo:.3f}\u2013{prec_hi:.3f}]",
            "recall@best":     f"{rec_pt:.3f} [{rec_lo:.3f}\u2013{rec_hi:.3f}]",
            "f1@best":         f"{f1_pt:.3f} [{f1_lo:.3f}\u2013{f1_hi:.3f}]",
        })

    return pd.DataFrame(rows)
