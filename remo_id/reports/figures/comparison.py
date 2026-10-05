"""Comparison figures: seed variability, thresholds, kfold, baselines, etc."""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc

from remo_id.reports import style
from remo_id.reports.figures._bootstrap import _bootstrap_roc_band


# ---------------------------------------------------------------------------
# Seed variability
# ---------------------------------------------------------------------------

def plot_seed_variability(
    metrics_by_seed: dict,
) -> plt.Figure:
    """Violin plots of metric distributions across seeds."""
    style.apply()
    names = list(metrics_by_seed.keys())
    values = [np.asarray(metrics_by_seed[k], dtype=float) for k in names]

    fig, ax = style.figure(width="double" if len(names) > 4 else "single", aspect=0.8)
    positions = list(range(1, len(names) + 1))
    style.tufte_violin(ax, values, positions=positions)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Seed Variability")
    style.trim_spines(ax)
    return fig


# ---------------------------------------------------------------------------
# Threshold sensitivity
# ---------------------------------------------------------------------------

def plot_threshold_sensitivity(
    y_true,
    y_probs,
    class_names: list,
) -> plt.Figure:
    """F1 / precision / recall vs threshold for each class (OvR)."""
    style.apply()
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    n_classes = len(class_names)

    fig, axes_flat = style.small_multiples_grid(
        n_classes, width="single", panel_aspect=0.9, max_cols=2,
        sharex=True, sharey=True,
    )

    thresholds_grid = np.linspace(0.01, 0.99, 99)

    for c in range(n_classes):
        ax = axes_flat[c]
        y_bin = (y_true == c).astype(int)
        y_sc = y_probs[:, c]

        precisions, recalls, f1s = [], [], []
        for thr in thresholds_grid:
            yp = (y_sc >= thr).astype(int)
            tp = int(((yp == 1) & (y_bin == 1)).sum())
            fp = int(((yp == 1) & (y_bin == 0)).sum())
            fn = int(((yp == 0) & (y_bin == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        ax.plot(thresholds_grid, f1s,        color=style.ACCENT, lw=1.0, label="F1")
        ax.plot(thresholds_grid, precisions, color=style.MUTE,   lw=0.7,
                ls=(0, (3, 2)), label="Precision")
        ax.plot(thresholds_grid, recalls,    color=style.FG,     lw=0.7,
                ls=(0, (1, 2)), label="Recall")
        best_thr = thresholds_grid[int(np.argmax(f1s))]
        ax.axvline(best_thr, color=style.MUTE_FAINT, lw=0.4, ls=(0, (1, 1.5)))

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_title(class_names[c])
        if c == 0:
            ax.legend(loc="lower left")

    fig.suptitle("Threshold Sensitivity (OvR)", y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Operating points
# ---------------------------------------------------------------------------

def plot_operating_points(
    y_true,
    y_probs,
    class_names: list,
    *,
    bootstrap_n: int = 1000,
) -> plt.Figure:
    """ROC space with 95% CI bands and operating-point scatter (star = Youden-J)."""
    style.apply()
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    n_classes = len(class_names)
    colors = style.class_colors_muted(class_names)

    fig, ax = style.figure(aspect=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], color=style.MUTE_FAINT, lw=0.5, ls=(0, (1, 1.5)), zorder=1)

    thresholds_grid = np.linspace(0.1, 0.9, 9)

    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        y_sc = y_probs[:, c]
        fpr_curve, tpr_curve, _ = roc_curve(y_bin, y_sc)
        roc_auc = auc(fpr_curve, tpr_curve)
        fpr_g, _, tpr_lo, tpr_hi = _bootstrap_roc_band(y_bin, y_sc, n=bootstrap_n)
        ax.fill_between(fpr_g, tpr_lo, tpr_hi, color=colors[c], alpha=0.18, linewidth=0)
        ax.plot(fpr_curve, tpr_curve, color=colors[c], lw=1.8,
                label=f"{class_names[c]} AUC={roc_auc:.3f}")

        for thr in thresholds_grid:
            yp = (y_sc >= thr).astype(int)
            tp = int(((yp == 1) & (y_bin == 1)).sum())
            fp = int(((yp == 1) & (y_bin == 0)).sum())
            fn = int(((yp == 0) & (y_bin == 1)).sum())
            tn = int(((yp == 0) & (y_bin == 0)).sum())
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            ax.scatter(fpr, tpr, color=colors[c], s=12, alpha=0.8, zorder=3, linewidths=0)

        j_best = int(np.argmax(tpr_curve - fpr_curve))
        ax.scatter(fpr_curve[j_best], tpr_curve[j_best],
                   color=colors[c], s=60, marker="*", zorder=4, linewidths=0)

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Operating Points (star = Youden-J)")
    ax.legend(loc="lower right")
    style.range_frame(ax)
    style.trim_spines(ax)
    return fig


# ---------------------------------------------------------------------------
# K-fold variability
# ---------------------------------------------------------------------------

def plot_kfold_variability(
    fold_metrics: dict,
    class_names: list,
) -> plt.Figure:
    """Box/strip plots of per-fold metrics across CV folds."""
    style.apply()
    names = list(fold_metrics.keys())
    values = [np.asarray(fold_metrics[k], dtype=float) for k in names]

    fig, ax = style.figure(width="double" if len(names) > 4 else "single", aspect=0.8)
    positions = list(range(1, len(names) + 1))
    style.tufte_violin(ax, values, positions=positions)

    # Overlay individual fold points in neutral gray
    for i, vals in enumerate(values):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                   color=style.MUTE, s=5, alpha=0.7, zorder=5, linewidths=0)

    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("K-Fold Variability")
    style.trim_spines(ax)
    return fig


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def plot_baseline_comparison(
    comparison_df: "pd.DataFrame",
    metric: str = "auc_macro",
) -> plt.Figure:
    """Bar chart comparing our model vs sklearn baselines."""
    style.apply()

    fig, ax = style.figure(width="single", aspect=0.8)
    x = np.arange(len(comparison_df))
    # First row (our model) gets the accent; baselines in muted gray
    colors = [style.ACCENT] + [style.MUTE] * (len(comparison_df) - 1)

    ax.bar(x, comparison_df[metric], color=colors, width=0.6, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["model"], rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison ({metric.replace('_', ' ').title()})")

    for i, v in enumerate(comparison_df[metric]):
        if not np.isnan(v):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=5, color=style.FG)

    style.trim_spines(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Post-training analysis plots
# ---------------------------------------------------------------------------

def plot_experiment_probabilities(
    mean_probs: "np.ndarray",
    std_probs: "np.ndarray",
    true_labels: "np.ndarray",
    splits: list,
    study_ids: list,
    *,
    class_names: list = None,
    model_type: str = "",
    n_runs: int = 1,
    n_images: int = 0,
) -> plt.Figure:
    """Experiment probability plot with error bars, sorted by probability."""
    style.apply()
    mean_probs = np.asarray(mean_probs)
    std_probs = np.asarray(std_probs)
    true_labels = np.asarray(true_labels)

    if class_names is None:
        class_names = ["Negative", "Positive"]
    neg_name, pos_name = class_names[0], class_names[1] if len(class_names) > 1 else "Positive"

    order = np.argsort(mean_probs)

    unique_studies = sorted(set(study_ids))
    study_to_marker = {sid: style.MARKERS[i % len(style.MARKERS)]
                       for i, sid in enumerate(unique_studies)}

    pos_color = style.ACCENT
    neg_color = style.MUTED_CLASS[0]

    fig, ax = style.figure(width="double", aspect=0.5)

    for idx, i in enumerate(order):
        marker = study_to_marker[study_ids[i]]
        color = pos_color if true_labels[i] == 1 else neg_color

        if splits[i] == "train":
            ax.errorbar(idx, mean_probs[i], yerr=std_probs[i], fmt=marker, color=color,
                        markerfacecolor=color, markersize=3, capsize=1.5,
                        elinewidth=0.4, alpha=0.8)
        else:
            ax.errorbar(idx, mean_probs[i], yerr=std_probs[i], fmt=marker, color=color,
                        markerfacecolor='white', markeredgecolor=color, markeredgewidth=0.6,
                        markersize=3, capsize=1.5, elinewidth=0.4, alpha=0.9)

    ax.axhline(y=0.5, color=style.MUTE, linestyle=(0, (3, 2)), linewidth=0.5)
    ax.set_xlabel("Experiments (sorted by probability)")
    ax.set_ylabel("Mean Probability")

    subtitle = f"[{model_type} | {n_runs} runs x {n_images} images]" if model_type else ""
    ax.set_title(f"Experiment Probabilities\n{subtitle}")
    ax.set_ylim(-0.05, 1.05)

    legend_elements = []
    for sid in unique_studies:
        m = study_to_marker[sid]
        legend_elements.append(
            Line2D([0], [0], marker=m, color='w',
                   markerfacecolor=style.MUTE, markersize=3, label=str(sid))
        )
    legend_elements.append(Line2D([0], [0], color='w', label=''))
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor=pos_color,
               markersize=3, label=f'True {pos_name}')
    )
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor=neg_color,
               markersize=3, label=f'True {neg_name}')
    )
    legend_elements.append(Line2D([0], [0], color='w', label=''))
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor=style.MUTE,
               markersize=3, label='Train (filled)')
    )
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor=style.MUTE, markeredgewidth=0.6,
               markersize=3, label='Val (hollow)')
    )
    legend_elements.append(Line2D([0], [0], color='w', label=''))
    legend_elements.append(Line2D([0], [0], color=style.MUTE,
                                  linestyle=(0, (3, 2)), label='Threshold (0.5)'))

    ax.legend(handles=legend_elements, loc='upper left')
    style.trim_spines(ax)
    fig.tight_layout()
    return fig


def plot_roc_binary_splits(
    y_true_all: "np.ndarray",
    y_prob_all: "np.ndarray",
    y_true_val: "np.ndarray",
    y_prob_val: "np.ndarray",
    *,
    model_type: str = "",
    n_runs: int = 1,
    n_images: int = 0,
) -> plt.Figure:
    """Simple binary ROC showing total vs validation curves."""
    style.apply()
    y_true_all = np.asarray(y_true_all)
    y_prob_all = np.asarray(y_prob_all)
    y_true_val = np.asarray(y_true_val)
    y_prob_val = np.asarray(y_prob_val)

    fig, ax = style.figure(aspect=1.0)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    ax.plot([0, 1], [0, 1], color=style.MUTE_FAINT, lw=0.5,
            ls=(0, (1, 1.5)), zorder=1)

    if len(np.unique(y_true_all)) > 1:
        fpr_all, tpr_all, _ = roc_curve(y_true_all, y_prob_all)
        auc_all = auc(fpr_all, tpr_all)
        ax.plot(fpr_all, tpr_all, color=style.MUTED_CLASS[0],
                label=f'Total (AUC = {auc_all:.3f})', linewidth=0.8)

    if len(np.unique(y_true_val)) > 1:
        fpr_val, tpr_val, _ = roc_curve(y_true_val, y_prob_val)
        auc_val = auc(fpr_val, tpr_val)
        ax.plot(fpr_val, tpr_val, color=style.ACCENT,
                label=f'Validation (AUC = {auc_val:.3f})', linewidth=0.9)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    subtitle = f"[{model_type} | {n_runs} runs x {n_images} images]" if model_type else ""
    ax.set_title(f"ROC Curve\n{subtitle}")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc='lower right')
    style.range_frame(ax)
    style.trim_spines(ax)
    return fig


def plot_repeatability_hist(
    stds: "np.ndarray",
    *,
    model_type: str = "",
    n_runs: int = 1,
    n_images: int = 0,
) -> plt.Figure:
    """Histogram of per-experiment standard deviations across inference runs."""
    style.apply()
    stds = np.asarray(stds)
    mean_std = float(stds.mean())

    fig, ax = style.figure(width="single", aspect=0.8)
    ax.hist(stds, bins=20, edgecolor=style.FG, linewidth=0.3,
            alpha=0.85, color=style.MUTE_LIGHT)
    ax.axvline(x=mean_std, color=style.ACCENT, linestyle=(0, (3, 2)),
               linewidth=0.7, label=f'Mean = {mean_std:.4f}')

    ax.set_xlabel('Standard Deviation (across runs)')
    ax.set_ylabel('Number of Experiments')

    subtitle = f"[{model_type} | {n_runs} runs x {n_images} images]" if model_type else ""
    ax.set_title(f"Repeatability Distribution\n{subtitle}")
    ax.legend()
    style.trim_spines(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Optimal thresholds (sensitivity-constrained)
# ---------------------------------------------------------------------------

def plot_per_fold_thresholds(
    per_fold_df,
    *,
    target_class,
    jitter: float = 0.08,
    seed: int = 0,
) -> plt.Figure:
    """Strip plot of per-fold operating thresholds per class (stability check).

    Each dot is one fold; the horizontal bar is the class median; the box
    spans the IQR. Target class highlighted in accent color.
    """
    style.apply()
    class_names = per_fold_df["class"].unique().tolist()
    target_idx = _resolve_target_class(class_names, target_class)
    rng = np.random.default_rng(seed)

    fig, ax = style.figure(width="double", aspect=0.5)
    for i, c in enumerate(class_names):
        vals = per_fold_df.loc[per_fold_df["class"] == c, "threshold"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        is_target = (i == target_idx)
        color = style.ACCENT if is_target else style.MUTE
        alpha = 0.85 if is_target else 0.55
        dot_size = 42 if is_target else 28

        jit = rng.normal(0, jitter, size=len(vals))
        ax.scatter(i + jit, vals, color=color, s=dot_size, alpha=alpha,
                   linewidths=0, zorder=3)

        med = float(np.median(vals))
        q25, q75 = np.percentile(vals, [25, 75])
        ax.fill_between([i - 0.28, i + 0.28], [q25, q25], [q75, q75],
                        color=color, alpha=0.12, linewidth=0, zorder=2)
        ax.plot([i - 0.32, i + 0.32], [med, med],
                color=color, lw=2.2 if is_target else 1.4, zorder=4)

    ax.set_xticks(range(len(class_names)))
    labels = [
        f"{c} (target)" if i == target_idx else c
        for i, c in enumerate(class_names)
    ]
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Operating threshold")
    ax.set_ylim(0, 1)
    n_folds = per_fold_df.groupby("class").size().max()
    ax.set_title(f"Per-fold operating thresholds  (n={n_folds} folds; bar = median, box = IQR)")
    style.trim_spines(ax)
    fig.tight_layout()
    return fig


def _resolve_target_class(class_names: list, target_class) -> int:
    """Return the index of ``target_class`` (a class name) in ``class_names``."""
    if target_class in class_names:
        return class_names.index(target_class)
    raise ValueError(
        f"target_class='{target_class}' not in class_names={class_names}"
    )


def plot_optimal_thresholds(
    y_true,
    y_probs,
    class_names: list,
    optimal_results: list[dict],
    *,
    target_class,
    bootstrap_n: int = 1000,
) -> plt.Figure:
    """ROC-space view: ROC curve per class with the sensitivity-constrained
    operating point marked.

    The target class (``target_class``, a class name) is
    drawn in the accent color with a 95% bootstrap CI band and a prominent
    star. Non-target classes are faded gray and differentiated by marker
    shape.
    """
    style.apply()
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    n_classes = len(class_names)
    target_idx = _resolve_target_class(class_names, target_class)
    non_target_markers = ["o", "s", "^", "D", "v", "P", "X"]

    fig, ax = style.figure(aspect=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], color=style.MUTE_FAINT, lw=0.5,
            ls=(0, (1, 1.5)), zorder=1)

    non_target_order = [c for c in range(n_classes) if c != target_idx]
    for marker_idx, c in enumerate(non_target_order):
        y_bin = (y_true == c).astype(int)
        y_sc = y_probs[:, c]
        fpr, tpr, _ = roc_curve(y_bin, y_sc)
        ax.plot(fpr, tpr, color=style.MUTE, lw=1.0, alpha=0.55, zorder=2)

        res = optimal_results[c]
        marker = non_target_markers[marker_idx % len(non_target_markers)]
        op_fpr = 1.0 - res["achieved_specificity"]
        op_tpr = res["achieved_sensitivity"]
        ax.scatter([op_fpr], [op_tpr], facecolor="white", edgecolor=style.MUTE,
                   s=70, marker=marker, linewidths=1.0, zorder=4,
                   label=f"{class_names[c]}  thr={res['optimal_threshold']:.2f}  "
                         f"sens={res['achieved_sensitivity']:.2f}  "
                         f"spec={res['achieved_specificity']:.2f}")

    y_bin = (y_true == target_idx).astype(int)
    y_sc = y_probs[:, target_idx]
    fpr, tpr, _ = roc_curve(y_bin, y_sc)
    fpr_g, _, tpr_lo, tpr_hi = _bootstrap_roc_band(y_bin, y_sc, n=bootstrap_n)
    ax.fill_between(fpr_g, tpr_lo, tpr_hi, color=style.ACCENT, alpha=0.18,
                    linewidth=0, zorder=3)
    ax.plot(fpr, tpr, color=style.ACCENT, lw=2.4, zorder=5)

    res = optimal_results[target_idx]
    op_fpr = 1.0 - res["achieved_specificity"]
    op_tpr = res["achieved_sensitivity"]
    ax.scatter([op_fpr], [op_tpr], color=style.ACCENT, s=200, marker="*",
               linewidths=0, zorder=7,
               label=f"{class_names[target_idx]} (target)  "
                     f"thr={res['optimal_threshold']:.2f}  "
                     f"sens={res['achieved_sensitivity']:.2f}  "
                     f"spec={res['achieved_specificity']:.2f}")

    target_sens = optimal_results[0].get("target_sensitivity", None)
    if target_sens is not None:
        ax.axhline(target_sens, color=style.MUTE, lw=0.5, ls=(0, (1, 2)))
        ax.annotate(f"target sens = {target_sens:.2f}", xy=(0.99, target_sens),
                    xytext=(-2, 2), textcoords="offset points",
                    ha="right", va="bottom", color=style.MUTE)

    ax.set_xlabel("False positive rate  (1 − specificity)")
    ax.set_ylabel("True positive rate  (sensitivity)")
    ax.set_title(f"Optimal Thresholds (target: {class_names[target_idx]})")
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)),
                   key=lambda i: 0 if "(target)" in labels[i] else 1)
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower right", title="class  (sens-constrained)")
    style.range_frame(ax)
    style.trim_spines(ax)
    fig.tight_layout()
    return fig


def plot_dual_operating_points(
    y_true,
    y_probs,
    class_names: list,
    youden_results: list[dict],
    highsens_results: list[dict],
    *,
    target_class,
    bootstrap_n: int = 1000,
) -> plt.Figure:
    """ROC-space view: ROC curve per class with both operating points marked.

    Stars = Youden-J; circles = high-sensitivity rule-out. Target class
    (``target_class``, a class name) drawn in accent with a bootstrap
    CI band; others faded.
    """
    style.apply()
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    n_classes = len(class_names)
    target_idx = _resolve_target_class(class_names, target_class)

    fig, ax = style.figure(aspect=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], color=style.MUTE_FAINT, lw=0.5,
            ls=(0, (1, 1.5)), zorder=1)

    def _op(res):
        return 1.0 - res["achieved_specificity"], res["achieved_sensitivity"]

    non_target_order = [c for c in range(n_classes) if c != target_idx]
    for c in non_target_order:
        y_bin = (y_true == c).astype(int)
        y_sc = y_probs[:, c]
        fpr, tpr, _ = roc_curve(y_bin, y_sc)
        ax.plot(fpr, tpr, color=style.MUTE, lw=1.0, alpha=0.55, zorder=2,
                label=f"{class_names[c]}  YJ thr={youden_results[c]['optimal_threshold']:.2f} "
                      f"(sens={youden_results[c]['achieved_sensitivity']:.2f}, "
                      f"spec={youden_results[c]['achieved_specificity']:.2f})")
        yj_fpr, yj_tpr = _op(youden_results[c])
        hs_fpr, hs_tpr = _op(highsens_results[c])
        ax.scatter([yj_fpr], [yj_tpr], facecolor="white", edgecolor=style.MUTE,
                   s=80, marker="*", linewidths=1.0, zorder=4)
        ax.scatter([hs_fpr], [hs_tpr], facecolor="white", edgecolor=style.MUTE,
                   s=40, marker="o", linewidths=1.0, zorder=4)

    y_bin = (y_true == target_idx).astype(int)
    y_sc = y_probs[:, target_idx]
    fpr, tpr, _ = roc_curve(y_bin, y_sc)
    fpr_g, _, tpr_lo, tpr_hi = _bootstrap_roc_band(y_bin, y_sc, n=bootstrap_n)
    ax.fill_between(fpr_g, tpr_lo, tpr_hi, color=style.ACCENT, alpha=0.18,
                    linewidth=0, zorder=3)
    ax.plot(fpr, tpr, color=style.ACCENT, lw=2.4, zorder=5,
            label=f"{class_names[target_idx]} (target)  "
                  f"YJ thr={youden_results[target_idx]['optimal_threshold']:.2f} "
                  f"(sens={youden_results[target_idx]['achieved_sensitivity']:.2f}, "
                  f"spec={youden_results[target_idx]['achieved_specificity']:.2f})")
    yj_fpr, yj_tpr = _op(youden_results[target_idx])
    hs_fpr, hs_tpr = _op(highsens_results[target_idx])
    ax.scatter([yj_fpr], [yj_tpr], color=style.ACCENT, s=220, marker="*",
               linewidths=0, zorder=7)
    ax.scatter([hs_fpr], [hs_tpr], color=style.ACCENT, s=90, marker="o",
               linewidths=0, zorder=7)

    target_sens = highsens_results[0].get("target_sensitivity", 0.95) if highsens_results else None
    if target_sens is not None:
        ax.axhline(target_sens, color=style.MUTE, lw=0.5, ls=(0, (1, 2)))
        ax.annotate(f"target sens = {target_sens:.2f}", xy=(0.99, target_sens),
                    xytext=(-2, 2), textcoords="offset points",
                    ha="right", va="bottom", color=style.MUTE)

    from matplotlib.lines import Line2D
    marker_legend = [
        Line2D([0], [0], marker="*", color=style.FG, lw=0,
               markersize=12, label="Youden-J"),
        Line2D([0], [0], marker="o", color=style.FG, lw=0,
               markersize=7,
               label=f"sens\u2265{target_sens:.2f}" if target_sens is not None else "sens rule-out"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)),
                   key=lambda i: 0 if "(target)" in labels[i] else 1)
    class_legend = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                             loc="lower right", title="class  (Youden-J operating point)")
    ax.add_artist(class_legend)
    ax.legend(handles=marker_legend, loc="upper left")

    ax.set_xlabel("False positive rate  (1 − specificity)")
    ax.set_ylabel("True positive rate  (sensitivity)")
    ax.set_title(f"Dual Operating Points (target: {class_names[target_idx]})")
    style.range_frame(ax)
    style.trim_spines(ax)
    fig.tight_layout()
    return fig
