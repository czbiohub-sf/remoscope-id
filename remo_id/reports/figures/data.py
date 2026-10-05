"""Data-related figures: bag size, sampling, dataset stats, errors, learning curves."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from remo_id.reports import style


# ---------------------------------------------------------------------------
# Bag-size sweep
# ---------------------------------------------------------------------------

def plot_bag_size_sweep(
    bag_sizes: list,
    aucs_by_seed,
) -> plt.Figure:
    """Mean +/- SD macro-AUC vs bag size."""
    style.apply()
    aucs = np.asarray(aucs_by_seed, dtype=float)
    mean_auc = aucs.mean(axis=0)
    std_auc = aucs.std(axis=0)

    fig, ax = style.figure()
    ax.fill_between(bag_sizes, mean_auc - std_auc, mean_auc + std_auc,
                    color=style.MUTE_LIGHT, alpha=0.5, linewidth=0, label="\u00b11 SD")
    ax.plot(bag_sizes, mean_auc, "o-", color=style.FG, lw=0.8, ms=2.5)

    # Mark plateau: first size where AUC reaches 99 % of max
    max_auc = mean_auc.max()
    plateau_idx = next(
        (i for i, v in enumerate(mean_auc) if v >= 0.99 * max_auc), None
    )
    if plateau_idx is not None:
        ax.axvline(bag_sizes[plateau_idx], color=style.ACCENT, lw=0.6,
                   ls=(0, (3, 2)),
                   label=f"Plateau (n={bag_sizes[plateau_idx]})")

    style.rug(ax, bag_sizes, side="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Bag size")
    ax.set_ylabel("Macro AUC")
    ax.set_title("AUC vs Bag Size")
    ax.legend()
    style.trim_spines(ax)
    return fig


# ---------------------------------------------------------------------------
# Sampling position
# ---------------------------------------------------------------------------

def plot_sampling_position(
    positions: list,
    aucs,
    *,
    bag_size: int,
) -> plt.Figure:
    """Bar chart comparing sampling strategies at a fixed bag size."""
    style.apply()
    aucs = np.asarray(aucs, dtype=float)
    means = aucs.mean(axis=1)
    sems = aucs.std(axis=1) / np.sqrt(aucs.shape[1])

    best_idx = int(np.argmax(means))
    colors = [style.ACCENT if i == best_idx else style.MUTE for i in range(len(positions))]

    fig, ax = style.figure(width="double", aspect=0.8)
    x = np.arange(len(positions))
    ax.bar(x, means, yerr=sems, color=colors, width=0.6, edgecolor="none",
           error_kw={"elinewidth": 0.4, "capsize": 2, "ecolor": style.FG})
    ax.set_xticks(x)
    ax.set_xticklabels(positions, rotation=20, ha="right")
    ax.set_ylabel("Macro AUC")
    ax.set_title(f"Sampling Position (Bag Size = {bag_size})")
    style.trim_spines(ax)
    return fig


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------

def plot_learning_curve(
    results_df: "pd.DataFrame",
    metric_name: str = "best_val_loss",
) -> plt.Figure:
    """Line plot of metric vs training data fraction."""
    style.apply()

    grouped = results_df.groupby("fraction")[metric_name].agg(["mean", "std"]).reset_index()

    fig, ax = style.figure(width="single", aspect=0.8)
    ax.fill_between(
        grouped["fraction"],
        grouped["mean"] - grouped["std"],
        grouped["mean"] + grouped["std"],
        color=style.MUTE_LIGHT, alpha=0.5, linewidth=0, label="\u00b11 SD",
    )
    ax.plot(grouped["fraction"], grouped["mean"], "o-",
            color=style.FG, lw=0.8, ms=2.5)

    for _, row in results_df.iterrows():
        ax.scatter(row["fraction"], row[metric_name],
                   color=style.MUTE, s=6, alpha=0.4, zorder=2, linewidths=0)

    style.rug(ax, grouped["fraction"].values, side="bottom")
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title("Learning Curve")
    ax.set_xlim(0, 1.05)
    ax.legend()
    style.trim_spines(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def plot_class_distribution(imbalance: dict, class_names: list) -> plt.Figure:
    """Pie chart of class distribution."""
    style.apply()

    counts = [imbalance["class_counts"].get(name, 0) for name in class_names]
    colors = style.class_colors_muted(class_names)

    fig, ax = style.figure(width="single", aspect=1.0)
    wedges, texts, autotexts = ax.pie(
        counts, labels=class_names, autopct="%1.1f%%",
        colors=colors,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        textprops={"fontsize": 6, "color": style.FG},
    )
    for autot in autotexts:
        autot.set_fontsize(5)
        autot.set_color("white")
    ax.set_title(f"Class Distribution (n={sum(counts)}, IR={imbalance['imbalance_ratio']:.1f})")
    return fig


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def plot_error_confidence(
    y_true: "np.ndarray",
    y_pred: "np.ndarray",
    y_probs: "np.ndarray",
    class_names: list,
) -> plt.Figure:
    """Confidence distribution for correct vs incorrect predictions."""
    style.apply()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_probs = np.asarray(y_probs)

    correct = y_true == y_pred
    conf = y_probs.max(axis=1)

    fig, ax = style.figure(width="single", aspect=0.8)

    ax.hist(conf[correct], bins=20, alpha=0.7, color=style.MUTE_LIGHT,
            edgecolor=style.FG, linewidth=0.3,
            label=f"Correct (n={correct.sum()})", density=True)
    ax.hist(conf[~correct], bins=20, alpha=0.7, color=style.ACCENT,
            edgecolor=style.FG, linewidth=0.3,
            label=f"Incorrect (n={(~correct).sum()})", density=True)

    ax.set_xlabel("Max bag probability")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Confidence")
    ax.legend()
    style.trim_spines(ax)
    fig.tight_layout()
    return fig


def plot_error_by_study(
    error_df: "pd.DataFrame",
    class_names: list,
) -> plt.Figure:
    """Error rate breakdown by study_id."""
    style.apply()

    if "study_id" not in error_df.columns or error_df["study_id"].isna().all():
        fig, ax = style.figure()
        ax.text(0.5, 0.5, "No study_id data available", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    study_stats = error_df.groupby("study_id").agg(
        n_total=("correct", "count"),
        n_correct=("correct", "sum"),
    ).reset_index()
    study_stats["error_rate"] = 1 - study_stats["n_correct"] / study_stats["n_total"]
    study_stats = study_stats.sort_values("error_rate", ascending=False)

    fig, ax = style.figure(width="double" if len(study_stats) > 4 else "single", aspect=0.8)
    x = np.arange(len(study_stats))

    # Top-3 worst performers get accent color; rest in muted gray
    n_highlight = min(3, len(study_stats))
    colors = [style.ACCENT if i < n_highlight else style.MUTE for i in range(len(study_stats))]

    ax.bar(x, study_stats["error_rate"], color=colors, width=0.6, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(study_stats["study_id"], rotation=30, ha="right")
    ax.set_ylabel("Error rate")
    ax.set_title("Error Rate by Study")
    ax.set_ylim(0, 1)

    for i, (_, row) in enumerate(study_stats.iterrows()):
        n_err = int(row["n_total"] - row["n_correct"])
        ax.text(i, row["error_rate"] + 0.02, f"{n_err}/{int(row['n_total'])}",
                ha="center", fontsize=5, color=style.FG)

    style.trim_spines(ax)
    fig.tight_layout()
    return fig


def plot_confusion_errors(
    error_df: "pd.DataFrame",
    class_names: list,
) -> plt.Figure:
    """Heatmap of error types (true->predicted confusion pairs)."""
    style.apply()

    errors_only = error_df[~error_df["correct"]].copy()
    if len(errors_only) == 0:
        fig, ax = style.figure()
        ax.text(0.5, 0.5, "No errors to display", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    n = len(class_names)
    error_matrix = np.zeros((n, n), dtype=int)
    for _, row in errors_only.iterrows():
        t = row["true_class_idx"]
        p = row["predicted_class_idx"]
        if 0 <= t < n and 0 <= p < n:
            error_matrix[t, p] += 1

    fig, ax = style.figure(width="single", aspect=1.0)
    # Mask the diagonal (not errors)
    np.fill_diagonal(error_matrix, 0)
    im = ax.imshow(error_matrix, cmap="Greys", vmin=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    style.thin_colorbar(cbar)
    style.strip_imshow_frame(ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Error Transitions (Off-Diagonal)")

    # Text contrast threshold against Greys cmap max
    max_err = int(error_matrix.max()) if error_matrix.size else 0
    for i in range(n):
        for j in range(n):
            if i != j and error_matrix[i, j] > 0:
                text_color = "white" if (max_err > 0 and error_matrix[i, j] / max_err > 0.55) else style.FG
                ax.text(j, i, str(error_matrix[i, j]),
                        ha="center", va="center", color=text_color)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Bag variance (prediction stability vs bag size)
# ---------------------------------------------------------------------------

def plot_bag_variance(
    bag_sizes: list,
    per_class_probs: dict,
    class_names: list,
) -> plt.Figure:
    """Violin plot of true-class probability at each bag size, per class.

    Args:
        bag_sizes: sorted list of bag sizes tested
        per_class_probs: {class_idx: {bag_size: [prob_of_true_class, ...]}}
        class_names: list of class names
    """
    style.apply()
    n_classes = len(class_names)
    colors = style.class_colors_muted(class_names)

    fig, axes = style.small_multiples_grid(
        n_classes, width="double", panel_aspect=0.9, max_cols=min(n_classes, 3),
    )
    if n_classes == 1:
        axes = [axes]

    for ci, ax in enumerate(axes):
        probs_by_bs = per_class_probs.get(ci, {})
        data = []
        positions = []
        for bi, bs in enumerate(bag_sizes):
            vals = probs_by_bs.get(bs, [])
            if vals:
                data.append(np.array(vals, dtype=float))
                positions.append(bi)

        if data:
            style.tufte_violin(
                ax, data, positions=positions,
                colors=[colors[ci]] * len(data),
            )

        ax.set_xticks(range(len(bag_sizes)))
        ax.set_xticklabels([str(b) for b in bag_sizes], rotation=45, ha="right")
        ax.set_xlabel("Bag size")
        ax.set_ylabel(f"P({class_names[ci]})")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(class_names[ci])
        ax.axhline(0.5, color=style.MUTE_LIGHT, lw=0.4, ls="--", zorder=0)
        style.trim_spines(ax)

    fig.suptitle("Prediction Stability vs Bag Size", fontsize=7, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-image probability heatmap
# ---------------------------------------------------------------------------

def plot_per_image_probabilities(
    experiment_probs: list,
    class_names: list,
    max_width: int = 2000,
) -> plt.Figure:
    """Stacked horizontal bars showing per-image class predictions.

    Each row is an experiment. Each pixel is an image colored by the predicted
    class probabilities (RGB mapped to class colors). Experiments are grouped
    by true class.

    Args:
        experiment_probs: list of dicts with keys:
            'experiment_id', 'y_true', 'probs' (N_images x num_classes array)
        class_names: list of class names
        max_width: max images to show per row (subsamples if more)
    """
    style.apply()
    import matplotlib.colors as mcolors

    if not experiment_probs:
        fig, ax = style.figure()
        ax.text(0.5, 0.5, "No per-image data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    num_classes = len(class_names)
    colors_hex = style.class_colors_muted(class_names)
    class_rgb = [mcolors.to_rgb(c) for c in colors_hex]

    # Sort by true class, then experiment_id
    experiment_probs.sort(key=lambda x: (x["y_true"], x["experiment_id"]))

    n_exps = len(experiment_probs)
    # Find max images across experiments (capped)
    max_imgs = min(max_width, max(len(e["probs"]) for e in experiment_probs))

    # Build RGB image array
    img = np.ones((n_exps, max_imgs, 3), dtype=float)  # white background
    labels = []

    for row_idx, entry in enumerate(experiment_probs):
        probs = entry["probs"]  # [N, C]
        n = len(probs)

        # Subsample if too many images
        if n > max_imgs:
            indices = np.linspace(0, n - 1, max_imgs, dtype=int)
            probs = probs[indices]
            n = max_imgs

        # Color each pixel: blend class colors by probability
        for col in range(n):
            pixel = np.zeros(3)
            for c in range(min(num_classes, len(class_rgb))):
                pixel += probs[col, c] * np.array(class_rgb[c])
            img[row_idx, col] = np.clip(pixel, 0, 1)

        labels.append(f"{entry['experiment_id']}")

    # Figure sizing
    row_height = 0.06  # inches per experiment
    fig_h = max(3, min(style.MAX_H, n_exps * row_height + 1.0))
    fig_w = style.DOUBLE

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img, aspect="auto", interpolation="nearest")

    # Class boundary lines
    boundaries = []
    prev_class = experiment_probs[0]["y_true"]
    for i, entry in enumerate(experiment_probs):
        if entry["y_true"] != prev_class:
            boundaries.append(i - 0.5)
            prev_class = entry["y_true"]
    for b in boundaries:
        ax.axhline(b, color="white", lw=1.5)

    # Y-axis labels (show every Nth for readability)
    step = max(1, n_exps // 40)
    tick_positions = list(range(0, n_exps, step))
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([labels[i] for i in tick_positions], fontsize=5)

    ax.set_xlabel("Image index")
    ax.set_title("Per-Image Class Predictions", fontsize=7)
    style.strip_imshow_frame(ax)

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=colors_hex[c], label=class_names[c])
                      for c in range(num_classes)]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=5,
              frameon=True, facecolor="white", edgecolor=style.MUTE_LIGHT)

    fig.tight_layout()
    return fig
