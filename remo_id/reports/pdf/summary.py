"""Shared summary utilities for inference reports."""

import io
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from remo_id.reports.pdf.primitives import fig_to_png_bytes, pdf_title


def make_probability_strip_plot(
    experiment_ids: List[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    has_null_labels: np.ndarray,
    pred_threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> bytes:
    """
    Create strip plot: x=experiment (sorted by prob), y=probability.

    Args:
        experiment_ids: List of experiment IDs
        y_true: Ground truth labels (0 or 1 for binary)
        y_prob: Predicted probabilities
        has_null_labels: Boolean array indicating null/unknown labels
        pred_threshold: Prediction threshold for classification

    Returns:
        PNG bytes of the strip plot

    Design:
        - Positive experiments: orange filled markers
        - Negative experiments: blue filled markers
        - Null/unknown labels: gray hollow markers
        - Horizontal threshold line
        - Background shading for correct/incorrect regions
    """
    n = len(experiment_ids)
    if n == 0:
        fig = plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, "No experiments to display", ha="center", va="center")
        plt.axis("off")
        return fig_to_png_bytes(fig)

    # Sort by probability
    sort_idx = np.argsort(y_prob)
    sorted_ids = [experiment_ids[i] for i in sort_idx]
    sorted_prob = y_prob[sort_idx]
    sorted_true = y_true[sort_idx]
    sorted_null = has_null_labels[sort_idx]

    fig, ax = plt.subplots(figsize=(max(8, min(n * 0.3, 14)), 5))

    # Background shading for correct/incorrect predictions
    for i in range(n):
        if sorted_null[i]:
            # Gray background for null labels (unknown correctness)
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.1, color="gray", zorder=0)
        else:
            pred = 1 if sorted_prob[i] >= pred_threshold else 0
            correct = (pred == sorted_true[i])
            color = "#d4edda" if correct else "#f8d7da"  # green/red tints
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.4, color=color, zorder=0)

    # Plot markers
    for i in range(n):
        if sorted_null[i]:
            # Null labels: gray hollow circles
            ax.scatter(i, sorted_prob[i], s=80, facecolors="none",
                      edgecolors="gray", linewidths=2, zorder=3)
        elif sorted_true[i] == 1:
            # Positive: orange filled
            ax.scatter(i, sorted_prob[i], s=80, c="orange", edgecolors="darkorange",
                      linewidths=1, zorder=3)
        else:
            # Negative: blue filled
            ax.scatter(i, sorted_prob[i], s=80, c="royalblue", edgecolors="navy",
                      linewidths=1, zorder=3)

    # Threshold line
    ax.axhline(y=pred_threshold, color="red", linestyle="--", linewidth=2,
               label=f"Threshold={pred_threshold:.2f}", zorder=2)

    # Labels and formatting
    ax.set_xlabel("Experiments (sorted by probability)", fontsize=11)
    ax.set_ylabel("Predicted Probability", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, n - 0.5)

    # Show experiment IDs on x-axis if small enough
    if n <= 20:
        ax.set_xticks(range(n))
        ax.set_xticklabels(sorted_ids, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xticks([])

    # Legend
    pos_label = f'{class_names[1]} (true)' if class_names and len(class_names) > 1 else 'Positive (true)'
    neg_label = f'{class_names[0]} (true)' if class_names else 'Negative (true)'
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markeredgecolor='darkorange', markersize=10, label=pos_label),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue',
               markeredgecolor='navy', markersize=10, label=neg_label),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='gray', markeredgewidth=2, markersize=10, label='Unknown label'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=f'Threshold ({pred_threshold:.2f})'),
        Patch(facecolor='#d4edda', alpha=0.6, label='Correct prediction'),
        Patch(facecolor='#f8d7da', alpha=0.6, label='Incorrect prediction'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              bbox_to_anchor=(1.02, 1), borderaxespad=0)

    ax.set_title("Experiment Predictions by Probability", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig_to_png_bytes(fig)


def make_confusion_matrix_heatmap(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
) -> bytes:
    """Create annotated confusion matrix heatmap.

    Args:
        confusion_matrix: NxN confusion matrix array
        class_names: List of class names for axis labels
        title: Title for the figure

    Returns:
        PNG bytes of the heatmap
    """
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(5, n * 1.2), max(4, n * 1.0)))

    im = ax.imshow(confusion_matrix, cmap="Blues", aspect="equal")

    # Annotate cells with counts
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            val = int(confusion_matrix[i, j])
            # Use white text on dark cells
            threshold = confusion_matrix.max() / 2.0
            color = "white" if val > threshold else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig_to_png_bytes(fig)


def make_multiclass_strip_plot(
    experiment_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_conf: np.ndarray,
    class_names: List[str],
    has_null_labels: np.ndarray,
) -> bytes:
    """Create strip plot for multiclass predictions.

    X: experiments sorted by confidence
    Y: max predicted probability
    Marker color: true class (categorical colormap)
    Background: green=correct, red=incorrect, gray=null

    Args:
        experiment_ids: List of experiment IDs
        y_true: Ground truth class indices
        y_pred: Predicted class indices
        y_conf: Max predicted probability per experiment
        class_names: List of class names
        has_null_labels: Boolean array for null labels

    Returns:
        PNG bytes of the strip plot
    """
    n = len(experiment_ids)
    if n == 0:
        fig = plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, "No experiments to display", ha="center", va="center")
        plt.axis("off")
        return fig_to_png_bytes(fig)

    # Sort by confidence
    sort_idx = np.argsort(y_conf)
    sorted_ids = [experiment_ids[i] for i in sort_idx]
    sorted_conf = y_conf[sort_idx]
    sorted_true = y_true[sort_idx]
    sorted_pred = y_pred[sort_idx]
    sorted_null = has_null_labels[sort_idx]

    num_classes = len(class_names)
    cmap = plt.cm.get_cmap("tab10", max(num_classes, 1))

    fig, ax = plt.subplots(figsize=(max(8, min(n * 0.3, 14)), 5))

    # Background shading for correct/incorrect
    for i in range(n):
        if sorted_null[i]:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.1, color="gray", zorder=0)
        else:
            correct = (sorted_pred[i] == sorted_true[i])
            color = "#d4edda" if correct else "#f8d7da"
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.4, color=color, zorder=0)

    # Plot markers colored by true class
    for i in range(n):
        if sorted_null[i]:
            ax.scatter(i, sorted_conf[i], s=80, facecolors="none",
                      edgecolors="gray", linewidths=2, zorder=3)
        else:
            cls = int(sorted_true[i])
            ax.scatter(i, sorted_conf[i], s=80, c=[cmap(cls)],
                      edgecolors="black", linewidths=0.5, zorder=3)

    ax.set_xlabel("Experiments (sorted by confidence)", fontsize=11)
    ax.set_ylabel("Max Predicted Probability", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, n - 0.5)

    if n <= 20:
        ax.set_xticks(range(n))
        ax.set_xticklabels(sorted_ids, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xticks([])

    # Legend: one entry per class + null + correct/incorrect
    legend_elements = []
    for ci, name in enumerate(class_names):
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(ci),
                   markeredgecolor='black', markersize=10, label=name)
        )
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='gray', markeredgewidth=2, markersize=10, label='Unknown label')
    )
    legend_elements.append(Patch(facecolor='#d4edda', alpha=0.6, label='Correct prediction'))
    legend_elements.append(Patch(facecolor='#f8d7da', alpha=0.6, label='Incorrect prediction'))

    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.set_title("Experiment Predictions by Confidence", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig_to_png_bytes(fig)


def make_per_class_metrics_table(
    f1_per_class: List[float],
    class_names: List[str],
) -> bytes:
    """Create a table figure showing per-class F1 scores.

    Args:
        f1_per_class: List of F1 scores per class
        class_names: List of class names

    Returns:
        PNG bytes of the table
    """
    # Ensure f1_per_class covers all classes (pad with 0.0 if needed)
    n = len(class_names)
    if len(f1_per_class) < n:
        f1_per_class = list(f1_per_class) + [0.0] * (n - len(f1_per_class))
    fig, ax = plt.subplots(figsize=(6, max(2, n * 0.5 + 1.5)))
    ax.axis("off")

    # Build table data
    cell_text = [[name, f"{f1:.4f}"] for name, f1 in zip(class_names, f1_per_class)]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Class", "F1 Score"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # Color header row
    for j in range(2):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Color F1 cells by value
    for i in range(n):
        f1 = f1_per_class[i]
        if f1 >= 0.8:
            table[(i + 1, 1)].set_facecolor("#d4edda")
        elif f1 >= 0.6:
            table[(i + 1, 1)].set_facecolor("#fff3cd")
        else:
            table[(i + 1, 1)].set_facecolor("#f8d7da")

    ax.set_title("Per-Class F1 Scores", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig_to_png_bytes(fig)


def make_metrics_box(
    metrics: Dict[str, float],
    task_type: str,
    n_experiments: int,
    n_null: int,
) -> bytes:
    """
    Create a compact metrics summary as a styled figure.

    Args:
        metrics: Dictionary of metric names to values
        task_type: "binary" or "multiclass"
        n_experiments: Total number of experiments
        n_null: Number of experiments with null labels

    Returns:
        PNG bytes of the metrics box
    """
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.axis("off")

    # Title
    ax.text(0.5, 0.95, "Performance Metrics", ha="center", va="top",
            fontsize=14, fontweight="bold", transform=ax.transAxes)

    # Experiment counts
    ax.text(0.5, 0.85, f"Experiments: {n_experiments} ({n_null} null)",
            ha="center", va="top", fontsize=10, color="gray", transform=ax.transAxes)

    # Metrics based on task type
    y_pos = 0.72
    line_height = 0.12

    if task_type == "binary":
        metric_items = [
            ("Accuracy", metrics.get("accuracy", 0)),
            ("Precision", metrics.get("precision", 0)),
            ("Recall", metrics.get("recall", 0)),
            ("F1 Score", metrics.get("f1", 0)),
        ]
    else:  # multiclass
        metric_items = [
            ("Accuracy", metrics.get("accuracy", 0)),
            ("F1 (macro)", metrics.get("f1_macro", 0)),
            ("F1 (weighted)", metrics.get("f1_weighted", 0)),
        ]

    for name, value in metric_items:
        # Color based on value (higher is better)
        if value >= 0.8:
            color = "green"
        elif value >= 0.6:
            color = "orange"
        else:
            color = "red"

        ax.text(0.15, y_pos, f"{name}:", ha="left", va="center",
                fontsize=11, transform=ax.transAxes)
        ax.text(0.85, y_pos, f"{value:.4f}", ha="right", va="center",
                fontsize=11, fontweight="bold", color=color, transform=ax.transAxes)
        y_pos -= line_height

    # Add border
    rect = plt.Rectangle((0.02, 0.08), 0.96, 0.9, fill=False,
                         edgecolor="gray", linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect)

    plt.tight_layout()
    return fig_to_png_bytes(fig)


def build_summary_first_page(
    c: canvas.Canvas,
    title: str,
    subtitle: str,
    metrics: Dict[str, float],
    experiment_ids: List[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    has_null_labels: np.ndarray,
    task_type: str,
    n_experiments: int,
    n_null: int,
    pred_threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
    y_pred: Optional[np.ndarray] = None,
):
    """
    Build a unified first page with metrics and probability strip plot.

    Layout:
    +-------------------------------------+
    | Title + Subtitle                    |
    +-------------+-----------------------+
    | Metrics Box | Probability Strip Plot|
    | (Precision, | (experiments vs prob) |
    |  Recall,F1) |                       |
    +-------------+-----------------------+

    Args:
        c: ReportLab canvas
        title: Page title
        subtitle: Page subtitle
        metrics: Dictionary of metrics
        experiment_ids: List of experiment IDs
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        has_null_labels: Boolean array for null labels
        task_type: "binary" or "multiclass"
        n_experiments: Total experiment count
        n_null: Null label count
        pred_threshold: Threshold for binary classification
    """
    w, h = letter

    # Draw title
    pdf_title(c, title, subtitle)

    # Create metrics box PNG
    metrics_png = make_metrics_box(metrics, task_type, n_experiments, n_null)

    # Create strip plot PNG
    if task_type == "multiclass" and class_names is not None and y_pred is not None:
        strip_png = make_multiclass_strip_plot(
            experiment_ids, y_true, y_pred, y_prob, class_names, has_null_labels
        )
    else:
        strip_png = make_probability_strip_plot(
            experiment_ids, y_true, y_prob, has_null_labels, pred_threshold,
            class_names=class_names,
        )

    # Position metrics box (left side)
    metrics_img = ImageReader(io.BytesIO(metrics_png))
    c.drawImage(
        metrics_img,
        0.5 * inch,
        h - 5.5 * inch,
        width=2.8 * inch,
        height=2.8 * inch,
        preserveAspectRatio=True,
        mask="auto",
    )

    # Position strip plot (right side, larger)
    strip_img = ImageReader(io.BytesIO(strip_png))
    c.drawImage(
        strip_img,
        3.5 * inch,
        h - 6.0 * inch,
        width=5.0 * inch,
        height=3.5 * inch,
        preserveAspectRatio=True,
        mask="auto",
    )

    # Add confusion matrix summary for binary classification
    if task_type == "binary" and all(k in metrics for k in ["tp", "tn", "fp", "fn"]):
        y = h - 6.5 * inch
        c.setFont("Helvetica-Bold", 10)
        c.drawString(0.5 * inch, y, "Confusion Matrix:")
        y -= 14

        c.setFont("Helvetica", 9)
        tp, tn = int(metrics.get("tp", 0)), int(metrics.get("tn", 0))
        fp, fn = int(metrics.get("fp", 0)), int(metrics.get("fn", 0))
        c.drawString(0.6 * inch, y, f"TP: {tp}  |  TN: {tn}")
        y -= 12
        c.drawString(0.6 * inch, y, f"FP: {fp}  |  FN: {fn}")

    c.showPage()


def make_null_predictions_table(
    experiment_ids: List[str],
    y_prob: np.ndarray,
    pred_threshold: float = 0.5,
    task_type: str = "binary",
    class_names: Optional[List[str]] = None,
    y_pred: Optional[np.ndarray] = None,
) -> bytes:
    """
    Create a table visualization for null-label predictions.

    Args:
        experiment_ids: Experiment IDs with null labels
        y_prob: Predicted probabilities/confidences for these experiments
        pred_threshold: Threshold for binary classification
        task_type: Task type for label formatting
        class_names: Class names for multiclass display
        y_pred: Predicted class indices for multiclass

    Returns:
        PNG bytes of the table
    """
    n = len(experiment_ids)
    if n == 0:
        fig = plt.figure(figsize=(8, 2))
        plt.text(0.5, 0.5, "No null-label experiments", ha="center", va="center")
        plt.axis("off")
        return fig_to_png_bytes(fig)

    # Sort by probability/confidence (descending)
    sort_idx = np.argsort(-y_prob)

    fig, ax = plt.subplots(figsize=(8, max(3, min(n * 0.4 + 1, 10))))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Title
    ax.text(0.5, 0.98, "Predictions for Unknown Labels", ha="center", va="top",
            fontsize=14, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.93, "(Ground truth unavailable)", ha="center", va="top",
            fontsize=10, color="gray", transform=ax.transAxes)

    # Table header
    y_pos = 0.85
    ax.text(0.1, y_pos, "Experiment ID", ha="left", fontsize=11, fontweight="bold")
    ax.text(0.5, y_pos, "Prediction", ha="center", fontsize=11, fontweight="bold")
    ax.text(0.8, y_pos, "Confidence", ha="center", fontsize=11, fontweight="bold")

    # Separator line
    y_pos -= 0.03
    ax.axhline(y=y_pos, xmin=0.05, xmax=0.95, color="gray", linewidth=0.5)

    # Table rows
    row_height = min(0.08, 0.7 / max(n, 1))
    y_pos -= 0.05

    # Colormap for multiclass
    if task_type == "multiclass" and class_names:
        cmap = plt.cm.get_cmap("tab10", max(len(class_names), 1))

    for i in sort_idx[:min(n, 15)]:  # Limit to 15 rows
        exp_id = experiment_ids[i]
        prob = y_prob[i]

        if task_type == "multiclass" and class_names and y_pred is not None:
            pred_cls = int(y_pred[i])
            pred_label = class_names[pred_cls] if pred_cls < len(class_names) else f"Class {pred_cls}"
            pred_color = cmap(pred_cls) if pred_cls < 10 else "black"
        else:
            pred_label = "POSITIVE" if prob >= pred_threshold else "NEGATIVE"
            pred_color = "darkorange" if prob >= pred_threshold else "royalblue"

        ax.text(0.1, y_pos, str(exp_id)[:20], ha="left", fontsize=10)
        ax.text(0.5, y_pos, pred_label, ha="center", fontsize=10,
                fontweight="bold", color=pred_color)
        ax.text(0.8, y_pos, f"{prob:.3f}", ha="center", fontsize=10)

        y_pos -= row_height

    if n > 15:
        ax.text(0.5, y_pos, f"... and {n - 15} more", ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    return fig_to_png_bytes(fig)
