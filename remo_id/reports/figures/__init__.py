"""Figure generation for remo-id reports."""

from remo_id.reports.figures._bootstrap import _bootstrap_ci

from remo_id.reports.figures.classification import (
    _expected_calibration_error,
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrix,
    plot_calibration,
    plot_prob_distributions,
    metrics_table_with_ci,
)

from remo_id.reports.figures.comparison import (
    plot_seed_variability,
    plot_threshold_sensitivity,
    plot_operating_points,
    plot_kfold_variability,
    plot_baseline_comparison,
    plot_experiment_probabilities,
    plot_roc_binary_splits,
    plot_repeatability_hist,
    plot_optimal_thresholds,
    plot_dual_operating_points,
    plot_per_fold_thresholds,
)

from remo_id.reports.figures.data import (
    plot_bag_size_sweep,
    plot_bag_variance,
    plot_per_image_probabilities,
    plot_sampling_position,
    plot_learning_curve,
    plot_class_distribution,
    plot_error_confidence,
    plot_error_by_study,
    plot_confusion_errors,
)

from remo_id.reports.figures.interpretation import (
    plot_top_attention_grid,
    plot_umap,
    plot_umap_panels,
    plot_attention_entropy,
)

# Private helpers still importable via direct module paths (e.g. from .classification import _youden_f1)
# but not re-exported at package level.

__all__ = [
    # Classification
    "plot_roc_curves", "plot_pr_curves", "plot_confusion_matrix",
    "plot_calibration", "plot_prob_distributions",
    "metrics_table_with_ci",
    # Comparison
    "plot_seed_variability", "plot_threshold_sensitivity", "plot_operating_points",
    "plot_kfold_variability", "plot_baseline_comparison",
    "plot_experiment_probabilities", "plot_roc_binary_splits", "plot_repeatability_hist",
    "plot_optimal_thresholds", "plot_dual_operating_points", "plot_per_fold_thresholds",
    # Data
    "plot_bag_size_sweep", "plot_bag_variance", "plot_per_image_probabilities",
    "plot_sampling_position",
    "plot_learning_curve", "plot_class_distribution",
    "plot_error_confidence", "plot_error_by_study", "plot_confusion_errors",
    # Interpretation
    "plot_top_attention_grid", "plot_umap", "plot_umap_panels", "plot_attention_entropy",
]
