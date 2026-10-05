#!/usr/bin/env python
"""Tests for remo_id/analysis/plots.py"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _helpers_analysis import (
    make_binary_data, make_binary_data_perfect,
    make_multiclass_data, make_multiclass_data_perfect,
    make_error_df,
)
from remo_id.reports import figures as plots, style


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_bootstrap_ci_basic():
    from sklearn.metrics import roc_auc_score
    y_true, _, y_prob = make_binary_data(n=100)
    point, lo, hi = plots._bootstrap_ci(y_true, y_prob, roc_auc_score, n=50, seed=0)
    assert isinstance(point, float)
    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert lo <= point <= hi
    print("  \u2713 _bootstrap_ci: 3-tuple floats, lo <= point <= hi")


def test_bootstrap_ci_deterministic():
    from sklearn.metrics import roc_auc_score
    y_true, _, y_prob = make_binary_data(n=100)
    r1 = plots._bootstrap_ci(y_true, y_prob, roc_auc_score, n=50, seed=42)
    r2 = plots._bootstrap_ci(y_true, y_prob, roc_auc_score, n=50, seed=42)
    assert r1 == r2
    print("  \u2713 _bootstrap_ci deterministic with seed")


def test_bootstrap_ci_perfect():
    from sklearn.metrics import roc_auc_score
    y_true, _, y_prob = make_binary_data_perfect(n=100)
    point, lo, hi = plots._bootstrap_ci(y_true, y_prob, roc_auc_score, n=50, seed=0)
    assert point == 1.0
    assert lo == 1.0
    assert hi == 1.0
    print("  \u2713 _bootstrap_ci perfect -> (1.0, 1.0, 1.0)")


def test_ece_perfect():
    y_bin = np.array([0, 0, 1, 1])
    y_sc = np.array([0.0, 0.0, 1.0, 1.0])
    ece = plots._expected_calibration_error(y_bin, y_sc)
    assert ece == 0.0
    print("  \u2713 _expected_calibration_error perfect -> 0.0")


def test_ece_worst():
    y_bin = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_sc = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    ece = plots._expected_calibration_error(y_bin, y_sc)
    assert ece > 0.5
    print("  \u2713 _expected_calibration_error worst > 0.5")


def test_ece_empty():
    ece = plots._expected_calibration_error(np.array([]), np.array([]))
    assert ece == 0.0
    print("  \u2713 _expected_calibration_error empty -> 0.0")


# ---------------------------------------------------------------------------
# Plot functions — each returns Figure, no crash
# ---------------------------------------------------------------------------

def _assert_fig(fig):
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_roc_curves():
    y_true, _, y_probs = make_multiclass_data(n=200, C=3)
    fig = plots.plot_roc_curves(y_true, y_probs, ["A", "B", "C"], bootstrap_n=20)
    _assert_fig(fig)
    print("  \u2713 plot_roc_curves")


def test_plot_pr_curves():
    y_true, _, y_probs = make_multiclass_data(n=200, C=3)
    fig = plots.plot_pr_curves(y_true, y_probs, ["A", "B", "C"], bootstrap_n=20)
    _assert_fig(fig)
    print("  \u2713 plot_pr_curves")


def test_plot_confusion_matrix():
    y_true, y_pred, _ = make_multiclass_data(n=200, C=3)
    fig = plots.plot_confusion_matrix(y_true, y_pred, ["A", "B", "C"])
    _assert_fig(fig)
    print("  \u2713 plot_confusion_matrix")


def test_plot_calibration():
    y_true, _, y_probs = make_multiclass_data(n=200, C=3)
    fig = plots.plot_calibration(y_true, y_probs, ["A", "B", "C"])
    _assert_fig(fig)
    print("  \u2713 plot_calibration")


def test_plot_prob_distributions():
    y_true, _, y_probs = make_multiclass_data(n=200, C=3)
    fig = plots.plot_prob_distributions(y_true, y_probs, ["A", "B", "C"])
    _assert_fig(fig)
    print("  \u2713 plot_prob_distributions")


def test_plot_prob_distributions_1class():
    # 1 class -> single axes (tests axes = [axes] branch)
    n = 100
    y_true = np.zeros(n, dtype=int)
    y_probs = np.random.default_rng(42).random((n, 1))
    fig = plots.plot_prob_distributions(y_true, y_probs, ["only"])
    _assert_fig(fig)
    print("  \u2713 plot_prob_distributions 1 class")


def test_plot_threshold_sensitivity():
    y_true, _, y_probs = make_multiclass_data(n=200, C=3)
    fig = plots.plot_threshold_sensitivity(y_true, y_probs, ["A", "B", "C"])
    _assert_fig(fig)
    print("  \u2713 plot_threshold_sensitivity")


def test_plot_threshold_sensitivity_1class():
    n = 100
    y_true = np.zeros(n, dtype=int)
    y_probs = np.random.default_rng(42).random((n, 1))
    fig = plots.plot_threshold_sensitivity(y_true, y_probs, ["only"])
    _assert_fig(fig)
    print("  \u2713 plot_threshold_sensitivity 1 class")


def test_plot_bag_size_sweep():
    bag_sizes = [50, 100, 200, 400]
    aucs = np.random.default_rng(42).uniform(0.7, 0.95, (3, 4))
    # Make AUCs increase with bag size for plateau detection
    aucs = np.sort(aucs, axis=1)
    fig = plots.plot_bag_size_sweep(bag_sizes, aucs)
    _assert_fig(fig)
    print("  \u2713 plot_bag_size_sweep")


def test_plot_sampling_position():
    fig = plots.plot_sampling_position(
        ["first 50%", "last 50%", "random"],
        np.random.default_rng(42).uniform(0.8, 0.95, (3, 5)),
        bag_size=200,
    )
    _assert_fig(fig)
    print("  \u2713 plot_sampling_position")


def test_plot_seed_variability():
    fig = plots.plot_seed_variability({
        "AUC": [0.90, 0.92, 0.91, 0.89, 0.93],
        "F1": [0.85, 0.87, 0.86, 0.84, 0.88],
    })
    _assert_fig(fig)
    print("  \u2713 plot_seed_variability")


def test_plot_kfold_variability():
    fig = plots.plot_kfold_variability(
        {"AUC": [0.9, 0.91, 0.89, 0.92, 0.88], "F1": [0.85, 0.86, 0.84, 0.87, 0.83]},
        ["A", "B", "C"],
    )
    _assert_fig(fig)
    print("  \u2713 plot_kfold_variability")


def test_plot_operating_points():
    y_true, _, y_probs = make_multiclass_data(n=200, C=3)
    fig = plots.plot_operating_points(y_true, y_probs, ["A", "B", "C"])
    _assert_fig(fig)
    print("  \u2713 plot_operating_points")


def test_plot_top_attention_grid():
    rng = np.random.default_rng(42)
    frames_by_class = []
    for _ in range(3):
        frames = []
        for j in range(10):
            raw = float(rng.random())
            frames.append({
                "image": rng.integers(0, 255, (64, 64, 3), dtype=np.uint8),
                "raw_attn": raw,
                "bag_attn_max": 0.01,
                "bag_attn_min": 0.003,
                "experiment_id": str(100 + j),
            })
        frames_by_class.append(frames)
    fig = plots.plot_top_attention_grid(frames_by_class, ["A", "B", "C"], n_top=4)
    _assert_fig(fig)
    print("  \u2713 plot_top_attention_grid")


def test_plot_umap():
    rng = np.random.default_rng(42)
    embeddings = rng.random((50, 10))
    labels = rng.integers(0, 3, 50)
    try:
        fig = plots.plot_umap(embeddings, labels, ["A", "B", "C"])
        _assert_fig(fig)
        print("  \u2713 plot_umap")
    except ImportError:
        fig = plots.plot_umap(embeddings, labels, ["A", "B", "C"])
        _assert_fig(fig)
        print("  \u2713 plot_umap (fallback, umap-learn not available)")


def test_plot_umap_with_repeats():
    rng = np.random.default_rng(42)
    # 10 experiments, 5 repeats each = 50 points
    embeddings = rng.random((50, 10))
    labels = np.repeat(rng.integers(0, 3, 10), 5)
    experiment_ids = [str(i) for i in np.repeat(np.arange(10), 5)]
    try:
        fig = plots.plot_umap(embeddings, labels, ["A", "B", "C"],
                              experiment_ids=experiment_ids)
        _assert_fig(fig)
        print("  \u2713 plot_umap with repeats")
    except ImportError:
        fig = plots.plot_umap(embeddings, labels, ["A", "B", "C"],
                              experiment_ids=experiment_ids)
        _assert_fig(fig)
        print("  \u2713 plot_umap with repeats (fallback)")


def test_plot_attention_entropy():
    rng = np.random.default_rng(42)
    entropy_by_class = [rng.random(20) for _ in range(3)]
    fig = plots.plot_attention_entropy(entropy_by_class, ["A", "B", "C"])
    _assert_fig(fig)
    print("  \u2713 plot_attention_entropy")


def test_metrics_table_with_ci():
    y_true, _, y_probs = make_multiclass_data(n=200, C=3)
    df = plots.metrics_table_with_ci(y_true, y_probs, ["A", "B", "C"], n=50)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    _assert_fig(plt.figure())  # just cleanup
    print("  \u2713 metrics_table_with_ci returns DataFrame")


def test_plot_baseline_comparison():
    df = pd.DataFrame({
        "model": ["Our model", "LR", "RF", "Majority"],
        "accuracy": [0.95, 0.80, 0.82, 0.50],
        "f1_macro": [0.93, 0.78, 0.80, 0.33],
        "auc_macro": [0.98, 0.85, 0.87, np.nan],
    })
    fig = plots.plot_baseline_comparison(df, metric="auc_macro")
    _assert_fig(fig)
    print("  \u2713 plot_baseline_comparison")


def test_plot_learning_curve():
    df = pd.DataFrame({
        "fraction": [0.2, 0.2, 0.5, 0.5, 1.0, 1.0],
        "seed": [42, 43, 42, 43, 42, 43],
        "best_val_loss": [0.5, 0.52, 0.35, 0.37, 0.25, 0.27],
    })
    fig = plots.plot_learning_curve(df)
    _assert_fig(fig)
    print("  \u2713 plot_learning_curve")


def test_plot_class_distribution():
    imbalance = {"class_counts": {"A": 20, "B": 30, "C": 50}, "imbalance_ratio": 2.5}
    fig = plots.plot_class_distribution(imbalance, ["A", "B", "C"])
    _assert_fig(fig)
    print("  \u2713 plot_class_distribution")


def test_plot_error_confidence():
    y_true, y_pred, y_probs = make_multiclass_data(n=200, C=3)
    fig = plots.plot_error_confidence(y_true, y_pred, y_probs, ["A", "B", "C"])
    _assert_fig(fig)
    print("  \u2713 plot_error_confidence")


def test_plot_error_confidence_all_correct():
    n = 50
    y_true = np.arange(n) % 3
    y_pred = y_true.copy()
    y_probs = np.zeros((n, 3))
    for i in range(n):
        y_probs[i, y_true[i]] = 0.9
    fig = plots.plot_error_confidence(y_true, y_pred, y_probs, ["A", "B", "C"])
    _assert_fig(fig)
    print("  \u2713 plot_error_confidence all correct -> only correct histogram")


def test_plot_error_by_study():
    error_df = make_error_df(n=100, C=3)
    fig = plots.plot_error_by_study(error_df, ["class_0", "class_1", "class_2"])
    _assert_fig(fig)
    print("  \u2713 plot_error_by_study")


def test_plot_error_by_study_all_nan():
    error_df = pd.DataFrame({
        "study_id": [np.nan] * 10,
        "correct": [True] * 10,
    })
    fig = plots.plot_error_by_study(error_df, ["A"])
    _assert_fig(fig)
    print("  \u2713 plot_error_by_study all-NaN -> fallback text")


def test_plot_confusion_errors():
    error_df = make_error_df(n=100, C=3)
    fig = plots.plot_confusion_errors(error_df, ["class_0", "class_1", "class_2"])
    _assert_fig(fig)
    print("  \u2713 plot_confusion_errors")


def test_plot_confusion_errors_all_correct():
    error_df = pd.DataFrame({
        "correct": [True] * 10,
        "true_class_idx": [0] * 10,
        "predicted_class_idx": [0] * 10,
    })
    fig = plots.plot_confusion_errors(error_df, ["A", "B"])
    _assert_fig(fig)
    print("  \u2713 plot_confusion_errors all correct -> 'No errors' text")


def test_plot_experiment_probabilities():
    rng = np.random.default_rng(42)
    n = 20
    fig = plots.plot_experiment_probabilities(
        mean_probs=rng.random(n),
        std_probs=rng.uniform(0, 0.1, n),
        true_labels=rng.integers(0, 2, n),
        splits=["train"] * 15 + ["val"] * 5,
        study_ids=["study_0"] * 10 + ["study_1"] * 10,
    )
    _assert_fig(fig)
    print("  \u2713 plot_experiment_probabilities")


def test_plot_roc_binary_splits():
    rng = np.random.default_rng(42)
    n = 50
    y_true = rng.integers(0, 2, n)
    y_prob = rng.random(n)
    fig = plots.plot_roc_binary_splits(y_true, y_prob, y_true[:10], y_prob[:10])
    _assert_fig(fig)
    print("  \u2713 plot_roc_binary_splits")


def test_plot_roc_binary_splits_single_class():
    # All same label -> handles unique check
    y_true = np.ones(20)
    y_prob = np.random.default_rng(42).random(20)
    fig = plots.plot_roc_binary_splits(y_true, y_prob, y_true[:5], y_prob[:5])
    _assert_fig(fig)
    print("  \u2713 plot_roc_binary_splits single-class y_true")


def test_plot_repeatability_hist():
    stds = np.random.default_rng(42).uniform(0, 0.1, 30)
    fig = plots.plot_repeatability_hist(stds)
    _assert_fig(fig)
    print("  \u2713 plot_repeatability_hist")


def test_plot_optimal_thresholds():
    from remo_id.analysis.optimal_thresholds import find_optimal_thresholds
    y_true, _, y_probs = make_multiclass_data(n=200, C=3)
    names = ["A", "B", "C"]
    targets = {n: 0.90 for n in names}
    results = find_optimal_thresholds(y_true, y_probs, names, targets)
    fig = plots.plot_optimal_thresholds(y_true, y_probs, names, results, target_class=names[0])
    _assert_fig(fig)
    print("  \u2713 plot_optimal_thresholds")


def test_plot_optimal_thresholds_all_classes():
    from remo_id.analysis.optimal_thresholds import find_optimal_thresholds
    y_true, _, y_probs = make_multiclass_data(n=200, C=4)
    names = ["A", "B", "C", "D"]
    targets = {n: 0.90 for n in names}
    results = find_optimal_thresholds(y_true, y_probs, names, targets)
    fig = plots.plot_optimal_thresholds(y_true, y_probs, names, results, target_class=names[0])
    axes = fig.get_axes()
    assert len(axes) == 1  # one ROC-space panel, one curve per class
    assert len(axes[0].get_lines()) >= 4
    _assert_fig(fig)
    print("  \u2713 plot_optimal_thresholds draws all 4 classes")


def test_plot_optimal_thresholds_with_warning():
    from remo_id.analysis.optimal_thresholds import find_optimal_thresholds
    y_true, _, y_probs = make_multiclass_data(n=200, C=2)
    names = ["A", "B"]
    targets = {n: 1.0 for n in names}  # may trigger warning
    results = find_optimal_thresholds(y_true, y_probs, names, targets)
    fig = plots.plot_optimal_thresholds(y_true, y_probs, names, results, target_class=names[0])
    _assert_fig(fig)
    print("  \u2713 plot_optimal_thresholds with warning")


# ---------------------------------------------------------------------------
# Visual gallery — generates all figures from DummyDataset and saves PNGs
# ---------------------------------------------------------------------------

def test_generate_figure_gallery(tmp_path):
    """Generate all figures using DummyDataset and save to disk for visual review.

    Run with: pytest tests/test_plots.py::test_generate_figure_gallery -s
    PNGs are saved under pytest's tmp_path (see also examples/generate_all_figures.py).
    """
    from dummy_dataset import DummyDataset
    from remo_id.analysis.optimal_thresholds import find_optimal_thresholds

    ds = DummyDataset()
    out = str(tmp_path / "figure_gallery")
    os.makedirs(out, exist_ok=True)

    def save(name, fig):
        fig.savefig(os.path.join(out, f"{name}.png"))
        plt.close(fig)
        print(f"  saved {name}.png")

    # Classification (multiclass)
    save("roc_curves", plots.plot_roc_curves(ds.y_true, ds.y_probs, ds.CLASS_NAMES, bootstrap_n=50))
    save("pr_curves", plots.plot_pr_curves(ds.y_true, ds.y_probs, ds.CLASS_NAMES, bootstrap_n=50))
    save("confusion_matrix", plots.plot_confusion_matrix(ds.y_true, ds.y_pred, ds.CLASS_NAMES))
    save("calibration", plots.plot_calibration(ds.y_true, ds.y_probs, ds.CLASS_NAMES))
    save("prob_distributions", plots.plot_prob_distributions(ds.y_true, ds.y_probs, ds.CLASS_NAMES))
    # per_class_f1 removed

    # Comparison
    save("threshold_sensitivity", plots.plot_threshold_sensitivity(ds.y_true, ds.y_probs, ds.CLASS_NAMES))
    save("seed_variability", plots.plot_seed_variability(ds.metrics_by_seed))
    save("kfold_variability", plots.plot_kfold_variability(ds.fold_metrics, ds.CLASS_NAMES))
    save("operating_points", plots.plot_operating_points(ds.y_true, ds.y_probs, ds.CLASS_NAMES))
    save("baseline_comparison", plots.plot_baseline_comparison(ds.baseline_df, metric="auc_macro"))
    save("experiment_probabilities", plots.plot_experiment_probabilities(
        ds.mean_probs, ds.std_probs, ds.y_true_binary, ds.splits[:len(ds.y_true_binary)],
        ds.study_ids[:len(ds.y_true_binary)],
    ))
    save("repeatability_hist", plots.plot_repeatability_hist(ds.std_probs))

    # Optimal thresholds
    targets = {n: 0.90 for n in ds.CLASS_NAMES}
    opt_results = find_optimal_thresholds(ds.y_true, ds.y_probs, ds.CLASS_NAMES, targets)
    save("optimal_thresholds", plots.plot_optimal_thresholds(ds.y_true, ds.y_probs, ds.CLASS_NAMES, opt_results, target_class=ds.CLASS_NAMES[0]))

    # Data
    save("bag_size_sweep", plots.plot_bag_size_sweep(ds.bag_sizes, ds.aucs_by_seed))
    save("sampling_position", plots.plot_sampling_position(ds.positions, ds.position_aucs, bag_size=200))
    save("learning_curve", plots.plot_learning_curve(ds.learning_curve_df))
    save("class_distribution", plots.plot_class_distribution(ds.imbalance, ds.CLASS_NAMES))
    save("error_confidence", plots.plot_error_confidence(ds.y_true, ds.y_pred, ds.y_probs, ds.CLASS_NAMES))
    save("error_by_study", plots.plot_error_by_study(ds.error_df, ds.CLASS_NAMES))
    save("confusion_errors", plots.plot_confusion_errors(ds.error_df, ds.CLASS_NAMES))

    # Interpretation
    save("top_attention_grid", plots.plot_top_attention_grid(ds.frames_by_class, ds.CLASS_NAMES))
    save("umap", plots.plot_umap(ds.embeddings, ds.embedding_labels, ds.CLASS_NAMES))
    save("attention_entropy", plots.plot_attention_entropy(ds.entropy_by_class, ds.CLASS_NAMES))

    # UMAP panels (batch-effect style)
    true_names = np.array([ds.CLASS_NAMES[i] for i in ds.y_true])
    pred_names = np.array([ds.CLASS_NAMES[i] for i in ds.y_pred])
    scope_names = np.array([f"scope_{i % 3}" for i in range(ds.n_experiments)])
    save("umap_panels", plots.plot_umap_panels(ds.embeddings, [
        {"labels": true_names, "title": "True class", "names": ds.CLASS_NAMES},
        {"labels": pred_names, "title": "Predicted class", "names": ds.CLASS_NAMES},
        {"labels": scope_names, "title": "scope"},
    ]))

    print(f"\nAll figures saved to {out}")


if __name__ == "__main__":
    print("test_plots.py")
    # Helpers
    test_bootstrap_ci_basic()
    test_bootstrap_ci_deterministic()
    test_bootstrap_ci_perfect()
    test_ece_perfect()
    test_ece_worst()
    test_ece_empty()
    # Plot functions
    test_plot_roc_curves()
    test_plot_pr_curves()
    test_plot_confusion_matrix()
    test_plot_calibration()
    test_plot_prob_distributions()
    test_plot_prob_distributions_1class()
    test_plot_threshold_sensitivity()
    test_plot_threshold_sensitivity_1class()
    test_plot_bag_size_sweep()
    test_plot_sampling_position()
    test_plot_seed_variability()
    test_plot_kfold_variability()
    test_plot_operating_points()
    test_plot_top_attention_grid()
    test_plot_umap()
    test_plot_umap_with_repeats()
    test_plot_attention_entropy()
    test_metrics_table_with_ci()
    test_plot_baseline_comparison()
    test_plot_learning_curve()
    test_plot_class_distribution()
    test_plot_error_confidence()
    test_plot_error_confidence_all_correct()
    test_plot_error_by_study()
    test_plot_error_by_study_all_nan()
    test_plot_confusion_errors()
    test_plot_confusion_errors_all_correct()
    test_plot_experiment_probabilities()
    test_plot_roc_binary_splits()
    test_plot_roc_binary_splits_single_class()
    test_plot_repeatability_hist()
    test_plot_optimal_thresholds()
    test_plot_optimal_thresholds_all_classes()
    test_plot_optimal_thresholds_with_warning()
    print("All tests passed.")
