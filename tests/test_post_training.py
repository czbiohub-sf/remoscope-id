#!/usr/bin/env python
"""Tests for remo_id/analysis/post_training.py (_compute_all_metrics and _compute_all_metrics_multiclass only)"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from _helpers_analysis import make_post_training_results
from remo_id.analysis.post_training import _compute_all_metrics, _compute_all_metrics_multiclass


def test_binary_keys():
    results = make_post_training_results(n=50)
    m = _compute_all_metrics(results, n_runs=5)
    assert "val" in m and "total" in m and "repeatability" in m
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        assert key in m["val"], f"missing val key: {key}"
        assert key in m["total"], f"missing total key: {key}"
    assert "mean_std" in m["repeatability"]
    assert "stds" in m["repeatability"]
    print("  \u2713 binary keys: val/total/repeatability present")


def test_binary_perfect():
    results = []
    for i in range(20):
        label = i % 2
        results.append({
            "split": "val" if i < 4 else "train",
            "true_label": label,
            "pred_label": label,
            "mean_prob": 0.9 if label == 1 else 0.1,
            "std_prob": 0.01,
            "run_probs": [0.9 if label == 1 else 0.1] * 5,
            "study_id": "s",
            "experiment_id": i,
        })
    m = _compute_all_metrics(results, n_runs=5)
    assert m["total"]["accuracy"] == 1.0
    assert m["total"]["auc"] == 1.0
    print("  \u2713 binary perfect -> accuracy=1.0, auc=1.0")


def test_binary_val_subset():
    results = make_post_training_results(n=50)
    m = _compute_all_metrics(results, n_runs=5)
    # Val results are the first 10 (n//5=10)
    val_count = sum(1 for r in results if r["split"] == "val")
    assert val_count > 0
    # Val metrics should differ from total (different subset)
    print("  \u2713 binary val subset: val metrics computed from val split only")


def test_binary_repeatability():
    results = make_post_training_results(n=50)
    m = _compute_all_metrics(results, n_runs=5)
    stds = [r["std_prob"] for r in results]
    expected_mean = np.mean(stds)
    assert abs(m["repeatability"]["mean_std"] - expected_mean) < 1e-6
    print("  \u2713 binary repeatability mean_std matches np.mean(stds)")


def test_binary_single_class_val():
    # All val experiments have same true_label -> auc=0.5 fallback
    results = []
    for i in range(20):
        label = 1  # all positive
        results.append({
            "split": "val" if i < 5 else "train",
            "true_label": label,
            "pred_label": 1,
            "mean_prob": 0.8,
            "std_prob": 0.02,
            "run_probs": [0.8] * 5,
            "study_id": "s",
            "experiment_id": i,
        })
    m = _compute_all_metrics(results, n_runs=5)
    assert m["val"]["auc"] == 0.5
    print("  \u2713 binary single-class val -> auc=0.5 fallback")


def test_multiclass_keys():
    results = []
    for i in range(30):
        label = i % 3
        results.append({
            "split": "val" if i < 6 else "train",
            "true_label": label,
            "pred_label": label,
            "mean_prob": 0.8,
            "std_prob": 0.02,
        })
    m = _compute_all_metrics_multiclass(results, num_classes=3)
    assert "val" in m and "total" in m and "repeatability" in m
    for key in ["accuracy", "f1_macro", "f1_weighted"]:
        assert key in m["val"], f"missing val key: {key}"
        assert key in m["total"], f"missing total key: {key}"
    print("  \u2713 multiclass keys: val/total have accuracy/f1_macro/f1_weighted")


def test_multiclass_perfect():
    results = []
    for i in range(30):
        label = i % 3
        results.append({
            "split": "val" if i < 6 else "train",
            "true_label": label,
            "pred_label": label,
            "mean_prob": 0.9,
            "std_prob": 0.01,
        })
    m = _compute_all_metrics_multiclass(results, num_classes=3)
    assert m["total"]["accuracy"] == 1.0
    assert m["total"]["f1_macro"] == 1.0
    print("  \u2713 multiclass perfect -> accuracy=1.0, f1_macro=1.0")


def test_multiclass_repeatability():
    results = []
    for i in range(20):
        results.append({
            "split": "val" if i < 4 else "train",
            "true_label": i % 3,
            "pred_label": i % 3,
            "mean_prob": 0.8,
            "std_prob": 0.05 * (i % 3 + 1),
        })
    m = _compute_all_metrics_multiclass(results, num_classes=3)
    stds = [r["std_prob"] for r in results]
    assert abs(m["repeatability"]["mean_std"] - np.mean(stds)) < 1e-6
    print("  \u2713 multiclass repeatability mean_std correct")


if __name__ == "__main__":
    print("test_post_training.py")
    test_binary_keys()
    test_binary_perfect()
    test_binary_val_subset()
    test_binary_repeatability()
    test_binary_single_class_val()
    test_multiclass_keys()
    test_multiclass_perfect()
    test_multiclass_repeatability()
    print("All tests passed.")
