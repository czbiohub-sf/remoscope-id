#!/usr/bin/env python
"""Tests for remo_id/analysis/metrics.py"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from _helpers_analysis import (
    make_binary_data, make_binary_data_perfect,
    make_multiclass_data, make_multiclass_data_perfect,
)
from remo_id.analysis.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics, binconf, precision_recall_f1,
)


def test_binary_hand_computed():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    m = compute_binary_metrics(y_true, y_pred)
    assert m["tp"] == 1
    assert m["tn"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 1
    assert m["accuracy"] == 0.75
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5
    assert abs(m["f1"] - 2 / 3) < 1e-6
    print("  \u2713 binary hand-computed")


def test_binary_perfect():
    y_true, y_pred, y_prob = make_binary_data_perfect()
    m = compute_binary_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    print("  \u2713 binary perfect")


def test_binary_all_wrong():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1])
    m = compute_binary_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.0
    print("  \u2713 binary all wrong")


def test_binary_with_prob():
    y_true, y_pred, y_prob = make_binary_data()
    m = compute_binary_metrics(y_true, y_pred, y_prob=y_prob)
    assert "brier_score" in m
    print("  \u2713 binary with prob has brier_score")


def test_binary_perfect_brier():
    y_true, y_pred, y_prob = make_binary_data_perfect()
    m = compute_binary_metrics(y_true, y_pred, y_prob=y_prob)
    assert m["brier_score"] == 0.0
    print("  \u2713 binary perfect brier = 0")


def test_binary_without_prob():
    y_true, y_pred, _ = make_binary_data()
    m = compute_binary_metrics(y_true, y_pred)
    assert "brier_score" not in m
    print("  \u2713 binary without prob: no brier_score")


def test_binary_all_positive():
    y_true = np.ones(10, dtype=int)
    y_pred = np.ones(10, dtype=int)
    m = compute_binary_metrics(y_true, y_pred)
    assert m["recall"] == 1.0
    assert m["accuracy"] == 1.0
    print("  \u2713 binary all-positive")


def test_binary_all_negative():
    y_true = np.zeros(10, dtype=int)
    y_pred = np.zeros(10, dtype=int)
    m = compute_binary_metrics(y_true, y_pred)
    assert m["specificity"] == 1.0
    assert m["accuracy"] == 1.0
    print("  \u2713 binary all-negative")


def test_binary_small_n():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    m = compute_binary_metrics(y_true, y_pred)
    assert "accuracy" in m and "f1" in m and "tp" in m
    print("  \u2713 binary n=2 no crash")


# --- Multiclass ---

def test_multiclass_basic():
    y_true, y_pred, y_probs = make_multiclass_data(n=200, C=4)
    m = compute_multiclass_metrics(y_true, y_pred, num_classes=4)
    assert m["confusion_matrix"].shape == (4, 4)
    assert len(m["f1_per_class"]) == 4
    print("  \u2713 multiclass basic shapes")


def test_multiclass_perfect():
    y_true, y_pred, y_probs = make_multiclass_data_perfect(n=200, C=4)
    m = compute_multiclass_metrics(y_true, y_pred, num_classes=4)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert all(s == 1.0 for s in m["sensitivity_per_class"])
    print("  \u2713 multiclass perfect")


def test_multiclass_with_probs():
    y_true, y_pred, y_probs = make_multiclass_data(n=200, C=4)
    m = compute_multiclass_metrics(y_true, y_pred, num_classes=4, y_probs=y_probs)
    assert "brier_score" in m
    assert m["brier_score"] >= 0
    print("  \u2713 multiclass with probs -> brier present")


def test_multiclass_2class_branch():
    y_true, y_pred, y_probs = make_multiclass_data(n=100, C=2, seed=99)
    m = compute_multiclass_metrics(y_true, y_pred, num_classes=2, y_probs=y_probs)
    assert "brier_score" in m
    print("  \u2713 multiclass 2-class branch (column_stack)")


# --- binconf / precision_recall_f1 ---

def test_binconf_hand_computed():
    y_true = np.array([1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 0, 0, 1])
    cm = binconf(y_true, y_pred)
    assert cm[0, 0] == 2   # tn
    assert cm[0, 1] == 1   # fp
    assert cm[1, 0] == 1   # fn
    assert cm[1, 1] == 1   # tp
    print("  \u2713 binconf hand-computed")


def test_precision_recall_f1_known():
    cm = np.array([[90, 10], [5, 95]], dtype=np.int64)
    p, r, f1 = precision_recall_f1(cm)
    assert abs(p - 95 / 105) < 1e-6
    assert abs(r - 95 / 100) < 1e-6
    expected_f1 = 2 * p * r / (p + r)
    assert abs(f1 - expected_f1) < 1e-6
    print("  \u2713 precision_recall_f1 known cm")


def test_precision_recall_f1_zero_division():
    cm = np.array([[10, 0], [0, 0]], dtype=np.int64)
    p, r, f1 = precision_recall_f1(cm)
    assert p == 0.0 and r == 0.0 and f1 == 0.0
    print("  \u2713 precision_recall_f1 zero-division")


if __name__ == "__main__":
    print("test_metrics.py")
    test_binary_hand_computed()
    test_binary_perfect()
    test_binary_all_wrong()
    test_binary_with_prob()
    test_binary_perfect_brier()
    test_binary_without_prob()
    test_binary_all_positive()
    test_binary_all_negative()
    test_binary_small_n()
    test_multiclass_basic()
    test_multiclass_perfect()
    test_multiclass_with_probs()
    test_multiclass_2class_branch()
    test_binconf_hand_computed()
    test_precision_recall_f1_known()
    test_precision_recall_f1_zero_division()
    print("All tests passed.")
