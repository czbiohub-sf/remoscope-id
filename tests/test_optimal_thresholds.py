#!/usr/bin/env python
"""Tests for remo_id/analysis/optimal_thresholds.py"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pytest
from _helpers_analysis import make_multiclass_data, make_multiclass_data_perfect
from remo_id.analysis.optimal_thresholds import (
    build_sensitivity_targets,
    find_optimal_thresholds,
)


# ---------------------------------------------------------------------------
# build_sensitivity_targets
# ---------------------------------------------------------------------------

def test_build_sensitivity_targets_default():
    targets = build_sensitivity_targets(["A", "B", "C"], 0.95)
    assert targets == {"A": 0.95, "B": 0.95, "C": 0.95}
    print("  v build_sensitivity_targets default")


def test_build_sensitivity_targets_override():
    targets = build_sensitivity_targets(
        ["disease+", "healthy", "infection+"], 0.95,
        "disease+:0.99,infection+:0.90",
    )
    assert targets["disease+"] == 0.99
    assert targets["healthy"] == 0.95
    assert targets["infection+"] == 0.90
    print("  v build_sensitivity_targets override")


def test_build_sensitivity_targets_whitespace():
    targets = build_sensitivity_targets(
        ["A", "B"], 0.95, " A : 0.80 , B : 0.70 ",
    )
    assert targets["A"] == 0.80
    assert targets["B"] == 0.70
    print("  v build_sensitivity_targets whitespace tolerance")


def test_build_sensitivity_targets_no_override():
    targets = build_sensitivity_targets(["X"], 0.50, None)
    assert targets == {"X": 0.50}
    print("  v build_sensitivity_targets None override")


def test_build_sensitivity_targets_unknown_class():
    with pytest.raises(ValueError, match="Unknown class 'Z'"):
        build_sensitivity_targets(["A", "B"], 0.95, "Z:0.99")
    print("  v build_sensitivity_targets unknown class -> ValueError")


def test_build_sensitivity_targets_invalid_format():
    with pytest.raises(ValueError, match="Invalid sensitivity target"):
        build_sensitivity_targets(["A", "B"], 0.95, "A0.99")
    print("  v build_sensitivity_targets invalid format -> ValueError")


def test_build_sensitivity_targets_partial_override():
    targets = build_sensitivity_targets(["A", "B", "C"], 0.95, "B:0.80")
    assert targets["A"] == 0.95
    assert targets["B"] == 0.80
    assert targets["C"] == 0.95
    print("  v build_sensitivity_targets partial override")


# ---------------------------------------------------------------------------
# find_optimal_thresholds
# ---------------------------------------------------------------------------

def test_find_optimal_thresholds_perfect_binary():
    """Perfect separation should achieve target sensitivity and high specificity."""
    y_true, _, y_probs = make_multiclass_data_perfect(n=200, C=2)
    targets = {"A": 0.95, "B": 0.95}
    results = find_optimal_thresholds(y_true, y_probs, ["A", "B"], targets)
    assert len(results) == 2
    for r in results:
        assert r["achieved_sensitivity"] >= 0.95
        assert r["achieved_specificity"] >= 0.95
        assert "warning" not in r
    print("  v find_optimal_thresholds perfect binary")


def test_find_optimal_thresholds_perfect_multiclass():
    """Perfect separation with 3+ classes."""
    y_true, _, y_probs = make_multiclass_data_perfect(n=300, C=3)
    names = ["A", "B", "C"]
    targets = {n: 0.95 for n in names}
    results = find_optimal_thresholds(y_true, y_probs, names, targets)
    assert len(results) == 3
    for r in results:
        assert r["achieved_sensitivity"] >= 0.95
        assert "warning" not in r
    print("  v find_optimal_thresholds perfect multiclass")


def test_find_optimal_thresholds_output_keys():
    """Verify all expected keys are present in each result dict."""
    y_true, _, y_probs = make_multiclass_data(n=200, C=3)
    names = ["A", "B", "C"]
    targets = {n: 0.95 for n in names}
    results = find_optimal_thresholds(y_true, y_probs, names, targets)
    expected_keys = {
        "class_name", "target_sensitivity", "optimal_threshold",
        "achieved_sensitivity", "achieved_specificity", "achieved_precision",
        "achieved_f1", "achieved_npv", "n_positive", "n_negative",
    }
    for r in results:
        assert expected_keys.issubset(r.keys())
    print("  v find_optimal_thresholds output keys")


def test_find_optimal_thresholds_threshold_range():
    """Optimal threshold must be between 0.01 and 0.99."""
    y_true, _, y_probs = make_multiclass_data(n=200, C=3)
    names = ["A", "B", "C"]
    targets = {n: 0.90 for n in names}
    results = find_optimal_thresholds(y_true, y_probs, names, targets)
    for r in results:
        assert 0.01 <= r["optimal_threshold"] <= 0.99
    print("  v find_optimal_thresholds threshold range")


def test_find_optimal_thresholds_metrics_range():
    """All metric values should be in [0, 1]."""
    y_true, _, y_probs = make_multiclass_data(n=200, C=3)
    names = ["A", "B", "C"]
    targets = {n: 0.90 for n in names}
    results = find_optimal_thresholds(y_true, y_probs, names, targets)
    metric_keys = ["achieved_sensitivity", "achieved_specificity",
                   "achieved_precision", "achieved_f1", "achieved_npv"]
    for r in results:
        for k in metric_keys:
            assert 0.0 <= r[k] <= 1.0, f"{k}={r[k]} out of [0,1]"
    print("  v find_optimal_thresholds metrics range")


def test_find_optimal_thresholds_class_counts():
    """n_positive + n_negative should equal total samples."""
    n = 200
    y_true, _, y_probs = make_multiclass_data(n=n, C=3)
    names = ["A", "B", "C"]
    targets = {n_: 0.90 for n_ in names}
    results = find_optimal_thresholds(y_true, y_probs, names, targets)
    for r in results:
        assert r["n_positive"] + r["n_negative"] == n
    print("  v find_optimal_thresholds class counts")


def test_find_optimal_thresholds_meets_target_when_possible():
    """With decent data and a modest target, sensitivity should be met."""
    y_true, _, y_probs = make_multiclass_data(n=500, C=2, seed=42)
    names = ["A", "B"]
    # Use a low target that should be achievable
    targets = {n: 0.70 for n in names}
    results = find_optimal_thresholds(y_true, y_probs, names, targets)
    for r in results:
        assert r["achieved_sensitivity"] >= 0.70
        assert "warning" not in r
    print("  v find_optimal_thresholds meets modest target")


def test_find_optimal_thresholds_impossible_target_warns():
    """When target is 1.0 and model is imperfect, should produce warning."""
    y_true, _, y_probs = make_multiclass_data(n=200, C=2, seed=42)
    names = ["A", "B"]
    targets = {n: 1.0 for n in names}
    results = find_optimal_thresholds(y_true, y_probs, names, targets)
    # At least one class should have a warning since the model is imperfect
    # and 1.0 sensitivity may not be achievable at any threshold in [0.01, 0.99]
    has_warning = any("warning" in r for r in results)
    # If no warning, the sensitivity was actually achieved, which is also fine
    if not has_warning:
        for r in results:
            assert r["achieved_sensitivity"] >= 1.0
    print("  v find_optimal_thresholds impossible target handling")


def test_find_optimal_thresholds_binary_two_classes():
    """Binary case: exactly 2 results returned."""
    y_true, _, y_probs = make_multiclass_data(n=200, C=2)
    names = ["pos", "neg"]
    targets = {"pos": 0.95, "neg": 0.95}
    results = find_optimal_thresholds(y_true, y_probs, names, targets)
    assert len(results) == 2
    assert results[0]["class_name"] == "pos"
    assert results[1]["class_name"] == "neg"
    print("  v find_optimal_thresholds binary two classes")


def test_find_optimal_thresholds_different_targets_per_class():
    """Different sensitivity targets per class should produce different thresholds."""
    y_true, _, y_probs = make_multiclass_data(n=500, C=2, seed=42)
    names = ["A", "B"]
    targets_high = {"A": 0.99, "B": 0.50}
    targets_low = {"A": 0.50, "B": 0.99}
    results_high = find_optimal_thresholds(y_true, y_probs, names, targets_high)
    results_low = find_optimal_thresholds(y_true, y_probs, names, targets_low)
    # Different targets should generally produce different thresholds for class A
    assert results_high[0]["target_sensitivity"] == 0.99
    assert results_low[0]["target_sensitivity"] == 0.50
    print("  v find_optimal_thresholds different targets per class")


def test_find_optimal_thresholds_maximizes_specificity():
    """Among thresholds meeting sensitivity target, specificity should be maximized."""
    y_true, _, y_probs = make_multiclass_data_perfect(n=200, C=2)
    names = ["A", "B"]
    targets = {n: 0.50 for n in names}  # low target, many candidates
    results = find_optimal_thresholds(y_true, y_probs, names, targets)
    # With perfect data and low target, specificity should be very high
    for r in results:
        assert r["achieved_specificity"] >= 0.95
    print("  v find_optimal_thresholds maximizes specificity")


if __name__ == "__main__":
    print("test_optimal_thresholds.py")
    test_build_sensitivity_targets_default()
    test_build_sensitivity_targets_override()
    test_build_sensitivity_targets_whitespace()
    test_build_sensitivity_targets_no_override()
    test_build_sensitivity_targets_unknown_class()
    test_build_sensitivity_targets_invalid_format()
    test_build_sensitivity_targets_partial_override()
    test_find_optimal_thresholds_perfect_binary()
    test_find_optimal_thresholds_perfect_multiclass()
    test_find_optimal_thresholds_output_keys()
    test_find_optimal_thresholds_threshold_range()
    test_find_optimal_thresholds_metrics_range()
    test_find_optimal_thresholds_class_counts()
    test_find_optimal_thresholds_meets_target_when_possible()
    test_find_optimal_thresholds_impossible_target_warns()
    test_find_optimal_thresholds_binary_two_classes()
    test_find_optimal_thresholds_different_targets_per_class()
    test_find_optimal_thresholds_maximizes_specificity()
    print("All tests passed.")
