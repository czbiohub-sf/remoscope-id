#!/usr/bin/env python
"""Tests for remo_id/analysis/error_analysis.py"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from types import SimpleNamespace
from _helpers_analysis import make_mock_results
from remo_id.analysis.error_analysis import identify_misclassifications


def test_basic_output():
    results = make_mock_results(n=50, C=3)
    df = identify_misclassifications(results, ["A", "B", "C"])
    assert len(df) > 0
    for col in ["experiment_id", "study_id", "true_class", "predicted_class",
                "confidence", "correct", "error_type", "true_class_idx", "predicted_class_idx"]:
        assert col in df.columns, f"missing column: {col}"
    print("  \u2713 basic output: DataFrame with expected columns")


def test_correct_column():
    results = make_mock_results(n=50, C=3)
    df = identify_misclassifications(results, ["A", "B", "C"])
    for _, row in df.iterrows():
        expected = row["true_class_idx"] == row["predicted_class_idx"]
        assert row["correct"] == expected
    print("  \u2713 correct column matches true_class_idx == predicted_class_idx")


def test_error_type_format():
    results = make_mock_results(n=50, C=3)
    df = identify_misclassifications(results, ["A", "B", "C"])
    for _, row in df.iterrows():
        if row["correct"]:
            assert row["error_type"] == "correct"
        else:
            assert "\u2192" in row["error_type"]
    print("  \u2713 error_type = 'correct' or 'X\u2192Y'")


def test_nan_prob_excluded():
    results = [
        SimpleNamespace(y_prob=float("nan"), has_null_label=False, y_true=0, y_pred=0,
                       y_probs_all=[0.5, 0.3, 0.2], experiment_id=1, study_id="s", attn_entropy=0.5),
        SimpleNamespace(y_prob=0.8, has_null_label=False, y_true=1, y_pred=1,
                       y_probs_all=[0.1, 0.8, 0.1], experiment_id=2, study_id="s", attn_entropy=0.5),
    ]
    df = identify_misclassifications(results, ["A", "B", "C"])
    assert len(df) == 1
    assert df.iloc[0]["experiment_id"] == 2
    print("  \u2713 NaN y_prob excluded")


def test_null_label_excluded():
    results = [
        SimpleNamespace(y_prob=0.9, has_null_label=True, y_true=0, y_pred=0,
                       y_probs_all=[0.9, 0.05, 0.05], experiment_id=1, study_id="s", attn_entropy=0.5),
        SimpleNamespace(y_prob=0.7, has_null_label=False, y_true=1, y_pred=1,
                       y_probs_all=[0.1, 0.7, 0.2], experiment_id=2, study_id="s", attn_entropy=0.5),
    ]
    df = identify_misclassifications(results, ["A", "B", "C"])
    assert len(df) == 1
    assert df.iloc[0]["experiment_id"] == 2
    print("  \u2713 has_null_label=True excluded")


def test_empty_input():
    df = identify_misclassifications([], ["A", "B", "C"])
    assert len(df) == 0
    print("  \u2713 empty input -> empty DataFrame")


def test_all_correct():
    results = []
    for i in range(10):
        results.append(SimpleNamespace(
            y_prob=0.9, has_null_label=False, y_true=0, y_pred=0,
            y_probs_all=[0.9, 0.05, 0.05], experiment_id=i, study_id="s", attn_entropy=0.3,
        ))
    df = identify_misclassifications(results, ["A", "B", "C"])
    assert (df["correct"] == True).all()
    print("  \u2713 all correct -> no incorrect rows")


def test_confidence_range():
    results = make_mock_results(n=50, C=3)
    df = identify_misclassifications(results, ["A", "B", "C"])
    assert (df["confidence"] >= 0).all()
    assert (df["confidence"] <= 1).all()
    print("  \u2713 confidence values in [0, 1]")


if __name__ == "__main__":
    print("test_error_analysis.py")
    test_basic_output()
    test_correct_column()
    test_error_type_format()
    test_nan_prob_excluded()
    test_null_label_excluded()
    test_empty_input()
    test_all_correct()
    test_confidence_range()
    print("All tests passed.")
