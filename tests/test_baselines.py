#!/usr/bin/env python
"""Tests for remo_id/analysis/baselines.py (run_baselines only)"""

import numpy as np
from sklearn.datasets import make_blobs
from remo_id.analysis.baselines import run_baselines


def test_binary_baselines_with_pretrained():
    X, y = make_blobs(n_samples=200, centers=2, n_features=10, random_state=42)
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]
    # Pretrained features: same shape but different (shifted)
    X_pt = X + np.random.RandomState(99).randn(*X.shape) * 0.5
    X_train_pt, X_val_pt = X_pt[:150], X_pt[150:]
    df = run_baselines(
        X_train, y_train, X_val, y_val,
        task_type="binary", num_classes=2,
        X_train_pretrained=X_train_pt, X_val_pretrained=X_val_pt,
    )
    assert len(df) == 3
    assert set(df["model"]) == {"Pretrained + Linear", "Linear", "MajorityClass"}
    for col in ["model", "accuracy", "balanced_accuracy", "f1_macro", "auc_macro"]:
        assert col in df.columns, f"missing column: {col}"
    print("  \u2713 binary baselines with pretrained: 3 rows, expected columns")


def test_binary_baselines_without_pretrained():
    X, y = make_blobs(n_samples=200, centers=2, n_features=10, random_state=42)
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]
    df = run_baselines(X_train, y_train, X_val, y_val, task_type="binary", num_classes=2)
    assert len(df) == 2
    assert set(df["model"]) == {"Linear", "MajorityClass"}
    print("  \u2713 binary baselines without pretrained: 2 rows")


def test_multiclass_baselines():
    X, y = make_blobs(n_samples=300, centers=4, n_features=10, random_state=42)
    X_train, X_val = X[:240], X[240:]
    y_train, y_val = y[:240], y[240:]
    df = run_baselines(X_train, y_train, X_val, y_val, task_type="multiclass", num_classes=4)
    assert len(df) == 2
    assert set(df["model"]) == {"Linear", "MajorityClass"}
    print("  \u2713 multiclass baselines: 2 rows")


def test_accuracy_range():
    X, y = make_blobs(n_samples=200, centers=2, n_features=10, random_state=42)
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]
    df = run_baselines(X_train, y_train, X_val, y_val, task_type="binary", num_classes=2)
    for _, row in df.iterrows():
        assert 0 <= row["accuracy"] <= 1
    print("  \u2713 all accuracy in [0, 1]")


def test_linear_beats_majority():
    X, y = make_blobs(n_samples=200, centers=2, n_features=10, random_state=42, cluster_std=1.0)
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]
    df = run_baselines(X_train, y_train, X_val, y_val, task_type="binary", num_classes=2)
    lr_acc = df[df["model"] == "Linear"]["accuracy"].values[0]
    maj_acc = df[df["model"] == "MajorityClass"]["accuracy"].values[0]
    assert lr_acc > maj_acc
    print("  \u2713 Linear accuracy > Majority on separable data")


def test_tiny_dataset():
    X, y = make_blobs(n_samples=16, centers=2, n_features=5, random_state=42)
    X_train, X_val = X[:8], X[8:]
    y_train, y_val = y[:8], y[8:]
    df = run_baselines(X_train, y_train, X_val, y_val, task_type="binary", num_classes=2)
    assert len(df) == 2
    print("  \u2713 tiny dataset (n=4 per split) no crash")


def test_model_names_with_pretrained():
    X, y = make_blobs(n_samples=200, centers=2, n_features=10, random_state=42)
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]
    X_pt = X * 0.8
    X_train_pt, X_val_pt = X_pt[:150], X_pt[150:]
    df = run_baselines(
        X_train, y_train, X_val, y_val,
        task_type="binary", num_classes=2,
        X_train_pretrained=X_train_pt, X_val_pretrained=X_val_pt,
    )
    expected_models = {"Pretrained + Linear", "Linear", "MajorityClass"}
    assert set(df["model"]) == expected_models
    print("  \u2713 model names correct (with pretrained)")


def test_pretrained_present_only_when_provided():
    X, y = make_blobs(n_samples=200, centers=2, n_features=10, random_state=42)
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]

    # Without pretrained
    df_no_pt = run_baselines(X_train, y_train, X_val, y_val, task_type="binary", num_classes=2)
    assert "Pretrained + Linear" not in df_no_pt["model"].values

    # With pretrained
    df_pt = run_baselines(
        X_train, y_train, X_val, y_val,
        task_type="binary", num_classes=2,
        X_train_pretrained=X_train, X_val_pretrained=X_val,
    )
    assert "Pretrained + Linear" in df_pt["model"].values
    print("  \u2713 Pretrained + Linear present only when pretrained features provided")


if __name__ == "__main__":
    print("test_baselines.py")
    test_binary_baselines_with_pretrained()
    test_binary_baselines_without_pretrained()
    test_multiclass_baselines()
    test_accuracy_range()
    test_linear_beats_majority()
    test_tiny_dataset()
    test_model_names_with_pretrained()
    test_pretrained_present_only_when_provided()
    print("All tests passed.")
