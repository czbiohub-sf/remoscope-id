#!/usr/bin/env python
"""Tests for remo_id/analysis/statistical_tests.py"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score
from _helpers_analysis import make_binary_data
from remo_id.analysis.statistical_tests import (
    mcnemar_test, delong_test, permutation_test, bootstrap_paired_diff,
)


# --- McNemar ---

def test_mcnemar_identical():
    y_true, y_pred, _ = make_binary_data(n=200)
    result = mcnemar_test(y_true, y_pred, y_pred)
    assert result["n_discordant"] == 0
    assert result["p_value"] == 1.0
    assert result["statistic"] == 0.0
    print("  \u2713 mcnemar identical models")


def test_mcnemar_different():
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    # Model A: 90% acc, Model B: 60% acc
    y_pred_a = y_true.copy()
    flip_a = rng.random(n) < 0.1
    y_pred_a[flip_a] = 1 - y_pred_a[flip_a]
    y_pred_b = y_true.copy()
    flip_b = rng.random(n) < 0.4
    y_pred_b[flip_b] = 1 - y_pred_b[flip_b]

    result = mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert result["p_value"] < 0.05
    print("  \u2713 mcnemar different models -> p < 0.05")


def test_mcnemar_output_types():
    y_true, y_pred, _ = make_binary_data(n=200)
    rng = np.random.default_rng(99)
    y_pred_b = y_pred.copy()
    flip = rng.random(200) < 0.3
    y_pred_b[flip] = 1 - y_pred_b[flip]
    result = mcnemar_test(y_true, y_pred, y_pred_b)
    assert isinstance(result["statistic"], float)
    assert isinstance(result["b"], int)
    assert isinstance(result["c"], int)
    print("  \u2713 mcnemar output types")


def test_mcnemar_small_sample_warning():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred_a = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred_b = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mcnemar_test(y_true, y_pred_a, y_pred_b)
        warned = any("Low statistical power" in str(x.message) for x in w)
        assert warned
    print("  \u2713 mcnemar small sample warning")


# --- DeLong ---

def test_delong_identical():
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_score = rng.random(n)
    result = delong_test(y_true, y_score, y_score)
    assert result["z_statistic"] == 0.0
    assert result["p_value"] == 1.0
    print("  \u2713 delong identical scores")


def test_delong_different():
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_score_a = y_true * 0.8 + rng.normal(0, 0.1, n)
    y_score_b = y_true * 0.3 + rng.normal(0, 0.3, n)
    result = delong_test(y_true, y_score_a, y_score_b)
    assert result["p_value"] < 0.05
    print("  \u2713 delong different scores -> p < 0.05")


def test_delong_auc_matches_sklearn():
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_score_a = rng.random(n)
    y_score_b = rng.random(n)
    result = delong_test(y_true, y_score_a, y_score_b)
    assert abs(result["auc_a"] - roc_auc_score(y_true, y_score_a)) < 1e-6
    assert abs(result["auc_b"] - roc_auc_score(y_true, y_score_b)) < 1e-6
    print("  \u2713 delong AUC matches sklearn")


def test_delong_output_types():
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_score_a = rng.random(n)
    y_score_b = rng.random(n)
    result = delong_test(y_true, y_score_a, y_score_b)
    for key in ["auc_a", "auc_b", "z_statistic", "p_value"]:
        assert isinstance(result[key], float), f"{key} not float"
    print("  \u2713 delong output types")


# --- Permutation test ---

def test_permutation_identical():
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_score = rng.random(n)
    result = permutation_test(y_true, y_score, y_score, roc_auc_score, n=500, seed=42)
    assert result["observed_diff"] == 0.0
    assert result["p_value"] >= 0.9
    print("  \u2713 permutation identical -> p ~1.0")


def test_permutation_different():
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_score_a = y_true * 0.8 + rng.normal(0, 0.1, n)
    y_score_b = rng.random(n)
    result = permutation_test(y_true, y_score_a, y_score_b, roc_auc_score, n=500, seed=42)
    assert result["p_value"] < 0.1
    print("  \u2713 permutation different -> p < 0.1")


def test_permutation_deterministic():
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_score_a = rng.random(n)
    y_score_b = rng.random(n)
    r1 = permutation_test(y_true, y_score_a, y_score_b, roc_auc_score, n=200, seed=42)
    r2 = permutation_test(y_true, y_score_a, y_score_b, roc_auc_score, n=200, seed=42)
    assert r1["p_value"] == r2["p_value"]
    print("  \u2713 permutation deterministic with same seed")


# --- Bootstrap paired diff ---

def test_bootstrap_identical():
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_score = rng.random(n)
    result = bootstrap_paired_diff(y_true, y_score, y_score, roc_auc_score, n=200, seed=42)
    assert result["observed_diff"] == 0.0
    assert result["ci_lo"] <= 0.0 <= result["ci_hi"]
    print("  \u2713 bootstrap identical -> diff=0, CI spans 0")


def test_bootstrap_a_better():
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_score_a = y_true * 0.8 + rng.normal(0, 0.1, n)
    y_score_b = rng.random(n)
    result = bootstrap_paired_diff(y_true, y_score_a, y_score_b, roc_auc_score, n=500, seed=42)
    assert result["observed_diff"] > 0
    assert result["ci_lo"] > 0
    print("  \u2713 bootstrap A clearly better -> ci_lo > 0")


def test_bootstrap_all_keys():
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_score_a = rng.random(n)
    y_score_b = rng.random(n)
    result = bootstrap_paired_diff(y_true, y_score_a, y_score_b, roc_auc_score, n=200, seed=42)
    for key in ["observed_diff", "ci_lo", "ci_hi", "p_value", "bootstrap_mean", "bootstrap_std"]:
        assert key in result, f"missing key: {key}"
    print("  \u2713 bootstrap all expected keys")


def test_bootstrap_deterministic():
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_score_a = rng.random(n)
    y_score_b = rng.random(n)
    r1 = bootstrap_paired_diff(y_true, y_score_a, y_score_b, roc_auc_score, n=200, seed=42)
    r2 = bootstrap_paired_diff(y_true, y_score_a, y_score_b, roc_auc_score, n=200, seed=42)
    assert r1["ci_lo"] == r2["ci_lo"]
    print("  \u2713 bootstrap deterministic with same seed")


def test_bootstrap_p_value_range():
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_score_a = rng.random(n)
    y_score_b = rng.random(n)
    result = bootstrap_paired_diff(y_true, y_score_a, y_score_b, roc_auc_score, n=200, seed=42)
    assert 0.0 <= result["p_value"] <= 1.0
    print("  \u2713 bootstrap p_value in [0,1]")


if __name__ == "__main__":
    print("test_statistical_tests.py")
    test_mcnemar_identical()
    test_mcnemar_different()
    test_mcnemar_output_types()
    test_mcnemar_small_sample_warning()
    test_delong_identical()
    test_delong_different()
    test_delong_auc_matches_sklearn()
    test_delong_output_types()
    test_permutation_identical()
    test_permutation_different()
    test_permutation_deterministic()
    test_bootstrap_identical()
    test_bootstrap_a_better()
    test_bootstrap_all_keys()
    test_bootstrap_deterministic()
    test_bootstrap_p_value_range()
    print("All tests passed.")
