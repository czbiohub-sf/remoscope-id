"""Statistical comparison tests for model evaluation.

Functions for comparing two models or two sets of predictions:
McNemar, DeLong, permutation test, bootstrap paired difference.
"""

import warnings
import numpy as np
from typing import Callable, Dict


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict:
    """McNemar's test for comparing two classifiers on the same data.

    Tests whether models A and B make the same pattern of errors.

    Args:
        y_true: Ground truth labels
        y_pred_a: Predictions from model A
        y_pred_b: Predictions from model B

    Returns:
        Dict with statistic, p_value, n_discordant, contingency_table
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency table of discordant pairs
    b = int((correct_a & ~correct_b).sum())  # A right, B wrong
    c = int((~correct_a & correct_b).sum())  # A wrong, B right

    n_discordant = b + c

    if n_discordant < 100:
        warnings.warn(f"Only {n_discordant} discordant pairs (< 100). Low statistical power.")

    # McNemar with continuity correction
    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_discordant": 0, "b": b, "c": c}

    statistic = (abs(b - c) - 1) ** 2 / (b + c)

    from scipy.stats import chi2
    p_value = float(chi2.sf(statistic, df=1))

    return {
        "statistic": float(statistic),
        "p_value": p_value,
        "n_discordant": n_discordant,
        "b": b,
        "c": c,
    }


def delong_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> Dict:
    """DeLong's test for comparing two AUC values on the same data.

    Args:
        y_true: Ground truth binary labels (0/1)
        y_score_a: Predicted scores from model A
        y_score_b: Predicted scores from model B

    Returns:
        Dict with auc_a, auc_b, z_statistic, p_value
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    y_score_a = np.asarray(y_score_a, dtype=float)
    y_score_b = np.asarray(y_score_b, dtype=float)

    n = len(y_true)
    if n < 100:
        warnings.warn(f"Only {n} samples (< 100). Low statistical power for DeLong test.")

    auc_a = float(roc_auc_score(y_true, y_score_a))
    auc_b = float(roc_auc_score(y_true, y_score_b))

    # Compute placement values (structural components of the AUC)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m = len(pos_idx)
    n_neg = len(neg_idx)

    def _placement_values(y_score):
        """Compute V statistics for DeLong's test."""
        v_pos = np.zeros(m)
        v_neg = np.zeros(n_neg)
        for i, pi in enumerate(pos_idx):
            v_pos[i] = np.mean(y_score[pi] > y_score[neg_idx]) + 0.5 * np.mean(y_score[pi] == y_score[neg_idx])
        for j, nj in enumerate(neg_idx):
            v_neg[j] = np.mean(y_score[pos_idx] > y_score[nj]) + 0.5 * np.mean(y_score[pos_idx] == y_score[nj])
        return v_pos, v_neg

    v_pos_a, v_neg_a = _placement_values(y_score_a)
    v_pos_b, v_neg_b = _placement_values(y_score_b)

    # Covariance matrix of (AUC_A, AUC_B)
    s_pos = np.cov(np.column_stack([v_pos_a, v_pos_b]), rowvar=False)
    s_neg = np.cov(np.column_stack([v_neg_a, v_neg_b]), rowvar=False)

    s = s_pos / m + s_neg / n_neg

    # z-statistic
    diff = auc_a - auc_b
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]

    if var_diff <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b, "z_statistic": 0.0, "p_value": 1.0}

    z = diff / np.sqrt(var_diff)

    from scipy.stats import norm
    p_value = float(2 * norm.sf(abs(z)))

    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "auc_diff": float(diff),
        "z_statistic": float(z),
        "p_value": p_value,
    }


def permutation_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    metric_fn: Callable,
    n: int = 10000,
    seed: int = 0,
) -> Dict:
    """Permutation test for comparing two models on any metric.

    Args:
        y_true: Ground truth labels
        y_score_a: Scores/predictions from model A
        y_score_b: Scores/predictions from model B
        metric_fn: callable(y_true, y_score) -> scalar metric
        n: Number of permutations
        seed: Random seed

    Returns:
        Dict with observed_diff, p_value, null_distribution
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score_a = np.asarray(y_score_a)
    y_score_b = np.asarray(y_score_b)

    if len(y_true) < 100:
        warnings.warn(f"Only {len(y_true)} samples (< 100). Low statistical power.")

    observed_a = metric_fn(y_true, y_score_a)
    observed_b = metric_fn(y_true, y_score_b)
    observed_diff = observed_a - observed_b

    null_diffs = np.empty(n)
    for i in range(n):
        swap = rng.random(len(y_true)) > 0.5
        perm_a = np.where(swap, y_score_b, y_score_a)
        perm_b = np.where(swap, y_score_a, y_score_b)
        null_diffs[i] = metric_fn(y_true, perm_a) - metric_fn(y_true, perm_b)

    p_value = float(np.mean(np.abs(null_diffs) >= np.abs(observed_diff)))

    return {
        "observed_diff": float(observed_diff),
        "metric_a": float(observed_a),
        "metric_b": float(observed_b),
        "p_value": p_value,
        "null_mean": float(null_diffs.mean()),
        "null_std": float(null_diffs.std()),
    }


def bootstrap_paired_diff(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_fn: Callable,
    n: int = 1000,
    seed: int = 0,
) -> Dict:
    """Bootstrap paired difference between two models' metrics with CI.

    Args:
        y_true: Ground truth labels
        scores_a: Scores from model A
        scores_b: Scores from model B
        metric_fn: callable(y_true, y_score) -> scalar
        n: Bootstrap iterations
        seed: Random seed

    Returns:
        Dict with observed_diff, ci_lo, ci_hi, p_value (proportion crossing 0)
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    if len(y_true) < 100:
        warnings.warn(f"Only {len(y_true)} samples (< 100). Low statistical power.")

    observed_diff = metric_fn(y_true, scores_a) - metric_fn(y_true, scores_b)

    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        try:
            diff_i = metric_fn(y_true[idx], scores_a[idx]) - metric_fn(y_true[idx], scores_b[idx])
            diffs[i] = diff_i
        except Exception:
            diffs[i] = np.nan

    valid = diffs[~np.isnan(diffs)]
    ci_lo = float(np.percentile(valid, 2.5)) if len(valid) else np.nan
    ci_hi = float(np.percentile(valid, 97.5)) if len(valid) else np.nan

    # p-value: proportion of bootstrap diffs that cross zero
    if observed_diff >= 0:
        p_value = float(np.mean(valid <= 0)) * 2  # two-sided
    else:
        p_value = float(np.mean(valid >= 0)) * 2

    return {
        "observed_diff": float(observed_diff),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": min(p_value, 1.0),
        "bootstrap_mean": float(np.mean(valid)) if len(valid) else np.nan,
        "bootstrap_std": float(np.std(valid)) if len(valid) else np.nan,
    }
