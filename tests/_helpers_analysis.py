"""Shared synthetic data generators for analysis module tests."""

import numpy as np
import pandas as pd
from types import SimpleNamespace


def make_binary_data(n=100, seed=42):
    """Generate synthetic binary classification data with ~80% accuracy.

    Returns:
        (y_true, y_pred, y_prob) arrays of length n
    """
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    # ~80% correct predictions
    flip = rng.random(n) < 0.2
    y_pred = y_true.copy()
    y_pred[flip] = 1 - y_pred[flip]
    # Probabilities correlated with true labels
    y_prob = np.clip(y_true * 0.7 + rng.normal(0, 0.15, n), 0.01, 0.99)
    return y_true, y_pred, y_prob


def make_binary_data_perfect(n=100):
    """Generate perfect binary classification data.

    Returns:
        (y_true, y_pred, y_prob) with 100% accuracy
    """
    y_true = np.array([0, 1] * (n // 2))
    y_pred = y_true.copy()
    y_prob = y_true.astype(float)
    return y_true, y_pred, y_prob


def make_multiclass_data(n=200, C=4, seed=42):
    """Generate synthetic multiclass data with ~70% accuracy.

    Returns:
        (y_true, y_pred, y_probs) where y_probs is [N, C]
    """
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, C, size=n)
    # ~70% correct
    flip = rng.random(n) < 0.3
    y_pred = y_true.copy()
    y_pred[flip] = rng.integers(0, C, size=flip.sum())
    # Probability matrix: elevated for true class
    y_probs = rng.dirichlet(np.ones(C) * 0.5, size=n)
    for i in range(n):
        y_probs[i, y_true[i]] += 1.5
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
    return y_true, y_pred, y_probs


def make_multiclass_data_perfect(n=200, C=4):
    """Generate perfect multiclass data.

    Returns:
        (y_true, y_pred, y_probs) with 100% accuracy and one-hot probs
    """
    y_true = np.tile(np.arange(C), n // C + 1)[:n]
    y_pred = y_true.copy()
    y_probs = np.zeros((n, C))
    for i in range(n):
        y_probs[i, y_true[i]] = 1.0
    return y_true, y_pred, y_probs


def make_mock_data_module(C=3, exp_per_class=10, img_per_exp=50):
    """Create a mock data module for dataset_stats tests.

    Returns:
        SimpleNamespace with .data, .experiment_labels, .experiment_ranges
    """
    rows = []
    experiment_labels = {}
    experiment_ranges = {}
    global_idx = 0

    for c in range(C):
        for e in range(exp_per_class):
            exp_id = c * 100 + e
            path = f"/data/class{c}/exp{exp_id}"
            study_id = f"study_{c}"
            sample_id = f"sample_{c}_{e // 3}"

            rows.append({
                "experiment_id": exp_id,
                "path": path,
                "study_id": study_id,
                "sample_id": sample_id,
                "num_images": img_per_exp,
            })
            experiment_labels[path] = c
            experiment_ranges[path] = (global_idx, global_idx + img_per_exp)
            global_idx += img_per_exp

    data = pd.DataFrame(rows)
    return SimpleNamespace(
        data=data,
        experiment_labels=experiment_labels,
        experiment_ranges=experiment_ranges,
    )


def make_mock_results(n=50, C=3, seed=42):
    """Create mock experiment results for error_analysis tests.

    Returns:
        list of SimpleNamespace objects with y_prob, has_null_label,
        y_true, y_pred, y_probs_all, experiment_id, study_id, attn_entropy
    """
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n):
        true_cls = rng.integers(0, C)
        probs = rng.dirichlet(np.ones(C) * 0.5)
        probs[true_cls] += 1.0
        probs = probs / probs.sum()
        pred_cls = int(probs.argmax())

        results.append(SimpleNamespace(
            y_prob=float(probs.max()),
            has_null_label=False,
            y_true=true_cls,
            y_pred=pred_cls,
            y_probs_all=probs.tolist(),
            experiment_id=1000 + i,
            study_id=f"study_{true_cls}",
            attn_entropy=float(rng.random()),
        ))
    return results


def make_post_training_results(n=50, seed=42):
    """Create mock post-training results dicts.

    Returns:
        list of dicts with split, true_label, pred_label, mean_prob,
        std_prob, run_probs, study_id, experiment_id
    """
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n):
        true_label = int(rng.integers(0, 2))
        mean_prob = np.clip(true_label * 0.7 + rng.normal(0, 0.15), 0.01, 0.99)
        std_prob = float(rng.uniform(0.01, 0.1))
        pred_label = int(mean_prob > 0.5)
        split = "val" if i < n // 5 else "train"

        results.append({
            "split": split,
            "true_label": true_label,
            "pred_label": pred_label,
            "mean_prob": float(mean_prob),
            "std_prob": std_prob,
            "run_probs": [float(mean_prob + rng.normal(0, 0.05)) for _ in range(5)],
            "study_id": f"study_{true_label}",
            "experiment_id": 2000 + i,
        })
    return results


def make_error_df(n=100, C=3, seed=42):
    """Create a DataFrame for error analysis plot tests.

    Returns:
        DataFrame with study_id, true_class_idx, predicted_class_idx,
        correct, error_type columns
    """
    rng = np.random.default_rng(seed)
    class_names = [f"class_{i}" for i in range(C)]
    rows = []
    for i in range(n):
        true_cls = rng.integers(0, C)
        pred_cls = true_cls if rng.random() > 0.3 else rng.integers(0, C)
        correct = true_cls == pred_cls
        error_type = "correct" if correct else f"{class_names[true_cls]}\u2192{class_names[pred_cls]}"
        rows.append({
            "study_id": f"study_{rng.integers(0, 3)}",
            "true_class_idx": true_cls,
            "predicted_class_idx": pred_cls,
            "true_class": class_names[true_cls],
            "predicted_class": class_names[pred_cls],
            "correct": correct,
            "error_type": error_type,
            "confidence": float(rng.uniform(0.3, 1.0)),
            "experiment_id": 3000 + i,
        })
    return pd.DataFrame(rows)
