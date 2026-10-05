"""Error analysis for misclassification investigation.

Identifies misclassified experiments, computes confidence distributions,
and generates per-study error breakdowns.
"""

import numpy as np
import pandas as pd


def identify_misclassifications(
    results: list,
    class_names: list,
) -> pd.DataFrame:
    """Build a DataFrame of misclassified experiments with metadata.

    Args:
        results: List of experiment result objects (from _predict_experiments)
        class_names: List of class name strings

    Returns:
        DataFrame with experiment_id, true_class, predicted_class, confidence,
        study_id, attention_entropy, error_type
    """
    rows = []
    for r in results:
        if np.isnan(r.y_prob) or r.has_null_label:
            continue

        true_cls = int(r.y_true)
        pred_cls = int(r.y_pred)
        is_correct = true_cls == pred_cls

        # Max probability as confidence
        probs = np.array(r.y_probs_all) if hasattr(r, "y_probs_all") else np.array([r.y_prob])
        confidence = float(probs.max()) if probs.ndim > 0 else float(r.y_prob)

        rows.append({
            "experiment_id": getattr(r, "experiment_id", None),
            "study_id": getattr(r, "study_id", None),
            "true_class": class_names[true_cls] if true_cls < len(class_names) else str(true_cls),
            "true_class_idx": true_cls,
            "predicted_class": class_names[pred_cls] if pred_cls < len(class_names) else str(pred_cls),
            "predicted_class_idx": pred_cls,
            "confidence": confidence,
            "correct": is_correct,
            "attention_entropy": getattr(r, "attn_entropy", None),
        })

    df = pd.DataFrame(rows)
    # Add error type column
    if len(df) > 0:
        df["error_type"] = df.apply(
            lambda row: "correct" if row["correct"]
            else f"{row['true_class']}→{row['predicted_class']}",
            axis=1,
        )
    return df


def identify_misclassifications_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: list,
    experiment_ids: np.ndarray = None,
    study_ids: np.ndarray = None,
) -> pd.DataFrame:
    """Build a DataFrame of misclassified experiments from arrays.

    Same output format as identify_misclassifications() but takes numpy arrays
    instead of ExperimentResult objects. Useful for CV pooled predictions where
    result objects are not available.

    Args:
        y_true: Ground truth class indices, shape (N,)
        y_pred: Predicted class indices, shape (N,)
        y_probs: Predicted probabilities, shape (N, C)
        class_names: List of class name strings
        experiment_ids: Optional experiment ID per sample, shape (N,)
        study_ids: Optional study ID per sample, shape (N,)

    Returns:
        DataFrame with experiment_id, true_class, predicted_class, confidence,
        study_id, error_type
    """
    rows = []
    for i in range(len(y_true)):
        true_cls = int(y_true[i])
        pred_cls = int(y_pred[i])
        is_correct = true_cls == pred_cls

        probs = y_probs[i] if y_probs is not None else None
        confidence = float(probs.max()) if probs is not None else None

        rows.append({
            "experiment_id": experiment_ids[i] if experiment_ids is not None else None,
            "study_id": study_ids[i] if study_ids is not None else None,
            "true_class": class_names[true_cls] if true_cls < len(class_names) else str(true_cls),
            "true_class_idx": true_cls,
            "predicted_class": class_names[pred_cls] if pred_cls < len(class_names) else str(pred_cls),
            "predicted_class_idx": pred_cls,
            "confidence": confidence,
            "correct": is_correct,
            "attention_entropy": None,
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["error_type"] = df.apply(
            lambda row: "correct" if row["correct"]
            else f"{row['true_class']}→{row['predicted_class']}",
            axis=1,
        )
    return df


