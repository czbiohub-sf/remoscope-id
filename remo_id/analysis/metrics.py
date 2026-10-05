"""Metrics computation for remo-id evaluation."""

import numpy as np
from typing import Dict, Any, Tuple


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted labels (0/1)
        y_prob: Predicted probabilities for positive class (optional, for Brier score)

    Returns:
        Dictionary with accuracy, balanced_accuracy, precision, recall, sensitivity,
        specificity, f1, brier_score, tp, tn, fp, fn
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    specificity = tn / max(1, (tn + fp))
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec))
    balanced_acc = (rec + specificity) / 2.0

    result = dict(
        accuracy=acc,
        balanced_accuracy=balanced_acc,
        precision=prec,
        recall=rec,
        sensitivity=rec,
        specificity=specificity,
        f1=f1,
        tp=tp, tn=tn, fp=fp, fn=fn,
    )

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        result["brier_score"] = float(np.mean((y_prob - y_true) ** 2))

    return result


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               num_classes: int,
                               y_probs: np.ndarray = None) -> Dict[str, Any]:
    """
    Compute multi-class metrics.

    Args:
        y_true: Ground truth class indices
        y_pred: Predicted class indices
        num_classes: Number of classes
        y_probs: Predicted probability matrix [N, C] (optional, for Brier score)

    Returns:
        Dictionary with confusion_matrix, accuracy, balanced_accuracy,
        f1_macro, f1_weighted, f1_per_class, sensitivity_per_class,
        specificity_per_class, brier_score
    """
    from sklearn.metrics import (
        confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score,
    )

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0, labels=labels))
    f1_weighted = float(f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels))
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels).tolist()

    # Per-class sensitivity (recall) and specificity
    sensitivity_per_class = []
    specificity_per_class = []
    for c in range(num_classes):
        tp_c = cm[c, c]
        fn_c = cm[c, :].sum() - tp_c
        fp_c = cm[:, c].sum() - tp_c
        tn_c = cm.sum() - tp_c - fn_c - fp_c
        sensitivity_per_class.append(float(tp_c / max(1, tp_c + fn_c)))
        specificity_per_class.append(float(tn_c / max(1, tn_c + fp_c)))

    result = dict(
        confusion_matrix=cm,
        accuracy=accuracy,
        balanced_accuracy=balanced_acc,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        f1_per_class=f1_per_class,
        sensitivity_per_class=sensitivity_per_class,
        specificity_per_class=specificity_per_class,
    )

    if y_probs is not None:
        y_probs = np.asarray(y_probs, dtype=float)
        # Multiclass Brier score: mean squared error of probability vectors
        from sklearn.preprocessing import label_binarize
        y_onehot = label_binarize(y_true, classes=labels)
        if num_classes == 2:
            y_onehot = np.column_stack([1 - y_onehot, y_onehot])
        result["brier_score"] = float(np.mean(np.sum((y_probs - y_onehot) ** 2, axis=1)))

    return result


def binconf(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute binary confusion matrix.

    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted labels (0/1)

    Returns:
        2x2 confusion matrix [[tn, fp], [fn, tp]]
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def precision_recall_f1(cm: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 from confusion matrix.

    Args:
        cm: 2x2 confusion matrix [[tn, fp], [fn, tp]]

    Returns:
        Tuple of (precision, recall, f1)
    """
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    def safe_div(a: float, b: float) -> float:
        return float(a / b) if b != 0 else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1
