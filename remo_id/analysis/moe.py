"""Mixture-of-experts (MOE) probability combination for multi-specialist models.

Given a gate model A (binary: condition-of-interest vs not) and a multiclass
expert B operating on the complement, combines per-experiment probabilities
into a unified multiclass output via the chain rule:

    P(gate_positive)          = P_A(positive)
    P(class_i in expert_B)    = P_A(negative) * P_B(class_i)

Assumes conditional independence of A and B given the sample — a strong
assumption that should be validated on cross-domain data.
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd



def combine_moe_probs(
    p_a: np.ndarray,
    p_b: np.ndarray,
    gate_positive_idx: int = 1,
) -> np.ndarray:
    """Combine binary gate probabilities with expert multiclass probabilities.

    Parameters
    ----------
    p_a : (N, 2) array
        Binary gate probabilities. Column ``gate_positive_idx`` is the
        condition-of-interest (e.g. an infection).
    p_b : (N, K) array
        Expert multiclass probabilities (e.g. healthy/control/disease).
    gate_positive_idx : int
        Index of the positive class in ``p_a`` (default 1).

    Returns
    -------
    combined : (N, K+1) array where column 0 is the gate-positive probability
        and columns 1..K are ``P_A(negative) * p_b[:, k]``.
    """
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    if p_a.shape[0] != p_b.shape[0]:
        raise ValueError(f"p_a and p_b must have equal rows, got {p_a.shape[0]} vs {p_b.shape[0]}")
    if p_a.shape[1] != 2:
        raise ValueError(f"p_a must be binary (N,2), got shape {p_a.shape}")

    p_pos = p_a[:, gate_positive_idx]
    p_neg = 1.0 - p_pos
    combined = np.empty((p_a.shape[0], 1 + p_b.shape[1]))
    combined[:, 0] = p_pos
    combined[:, 1:] = p_neg[:, None] * p_b
    return combined


def load_cv_predictions_df(cv_dir: str, use_last: bool = False) -> pd.DataFrame:
    """Load pooled per-experiment predictions from a cv_dir as a DataFrame.

    Returns columns: experiment_id, y_true, y_pred, and y_probs_class_<i>.
    Averages across repeats (same dedup as cv-performance).
    """
    from remo_id.workflows.cv import load_cached_cv_predictions, _deduplicate_pooled_predictions

    pooled, per_fold, missing = load_cached_cv_predictions(
        cv_dir, data_module=None, use_last=use_last,
    )
    if missing:
        raise FileNotFoundError(
            f"{cv_dir}: {len(missing)} folds missing cached predictions"
        )
    y_true, y_probs, y_pred, exp_ids, _, _ = _deduplicate_pooled_predictions(per_fold)

    df = pd.DataFrame({
        "experiment_id": np.asarray(exp_ids).astype(str),
        "y_true": y_true,
        "y_pred": y_pred,
    })
    for k in range(y_probs.shape[1]):
        df[f"y_probs_class_{k}"] = y_probs[:, k]
    return df


def align_moe_predictions(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    num_classes_a: int,
    num_classes_b: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inner-join predictions by experiment_id and return aligned arrays.

    Returns ``(p_a, p_b, experiment_ids)`` with shapes
    ``(M, num_classes_a)``, ``(M, num_classes_b)``, ``(M,)``.
    """
    prob_cols_a = [f"y_probs_class_{k}" for k in range(num_classes_a)]
    prob_cols_b = [f"y_probs_class_{k}" for k in range(num_classes_b)]

    a = df_a[["experiment_id"] + prob_cols_a].rename(
        columns={c: f"a_{c}" for c in prob_cols_a},
    )
    b = df_b[["experiment_id"] + prob_cols_b].rename(
        columns={c: f"b_{c}" for c in prob_cols_b},
    )
    merged = a.merge(b, on="experiment_id", how="inner")
    if merged.empty:
        raise RuntimeError(
            "No experiments overlap between the two cv_dirs (inner join on experiment_id)."
        )

    p_a = merged[[f"a_y_probs_class_{k}" for k in range(num_classes_a)]].to_numpy()
    p_b = merged[[f"b_y_probs_class_{k}" for k in range(num_classes_b)]].to_numpy()
    return p_a, p_b, merged["experiment_id"].to_numpy()


def run_moe(
    cv_dir_a: str,
    cv_dir_b: str,
    expert_b_class_names: list,
    gate_class_name: str,
    output_dir: str,
    bootstrap_n: int = 1000,
    use_last: bool = False,
    y_true_combined: Optional[np.ndarray] = None,
) -> dict:
    """End-to-end MOE: combine Model A (binary gate) + Model B (expert) outputs.

    Produces under ``output_dir``:
      - combined_predictions.csv — per-experiment probabilities for each of the
        (1 + |expert_b|) classes plus the argmax prediction.
      - metrics.json — multiclass metrics if ``y_true_combined`` is provided.

    Parameters
    ----------
    cv_dir_a : Path to binary gate CV directory.
    cv_dir_b : Path to expert multiclass CV directory.
    expert_b_class_names : list of class names in Model B's output, in order.
    gate_class_name : name of the gate-positive class (e.g. "infected").
    y_true_combined : optional (M,) integer array with 4-class labels indexed
        by the inner-join output order (0 = gate class, 1..K = expert_b order).
        If provided, metrics are computed.
    """
    os.makedirs(output_dir, exist_ok=True)

    df_a = load_cv_predictions_df(cv_dir_a, use_last=use_last)
    df_b = load_cv_predictions_df(cv_dir_b, use_last=use_last)
    num_a = sum(1 for c in df_a.columns if c.startswith("y_probs_class_"))
    num_b = sum(1 for c in df_b.columns if c.startswith("y_probs_class_"))
    if num_a != 2:
        raise ValueError(f"Expected binary gate (2 classes); cv_dir_a has {num_a}")

    p_a, p_b, exp_ids = align_moe_predictions(df_a, df_b, num_a, num_b)
    print(f"MOE aligned: {len(exp_ids)} experiments (gate={cv_dir_a}, expert={cv_dir_b})")

    combined = combine_moe_probs(p_a, p_b, gate_positive_idx=1)
    y_pred = combined.argmax(axis=1)

    combined_class_names = [gate_class_name] + list(expert_b_class_names)

    rows = {"experiment_id": exp_ids, "y_pred": y_pred}
    for k, name in enumerate(combined_class_names):
        rows[f"p_{name}"] = combined[:, k]
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "combined_predictions.csv"), index=False,
    )

    result = {
        "n_experiments": int(len(exp_ids)),
        "combined_class_names": combined_class_names,
    }

    if y_true_combined is not None:
        from remo_id.analysis.metrics import compute_multiclass_metrics
        from remo_id.analysis.results import save_metrics

        y_true_combined = np.asarray(y_true_combined)
        if len(y_true_combined) != len(exp_ids):
            raise ValueError(
                f"y_true_combined has {len(y_true_combined)} entries but MOE aligned "
                f"{len(exp_ids)} experiments"
            )
        metrics = compute_multiclass_metrics(
            y_true_combined, y_pred, num_classes=len(combined_class_names), y_probs=combined,
        )
        metrics["accuracy"] = float(metrics["accuracy"])
        metrics["error_rate"] = float((y_pred != y_true_combined).mean())
        save_metrics(os.path.join(output_dir, "metrics.json"), metrics)
        result["metrics"] = metrics
        print(f"MOE multiclass: acc={metrics['accuracy']:.3f}  "
              f"f1_macro={metrics['f1_macro']:.3f}  error={metrics['error_rate']:.3f}")

    print(f"Saved MOE combination to {output_dir}")
    return result
