"""Cascade metrics: per-node (all + cascade-routed subset), leaf-level
confusion + accuracy, per-depth accuracy, error attribution, bootstrap CIs."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from remo_id.analysis.cascade.loader import ModelPredictions
from remo_id.analysis.cascade.tree import CascadeTree
from remo_id.analysis.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
)


# ---------------------------------------------------------------------------
# Per-node metrics
# ---------------------------------------------------------------------------

def per_node_metrics(
    tree: CascadeTree,
    model_preds: dict[str, ModelPredictions],
    pred_walk: pd.DataFrame,
) -> dict[str, dict]:
    """For each node: (a) metrics over all of its predictions; (b) metrics
    restricted to experiments the cascade actually routed to this node
    (via parent y_pred)."""
    results: dict[str, dict] = {}
    for node_id, preds in model_preds.items():
        depth = tree.node_depth[node_id]
        col = f"pred_depth_{depth}_node"
        routed_ids = set()
        if col in pred_walk.columns:
            routed_ids = set(
                pred_walk.loc[pred_walk[col] == node_id, "experiment_id"].astype(str)
            )

        full = _compute_one_node(preds)
        routed = _compute_one_node(preds, subset_eids=routed_ids)
        results[node_id] = {
            "all": full,
            "routed_only": routed,
            "n_all": int(len(preds.df)),
            "n_routed": int(len(routed_ids)),
        }
    return results


def _compute_one_node(
    preds: ModelPredictions, subset_eids: Optional[set[str]] = None
) -> dict:
    df = preds.df
    if subset_eids is not None:
        df = df[df["experiment_id"].astype(str).isin(subset_eids)]
    if df.empty:
        return {"n": 0}
    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()
    n_classes = len(preds.class_names)

    y_probs = None
    if "y_probs_all" in df.columns:
        y_probs = np.asarray(df["y_probs_all"].tolist(), dtype=float)

    if n_classes == 2:
        y_prob_pos = y_probs[:, 1] if y_probs is not None else None
        m = compute_binary_metrics(y_true, y_pred, y_prob_pos)
    else:
        m = compute_multiclass_metrics(y_true, y_pred, n_classes, y_probs)
        if "confusion_matrix" in m:
            m["confusion_matrix"] = m["confusion_matrix"].tolist()
    m["n"] = int(len(df))
    return m


# ---------------------------------------------------------------------------
# Leaf-level metrics
# ---------------------------------------------------------------------------

def leaf_level_metrics(
    pred_walk: pd.DataFrame,
    gt_walk: pd.DataFrame,
    canonical_leaves: list[str],
    bootstrap_n: int = 1000,
    seed: int = 0,
) -> dict:
    """Compute leaf-level metrics over canonical leaves only.

    Cascade-rejected experiments (unresolved/invalid) are excluded from
    metrics and confusion matrix but reported as separate counts.
    """
    merged = pred_walk.merge(gt_walk, on="experiment_id", how="inner")

    n_total = len(merged)
    is_rejected = (
        merged["pred_terminal_leaf"].astype(str).str.startswith("unresolved_at_")
        | merged["pred_terminal_leaf"].astype(str).str.startswith("invalid_class_at_")
    )
    n_rejected = int(is_rejected.sum())
    resolved = merged[~is_rejected]

    leaf_to_idx = {leaf: i for i, leaf in enumerate(canonical_leaves)}

    y_true = resolved["gt_terminal_leaf"].astype(str).map(leaf_to_idx)
    y_pred = resolved["pred_terminal_leaf"].astype(str).map(leaf_to_idx)
    missing = resolved[y_true.isna() | y_pred.isna()]
    if not missing.empty:
        sample = missing[["experiment_id", "gt_terminal_leaf", "pred_terminal_leaf"]].head(5).to_dict("records")
        raise RuntimeError(
            f"{len(missing)} experiment(s) have terminal leaves outside the canonical set. "
            f"Sample: {sample}. Check tree inference and canonical leaf list."
        )
    y_true = y_true.astype(int).to_numpy()
    y_pred = y_pred.astype(int).to_numpy()

    point = compute_multiclass_metrics(y_true, y_pred, num_classes=len(canonical_leaves))
    point["confusion_matrix"] = point["confusion_matrix"].tolist()

    cis = _bootstrap_leaf_ci(y_true, y_pred, len(canonical_leaves), bootstrap_n, seed)

    return {
        "n": int(len(y_true)),
        "n_total": n_total,
        "leaves": list(canonical_leaves),
        "point_estimates": {k: v for k, v in point.items() if k != "confusion_matrix"},
        "confusion_matrix": point["confusion_matrix"],
        "ci_95": cis,
        "cascade_rejection_rate": n_rejected / max(1, n_total),
        "n_cascade_rejected": n_rejected,
    }


def _bootstrap_leaf_ci(
    y_true: np.ndarray, y_pred: np.ndarray,
    num_classes: int, n_iters: int, seed: int,
) -> dict:
    """Bootstrap 95% CIs for accuracy, balanced_accuracy, f1_macro."""
    if n_iters <= 0 or len(y_true) == 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc_samples = np.empty(n_iters)
    bacc_samples = np.empty(n_iters)
    f1_samples = np.empty(n_iters)
    for i in range(n_iters):
        idx = rng.integers(0, n, size=n)
        m = compute_multiclass_metrics(y_true[idx], y_pred[idx], num_classes)
        acc_samples[i] = m["accuracy"]
        bacc_samples[i] = m["balanced_accuracy"]
        f1_samples[i] = m["f1_macro"]

    def pctl(arr):
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {
        "accuracy": pctl(acc_samples),
        "balanced_accuracy": pctl(bacc_samples),
        "f1_macro": pctl(f1_samples),
    }


# ---------------------------------------------------------------------------
# Per-depth accuracy + error attribution
# ---------------------------------------------------------------------------

def per_depth_accuracy(
    pred_walk: pd.DataFrame, gt_walk: pd.DataFrame
) -> pd.DataFrame:
    merged = pred_walk.merge(gt_walk, on="experiment_id", how="inner")
    max_depth = max(
        int(c.split("_")[-2]) for c in gt_walk.columns if c.startswith("gt_depth_")
    ) + 1 if any(c.startswith("gt_depth_") for c in gt_walk.columns) else 0

    rows = []
    for d in range(max_depth):
        gt_col = f"gt_depth_{d}_class"
        pred_col = f"pred_depth_{d}_class"
        if gt_col not in merged.columns:
            continue
        mask = merged[gt_col].notna()
        if pred_col not in merged.columns:
            rows.append({"depth": d, "n": int(mask.sum()), "accuracy": 0.0})
            continue
        gt_vals = merged.loc[mask, gt_col].astype(str)
        pred_vals = merged.loc[mask, pred_col].astype(str).fillna("__missing__")
        n = int(mask.sum())
        acc = float((gt_vals == pred_vals).mean()) if n else float("nan")
        rows.append({"depth": d, "n": n, "accuracy": acc})
    return pd.DataFrame(rows)


def error_attribution(
    pred_walk: pd.DataFrame, gt_walk: pd.DataFrame
) -> pd.DataFrame:
    merged = pred_walk.merge(gt_walk, on="experiment_id", how="inner")
    gt_depths = sorted(
        int(c.split("_")[-2]) for c in gt_walk.columns if c.startswith("gt_depth_") and c.endswith("_class")
    )

    rows = []
    for _, row in merged.iterrows():
        first_error_depth = None
        first_error_node = None
        for d in gt_depths:
            gt_class = row.get(f"gt_depth_{d}_class")
            pred_class = row.get(f"pred_depth_{d}_class")
            gt_node = row.get(f"gt_depth_{d}_node")
            if pd.isna(gt_class):
                break
            if pd.isna(pred_class) or pred_class != gt_class:
                first_error_depth = d
                first_error_node = gt_node
                break
        if first_error_depth is None and row["pred_terminal_leaf"] != row["gt_terminal_leaf"]:
            first_error_depth = int(row.get("gt_terminal_depth", -1))
            first_error_node = "terminal"
        rows.append({
            "experiment_id": row["experiment_id"],
            "gt_leaf": row["gt_terminal_leaf"],
            "pred_leaf": row["pred_terminal_leaf"],
            "correct": row["pred_terminal_leaf"] == row["gt_terminal_leaf"],
            "first_error_depth": first_error_depth,
            "first_error_node": first_error_node,
            "cascade_rejected": row.get("pred_cascade_rejected", False),
        })
    return pd.DataFrame(rows)
