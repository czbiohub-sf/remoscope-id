"""Save and load inference prediction results.

Prediction files are saved in the same directory as the checkpoint:
    predictions.npz   — one row per experiment (structured arrays)
    attention.npz     — per-image attention weights (MIL only)
    embeddings.npz    — pooled experiment embeddings
    features.npz      — per-image backbone features
    metadata.json     — run metadata
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ExperimentResult:
    """Per-experiment inference result.

    Unified across instance and MIL models so that save_predictions()
    works identically for both.
    """
    experiment_id: str
    sample_id: str
    study_id: str
    path: str
    y_true: float
    y_prob: float
    y_pred: float
    bag_size_used: int
    bag_logit: float
    attn_max: float
    attn_mean: float
    has_null_label: bool = False
    split: str = ""
    n_bags: int = 1
    # Optional fields populated by analysis
    y_probs_all: Optional[List[float]] = None   # full softmax vector [num_classes]
    feat_emb: Optional[List[float]] = None       # attention-weighted bag embedding [D]
    attn_entropy: Optional[float] = None         # -sum(a * log(a + eps))


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def predictions_path(checkpoint_path: str) -> str:
    return os.path.join(os.path.dirname(checkpoint_path), "predictions.npz")


def attention_path(checkpoint_path: str) -> str:
    return os.path.join(os.path.dirname(checkpoint_path), "attention.npz")


def embeddings_path(checkpoint_path: str) -> str:
    return os.path.join(os.path.dirname(checkpoint_path), "embeddings.npz")


def features_path(checkpoint_path: str) -> str:
    return os.path.join(os.path.dirname(checkpoint_path), "features.npz")


def metadata_path(checkpoint_path: str) -> str:
    return os.path.join(os.path.dirname(checkpoint_path), "metadata.json")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_predictions(
    checkpoint_path: str,
    results: List[ExperimentResult],
    metadata: Optional[Dict] = None,
    attention_weights: Optional[Dict[str, np.ndarray]] = None,
    attention_indices: Optional[Dict[str, List[int]]] = None,
    embeddings: Optional[Dict[str, np.ndarray]] = None,
    per_image_features: Optional[Dict[str, np.ndarray]] = None,
) -> str:
    """Save inference results alongside a checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint (files are saved as
            ``predictions.npz``, etc. in the checkpoint directory)
        results: List of ExperimentResult objects.
        metadata: Optional dict with run metadata (task_type, num_classes, etc.).
        attention_weights: Optional dict mapping ``exp_{id}`` to attention
            weight arrays (MIL only).
        attention_indices: Optional dict mapping ``exp_{id}`` to global image
            index lists corresponding to attention_weights.
        embeddings: Optional dict mapping ``exp_{id}`` to embedding vectors.

    Returns:
        Path to the saved predictions NPZ.
    """
    npz_path = predictions_path(checkpoint_path)

    # --- predictions.npz ---
    _save_predictions_npz(npz_path, results)

    # --- metadata.json ---
    if metadata is not None:
        _save_metadata(metadata_path(checkpoint_path), metadata)

    # --- attention.npz (MIL only) ---
    if attention_weights:
        npz_data = {}
        for key, weights in attention_weights.items():
            npz_data[f"{key}_weights"] = np.asarray(weights, dtype=np.float32)
        if attention_indices:
            for key, indices in attention_indices.items():
                npz_data[f"{key}_indices"] = np.asarray(indices, dtype=np.int64)
        np.savez_compressed(attention_path(checkpoint_path), **npz_data)
        print(f"Saved attention: {attention_path(checkpoint_path)}")

    # --- embeddings.npz ---
    if embeddings:
        npz_data = {k: np.asarray(v, dtype=np.float32) for k, v in embeddings.items()}
        np.savez_compressed(embeddings_path(checkpoint_path), **npz_data)
        print(f"Saved embeddings: {embeddings_path(checkpoint_path)}")

    # --- features.npz ---
    if per_image_features:
        npz_data = {k: np.asarray(v, dtype=np.float32) for k, v in per_image_features.items()}
        np.savez_compressed(features_path(checkpoint_path), **npz_data)
        print(f"Saved features: {features_path(checkpoint_path)}")

    return npz_path


def _save_predictions_npz(npz_path: str, results: List[ExperimentResult]):
    """Write predictions NPZ with structured arrays."""
    if not results:
        print(f"Warning: no results to save")
        return

    data = {
        "experiment_id": np.array([r.experiment_id for r in results], dtype=str),
        "sample_id": np.array([r.sample_id for r in results], dtype=str),
        "study_id": np.array([r.study_id for r in results], dtype=str),
        "split": np.array([r.split for r in results], dtype=str),
        "has_null_label": np.array([r.has_null_label for r in results], dtype=bool),
        "y_true": np.array([r.y_true for r in results], dtype=np.float64),
        "y_pred": np.array([r.y_pred for r in results], dtype=np.float64),
        "y_prob": np.array([r.y_prob for r in results], dtype=np.float64),
        "bag_size": np.array([r.bag_size_used for r in results], dtype=np.int64),
        "n_bags": np.array([r.n_bags for r in results], dtype=np.int64),
        "bag_logit": np.array([r.bag_logit for r in results], dtype=np.float64),
        "attn_max": np.array([r.attn_max for r in results], dtype=np.float64),
        "attn_mean": np.array([r.attn_mean for r in results], dtype=np.float64),
        "attn_entropy": np.array([
            r.attn_entropy if r.attn_entropy is not None else float("nan")
            for r in results
        ], dtype=np.float64),
    }

    # Build 2D y_probs array [N, num_classes]
    if results[0].y_probs_all is not None:
        data["y_probs"] = np.array([r.y_probs_all for r in results], dtype=np.float64)

    np.savez(npz_path, **data)
    print(f"Saved predictions: {npz_path} ({len(results)} experiments)")


def _save_metadata(json_path: str, metadata: Dict):
    """Write metadata JSON."""
    out = dict(metadata)
    out.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    # Ensure JSON-serializable
    for k, v in out.items():
        if isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved metadata: {json_path}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_predictions(checkpoint_path: str) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """Load saved predictions and metadata for a checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint.

    Returns:
        Tuple of (predictions_df, metadata_dict).
        metadata_dict is None if the metadata file doesn't exist.
    """
    npz_path = predictions_path(checkpoint_path)

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No predictions found at {npz_path}")

    df = _load_predictions_npz(npz_path)

    meta = None
    meta_path = metadata_path(checkpoint_path)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    return df, meta


def _load_predictions_npz(npz_path: str) -> pd.DataFrame:
    """Load predictions from NPZ format into a DataFrame."""
    data = np.load(npz_path, allow_pickle=False)

    df = pd.DataFrame({
        "experiment_id": data["experiment_id"].astype(str),
        "sample_id": data["sample_id"].astype(str),
        "study_id": data["study_id"].astype(str),
        "split": data["split"].astype(str),
        "has_null_label": data["has_null_label"].astype(bool),
        "y_true": data["y_true"],
        "y_pred": data["y_pred"],
        "y_prob": data["y_prob"],
        "bag_size": data["bag_size"],
        "n_bags": data["n_bags"],
        "bag_logit": data["bag_logit"],
        "attn_max": data["attn_max"],
        "attn_mean": data["attn_mean"],
        "attn_entropy": data["attn_entropy"],
    })

    # Reconstruct y_probs_class_* columns from 2D y_probs array
    if "y_probs" in data:
        y_probs = data["y_probs"]
        for i in range(y_probs.shape[1]):
            df[f"y_probs_class_{i}"] = y_probs[:, i]

    return df


def load_attention(checkpoint_path: str) -> Dict[str, np.ndarray]:
    """Load attention weights from NPZ.

    Returns dict with keys like ``exp_123_weights``, ``exp_123_indices``.
    """
    path = attention_path(checkpoint_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No attention file at {path}")
    return dict(np.load(path, allow_pickle=False))


def load_embeddings(checkpoint_path: str) -> Dict[str, np.ndarray]:
    """Load embeddings from NPZ.

    Returns dict with keys like ``exp_123``.
    """
    path = embeddings_path(checkpoint_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No embeddings file at {path}")
    return dict(np.load(path, allow_pickle=False))


def has_saved_predictions(checkpoint_path: str) -> bool:
    """Check if predictions have already been saved for this checkpoint."""
    return os.path.exists(predictions_path(checkpoint_path))


def load_embeddings_with_predictions(
    checkpoint_path: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings matrix and matched predictions for a checkpoint.

    Returns:
        embeddings: (N, D) array, row-aligned with df
        df: DataFrame with experiment_id, y_true, y_pred, split, study_id, etc.
    """
    df, _ = load_predictions(checkpoint_path)
    emb_dict = load_embeddings(checkpoint_path)

    # Match embeddings to predictions by experiment_id
    rows = []
    emb_list = []
    for _, row in df.iterrows():
        key = f"exp_{row['experiment_id']}"
        if key in emb_dict:
            rows.append(row)
            emb_list.append(emb_dict[key])

    if not emb_list:
        raise ValueError(
            f"No matching embeddings found for predictions at {checkpoint_path}"
        )

    matched_df = pd.DataFrame(rows).reset_index(drop=True)
    embeddings = np.stack(emb_list, axis=0)
    return embeddings, matched_df


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def results_to_arrays(
    results: List[ExperimentResult],
    exclude_null: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (y_true, y_pred, y_prob) arrays from results.

    Args:
        results: List of ExperimentResult.
        exclude_null: If True, exclude experiments with has_null_label=True
            and experiments with NaN y_prob.

    Returns:
        (y_true, y_pred, y_prob) numpy arrays.
    """
    filtered = [
        r for r in results
        if not (exclude_null and r.has_null_label)
        and not np.isnan(r.y_prob)
    ]
    y_true = np.array([r.y_true for r in filtered])
    y_pred = np.array([r.y_pred for r in filtered])
    y_prob = np.array([r.y_prob for r in filtered])
    return y_true, y_pred, y_prob


def dataframe_to_results(df: pd.DataFrame) -> List[ExperimentResult]:
    """Reconstruct ExperimentResult objects from a predictions DataFrame.

    Useful for round-trip testing and for code that expects the dataclass.
    """
    results = []
    # Detect per-class probability columns
    prob_cols = sorted([c for c in df.columns if c.startswith("y_probs_class_")])

    for _, row in df.iterrows():
        y_probs_all = None
        if prob_cols:
            probs = [row[c] for c in prob_cols]
            if not all(np.isnan(p) for p in probs):
                y_probs_all = probs

        results.append(ExperimentResult(
            experiment_id=str(row["experiment_id"]),
            sample_id=str(row.get("sample_id", "")),
            study_id=str(row.get("study_id", "")),
            path="",  # not stored in predictions file
            y_true=float(row["y_true"]),
            y_prob=float(row["y_prob"]),
            y_pred=float(row["y_pred"]),
            bag_size_used=int(row.get("bag_size", 0)),
            bag_logit=float(row.get("bag_logit", float("nan"))),
            attn_max=float(row.get("attn_max", float("nan"))),
            attn_mean=float(row.get("attn_mean", float("nan"))),
            has_null_label=bool(row.get("has_null_label", False)),
            split=str(row.get("split", "")),
            n_bags=int(row.get("n_bags", 1)),
            y_probs_all=y_probs_all,
            attn_entropy=float(row.get("attn_entropy", float("nan")))
                if not np.isnan(row.get("attn_entropy", float("nan"))) else None,
        ))
    return results
