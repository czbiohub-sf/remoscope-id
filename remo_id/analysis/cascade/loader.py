"""Kind-dispatched prediction loader.

Normalises single-run, CV, and LOO prediction artifacts to a common
DataFrame with columns: ``experiment_id, y_true, y_pred, y_probs_all, split``.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from remo_id.analysis.cascade.models import Model


@dataclass
class ModelPredictions:
    model: Model
    df: pd.DataFrame          # experiment_id, y_true, y_pred, y_probs_all (list), split
    class_names: list[str]
    source_info: str          # human-readable description of where preds came from


def load_model_predictions(model: Model, split: str) -> ModelPredictions:
    """Load predictions for a cascade node based on its ``predictions_kind``.

    After loading, y_pred is recomputed using Youden-J thresholds (margin-
    based ranking) so that cascade routing is consistent with per-node LOO
    report metrics.
    """
    kind = model.predictions_kind
    if kind == "single":
        preds = _load_single(model, split)
    elif kind == "cv":
        preds = _load_cv(model)
    elif kind == "loo":
        preds = _load_loo(model)
    else:
        raise ValueError(f"Unknown predictions_kind={kind!r} for {model.id}")
    _apply_youden_thresholds(preds)
    return preds


def _apply_youden_thresholds(preds: ModelPredictions) -> None:
    """Replace y_pred with Youden-J threshold-based predictions."""
    if "y_probs_all" not in preds.df.columns:
        return
    from remo_id.analysis.optimal_thresholds import find_youden_thresholds

    y_true = preds.df["y_true"].astype(int).to_numpy()
    y_probs = np.asarray(preds.df["y_probs_all"].tolist(), dtype=float)

    youden = find_youden_thresholds(y_true, y_probs, preds.class_names)
    thresholds = np.array([r["optimal_threshold"] for r in youden])
    margins = y_probs - thresholds
    y_pred_thr = margins.argmax(axis=1)

    thr_str = ", ".join(f"{r['class_name']}={r['optimal_threshold']:.3f}" for r in youden)
    print(f"    Youden-J thresholds: {thr_str}")
    preds.df["y_pred"] = y_pred_thr


def _load_single(model: Model, split: str) -> ModelPredictions:
    from remo_id.inference.results import load_predictions

    df, meta = load_predictions(model.checkpoint)
    if "split" in df.columns and split != "all":
        df = df[df["split"] == split].copy()
    if "has_null_label" in df.columns:
        df = df[~df["has_null_label"].astype(bool)].copy()
    df = df.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No predictions in split={split!r} for {model.id}. "
            f"Check split filter and that predictions.npz has a 'split' column."
        )

    prob_cols = sorted(
        [c for c in df.columns if c.startswith("y_probs_class_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    y_probs_all = (
        df[prob_cols].to_numpy(dtype=float).tolist() if prob_cols else None
    )

    class_names = _class_names_from_meta(meta)
    if class_names is None:
        from remo_id.data.class_config import parse_class_config
        class_names = parse_class_config(model.class_config).class_names

    out = pd.DataFrame({
        "experiment_id": df["experiment_id"].astype(str).to_numpy(),
        "y_true": df["y_true"].astype(int).to_numpy(),
        "y_pred": df["y_pred"].astype(int).to_numpy(),
        "split": df["split"] if "split" in df.columns else split,
    })
    if y_probs_all is not None:
        out["y_probs_all"] = y_probs_all

    return ModelPredictions(
        model=model, df=out, class_names=class_names,
        source_info=f"single run at {model.checkpoint}",
    )


def _load_cv(model: Model) -> ModelPredictions:
    from remo_id.workflows.cv import (
        load_cached_cv_predictions, _deduplicate_pooled_predictions,
    )

    cv_dir = model.resolved_predictions_dir()
    _, per_fold, missing = load_cached_cv_predictions(cv_dir, data_module=None)
    if missing:
        raise ValueError(
            f"CV directory {cv_dir} has {len(missing)} folds missing cached "
            f"predictions ({', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}). "
            f"Regenerate CV predictions before running cascade analysis."
        )
    if not per_fold:
        raise ValueError(
            f"No repeat*_fold* directories with predictions found under {cv_dir}."
        )

    y_true, y_probs, y_pred, experiment_ids, _study_ids, _sample_ids = (
        _deduplicate_pooled_predictions(per_fold)
    )

    from remo_id.data.class_config import parse_class_config
    class_names = parse_class_config(model.class_config).class_names

    out = pd.DataFrame({
        "experiment_id": np.asarray(experiment_ids, dtype=str),
        "y_true": np.asarray(y_true, dtype=int),
        "y_pred": np.asarray(y_pred, dtype=int),
        "split": "val",   # CV cached preds are always val
    })
    if y_probs is not None:
        out["y_probs_all"] = list(np.asarray(y_probs, dtype=float))

    return ModelPredictions(
        model=model, df=out, class_names=class_names,
        source_info=f"CV (pooled across {len(per_fold)} folds) at {cv_dir}",
    )


def _load_loo(model: Model) -> ModelPredictions:
    import json as _json
    from remo_id.workflows.loo_predict import load_epoch_predictions

    cv_dir = model.resolved_predictions_dir()
    pred_dir = os.path.join(cv_dir, "predictions")
    if not os.path.isdir(pred_dir):
        raise ValueError(
            f"LOO predictions dir {pred_dir} does not exist. "
            f"Expected structure: {cv_dir}/predictions/epoch_*.npz"
        )
    if not glob.glob(os.path.join(pred_dir, "epoch_*.npz")):
        raise ValueError(f"No epoch_*.npz files found under {pred_dir}.")

    by_epoch = load_epoch_predictions(cv_dir)
    if not by_epoch:
        raise ValueError(f"Loaded zero epochs from {pred_dir}.")

    # Resolve epoch: explicit > best_epoch.json > fallback to max
    chosen_epoch = None
    if model.epoch is not None and model.epoch in by_epoch:
        chosen_epoch = model.epoch
        print(f"    Using explicit epoch {chosen_epoch}")
    else:
        search_dirs = [cv_dir]
        if model.report_dir:
            search_dirs.insert(0, model.report_dir)
        for search_dir in search_dirs:
            best_epoch_path = os.path.join(search_dir, "best_epoch.json")
            if os.path.exists(best_epoch_path):
                with open(best_epoch_path) as f:
                    best_epoch_info = _json.load(f)
                chosen_epoch = best_epoch_info.get("epoch")
                if chosen_epoch is not None and chosen_epoch in by_epoch:
                    print(f"    Using best epoch {chosen_epoch} from {best_epoch_path}")
                    break
                chosen_epoch = None
    if chosen_epoch is None:
        chosen_epoch = max(e for e in by_epoch.keys() if e >= 0)
        print(f"    Warning: no best_epoch.json found, falling back to epoch {chosen_epoch}")
    data = by_epoch[chosen_epoch]

    class_names = data.get("class_names")
    if not class_names:
        from remo_id.data.class_config import parse_class_config
        class_names = parse_class_config(model.class_config).class_names

    y_probs = np.asarray(data["y_probs"], dtype=float)

    out = pd.DataFrame({
        "experiment_id": np.asarray(data["experiment_ids"], dtype=str),
        "y_true": np.asarray(data["y_true"], dtype=int),
        "y_pred": np.asarray(data["y_pred"], dtype=int),
        "split": "val",
    })
    out["y_probs_all"] = list(y_probs)

    return ModelPredictions(
        model=model, df=out, class_names=list(class_names),
        source_info=f"LOO epoch {chosen_epoch} at {cv_dir}",
    )


def _class_names_from_meta(meta):
    if not meta:
        return None
    names = meta.get("class_names")
    if names and isinstance(names, (list, tuple)) and len(names) >= 2:
        return list(names)
    return None
