"""Context builders for analysis and reports.

Provides helpers for loading class configs, resolving experiment IDs,
and building the data context needed for analysis figures — either
by running inference (GPU) or loading saved predictions (CPU).
"""

import yaml
import numpy as np


def resolve_split_experiment_ids(splits_data: dict, split: str) -> list:
    """Flatten experiment IDs from a splits YAML for the named split.

    If split is "all", returns IDs from all splits combined.
    """
    if split == "all":
        ids = []
        for split_name, split_data in splits_data.items():
            for v in split_data.values():
                if isinstance(v, list):
                    ids.extend(v)
        return list(dict.fromkeys(str(x) for x in ids))  # dedupe, preserve order

    split_data = splits_data.get(split, {})
    ids = []
    for v in split_data.values():
        if isinstance(v, list):
            ids.extend(v)
    return [str(x) for x in ids]


def load_class_config(class_config_path: str):
    """Parse class config YAML; return (class_definitions, class_names, num_classes,
    task_type, exclude_filters, include_filters, default_class_idx)."""
    from remo_id.data.class_config import parse_class_config
    cfg = parse_class_config(class_config_path)
    default_class_idx = cfg.class_names.index(cfg.default_class) if cfg.default_class in cfg.class_names else None
    return cfg.class_definitions, cfg.class_names, cfg.num_classes, cfg.task_type, cfg.exclude_filters, cfg.include_filters, default_class_idx


def build_analysis_context(checkpoint, class_config_path, splits_path, split_name,
                        bag_size, seed, batch_size,
                        n_top_attention=5, n_bottom_attention=5,
                        sample_mode="random", sample_percent=100.0,
                        capture_embeddings=False):
    """Set up data, model, run inference, and return all arrays needed for figures."""
    import torch
    from remo_id.inference.core import (
        load_checkpoint, extract_checkpoint_metadata, set_inference_seeds,
    )
    from remo_id.inference.mil import _predict_experiments
    from remo_id.data import RemoIDDataModule

    class_definitions, class_names, num_classes, task_type, exclude_filters, include_filters, default_class_idx = (
        load_class_config(class_config_path)
    )

    with open(splits_path) as f:
        splits_data = yaml.safe_load(f)
    experiment_ids = resolve_split_experiment_ids(splits_data, split_name)
    if not experiment_ids:
        raise ValueError(
            f"No experiment IDs found for split '{split_name}' in {splits_path}"
        )

    set_inference_seeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Infer backbone_name from checkpoint
    ckpt_meta = extract_checkpoint_metadata(checkpoint)
    model_name = ckpt_meta.get("model_name", "resnet18")
    from remo_id.models import get_augmentation_backbone
    backbone_name = get_augmentation_backbone(model_name)

    ds = RemoIDDataModule(
        database_column="class_label",
        database_label=None,
        study_id=None,
        threshold=None,
        threshold_mode="==",
        negative_study_ids=[],
        negative_experiment_ids=[],
        include_experiment_ids=experiment_ids,
        batch_size=batch_size,
        backbone_name=backbone_name,
        train=False,
        task_type=task_type,
        num_classes=num_classes,
        class_definitions=class_definitions,
        exclude_filters=exclude_filters,
        include_filters=include_filters,
        default_class_idx=default_class_idx,
        skip_null_labels=True,
        background_correction='none',
    )

    df = ds.data.copy()
    df["experiment_id"] = df["experiment_id"].astype(str)
    df = df[df["experiment_id"].isin(experiment_ids)].copy()
    df = df.sort_values("experiment_id").reset_index(drop=True)
    df["_has_null_label"] = df["path"].isin(ds.null_label_paths)

    if len(df) == 0:
        raise RuntimeError("No matching experiments found.")

    model, device, _in_channels = load_checkpoint(
        checkpoint, model_name,
        task_type=task_type, num_classes=num_classes,
        n_experiments=1, device=device,
    )

    results, _yt, _yp, _ypred, attention_pages_data, _captured = _predict_experiments(
        model=model, ds=ds, df=df, database_column="class_label",
        task_type=task_type, num_classes=num_classes,
        bag_size=bag_size, seed=seed, device=device,
        pred_threshold=0.5,
        sample_mode=sample_mode, sample_percent=sample_percent,
        n_top_attention=n_top_attention, n_bottom_attention=n_bottom_attention,
        class_names=class_names,
        capture_embeddings=capture_embeddings,
    )

    valid = [r for r in results if not np.isnan(r.y_prob) and not r.has_null_label]
    y_true  = np.array([r.y_true       for r in valid], dtype=int)
    y_pred  = np.array([r.y_pred       for r in valid], dtype=int)
    y_probs = np.array([r.y_probs_all  for r in valid], dtype=float)  # [N, C]

    return dict(
        model=model,
        ds=ds,
        df=df,
        results=results,
        valid=valid,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        attention_pages_data=attention_pages_data,
        class_names=class_names,
        num_classes=num_classes,
        task_type=task_type,
        class_definitions=class_definitions,
        device=device,
        database_column="class_label",
    )


def build_context_from_cv_predictions(cv_dir, class_config_path, use_last=False):
    """Build context arrays from pooled CV predictions (no GPU needed).

    Loads cached fold predictions from all folds in a CV directory,
    deduplicates across repeats, and returns the same dict format as
    build_context_from_predictions() plus per_fold and experiment_ids.

    Raises UsageError (via click) if any folds are missing cached predictions.
    """
    import click
    from remo_id.workflows.cv import (
        load_cached_cv_predictions, _deduplicate_pooled_predictions,
    )

    _, class_names, num_classes, task_type, _, _, _ = load_class_config(class_config_path)

    pooled_tuple, per_fold, missing = load_cached_cv_predictions(
        cv_dir, data_module=None, use_last=use_last,
    )

    if missing:
        raise click.UsageError(
            f"Missing cached predictions for {len(missing)} fold(s): "
            f"{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}. "
            f"Run 'remo-id analysis cv-performance --cv_dir {cv_dir} ...' first "
            f"to generate predictions."
        )

    # Get full dedup results including experiment_ids and study_ids
    y_true, y_probs, y_pred, experiment_ids, study_ids, _sample_ids = (
        _deduplicate_pooled_predictions(per_fold)
    )

    return dict(
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        class_names=class_names,
        num_classes=num_classes,
        task_type=task_type,
        class_definitions=None,
        experiment_ids=experiment_ids,
        study_ids=study_ids,
        per_fold=per_fold,
        # Not available from CV predictions (GPU-dependent fields):
        model=None, ds=None, df=None, results=None, valid=None,
        attention_pages_data=None, device=None,
    )


def build_context_from_predictions(checkpoint, split_name):
    """Build context arrays from saved predictions (no GPU needed).

    Returns the same dict keys as build_analysis_context() for array data,
    or None if predictions are not available.
    """
    from remo_id.inference.results import load_predictions

    try:
        df, meta = load_predictions(checkpoint)
    except (FileNotFoundError, Exception):
        return None

    task_type = meta.get("task_type", "binary") if meta else "binary"
    num_classes = meta.get("num_classes", 2) if meta else 2
    class_names = meta.get("class_names") if meta else None

    # Filter by split
    if split_name != "all" and "split" in df.columns:
        mask = df["split"] == split_name
        if mask.sum() > 0:
            df = df[mask].copy()

    # Exclude nulls and NaN
    if "has_null_label" in df.columns:
        df = df[~df["has_null_label"]].copy()
    df = df.dropna(subset=["y_prob"])

    if len(df) == 0:
        return None

    y_true = df["y_true"].to_numpy(dtype=int)
    y_pred = df["y_pred"].to_numpy(dtype=int)
    prob_cols = sorted([c for c in df.columns if c.startswith("y_probs_class_")])
    y_probs = df[prob_cols].to_numpy(dtype=float) if prob_cols else None

    return dict(
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        class_names=class_names,
        num_classes=num_classes,
        task_type=task_type,
        class_definitions=None,
        # Not available from predictions (GPU-dependent fields):
        model=None, ds=None, df=None, results=None, valid=None,
        attention_pages_data=None, device=None,
    )
