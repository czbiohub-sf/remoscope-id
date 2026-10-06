"""CV workflow orchestration - moved from training/cross_validation.py"""

import glob
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Tuple, Optional
from lightning.pytorch import seed_everything

from remo_id.training.cross_validation import CVConfig, create_cv_splits, create_loo_splits, save_cv_splits, load_cv_splits
from remo_id.training.config import TrainingConfig


def prepare_cv_dir(
    data_module,
    cv_config: CVConfig,
    save_dir: str,
    model_name: str,
    loo: bool = False,
) -> str:
    """Create timestamped CV directory and save splits. Returns the cv_dir path.

    Called from the login node before submitting the array job, so all
    array tasks share the same directory and splits.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if loo:
        cv_dir = os.path.join(save_dir, f"{timestamp}_cv_{model_name}_loo")
    else:
        cv_dir = os.path.join(save_dir, f"{timestamp}_cv_{model_name}_{cv_config.n_folds}fold_{cv_config.n_repeats}repeat")
    os.makedirs(cv_dir, exist_ok=True)

    splits_path = os.path.join(cv_dir, "cv_splits.yaml")
    if loo:
        all_repeats = create_loo_splits(data_module)
        cv_config.n_folds = len(all_repeats[0])
        cv_config.n_repeats = 1
    else:
        all_repeats = create_cv_splits(data_module, cv_config)
    save_cv_splits(all_repeats, data_module, splits_path)

    print(f"CV directory: {cv_dir}")
    print(f"Saved splits: {splits_path}")
    return cv_dir


def run_cross_validation(
    data_module,
    cv_config: CVConfig,
    save_dir: str,
    train_kwargs: dict,
    loo: bool = False,
) -> dict:
    """Run full k-fold cross-validation.

    Trains each fold via run_single_fold(), saving checkpoints and splits.
    W&B: one run group per CV experiment, child run per fold.

    Args:
        data_module: Full RemoIDDataModule (unsplit)
        cv_config: CVConfig with n_folds, n_repeats, group_by, seed
        save_dir: Base directory for fold checkpoints
        train_kwargs: All kwargs to pass through to run_single_fold()
        loo: If True, use leave-one-out splits (one fold per experiment)

    Returns:
        Dict with per-fold results and aggregate metrics
    """
    from remo_id.training.train import run_single_fold
    from remo_id.models import get_model

    model_name = train_kwargs.get("model", "resnet18")
    array_job_id = os.environ.get('SLURM_ARRAY_JOB_ID')
    array_task_id_env = os.environ.get('SLURM_ARRAY_TASK_ID')
    is_array = array_task_id_env is not None
    is_task_zero = is_array and int(array_task_id_env) == 0

    if os.path.exists(os.path.join(save_dir, "cv_splits.yaml")):
        cv_dir = save_dir
    else:
        if array_job_id:
            # Rendezvous file: task 0 creates timestamped dir and writes
            # its path here; other tasks read it.
            rendezvous_file = os.path.join(save_dir, f".cv_dir_{array_job_id}")
            if is_task_zero or not is_array:
                # Task 0 (or non-array): create timestamped directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if loo:
                    cv_dir = os.path.join(save_dir, f"{timestamp}_cv_{model_name}_loo")
                else:
                    cv_dir = os.path.join(
                        save_dir,
                        f"{timestamp}_cv_{model_name}_{cv_config.n_folds}fold_{cv_config.n_repeats}repeat",
                    )
                os.makedirs(cv_dir, exist_ok=True)
                with open(rendezvous_file, 'w') as f:
                    f.write(cv_dir)
            else:
                # Non-zero tasks: wait for task 0's rendezvous file
                import time
                for _ in range(120):
                    if os.path.exists(rendezvous_file):
                        break
                    time.sleep(0.5)
                if not os.path.exists(rendezvous_file):
                    raise RuntimeError(
                        f"Timed out waiting for {rendezvous_file} — task 0 may have failed"
                    )
                with open(rendezvous_file) as f:
                    cv_dir = f.read().strip()
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if loo:
                cv_dir = os.path.join(save_dir, f"{timestamp}_cv_{model_name}_loo")
            else:
                cv_dir = os.path.join(
                    save_dir,
                    f"{timestamp}_cv_{model_name}_{cv_config.n_folds}fold_{cv_config.n_repeats}repeat",
                )
    os.makedirs(cv_dir, exist_ok=True)
    save_dir = cv_dir

    # Create and save splits.
    # For array jobs, task 0 creates the splits file; other tasks poll until
    # it appears (max ~60 s, which covers DB + split generation time).
    splits_path = os.path.join(save_dir, "cv_splits.yaml")

    if os.path.exists(splits_path):
        print(f"Loading existing CV splits from {splits_path}")
        all_repeats = load_cv_splits(splits_path, data_module)
    elif not is_array or is_task_zero:
        # Single job or task 0: create splits
        if loo:
            all_repeats = create_loo_splits(data_module)
            cv_config.n_folds = len(all_repeats[0])
            cv_config.n_repeats = 1
        else:
            all_repeats = create_cv_splits(data_module, cv_config)
        save_cv_splits(all_repeats, data_module, splits_path)
    else:
        # Non-zero array tasks: wait for task 0 to write the splits file
        import time
        for _wait in range(120):
            if os.path.exists(splits_path):
                break
            time.sleep(0.5)
        if not os.path.exists(splits_path):
            raise RuntimeError(
                f"Timed out waiting for {splits_path} — task 0 may have failed"
            )
        print(f"Loading CV splits from {splits_path} (created by task 0)")
        all_repeats = load_cv_splits(splits_path, data_module)

    # Build TrainingConfig from train_kwargs
    cv_config_obj = TrainingConfig(
        model_name=model_name,
        task_type=train_kwargs.get("task_type", "binary"),
        num_classes=train_kwargs.get("num_classes", 2),
        batch_size=train_kwargs.get("batch_size", 200),
        n_experiments=train_kwargs.get("n_experiments", 8),
        num_batches=train_kwargs.get("num_batches", 100),
        cross_experiment_sampling=train_kwargs.get("cross_experiment_sampling", False),
        sample_aware_sampling=train_kwargs.get("sample_aware_sampling", False),
        sample_aware_mode=train_kwargs.get("sample_aware_mode", "single"),
        max_experiments_per_bag=train_kwargs.get("max_experiments_per_bag"),
        majority_min_pct=train_kwargs.get("majority_min_pct", 1.0),
        bag_label_mode=train_kwargs.get("bag_label_mode", "hard"),
        attn_entropy_weight=train_kwargs.get("attn_entropy_weight", 0.0),
        lr=train_kwargs.get("lr", 1e-4),
        lr_schedule=train_kwargs.get("lr_schedule", "cosine"),
        warmup_epochs=train_kwargs.get("warmup_epochs", 5),
        max_epochs=train_kwargs.get("max_epochs", 100),
        min_epochs=train_kwargs.get("min_epochs"),
        weight_decay=train_kwargs.get("weight_decay", 0.01),
        attn_dim=train_kwargs.get("attn_dim", 128),
        pretrained=train_kwargs.get("pretrained", True),
        dino_weights=train_kwargs.get("dino_weights"),
        dropout=train_kwargs.get("dropout", 0.3),
        finetune_layers=train_kwargs.get("finetune_layers", "layer3,layer4"),
        num_heads=train_kwargs.get("num_heads", 1),
        projection_dim=train_kwargs.get("projection_dim"),
        pooling=train_kwargs.get("pooling", "attention"),
        classifier_hidden=train_kwargs.get("classifier_hidden"),
        top_k=train_kwargs.get("top_k"),
        transformer_layers=train_kwargs.get("transformer_layers", 2),
        encode_chunk_size=train_kwargs.get("encode_chunk_size"),
        save_dir=save_dir,
        project=train_kwargs.get("project", "remo-id"),
        accelerator=train_kwargs.get("accelerator", "auto"),
        devices=train_kwargs.get("devices", 1),
        seed=cv_config.seed,
        log_every_n_steps=train_kwargs.get("log_every_n_steps", 50),
        early_stopping_patience=train_kwargs.get("early_stopping_patience", 15),
        accumulate_grad_batches=train_kwargs.get("accumulate_grad_batches", 1),
        precision=train_kwargs.get("precision", "auto"),
        compile=train_kwargs.get("compile", False),
        deterministic=train_kwargs.get("deterministic", True),
        database_column=train_kwargs.get("database_column"),
        database_label=train_kwargs.get("database_label"),
        study_id=train_kwargs.get("study_id"),
        negative_study_ids=train_kwargs.get("negative_study_ids", []),
        skip_null_labels=train_kwargs.get("skip_null_labels", True),
        negative_experiment_ids=train_kwargs.get("negative_experiment_ids", []),
        class_definitions=train_kwargs.get("class_definitions"),
        class_config_raw=train_kwargs.get("class_config_raw"),
        manifest=train_kwargs.get("manifest"),
        label_column=train_kwargs.get("label_column", "label"),
        classes=train_kwargs.get("classes"),
        run_infer=train_kwargs.get("run_infer", True),
        background_correction=train_kwargs.get("background_correction", "none"),
        background_value=train_kwargs.get("background_value"),
        intensity_jitter=train_kwargs.get("intensity_jitter"),
        noise_std=train_kwargs.get("noise_std"),
        blur_sigma=train_kwargs.get("blur_sigma"),
        gradient_blur_sigma=train_kwargs.get("gradient_blur_sigma"),
        erase_num=train_kwargs.get("erase_num"),
        erase_scale=train_kwargs.get("erase_scale"),
        min_image_index=train_kwargs.get("min_image_index", 0),
        crop_size=train_kwargs.get("crop_size"),
        save_every_n_epochs=train_kwargs.get("save_every_n_epochs"),
        save_epochs=train_kwargs.get("save_epochs"),
        patch_pool=train_kwargs.get("patch_pool", False),
        # Embedding-MIL fields
        model_version=train_kwargs.get("model_version"),
        cell_config=train_kwargs.get("cell_config"),
        cells_per_bag=train_kwargs.get("cells_per_bag"),
        load_workers=train_kwargs.get("load_workers"),
        pool_cap=train_kwargs.get("pool_cap"),
    )

    # Embedding-MIL resolves its feature dim from the loaded store, and counts cells (not
    # images): cells_per_bag * n_experiments must be the batch size (run_training does this
    # for the single-split path; mirror it here so batch_size stays divisible).
    if model_name == "embedding-mil":
        cv_config_obj.embedding_dim = getattr(data_module, "embedding_dim", None)
        if cv_config_obj.cells_per_bag:
            cv_config_obj.batch_size = cv_config_obj.cells_per_bag * cv_config_obj.n_experiments

    task_type = cv_config_obj.task_type
    num_classes = cv_config_obj.num_classes
    batch_size = cv_config_obj.batch_size
    n_experiments = cv_config_obj.n_experiments
    n_pos = cv_config_obj.n_pos

    from remo_id.training.train import build_model_kwargs_from_config
    model_kwargs = build_model_kwargs_from_config(cv_config_obj)

    # Auto-detect SLURM array job context
    # Round-robin distribution: each task runs folds where global_idx % task_count == task_id
    # - 1 task (--array=0-0): all folds sequential
    # - N tasks (--array=0-(N-1)): folds balanced across N GPUs
    # - total tasks (--array=0-(total-1)): 1 fold per GPU (legacy behavior)
    array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    array_task_count = os.environ.get('SLURM_ARRAY_TASK_COUNT')
    if array_task_id is not None:
        task_id = int(array_task_id)
        task_count = int(array_task_count) if array_task_count else 1
        total = cv_config.n_folds * cv_config.n_repeats
        my_count = sum(1 for i in range(total) if i % task_count == task_id)
        print(f"SLURM array task {task_id}/{task_count}: training {my_count}/{total} folds")
    else:
        task_id = None
        task_count = None

    fold_results = []
    fold_global_idx = 0

    for repeat_idx, folds in enumerate(all_repeats):
        for fold_idx, (train_paths, val_paths) in enumerate(folds):
            if task_id is not None and fold_global_idx % task_count != task_id:
                fold_global_idx += 1
                continue
            fold_global_idx += 1
            fold_label = f"repeat{repeat_idx}_fold{fold_idx}"
            fold_save_dir = os.path.join(save_dir, fold_label)

            # A fold is complete when predictions.npz has been written (end of fold).
            # last.ckpt alone means the job was preempted mid-training — resume from it.
            predictions_file = os.path.join(fold_save_dir, "predictions.npz")
            last_ckpt = os.path.join(fold_save_dir, "last.ckpt")

            if os.path.exists(predictions_file):
                existing_ckpts = glob.glob(os.path.join(fold_save_dir, "*.ckpt"))
                ckpt = existing_ckpts[0] if existing_ckpts else predictions_file
                print(f"Skipping {fold_label}: already complete")
                fold_results.append({
                    "repeat": repeat_idx,
                    "fold": fold_idx,
                    "checkpoint": ckpt,
                    "skipped": True,
                })
                continue

            ckpt_path = last_ckpt if os.path.exists(last_ckpt) else None
            if ckpt_path:
                print(f"Resuming {fold_label} from {last_ckpt}")

            print(f"\n{'='*60}")
            print(f"Training {fold_label} ({len(train_paths)} train, {len(val_paths)} val)")
            print(f"{'='*60}")

            seed_everything(cv_config.seed + repeat_idx * 1000 + fold_idx, workers=True)

            train_module, val_module = data_module.split_from_paths(train_paths, val_paths)

            model_instance = get_model(model_name, **model_kwargs)
            model_cls = model_instance.__class__

            try:
                fold_config = TrainingConfig(
                    **{k: v for k, v in cv_config_obj.__dict__.items()},
                )
                fold_config.seed = cv_config.seed + repeat_idx * 1000 + fold_idx
                fold_config.save_dir = save_dir

                model_instance, trainer, _, _ = run_single_fold(
                    config=fold_config,
                    model_instance=model_instance,
                    model_cls=model_cls,
                    model_kwargs=model_kwargs,
                    train_module=train_module,
                    val_module=val_module,
                    fold_label=fold_label,
                    ckpt_path=ckpt_path,
                )

                from lightning.pytorch.callbacks import ModelCheckpoint
                ckpt_callback = next(c for c in trainer.callbacks if isinstance(c, ModelCheckpoint))
                best_ckpt = ckpt_callback.best_model_path
                best_score = ckpt_callback.best_model_score
                fold_results.append({
                    "repeat": repeat_idx,
                    "fold": fold_idx,
                    "checkpoint": best_ckpt,
                    "best_val_loss": float(best_score) if best_score is not None else None,
                    "skipped": False,
                })
            except Exception as e:
                print(f"FAILED {fold_label}: {e}")
                import traceback
                traceback.print_exc()
                fold_results.append({
                    "repeat": repeat_idx,
                    "fold": fold_idx,
                    "checkpoint": None,
                    "error": str(e),
                    "skipped": False,
                })

    # Save fold results summary (per-task file in array mode to avoid write conflicts)
    results_df = pd.DataFrame(fold_results)
    if array_task_id is not None:
        results_path = os.path.join(save_dir, f"cv_results_task{task_id}.csv")
    else:
        results_path = os.path.join(save_dir, "cv_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nCV results saved to {results_path}")

    return {"fold_results": fold_results, "results_df": results_df}


def _fold_predictions_path(fold_dir: str, use_last: bool = False) -> str:
    """Path for cached fold predictions npz file."""
    name = "predictions_last.npz" if use_last else "predictions.npz"
    return os.path.join(fold_dir, name)


def _save_fold_predictions(fold_dir: str, repeat_idx: int, fold_idx: int,
                           y_true: np.ndarray, y_probs: np.ndarray, y_pred: np.ndarray,
                           use_last: bool = False,
                           experiment_ids: np.ndarray = None):
    """Save fold predictions to disk as npz."""
    path = _fold_predictions_path(fold_dir, use_last=use_last)
    save_kwargs = dict(y_true=y_true, y_probs=y_probs, y_pred=y_pred,
                       repeat=repeat_idx, fold=fold_idx)
    if experiment_ids is not None:
        save_kwargs["experiment_ids"] = experiment_ids
    np.savez(path, **save_kwargs)
    print(f"Saved predictions to {path}")


def _load_fold_predictions(fold_dir: str, use_last: bool = False) -> Optional[dict]:
    """Load cached fold predictions from disk. Returns None if not found.

    Loads results-format predictions.npz from save_predictions() and filters to val split.
    """
    # Parse repeat/fold from directory name (e.g. "repeat0_fold3")
    fold_label = os.path.basename(fold_dir)
    try:
        parts = fold_label.replace("repeat", "").replace("fold", " ").split("_")
        repeat_idx = int(parts[0])
        fold_idx = int(parts[1].strip())
    except (ValueError, IndexError):
        return None

    # Find a checkpoint in this fold dir to use as the path key
    import glob as glob_module
    if use_last:
        ckpt_candidates = [os.path.join(fold_dir, "last.ckpt")]
    else:
        all_ckpts = sorted(glob_module.glob(os.path.join(fold_dir, "*.ckpt")))
        ckpt_candidates = [c for c in all_ckpts if "last" not in os.path.basename(c)]
        if not ckpt_candidates:
            ckpt_candidates = [c for c in all_ckpts if "last" in os.path.basename(c)]
    if not ckpt_candidates:
        return None

    ckpt_path = ckpt_candidates[0]
    try:
        from remo_id.inference.results import load_predictions
        df, _ = load_predictions(ckpt_path)
    except FileNotFoundError:
        return None

    # Filter to val split, exclude null labels
    mask = (df["has_null_label"] == False)  # noqa: E712
    if "split" in df.columns:
        mask = mask & (df["split"] == "val")
    df = df[mask]

    if df.empty:
        return None

    y_true = df["y_true"].values.astype(int)
    y_pred = df["y_pred"].values.astype(int)
    experiment_ids = df["experiment_id"].values
    study_ids = df["study_id"].values if "study_id" in df.columns else None
    sample_ids = df["sample_id"].values if "sample_id" in df.columns else None

    # Build 2D y_probs from y_probs_class_* columns
    prob_cols = sorted([c for c in df.columns if c.startswith("y_probs_class_")])
    if prob_cols:
        y_probs = df[prob_cols].values.astype(float)
    else:
        # Binary fallback: use y_prob column
        y_probs = np.column_stack([1 - df["y_prob"].values, df["y_prob"].values])

    result = {
        "repeat": repeat_idx,
        "fold": fold_idx,
        "y_true": y_true,
        "y_probs": y_probs,
        "y_pred": y_pred,
        "experiment_ids": experiment_ids,
    }
    if study_ids is not None:
        result["study_ids"] = study_ids
    if sample_ids is not None:
        result["sample_ids"] = sample_ids
    return result


def _predict_fold(save_dir: str, data_module, class_names: list,
                  repeat_idx: int, fold_idx: int,
                  train_paths: set, val_paths: set,
                  bag_size: int, seed: int, n_bags: int = 1,
                  use_last: bool = False):
    """Run inference for a single fold and save predictions to disk."""
    from remo_id.inference.core import load_checkpoint, extract_checkpoint_metadata, set_inference_seeds
    from remo_id.inference.mil import _predict_experiments
    import glob

    fold_label = f"repeat{repeat_idx}_fold{fold_idx}"
    fold_dir = os.path.join(save_dir, fold_label)

    # Check for cached predictions
    cached = _load_fold_predictions(fold_dir, use_last=use_last)
    if cached is not None:
        print(f"Loaded cached predictions for {fold_label}")
        return cached

    # Find checkpoint
    if use_last:
        ckpt_path = os.path.join(fold_dir, "last.ckpt")
        if not os.path.exists(ckpt_path):
            print(f"Warning: no last.ckpt found for {fold_label}")
            return None
    else:
        all_ckpts = sorted(glob.glob(os.path.join(fold_dir, "*.ckpt")))
        ckpts = [c for c in all_ckpts if "last" not in os.path.basename(c)]
        if not ckpts:
            ckpts = [c for c in all_ckpts if "last" in os.path.basename(c)]
        if not ckpts:
            print(f"Warning: no checkpoint found for {fold_label}")
            return None
        ckpt_path = ckpts[0]

    task_type = data_module.task_type
    num_classes = data_module.num_classes

    print(f"Predicting {fold_label} ({len(val_paths)} val experiments)...")

    train_paths_set = set(data_module.data["path"].values) - val_paths
    _, val_module = data_module.split_from_paths(train_paths_set, val_paths)

    ckpt_meta = extract_checkpoint_metadata(ckpt_path)
    model_name = ckpt_meta.get("model_name", "resnet18")

    set_inference_seeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, device, _ = load_checkpoint(
        ckpt_path, model_name,
        task_type=task_type, num_classes=num_classes,
        n_experiments=1, device=device,
    )

    df = val_module.data.copy()
    df["experiment_id"] = df["experiment_id"].astype(str)

    results, yt, yp, ypred, _, _ = _predict_experiments(
        model=model, ds=val_module, df=df,
        database_column=val_module.database_column,
        task_type=task_type, num_classes=num_classes,
        bag_size=bag_size, seed=seed, device=device,
        pred_threshold=0.5,
        class_names=class_names,
        n_bags=n_bags,
    )

    valid = [r for r in results if not np.isnan(r.y_prob) and not r.has_null_label]
    if not valid:
        return None

    # Save predictions in the unified results format
    from remo_id.inference.results import save_predictions
    save_predictions(
        checkpoint_path=ckpt_path,
        results=results,
        metadata={"task_type": task_type, "num_classes": num_classes},
    )

    fold_yt = np.array([r.y_true for r in valid], dtype=int)
    fold_yp = np.array([r.y_probs_all for r in valid], dtype=float)
    fold_ypred = np.array([r.y_pred for r in valid], dtype=int)
    fold_exp_ids = np.array([r.experiment_id for r in valid])
    fold_sample_ids = np.array([r.sample_id for r in valid])

    result = {
        "repeat": repeat_idx,
        "fold": fold_idx,
        "y_true": fold_yt,
        "y_probs": fold_yp,
        "y_pred": fold_ypred,
        "experiment_ids": fold_exp_ids,
        "sample_ids": fold_sample_ids,
    }
    return result


def _deduplicate_pooled_predictions(
    per_fold: list,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Deduplicate pooled CV predictions by averaging across repeats.

    When multiple repeats are used, the same experiment appears once per repeat
    (in different folds). Naive concatenation inflates effective N and produces
    artificially narrow bootstrap CIs. This function averages each experiment's
    predicted probabilities across repeats and derives y_pred from the averaged
    probabilities.

    Falls back to simple concatenation if experiment_ids are not available
    (backward compatibility with old cached predictions).

    Returns:
        (y_true, y_probs, y_pred, experiment_ids, study_ids) arrays with one
        entry per unique experiment. study_ids is None if not available.
    """
    from collections import defaultdict

    # Verify all folds have experiment_ids
    missing = [i for i, f in enumerate(per_fold) if "experiment_ids" not in f]
    if missing:
        raise ValueError(
            f"Folds {missing} lack experiment_ids. "
            "Re-run CV predictions to generate them."
        )

    # Group predictions by experiment_id
    exp_data = defaultdict(lambda: {"y_true": None, "y_probs_list": [], "study_id": None, "sample_id": None})
    for f in per_fold:
        has_study_ids = "study_ids" in f and f["study_ids"] is not None
        has_sample_ids = "sample_ids" in f and f["sample_ids"] is not None
        for i, eid in enumerate(f["experiment_ids"]):
            eid_str = str(eid)
            exp_data[eid_str]["y_true"] = int(f["y_true"][i])
            exp_data[eid_str]["y_probs_list"].append(f["y_probs"][i])
            if has_study_ids:
                exp_data[eid_str]["study_id"] = f["study_ids"][i]
            if has_sample_ids:
                exp_data[eid_str]["sample_id"] = str(f["sample_ids"][i])

    # Average probabilities across repeats
    y_true_list, y_probs_list, eid_list, sid_list, sample_id_list = [], [], [], [], []
    has_any_study_id = any(d["study_id"] is not None for d in exp_data.values())
    has_any_sample_id = any(d["sample_id"] is not None for d in exp_data.values())
    for eid in sorted(exp_data.keys()):
        d = exp_data[eid]
        y_true_list.append(d["y_true"])
        y_probs_list.append(np.mean(d["y_probs_list"], axis=0))
        eid_list.append(eid)
        if has_any_study_id:
            sid_list.append(d["study_id"])
        if has_any_sample_id:
            sample_id_list.append(d["sample_id"])

    y_true = np.array(y_true_list, dtype=int)
    y_probs = np.array(y_probs_list, dtype=float)
    experiment_ids = np.array(eid_list)
    study_ids = np.array(sid_list) if sid_list else None
    sample_ids = np.array(sample_id_list) if sample_id_list else None

    # Derive y_pred from averaged probabilities
    if y_probs.ndim == 2:
        y_pred = y_probs.argmax(axis=1)
    else:
        y_pred = (y_probs > 0.5).astype(int)

    return y_true, y_probs, y_pred, experiment_ids, study_ids, sample_ids


def aggregate_to_patient_level(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    y_pred: np.ndarray,
    experiment_ids: np.ndarray,
    sample_ids: np.ndarray,
    mode: str = "first",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate experiment-level predictions to patient (sample_id) level.

    Args:
        y_true: True labels per experiment
        y_probs: Predicted probabilities per experiment (N x C)
        y_pred: Predicted classes per experiment
        experiment_ids: Experiment IDs
        sample_ids: Sample (patient) IDs
        mode: Aggregation mode:
            - "first": Keep only the first experiment per patient (earliest experiment_id)
            - "combined": Average probabilities across all experiments per patient

    Returns:
        (y_true, y_probs, y_pred, sample_ids) at patient level
    """
    from collections import defaultdict

    if mode not in ("first", "combined"):
        raise ValueError(f"Unknown patient_level mode: {mode!r}. Use 'first' or 'combined'.")

    # Group by sample_id
    patient_data = defaultdict(lambda: {"y_true": None, "experiment_ids": [], "y_probs_list": []})
    for i, sid in enumerate(sample_ids):
        sid_str = str(sid)
        patient_data[sid_str]["y_true"] = int(y_true[i])
        patient_data[sid_str]["experiment_ids"].append(str(experiment_ids[i]))
        patient_data[sid_str]["y_probs_list"].append(y_probs[i])

    yt_list, yp_list, sid_list = [], [], []

    for sid in sorted(patient_data.keys()):
        d = patient_data[sid]
        yt_list.append(d["y_true"])
        sid_list.append(sid)

        if mode == "first":
            # Keep first experiment (earliest by experiment_id, which is a timestamp)
            sorted_indices = np.argsort(d["experiment_ids"])
            yp_list.append(d["y_probs_list"][sorted_indices[0]])
        else:
            # Average probabilities across replicates
            yp_list.append(np.mean(d["y_probs_list"], axis=0))

    yt_out = np.array(yt_list, dtype=int)
    yp_out = np.array(yp_list, dtype=float)
    sids_out = np.array(sid_list)

    # Derive y_pred from probabilities
    if yp_out.ndim == 2:
        ypred_out = yp_out.argmax(axis=1)
    else:
        ypred_out = (yp_out > 0.5).astype(int)

    return yt_out, yp_out, ypred_out, sids_out


def collect_cv_predictions_by_fold(
    save_dir: str,
    data_module,
    class_names: list,
    bag_size: int = 200,
    seed: int = 403,
    n_bags: int = 1,
    use_last: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]], list]:
    """Load fold checkpoints, predict on held-out folds, return pooled and per-fold arrays.

    SLURM array aware: when SLURM_ARRAY_TASK_ID is set, only processes folds
    assigned to this task (round-robin). Predictions are cached as .npz files
    so aggregation can load them without re-running inference.

    Returns:
        pooled: (y_true, y_probs, y_pred, experiment_ids, sample_ids) covering all experiments
        per_fold: list of dicts with keys: repeat, fold, y_true, y_probs, y_pred
    """
    splits_path = os.path.join(save_dir, "cv_splits.yaml")
    all_repeats = load_cv_splits(splits_path, data_module)

    # SLURM array task distribution (same round-robin as training)
    array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    array_task_count = os.environ.get('SLURM_ARRAY_TASK_COUNT')
    if array_task_id is not None:
        task_id = int(array_task_id)
        task_count = int(array_task_count) if array_task_count else 1
        total = sum(len(folds) for folds in all_repeats)
        my_count = sum(1 for i in range(total) if i % task_count == task_id)
        print(f"SLURM array task {task_id}/{task_count}: predicting {my_count}/{total} folds")
    else:
        task_id = None
        task_count = None

    per_fold = []
    fold_global_idx = 0

    for repeat_idx, folds in enumerate(all_repeats):
        for fold_idx, (train_paths, val_paths) in enumerate(folds):
            if task_id is not None and fold_global_idx % task_count != task_id:
                fold_global_idx += 1
                continue
            fold_global_idx += 1

            result = _predict_fold(
                save_dir, data_module, class_names,
                repeat_idx, fold_idx, train_paths, val_paths,
                bag_size, seed, n_bags=n_bags, use_last=use_last,
            )
            if result is not None:
                per_fold.append(result)

    # Deduplicate across repeats: average each experiment's predictions
    # so that bootstrap CIs reflect the true effective sample size
    y_true, y_probs, y_pred, eids, _sids, sample_ids = _deduplicate_pooled_predictions(per_fold)

    return (y_true, y_probs, y_pred, eids, sample_ids), per_fold


def load_cached_cv_predictions(
    save_dir: str,
    data_module,
    use_last: bool = False,
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]], list, list]:
    """Load all cached fold predictions from disk.

    Returns:
        pooled: (y_true, y_probs, y_pred, experiment_ids, sample_ids) or None if any fold missing
        per_fold: list of dicts for folds that have cached predictions
        missing: list of fold labels that are missing predictions
    """
    import glob as glob_module

    fold_dirs = sorted(glob_module.glob(os.path.join(save_dir, "repeat*_fold*")))

    per_fold = []
    missing = []

    for fold_dir in fold_dirs:
        fold_label = os.path.basename(fold_dir)
        cached = _load_fold_predictions(fold_dir, use_last=use_last)
        if cached is not None:
            per_fold.append(cached)
        else:
            missing.append(fold_label)

    if missing:
        return None, per_fold, missing

    # Deduplicate across repeats: average each experiment's predictions
    # so that bootstrap CIs reflect the true effective sample size
    y_true, y_probs, y_pred, eids, _sids, sample_ids = _deduplicate_pooled_predictions(per_fold)
    return (y_true, y_probs, y_pred, eids, sample_ids), per_fold, missing


def aggregate_cv_results(
    per_fold: list,
    pooled: Tuple[np.ndarray, np.ndarray, np.ndarray],
    class_names: list,
    bootstrap_n: int = 1000,
    task_type: str = "multiclass",
) -> dict:
    """Aggregate per-fold metrics and pooled metrics with bootstrap CIs.

    Args:
        per_fold: list of dicts from collect_cv_predictions_by_fold
        pooled: (y_true, y_probs, y_pred) tuple
        class_names: class name strings
        bootstrap_n: bootstrap iterations for CIs
        task_type: 'binary' or 'multiclass'

    Returns:
        Dict with:
            per_fold_metrics: {metric_name: [values]} for plot_kfold_variability
            pooled_metrics_df: DataFrame from metrics_table_with_ci
            y_true, y_probs, y_pred: pooled arrays
    """
    from remo_id.reports.figures import metrics_table_with_ci
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, precision_score, recall_score,
    )
    from sklearn.preprocessing import label_binarize

    num_classes = len(class_names)
    y_true_pooled, y_probs_pooled, y_pred_pooled = pooled
    is_binary = task_type == "binary"

    per_fold_metrics = {"AUC": [], "AP": [], "F1": [], "Precision": [], "Recall": []}

    for fold_data in per_fold:
        yt = fold_data["y_true"]
        yp = fold_data["y_probs"]
        ypred = fold_data["y_pred"]

        if is_binary:
            # Standard binary metrics using positive class probability
            yp_pos = yp[:, 1] if yp.ndim == 2 else yp

            try:
                per_fold_metrics["AUC"].append(roc_auc_score(yt, yp_pos))
            except Exception:
                per_fold_metrics["AUC"].append(np.nan)

            try:
                per_fold_metrics["AP"].append(average_precision_score(yt, yp_pos))
            except Exception:
                per_fold_metrics["AP"].append(np.nan)

            try:
                per_fold_metrics["F1"].append(
                    f1_score(yt, ypred, pos_label=1, zero_division=0))
            except Exception:
                per_fold_metrics["F1"].append(np.nan)

            try:
                per_fold_metrics["Precision"].append(
                    precision_score(yt, ypred, pos_label=1, zero_division=0))
            except Exception:
                per_fold_metrics["Precision"].append(np.nan)

            try:
                per_fold_metrics["Recall"].append(
                    recall_score(yt, ypred, pos_label=1, zero_division=0))
            except Exception:
                per_fold_metrics["Recall"].append(np.nan)
        else:
            # Binarize for OvR metrics
            try:
                yb = label_binarize(yt, classes=list(range(num_classes)))
                if num_classes == 2:
                    yb = np.column_stack([1 - yb, yb])
            except Exception:
                yb = None

            # Macro AUC
            try:
                per_fold_metrics["AUC"].append(
                    roc_auc_score(yb, yp, average="macro", multi_class="ovr"))
            except Exception:
                per_fold_metrics["AUC"].append(np.nan)

            # Macro AP
            try:
                per_fold_metrics["AP"].append(
                    average_precision_score(yb, yp, average="macro"))
            except Exception:
                per_fold_metrics["AP"].append(np.nan)

            # Macro F1, Precision, Recall
            try:
                per_fold_metrics["F1"].append(
                    f1_score(yt, ypred, average="macro", zero_division=0))
            except Exception:
                per_fold_metrics["F1"].append(np.nan)

            try:
                per_fold_metrics["Precision"].append(
                    precision_score(yt, ypred, average="macro", zero_division=0))
            except Exception:
                per_fold_metrics["Precision"].append(np.nan)

            try:
                per_fold_metrics["Recall"].append(
                    recall_score(yt, ypred, average="macro", zero_division=0))
            except Exception:
                per_fold_metrics["Recall"].append(np.nan)

    # Pooled metrics with bootstrap CIs
    pooled_metrics_df = metrics_table_with_ci(
        y_true_pooled, y_probs_pooled, class_names, n=bootstrap_n,
        task_type=task_type,
    )

    return {
        "per_fold_metrics": per_fold_metrics,
        "pooled_metrics_df": pooled_metrics_df,
        "y_true": y_true_pooled,
        "y_probs": y_probs_pooled,
        "y_pred": y_pred_pooled,
    }
