"""Post-training analysis module for remo-id.

Runs inference on train+val experiments multiple times to compute:
- Accuracy, precision, recall, F1 (total and val-only)
- Repeatability metrics (prediction variance across runs)
- ROC curves and AUC
- Visualizations logged to W&B
"""

import json
import os

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

from remo_id.inference.core import (
    get_augmentations,
    predict_proba_batch,
    aggregate_image_predictions_binary,
    INFERENCE_SEEDS,
)
from remo_id.analysis.metrics import compute_binary_metrics, compute_multiclass_metrics
from remo_id.reports import figures as plots


def run_post_training_analysis(
    model,
    train_module,
    val_module,
    task_type: str = "binary",
    num_classes: int = 2,
    class_names: Optional[List[str]] = None,
    n_images: int = 200,
    n_runs: int = 5,
    conf_threshold: float = 0.5,
    pos_pct_threshold: float = 0.35,
    device: str = None,
    checkpoint_path: str = None,
    crop_size: int = None,
) -> Dict[str, Any]:
    """
    Run post-training analysis on train and validation experiments.

    Args:
        model: Trained Lightning module
        train_module: RemoIDDataModule for training set
        val_module: RemoIDDataModule for validation set
        task_type: Task type (binary or multiclass)
        num_classes: Number of classes for multiclass
        class_names: List of class names for multiclass
        n_images: Number of images to sample per experiment per run
        n_runs: Number of inference runs per experiment (uses different seeds)
        conf_threshold: Image confidence threshold for instance models
        pos_pct_threshold: Experiment positive if >this fraction of images exceed threshold
        device: Device for inference

    Returns:
        Dictionary with metrics and results for W&B logging
    """
    import wandb

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    # Detect if MIL model
    is_mil = hasattr(model, 'forward_with_attention')
    pooling = getattr(model, "pooling", "attention")
    model_type = f"MIL ({pooling.capitalize()} Pooling)" if is_mil else "Instance"

    print(f"\n{'='*60}", flush=True)
    print("POST-TRAINING ANALYSIS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Model type: {model_type}", flush=True)
    print(f"Images per run: {n_images}", flush=True)
    print(f"Number of runs: {n_runs}", flush=True)
    if not is_mil:
        print(f"Confidence threshold: {conf_threshold}", flush=True)
        print(f"Positive % threshold: {pos_pct_threshold*100:.0f}%", flush=True)

    # Get experiment info with study_id mapping
    train_exp_ids = train_module.data['experiment_id'].tolist()
    val_exp_ids = val_module.data['experiment_id'].tolist()
    all_exp_ids = train_exp_ids + val_exp_ids

    # Build study_id mapping from both modules
    exp_id_to_study_id = {}
    for module in [train_module, val_module]:
        if 'study_id' in module.data.columns:
            for exp_id, study_id in zip(module.data['experiment_id'], module.data['study_id']):
                exp_id_to_study_id[exp_id] = study_id

    print(f"Training experiments: {len(train_exp_ids)}")
    print(f"Validation experiments: {len(val_exp_ids)}")
    print(f"Total experiments: {len(all_exp_ids)}")

    # Set validation augmentations on data loaders for inference
    backbone_name = getattr(model, 'backbone_name', 'resnet18')
    val_augments = get_augmentations(backbone_name, train=False, crop_size=crop_size)
    train_module.dataloader.augments = val_augments
    val_module.dataloader.augments = val_augments

    # Run inference on all experiments
    results = []

    infer_kwargs = dict(
        model=model,
        n_images=n_images,
        n_runs=n_runs,
        conf_threshold=conf_threshold,
        pos_pct_threshold=pos_pct_threshold,
        is_mil=is_mil,
        device=device,
        exp_id_to_study_id=exp_id_to_study_id,
        task_type=task_type,
        num_classes=num_classes,
    )

    # Process training experiments
    print("\nProcessing training experiments...")
    train_results = _run_inference_on_experiments(
        data_module=train_module, exp_ids=train_exp_ids, split="train", **infer_kwargs,
    )
    results.extend(train_results)

    # Process validation experiments
    print("\nProcessing validation experiments...")
    val_results = _run_inference_on_experiments(
        data_module=val_module, exp_ids=val_exp_ids, split="val", **infer_kwargs,
    )
    results.extend(val_results)

    # Compute metrics
    print("\nComputing metrics...")
    if task_type == "multiclass":
        metrics_dict = _compute_all_metrics_multiclass(results, num_classes)
    else:
        metrics_dict = _compute_all_metrics(results, n_runs)

    # Create visualizations using plots module
    print("Creating visualizations...")
    effective_class_names = class_names or (
        [f"Class {i}" for i in range(num_classes)] if task_type == "multiclass" else None
    )

    # Extract arrays from results
    y_true = np.array([r["true_label"] for r in results])
    y_pred = np.array([r["pred_label"] for r in results])
    stds = np.array([r["std_prob"] for r in results])

    figures = {}
    plot_kwargs = dict(model_type=model_type, n_runs=n_runs, n_images=n_images)

    if task_type == "multiclass":
        figures["confusion_matrix"] = plots.plot_confusion_matrix(
            y_true, y_pred, effective_class_names)
    else:
        # Binary-specific plots
        mean_probs = np.array([r["mean_prob"] for r in results])
        std_probs = np.array([r["std_prob"] for r in results])
        splits_list = [r["split"] for r in results]
        study_ids_list = [r["study_id"] for r in results]

        figures["probability_plot"] = plots.plot_experiment_probabilities(
            mean_probs, std_probs, y_true, splits_list, study_ids_list,
            class_names=effective_class_names, **plot_kwargs)

        val_mask = np.array([r["split"] == "val" for r in results])
        figures["roc_curve"] = plots.plot_roc_binary_splits(
            y_true, mean_probs, y_true[val_mask], mean_probs[val_mask],
            **plot_kwargs)

    figures["repeatability_hist"] = plots.plot_repeatability_hist(stds, **plot_kwargs)

    # Prepare W&B logging dict
    if task_type == "multiclass":
        wandb_log = {
            "analysis/val_accuracy": metrics_dict["val"]["accuracy"],
            "analysis/val_f1_macro": metrics_dict["val"]["f1_macro"],
            "analysis/val_f1_weighted": metrics_dict["val"]["f1_weighted"],
            "analysis/total_accuracy": metrics_dict["total"]["accuracy"],
            "analysis/total_f1_macro": metrics_dict["total"]["f1_macro"],
            "analysis/total_f1_weighted": metrics_dict["total"]["f1_weighted"],
            "analysis/mean_repeatability": metrics_dict["repeatability"]["mean_std"],
            "analysis/confusion_matrix": wandb.Image(figures["confusion_matrix"]),
            "analysis/repeatability_hist": wandb.Image(figures["repeatability_hist"]),
            "analysis/experiment_results": _create_results_table(results, task_type=task_type),
        }
        if "per_class_f1" in figures:
            wandb_log["analysis/per_class_f1"] = wandb.Image(figures["per_class_f1"])
    else:
        wandb_log = {
            "analysis/val_accuracy": metrics_dict["val"]["accuracy"],
            "analysis/val_precision": metrics_dict["val"]["precision"],
            "analysis/val_recall": metrics_dict["val"]["recall"],
            "analysis/val_f1": metrics_dict["val"]["f1"],
            "analysis/val_auc": metrics_dict["val"]["auc"],
            "analysis/total_accuracy": metrics_dict["total"]["accuracy"],
            "analysis/total_precision": metrics_dict["total"]["precision"],
            "analysis/total_recall": metrics_dict["total"]["recall"],
            "analysis/total_f1": metrics_dict["total"]["f1"],
            "analysis/total_auc": metrics_dict["total"]["auc"],
            "analysis/mean_repeatability": metrics_dict["repeatability"]["mean_std"],
            "analysis/probability_plot": wandb.Image(figures["probability_plot"]),
            "analysis/roc_curve": wandb.Image(figures["roc_curve"]),
            "analysis/repeatability_hist": wandb.Image(figures["repeatability_hist"]),
            "analysis/experiment_results": _create_results_table(results),
        }

    # Save metrics.json alongside checkpoint
    if checkpoint_path:
        metrics_dir = os.path.join(os.path.dirname(checkpoint_path), "analysis", "performance")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_json = {
            "val": metrics_dict["val"],
            "total": metrics_dict["total"],
            "repeatability": {
                "mean_std": float(metrics_dict["repeatability"]["mean_std"]),
                "per_experiment_stds": [float(s) for s in metrics_dict["repeatability"]["stds"]],
            },
        }
        metrics_path = os.path.join(metrics_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_json, f, indent=2)
        print(f"Saved metrics: {metrics_path}")

    # Log to W&B
    wandb.log(wandb_log)

    # Close figures
    for fig in figures.values():
        plt.close(fig)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    if task_type == "multiclass":
        print(f"Val Accuracy: {metrics_dict['val']['accuracy']:.3f}")
        print(f"Val F1 (macro): {metrics_dict['val']['f1_macro']:.3f}")
        print(f"Total Accuracy: {metrics_dict['total']['accuracy']:.3f}")
        print(f"Total F1 (macro): {metrics_dict['total']['f1_macro']:.3f}")
    else:
        print(f"Val F1: {metrics_dict['val']['f1']:.3f}")
        print(f"Val AUC: {metrics_dict['val']['auc']:.3f}")
        print(f"Total F1: {metrics_dict['total']['f1']:.3f}")
        print(f"Total AUC: {metrics_dict['total']['auc']:.3f}")
    print(f"Mean Repeatability (std): {metrics_dict['repeatability']['mean_std']:.4f}")

    return wandb_log


def _run_inference_on_experiments(
    model,
    data_module,
    exp_ids: List[int],
    n_images: int,
    n_runs: int,
    conf_threshold: float,
    pos_pct_threshold: float,
    is_mil: bool,
    device: str,
    split: str,
    exp_id_to_study_id: Dict = None,
    task_type: str = "binary",
    num_classes: int = 2,
) -> List[Dict]:
    """Run inference on a list of experiments multiple times."""
    results = []
    if exp_id_to_study_id is None:
        exp_id_to_study_id = {}

    # Get experiment ranges and labels
    exp_ranges = data_module.experiment_ranges
    exp_labels = data_module.experiment_labels

    # Map experiment_id to path using the data DataFrame
    exp_id_to_path = dict(zip(
        data_module.data['experiment_id'],
        data_module.data['path']
    ))

    for exp_id in exp_ids:
        path = exp_id_to_path.get(exp_id)
        if path is None:
            print(f"  Warning: No path found for experiment {exp_id}")
            continue

        start_idx, end_idx = exp_ranges[path]
        n_available = end_idx - start_idx
        label = exp_labels[path]

        # Run multiple inference passes
        run_probs = []  # For binary: float prob; for multiclass: (pred_class, max_prob)
        run_preds = []  # For binary instance: per-run threshold-based predictions

        for run_idx in range(n_runs):
            seed = INFERENCE_SEEDS[run_idx % len(INFERENCE_SEEDS)]
            rng = random.Random(seed + hash(str(exp_id)) % 10000)

            n_pick = min(n_images, n_available)
            local_indices = rng.sample(range(n_available), n_pick)
            global_indices = [start_idx + i for i in local_indices]

            images, _, _ = data_module.dataloader.get_items(global_indices)
            images = images.to(device)

            with torch.no_grad():
                if task_type == "multiclass":
                    if is_mil:
                        pred_class, max_prob = _infer_mil_multiclass(model, images)
                    else:
                        pred_class, max_prob = _infer_instance_multiclass(model, images, num_classes)
                    run_probs.append((pred_class, max_prob))
                else:
                    if is_mil:
                        prob = _infer_mil(model, images)
                        run_probs.append(prob)
                    else:
                        prob, exp_pred = _infer_instance(
                            model, images, conf_threshold, pos_pct_threshold
                        )
                        run_probs.append(prob)
                        run_preds.append(exp_pred)

            del images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Aggregate across runs
        if task_type == "multiclass":
            # Majority vote for predicted class
            from collections import Counter
            pred_classes = [r[0] for r in run_probs]
            max_probs = [r[1] for r in run_probs]
            pred_label = Counter(pred_classes).most_common(1)[0][0]
            mean_prob = float(np.mean(max_probs))
            std_prob = float(np.std(max_probs))

            results.append({
                "experiment_id": exp_id,
                "study_id": exp_id_to_study_id.get(exp_id, "unknown"),
                "split": split,
                "true_label": int(label),
                "run_probs": max_probs,
                "mean_prob": mean_prob,
                "std_prob": std_prob,
                "pred_label": pred_label,
            })
        else:
            mean_prob = np.mean(run_probs)
            std_prob = np.std(run_probs)

            # For instance models, use majority vote of threshold-based
            # predictions for consistency with inference pipeline
            if run_preds:
                from collections import Counter
                pred_label = Counter(run_preds).most_common(1)[0][0]
            else:
                pred_label = int(mean_prob > 0.5)

            results.append({
                "experiment_id": exp_id,
                "study_id": exp_id_to_study_id.get(exp_id, "unknown"),
                "split": split,
                "true_label": int(label),
                "run_probs": run_probs,
                "mean_prob": mean_prob,
                "std_prob": std_prob,
                "pred_label": pred_label,
            })

        if len(results) % 5 == 0:
            print(f"  Processed {len(results)} experiments...")

    return results


def _infer_mil(model, images: torch.Tensor) -> float:
    """Run MIL inference on a bag of images as a single experiment."""
    # Temporarily set n_experiments=1 for single-bag inference
    original_n_experiments = model.n_experiments
    model.n_experiments = 1

    try:
        if hasattr(model, 'forward_with_attention'):
            logits, attn, feats = model.forward_with_attention(images)
        else:
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

        # logits is now [1] for a single experiment
        prob = torch.sigmoid(logits).squeeze().item()
    finally:
        # Restore original n_experiments
        model.n_experiments = original_n_experiments

    return prob


def _infer_instance(
    model,
    images: torch.Tensor,
    conf_threshold: float,
    pos_pct_threshold: float,
) -> tuple:
    """Run instance inference and aggregate to experiment prediction.

    Uses aggregate_image_predictions_binary() for consistency with the
    inference pipeline. Returns both the mean probability (for calibration
    analysis) and the threshold-based prediction.

    Returns:
        (mean_prob, exp_pred) tuple
    """
    probs = predict_proba_batch(model, images, task_type="binary", batch_size=32)
    exp_pred, _, _, mean_prob = aggregate_image_predictions_binary(
        list(probs), conf_threshold, pos_pct_threshold * 100,
    )
    return float(mean_prob), int(exp_pred)


def _infer_mil_multiclass(model, images: torch.Tensor) -> Tuple[int, float]:
    """Run MIL inference for multiclass, returning (pred_class, max_prob)."""
    original_n_experiments = model.n_experiments
    model.n_experiments = 1
    try:
        if hasattr(model, 'forward_with_attention'):
            logits, attn, feats = model.forward_with_attention(images)
        else:
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
        probs = torch.softmax(logits.squeeze(0), dim=-1)
        pred_class = int(probs.argmax().item())
        max_prob = float(probs.max().item())
    finally:
        model.n_experiments = original_n_experiments
    return pred_class, max_prob


def _infer_instance_multiclass(model, images: torch.Tensor, num_classes: int) -> Tuple[int, float]:
    """Run instance inference for multiclass, aggregating via majority vote."""
    preds = predict_proba_batch(model, images, task_type="multiclass", batch_size=32)
    # preds is list of (pred_class, max_prob, all_probs)
    from collections import Counter
    pred_classes = [p[0] for p in preds]
    max_probs = [p[1] for p in preds]
    pred_class = Counter(pred_classes).most_common(1)[0][0]
    mean_max_prob = float(np.mean(max_probs))
    return pred_class, mean_max_prob


def _compute_all_metrics_multiclass(results: List[Dict], num_classes: int) -> Dict:
    """Compute multiclass metrics from results."""
    val_results = [r for r in results if r["split"] == "val"]

    def compute_for_subset(subset):
        y_true = np.array([r["true_label"] for r in subset])
        y_pred = np.array([r["pred_label"] for r in subset])
        return compute_multiclass_metrics(y_true, y_pred, num_classes)

    val_metrics = compute_for_subset(val_results)
    total_metrics = compute_for_subset(results)

    stds = [r["std_prob"] for r in results]
    mean_std = np.mean(stds)

    return {
        "val": val_metrics,
        "total": total_metrics,
        "repeatability": {"mean_std": mean_std, "stds": stds},
    }



def _compute_all_metrics(results: List[Dict], n_runs: int) -> Dict:
    """Compute all metrics from results."""
    from sklearn.metrics import roc_curve, auc

    # Split by train/val
    val_results = [r for r in results if r["split"] == "val"]
    all_results = results

    def compute_metrics_for_subset(subset):
        y_true = np.array([r["true_label"] for r in subset])
        y_pred = np.array([r["pred_label"] for r in subset])
        y_prob = np.array([r["mean_prob"] for r in subset])

        metrics = compute_binary_metrics(y_true, y_pred)

        # ROC AUC
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
        else:
            roc_auc = 0.5

        metrics["auc"] = roc_auc
        return metrics

    val_metrics = compute_metrics_for_subset(val_results)
    total_metrics = compute_metrics_for_subset(all_results)

    # Repeatability
    stds = [r["std_prob"] for r in all_results]
    mean_std = np.mean(stds)

    return {
        "val": val_metrics,
        "total": total_metrics,
        "repeatability": {
            "mean_std": mean_std,
            "stds": stds,
        }
    }



def _create_results_table(results: List[Dict], task_type: str = "binary"):
    """Create W&B table with experiment-level results."""
    import wandb
    columns = ["experiment_id", "study_id", "split", "true_label", "mean_prob", "std_prob", "pred_label", "correct"]
    data = []

    for r in results:
        correct = int(r["true_label"] == r["pred_label"])
        data.append([
            r["experiment_id"],
            r["study_id"],
            r["split"],
            r["true_label"],
            round(r["mean_prob"], 4),
            round(r["std_prob"], 4),
            r["pred_label"],
            correct,
        ])

    return wandb.Table(columns=columns, data=data)
