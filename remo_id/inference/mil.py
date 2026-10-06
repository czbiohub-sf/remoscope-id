"""MIL (Multiple Instance Learning) inference for remo-id."""

import os
import json
import random
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from remo_id.data import RemoIDDataModule
from remo_id.inference.core import load_checkpoint, parse_scalar, set_inference_seeds, resolve_experiment_ids, extract_checkpoint_metadata
from remo_id.inference.results import ExperimentResult
from remo_id.analysis.metrics import (
    compute_multiclass_metrics,
    binconf,
    precision_recall_f1,
)
from remo_id.reports.pdf import (
    fig_to_png_bytes,
    pdf_title,
    pdf_table,
    add_png_page,
)
from remo_id.interpret import (
    base_rgb_from_tensor,
    make_instance_overlay,
)
from remo_id.reports.pdf.summary import (
    build_summary_first_page,
    make_null_predictions_table,
    make_confusion_matrix_heatmap,
    make_per_class_metrics_table,
)


# ExperimentResult is imported from inference.results (canonical location)




def pick_global_indices_for_experiment(
    exp_ranges: Dict[str, Tuple[int, int]],
    experiment_path: str,
    n_images: int,
    seed: int,
    sample_mode: str = "random",
    sample_percent: float = 100.0,
) -> List[int]:
    """Pick global image indices for an experiment.

    Args:
        exp_ranges: Dict mapping experiment paths to (start, end) index ranges
        experiment_path: Path to the experiment
        n_images: Number of images to sample
        seed: Random seed for reproducible sampling
        sample_mode: Sampling strategy - "random", "first", "last", or "start"
        sample_percent: Percentage of experiment to sample from (e.g., 50.0 = first/last 50%)

    Returns:
        List of global indices to sample
    """
    start, end = exp_ranges[experiment_path]
    n_avail = max(0, end - start)

    # Apply sample_percent to restrict the sampling region
    if sample_mode == "start":
        # Strictly sequential: first n_images images, no randomness
        n_pick = min(int(n_images), n_avail)
        if n_pick <= 0:
            return []
        return [start + i for i in range(n_pick)]
    elif sample_mode == "first":
        # Sample from first x% of images
        region_size = max(1, int(n_avail * sample_percent / 100.0))
        region_start = 0
        region_end = region_size
    elif sample_mode == "last":
        # Sample from last x% of images
        region_size = max(1, int(n_avail * sample_percent / 100.0))
        region_start = n_avail - region_size
        region_end = n_avail
    else:  # "random" - sample from entire experiment
        region_start = 0
        region_end = n_avail

    region_avail = region_end - region_start
    n_pick = min(int(n_images), region_avail)
    if n_pick <= 0:
        return []

    rng = random.Random(int(seed))
    if n_pick < region_avail:
        local = rng.sample(range(region_avail), n_pick)
    else:
        local = list(range(region_avail))

    return [start + region_start + li for li in local]


def _pick_bag_indices(ds, experiment_path, n_images, seed,
                      sample_mode="random", sample_percent=100.0):
    """Pick one experiment's bag indices. Embedding modules expose ``sample_bag_indices``,
    which composes the bag with the SAME cell_config weighting used in training (so eval
    bags match the training distribution); image modules sample uniformly over the range.
    """
    if hasattr(ds, "sample_bag_indices"):
        return ds.sample_bag_indices(experiment_path, n_images, seed=seed)
    return pick_global_indices_for_experiment(
        ds.experiment_ranges, experiment_path, n_images, seed,
        sample_mode=sample_mode, sample_percent=sample_percent,
    )


@torch.no_grad()
def get_instance_overlay(
    model: torch.nn.Module,
    x1: torch.Tensor,
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
    image_weight: float = 0.85,
) -> np.ndarray:
    """
    Get overlay visualization for a single instance.
    Uses unified make_instance_overlay which supports both ViT and ResNet.
    """
    return make_instance_overlay(
        model, x1,
        head_fusion=head_fusion,
        discard_ratio=discard_ratio,
        image_weight=image_weight
    )


def make_top_bottom_attention_grid_png_preselected(
    model: torch.nn.Module,
    x_vis: torch.Tensor,
    attn_vis: np.ndarray,
    original_indices: List[int],
    title: str,
    n_top: int = 5,
    n_bottom: int = 5,
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
    device: str = None,
) -> bytes:
    """
    Create attention grid from pre-selected visualization images.

    Args:
        model: The MIL model
        x_vis: Pre-selected images [n_top + n_bottom, C, H, W]
        attn_vis: Attention values for selected images
        original_indices: Original bag indices (for labeling)
        title: Title for the figure
        n_top: Number of top attention instances
        n_bottom: Number of bottom attention instances
        head_fusion: How to fuse attention heads for ViT
        discard_ratio: Fraction of attention to discard for ViT
        device: Device for processing

    Returns:
        PNG bytes of the attention grid
    """
    n_images = x_vis.shape[0]
    if n_images <= 0:
        fig = plt.figure(figsize=(12, 2))
        plt.text(0.5, 0.5, "No instances to display", ha="center", va="center")
        plt.axis("off")
        return fig_to_png_bytes(fig)

    # Split into top and bottom (already pre-sorted)
    n_top_actual = min(n_top, n_images)
    n_bottom_actual = min(n_bottom, max(0, n_images - n_top))

    # Create overlays for top instances
    top_overlays = []
    for i in range(n_top_actual):
        x1 = x_vis[i:i+1]
        if device is not None:
            x1 = x1.to(device)
        try:
            ov = get_instance_overlay(model, x1, head_fusion=head_fusion, discard_ratio=discard_ratio)
        except Exception:
            base = base_rgb_from_tensor(x1)
            ov = (base * 255).astype(np.uint8)
        orig_idx = original_indices[i] if i < len(original_indices) else i
        top_overlays.append((orig_idx, ov, attn_vis[i]))
        if device is not None:
            del x1
            torch.cuda.empty_cache()

    # Create overlays for bottom instances
    bottom_overlays = []
    for i in range(n_bottom_actual):
        vis_idx = n_top_actual + i
        if vis_idx >= n_images:
            break
        x1 = x_vis[vis_idx:vis_idx+1]
        if device is not None:
            x1 = x1.to(device)
        try:
            ov = get_instance_overlay(model, x1, head_fusion=head_fusion, discard_ratio=discard_ratio)
        except Exception:
            base = base_rgb_from_tensor(x1)
            ov = (base * 255).astype(np.uint8)
        orig_idx = original_indices[vis_idx] if vis_idx < len(original_indices) else vis_idx
        bottom_overlays.append((orig_idx, ov, attn_vis[vis_idx]))
        if device is not None:
            del x1
            torch.cuda.empty_cache()

    # Create figure with GridSpec
    ncols = max(n_top_actual, n_bottom_actual, 1)
    fig = plt.figure(figsize=(ncols * 2.2, 8))
    gs = GridSpec(4, ncols, height_ratios=[0.08, 1, 0.08, 1], hspace=0.25, wspace=0.1)

    # Title
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    # HIGH ATTENTION section header
    ax_header_top = fig.add_subplot(gs[0, :])
    ax_header_top.text(0.5, 0.5, "HIGH ATTENTION", ha="center", va="center",
                       fontsize=11, fontweight="bold", color="darkgreen")
    ax_header_top.axis("off")

    # Top attention images (row 1)
    for i in range(ncols):
        ax = fig.add_subplot(gs[1, i])
        ax.axis("off")
        if i < len(top_overlays):
            idx, ov, a = top_overlays[i]
            ax.imshow(ov)
            ax.set_title(f"TOP-{i+1}\ni={idx}\na={a:.4f}", fontsize=9, color="darkgreen")

    # LOW ATTENTION section header
    ax_header_bot = fig.add_subplot(gs[2, :])
    ax_header_bot.text(0.5, 0.5, "LOW ATTENTION", ha="center", va="center",
                       fontsize=11, fontweight="bold", color="darkred")
    ax_header_bot.axis("off")

    # Bottom attention images (row 3)
    for i in range(ncols):
        ax = fig.add_subplot(gs[3, i])
        ax.axis("off")
        if i < len(bottom_overlays):
            idx, ov, a = bottom_overlays[i]
            ax.imshow(ov)
            ax.set_title(f"BOT-{i+1}\ni={idx}\na={a:.4f}", fontsize=9, color="darkred")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig_to_png_bytes(fig)




def _predict_experiments(
    model,
    ds,
    df: "pd.DataFrame",
    database_column: str,
    task_type: str,
    num_classes: int,
    bag_size: int,
    seed: int,
    device: str,
    pred_threshold: float = 0.5,
    sample_mode: str = "random",
    sample_percent: float = 100.0,
    n_top_attention: int = 5,
    n_bottom_attention: int = 5,
    class_names: Optional[List[str]] = None,
    capture_embeddings: bool = False,
    capture_features: bool = False,
    n_bags: int = 1,
    scan_size: Optional[int] = None,
    top_k: Optional[int] = None,
) -> tuple:
    """Run per-experiment MIL prediction loop.

    Extracted from ``run_mil_inference`` so analysis can reuse it
    with different bag sizes, seeds, and sampling strategies.

    When ``n_bags > 1``, each experiment is predicted with multiple
    independently sampled bags and the probabilities are averaged.
    Attention/visualization data comes from the first bag only.

    Two-stage hard-attention inference (``scan_size`` + ``top_k``):
    When both are set, samples ``scan_size`` images per experiment,
    encodes all and computes attention, selects the ``top_k`` instances
    with highest attention, then re-aggregates on only those instances
    for the final prediction. This focuses the model on the most
    informative instances from a wider scan.

    Returns
    -------
    results : List[ExperimentResult]
    y_true_all, y_prob_all, y_pred_all : lists of scalars
    attention_pages_data : list[dict]  (top/bottom images for PDF pages)
    """
    results: List[ExperimentResult] = []
    y_true_all = []
    y_prob_all = []
    y_pred_all = []
    attention_pages_data = []
    captured_features = {}
    captured_attention_weights = {}
    captured_attention_indices = {}
    max_chunk_size = 64

    for i, row in df.iterrows():
        exp_id = str(row["experiment_id"])
        exp_sample_id = str(row.get("sample_id", ""))
        exp_study_id = str(row.get("study_id", ""))
        exp_path = str(row["path"])
        has_null = bool(row.get("_has_null_label", False))

        if database_column in row:
            y_true = int(row[database_column])
        else:
            raise RuntimeError(f"Missing label column: {database_column}")

        # Two-stage hard-attention: scan wide, select top_k, re-aggregate
        use_two_stage = scan_size is not None and top_k is not None

        bag_logits_list = []
        first_bag_data = None

        if use_two_stage:
            bag_seed = int(seed + 1009 * i)
            set_inference_seeds(bag_seed)

            # Stage 1: sample scan_size images and encode all
            idxs = _pick_bag_indices(
                ds, exp_path, scan_size, bag_seed,
                sample_mode=sample_mode, sample_percent=sample_percent,
            )

            if len(idxs) > 0:
                x, _labels, _exp_ids = ds.dataloader.get_items(idxs)

                with torch.no_grad():
                    # Encode in chunks (scan_size can be large)
                    all_feats = []
                    for chunk_start in range(0, x.shape[0], max_chunk_size):
                        chunk_end = min(chunk_start + max_chunk_size, x.shape[0])
                        x_chunk = x[chunk_start:chunk_end].to(device)
                        all_feats.append(model._encode_instances(x_chunk).cpu())
                        del x_chunk
                        torch.cuda.empty_cache()
                    all_feats = torch.cat(all_feats, dim=0).to(device)
                    if hasattr(model, 'projection'):
                        all_feats = model.projection(all_feats)

                    # Get attention weights for all scan_size instances
                    scan_feats = all_feats.unsqueeze(0)  # [1, scan_size, D]
                    _scan_logits, scan_attn = model._aggregate(scan_feats)
                    if scan_attn is None:
                        scan_attn = torch.ones(1, all_feats.shape[0], device=all_feats.device) / all_feats.shape[0]
                    scan_attn0 = scan_attn[0].detach().cpu().numpy().astype(np.float32)

                    # Stage 2: select top_k instances by attention
                    k = min(top_k, len(scan_attn0))
                    top_indices = np.argsort(-scan_attn0)[:k]

                    selected_feats = all_feats[top_indices].unsqueeze(0)  # [1, top_k, D]
                    bag_logits, attn = model._aggregate(selected_feats)
                    if attn is None:
                        attn = torch.ones(1, k, device=selected_feats.device) / k

                bag_logits_list.append(bag_logits)

                # Map attention back to scan-level indices for visualization
                attn0 = attn[0].detach().cpu().numpy().astype(np.float32)
                # x_vis needs original images — index into x using top_indices
                selected_idxs = [idxs[j] for j in top_indices]
                x_selected = x[top_indices]

                first_bag_data = {
                    "x": x_selected, "attn0": attn0, "feats": selected_feats,
                    "bag_size_used": k,
                    "idxs": selected_idxs,
                    "scan_size_used": len(idxs),
                    "scan_attn": scan_attn0,
                }

        else:
            # Standard: multi-bag prediction
            for bag_idx in range(n_bags):
                bag_seed = int(seed + 1009 * i + 7919 * bag_idx)
                set_inference_seeds(bag_seed)

                idxs = _pick_bag_indices(
                    ds, exp_path, bag_size, bag_seed,
                    sample_mode=sample_mode, sample_percent=sample_percent,
                )

                if len(idxs) == 0:
                    continue

                x, _labels, _exp_ids = ds.dataloader.get_items(idxs)

                with torch.no_grad():
                    if x.shape[0] <= max_chunk_size:
                        x = x.to(device)
                        bag_logits, attn, feats = model.forward_with_attention(x)
                    else:
                        all_feats = []
                        for chunk_start in range(0, x.shape[0], max_chunk_size):
                            chunk_end = min(chunk_start + max_chunk_size, x.shape[0])
                            x_chunk = x[chunk_start:chunk_end].to(device)
                            all_feats.append(model._encode_instances(x_chunk).cpu())
                            del x_chunk
                            torch.cuda.empty_cache()
                        all_feats = torch.cat(all_feats, dim=0).to(device)
                        if hasattr(model, 'projection'):
                            all_feats = model.projection(all_feats)
                        all_feats = all_feats.unsqueeze(0)
                        bag_logits, attn = model._aggregate(all_feats)
                        if attn is None:
                            bag_sz = all_feats.shape[1]
                            attn = torch.ones(1, bag_sz, device=all_feats.device) / bag_sz
                        feats = all_feats

                bag_logits_list.append(bag_logits)

                # Keep attention/vis data from first bag only
                if first_bag_data is None:
                    attn0 = attn[0].detach().cpu().numpy().astype(np.float32)
                    first_bag_data = {
                        "x": x, "attn0": attn0, "feats": feats,
                        "bag_size_used": len(idxs),
                        "idxs": idxs,
                    }

        if not bag_logits_list:
            results.append(ExperimentResult(
                experiment_id=exp_id, sample_id=exp_sample_id, study_id=exp_study_id,
                path=exp_path, y_true=y_true,
                y_prob=float("nan"), y_pred=0, bag_size_used=0,
                bag_logit=float("nan"), attn_max=float("nan"),
                attn_mean=float("nan"), has_null_label=has_null,
            ))
            continue

        # Average logits across bags
        avg_logits = torch.stack(bag_logits_list).mean(dim=0)

        attn0 = first_bag_data["attn0"]

        # Capture per-image features and attention for saving
        if capture_features and first_bag_data.get("feats") is not None:
            captured_features[f"exp_{exp_id}"] = first_bag_data["feats"][0].detach().cpu().numpy().astype(np.float32)
            captured_attention_weights[f"exp_{exp_id}"] = attn0
            captured_attention_indices[f"exp_{exp_id}"] = np.array(first_bag_data["idxs"], dtype=np.int64)

        attn_max = float(np.max(attn0)) if attn0.size else float("nan")
        attn_mean = float(np.mean(attn0)) if attn0.size else float("nan")
        attn_entropy = float(-np.sum(attn0 * np.log(attn0 + 1e-9))) if attn0.size else float("nan")

        if task_type == "binary":
            logit = float(avg_logits[0].item())
            prob = float(torch.sigmoid(avg_logits[0]).item())
            y_pred = int(prob >= pred_threshold)
            y_probs_all = [1.0 - prob, prob]
            pred_display = f"p={prob:.3f}"
            if n_bags > 1:
                pred_display += f" ({n_bags} bags)"
        elif task_type == "multiclass":
            logits_0 = avg_logits[0] if avg_logits.dim() > 1 else avg_logits
            probs_t = torch.softmax(logits_0, dim=-1)
            y_pred = int(probs_t.argmax().item())
            prob = float(probs_t.max().item())
            logit = float(logits_0[y_pred].item())
            y_probs_all = probs_t.cpu().tolist()
            mc_class_names = class_names or [f"Class {i}" for i in range(num_classes)]
            cls_name = mc_class_names[y_pred] if y_pred < len(mc_class_names) else f"Class {y_pred}"
            pred_display = f"{cls_name}, conf={prob:.3f}"
            if n_bags > 1:
                pred_display += f" ({n_bags} bags)"

        x = first_bag_data["x"]
        feats = first_bag_data["feats"]

        sorted_indices = np.argsort(-attn0)
        top_indices = sorted_indices[:n_top_attention].tolist()
        bottom_indices = sorted_indices[-n_bottom_attention:].tolist()
        vis_indices = top_indices + bottom_indices
        x_vis = x[vis_indices].cpu() if len(vis_indices) > 0 else x.cpu()
        attn_vis = attn0[vis_indices] if len(vis_indices) > 0 else attn0

        attention_pages_data.append({
            "exp_id": exp_id, "y_true": y_true, "pred_display": pred_display,
            "x_vis": x_vis, "attn_vis": attn_vis,
            "original_indices": vis_indices, "has_null": has_null,
            "bag_attn_max": float(np.max(attn0)),
            "bag_attn_min": float(np.min(attn0)),
        })

        feat_emb = None
        if capture_embeddings:
            feats_np = feats[0].detach().cpu().numpy().astype(np.float32)  # [bag_size, D]
            w = attn0 / (attn0.sum() + 1e-9)
            feat_emb = (w[:, None] * feats_np).sum(0).tolist()

        res = ExperimentResult(
            experiment_id=exp_id, sample_id=exp_sample_id, study_id=exp_study_id,
            path=exp_path, y_true=y_true,
            y_prob=prob, y_pred=y_pred,
            bag_size_used=first_bag_data["bag_size_used"],
            bag_logit=logit, attn_max=attn_max, attn_mean=attn_mean,
            has_null_label=has_null,
            y_probs_all=y_probs_all,
            feat_emb=feat_emb,
            attn_entropy=attn_entropy,
        )
        results.append(res)
        y_true_all.append(y_true)
        y_prob_all.append(prob)
        y_pred_all.append(y_pred)

        scan_info = ""
        if use_two_stage and first_bag_data.get("scan_size_used"):
            scan_info = f" scan={first_bag_data['scan_size_used']}→top{first_bag_data['bag_size_used']}"
        print(f"[{i+1}/{len(df)}] exp={exp_id} y={y_true} {pred_display}{scan_info} null={has_null}")

    captured_data = None
    if capture_features:
        captured_data = {
            "features": captured_features,
            "attention_weights": captured_attention_weights,
            "attention_indices": captured_attention_indices,
        }

    return results, y_true_all, y_prob_all, y_pred_all, attention_pages_data, captured_data


def _predict_experiments_multi_embed(
    model,
    ds,
    df: "pd.DataFrame",
    database_column: str,
    task_type: str,
    num_classes: int,
    bag_size: int,
    seed: int,
    device: str,
    n_repeats: int = 10,
    class_names: Optional[List[str]] = None,
):
    """Run multiple independent bags per experiment for UMAP embedding variability.

    Returns
    -------
    embeddings : np.ndarray of shape [N, D]
    labels : np.ndarray of shape [N]
    experiment_ids : list[str] of length N
    """
    max_chunk_size = 64
    embeddings = []
    labels = []
    experiment_ids = []

    for i, row in df.iterrows():
        exp_id = str(row["experiment_id"])
        exp_path = str(row["path"])
        has_null = bool(row.get("_has_null_label", False))

        if has_null:
            continue

        if database_column in row:
            y_true = int(row[database_column])
        else:
            continue

        for repeat_idx in range(n_repeats):
            bag_seed = int(seed + 1009 * i + 7919 * repeat_idx)
            set_inference_seeds(bag_seed)

            idxs = _pick_bag_indices(ds, exp_path, bag_size, bag_seed)
            if len(idxs) == 0:
                continue

            x, _labels, _exp_ids = ds.dataloader.get_items(idxs)

            with torch.no_grad():
                if x.shape[0] <= max_chunk_size:
                    x = x.to(device)
                    _logits, attn, feats = model.forward_with_attention(x)
                else:
                    all_feats = []
                    for chunk_start in range(0, x.shape[0], max_chunk_size):
                        chunk_end = min(chunk_start + max_chunk_size, x.shape[0])
                        x_chunk = x[chunk_start:chunk_end].to(device)
                        all_feats.append(model._encode_instances(x_chunk).cpu())
                        del x_chunk
                        torch.cuda.empty_cache()
                    all_feats = torch.cat(all_feats, dim=0).to(device)
                    if hasattr(model, 'projection'):
                        all_feats = model.projection(all_feats)
                    all_feats = all_feats.unsqueeze(0)
                    _logits, attn = model._aggregate(all_feats)
                    if attn is None:
                        bag_sz = all_feats.shape[1]
                        attn = torch.ones(1, bag_sz, device=all_feats.device) / bag_sz
                    feats = all_feats

            attn0 = attn[0].detach().cpu().numpy().astype(np.float32)
            feats_np = feats[0].detach().cpu().numpy().astype(np.float32)
            w = attn0 / (attn0.sum() + 1e-9)
            emb = (w[:, None] * feats_np).sum(0)

            embeddings.append(emb)
            labels.append(y_true)
            experiment_ids.append(exp_id)

        print(f"[{i+1}/{len(df)}] exp={exp_id} {n_repeats} bags embedded")

    return np.array(embeddings), np.array(labels), experiment_ids


def run_mil_inference(
    checkpoint: str,
    study_id: str = None,
    model_name: str = "vit-l-16-mil",
    task_type: str = "binary",
    num_classes: int = 2,
    database_column: str = None,
    database_label: str = None,
    label_threshold: float = None,
    label_threshold_mode: str = "eq",
    experiment_ids: list = None,
    experiment_ids_file: str = None,
    bag_size: int = 200,
    pred_threshold: float = 0.5,
    output_path: str = "./reports/",
    seed: int = 403,
    negative_study_ids: list = None,
    negative_experiment_ids: list = None,
    skip_null_labels: bool = False,
    backbone_name: str = None,
    batch_size: int = 256,
    grid_cols: int = 6,
    max_grid_images: int = 36,
    gradcam_subsample: int = 8,
    rollout_max_instances: int = 36,
    rollout_head_fusion: str = "mean",
    rollout_discard_ratio: float = 0.0,
    n_top_attention: int = 5,
    n_bottom_attention: int = 5,
    sample_mode: str = "random",
    sample_percent: float = 100.0,
    class_definitions: list = None,
    exclude_filters: list = None,
    include_filters: list = None,
    default_class_idx: int = None,
    class_names: list = None,
    scan_size: int = None,
    top_k: int = None,
    **kwargs
):
    """
    Run MIL inference with attention visualization and PDF report.

    The PDF report includes:
    1. Summary page with metrics and probability strip plot (first page)
    2. Per-experiment attention pages showing top 5 + bottom 5 attention instances
    3. Null label predictions page (if any experiments have unknown labels)
    4. Detailed per-experiment statistics table
    """
    # Set defaults
    database_column = "class_label" if class_definitions else database_column
    database_label = database_label or "True"
    if backbone_name is None:
        from remo_id.models import get_augmentation_backbone
        backbone_name = get_augmentation_backbone(model_name)

    # Parse database_label
    database_label = parse_scalar(str(database_label))

    os.makedirs(output_path, exist_ok=True)
    out_pdf = os.path.join(output_path, "mil_eval_report.pdf")
    out_json = os.path.join(output_path, "per_experiment_stats.json")
    out_csv = os.path.join(output_path, "per_experiment_stats.csv")

    # Set seeds
    set_inference_seeds(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve experiment IDs (optional when using class_config)
    req_ids = resolve_experiment_ids(experiment_ids, experiment_ids_file)

    if not req_ids and not class_definitions:
        raise RuntimeError("No experiment IDs provided. Use experiment_ids, experiment_ids_file, or class_config.")

    # Create data module (train=False disables random augmentations like erasing)
    # Inference always uses scalar normalization (background_correction='none')
    ds = RemoIDDataModule(
        database_column=database_column,
        database_label=database_label,
        study_id=study_id,
        threshold=label_threshold,
        threshold_mode=label_threshold_mode,
        negative_study_ids=negative_study_ids or [],
        negative_experiment_ids=negative_experiment_ids or [],
        include_experiment_ids=req_ids,
        batch_size=batch_size,
        backbone_name=backbone_name,
        train=False,
        task_type=task_type,
        num_classes=num_classes,
        class_definitions=class_definitions,
        exclude_filters=exclude_filters or [],
        include_filters=include_filters or [],
        default_class_idx=default_class_idx,
        skip_null_labels=skip_null_labels,
        background_correction='none',
    )

    df = ds.data.copy()
    df["experiment_id"] = df["experiment_id"].astype(str)
    if req_ids:
        df = df[df["experiment_id"].isin(req_ids)].copy()
    df = df.sort_values("experiment_id").reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("No matching experiments found.")

    source_desc = f"class_config ({num_classes} classes)" if class_definitions else f"study_id={study_id}"
    print(f"Dataset contains {len(df)} experiments from {source_desc}")

    # Track null labels using the datamodule's tracking (by path)
    # The datamodule tracks experiments that originally had null labels before conversion
    null_label_paths = ds.null_label_paths
    df["_has_null_label"] = df["path"].isin(null_label_paths)
    n_null = df["_has_null_label"].sum()

    if n_null > 0:
        print(f"Found {n_null} experiments with null labels")
        if skip_null_labels:
            print("  - Will exclude from metrics calculation")
        print("  - Will include in report with predictions")

    # Extract class_names: prefer CLI arg, then checkpoint metadata
    if not class_names:
        ckpt_meta = extract_checkpoint_metadata(checkpoint)
        class_names = ckpt_meta.get("class_names")

    # Load model
    model, device, in_channels = load_checkpoint(
        checkpoint, model_name, task_type=task_type, num_classes=num_classes,
        n_experiments=1, device=device,
    )

    # Log sampling configuration
    if scan_size is not None and top_k is not None:
        print(f"Two-stage inference: scan {scan_size} images → select top {top_k} → re-aggregate")
    elif sample_mode != "random" or sample_percent != 100.0:
        print(f"Sampling mode: {sample_mode} {sample_percent:.0f}% of each experiment")
    else:
        print(f"Sampling mode: random (full experiment)")

    results, y_true_all, y_prob_all, y_pred_all, attention_pages_data, _ = _predict_experiments(
        model=model, ds=ds, df=df, database_column=database_column,
        task_type=task_type, num_classes=num_classes,
        bag_size=bag_size, seed=seed, device=device,
        pred_threshold=pred_threshold,
        sample_mode=sample_mode, sample_percent=sample_percent,
        n_top_attention=n_top_attention, n_bottom_attention=n_bottom_attention,
        class_names=class_names,
        scan_size=scan_size, top_k=top_k,
    )

    # Compute metrics (ALWAYS excluding null labels - they have no ground truth)
    # Null labels are still included in the probability plot for visualization
    valid_mask = np.array([not r.has_null_label for r in results if not np.isnan(r.y_prob)])
    y_true_eval = np.array([r.y_true for r in results if not np.isnan(r.y_prob)])[valid_mask]
    y_pred_eval = np.array([r.y_pred for r in results if not np.isnan(r.y_prob)])[valid_mask]
    y_prob_eval = np.array([r.y_prob for r in results if not np.isnan(r.y_prob)])[valid_mask]
    n_excluded = sum(1 for r in results if r.has_null_label and not np.isnan(r.y_prob))
    if n_excluded > 0:
        print(f"Computing metrics on {len(y_true_eval)} experiments (excluded {n_excluded} with null labels)")

    # Calculate metrics based on task type
    if task_type == "binary":
        y_true_arr = y_true_eval.astype(np.int64)
        y_pred_arr = y_pred_eval.astype(np.int64)
        cm = binconf(y_true_arr, y_pred_arr)
        precision, recall, f1 = precision_recall_f1(cm)
        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum() if cm.sum() > 0 else 0

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": int(cm[1, 1]),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
        }
        metrics_str = f"Accuracy={accuracy:.4f} Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}"

    elif task_type == "multiclass":
        y_true_arr = y_true_eval.astype(np.int64)
        y_pred_arr = y_pred_eval.astype(np.int64)
        metrics = compute_multiclass_metrics(y_true_arr, y_pred_arr, num_classes)
        metrics_str = f"Accuracy={metrics['accuracy']:.4f} F1_macro={metrics['f1_macro']:.4f}"

    # Count null labels
    n_null = sum(1 for r in results if r.has_null_label)
    n_total = len(results)

    # Build PDF with new structure
    c = canvas.Canvas(out_pdf, pagesize=letter)

    subtitle = (
        f"model={model_name} | study_id={study_id} | "
        f"label=({database_column} == {database_label}) | "
        f"bag_size={bag_size} | pred_threshold={pred_threshold}"
    )

    # PAGE 1: Summary with metrics and strip plot
    # Prepare data for strip plot
    all_exp_ids = [r.experiment_id for r in results if not np.isnan(r.y_prob)]
    all_y_true = np.array([r.y_true for r in results if not np.isnan(r.y_prob)])
    all_y_prob = np.array([r.y_prob for r in results if not np.isnan(r.y_prob)])
    all_y_pred = np.array([r.y_pred for r in results if not np.isnan(r.y_prob)])
    all_has_null = np.array([r.has_null_label for r in results if not np.isnan(r.y_prob)])

    mc_class_names = class_names or [f"Class {i}" for i in range(num_classes)]

    build_summary_first_page(
        c=c,
        title="MIL Evaluation Report",
        subtitle=subtitle,
        metrics=metrics,
        experiment_ids=all_exp_ids,
        y_true=all_y_true,
        y_prob=all_y_prob,
        has_null_labels=all_has_null,
        task_type=task_type,
        n_experiments=n_total,
        n_null=n_null,
        pred_threshold=pred_threshold,
        class_names=mc_class_names,
        y_pred=all_y_pred if task_type in ("multiclass", "binary") else None,
    )

    # Binary/Multiclass: add confusion matrix and per-class F1 pages
    if task_type in ("binary", "multiclass") and mc_class_names:
        cm = metrics.get("confusion_matrix")
        if cm is None and task_type == "binary":
            # Build confusion matrix from binary tp/tn/fp/fn
            tp = metrics.get("tp", 0)
            tn = metrics.get("tn", 0)
            fp = metrics.get("fp", 0)
            fn = metrics.get("fn", 0)
            cm = np.array([[tn, fp], [fn, tp]])
        if cm is not None:
            cm_png = make_confusion_matrix_heatmap(cm, mc_class_names)
            add_png_page(c, "Confusion Matrix", cm_png)

        f1_per_class = metrics.get("f1_per_class")
        if f1_per_class is None and task_type == "binary":
            # Compute per-class F1 from binary predictions
            from sklearn.metrics import f1_score as sklearn_f1
            f1_per_class = sklearn_f1(y_true_arr, y_pred_arr, average=None,
                                       labels=list(range(num_classes)), zero_division=0).tolist()
        if f1_per_class is not None:
            f1_png = make_per_class_metrics_table(f1_per_class, mc_class_names)
            add_png_page(c, "Per-Class F1 Scores", f1_png)

    # PAGES 2+: Per-experiment attention pages (top 5 + bottom 5)
    for page_data in attention_pages_data:
        exp_id = page_data["exp_id"]
        y_true = page_data["y_true"]
        pred_display = page_data["pred_display"]
        x_vis = page_data["x_vis"]  # Only visualization images (top + bottom)
        attn_vis = page_data["attn_vis"]
        original_indices = page_data["original_indices"]
        has_null = page_data["has_null"]

        null_marker = " [NULL LABEL]" if has_null else ""
        rollout_title = f"exp={exp_id} | y={y_true}{null_marker} | {pred_display}"

        # Pass pre-selected visualization images
        rollout_png = make_top_bottom_attention_grid_png_preselected(
            model=model, x_vis=x_vis, attn_vis=attn_vis,
            original_indices=original_indices, title=rollout_title,
            n_top=n_top_attention, n_bottom=n_bottom_attention,
            head_fusion=rollout_head_fusion, discard_ratio=rollout_discard_ratio,
            device=device,
        )
        add_png_page(c, f"Experiment {exp_id}: Top & Bottom Attention", rollout_png)

    # NULL LABELS PAGE: Show predictions for experiments with unknown ground truth
    null_results = [r for r in results if r.has_null_label and not np.isnan(r.y_prob)]
    if null_results:
        null_exp_ids = [r.experiment_id for r in null_results]
        null_probs = np.array([r.y_prob for r in null_results])
        null_preds = np.array([r.y_pred for r in null_results]) if task_type == "multiclass" else None

        null_table_png = make_null_predictions_table(
            null_exp_ids, null_probs, pred_threshold,
            task_type=task_type, class_names=mc_class_names, y_pred=null_preds,
        )
        add_png_page(c, "Predictions for Unknown Labels", null_table_png)

    # FINAL PAGE: Per-experiment summary table
    pdf_title(c, "Per-Experiment Statistics", "")
    y = letter[1] - 1.5 * inch

    df_results = pd.DataFrame([asdict(r) for r in results])
    # Drop path (too long) but keep all other columns including null labels
    if "path" in df_results.columns:
        df_results = df_results.drop(columns=["path"])

    # Rename columns to be more compact for PDF display
    rename_map = {
        "experiment_id": "exp_id",
        "sample_id": "sample",
        "study_id": "study",
        "bag_size_used": "bag_sz",
        "bag_logit": "logit",
        "attn_max": "attn_max",
        "attn_mean": "attn_avg",
        "has_null_label": "null_lbl",
    }
    df_results = df_results.rename(columns=rename_map)

    for col in df_results.select_dtypes(include=["float"]).columns:
        df_results[col] = df_results[col].map(lambda v: f"{v:.3f}" if not np.isnan(v) else "N/A")

    # Calculate column widths - use smaller font multiplier to fit more columns
    col_widths = []
    page_width = letter[0] - 1.0 * inch  # Available width
    for col in df_results.columns:
        max_len = max(df_results[col].map(lambda v: len(str(v))).max(), len(str(col)))
        col_widths.append(min(max_len * 6 + 2, 90))

    # Scale down if total exceeds page width
    total_width = sum(col_widths)
    if total_width > page_width:
        scale = page_width / total_width
        col_widths = [int(w * scale) for w in col_widths]

    y = pdf_table(c, 0.4*inch, y, df_results, col_widths, row_h=12)
    c.showPage()

    c.save()
    print(f"Saved PDF: {out_pdf}")

    # Save JSON/CSV
    with open(out_json, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    cols = list(asdict(results[0]).keys()) if results else []
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            d = asdict(r)
            f.write(",".join(str(d[k]) for k in cols) + "\n")

    print(f"Saved JSON: {out_json}")
    print(f"Saved CSV: {out_csv}")
    print(metrics_str)

    # Save structured prediction files alongside checkpoint
    from remo_id.inference.results import save_predictions
    try:
        metadata = {
            "task_type": task_type, "num_classes": num_classes,
            "class_names": class_names, "model_name": model_name,
            "seed": seed, "bag_size": bag_size,
        }
        save_predictions(
            checkpoint_path=checkpoint,
            results=results,
            metadata=metadata,
        )
    except Exception as e:
        print(f"Warning: Failed to save structured predictions: {e}")
