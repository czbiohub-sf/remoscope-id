"""Instance-level inference for remo-id."""

import os
import io
import random
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from remo_id.data.zarr_io import get_image_from_zarr

from remo_id.inference.core import (
    load_checkpoint,
    get_augmentations,
    label_from_row,
    preprocess_batch,
    predict_proba_batch,
    query_experiments_with_labels,
    resolve_experiments_from_class_definitions,
    resolve_experiment_ids,
    resolve_negative_studies,
    aggregate_image_predictions_binary,
    aggregate_image_predictions_multiclass,
    extract_checkpoint_metadata,
)
from remo_id.inference.results import ExperimentResult
from remo_id.analysis.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
)
from remo_id.reports.pdf import (
    fig_to_png_bytes,
    pdf_title,
    pdf_table,
    add_png_page,
)
from remo_id.interpret import (
    base_rgb_from_tensor,
    resize_mask_to_image,
    is_vit_like,
    is_dino_like,
    dino_cls_attention,
    vit_attention_rollout,
    get_target_layer_resnet,
    make_instance_overlay_signed,
    compute_signed_gradcam,
    make_signed_overlay,
    normalize_cams_globally,
)
from remo_id.reports.pdf.summary import (
    build_summary_first_page,
    make_null_predictions_table,
    make_confusion_matrix_heatmap,
    make_per_class_metrics_table,
)


def choose_image_ids(n_available: int, n_pick: int, strategy: str, seed: int) -> List[int]:
    """Select image indices based on strategy."""
    rng = random.Random(seed)
    n_pick = min(n_pick, n_available)
    if n_pick <= 0:
        return []
    if strategy == "first":
        return list(range(n_pick))
    elif strategy == "random":
        return rng.sample(range(n_available), n_pick)
    else:
        raise ValueError("strategy must be 'first' or 'random'")



def make_gradcam_overlay(
    model, device, img_np: np.ndarray, target_layer, augments=None, target_channels: int = 1,
    signed: bool = False, raw_cam: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Grad-CAM or attention rollout overlay.

    Args:
        model: The neural network model
        device: Device to run on
        img_np: Input image as numpy array
        target_layer: Target layer for Grad-CAM (unused for ViT)
        augments: Optional augmentation transform
        target_channels: Number of channels the model expects
        signed: If True, use signed Grad-CAM with diverging colormap
        raw_cam: Pre-computed (globally normalized) CAM array. If provided,
                 skips CAM computation and uses this directly.

    Returns:
        Tuple of (base_rgb, overlay) as uint8 arrays
    """
    from remo_id.inference.core import preprocess

    x = preprocess(img_np, augments=augments, target_channels=target_channels).to(device)
    base_rgb = base_rgb_from_tensor(x)

    if is_dino_like(model):
        mask = dino_cls_attention(model, x, layer_idx=-1)
        H, W, _ = base_rgb.shape
        mask = resize_mask_to_image(mask, H, W)
        overlay = show_cam_on_image(base_rgb, mask, use_rgb=True, image_weight=0.85)
        return base_rgb, overlay

    if is_vit_like(model):
        mask = vit_attention_rollout(model, x, head_fusion="mean", discard_ratio=0.0)
        H, W, _ = base_rgb.shape
        mask = resize_mask_to_image(mask, H, W)
        overlay = show_cam_on_image(base_rgb, mask, use_rgb=True, image_weight=0.85)
        return base_rgb, overlay

    # ResNet: Grad-CAM
    if signed:
        if raw_cam is not None:
            # Use pre-computed (globally normalized) CAM
            signed_overlay = make_signed_overlay(base_rgb, raw_cam, image_weight=0.6, colormap="RdBu_r")
        else:
            # Compute fresh signed Grad-CAM
            base_uint8, signed_overlay, _ = make_instance_overlay_signed(
                model, x, image_weight=0.6, colormap="RdBu_r"
            )
        return base_rgb, signed_overlay

    # Standard Grad-CAM
    x = x.requires_grad_(True)
    targets = [ClassifierOutputTarget(0)]

    with torch.enable_grad():
        with GradCAM(model=model, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=x, targets=targets)[0]

    overlay = show_cam_on_image(base_rgb, grayscale_cam, use_rgb=True, image_weight=0.85)
    return base_rgb, overlay


def compute_raw_signed_cam(
    model, device, img_np: np.ndarray, augments=None, target_channels: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute raw (unnormalized) signed Grad-CAM for global normalization.

    Args:
        model: The neural network model
        device: Device to run on
        img_np: Input image as numpy array
        augments: Optional augmentation transform
        target_channels: Number of channels the model expects

    Returns:
        Tuple of (raw_cam, base_rgb) where raw_cam has unnormalized values
    """
    from remo_id.inference.core import preprocess

    x = preprocess(img_np, augments=augments, target_channels=target_channels).to(device)
    raw_cam, base_rgb01 = compute_signed_gradcam(model, x, normalize=False)
    base_rgb = base_rgb_from_tensor(x)
    return raw_cam, base_rgb


def make_side_by_side_png(
    base_rgb: np.ndarray,
    overlay: np.ndarray,
    title_left: str,
    title_right: str,
    suptitle: str,
    signed: bool = False
) -> bytes:
    """Create side-by-side visualization PNG.

    Args:
        base_rgb: Original image as uint8 array
        overlay: Overlay visualization as uint8 array
        title_left: Title for left (original) image
        title_right: Title for right (overlay) image
        suptitle: Main title for the figure
        signed: If True, add colorbar for signed Grad-CAM interpretation

    Returns:
        PNG bytes of the figure
    """
    if signed:
        # Create figure with extra space for colorbar and legend
        fig = plt.figure(figsize=(12, 5))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.08], width_ratios=[1, 1, 0.05])

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        cax = fig.add_subplot(gs[0, 2])

        ax0.imshow(base_rgb)
        ax0.set_title(title_left, fontsize=10)
        ax0.axis("off")

        ax1.imshow(overlay)
        ax1.set_title(title_right, fontsize=10)
        ax1.axis("off")

        # Add colorbar for signed interpretation
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=-1, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cm.get_cmap("RdBu_r"), norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(["Against", "Neutral", "For"])
        cbar.ax.tick_params(labelsize=8)

        fig.suptitle(suptitle, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(base_rgb)
        axes[0].set_title(title_left, fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(title_right, fontsize=10)
        axes[1].axis("off")

        fig.suptitle(suptitle, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.92])

    return fig_to_png_bytes(fig)


def make_heatmap(exp_records: pd.DataFrame, image_thresh_grid: np.ndarray, pct_grid: np.ndarray) -> bytes:
    """Create threshold sweep heatmap."""
    f1_mat = np.zeros((len(pct_grid), len(image_thresh_grid)), dtype=float)

    for i, pct_thr in enumerate(pct_grid):
        for j, img_thr in enumerate(image_thresh_grid):
            y_true = exp_records["y_true"].to_numpy().astype(int)
            y_pred = []
            for probs in exp_records["probs"].tolist():
                exp_pred, *_ = aggregate_image_predictions_binary(
                    probs, image_thresh=float(img_thr), exp_pos_pct=float(pct_thr)
                )
                y_pred.append(exp_pred)
            y_pred = np.array(y_pred, dtype=int)
            m = compute_binary_metrics(y_true, y_pred)
            f1_mat[i, j] = m["f1"]

    fig = plt.figure(figsize=(8.5, 6.5))
    ax = fig.add_subplot(111)
    f1_mat = np.clip(f1_mat, 0.01, 1.0)
    im = ax.imshow(f1_mat, aspect="auto", origin="lower", norm=LogNorm(vmin=0.01, vmax=1.0), cmap="viridis")

    n_xticks = 10
    n_yticks = 10
    xtick_idx = np.linspace(0, len(image_thresh_grid) - 1, n_xticks, dtype=int)
    ytick_idx = np.linspace(0, len(pct_grid) - 1, n_yticks, dtype=int)

    ax.set_xticks(xtick_idx)
    ax.set_yticks(ytick_idx)
    ax.set_xticklabels([f"{image_thresh_grid[i]:.2f}" for i in xtick_idx])
    ax.set_yticklabels([f"{pct_grid[i]:.1f}" for i in ytick_idx])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_xlabel("Image threshold (p > thresh)")
    ax.set_ylabel("Experiment %positive threshold")
    ax.set_title("Experiment-level F1 across thresholds")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="F1")

    return fig_to_png_bytes(fig)


def make_conf_histogram(image_probs: List[float], y_true: List[int], image_thresh: float) -> bytes:
    """Create confidence histogram."""
    probs = np.array(image_probs, dtype=float)
    y_true = np.array(y_true, dtype=int)
    bins = np.linspace(0.0, 1.0, 21)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.hist(probs[y_true == 0], bins=bins, alpha=0.6, label="Negative", color="blue")
    ax.hist(probs[y_true == 1], bins=bins, alpha=0.6, label="Positive", color="orange")
    ax.axvline(x=image_thresh, color="red", linestyle="--", label=f"Threshold={image_thresh:.2f}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Number of images")
    ax.set_title("Histogram of image-level predicted probabilities")
    ax.legend()

    return fig_to_png_bytes(fig)


def build_pdf_binary(
    out_pdf: str,
    args,
    exp_metrics_df: pd.DataFrame,
    overall_metrics: Dict[str, float],
    heatmap_png: bytes,
    gradcam_by_exp: Dict[str, List[bytes]],
    conf_hist_png: bytes,
    null_label_exp_ids: Optional[List[str]] = None,
):
    """Build PDF report for binary classification with summary first page."""
    c = canvas.Canvas(out_pdf, pagesize=letter)
    w, h = letter

    # Prepare data for summary page
    experiment_ids = exp_metrics_df["experiment_id"].tolist()
    y_true = exp_metrics_df["y_true"].to_numpy()
    y_prob = exp_metrics_df["mean_prob"].to_numpy() if "mean_prob" in exp_metrics_df.columns else np.zeros(len(exp_metrics_df))

    # Determine null labels
    null_label_exp_ids = null_label_exp_ids or []
    has_null_labels = np.array([str(eid) in null_label_exp_ids for eid in experiment_ids])
    n_null = int(has_null_labels.sum())

    subtitle = (
        f"study_id={args.study_id} | database_column={args.database_column} | "
        f"bag_size={args.bag_size} | image_thresh={args.image_thresh}"
    )

    # PAGE 1: Summary with metrics and strip plot
    build_summary_first_page(
        c=c,
        title="RemoID Instance Inference Report",
        subtitle=subtitle,
        metrics=overall_metrics,
        experiment_ids=experiment_ids,
        y_true=y_true,
        y_prob=y_prob,
        has_null_labels=has_null_labels,
        task_type="binary",
        n_experiments=len(exp_metrics_df),
        n_null=n_null,
        pred_threshold=args.image_thresh,
    )

    # PAGE 2: Heatmap
    pdf_title(c, "Threshold Sweep Heatmap", "Sweeping image threshold vs %positive threshold")
    img = ImageReader(io.BytesIO(heatmap_png))
    c.drawImage(img, 0.75*inch, 1.25*inch, width=7.0*inch, height=9.0*inch, preserveAspectRatio=True, mask='auto')
    c.showPage()

    # Grad-CAM pages
    for i, (exp_id, imgs) in enumerate(gradcam_by_exp.items()):
        if i >= args.gradcam_max_experiments:
            break
        pdf_title(c, f"Grad-CAM: experiment {exp_id}", "")
        y_top = h - 1.6*inch
        box_h = 2.6*inch
        y = y_top
        for png_bytes in imgs[:3]:
            img = ImageReader(io.BytesIO(png_bytes))
            c.drawImage(img, 0.75*inch, y - box_h, width=7.0*inch, height=box_h, preserveAspectRatio=True, mask='auto')
            y -= (box_h + 0.25*inch)
        c.showPage()

    # Null label predictions page (if any)
    if n_null > 0:
        null_exp_ids = [eid for eid, is_null in zip(experiment_ids, has_null_labels) if is_null]
        null_probs = np.array([y_prob[i] for i, is_null in enumerate(has_null_labels) if is_null])
        null_table_png = make_null_predictions_table(null_exp_ids, null_probs, args.image_thresh)
        add_png_page(c, "Predictions for Unknown Labels", null_table_png)

    # Confidence histogram
    pdf_title(c, "Image-level Confidence Histogram", "")
    img = ImageReader(io.BytesIO(conf_hist_png))
    c.drawImage(img, 0.75*inch, 1.25*inch, width=7.0*inch, height=9.0*inch, preserveAspectRatio=True, mask='auto')
    c.showPage()

    # Per-experiment summary table
    pdf_title(c, "Per-Experiment Statistics", "")
    y = h - 1.5 * inch

    view = exp_metrics_df.copy()
    view["err"] = (view["y_true"] - view["y_pred"]).abs()
    view = view.sort_values(["err", "mean_prob"], ascending=[False, False])
    cols = ["experiment_id", "y_true", "y_pred", "n_images", "n_pos_images", "mean_prob"]
    cols = [col for col in cols if col in view.columns]
    view = view[cols]
    if "mean_prob" in view.columns:
        view["mean_prob"] = view["mean_prob"].round(3)

    y = pdf_table(c, 0.75*inch, y, view, [1.3*inch]*len(cols), row_h=14)
    c.showPage()

    c.save()


def build_pdf_multiclass(
    out_pdf: str,
    args,
    exp_metrics_df: pd.DataFrame,
    overall_metrics: Dict[str, float],
    gradcam_by_exp: Dict[str, List[bytes]],
    class_names: List[str],
    null_label_exp_ids: Optional[List[str]] = None,
):
    """Build PDF report for multiclass classification."""
    c = canvas.Canvas(out_pdf, pagesize=letter)
    w, h = letter

    experiment_ids = exp_metrics_df["experiment_id"].tolist()
    y_true = exp_metrics_df["y_true"].to_numpy()
    y_pred = exp_metrics_df["y_pred"].to_numpy()
    y_conf = exp_metrics_df["mean_conf"].to_numpy() if "mean_conf" in exp_metrics_df.columns else np.zeros(len(exp_metrics_df))

    null_label_exp_ids = null_label_exp_ids or []
    has_null_labels = np.array([str(eid) in null_label_exp_ids for eid in experiment_ids])
    n_null = int(has_null_labels.sum())

    subtitle = (
        f"study_id={args.study_id} | database_column={args.database_column} | "
        f"classes={', '.join(class_names)}"
    )

    # PAGE 1: Summary with metrics and multiclass strip plot
    build_summary_first_page(
        c=c,
        title="RemoID Multiclass Inference Report",
        subtitle=subtitle,
        metrics=overall_metrics,
        experiment_ids=experiment_ids,
        y_true=y_true,
        y_prob=y_conf,
        has_null_labels=has_null_labels,
        task_type="multiclass",
        n_experiments=len(exp_metrics_df),
        n_null=n_null,
        class_names=class_names,
        y_pred=y_pred,
    )

    # PAGE 2: Full-page confusion matrix heatmap
    cm = overall_metrics.get("confusion_matrix")
    if cm is not None:
        cm_png = make_confusion_matrix_heatmap(cm, class_names)
        add_png_page(c, "Confusion Matrix", cm_png)

    # PAGE 3: Per-class F1 scores
    f1_per_class = overall_metrics.get("f1_per_class")
    if f1_per_class is not None:
        f1_png = make_per_class_metrics_table(f1_per_class, class_names)
        add_png_page(c, "Per-Class F1 Scores", f1_png)

    # Grad-CAM pages (target predicted class)
    for i, (exp_id, imgs) in enumerate(gradcam_by_exp.items()):
        if i >= args.gradcam_max_experiments:
            break
        if not imgs:
            continue
        pdf_title(c, f"Grad-CAM: experiment {exp_id}", "")
        y_top = h - 1.6 * inch
        box_h = 2.6 * inch
        y = y_top
        for png_bytes in imgs[:3]:
            img = ImageReader(io.BytesIO(png_bytes))
            c.drawImage(img, 0.75 * inch, y - box_h, width=7.0 * inch, height=box_h,
                       preserveAspectRatio=True, mask='auto')
            y -= (box_h + 0.25 * inch)
        c.showPage()

    # Null label predictions page
    if n_null > 0:
        null_exp_ids_list = [eid for eid, is_null in zip(experiment_ids, has_null_labels) if is_null]
        null_confs = np.array([y_conf[i] for i, is_null in enumerate(has_null_labels) if is_null])
        null_preds = np.array([y_pred[i] for i, is_null in enumerate(has_null_labels) if is_null])
        null_table_png = make_null_predictions_table(
            null_exp_ids_list, null_confs, task_type="multiclass",
            class_names=class_names, y_pred=null_preds,
        )
        add_png_page(c, "Predictions for Unknown Labels", null_table_png)

    # Per-experiment summary table
    pdf_title(c, "Per-Experiment Statistics", "")
    y = h - 1.5 * inch

    view = exp_metrics_df.copy()
    # Map class indices to names for display
    view["true_class"] = view["y_true"].apply(
        lambda x: class_names[int(x)] if 0 <= int(x) < len(class_names) else f"Class {int(x)}"
    )
    view["pred_class"] = view["y_pred"].apply(
        lambda x: class_names[int(x)] if 0 <= int(x) < len(class_names) else f"Class {int(x)}"
    )
    view["correct"] = (view["y_true"] == view["y_pred"]).astype(int)
    view = view.sort_values("correct", ascending=True)

    cols = ["experiment_id", "true_class", "pred_class", "mean_conf", "n_images"]
    cols = [col for col in cols if col in view.columns]
    if "mean_conf" in view.columns:
        view["mean_conf"] = view["mean_conf"].round(3)

    y = pdf_table(c, 0.75 * inch, y, view[cols], [1.5 * inch] * len(cols), row_h=14)
    c.showPage()

    c.save()


def _predict_experiments_instance(
    model,
    df: "pd.DataFrame",
    database_column: str,
    task_type: str,
    num_classes: int,
    bag_size: int,
    seed: int,
    device: str,
    augments,
    in_channels: int = 1,
    image_thresh: float = 0.5,
    exp_pos_pct: float = None,
    class_names: Optional[List[str]] = None,
) -> Tuple[List["ExperimentResult"], List[float], List[float], List[float]]:
    """Run per-experiment instance-level prediction loop.

    Parallel to ``_predict_experiments`` in ``inference/mil.py`` so that
    ``save_predictions()`` works identically for both model types.

    Returns
    -------
    results : List[ExperimentResult]
    y_true_all, y_prob_all, y_pred_all : lists of scalars
    """
    results: List[ExperimentResult] = []
    y_true_all = []
    y_prob_all = []
    y_pred_all = []

    for exp_idx, (_, row) in enumerate(df.iterrows()):
        exp_id = str(row["experiment_id"])
        exp_sample_id = str(row.get("sample_id", ""))
        exp_study_id = str(row.get("study_id", ""))
        exp_path = str(row["path"])
        n_avail = int(row["num_images"])
        has_null = bool(row.get("_has_null_label", False))

        y_true = int(row["y_true"])

        img_ids = choose_image_ids(n_avail, bag_size, "random", seed=seed)
        if not img_ids:
            results.append(ExperimentResult(
                experiment_id=exp_id, sample_id=exp_sample_id,
                study_id=exp_study_id, path=exp_path, y_true=y_true,
                y_prob=float("nan"), y_pred=0, bag_size_used=0,
                bag_logit=float("nan"), attn_max=float("nan"),
                attn_mean=float("nan"), has_null_label=has_null,
            ))
            continue

        # Load images
        try:
            all_images = get_image_from_zarr(exp_path, img_ids=[int(i) for i in img_ids])
            all_images = np.array(all_images)
            if all_images.ndim == 3 and all_images.shape[-1] == len(img_ids):
                all_images = [all_images[:, :, i] for i in range(all_images.shape[-1])]
            elif all_images.ndim == 2:
                all_images = [all_images]
            elif not isinstance(all_images, (list, tuple)):
                all_images = [all_images]
        except Exception as e:
            print(f"[{exp_id}] Error loading images: {e}")
            continue

        images_np = []
        for img in all_images:
            try:
                img_np = np.array(img)
                if img_np.ndim == 3 and img_np.shape[-1] == 1:
                    img_np = img_np[..., 0]
                images_np.append(img_np)
            except Exception:
                pass

        if not images_np:
            continue

        batch_tensor = preprocess_batch(images_np, augments=augments, target_channels=in_channels).to(device)
        image_preds = predict_proba_batch(model, batch_tensor, task_type=task_type, batch_size=32)

        if task_type == "binary":
            probs = image_preds
            exp_pred, n_pos, pct_pos, mean_prob = aggregate_image_predictions_binary(
                probs, image_thresh=image_thresh, exp_pos_pct=exp_pos_pct
            )
            y_probs_all_val = [1.0 - mean_prob, mean_prob]
            res = ExperimentResult(
                experiment_id=exp_id, sample_id=exp_sample_id,
                study_id=exp_study_id, path=exp_path, y_true=y_true,
                y_prob=mean_prob, y_pred=int(exp_pred),
                bag_size_used=len(probs), bag_logit=float("nan"),
                attn_max=float("nan"), attn_mean=float("nan"),
                has_null_label=has_null, y_probs_all=y_probs_all_val,
            )

        elif task_type == "multiclass":
            exp_pred, mean_conf, class_counts = aggregate_image_predictions_multiclass(
                image_preds, num_classes
            )
            # Average class probabilities across images
            all_class_probs = np.array([p[2] for p in image_preds])
            mean_class_probs = all_class_probs.mean(axis=0).tolist()
            res = ExperimentResult(
                experiment_id=exp_id, sample_id=exp_sample_id,
                study_id=exp_study_id, path=exp_path, y_true=y_true,
                y_prob=mean_conf, y_pred=int(exp_pred),
                bag_size_used=len(image_preds), bag_logit=float("nan"),
                attn_max=float("nan"), attn_mean=float("nan"),
                has_null_label=has_null, y_probs_all=mean_class_probs,
            )

        results.append(res)
        y_true_all.append(y_true)
        y_prob_all.append(res.y_prob)
        y_pred_all.append(res.y_pred)

        print(f"[{exp_idx+1}/{len(df)}] exp={exp_id} y={y_true} pred={res.y_pred} null={has_null}")

    return results, y_true_all, y_prob_all, y_pred_all


def run_instance_inference(
    checkpoint: str,
    study_id: str,
    model_name: str = "resnet18",
    task_type: str = "binary",
    num_classes: int = 2,
    database_column: str = None,
    database_label: str = None,
    label_threshold: float = None,
    label_threshold_mode: str = "eq",
    experiment_ids: list = None,
    experiment_ids_file: str = None,
    bag_size: int = 200,
    image_pick_strategy: str = "random",
    image_thresh: float = 0.5,
    exp_pos_pct: float = None,
    out_pdf: str = "./reports/inference_report.pdf",
    seed: int = 403,
    gradcam_images_per_exp: int = 3,
    gradcam_max_experiments: int = 8,
    negative_study_ids: list = None,
    negative_experiment_ids: list = None,
    skip_null_labels: bool = False,
    signed_gradcam: bool = False,
    normalize_global: bool = False,
    class_definitions: list = None,
    exclude_filters: list = None,
    include_filters: list = None,
    default_class_idx: int = None,
    **kwargs
):
    """Run instance-level inference with full PDF report generation.

    Experiments come from ``class_definitions`` (class config or manifest) when
    given, otherwise from the database (``study_id``/``database_column``).

    Args:
        normalize_global: If True and signed_gradcam is True, normalize all
            GradCAM heatmaps to a global scale so they are directly comparable
            across images. Uses 99th percentile of absolute values.
    """
    import argparse

    # Build args namespace for PDF functions
    args = argparse.Namespace(
        checkpoint=checkpoint, model_name=model_name, out_pdf=out_pdf,
        task_type=task_type, num_classes=num_classes, study_id=study_id,
        database_column="class_label" if class_definitions else database_column,
        label_threshold_mode=label_threshold_mode,
        database_label=database_label, label_threshold=label_threshold, bag_size=bag_size,
        image_thresh=image_thresh, exp_pos_pct=exp_pos_pct, seed=seed,
        gradcam_images_per_exp=gradcam_images_per_exp,
        gradcam_max_experiments=gradcam_max_experiments,
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)

    # Extract metadata from checkpoint for class names
    ckpt_meta = extract_checkpoint_metadata(checkpoint)
    class_names = ckpt_meta.get("class_names")

    # Load model
    print(f"Loading model {model_name} from {checkpoint}")
    model, device, in_channels = load_checkpoint(checkpoint, model_name, task_type, num_classes)
    augments = get_augmentations(model.backbone_name, crop_size=ckpt_meta.get("crop_size"))
    target_layer = None if (is_vit_like(model) or is_dino_like(model)) else get_target_layer_resnet(model)

    if signed_gradcam:
        if normalize_global:
            print("Using signed Grad-CAM with GLOBAL normalization (comparable across images)")
        else:
            print("Using signed Grad-CAM (red=for diagnosis, blue=against)")

    # Resolve experiment IDs
    exp_ids = resolve_experiment_ids(experiment_ids, experiment_ids_file)

    if class_definitions:
        df = resolve_experiments_from_class_definitions(
            class_definitions, task_type, num_classes, exclude_filters=exclude_filters,
            include_filters=include_filters, default_class_idx=default_class_idx,
            experiment_ids=exp_ids or None,
        )
        df["y_true"] = df["y"]
        df["_has_null_label"] = False
        if len(df) == 0:
            raise RuntimeError("No experiments found.")
    else:
        # Get experiment data with labels
        df = query_experiments_with_labels(study_id, args.database_column, exp_ids or None)

        # Handle negative studies/experiments
        df["_has_null_label"] = df[args.database_column].isna()
        df = resolve_negative_studies(df, args.database_column, negative_study_ids, None)
        # Mark forced-negative experiments as not having null labels
        df.loc[df["_is_forced_negative"], "_has_null_label"] = False

        if len(df) == 0:
            raise RuntimeError("No experiments found.")

        # Compute labels
        def compute_label(row):
            if row.get("_is_forced_negative", False):
                return 0
            return label_from_row(row, args.database_column, args.label_threshold_mode,
                                  args.database_label, args.label_threshold, task_type=task_type)

        df["y_true"] = df.apply(compute_label, axis=1)
        df = df.drop(columns=["_is_forced_negative"], errors="ignore")

    print(f"Processing {len(df)} experiments...")

    exp_rows = []
    gradcam_by_exp: Dict[str, List[bytes]] = {}

    # For global normalization: collect raw CAMs first, then normalize and create overlays
    deferred_gradcam_data = []  # List of (exp_id, img_idx, img_np, raw_cam, base_rgb, prob) tuples

    for exp_idx, (_, row) in enumerate(df.iterrows()):
        exp_id = str(row["experiment_id"])
        path = row["path"]
        n_avail = int(row["num_images"])
        y_true = int(row["y_true"])

        img_ids = choose_image_ids(n_avail, bag_size, image_pick_strategy, seed=seed)
        if not img_ids:
            continue

        print(f"[{exp_idx+1}/{len(df)}] {exp_id}: loading {len(img_ids)} images...")

        # Load images
        try:
            all_images = get_image_from_zarr(path, img_ids=[int(i) for i in img_ids])
            all_images = np.array(all_images)
            if all_images.ndim == 3 and all_images.shape[-1] == len(img_ids):
                all_images = [all_images[:, :, i] for i in range(all_images.shape[-1])]
            elif all_images.ndim == 2:
                all_images = [all_images]
            elif not isinstance(all_images, (list, tuple)):
                all_images = [all_images]
        except Exception as e:
            print(f"[{exp_id}] Error loading images: {e}")
            continue

        images_np = []
        valid_img_ids = []
        for i, img in enumerate(all_images):
            try:
                img_np = np.array(img)
                if img_np.ndim == 3 and img_np.shape[-1] == 1:
                    img_np = img_np[..., 0]
                images_np.append(img_np)
                valid_img_ids.append(img_ids[i])
            except Exception:
                pass

        if not images_np:
            continue

        # Batch predictions
        batch_tensor = preprocess_batch(images_np, augments=augments, target_channels=in_channels).to(device)
        image_preds = predict_proba_batch(model, batch_tensor, task_type=task_type, batch_size=32)

        # Aggregate and create Grad-CAM
        overlays = []
        if task_type == "binary":
            probs = image_preds
            exp_pred, n_pos, pct_pos, mean_prob = aggregate_image_predictions_binary(
                probs, image_thresh=image_thresh, exp_pos_pct=exp_pos_pct
            )

            # Select images for Grad-CAM
            order = np.argsort(np.array(probs))
            pick_idx = [int(order[-1]), int(order[0]), int(order[len(order)//2])]
            pick_idx = list(dict.fromkeys([i for i in pick_idx if i < len(valid_img_ids)]))[:gradcam_images_per_exp]

            if signed_gradcam and normalize_global:
                # Collect raw CAMs for deferred global normalization
                for pi in pick_idx:
                    try:
                        raw_cam, base_rgb = compute_raw_signed_cam(
                            model, device, images_np[pi], augments, target_channels=in_channels
                        )
                        deferred_gradcam_data.append({
                            "exp_id": exp_id,
                            "img_idx": valid_img_ids[pi],
                            "img_np": images_np[pi].copy(),
                            "raw_cam": raw_cam,
                            "base_rgb": base_rgb,
                            "prob": probs[pi],
                        })
                    except Exception as e:
                        print(f"[{exp_id}] Raw CAM computation failed: {e}")
                # Overlays will be created after all experiments are processed
                gradcam_by_exp[exp_id] = []  # Placeholder
            else:
                # Create overlays immediately (original behavior)
                for pi in pick_idx:
                    try:
                        base_rgb, overlay = make_gradcam_overlay(
                            model, device, images_np[pi], target_layer, augments,
                            target_channels=in_channels, signed=signed_gradcam
                        )
                        overlay_label = "Signed Grad-CAM" if signed_gradcam else "Grad-CAM"
                        suptitle = f"exp={exp_id} | img={valid_img_ids[pi]} | p={probs[pi]:.3f}"
                        png = make_side_by_side_png(
                            base_rgb, overlay, "Original", overlay_label, suptitle,
                            signed=signed_gradcam
                        )
                        overlays.append(png)
                    except Exception as e:
                        print(f"[{exp_id}] GradCAM failed: {e}")
                gradcam_by_exp[exp_id] = overlays
            has_null = bool(row.get("_has_null_label", False))

            exp_rows.append({
                "experiment_id": exp_id, "y_true": y_true, "y_pred": int(exp_pred),
                "n_images": len(probs), "n_pos_images": int(n_pos), "mean_prob": float(mean_prob),
                "probs": probs, "_has_null_label": has_null,
            })

        elif task_type == "multiclass":
            exp_pred, mean_conf, class_counts = aggregate_image_predictions_multiclass(image_preds, num_classes)
            has_null = bool(row.get("_has_null_label", False))

            # GradCAM for multiclass: target the predicted class
            if not (is_vit_like(model) or is_dino_like(model)):
                confs = [p[1] for p in image_preds]
                order = np.argsort(np.array(confs))
                pick_idx = [int(order[-1]), int(order[0]), int(order[len(order)//2])]
                pick_idx = list(dict.fromkeys([i for i in pick_idx if i < len(valid_img_ids)]))[:gradcam_images_per_exp]

                mc_class_names_local = class_names or [f"Class {i}" for i in range(num_classes)]
                for pi in pick_idx:
                    try:
                        pred_cls = image_preds[pi][0]
                        base_rgb, overlay = make_gradcam_overlay(
                            model, device, images_np[pi], target_layer, augments,
                            target_channels=in_channels, signed=signed_gradcam
                        )
                        cls_name = mc_class_names_local[pred_cls] if pred_cls < len(mc_class_names_local) else f"Class {pred_cls}"
                        suptitle = f"exp={exp_id} | img={valid_img_ids[pi]} | pred={cls_name} conf={image_preds[pi][1]:.3f}"
                        overlay_label = "Signed Grad-CAM" if signed_gradcam else "Grad-CAM"
                        png = make_side_by_side_png(
                            base_rgb, overlay, "Original", overlay_label, suptitle,
                            signed=signed_gradcam
                        )
                        overlays.append(png)
                    except Exception as e:
                        print(f"[{exp_id}] GradCAM failed: {e}")
                gradcam_by_exp[exp_id] = overlays

            exp_rows.append({
                "experiment_id": exp_id, "y_true": y_true, "y_pred": int(exp_pred),
                "n_images": len(image_preds), "mean_conf": float(mean_conf),
                "_has_null_label": has_null,
            })

        print(f"[{exp_idx+1}/{len(df)}] {exp_id}: done (pred={exp_rows[-1]['y_pred']})")

    # Process deferred GradCAM data with global normalization
    if deferred_gradcam_data and signed_gradcam and normalize_global:
        print(f"Applying global normalization to {len(deferred_gradcam_data)} GradCAM heatmaps...")

        # Extract raw CAMs and normalize globally
        raw_cams = [d["raw_cam"] for d in deferred_gradcam_data]
        normalized_cams = normalize_cams_globally(raw_cams, percentile=99.0)

        # Create overlays with normalized CAMs
        for i, (data, norm_cam) in enumerate(zip(deferred_gradcam_data, normalized_cams)):
            try:
                # Create overlay using pre-normalized CAM
                base_rgb, overlay = make_gradcam_overlay(
                    model, device, data["img_np"], target_layer, augments,
                    target_channels=in_channels, signed=True, raw_cam=norm_cam
                )
                suptitle = f"exp={data['exp_id']} | img={data['img_idx']} | p={data['prob']:.3f}"
                png = make_side_by_side_png(
                    base_rgb, overlay, "Original", "Signed Grad-CAM (Global)",
                    suptitle, signed=True
                )
                # Add to appropriate experiment
                if data["exp_id"] not in gradcam_by_exp:
                    gradcam_by_exp[data["exp_id"]] = []
                gradcam_by_exp[data["exp_id"]].append(png)
            except Exception as e:
                print(f"[{data['exp_id']}] Overlay creation failed: {e}")

        print(f"Created {sum(len(v) for v in gradcam_by_exp.values())} GradCAM overlays")

    exp_df = pd.DataFrame(exp_rows)
    if len(exp_df) == 0:
        raise RuntimeError("No predictions generated.")

    # ALWAYS filter out null labels for metrics - they have no ground truth
    # Null labels are still included in the PDF report for visualization
    eval_df = exp_df[~exp_df["_has_null_label"]]

    # Get list of experiment IDs with null labels for PDF report
    null_label_exp_ids = exp_df[exp_df["_has_null_label"]]["experiment_id"].astype(str).tolist()
    n_null = len(null_label_exp_ids)
    if n_null > 0:
        print(f"Found {n_null} experiments with null labels - excluded from metrics, included in report")
        print(f"Computing metrics on {len(eval_df)} experiments")

    # Compute metrics and build PDF
    if task_type == "binary":
        overall = compute_binary_metrics(eval_df["y_true"].to_numpy(), eval_df["y_pred"].to_numpy())

        image_thresh_grid = np.linspace(0.01, 0.99, 40)
        pct_grid = np.linspace(0.0, 100.0, 50)
        heatmap_png = make_heatmap(eval_df[["y_true", "probs"]], image_thresh_grid, pct_grid)

        conf_hist_png = make_conf_histogram(
            [p for probs in eval_df["probs"] for p in probs],
            [yt for _, row in eval_df.iterrows() for yt in [row["y_true"]]*int(row["n_images"])],
            image_thresh
        )

        pdf_df = exp_df.drop(columns=["probs", "_has_null_label"], errors="ignore")
        build_pdf_binary(out_pdf, args, pdf_df, overall, heatmap_png, gradcam_by_exp, conf_hist_png, null_label_exp_ids)

        print(f"\nSaved PDF: {out_pdf}")
        print(f"Metrics: accuracy={overall['accuracy']:.4f}, precision={overall['precision']:.4f}, "
              f"recall={overall['recall']:.4f}, f1={overall['f1']:.4f}")

    elif task_type == "multiclass":
        overall = compute_multiclass_metrics(eval_df["y_true"].to_numpy(), eval_df["y_pred"].to_numpy(), num_classes)

        # Use generic class names if none from checkpoint
        mc_class_names = class_names or [f"Class {i}" for i in range(num_classes)]

        pdf_df = exp_df.drop(columns=["_has_null_label"], errors="ignore")
        build_pdf_multiclass(out_pdf, args, pdf_df, overall, gradcam_by_exp, mc_class_names, null_label_exp_ids)

        print(f"\nSaved PDF: {out_pdf}")
        print(f"Metrics: accuracy={overall['accuracy']:.4f}, f1_macro={overall['f1_macro']:.4f}")

    # Save structured prediction files alongside checkpoint
    from remo_id.inference.results import save_predictions
    try:
        instance_results = []
        for _, row_data in exp_df.iterrows():
            y_prob = float(row_data.get("mean_prob", row_data.get("y_pred", float("nan"))))
            y_probs_all = None
            if task_type == "multiclass" and "mean_conf" in row_data:
                y_probs_all = None  # not available from aggregated instance results
            instance_results.append(ExperimentResult(
                experiment_id=str(row_data["experiment_id"]),
                sample_id=str(df.loc[df["experiment_id"].astype(str) == str(row_data["experiment_id"]), "sample_id"].iloc[0])
                    if "sample_id" in df.columns else "",
                study_id=str(df.loc[df["experiment_id"].astype(str) == str(row_data["experiment_id"]), "study_id"].iloc[0])
                    if "study_id" in df.columns else "",
                path=str(df.loc[df["experiment_id"].astype(str) == str(row_data["experiment_id"]), "path"].iloc[0])
                    if "path" in df.columns else "",
                y_true=float(row_data["y_true"]),
                y_prob=y_prob,
                y_pred=float(row_data["y_pred"]),
                bag_size_used=int(row_data.get("n_images", 0)),
                bag_logit=float("nan"),
                attn_max=float("nan"),
                attn_mean=float("nan"),
                has_null_label=bool(row_data.get("_has_null_label", False)),
            ))
        metadata = {
            "task_type": task_type, "num_classes": num_classes,
            "class_names": class_names, "model_name": model_name,
            "seed": seed, "bag_size": bag_size,
        }
        save_predictions(
            checkpoint_path=checkpoint,
            results=instance_results,
            metadata=metadata,
        )
    except Exception as e:
        print(f"Warning: Failed to save structured predictions: {e}")
