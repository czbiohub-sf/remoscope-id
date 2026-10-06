"""Gallery generation for high-activation crop visualization.

This module generates PDF galleries showing high-resolution crops around
regions of highest activation in Grad-CAM heatmaps. The gallery is organized
into two sections:

1. Positive Experiments - "Diagnostic Features"
   Shows crops where red (positive gradient) indicates correct diagnostic signals

2. Negative Experiments - "False Positive Triggers"
   Shows crops where red indicates features that look positive in healthy samples
"""

import os
import io
import math
import random
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from remo_id.data.zarr_io import get_image_from_zarr

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from remo_id.inference.core import (
    load_checkpoint,
    get_augmentations,
    get_experiment_metadata,
    label_from_row,
    parse_scalar,
    preprocess,
    query_experiments_with_labels,
    resolve_experiments_from_class_config,
    resolve_negative_studies,
    set_inference_seeds,
    extract_checkpoint_metadata,
)
from remo_id.interpret import (
    base_rgb_from_tensor,
    is_vit_like,
    is_dino_like,
    dino_cls_attention,
    compute_signed_gradcam,
    get_target_layer_resnet,
    vit_attention_rollout,
    resize_mask_to_image,
    _MILOutputWrapper,
)
from remo_id.reports.pdf import (
    fig_to_png_bytes,
)


def fig_to_png_bytes_hires(fig, dpi: int = 400) -> bytes:
    """Convert a matplotlib figure to high-resolution PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return buf.getvalue()


@dataclass
class ImageScore:
    """Score for ranking images by activation."""
    experiment_id: str
    sample_id: str
    image_idx: int
    img_np: np.ndarray
    activation_score: float  # Max absolute activation in the image
    y_true: int


@dataclass
class CropInfo:
    """Information about a single crop."""
    experiment_id: str
    sample_id: str
    image_idx: int
    row: int
    col: int
    activation_value: float
    activation_sign: int  # +1 for positive, -1 for negative
    y_true: int
    crop_image: np.ndarray  # The cropped image with GradCAM overlay
    image_activation_score: float  # Overall image activation score
    full_overlay: np.ndarray = None  # Full image with GradCAM overlay (for bounding box view)


def extract_crop(
    img: np.ndarray,
    row: int,
    col: int,
    crop_size: int = 120,
) -> np.ndarray:
    """Extract a crop centered at (row, col) from an image.

    Args:
        img: Input image [H, W] or [H, W, C]
        row: Center row coordinate
        col: Center column coordinate
        crop_size: Size of the crop in pixels

    Returns:
        Cropped image of shape [crop_size, crop_size] or [crop_size, crop_size, C]
    """
    H, W = img.shape[:2]
    half = crop_size // 2

    # Calculate crop bounds with padding for edge cases
    r_start = max(0, row - half)
    r_end = min(H, row + half)
    c_start = max(0, col - half)
    c_end = min(W, col + half)

    crop = img[r_start:r_end, c_start:c_end]

    # Pad if necessary to achieve exact crop_size
    if crop.shape[0] < crop_size or crop.shape[1] < crop_size:
        if img.ndim == 3:
            padded = np.zeros((crop_size, crop_size, img.shape[2]), dtype=crop.dtype)
        else:
            padded = np.zeros((crop_size, crop_size), dtype=crop.dtype)

        # Calculate where to place the crop in the padded array
        pad_r = (crop_size - crop.shape[0]) // 2
        pad_c = (crop_size - crop.shape[1]) // 2
        padded[pad_r:pad_r+crop.shape[0], pad_c:pad_c+crop.shape[1]] = crop
        return padded

    return crop


def compute_image_activation_score(cam: np.ndarray) -> float:
    """Compute a single score representing the image's overall activation.

    Uses the max absolute activation value as the score, which captures
    how strongly any region in the image activates the model.

    Args:
        cam: 2D array of activation values (signed or unsigned)

    Returns:
        Score representing overall image importance
    """
    return float(np.max(np.abs(cam)))


def make_gallery_page(
    crops: List[CropInfo],
    title: str,
    subtitle: str,
    ncols: int = 5,
    max_crops_per_page: int = 20,
) -> bytes:
    """Create a gallery page with crops.

    Args:
        crops: List of CropInfo objects to display
        title: Page title
        subtitle: Page subtitle
        ncols: Number of columns in the grid
        max_crops_per_page: Maximum number of crops per page

    Returns:
        PNG bytes of the gallery page
    """
    crops = crops[:max_crops_per_page]
    n = len(crops)
    if n == 0:
        fig = plt.figure(figsize=(10, 2))
        plt.text(0.5, 0.5, "No crops to display", ha="center", va="center")
        plt.axis("off")
        return fig_to_png_bytes(fig)

    nrows = math.ceil(n / ncols)
    # Use letter page aspect ratio (8.5 x 11), fill the page
    fig = plt.figure(figsize=(11, 13))

    # Add title with minimal space
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.995)

    for i, crop in enumerate(crops):
        ax = fig.add_subplot(nrows, ncols, i + 1)

        # Display crop - no border, no labels
        if crop.crop_image.ndim == 2:
            ax.imshow(crop.crop_image, cmap="gray", interpolation='nearest')
        else:
            ax.imshow(crop.crop_image, interpolation='nearest')

        ax.axis("off")

    # Minimize spacing between subplots to fill the page
    plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.02, hspace=0.02)
    return fig_to_png_bytes_hires(fig)


def run_gallery(
    checkpoint: str,
    experiment_ids: List[int] = None,
    study_id: str = None,
    database_column: str = None,
    database_label: str = None,
    threshold: float = None,
    threshold_mode: str = "eq",
    num_crops: int = 50,
    num_images: int = 100,
    crop_size: int = 120,
    output: str = "gallery.pdf",
    model_name: str = "resnet18",
    task_type: str = "binary",
    num_classes: int = 2,
    seed: int = 403,
    negative_study_ids: list = None,
    negative_experiment_ids: list = None,
    top_images_per_experiment: int = 10,
    min_peak_distance: int = 30,
    peak_significance: float = 0.7,
    max_peaks_per_image: int = 3,
    normalize_global: bool = False,
    positive_only: bool = False,
    heatmap_opacity: float = 0.5,
    class_config: str = None,
    max_experiments_per_class: int = None,
    **kwargs
):
    """Generate a gallery of high-activation crops with GradCAM overlay.

    Two-pass approach:
    1. First pass: Compute activation scores for all images, rank by importance
    2. Second pass: Full GradCAM on top-activated images, extract significant peaks

    This function creates a PDF gallery showing high-resolution crops around
    the regions with highest GradCAM activation. The gallery is organized into
    two sections:

    1. "Positive Experiments - Diagnostic Features"
       Crops from experiments labeled positive, showing what the model identifies
       as diagnostic features.

    2. "Negative Experiments - False Positive Triggers"
       Crops from experiments labeled negative, showing what triggers false
       positive signals.

    Args:
        checkpoint: Path to model checkpoint
        experiment_ids: List of experiment IDs to analyze
        study_id: Study ID for label lookup
        database_column: Column for labeling
        database_label: Positive label value for categorical matching
        threshold: Threshold for numeric labels
        threshold_mode: How to apply threshold ("eq", "gt", "lt")
        num_crops: Total number of crops to show in gallery
        num_images: Number of images to scan per experiment (for scoring pass)
        crop_size: Size of crop in pixels
        output: Output PDF path
        model_name: Model architecture name
        task_type: Task type (binary or multiclass)
        num_classes: Number of classes for multiclass
        seed: Random seed
        negative_study_ids: Study IDs to merge as negatives
        negative_experiment_ids: Experiment IDs to merge as negatives
        top_images_per_experiment: Number of top-activated images to process per experiment
        min_peak_distance: Minimum distance between peaks in pixels
        peak_significance: Threshold for significant peaks (0-1, relative to max)
        normalize_global: If True, normalize all activation values to a global
            scale (99th percentile) so they are directly comparable across images.
    """
    # Set seeds
    set_inference_seeds(seed)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    # Extract metadata from checkpoint
    ckpt_meta = extract_checkpoint_metadata(checkpoint)
    class_names = ckpt_meta.get("class_names")

    # Load model
    print(f"Loading model {model_name} from {checkpoint}")
    model, device, in_channels = load_checkpoint(checkpoint, model_name, task_type, num_classes)
    augments = get_augmentations(model.backbone_name, crop_size=ckpt_meta.get("crop_size"))
    target_layer = None if (is_vit_like(model) or is_dino_like(model)) else get_target_layer_resnet(model, kwargs.get("gradcam_layer") or "layer4")

    # MIL models output a 0-dim scalar for binary tasks; wrap so the
    # pytorch-grad-cam library can index it as [B, C] (same as whole mode).
    from remo_id.models.mil import BaseMILModel
    gradcam_model = _MILGradCAMWrapper(model).to(device) if isinstance(model, BaseMILModel) else model

    # Check if model supports signed GradCAM (not ViT or DINO)
    if is_dino_like(model):
        print("DINOv3 model: using CLS token attention for visualization.")
    elif is_vit_like(model):
        print("Warning: ViT models use attention rollout instead of Grad-CAM.")
        print("Gallery will use attention weights instead of signed gradients.")

    # Get experiment data with labels
    if class_config:
        exp_ids = [str(x) for x in experiment_ids] if experiment_ids else None
        df, cc_class_names, cc_num_classes, cc_task_type = resolve_experiments_from_class_config(
            class_config, experiment_ids=exp_ids
        )
        class_names = cc_class_names
        task_type = cc_task_type
        num_classes = cc_num_classes
        df["y_true"] = df["y"]
    else:
        database_column = database_column or "operator_id"
        database_label = database_label or "True"
        database_label = parse_scalar(str(database_label))

        exp_ids = [str(x) for x in experiment_ids] if experiment_ids else []
        exp_ids = list(dict.fromkeys(exp_ids))
        df = query_experiments_with_labels(study_id, database_column, exp_ids)

        # Handle negative studies/experiments
        df = resolve_negative_studies(df, database_column, negative_study_ids, negative_experiment_ids)

        if len(df) == 0:
            raise RuntimeError("No experiments found.")

        # Build class_mapping for multiclass from class_definitions
        class_mapping = None
        if task_type == "multiclass" and ckpt_meta.get("class_definitions"):
            class_defs = ckpt_meta["class_definitions"]
            unique_names = list(dict.fromkeys(d["name"] for d in class_defs))
            class_mapping = {}
            for d in class_defs:
                cls_idx = unique_names.index(d["name"])
                if "value" in d:
                    class_mapping[parse_scalar(str(d["value"]))] = cls_idx

        # Compute labels
        def compute_label(row):
            if row.get("_is_forced_negative", False):
                return 0
            return label_from_row(
                row, database_column, threshold_mode, database_label, threshold,
                task_type=task_type, class_mapping=class_mapping,
            )

        df["y_true"] = df.apply(compute_label, axis=1)

    if len(df) == 0:
        raise RuntimeError("No experiments found.")

    # Apply max_experiments_per_class filtering
    if max_experiments_per_class is not None:
        filtered_dfs = []
        for cls_idx in sorted(df["y_true"].unique()):
            cls_df = df[df["y_true"] == cls_idx]
            if len(cls_df) > max_experiments_per_class:
                cls_df = cls_df.sample(n=max_experiments_per_class, random_state=seed)
            filtered_dfs.append(cls_df)
        df = pd.concat(filtered_dfs, ignore_index=True)

    if task_type == "multiclass" and class_names:
        print(f"Found {len(df)} experiments")
        for ci, name in enumerate(class_names):
            count = (df['y_true'] == ci).sum()
            print(f"  {name} (class {ci}): {count}")
    else:
        print(f"Found {len(df)} experiments")
        print(f"  Positive: {(df['y_true'] == 1).sum()}")
        print(f"  Negative: {(df['y_true'] == 0).sum()}")

    # =========================================================================
    # PASS 1: Score all images by activation magnitude
    # =========================================================================
    print("\n=== Pass 1: Scoring images by activation ===")

    all_image_scores: List[ImageScore] = []

    for exp_idx, (_, row) in enumerate(df.iterrows()):
        exp_id = str(row["experiment_id"])
        sample_id = str(row.get("sample_id", ""))
        path = row["path"]
        n_avail = int(row["num_images"])
        y_true = int(row["y_true"])

        # Select images to scan
        n_pick = min(num_images, n_avail)
        if n_pick <= 0:
            continue

        rng = random.Random(seed + exp_idx)
        img_ids = rng.sample(range(n_avail), n_pick) if n_pick < n_avail else list(range(n_avail))

        print(f"[{exp_idx+1}/{len(df)}] {exp_id}: scoring {len(img_ids)} images...")

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

        # Score each image
        for i, img_np in enumerate(all_images):
            if img_np is None:
                continue

            try:
                img_np = np.array(img_np)
                if img_np.ndim == 3 and img_np.shape[-1] == 1:
                    img_np = img_np[..., 0]

                # Preprocess for model
                x = preprocess(img_np, augments=augments, target_channels=in_channels).to(device)

                # Compute activation score for image ranking
                if is_vit_like(model):
                    # For ViT models, use prediction probability as activation score
                    with torch.no_grad():
                        out = model(x)
                        if hasattr(out, "logits"):
                            out = out.logits
                        if task_type == "multiclass":
                            probs = torch.softmax(out, dim=-1).cpu().squeeze()
                            activation_score = probs.max().item()
                        else:
                            activation_score = torch.sigmoid(out).cpu().item()
                else:
                    # For ResNet models, use raw signed GradCAM
                    raw_cam, _ = compute_signed_gradcam(model, x, target_layer=target_layer, normalize=False)
                    activation_score = compute_image_activation_score(raw_cam)

                all_image_scores.append(ImageScore(
                    experiment_id=exp_id,
                    sample_id=sample_id,
                    image_idx=img_ids[i],
                    img_np=img_np.copy(),
                    activation_score=activation_score,
                    y_true=y_true,
                ))

            except Exception as e:
                print(f"[{exp_id}] Image {img_ids[i]} scoring failed: {e}")
                continue

    print(f"Scored {len(all_image_scores)} images total")

    # =========================================================================
    # Select top images per experiment
    # =========================================================================
    print(f"\n=== Selecting top {top_images_per_experiment} images per experiment ===")

    # Group by experiment and select top images
    from collections import defaultdict
    exp_images = defaultdict(list)
    for score in all_image_scores:
        exp_images[score.experiment_id].append(score)

    selected_images: List[ImageScore] = []
    for exp_id, images in exp_images.items():
        # Sort by activation score descending
        images.sort(key=lambda x: x.activation_score, reverse=True)
        # Take top N
        top_images = images[:top_images_per_experiment]
        selected_images.extend(top_images)
        print(f"  {exp_id}: selected {len(top_images)} images (top score: {top_images[0].activation_score:.4f})")

    print(f"Selected {len(selected_images)} images for detailed analysis")

    # =========================================================================
    # PASS 2: Full GradCAM and crop extraction on selected images
    # =========================================================================
    print("\n=== Pass 2: Full GradCAM on selected images ===")

    # Collect CAMs using standard GradCAM (same as inference report)
    selected_cam_data = []

    use_attention_rollout = is_vit_like(model)
    for i, img_score in enumerate(selected_images):
        try:
            # Preprocess for model
            x = preprocess(img_score.img_np, augments=augments, target_channels=in_channels).to(device)
            base_rgb = base_rgb_from_tensor(x)  # RGB in [0, 1] range

            if use_attention_rollout:
                # Use attention rollout for ViT models
                mask = vit_attention_rollout(model, x, head_fusion="mean", discard_ratio=0.0)
                H, W = base_rgb.shape[:2]
                grayscale_cam = resize_mask_to_image(mask, H, W)
            else:
                # Compute standard GradCAM for ResNet models
                x = x.requires_grad_(True)
                # For multiclass, target the predicted class; for binary, target class 0
                if task_type == "multiclass":
                    with torch.no_grad():
                        out = model(x)
                        if hasattr(out, "logits"):
                            out = out.logits
                        target_class = torch.softmax(out, dim=-1).argmax(dim=-1).item()
                else:
                    target_class = 0
                targets = [ClassifierOutputTarget(target_class)]

                with torch.enable_grad():
                    with GradCAM(model=gradcam_model, target_layers=[target_layer]) as cam:
                        grayscale_cam = cam(input_tensor=x, targets=targets)[0]

            selected_cam_data.append({
                "img_score": img_score,
                "grayscale_cam": grayscale_cam,  # Standard CAM [0, 1]
                "base_rgb": base_rgb,
            })

        except Exception as e:
            print(f"[{img_score.experiment_id}] Image {img_score.image_idx} CAM failed: {e}")
            continue

    print(f"Computed {len(selected_cam_data)} detailed CAMs")

    # =========================================================================
    # Extract crops with GradCAM overlay from significant peaks
    # =========================================================================
    print("\n=== Extracting crops from significant peaks ===")

    all_crops: List[CropInfo] = []

    for data in selected_cam_data:
        img_score = data["img_score"]
        grayscale_cam = data["grayscale_cam"]  # Standard CAM [0, 1]
        base_rgb = data["base_rgb"]  # RGB in [0, 1]
        img_np = img_score.img_np

        # Create overlay using show_cam_on_image (same as inference report)
        # image_weight=0.85 means 85% original image, 15% heatmap (inference default)
        # Lower image_weight = more prominent heatmap
        image_weight = 1.0 - heatmap_opacity
        overlay = show_cam_on_image(base_rgb, grayscale_cam, use_rgb=True, image_weight=image_weight)

        # Find peaks from the grayscale CAM
        # Standard GradCAM is always positive [0, 1], so all peaks are "positive"
        cam_max = grayscale_cam.max()
        if cam_max > 0:
            threshold = peak_significance * cam_max
            peaks = []

            # Find local maxima above threshold
            from scipy import ndimage
            labeled, num_features = ndimage.label(grayscale_cam >= threshold)
            for i in range(1, num_features + 1):
                region = (labeled == i)
                region_vals = grayscale_cam * region
                max_idx = np.unravel_index(region_vals.argmax(), region_vals.shape)
                val = float(grayscale_cam[max_idx])
                peaks.append((max_idx[0], max_idx[1], val, +1))  # All peaks are positive

            # Sort by value descending and filter by distance
            peaks.sort(key=lambda x: x[2], reverse=True)
            filtered = []
            for peak in peaks:
                too_close = False
                for existing in filtered:
                    dist = np.sqrt((peak[0] - existing[0])**2 + (peak[1] - existing[1])**2)
                    if dist < min_peak_distance:
                        too_close = True
                        break
                if not too_close:
                    filtered.append(peak)
                if len(filtered) >= max_peaks_per_image:
                    break
            peaks = filtered
        else:
            peaks = []

        if not peaks:
            continue

        # Create crops for each peak
        H, W = img_np.shape[:2]
        cam_h, cam_w = grayscale_cam.shape

        for row_idx, col_idx, value, sign in peaks:
            # Scale coordinates from cam size to original image size
            orig_row = int(row_idx * H / cam_h)
            orig_col = int(col_idx * W / cam_w)

            # Extract crop from the OVERLAY image (with GradCAM visualization)
            crop_overlay = extract_crop(overlay, orig_row, orig_col, crop_size)

            crop_info = CropInfo(
                experiment_id=img_score.experiment_id,
                sample_id=img_score.sample_id,
                image_idx=img_score.image_idx,
                row=orig_row,
                col=orig_col,
                activation_value=value,
                activation_sign=sign,
                y_true=img_score.y_true,
                crop_image=crop_overlay,
                image_activation_score=img_score.activation_score,
                full_overlay=overlay,  # Store full image for bounding box visualization
            )
            all_crops.append(crop_info)

    print(f"Collected {len(all_crops)} crops total")

    # Sort all crops by image activation score first, then peak value
    # This prioritizes crops from highly-activated images
    all_crops.sort(key=lambda c: (c.image_activation_score, abs(c.activation_value)), reverse=True)

    # Build PDF - crops only, no title/legend pages
    c_pdf = canvas.Canvas(output, pagesize=letter)
    w, h = letter
    crops_per_page = 35  # 7 columns x 5 rows

    if task_type == "multiclass" and class_names:
        # Per-class sections for multiclass
        n_classes = len(class_names)
        crops_per_class = max(1, num_crops // n_classes)

        for ci, cls_name in enumerate(class_names):
            cls_crops = [c for c in all_crops if c.y_true == ci][:crops_per_class]
            print(f"Selected {len(cls_crops)} crops for class '{cls_name}'")

            if cls_crops:
                for page_idx in range(0, len(cls_crops), crops_per_page):
                    page_crops = cls_crops[page_idx:page_idx + crops_per_page]
                    page_num = page_idx // crops_per_page + 1
                    total_pages = math.ceil(len(cls_crops) / crops_per_page)
                    gallery_png = make_gallery_page(
                        page_crops,
                        title=f"Class: {cls_name} - Features (Page {page_num}/{total_pages})",
                        subtitle=f"Crops {page_idx+1}-{page_idx+len(page_crops)} of {len(cls_crops)}",
                        ncols=7,
                        max_crops_per_page=crops_per_page,
                    )
                    img = ImageReader(io.BytesIO(gallery_png))
                    c_pdf.drawImage(img, 0.25*inch, 0.25*inch, width=8.0*inch, height=10.5*inch,
                                    preserveAspectRatio=True, mask='auto')
                    c_pdf.showPage()
    else:
        # Binary: split into positive and negative experiments
        pos_crops = [c for c in all_crops if c.y_true == 1]
        neg_crops = [c for c in all_crops if c.y_true == 0]

        n_pos = num_crops // 2
        n_neg = num_crops - n_pos
        pos_crops = pos_crops[:n_pos]
        neg_crops = neg_crops[:n_neg]

        print(f"Selected {len(pos_crops)} crops from positive experiments")
        print(f"Selected {len(neg_crops)} crops from negative experiments")

        if pos_crops:
            for page_idx in range(0, len(pos_crops), crops_per_page):
                page_crops = pos_crops[page_idx:page_idx + crops_per_page]
                page_num = page_idx // crops_per_page + 1
                total_pages = math.ceil(len(pos_crops) / crops_per_page)
                gallery_png = make_gallery_page(
                    page_crops,
                    title=f"Positive Experiments - Diagnostic Features (Page {page_num}/{total_pages})",
                    subtitle=f"Crops {page_idx+1}-{page_idx+len(page_crops)} of {len(pos_crops)}",
                    ncols=7,
                    max_crops_per_page=crops_per_page,
                )
                img = ImageReader(io.BytesIO(gallery_png))
                c_pdf.drawImage(img, 0.25*inch, 0.25*inch, width=8.0*inch, height=10.5*inch,
                                preserveAspectRatio=True, mask='auto')
                c_pdf.showPage()

        if neg_crops:
            for page_idx in range(0, len(neg_crops), crops_per_page):
                page_crops = neg_crops[page_idx:page_idx + crops_per_page]
                page_num = page_idx // crops_per_page + 1
                total_pages = math.ceil(len(neg_crops) / crops_per_page)
                gallery_png = make_gallery_page(
                    page_crops,
                    title=f"Negative Experiments - False Positive Triggers (Page {page_num}/{total_pages})",
                    subtitle=f"Crops {page_idx+1}-{page_idx+len(page_crops)} of {len(neg_crops)}",
                    ncols=7,
                    max_crops_per_page=crops_per_page,
                )
                img = ImageReader(io.BytesIO(gallery_png))
                c_pdf.drawImage(img, 0.25*inch, 0.25*inch, width=8.0*inch, height=10.5*inch,
                                preserveAspectRatio=True, mask='auto')
                c_pdf.showPage()

    c_pdf.save()
    print(f"Saved gallery PDF: {output}")

    # Print summary statistics
    if pos_crops:
        pos_activations = [c.activation_value for c in pos_crops]
        print(f"\nPositive experiments activation stats:")
        print(f"  Mean: {np.mean(pos_activations):.4f}")
        print(f"  Max:  {np.max(pos_activations):.4f}")

    if neg_crops:
        neg_activations = [c.activation_value for c in neg_crops]
        print(f"\nNegative experiments activation stats:")
        print(f"  Mean: {np.mean(neg_activations):.4f}")
        print(f"  Max:  {np.max(neg_activations):.4f}")


def make_experiment_page(
    images: List[Dict],
    experiment_id: str,
    sample_id: str,
    label: int,
    heatmap_opacity: float = 0.15,
    class_names: List[str] = None,
    class_probs: List[float] = None,
    pred_class: int = None,
    metadata: dict = None,
    study_id: str = None,
    split: str = None,
    minimal: bool = False,
) -> bytes:
    """Create a grid page showing top whole images with GradCAM overlay.

    Args:
        images: List of dicts with 'base_rgb', 'grayscale_cam', 'activation_score'
        experiment_id: Experiment ID for title
        sample_id: Sample ID for title
        label: Label value (class index)
        heatmap_opacity: Opacity of heatmap overlay
        class_names: List of class names for multiclass display
        class_probs: Mean per-class probabilities across images
        pred_class: Predicted class index
        metadata: Study table metadata dict for this experiment
        study_id: Study ID for display
        split: Train/val split label for this experiment

    Returns:
        PNG bytes of the page
    """
    n = len(images)
    ncols = min(3, n) if n > 4 else min(2, n)
    nrows = math.ceil(n / ncols)

    if minimal:
        header_height = 0
        fig = plt.figure(figsize=(11, 3.5 * nrows), facecolor="black")
        top_margin = 1.0
    else:
        # Calculate extra header space for metadata/probabilities
        header_lines = 2  # base title lines
        if class_names and class_probs is not None:
            header_lines += 1  # class probabilities line
        if metadata:
            # Show up to 3 metadata items per line, max 2 lines
            n_meta = min(len(metadata), 6)
            header_lines += math.ceil(n_meta / 3)

        header_height = 0.3 * header_lines + 0.3
        fig = plt.figure(figsize=(11, 3.5 * nrows + header_height + 0.5))

        # Build title
        split_tag = f" [{split}]" if split else ""
        if class_names and len(class_names) > 2:
            # Multiclass: show true class name and predicted class
            true_name = class_names[label] if label < len(class_names) else f"Class {label}"
            if pred_class is not None and pred_class < len(class_names):
                pred_name = class_names[pred_class]
                pred_conf = class_probs[pred_class] if class_probs else 0
                title = f"Experiment: {experiment_id}{split_tag} | True: {true_name} | Pred: {pred_name} ({pred_conf:.2f})"
            else:
                title = f"Experiment: {experiment_id}{split_tag} | True: {true_name}"
        else:
            # Binary
            label_str = "Positive" if label == 1 else "Negative"
            title = f"Experiment: {experiment_id}{split_tag} ({label_str})"

        # Second line: sample and study info
        subtitle_parts = []
        if sample_id and sample_id != "None":
            subtitle_parts.append(f"Sample: {sample_id}")
        if study_id:
            subtitle_parts.append(f"Study: {study_id}")
        subtitle = " | ".join(subtitle_parts) if subtitle_parts else ""

        # Build full title text
        title_text = title
        if subtitle:
            title_text += f"\n{subtitle}"

        # Add class probabilities line for multiclass
        if class_names and class_probs is not None and len(class_names) > 2:
            prob_parts = [f"{name}: {prob:.2f}" for name, prob in zip(class_names, class_probs)]
            title_text += f"\n{' | '.join(prob_parts)}"

        # Add metadata line
        if metadata:
            skip_keys = {"experiment_id", "sample_id", "study_id"}
            meta_items = [(k, v) for k, v in metadata.items()
                          if k not in skip_keys and v is not None and str(v) != "nan"]
            if meta_items:
                meta_str = ", ".join(f"{k}={v}" for k, v in meta_items[:6])
                title_text += f"\n{meta_str}"

        top_margin = 1.0 - header_height / (3.5 * nrows + header_height + 0.5)
        fig.suptitle(title_text, fontsize=10, fontweight="bold", y=0.99,
                     va="top", linespacing=1.4)

    for i, img_data in enumerate(images):
        ax = fig.add_subplot(nrows, ncols, i + 1)

        base_rgb = img_data["base_rgb"]
        grayscale_cam = img_data["grayscale_cam"]
        score = img_data["activation_score"]
        rank_type = img_data.get("rank_type", "")

        # Create overlay
        image_weight = 1.0 - heatmap_opacity
        overlay = show_cam_on_image(base_rgb, grayscale_cam, use_rgb=True, image_weight=image_weight)

        ax.imshow(overlay)
        if not minimal:
            rank_label = f" ({rank_type})" if rank_type else ""
            ax.set_title(f"Score: {score:.4f}{rank_label}", fontsize=10)
        ax.axis("off")

    if minimal:
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.02, hspace=0.02)
    else:
        plt.subplots_adjust(left=0.02, right=0.98, top=top_margin, bottom=0.02, wspace=0.05, hspace=0.10)
    return fig_to_png_bytes_hires(fig)


_MILGradCAMWrapper = _MILOutputWrapper  # Alias for backward compat in this module


def run_whole_image_gallery(
    checkpoint: str,
    experiment_ids: List[int] = None,
    study_id: str = None,
    database_column: str = None,
    database_label: str = None,
    threshold: float = None,
    threshold_mode: str = "eq",
    num_images: int = 1000,
    top_images: int = 4,
    bottom_images: int = 0,
    output: str = "whole_image_gallery.pdf",
    model_name: str = "resnet18",
    task_type: str = "binary",
    num_classes: int = 2,
    seed: int = 403,
    negative_study_ids: list = None,
    negative_experiment_ids: list = None,
    heatmap_opacity: float = 0.15,
    class_config: str = None,
    max_experiments_per_class: int = None,
    minimal: bool = False,
    **kwargs
):
    """Generate a gallery of whole images with GradCAM overlay.

    For each experiment:
    1. Score `num_images` images by activation
    2. Select top `top_images` (default 4)
    3. Generate 2x2 page with full GradCAM overlays

    Args:
        checkpoint: Path to model checkpoint
        experiment_ids: List of experiment IDs to analyze
        study_id: Study ID for label lookup
        database_column: Column for labeling
        database_label: Positive label value for categorical matching
        threshold: Threshold for numeric labels
        threshold_mode: How to apply threshold ("eq", "gt", "lt")
        num_images: Number of images to scan per experiment
        top_images: Number of top images to show per experiment (default 4)
        output: Output PDF path
        model_name: Model architecture name
        task_type: Task type (binary or multiclass)
        num_classes: Number of classes for multiclass
        seed: Random seed
        negative_study_ids: Study IDs to merge as negatives
        negative_experiment_ids: Experiment IDs to merge as negatives
        heatmap_opacity: Opacity of GradCAM heatmap (0-1)
    """
    # Set seeds
    set_inference_seeds(seed)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    # Extract metadata from checkpoint
    ckpt_meta = extract_checkpoint_metadata(checkpoint)
    class_names = ckpt_meta.get("class_names")

    # Load model
    print(f"Loading model {model_name} from {checkpoint}")
    model, device, in_channels = load_checkpoint(checkpoint, model_name, task_type, num_classes)
    augments = get_augmentations(model.backbone_name, crop_size=ckpt_meta.get("crop_size"))

    # For MIL models, wrap for GradCAM compatibility
    from remo_id.models.mil import BaseMILModel
    is_mil = isinstance(model, BaseMILModel)
    if is_mil:
        gradcam_model = _MILGradCAMWrapper(model).to(device)
    else:
        gradcam_model = model

    target_layer = None if (is_vit_like(model) or is_dino_like(model)) else get_target_layer_resnet(model, kwargs.get("gradcam_layer") or "layer4")

    if is_dino_like(model):
        print("DINOv3 model: using CLS token attention for visualization.")
    elif is_vit_like(model):
        print("Warning: ViT models use attention rollout instead of Grad-CAM.")

    # Get experiment data with labels
    if class_config:
        exp_ids = [str(x) for x in experiment_ids] if experiment_ids else None
        df, cc_class_names, cc_num_classes, cc_task_type = resolve_experiments_from_class_config(
            class_config, experiment_ids=exp_ids
        )
        class_names = cc_class_names
        task_type = cc_task_type
        num_classes = cc_num_classes
    else:
        database_column = database_column or "operator_id"
        database_label = database_label or "True"
        database_label = parse_scalar(str(database_label))

        exp_ids = [str(x) for x in experiment_ids] if experiment_ids else []
        exp_ids = list(dict.fromkeys(exp_ids))
        df = query_experiments_with_labels(study_id, database_column, exp_ids)

        # Handle negative studies/experiments
        df = resolve_negative_studies(df, database_column, negative_study_ids, negative_experiment_ids)

        if len(df) == 0:
            raise ValueError("No experiments found")

        # Determine labels
        label_mode = "eq" if threshold is None else threshold_mode

        def get_label(row):
            if row["_is_forced_negative"]:
                return 0
            return label_from_row(row, database_column, label_mode, database_label, threshold)

        df["y"] = df.apply(get_label, axis=1)

    if len(df) == 0:
        raise ValueError("No experiments found")

    # Load splits to tag train/val
    splits_path = os.path.join(os.path.dirname(checkpoint), "splits.yaml")
    val_experiment_ids = set()
    if os.path.exists(splits_path):
        import yaml
        with open(splits_path) as f:
            splits_data = yaml.safe_load(f)
        for cls_exps in (splits_data.get("val") or {}).values():
            val_experiment_ids.update(str(e) for e in cls_exps)
        df["split"] = df["experiment_id"].astype(str).apply(
            lambda x: "val" if x in val_experiment_ids else "train"
        )
        print(f"Loaded splits: {len(val_experiment_ids)} val experiments")
    else:
        df["split"] = "unknown"

    # Apply max_experiments_per_class filtering (prefer val experiments)
    if max_experiments_per_class is not None:
        filtered_dfs = []
        for cls_idx in sorted(df["y"].unique()):
            cls_df = df[df["y"] == cls_idx]
            if len(cls_df) > max_experiments_per_class:
                val_df = cls_df[cls_df["split"] == "val"]
                train_df = cls_df[cls_df["split"] != "val"]
                if len(val_df) >= max_experiments_per_class:
                    cls_df = val_df.sample(n=max_experiments_per_class, random_state=seed)
                else:
                    n_train = max_experiments_per_class - len(val_df)
                    train_sample = train_df.sample(n=min(n_train, len(train_df)), random_state=seed)
                    cls_df = pd.concat([val_df, train_sample], ignore_index=True)
            filtered_dfs.append(cls_df)
        df = pd.concat(filtered_dfs, ignore_index=True)

    # Sort by class, then val first within each class
    split_order = df["split"].map({"val": 0, "train": 1, "unknown": 2})
    df = df.assign(_split_order=split_order).sort_values(["y", "_split_order"]).drop(columns="_split_order").reset_index(drop=True)

    if task_type == "multiclass" and class_names:
        print(f"Found {len(df)} experiments")
        for ci, name in enumerate(class_names):
            count = (df['y'] == ci).sum()
            print(f"  {name} (class {ci}): {count}")
    else:
        print(f"Found {len(df)} experiments")
        print(f"  Positive: {(df['y'] == 1).sum()}")
        print(f"  Negative: {(df['y'] == 0).sum()}")

    # Create PDF
    c_pdf = canvas.Canvas(output, pagesize=letter)

    # Track current class for section breaks
    current_class = None

    # Process each experiment
    for exp_counter, (idx, row) in enumerate(df.iterrows()):
        # Add section break page when class changes (skip in minimal mode)
        exp_class = int(row["y"])
        if not minimal and exp_class != current_class:
            current_class = exp_class
            if class_names and exp_class < len(class_names):
                section_title = f"Class: {class_names[exp_class]}"
            elif task_type == "binary":
                section_title = "Positive" if exp_class == 1 else "Negative"
            else:
                section_title = f"Class {exp_class}"
            cls_count = (df["y"] == exp_class).sum()
            section_subtitle = f"{cls_count} experiments"

            fig, ax = plt.subplots(figsize=(11, 8.5))
            fig.patch.set_facecolor("white")
            ax.text(0.5, 0.55, section_title, fontsize=36, fontweight="bold",
                    ha="center", va="center", transform=ax.transAxes)
            ax.text(0.5, 0.42, section_subtitle, fontsize=18, color="gray",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            section_png = fig_to_png_bytes_hires(fig)
            img = ImageReader(io.BytesIO(section_png))
            c_pdf.drawImage(img, 0.25*inch, 0.25*inch, width=8.0*inch, height=10.5*inch,
                            preserveAspectRatio=True, mask=None)
            c_pdf.showPage()

        exp_id = str(row["experiment_id"])
        sample_id = str(row.get("sample_id", ""))
        exp_study_id = str(row.get("study_id", study_id or ""))
        exp_split = row.get("split", "unknown")
        path = row["path"]
        n_imgs = int(row["num_images"])
        y_true = int(row["y"])

        print(f"\n[{exp_counter+1}/{len(df)}] {exp_id}: scanning {min(num_images, n_imgs)} images...")

        # Select image indices
        n_to_score = min(num_images, n_imgs)
        if n_to_score < n_imgs:
            img_indices = random.sample(range(n_imgs), n_to_score)
        else:
            img_indices = list(range(n_imgs))

        # Score images
        image_scores = []
        all_image_probs = []  # Collect per-class probs for multiclass
        for img_idx in img_indices:
            try:
                img_np = get_image_from_zarr(path, img_idx)
                if img_np is None:
                    continue

                x = preprocess(img_np, augments=augments, target_channels=in_channels).to(device)

                with torch.no_grad():
                    out = model(x)
                    if hasattr(out, "logits"):
                        out = out.logits

                    if task_type == "multiclass":
                        probs = torch.softmax(out, dim=-1).cpu().squeeze()
                        pred_cls = probs.argmax().item()
                        score = probs[pred_cls].item()
                        all_probs = probs.tolist()
                    else:
                        score = torch.sigmoid(out).cpu().item()
                        pred_cls = int(score > 0.5)
                        all_probs = [1 - score, score]

                image_scores.append({
                    "img_idx": img_idx,
                    "img_np": img_np,
                    "score": score,
                    "pred_class": pred_cls,
                    "all_probs": all_probs,
                })
                all_image_probs.append(all_probs)

            except Exception as e:
                print(f"  Error scoring image {img_idx}: {e}")
                continue

        if len(image_scores) == 0:
            print(f"  No valid images found, skipping")
            continue

        # Compute mean per-class probabilities across all scored images
        mean_class_probs = None
        exp_pred_class = None
        if all_image_probs:
            mean_class_probs = [float(x) for x in np.mean(all_image_probs, axis=0)]
            exp_pred_class = int(np.argmax(mean_class_probs))

        # Sort by score and select top + bottom
        image_scores.sort(key=lambda x: x["score"], reverse=True)
        selected = []
        top_list = image_scores[:top_images]
        for item in top_list:
            item["rank_type"] = "top"
            selected.append(item)
        if bottom_images > 0:
            bot_list = image_scores[-bottom_images:]
            for item in bot_list:
                item["rank_type"] = "bottom"
            selected.extend(bot_list)
        top_images_list = selected
        scores_str = [f'{x["score"]:.4f}' for x in top_images_list]
        print(f"  Selected {len(top_images_list)} images (scores: {scores_str})")

        # Fetch study metadata for display
        metadata = None
        if exp_study_id and exp_study_id != "None":
            metadata = get_experiment_metadata(exp_id, exp_study_id)

        # Compute activation maps for top images (DINO attention, ViT rollout, or GradCAM)
        page_images = []
        use_dino_attention = is_dino_like(model)
        use_attention_rollout = is_vit_like(model)
        for img_data in top_images_list:
            try:
                img_np = img_data["img_np"]
                x = preprocess(img_np, augments=augments, target_channels=in_channels).to(device)
                base_rgb = base_rgb_from_tensor(x)

                if use_dino_attention:
                    # Use CLS token attention from DINOv3
                    mask = dino_cls_attention(model, x, layer_idx=-1)
                    H, W = base_rgb.shape[:2]
                    grayscale_cam = resize_mask_to_image(mask, H, W)
                elif use_attention_rollout:
                    # Use attention rollout for ViT models
                    mask = vit_attention_rollout(model, x, head_fusion="mean", discard_ratio=0.0)
                    H, W = base_rgb.shape[:2]
                    grayscale_cam = resize_mask_to_image(mask, H, W)
                else:
                    # Use GradCAM for ResNet models (use wrapper for MIL)
                    x = x.requires_grad_(True)
                    # For multiclass, target the predicted class; for binary, always target class 0
                    target_class = 0 if task_type == "binary" else img_data.get("pred_class", 0)
                    targets = [ClassifierOutputTarget(target_class)]

                    with torch.enable_grad():
                        with GradCAM(model=gradcam_model, target_layers=[target_layer]) as cam:
                            grayscale_cam = cam(input_tensor=x, targets=targets)[0]

                page_images.append({
                    "base_rgb": base_rgb,
                    "grayscale_cam": grayscale_cam,
                    "activation_score": img_data["score"],
                    "rank_type": img_data.get("rank_type", ""),
                })

            except Exception as e:
                import traceback
                print(f"  Activation map failed for image: {e}")
                traceback.print_exc()
                continue

        if len(page_images) == 0:
            print(f"  No GradCAM computed, skipping")
            continue

        # Create page for this experiment
        page_png = make_experiment_page(
            page_images,
            experiment_id=exp_id,
            sample_id=sample_id,
            label=y_true,
            heatmap_opacity=heatmap_opacity,
            class_names=class_names,
            class_probs=mean_class_probs,
            pred_class=exp_pred_class,
            metadata=metadata,
            study_id=exp_study_id if exp_study_id != "None" else None,
            split=exp_split,
            minimal=minimal,
        )

        img = ImageReader(io.BytesIO(page_png))
        c_pdf.drawImage(img, 0.25*inch, 0.25*inch, width=8.0*inch, height=10.5*inch,
                        preserveAspectRatio=True, mask=None)
        c_pdf.showPage()

    c_pdf.save()
    print(f"\nSaved whole image gallery PDF: {output}")
