"""CPU-only image processing and overlay utilities."""

import numpy as np
import torch
import torch.nn.functional as F


def base_rgb_from_tensor(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to RGB float array for visualization.

    Args:
        x: [1,C,H,W] tensor in any value range (may be normalized)
           C can be 1 (grayscale) or 3 (RGB)

    Returns:
        [H,W,3] float array in [0,1] for visualization
    """
    base = x.detach()[0].float()  # [C, H, W]
    base = base - base.min()
    base = base / (base.max().clamp_min(1e-6))

    # Convert single-channel to 3-channel grayscale for visualization
    if base.shape[0] == 1:
        base = base.repeat(3, 1, 1)  # [3, H, W]

    return base.permute(1, 2, 0).cpu().numpy()


def tensor_to_rgb_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to RGB numpy array for visualization.

    Args:
        x: [C,H,W] float tensor (may be normalized)

    Returns:
        [H,W,3] uint8 RGB array
    """
    base = x.detach().cpu()

    # Handle single-channel input by converting to 3-channel grayscale
    if base.shape[0] == 1:
        base = base.repeat(3, 1, 1)  # [1,H,W] -> [3,H,W]

    base = (base - base.min()) / (base.max() - base.min() + 1e-6)
    rgb = (base.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return rgb


def resize_mask_to_image(mask_hw: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize a mask to match image dimensions.

    Args:
        mask_hw: [h,w] or [h,w,1] mask in [0,1]
        H: Target height
        W: Target width

    Returns:
        [H,W] mask in [0,1]
    """
    m = torch.from_numpy(mask_hw).float()
    if m.ndim == 2:
        m = m[None, None, :, :]          # [1,1,h,w]
    elif m.ndim == 3:
        m = m.permute(2, 0, 1)[None, ...]  # [1,1,h,w] if last dim was 1
    m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
    m = m[0, 0].cpu().numpy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m


def fuse_heads(attn: torch.Tensor, head_fusion: str = "mean") -> torch.Tensor:
    """Fuse multi-head attention weights.

    Args:
        attn: [B, heads, S, S] attention weights
        head_fusion: "mean", "max", or "min"

    Returns:
        [B, S, S] fused attention
    """
    if head_fusion == "mean":
        return attn.mean(dim=1)
    if head_fusion == "max":
        return attn.max(dim=1).values
    if head_fusion == "min":
        return attn.min(dim=1).values
    raise ValueError("head_fusion must be one of: mean|max|min")


def make_signed_overlay(
    base_rgb01: np.ndarray,
    signed_cam: np.ndarray,
    image_weight: float = 0.6,
    colormap: str = "RdBu_r",
) -> np.ndarray:
    """Create an overlay visualization with a diverging colormap for signed Grad-CAM.

    Args:
        base_rgb01: Base RGB image in [0, 1] range, shape [H, W, 3]
        signed_cam: Signed Grad-CAM in [-1, 1] range, shape [H, W]
        image_weight: Weight for blending (0 = all heatmap, 1 = all image)
        colormap: Matplotlib colormap name (diverging recommended)

    Returns:
        RGB uint8 array of shape [H, W, 3] with overlay visualization

    Interpretation:
        - Red regions: Features pushing toward positive class prediction
        - Blue regions: Features pushing toward negative class prediction
        - White regions: Neutral/low importance
    """
    import matplotlib.cm as cm

    # Get colormap
    cmap = cm.get_cmap(colormap)

    # Map signed_cam from [-1, 1] to [0, 1] for colormap lookup
    cam_normalized = (signed_cam + 1) / 2  # Now in [0, 1]
    cam_normalized = np.clip(cam_normalized, 0, 1)

    # Apply colormap
    heatmap_rgba = cmap(cam_normalized)  # [H, W, 4]
    heatmap_rgb = heatmap_rgba[:, :, :3]  # [H, W, 3]

    # Blend with original image
    overlay = image_weight * base_rgb01 + (1 - image_weight) * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)

    return (overlay * 255).astype(np.uint8)


def normalize_cams_globally(
    cams: list,
    percentile: float = 99.0,
) -> list:
    """Normalize a list of CAM arrays to a common global scale.

    Args:
        cams: List of 2D numpy arrays (raw, unnormalized CAM values)
        percentile: Percentile of absolute values to use as the normalization factor

    Returns:
        List of normalized CAM arrays in [-1, 1] range
    """
    if not cams:
        return cams

    # Collect all absolute values directly as numpy
    all_abs_values = np.concatenate([np.abs(cam).ravel() for cam in cams])

    if len(all_abs_values) == 0:
        return cams

    # Compute normalization factor from percentile
    norm_factor = np.percentile(all_abs_values, percentile)
    norm_factor = max(norm_factor, 1e-8)  # Avoid division by zero

    # Normalize all cams
    normalized = []
    for cam in cams:
        norm_cam = cam / norm_factor
        norm_cam = np.clip(norm_cam, -1, 1)  # Clip to [-1, 1] after normalization
        normalized.append(norm_cam)

    return normalized


def find_activation_peaks(
    cam: np.ndarray,
    num_peaks: int = 10,
    min_distance: int = 20,
    threshold_rel: float = 0.3,
) -> list:
    """Find local maxima in a Grad-CAM heatmap.

    Args:
        cam: 2D array of activation values (can be signed or unsigned)
        num_peaks: Maximum number of peaks to return
        min_distance: Minimum pixel distance between peaks
        threshold_rel: Minimum activation relative to max (0-1)

    Returns:
        List of (row, col, value, sign) tuples sorted by absolute value descending.
        sign is +1 for positive activations, -1 for negative.
    """
    from scipy import ndimage

    peaks = []

    # Find peaks in positive activations
    pos_cam = np.maximum(cam, 0)
    if pos_cam.max() > 0:
        pos_threshold = threshold_rel * pos_cam.max()
        pos_mask = pos_cam >= pos_threshold
        labeled, num_features = ndimage.label(pos_mask)
        for i in range(1, num_features + 1):
            region = (labeled == i)
            region_vals = pos_cam * region
            max_idx = np.unravel_index(region_vals.argmax(), region_vals.shape)
            peaks.append((max_idx[0], max_idx[1], float(pos_cam[max_idx]), +1))

    # Find peaks in negative activations (for signed cam)
    neg_cam = np.maximum(-cam, 0)
    if neg_cam.max() > 0:
        neg_threshold = threshold_rel * neg_cam.max()
        neg_mask = neg_cam >= neg_threshold
        labeled, num_features = ndimage.label(neg_mask)
        for i in range(1, num_features + 1):
            region = (labeled == i)
            region_vals = neg_cam * region
            max_idx = np.unravel_index(region_vals.argmax(), region_vals.shape)
            peaks.append((max_idx[0], max_idx[1], float(neg_cam[max_idx]), -1))

    # Sort by absolute value descending
    peaks.sort(key=lambda x: abs(x[2]), reverse=True)

    # Filter peaks that are too close together
    filtered = []
    for peak in peaks:
        too_close = False
        for existing in filtered:
            dist = np.sqrt((peak[0] - existing[0])**2 + (peak[1] - existing[1])**2)
            if dist < min_distance:
                too_close = True
                break
        if not too_close:
            filtered.append(peak)
        if len(filtered) >= num_peaks:
            break

    return filtered


def class_tint_overlay(
    base01: np.ndarray,
    cams: np.ndarray,
    colors,
    threshold: float = 0.30,
    alpha: float = 0.60,
) -> np.ndarray:
    """Tint each pixel by the class its per-class Grad-CAM pushes toward most.

    Args:
        base01: [H, W] grayscale image in [0, 1] (already display-mapped).
        cams: [H, W, K] per-class maps in [0, 1]
            (:func:`remo_id.interpret.gradcam.compute_class_gradcams`).
        colors: K RGB triples in [0, 1], one per class (class index order).
        threshold: only pushes above this are tinted.
        alpha: maximum tint opacity; opacity = alpha * w**1.5 with
            w = (push - threshold) / (1 - threshold).

    Returns:
        [H, W, 3] float RGB in [0, 1].
    """
    base = np.repeat(base01[..., None], 3, axis=2)
    tint = np.asarray(colors, dtype=float)[cams.argmax(-1)]
    w = np.clip((cams.max(-1) - threshold) / (1.0 - threshold), 0.0, 1.0)
    a = (alpha * w ** 1.5)[..., None]
    return np.clip((1 - a) * base + a * tint, 0, 1)
