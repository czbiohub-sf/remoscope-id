"""GPU-dependent Grad-CAM computation."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from remo_id.interpret.core import (
    get_target_layer_resnet,
)
from remo_id.interpret.overlays import (
    base_rgb_from_tensor,
    make_signed_overlay,
)


def compute_signed_gradcam(
    model: nn.Module,
    x1: torch.Tensor,
    target_layer: Optional[nn.Module] = None,
    normalize: bool = True,
    target_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute signed Grad-CAM that preserves gradient direction.

    Unlike standard Grad-CAM which uses ReLU to keep only positive gradients,
    signed Grad-CAM preserves the sign to indicate whether features push
    toward (positive) or away from (negative) the target class prediction.

    Args:
        model: The neural network model
        x1: Input tensor of shape [1, C, H, W]
        target_layer: Target layer for Grad-CAM (auto-detected if None)
        normalize: If True, normalize to [-1, 1] range.
        target_class: Class index to compute gradients for (default: 0 for
            binary, required for multiclass).

    Returns:
        Tuple of (signed_cam, base_rgb01):
            - signed_cam: [H, W] array in [-1, 1] range (if normalize=True) or raw values
            - base_rgb01: [H, W, 3] array in [0, 1] range
    """
    device = next(model.parameters()).device
    x1 = x1.to(device)

    base_rgb01 = base_rgb_from_tensor(x1)
    H, W = base_rgb01.shape[:2]

    if target_layer is None:
        target_layer = get_target_layer_resnet(model)

    if target_layer is None:
        return np.zeros((H, W), dtype=np.float32), base_rgb01

    # Hook to capture activations and gradients
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        x_grad = x1.clone().requires_grad_(True)

        with torch.enable_grad():
            # Forward pass
            output = model(x_grad)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # Get the target class logit
            tc = target_class or 0
            if output.dim() == 0:
                target = output
            elif output.dim() == 1:
                target = output[tc] if output.shape[0] > 1 else output[0]
            else:
                target = output[0, tc]

            # Backward pass
            model.zero_grad()
            target.backward()

        if not activations or not gradients:
            return np.zeros((H, W), dtype=np.float32), base_rgb01

        # Get activations and gradients
        acts = activations[0]  # [1, C, h, w]
        grads = gradients[0]   # [1, C, h, w]

        # Weight activations by gradients (without ReLU - this preserves sign)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        weighted = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = weighted[0, 0].detach().cpu().numpy()  # [h, w]

        # Resize to image dimensions
        cam_tensor = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)
        cam_resized = F.interpolate(cam_tensor, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam_resized[0, 0].numpy()

        # Optionally normalize to [-1, 1] range while preserving sign
        if normalize:
            max_abs = max(abs(cam.max()), abs(cam.min()), 1e-8)
            cam = cam / max_abs

        return cam, base_rgb01

    finally:
        forward_handle.remove()
        backward_handle.remove()


def gradcam_overlay(
    model: nn.Module,
    x1: torch.Tensor,
    target_layer: Optional[nn.Module] = None,
    image_weight: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Grad-CAM overlay for ResNet-like models.

    Args:
        model: The neural network model
        x1: Input tensor of shape [1, C, H, W]
        target_layer: Target layer for Grad-CAM (auto-detected if None)
        image_weight: Weight for blending original image

    Returns:
        Tuple of (base_rgb, overlay) as uint8 arrays
    """
    device = next(model.parameters()).device
    x1 = x1.to(device)

    base_rgb01 = base_rgb_from_tensor(x1)
    H, W = base_rgb01.shape[:2]

    if target_layer is None:
        target_layer = get_target_layer_resnet(model)

    if target_layer is None:
        base_uint8 = (base_rgb01 * 255).astype(np.uint8)
        return base_uint8, base_uint8

    x_grad = x1.clone().requires_grad_(True)
    targets = [ClassifierOutputTarget(0)]

    with torch.enable_grad():
        with GradCAM(model=model, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=x_grad, targets=targets)[0]

    overlay = show_cam_on_image(base_rgb01, grayscale_cam, use_rgb=True, image_weight=image_weight)
    return (base_rgb01 * 255).astype(np.uint8), overlay


def make_instance_overlay_signed(
    model: nn.Module,
    x1: torch.Tensor,
    image_weight: float = 0.6,
    colormap: str = "RdBu_r",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create signed Grad-CAM overlay for an image.

    Args:
        model: The neural network model
        x1: Input tensor of shape [1, C, H, W]
        image_weight: Weight for blending (0 = all heatmap, 1 = all image)
        colormap: Matplotlib colormap name

    Returns:
        Tuple of (base_rgb, signed_overlay, signed_cam):
            - base_rgb: uint8 array [H, W, 3]
            - signed_overlay: uint8 array [H, W, 3] with diverging colormap
            - signed_cam: float array [H, W] in [-1, 1] range
    """
    signed_cam, base_rgb01 = compute_signed_gradcam(model, x1)
    base_rgb = (base_rgb01 * 255).astype(np.uint8)
    signed_overlay = make_signed_overlay(base_rgb01, signed_cam, image_weight, colormap)
    return base_rgb, signed_overlay, signed_cam


def compute_class_gradcams(
    model: nn.Module,
    x1: torch.Tensor,
    target_layer: Optional[nn.Module] = None,
    classes=None,
) -> np.ndarray:
    """Per-class signed Grad-CAM of one image, positive part only.

    For every class ``c`` in ``classes`` (default: every output), the signed Grad-CAM
    of logit ``c`` normalised to [-1, 1] (:func:`compute_signed_gradcam`) and clipped
    at 0, so each map shows where the image pushes the model toward that class.

    Returns:
        float32 array [H, W, len(classes)] in [0, 1]. Feed it to
        :func:`remo_id.interpret.overlays.class_tint_overlay` for a per-class tint.
    """
    if classes is None:
        with torch.no_grad():
            out = model(x1.to(next(model.parameters()).device))
        out = out[0] if isinstance(out, (tuple, list)) else out
        classes = range(max(1, out.shape[-1]) if out.dim() > 0 else 1)
    cams = []
    for tc in classes:
        sc, _ = compute_signed_gradcam(model, x1, target_layer=target_layer, normalize=True,
                                       target_class=tc)
        cams.append(np.clip(sc, 0.0, 1.0))
    return np.stack(cams, -1).astype(np.float32)


def frame_class_gradcams(checkpoint: str, experiment_path: str, image_id: int,
                         device: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Per-class Grad-CAM (:func:`compute_class_gradcams`) of one full frame.

    Loads the checkpoint (model settings from its metadata), reads frame ``image_id`` of
    the experiment's image stack and preprocesses it exactly as at evaluation.

    Returns:
        (cams [H, W, K] float32 in [0, 1], raw frame [H, W] uint8)
    """
    from remo_id.data.zarr_io import get_image_from_zarr
    from remo_id.inference.core import (extract_background_value, extract_checkpoint_metadata,
                                        get_augmentations, load_checkpoint, preprocess)
    from remo_id.interpret.core import _get_gradcam_model_and_layer
    meta = extract_checkpoint_metadata(checkpoint)
    model, device, in_ch = load_checkpoint(checkpoint, meta["model_name"], meta["task_type"],
                                           meta["num_classes"], n_experiments=1, device=device)
    gcm, layer = _get_gradcam_model_and_layer(model)
    img = np.array(get_image_from_zarr(experiment_path, img_ids=[int(image_id)]))
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[:, :, 0]
    x = preprocess(img, augments=get_augmentations(model.backbone_name), target_channels=in_ch,
                   background_value=extract_background_value(checkpoint)).to(device)
    classes = range(meta["num_classes"]) if meta["task_type"] == "multiclass" else [0]
    return compute_class_gradcams(gcm, x, target_layer=layer, classes=classes), np.asarray(img)
