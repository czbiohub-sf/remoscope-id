"""Interpret module: visualization and model introspection for remo-id.

Submodules imported on demand: ``gallery`` (Grad-CAM crop galleries, ``remo-id gallery``),
``occlusion`` (cell-removal fill model) and ``cell_occlusion`` (cell occlusion screen,
``remo-id analysis occlusion``).
"""

from remo_id.interpret.core import (
    is_vit_like,
    is_dino_like,
    find_backbone,
    get_target_layer_resnet,
    _get_gradcam_model_and_layer,
    _MILOutputWrapper,
)

from remo_id.interpret.overlays import (
    base_rgb_from_tensor,
    tensor_to_rgb_numpy,
    resize_mask_to_image,
    fuse_heads,
    make_signed_overlay,
    normalize_cams_globally,
    find_activation_peaks,
    class_tint_overlay,
)

from remo_id.interpret.gradcam import (
    compute_signed_gradcam,
    gradcam_overlay,
    make_instance_overlay_signed,
    compute_class_gradcams,
    frame_class_gradcams,
)

from remo_id.interpret.attention import (
    vit_attention_rollout,
    dino_cls_attention,
    make_instance_overlay,
)

__all__ = [
    # Core
    "is_vit_like", "is_dino_like", "find_backbone",
    "get_target_layer_resnet", "_MILOutputWrapper",
    # Overlays
    "base_rgb_from_tensor", "tensor_to_rgb_numpy", "resize_mask_to_image",
    "fuse_heads", "make_signed_overlay", "normalize_cams_globally", "find_activation_peaks",
    "class_tint_overlay",
    # Grad-CAM
    "compute_signed_gradcam", "gradcam_overlay", "make_instance_overlay_signed",
    "compute_class_gradcams", "frame_class_gradcams",
    # Attention
    "vit_attention_rollout", "dino_cls_attention", "make_instance_overlay",
]
