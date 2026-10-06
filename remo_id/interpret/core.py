"""CPU-only model introspection utilities."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def is_vit_like(model: nn.Module) -> bool:
    """Check if a model is ViT-like (has encoder and class token).

    Also checks the backbone attribute for MIL models.
    """
    # First try the model directly
    m = getattr(model, "model", model)
    if hasattr(m, "encoder") and (hasattr(m, "class_token") or hasattr(m, "cls_token")):
        return True

    # For MIL models, check the backbone
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        b = getattr(backbone, "model", backbone)
        if hasattr(b, "encoder") and (hasattr(b, "class_token") or hasattr(b, "cls_token")):
            return True

    return False


def find_backbone(module: nn.Module) -> nn.Module:
    """Find the actual vision backbone inside a model wrapper."""
    for name in ["backbone", "dino", "vit", "encoder", "feature_extractor", "model"]:
        if hasattr(module, name):
            cand = getattr(module, name)
            if isinstance(cand, nn.Module):
                return cand
    return module  # fallback


def is_dino_like(model: nn.Module) -> bool:
    """Check if a model uses a DINOv3 backbone (has blocks + cls_token)."""
    for attr in ("dino", "backbone", "model"):
        b = getattr(model, attr, None)
        if b is not None and hasattr(b, "blocks") and hasattr(b, "cls_token"):
            return True
    return False


def _find_dino_backbone(model: nn.Module) -> nn.Module:
    """Find the DINOv3 backbone inside a model."""
    for attr in ("dino", "backbone", "model"):
        b = getattr(model, attr, None)
        if b is not None and hasattr(b, "blocks") and hasattr(b, "cls_token"):
            return b
    raise RuntimeError("No DINOv3 backbone found")


def get_target_layer_resnet(model: nn.Module, which: str = "layer4") -> Optional[nn.Module]:
    """Get the target layer for Grad-CAM on ResNet models.

    `which` selects the block: layer4 (7x7 @224, coarsest/default), layer3
    (14x14), layer2 (28x28) for progressively sharper (less blurry) CAMs.
    """
    backbone = find_backbone(model)
    b = getattr(backbone, "model", backbone)
    layer = getattr(b, which, None)
    if layer is None:
        layer = getattr(b, "layer4", None)
    if layer is None:
        return None
    try:
        return layer[-1]
    except Exception:
        return layer


class _MILOutputWrapper(nn.Module):
    """Wrapper to ensure MIL model output has [B, C] shape for GradCAM.

    GradCAM expects output with at least 2 dimensions [batch, classes].
    MIL models output a scalar (0-dim) for binary tasks after squeeze.
    This wrapper reshapes the output so GradCAM can index it properly.

    The full MIL model (backbone + attention + classifier) runs normally,
    so gradients flow through the entire pipeline and Grad-CAM reflects
    the actual classification decision, not just backbone feature sums.
    """
    def __init__(self, mil_model: nn.Module):
        super().__init__()
        self.mil_model = mil_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mil_model(x)
        if out.dim() == 0:
            out = out.unsqueeze(0).unsqueeze(0)
        elif out.dim() == 1:
            out = out.unsqueeze(-1)
        return out


def _get_gradcam_model_and_layer(model: nn.Module) -> Tuple[nn.Module, Optional[nn.Module]]:
    """Get the appropriate model and target layer for Grad-CAM.

    For MIL models, we wrap the full model (backbone + attention + classifier)
    so gradients flow through the real classification path. Requires
    n_experiments=1 (standard for inference).
    For instance models, we use the full model directly.
    """
    backbone = find_backbone(model)

    # Check if this is a MIL model
    from remo_id.models.mil import BaseMILModel
    is_mil = isinstance(model, BaseMILModel)

    if is_mil:
        resnet = getattr(backbone, "model", backbone)
        if hasattr(resnet, "layer4"):
            wrapped = _MILOutputWrapper(model)
            return wrapped, resnet.layer4[-1]

    # For non-MIL models, use the full model
    target_layer = get_target_layer_resnet(model)
    return model, target_layer
