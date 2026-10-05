"""Instance-level classification models."""

import os

from torchvision import models
import torch
import torch.nn as nn

from remo_id.models.base import BaseModel

# (torch.hub entry point, feat_dim)
_DINO_CONFIGS = {
    "dino-vits16": ("dinov3_vits16", 384),
    "dino-vitsplus": ("dinov3_vits16plus", 384),
    "dino-vitb": ("dinov3_vitb16", 768),
}


def load_dino_backbone(backbone: str, pretrained: bool = True, dino_weights: str | None = None):
    """Build a DINOv3 ViT via ``torch.hub``; returns ``(module, feat_dim)``.

    The model code comes from ``$REMO_ID_DINO_REPO`` (a local clone of
    https://github.com/facebookresearch/dinov3) or, when unset, is fetched from
    GitHub by ``torch.hub``. DINOv3 weights are distributed by Meta under the
    DINOv3 License and are not bundled: download the checkpoint matching
    ``backbone`` and pass it as ``dino_weights`` (``--dino_weights``) or set
    ``$REMO_ID_DINO_WEIGHTS``. Not needed with ``pretrained=False`` (e.g. when
    restoring a trained remo-id checkpoint).
    """
    hub_fn, feat_dim = _DINO_CONFIGS[backbone]
    repo = os.environ.get("REMO_ID_DINO_REPO") or "facebookresearch/dinov3"
    source = "local" if os.path.isdir(repo) else "github"
    weights = None
    if pretrained:
        weights = dino_weights or os.environ.get("REMO_ID_DINO_WEIGHTS")
        if not weights:
            raise ValueError(
                f"{backbone} needs pretrained DINOv3 weights, which are released by Meta "
                f"under the DINOv3 License and are not bundled with remo-id. Accept the "
                f"license at https://github.com/facebookresearch/dinov3, download the "
                f"'{hub_fn}' checkpoint, and pass --dino_weights PATH (or set "
                f"REMO_ID_DINO_WEIGHTS). Use --no-pretrained to train from scratch."
            )
    dino = torch.hub.load(repo, hub_fn, source=source, pretrained=weights is not None, weights=weights)
    return dino, feat_dim


# ----------------------------
# Base class for instance-level models
# ----------------------------
class BaseInstanceModel(BaseModel):
    """Thin base for instance-level models. Implements _forward_step()."""

    def _forward_step(self, batch):
        x, label, _exp_ids = batch
        logits = self(x)
        loss = self._compute_loss(logits, label)
        return loss, logits, label


# ----------------------------
# Helper: Backbone creation with configurable input channels
# ----------------------------
def _adapt_conv1_channels(conv, in_channels):
    """Replace a Conv2d's input channels, initializing from averaged pretrained weights."""
    new_conv = nn.Conv2d(
        in_channels, conv.out_channels,
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, bias=(conv.bias is not None)
    )
    with torch.no_grad():
        new_conv.weight[:] = conv.weight.mean(dim=1, keepdim=True)
        if conv.bias is not None:
            new_conv.bias[:] = conv.bias
    return new_conv


def _apply_finetune_layers(backbone: nn.Module, finetune_layers: str):
    """Apply fine-tuning configuration to a backbone.

    Args:
        backbone: The backbone module to configure
        finetune_layers: Comma-separated layer names to fine-tune,
            'all' for full fine-tuning, or 'none' to freeze everything.
    """
    ft = finetune_layers.strip().lower()
    if ft == "all":
        return  # all params trainable
    # Freeze everything first
    for param in backbone.parameters():
        param.requires_grad = False
    if ft == "none":
        return
    # Unfreeze specified layers
    for layer_name in finetune_layers.split(","):
        layer_name = layer_name.strip()
        if hasattr(backbone, layer_name):
            for param in getattr(backbone, layer_name).parameters():
                param.requires_grad = True


def _apply_finetune_layers_with_resolver(backbone: nn.Module, finetune_layers: str, layer_resolver):
    """Apply fine-tuning configuration using a custom layer resolver.

    Args:
        backbone: The backbone module to configure
        finetune_layers: Comma-separated layer specs, 'all', or 'none'
        layer_resolver: Callable(spec: str) -> Optional[nn.Module] that returns
            the sub-module for a given layer spec, or None if not applicable.
    """
    ft = finetune_layers.strip().lower()
    if ft == "all":
        return
    for param in backbone.parameters():
        param.requires_grad = False
    if ft == "none":
        return
    for layer_spec in finetune_layers.split(","):
        layer_spec = layer_spec.strip()
        layer = layer_resolver(layer_spec)
        if layer is not None:
            for param in layer.parameters():
                param.requires_grad = True


def create_single_channel_resnet(base: str = 'resnet18', in_channels: int = 1, pretrained: bool = True):
    """Create a ResNet backbone with configurable input channels."""
    weights = 'DEFAULT' if pretrained else None
    if base == 'resnet18':
        backbone = models.resnet18(weights=weights)
        feat_dim = 512
    elif base == 'resnet50':
        backbone = models.resnet50(weights=weights)
        feat_dim = 2048
    else:
        raise ValueError(f"Unknown ResNet base: {base}")

    if in_channels != 3:
        backbone.conv1 = _adapt_conv1_channels(backbone.conv1, in_channels)

    return backbone, feat_dim


# ----------------------------
# ResNet Model (resnet18 and resnet50)
# ----------------------------
class ResNetModel(BaseInstanceModel):
    """ResNet-based classifier for grayscale or RGB images.

    Supports resnet18 (default) and resnet50 via the backbone parameter.
    """

    def __init__(self, backbone="resnet18", task_type="binary", num_classes=2,
                 class_weights=None, lr=1e-4, lr_schedule="cosine", warmup_epochs=5, max_epochs=100,
                 in_channels=1, weight_decay=0.01, pretrained=True, finetune_layers="layer3,layer4"):
        super().__init__()
        self.in_channels = in_channels
        self.model, feat_dim = create_single_channel_resnet(backbone, in_channels=in_channels, pretrained=pretrained)

        _apply_finetune_layers(self.model, finetune_layers)

        out_features = num_classes if task_type == "multiclass" else 1
        self.model.fc = nn.Linear(feat_dim, out_features)

        self._setup_loss(task_type, num_classes, class_weights)
        self._setup_lr(lr, lr_schedule, warmup_epochs, max_epochs, weight_decay)
        backbone_label = backbone.replace("resnet", "ResNet")
        self.backbone_name = backbone_label
        self.name = backbone_label
        self.save_hyperparameters(logger=False)

    def forward(self, x):
        return self.model(x)


# ----------------------------
# ViT Models (vit-b-16 and vit-l-16)
# ----------------------------
_VIT_CONFIGS = {
    "vit-b-16": (models.vit_b_16, "DEFAULT", 768),
    "vit-l-16": (models.vit_l_16, "IMAGENET1K_SWAG_E2E_V1", 1024),
}


class ViTModel(BaseInstanceModel):
    """Vision Transformer based classifier for grayscale or RGB images.

    Supports vit-b-16 (default) and vit-l-16 via the backbone parameter.
    Uses 'layer-N' spec for finetune_layers (e.g. 'layer-1' = last encoder layer).
    """

    def __init__(self, backbone="vit-b-16", task_type="binary", num_classes=2,
                 class_weights=None, lr=1e-4, lr_schedule="cosine", warmup_epochs=5, max_epochs=100,
                 in_channels=1, weight_decay=0.01, finetune_layers="layer-1"):
        super().__init__()
        self.in_channels = in_channels

        factory, weights_name, feat_dim = _VIT_CONFIGS[backbone]
        self.model = factory(weights=weights_name)
        if in_channels != 3:
            self.model.conv_proj = _adapt_conv1_channels(self.model.conv_proj, in_channels)

        _apply_finetune_layers_with_resolver(
            self.model, finetune_layers,
            lambda spec: self.model.encoder.layers[int(spec.replace("layer", ""))] if spec.startswith("layer") else None,
        )

        out_features = num_classes if task_type == "multiclass" else 1
        # vit-b-16 uses model.heads directly (Linear), vit-l-16 uses model.heads.head
        if hasattr(self.model.heads, 'head'):
            in_features = self.model.heads.head.in_features
        else:
            in_features = feat_dim
        self.model.heads = nn.Linear(in_features, out_features)

        self._setup_loss(task_type, num_classes, class_weights)
        self._setup_lr(lr, lr_schedule, warmup_epochs, max_epochs, weight_decay)
        backbone_label = backbone.upper()
        self.backbone_name = backbone_label
        self.name = backbone_label
        self.save_hyperparameters(logger=False)

    def forward(self, x):
        return self.model(x)


# ----------------------------
# Patch-level gated attention pooling
# ----------------------------
class PatchGatedAttention(nn.Module):
    """Gated attention pooling over patch tokens within a single image.

    Replaces the CLS token with a learned weighted combination of patch features,
    letting the model focus on the most informative spatial regions.
    """

    def __init__(self, feat_dim: int, attn_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.attn_V = nn.Linear(feat_dim, attn_dim)
        self.attn_U = nn.Linear(feat_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patches: torch.Tensor):
        """
        Args:
            patches: [B, N_patches, feat_dim]
        Returns:
            pooled: [B, feat_dim]
            attn: [B, N_patches]
        """
        V = torch.tanh(self.attn_V(patches))
        U = torch.sigmoid(self.attn_U(patches))
        gated = self.dropout(V * U)
        A = self.attn_w(gated)
        attn = torch.softmax(A, dim=1)
        pooled = (attn * patches).sum(dim=1)
        return pooled, attn.squeeze(-1)


# ----------------------------
# DINOv3 Models (ViT-S/16, ViT-S/16+, ViT-B/16)
# ----------------------------
class DINOv3Model(BaseInstanceModel):
    """DINOv3 ViT-based classifier. Expects single-channel input, handles
    grayscale→RGB conversion and ImageNet normalization internally."""

    def __init__(self, backbone="dino-vits16", task_type="binary", num_classes=2, class_weights=None,
                 lr=1e-4, lr_schedule="cosine", warmup_epochs=5, max_epochs=100,
                 in_channels=1, weight_decay=0.01, pretrained=True, finetune_layers="none",
                 patch_pool=False, dino_weights=None):
        super().__init__()
        self.in_channels = in_channels
        self.patch_pool = patch_pool

        self.dino, feat_dim = load_dino_backbone(backbone, pretrained, dino_weights)

        _apply_finetune_layers(self.dino, finetune_layers)

        # ImageNet normalization buffers
        self.register_buffer('norm_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('norm_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if patch_pool:
            self.patch_attention = PatchGatedAttention(feat_dim)

        out_features = num_classes if task_type == "multiclass" else 1
        self.classifier = nn.Linear(feat_dim, out_features)

        self._setup_loss(task_type, num_classes, class_weights)
        self._setup_lr(lr, lr_schedule, warmup_epochs, max_epochs, weight_decay)
        label = backbone.upper().replace("DINO-", "DINOv3-")
        self.backbone_name = label
        self.name = label
        self.save_hyperparameters(logger=False)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.norm_mean) / self.norm_std
        out = self.dino.forward_features(x)
        if self.patch_pool:
            features, _ = self.patch_attention(out["x_norm_patchtokens"])
        else:
            features = out["x_norm_clstoken"]
        return self.classifier(features)


# ----------------------------
# Custom Microscopy Encoder
# ----------------------------
class MicroscopyEncoder(nn.Module):
    """Custom encoder optimized for microscopy images (~200K params)."""

    def __init__(self, feat_dim=256, in_channels=1):
        super().__init__()
        self.feat_dim = feat_dim
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MicroscopyCNNModel(BaseInstanceModel):
    """Instance-level classification with custom microscopy encoder."""

    def __init__(self, task_type="binary", num_classes=2,
                 class_weights=None, lr=1e-4, lr_schedule="cosine", warmup_epochs=5,
                 max_epochs=100, feat_dim=256, in_channels=1, weight_decay=0.01,
                 pretrained=False):
        super().__init__()
        self.in_channels = in_channels
        self.backbone = MicroscopyEncoder(feat_dim=feat_dim, in_channels=in_channels)

        out_features = num_classes if task_type == "multiclass" else 1
        self.classifier = nn.Linear(feat_dim, out_features)

        self._setup_loss(task_type, num_classes, class_weights)
        self._setup_lr(lr, lr_schedule, warmup_epochs, max_epochs, weight_decay)
        self.backbone_name = 'MicroscopyCNN'
        self.name = 'MicroscopyCNN'
        self.save_hyperparameters(logger=False)

    def forward(self, x):
        return self.classifier(self.backbone(x))
