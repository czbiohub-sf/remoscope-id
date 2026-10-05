"""GPU-dependent attention visualization (ViT rollout, DINO attention)."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from remo_id.interpret.core import (
    is_vit_like,
    find_backbone,
    _find_dino_backbone,
    _get_gradcam_model_and_layer,
)
from remo_id.interpret.overlays import (
    base_rgb_from_tensor,
    resize_mask_to_image,
    fuse_heads,
)


@torch.no_grad()
def vit_attention_rollout(
    model: nn.Module,
    x: torch.Tensor,
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
) -> np.ndarray:
    """Compute ViT attention rollout (CLS->patches).

    Args:
        model: ViT model (torchvision style)
        x: [B,C,H,W] input tensor
        head_fusion: How to fuse attention heads
        discard_ratio: Fraction of lowest attention to zero out

    Returns:
        [h,w] mask in [0,1] (token grid size)
    """
    backbone = find_backbone(model)
    vit = getattr(backbone, "model", backbone)

    if not (is_vit_like(vit) and hasattr(vit, "_process_input")):
        raise RuntimeError("Model is not a torchvision ViT-like module with _process_input().")

    encoder = vit.encoder
    layers = encoder.layers

    # patch embeddings: [B, N, D]
    tokens = vit._process_input(x)

    # CLS token
    cls_tok = getattr(vit, "class_token", None)
    if cls_tok is None:
        cls_tok = getattr(vit, "cls_token")
    cls_tok = cls_tok.expand(tokens.shape[0], -1, -1)  # [B,1,D]
    tokens = torch.cat([cls_tok, tokens], dim=1)       # [B, 1+N, D]

    # pos embedding + dropout
    if hasattr(encoder, "pos_embedding"):
        tokens = tokens + encoder.pos_embedding
    if hasattr(encoder, "dropout"):
        tokens = encoder.dropout(tokens)

    B, S, _ = tokens.shape
    device = x.device
    joint = torch.eye(S, device=device).unsqueeze(0).repeat(B, 1, 1)

    for blk in layers:
        x_ln = blk.ln_1(tokens)

        # per-head attention weights if available
        try:
            attn_out, attn_w = blk.self_attention(
                x_ln, x_ln, x_ln,
                need_weights=True,
                average_attn_weights=False
            )  # [B, heads, S, S]
        except TypeError:
            attn_out, attn_w = blk.self_attention(x_ln, x_ln, x_ln, need_weights=True)
            if attn_w.dim() == 3:
                attn_w = attn_w.unsqueeze(1)  # [B,1,S,S]

        # advance tokens (matches torchvision)
        tokens = tokens + blk.dropout(attn_out)
        tokens = tokens + blk.mlp(blk.ln_2(tokens))

        attn = fuse_heads(attn_w, head_fusion=head_fusion)  # [B,S,S]

        if discard_ratio > 0.0:
            flat = attn.view(B, -1)
            k = int(flat.shape[1] * discard_ratio)
            if k > 0:
                thresh, _ = torch.kthvalue(flat, k=k, dim=1)
                attn = torch.where(attn <= thresh.view(B, 1, 1),
                                   torch.zeros_like(attn),
                                   attn)

        # residual + row normalize
        attn = attn + torch.eye(S, device=device).unsqueeze(0)
        attn = attn / attn.sum(dim=-1, keepdim=True)

        joint = attn @ joint

    # CLS -> patches
    cls_to_patches = joint[:, 0, 1:]  # [B, N]
    N = cls_to_patches.shape[-1]
    h = w = int(math.sqrt(N))
    assert h * w == N, f"Non-square token map: N={N}"

    mask = cls_to_patches[0].reshape(h, w).detach().cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


@torch.no_grad()
def dino_cls_attention(
    model: nn.Module,
    x: torch.Tensor,
    layer_idx: int = -1,
) -> np.ndarray:
    """Compute CLS token attention map from a DINOv3 ViT model.

    Args:
        model: Model with a DINOv3 backbone (self.dino with .blocks)
        x: [B, C, H, W] input tensor (already preprocessed)
        layer_idx: Which transformer block to extract attention from (-1 = last)

    Returns:
        [h, w] attention mask in [0, 1] (patch grid size)
    """
    dino = _find_dino_backbone(model)
    device = x.device

    if hasattr(model, 'norm_mean'):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - model.norm_mean) / model.norm_std

    has_rope = hasattr(dino, "rope_embed") and dino.rope_embed is not None

    if has_rope and hasattr(dino, "prepare_tokens_with_masks"):
        tokens, (H, W) = dino.prepare_tokens_with_masks(x, masks=None)
        rope = dino.rope_embed(H=H, W=W)
    elif hasattr(dino, "prepare_tokens"):
        tokens = dino.prepare_tokens(x)
        rope = None
    else:
        raise RuntimeError("Cannot find token preparation method on DINOv3 backbone")

    blocks = list(dino.blocks)
    target_idx = layer_idx if layer_idx >= 0 else len(blocks) + layer_idx

    for i, block in enumerate(blocks[:target_idx]):
        if has_rope:
            tokens = block(tokens, rope)
        else:
            tokens = block(tokens)

    target_block = blocks[target_idx]
    attn_module = target_block.attn
    B, N, C = tokens.shape

    norm_x = target_block.norm1(tokens) if hasattr(target_block, "norm1") else tokens

    qkv = attn_module.qkv(norm_x)
    qkv = qkv.reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
    q, k, _v = torch.unbind(qkv, dim=2)

    q = q.transpose(1, 2)  # (B, num_heads, N, head_dim)
    k = k.transpose(1, 2)

    # Apply RoPE if present
    if has_rope and rope is not None:
        sin, cos = rope
        rope_dtype = sin.dtype
        q_rot, k_rot = q.to(rope_dtype), k.to(rope_dtype)

        prefix = N - sin.shape[-2]
        q_prefix, k_prefix = q_rot[:, :, :prefix, :], k_rot[:, :, :prefix, :]
        q_patches, k_patches = q_rot[:, :, prefix:, :], k_rot[:, :, prefix:, :]

        def rotate_half(t):
            t1, t2 = t.chunk(2, dim=-1)
            return torch.cat([-t2, t1], dim=-1)

        q_rot = torch.cat([q_prefix, (q_patches * cos) + (rotate_half(q_patches) * sin)], dim=-2)
        k_rot = torch.cat([k_prefix, (k_patches * cos) + (rotate_half(k_patches) * sin)], dim=-2)
        q, k = q_rot.to(q.dtype), k_rot.to(k.dtype)

    scale = (C // attn_module.num_heads) ** -0.5
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)

    n_storage = getattr(dino, "n_storage_tokens", 0)
    prefix_tokens = 1 + n_storage
    cls_attn = attn_weights[:, :, 0, prefix_tokens:].mean(dim=1)  # (B, num_patches)

    num_patches = cls_attn.shape[-1]
    _, _, img_H, img_W = x.shape
    patch_size = getattr(dino, "patch_size", 16)
    h_patches = img_H // patch_size
    w_patches = img_W // patch_size

    if h_patches * w_patches != num_patches:
        h_patches = w_patches = int(math.sqrt(num_patches))

    mask = cls_attn[0].reshape(h_patches, w_patches).cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


@torch.no_grad()
def make_instance_overlay(
    model: nn.Module,
    x1: torch.Tensor,
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
    image_weight: float = 0.85,
) -> np.ndarray:
    """Create a unified overlay visualization for any backbone type.

    Automatically detects the model architecture and applies
    the appropriate visualization technique:
    - ViT models: attention rollout
    - ResNet models: Grad-CAM on layer4[-1]
    - MIL models: Grad-CAM on backbone only
    - Fallback: raw image

    Args:
        model: The neural network model (can be MIL or instance-level)
        x1: Input tensor of shape [1, C, H, W]
        head_fusion: How to fuse attention heads for ViT
        discard_ratio: Fraction of lowest attention to discard for ViT
        image_weight: Weight for blending original image (0-1)

    Returns:
        RGB uint8 array of shape [H, W, 3] with overlay visualization
    """
    device = next(model.parameters()).device
    x1 = x1.to(device)

    base_rgb01 = base_rgb_from_tensor(x1)
    H, W = base_rgb01.shape[:2]

    backbone = find_backbone(model)
    vit = getattr(backbone, "model", backbone)

    # Try ViT attention rollout first
    if is_vit_like(vit) and hasattr(vit, "_process_input"):
        try:
            mask_hw = vit_attention_rollout(backbone, x1, head_fusion=head_fusion, discard_ratio=discard_ratio)
            mask = resize_mask_to_image(mask_hw, H, W)
            overlay = show_cam_on_image(base_rgb01, mask, use_rgb=True, image_weight=image_weight)
            return overlay
        except Exception:
            pass

    # Try Grad-CAM for ResNet-like models (handles both MIL and instance models)
    gradcam_model, target_layer = _get_gradcam_model_and_layer(model)
    if target_layer is not None:
        try:
            # Need to enable gradients for Grad-CAM
            x_grad = x1.clone().requires_grad_(True)
            targets = [ClassifierOutputTarget(0)]

            with torch.enable_grad():
                with GradCAM(model=gradcam_model, target_layers=[target_layer]) as cam:
                    grayscale_cam = cam(input_tensor=x_grad, targets=targets)[0]

            overlay = show_cam_on_image(base_rgb01, grayscale_cam, use_rgb=True, image_weight=image_weight)
            return overlay
        except Exception:
            pass

    # Fallback: return raw image
    return (base_rgb01 * 255).astype(np.uint8)
