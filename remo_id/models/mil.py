"""Multiple Instance Learning (MIL) models."""

from torchvision import models
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from remo_id.models.base import BaseModel
from remo_id.models.instance import (
    create_single_channel_resnet,
    _adapt_conv1_channels,
    _apply_finetune_layers,
    _apply_finetune_layers_with_resolver,
    MicroscopyEncoder,
    PatchGatedAttention,
    load_dino_backbone,
)


def _build_projection(feat_dim, projection_dim, dropout):
    """Build optional projection layer to reduce feature dimensionality."""
    if projection_dim is not None and projection_dim > 0:
        return nn.Sequential(
            nn.Linear(feat_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ), projection_dim
    return nn.Identity(), feat_dim


def _make_classifier(feat_dim, dropout, classifier_hidden=None, num_classes=2, task_type="binary"):
    """Build classifier head (nn.Module only, no loss).

    Args:
        classifier_hidden: Controls hidden layer size.
            None = default (feat_dim // 2),
            0 = linear only (no hidden layer),
            N > 0 = custom hidden dim.
    """
    out_features = num_classes if task_type == "multiclass" else 1

    if classifier_hidden == 0:
        # Linear only - no hidden layer
        return nn.Linear(feat_dim, out_features)
    elif classifier_hidden is not None and classifier_hidden > 0:
        # Custom hidden dim
        return nn.Sequential(
            nn.Linear(feat_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, out_features),
        )
    else:
        # Default: feat_dim // 2
        hidden_dim = feat_dim // 2
        return nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
        )


class MultiHeadGatedAttention(nn.Module):
    """Multi-head gated attention for MIL pooling.

    When num_heads=1, equivalent to single-head gated attention.
    """

    def __init__(self, feat_dim: int, attn_dim: int, num_heads: int = 1, dropout: float = 0.3):
        super().__init__()
        self.num_heads = num_heads
        self.attn_V = nn.ModuleList([nn.Linear(feat_dim, attn_dim) for _ in range(num_heads)])
        self.attn_U = nn.ModuleList([nn.Linear(feat_dim, attn_dim) for _ in range(num_heads)])
        self.attn_w = nn.ModuleList([nn.Linear(attn_dim, 1, bias=False) for _ in range(num_heads)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats: torch.Tensor):
        """
        Args:
            feats: [num_bags, bag_size, feat_dim]
        Returns:
            bag_repr: [num_bags, feat_dim * num_heads]  (heads concatenated)
            attn: [num_bags, bag_size]  (mean attention map across heads, for viz/topk)
        """
        bag_reprs, attns = [], []
        for h in range(self.num_heads):
            V_h = torch.tanh(self.attn_V[h](feats))
            U_h = torch.sigmoid(self.attn_U[h](feats))
            gated = self.dropout(V_h * U_h)
            A_h = self.attn_w[h](gated)
            attn_h = torch.softmax(A_h, dim=1)
            bag_reprs.append((attn_h * feats).sum(dim=1))
            attns.append(attn_h.squeeze(-1))

        # Concatenate per-head bag vectors so each head can specialize on a
        # different subpopulation and the classifier sees all of them. (num_heads=1
        # reduces to the original single feat_dim vector — backward compatible.)
        bag_repr = torch.cat(bag_reprs, dim=-1)
        attn = torch.stack(attns, dim=0).mean(dim=0)
        return bag_repr, attn


# ----------------------------
# Base class for all MIL models
# ----------------------------
class BaseMILModel(BaseModel):
    """Base class for MIL models.

    Subclasses must implement:
      - _encode_instances(x) -> features
      - _aggregate(feats) -> (bag_logits, attn_or_none)

    The base class handles forward(), forward_with_attention(), training/validation
    steps, and optimizer configuration (via BaseModel).
    """

    def _setup_mil(self, task_type, num_classes, n_experiments, lr, lr_schedule,
                   warmup_epochs, max_epochs, weight_decay, encode_chunk_size=None,
                   bag_label_mode="hard", attn_entropy_weight=0.0):
        # task_type and num_classes are needed by BaseModel._forward_step/_log_validation_metrics
        self.task_type = task_type
        self.num_classes = num_classes
        self.n_experiments = n_experiments
        self.encode_chunk_size = encode_chunk_size
        self.bag_label_mode = bag_label_mode
        self.attn_entropy_weight = attn_entropy_weight
        # Note: self.loss is already set by _init_attention_head via _setup_loss,
        # which correctly handles class_weights for both binary and multiclass.
        self._setup_lr(lr, lr_schedule, warmup_epochs, max_epochs, weight_decay)

    def _encode_instances(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _aggregate(self, feats: torch.Tensor):
        """Aggregate per-instance features into bag-level logits.

        Args:
            feats: [num_bags, bag_size, D]
        Returns:
            bag_logits: [num_bags] for binary, [num_bags, C] for multiclass
            attn: [num_bags, bag_size] or None (for non-attention models)
        """
        raise NotImplementedError

    def _encode_chunk(self, x_chunk: torch.Tensor) -> torch.Tensor:
        """Encode a chunk of images and apply projection."""
        feats = self._encode_instances(x_chunk)
        if hasattr(self, 'projection'):
            feats = self.projection(feats)
        return feats

    def _encode_and_reshape(self, x: torch.Tensor, n_experiments: int = None):
        """Encode instances, apply projection, and reshape into bags.

        When encode_chunk_size is set, the backbone processes images in chunks
        with gradient checkpointing: intermediate activations are discarded
        during the forward pass and recomputed during backward, trading ~30%
        more compute for dramatically less GPU memory.
        """
        n_exp = n_experiments or self.n_experiments
        B, C, H, W = x.shape
        assert B % n_exp == 0, \
            f"Batch size {B} not divisible by n_experiments={n_exp}"
        bag_size = B // n_exp
        x_flat = x.view(-1, C, H, W)

        chunk_size = self.encode_chunk_size
        if chunk_size and chunk_size < B:
            chunks = []
            for i in range(0, B, chunk_size):
                chunk = x_flat[i:i + chunk_size]
                if self.training:
                    chunk_feats = grad_checkpoint(
                        self._encode_chunk, chunk, use_reentrant=False,
                    )
                else:
                    chunk_feats = self._encode_chunk(chunk)
                chunks.append(chunk_feats)
            feats_flat = torch.cat(chunks, dim=0)
        else:
            feats_flat = self._encode_instances(x_flat)
            if hasattr(self, 'projection'):
                feats_flat = self.projection(feats_flat)

        D = feats_flat.shape[-1]
        feats = feats_flat.view(n_exp, bag_size, D)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._encode_and_reshape(x)
        bag_logits, _ = self._aggregate(feats)
        return bag_logits

    def forward_with_attention(self, x: torch.Tensor, n_experiments: int = None):
        """Returns (bag_logits, attn_weights, features).

        For models without attention, returns uniform weights.
        """
        feats = self._encode_and_reshape(x, n_experiments=n_experiments)
        bag_logits, attn = self._aggregate(feats)
        if attn is None:
            n_exp = feats.shape[0]
            bag_size = feats.shape[1]
            attn = torch.ones(n_exp, bag_size, device=x.device) / bag_size
        return bag_logits, attn, feats

    def _forward_step(self, batch):
        x, labels, _exp_ids = batch
        B = x.shape[0]
        # The sampler lays the batch out as a whole number of equal-size bags. It builds
        # self.n_experiments bags (train AND val — the val sampler pads to n_experiments by
        # repeating), so when B divides evenly use self.n_experiments. The fallback (infer
        # from unique exp_ids) only triggers for image-val batches whose size isn't a
        # multiple of n_experiments.
        if B % self.n_experiments == 0:
            n_exp = self.n_experiments
        else:
            n_exp = len(set(_exp_ids.tolist())) if hasattr(_exp_ids, "tolist") else self.n_experiments
            if n_exp == 0 or B % n_exp != 0:
                n_exp = 1
        bag_size = B // n_exp

        labels = labels.view(n_exp, bag_size)

        bag_label_mode = getattr(self, "bag_label_mode", "hard")
        if bag_label_mode == "soft" and self.task_type == "multiclass":
            # Collate kept raw per-image labels; compute bag proportion vector.
            bag_labels = torch.zeros(
                n_exp, self.num_classes, device=labels.device
            )
            for c in range(self.num_classes):
                bag_labels[:, c] = (labels == c).float().mean(dim=1)
        else:
            # Hard mode: collate already replicated the bag-level label across the
            # bag. Applies to single-class bags (majority_min_pct=1.0) AND mixed
            # bags (idea 6) — Counter.most_common collapses to majority before here.
            assert (labels.float().std(dim=1) == 0).all(), \
                "Post-collate per-instance labels must be uniform in hard mode"
            bag_labels = labels[:, 0]

        # Encode + aggregate directly so we can capture attention for regularization.
        feats = self._encode_and_reshape(x, n_experiments=n_exp)
        bag_logits, attn = self._aggregate(feats)

        loss = self._compute_loss(bag_logits, bag_labels)

        # Attention-entropy regularizer: encourages attention to spread across
        # instances instead of collapsing onto confidently-predicted majority cells.
        # Only active when soft labels are used and pooling produces attention weights.
        attn_entropy_weight = getattr(self, "attn_entropy_weight", 0.0)
        if (bag_label_mode == "soft"
                and attn_entropy_weight > 0
                and attn is not None
                and getattr(self, "pooling", None) in ("attention", "topk")):
            attn_entropy = -(attn * (attn + 1e-8).log()).sum(-1).mean()
            loss = loss - attn_entropy_weight * attn_entropy

        return loss, bag_logits, bag_labels


class TransformerAggregator(nn.Module):
    """Self-attention MIL pooling: cells attend to EACH OTHER via a small
    transformer encoder, then a CLS token reads out the bag embedding.

    Unlike gated attention (which scores each cell independently and averages),
    here a cell's representation depends on the whole bag, so it can encode
    relational / population features (e.g. relative abnormality, subpopulation
    ratios). d_model = attn_dim; nhead = num_heads (falls back to 1 if it doesn't
    divide). Uses PyTorch SDPA (FlashAttention) so N up to a few thousand fits.
    """

    def __init__(self, feat_dim, dim=128, num_heads=4, num_layers=2, dropout=0.3):
        super().__init__()
        nhead = num_heads if (dim % max(num_heads, 1) == 0 and num_heads >= 1) else 1
        self.proj = nn.Linear(feat_dim, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim

    def forward(self, feats):  # feats: [num_bags, bag_size, feat_dim]
        x = self.proj(feats)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)          # [B, N+1, dim], CLS prepended
        x = self.encoder(x)
        cls_out = self.norm(x[:, 0])            # [B, dim] bag embedding
        # proxy per-cell attention (alignment of each cell token with CLS) for viz/topk
        attn = torch.softmax((x[:, 1:] @ cls_out.unsqueeze(-1)).squeeze(-1), dim=1)
        return cls_out, attn


# ----------------------------
# Attention-based MIL models
# ----------------------------
class _AttentionMILMixin:
    """Mixin for MIL models that use gated attention pooling (or mean/max pooling)."""

    def _init_attention_head(self, feat_dim, attn_dim, num_heads, dropout,
                             task_type, num_classes, class_weights,
                             pooling="attention", classifier_hidden=None, top_k=None,
                             transformer_layers=2):
        self.pooling = pooling
        self.top_k = top_k
        if pooling in ("attention", "topk", "additive"):
            self.attention = MultiHeadGatedAttention(feat_dim, attn_dim, num_heads, dropout)
        elif pooling == "transformer":
            self.aggregator = TransformerAggregator(feat_dim, dim=attn_dim,
                                                    num_heads=num_heads,
                                                    num_layers=transformer_layers,
                                                    dropout=dropout)
        # classifier input dim: attention concats heads -> feat_dim*num_heads;
        # transformer reads out a dim-d CLS token; topk/mean/max/additive use feat_dim
        # (additive classifies each cell pre-pool, so it sees a single feat_dim vector).
        if pooling == "attention":
            clf_in = feat_dim * num_heads
        elif pooling == "transformer":
            clf_in = self.aggregator.out_dim
        else:
            clf_in = feat_dim
        self.classifier = _make_classifier(
            clf_in, dropout,
            classifier_hidden=classifier_hidden, num_classes=num_classes, task_type=task_type,
        )
        self._setup_loss(task_type, num_classes, class_weights)
        self.feat_dim = feat_dim

    def _aggregate(self, feats):
        # Additive MIL (Javed et al. 2022): classify each cell, weight by attention, then
        # sum. Because the sum is linear, contribution(k,c) = a_k * classifier_c(h_k) is an
        # EXACT per-cell decomposition of bag_logits (summing to it), unlike attention
        # pooling where the classifier acts on the pooled vector. Early return.
        if self.pooling == "additive":
            _, attn = self.attention(feats)                       # [B, bag]
            per_inst_logits = self.classifier(feats)              # [B, bag, out]
            contributions = attn.unsqueeze(-1) * per_inst_logits  # [B, bag, out]
            bag_logits = contributions.sum(dim=1)                 # [B, out]
            if self.task_type == "binary":
                bag_logits = bag_logits.squeeze(-1)
            return bag_logits, attn

        # mi-Net: per-instance classification, pool logits (early return)
        if self.pooling == "linear":
            per_inst_logits = self.classifier(feats)  # [B, bag, out]
            bag_logits = per_inst_logits.mean(dim=1)  # [B, out]
            if self.task_type == "binary":
                scores = per_inst_logits.squeeze(-1)  # [B, bag]
                bag_logits = bag_logits.squeeze(-1)
            else:
                scores = per_inst_logits.max(dim=-1).values  # [B, bag]
            attn = torch.softmax(scores, dim=1)  # [B, bag]
            return bag_logits, attn

        if self.pooling == "mean":
            bag_repr = feats.mean(dim=1)
            attn = None
        elif self.pooling == "max":
            bag_repr = feats.max(dim=1).values
            attn = None
        elif self.pooling == "topk":
            bag_repr_full, attn_full = self.attention(feats)  # attn: [B, bag]
            k = min(self.top_k, feats.shape[1])
            topk_attn, topk_idx = torch.topk(attn_full, k, dim=1)
            topk_attn = topk_attn / (topk_attn.sum(dim=1, keepdim=True) + 1e-8)
            topk_feats = torch.gather(feats, 1, topk_idx.unsqueeze(-1).expand(-1, -1, feats.shape[-1]))
            bag_repr = (topk_attn.unsqueeze(-1) * topk_feats).sum(dim=1)
            attn = attn_full
        elif self.pooling == "transformer":
            bag_repr, attn = self.aggregator(feats)  # self-attention over cells
        else:  # attention
            bag_repr, attn = self.attention(feats)

        bag_logits = self.classifier(bag_repr)
        if self.task_type == "binary":
            bag_logits = bag_logits.squeeze(-1)
        return bag_logits, attn


class ResNetAttentionMIL(_AttentionMILMixin, BaseMILModel):
    """ResNet-based MIL model with attention pooling."""

    def __init__(self, backbone: str = 'resnet18', n_experiments=8,
                 attn_dim=128, lr=1e-4, lr_schedule="cosine", warmup_epochs=5, max_epochs=100,
                 task_type="binary", num_classes=2,
                 class_weights=None, in_channels=1, pretrained=True,
                 dropout=0.3, weight_decay=0.01, finetune_layers="layer3,layer4", num_heads=1,
                 projection_dim=None, pooling="attention", classifier_hidden=None, top_k=None,
                 encode_chunk_size=None, bag_label_mode="hard", attn_entropy_weight=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.dropout_rate = dropout
        self.num_heads = num_heads

        self.resnet_backbone, feat_dim = create_single_channel_resnet(backbone, in_channels=in_channels, pretrained=pretrained)

        _apply_finetune_layers(self.resnet_backbone, finetune_layers)

        self.resnet_backbone.fc = nn.Identity()
        self.backbone = self.resnet_backbone
        self.backbone_name = backbone.upper()
        self.name = f'MIL-{backbone.upper()}'

        self.projection, proj_dim = _build_projection(feat_dim, projection_dim, dropout)

        self._init_attention_head(proj_dim, attn_dim, num_heads, dropout,
                                  task_type, num_classes, class_weights,
                                  pooling=pooling, classifier_hidden=classifier_hidden, top_k=top_k)
        self._setup_mil(task_type, num_classes, n_experiments, lr, lr_schedule,
                        warmup_epochs, max_epochs, weight_decay, encode_chunk_size=encode_chunk_size,
                        bag_label_mode=bag_label_mode, attn_entropy_weight=attn_entropy_weight)
        self.save_hyperparameters(logger=False)

    def _encode_instances(self, x):
        return self.backbone(x)


class ViTLAttentionMIL(_AttentionMILMixin, BaseMILModel):
    """ViT-L-16 based MIL model with attention pooling."""

    def __init__(self, n_experiments=8, attn_dim=128, lr=1e-4,
                 lr_schedule="cosine", warmup_epochs=5, max_epochs=100,
                 task_type="binary", num_classes=2, class_weights=None, in_channels=1,
                 pretrained=True, dropout=0.3, weight_decay=0.01, finetune_layers="layer-1", num_heads=1,
                 projection_dim=None, pooling="attention", classifier_hidden=None, top_k=None,
                 encode_chunk_size=None, bag_label_mode="hard", attn_entropy_weight=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.dropout_rate = dropout
        self.num_heads = num_heads

        vit = models.vit_l_16(weights='IMAGENET1K_SWAG_E2E_V1' if pretrained else None)
        if in_channels != 3:
            vit.conv_proj = _adapt_conv1_channels(vit.conv_proj, in_channels)

        # Apply finetune_layers: ViT uses "layer{idx}" to reference encoder layers
        _apply_finetune_layers_with_resolver(
            vit, finetune_layers,
            lambda spec: vit.encoder.layers[int(spec.replace("layer", ""))] if spec.startswith("layer") else None,
        )

        feat_dim = vit.heads.head.in_features
        vit.heads = nn.Identity()
        self.backbone = vit
        self.backbone_name = 'ViT-L-16'
        self.name = 'MIL-ViT-L-16'

        self.projection, proj_dim = _build_projection(feat_dim, projection_dim, dropout)

        self._init_attention_head(proj_dim, attn_dim, num_heads, dropout,
                                  task_type, num_classes, class_weights,
                                  pooling=pooling, classifier_hidden=classifier_hidden, top_k=top_k)
        self._setup_mil(task_type, num_classes, n_experiments, lr, lr_schedule,
                        warmup_epochs, max_epochs, weight_decay, encode_chunk_size=encode_chunk_size,
                        bag_label_mode=bag_label_mode, attn_entropy_weight=attn_entropy_weight)
        self.save_hyperparameters(logger=False)

    def _encode_instances(self, x):
        return self.backbone(x)


class DINOv3AttentionMIL(_AttentionMILMixin, BaseMILModel):
    """DINOv3 ViT MIL model with attention pooling. Handles grayscale→RGB
    and ImageNet normalization internally."""

    def __init__(self, backbone="dino-vits16", n_experiments=8, attn_dim=128, lr=1e-4,
                 lr_schedule="cosine", warmup_epochs=5, max_epochs=100,
                 task_type="binary", num_classes=2, class_weights=None, in_channels=1,
                 dropout=0.3, weight_decay=0.01, finetune_layers="block10,block11",
                 num_heads=1, pretrained=True,
                 projection_dim=None, pooling="attention", classifier_hidden=None, top_k=None,
                 encode_chunk_size=None, bag_label_mode="hard", attn_entropy_weight=0.0,
                 patch_pool=False, dino_weights=None):
        super().__init__()
        self.in_channels = in_channels
        self.dropout_rate = dropout
        self.num_heads = num_heads
        self.patch_pool = patch_pool

        self.dino, feat_dim = load_dino_backbone(backbone, pretrained, dino_weights)

        # Apply finetune_layers: DINO uses "block{idx}" to reference transformer blocks
        _apply_finetune_layers_with_resolver(
            self.dino, finetune_layers,
            lambda spec: self.dino.blocks[int(spec.replace("block", ""))] if spec.startswith("block") else None,
        )

        # ImageNet normalization buffers
        self.register_buffer('norm_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('norm_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if patch_pool:
            self.patch_attention = PatchGatedAttention(feat_dim, attn_dim, dropout)

        label = backbone.upper().replace("DINO-", "DINOv3-")
        self.backbone_name = label
        self.name = f'MIL-{label}'

        self.projection, proj_dim = _build_projection(feat_dim, projection_dim, dropout)

        self._init_attention_head(proj_dim, attn_dim, num_heads, dropout,
                                  task_type, num_classes, class_weights,
                                  pooling=pooling, classifier_hidden=classifier_hidden, top_k=top_k)
        self._setup_mil(task_type, num_classes, n_experiments, lr, lr_schedule,
                        warmup_epochs, max_epochs, weight_decay, encode_chunk_size=encode_chunk_size,
                        bag_label_mode=bag_label_mode, attn_entropy_weight=attn_entropy_weight)
        self.save_hyperparameters(logger=False)

    def _encode_instances(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.norm_mean) / self.norm_std
        out = self.dino.forward_features(x)
        if self.patch_pool:
            features, _ = self.patch_attention(out["x_norm_patchtokens"])
        else:
            features = out["x_norm_clstoken"]
        return features


class MicroscopyCNNAttentionMIL(_AttentionMILMixin, BaseMILModel):
    """MIL model with custom microscopy encoder and attention pooling."""

    def __init__(self, n_experiments=8, attn_dim=128, lr=1e-4,
                 lr_schedule="cosine", warmup_epochs=5, max_epochs=100,
                 task_type="binary", num_classes=2, class_weights=None, feat_dim=256, in_channels=1,
                 dropout=0.3, weight_decay=0.01, num_heads=1,
                 projection_dim=None, pooling="attention", classifier_hidden=None,
                 pretrained=False, top_k=None, encode_chunk_size=None,
                 bag_label_mode="hard", attn_entropy_weight=0.0, finetune_layers="all"):
        # pretrained / finetune_layers are accepted for interface parity with the other
        # MIL models; the custom encoder has no pretrained weights and is always trained.
        super().__init__()
        self.in_channels = in_channels
        self.dropout_rate = dropout
        self.num_heads = num_heads

        self.backbone = MicroscopyEncoder(feat_dim=feat_dim, in_channels=in_channels)
        self.backbone_name = 'MicroscopyCNN'
        self.name = 'MIL-MicroscopyCNN'

        self.projection, proj_dim = _build_projection(feat_dim, projection_dim, dropout)

        self._init_attention_head(proj_dim, attn_dim, num_heads, dropout,
                                  task_type, num_classes, class_weights,
                                  pooling=pooling, classifier_hidden=classifier_hidden, top_k=top_k)
        self._setup_mil(task_type, num_classes, n_experiments, lr, lr_schedule,
                        warmup_epochs, max_epochs, weight_decay, encode_chunk_size=encode_chunk_size,
                        bag_label_mode=bag_label_mode, attn_entropy_weight=attn_entropy_weight)
        self.save_hyperparameters(logger=False)

    def _encode_instances(self, x):
        return self.backbone(x)
