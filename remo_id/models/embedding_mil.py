"""MIL model that consumes precomputed per-cell embeddings.

Unlike the image-based MIL models in ``mil.py`` (which run a backbone on image crops),
``EmbeddingMIL`` takes precomputed ``[D]`` cell-embedding vectors as instances. The
backbone is ``nn.Identity`` and the whole attention/aggregation stack from
``_AttentionMILMixin`` is reused unchanged. The only structural difference is that the
input batch is 2D ``[B, D]`` (a flat stack of cell vectors) rather than 4D
``[B, C, H, W]`` images, so ``_encode_and_reshape`` is overridden to unpack 2D.

The embeddings come from per-experiment stores (``{exp_id}_emb__<version>.zarr``, see
``remo_id.data.embedding_store``); the EmbeddingDataModule reads them, gates cells, and feeds ``[B, D]`` batches here.
"""

import torch.nn as nn

from remo_id.models.mil import (
    BaseMILModel,
    _AttentionMILMixin,
    _build_projection,
)


class EmbeddingMIL(_AttentionMILMixin, BaseMILModel):
    """Attention-MIL over precomputed cell embeddings (instance = a [D] vector)."""

    def __init__(self, embedding_dim=384, n_experiments=8, attn_dim=128, lr=1e-4,
                 lr_schedule="cosine", warmup_epochs=5, max_epochs=100,
                 task_type="binary", num_classes=2, class_weights=None,
                 dropout=0.3, weight_decay=0.01, num_heads=1,
                 projection_dim=None, pooling="attention", classifier_hidden=None,
                 top_k=None, transformer_layers=2, bag_label_mode="hard",
                 attn_entropy_weight=0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout
        self.num_heads = num_heads

        # No backbone: instances arrive pre-embedded.
        self.backbone = nn.Identity()
        self.backbone_name = 'Embedding'
        self.name = 'MIL-Embedding'

        # Optional learned projection is the main trainable adapter on frozen embeddings.
        self.projection, proj_dim = _build_projection(embedding_dim, projection_dim, dropout)

        self._init_attention_head(proj_dim, attn_dim, num_heads, dropout,
                                  task_type, num_classes, class_weights,
                                  pooling=pooling, classifier_hidden=classifier_hidden,
                                  top_k=top_k, transformer_layers=transformer_layers)
        self._setup_mil(task_type, num_classes, n_experiments, lr, lr_schedule,
                        warmup_epochs, max_epochs, weight_decay,
                        bag_label_mode=bag_label_mode, attn_entropy_weight=attn_entropy_weight)
        self.save_hyperparameters(logger=False)
        # Persist model_name so downstream loaders (inference/CV/LOO) identify the arch.
        self.hparams["model_name"] = "embedding-mil"

    def _encode_instances(self, x):
        return self.backbone(x)  # Identity: [B, D] -> [B, D]

    def _encode_and_reshape(self, x, n_experiments=None):
        """Reshape a flat batch of cell embeddings ``[B, D]`` into bags.

        Overrides the base ``[B, C, H, W]`` (4D) path: instances are vectors, not images.
        """
        n_exp = n_experiments or self.n_experiments
        B, D = x.shape
        assert B % n_exp == 0, \
            f"Batch size {B} not divisible by n_experiments={n_exp}"
        bag_size = B // n_exp
        feats_flat = self._encode_instances(x)
        if hasattr(self, 'projection'):
            feats_flat = self.projection(feats_flat)
        Dp = feats_flat.shape[-1]
        return feats_flat.view(n_exp, bag_size, Dp)
