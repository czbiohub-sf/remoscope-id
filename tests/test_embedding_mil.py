"""Unit tests for EmbeddingMIL and the `additive` pooling aggregator.

Covers:
1. Additive invariant: per-cell contributions sum EXACTLY to the bag logits, attention
   normalizes to 1, output shapes are correct for binary + multiclass.
2. EmbeddingMIL 2D reshape: `[B, D]` -> bags `[n_exp, bag_size, Dp]`.
3. Forward parity: `pooling="mean"` forward equals a manual mean-pool + classifier.
"""

import torch

from remo_id.models import get_model
from remo_id.models.embedding_mil import EmbeddingMIL


D = 32


def _make(pooling, task_type, num_classes=2, **kw):
    return get_model(
        "embedding-mil", embedding_dim=D, n_experiments=2, attn_dim=16,
        pooling=pooling, task_type=task_type, num_classes=num_classes,
        max_epochs=1, **kw,
    ).eval()


def test_registered():
    assert isinstance(_make("attention", "binary"), EmbeddingMIL)


def test_additive_invariant_multiclass():
    model = _make("additive", "multiclass", num_classes=3)
    feats = torch.randn(2, 50, D)
    with torch.no_grad():
        bag_logits, attn = model._aggregate(feats)
        # Recompute the exact decomposition and confirm it sums to the bag logits.
        _, attn2 = model.attention(feats)
        contributions = attn2.unsqueeze(-1) * model.classifier(feats)
        recomputed = contributions.sum(dim=1)
    assert bag_logits.shape == (2, 3)
    assert attn.shape == (2, 50)
    assert torch.allclose(bag_logits, recomputed, atol=1e-5)
    # Attention is a softmax over instances -> sums to 1 per bag.
    assert torch.allclose(attn.sum(dim=1), torch.ones(2), atol=1e-5)


def test_additive_invariant_binary():
    model = _make("additive", "binary")
    feats = torch.randn(2, 40, D)
    with torch.no_grad():
        bag_logits, attn = model._aggregate(feats)
        _, attn2 = model.attention(feats)
        contributions = attn2.unsqueeze(-1) * model.classifier(feats)
        recomputed = contributions.sum(dim=1).squeeze(-1)
    assert bag_logits.shape == (2,)
    assert torch.allclose(bag_logits, recomputed, atol=1e-5)


def test_encode_and_reshape_2d():
    model = _make("attention", "binary")
    x = torch.randn(16, D)  # 16 cells = 4 experiments x 4 cells (n_exp=4 here)
    feats = model._encode_and_reshape(x, n_experiments=4)
    assert feats.shape == (4, 4, D)


def test_forward_mean_parity():
    model = _make("mean", "binary", projection_dim=None)
    x = torch.randn(8, D)  # n_experiments=2 -> bags of 4
    with torch.no_grad():
        out = model.forward(x)
        feats = x.view(2, 4, D).mean(dim=1)
        manual = model.classifier(feats).squeeze(-1)
    assert out.shape == (2,)
    assert torch.allclose(out, manual, atol=1e-5)


def test_additive_also_works_on_image_mil():
    # The additive branch lives in the shared mixin, so it works for any MIL model.
    model = get_model("microscopy-cnn-mil", n_experiments=2, attn_dim=16,
                      pooling="additive", task_type="multiclass", num_classes=3,
                      max_epochs=1).eval()
    feats = torch.randn(2, 20, model.feat_dim)
    with torch.no_grad():
        bag_logits, attn = model._aggregate(feats)
        recomputed = (attn.unsqueeze(-1) * model.classifier(feats)).sum(dim=1)
    assert torch.allclose(bag_logits, recomputed, atol=1e-5)
