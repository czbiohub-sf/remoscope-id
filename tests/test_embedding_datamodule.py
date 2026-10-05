"""Offline test of the embedding data path: cell_config classify + capped pools + true
counts + weighted-by-true-composition bag sampler. No GPU, no DB, no real data."""
import os

import numpy as np
import pandas as pd
import pytest
import torch

from remo_id.data.embedding_store import write_experiment_embeddings
from remo_id.data.cell_config import CellClass, CellConfig, classify_cells, load_cell_config
from remo_id.data.embedding_datamodule import EmbeddingDataModule
from remo_id.data.sampler import EmbeddingBatchSampler
from remo_id.models import get_model

D = 16
MV = "testver"
# class order = priority: platelet(0), wbc(1), rbc(2 catch-all). Weights enrich wbc.
CELLS = [
    CellClass("platelet", "(area < 500) & (intensity_mean > 140)", 5.0),
    CellClass("wbc", "(area > 2300) | (intensity_mean > 140)", 20.0),
    CellClass("rbc", "*", 1.0),
]
PLT, WBC, RBC = 0, 1, 2


def _write_store(exp_dir, exp_id, n_plt, n_wbc, n_rbc):
    os.makedirs(exp_dir, exist_ok=True)
    rng = np.random.default_rng(abs(hash(exp_id)) % 2**31)
    area = np.concatenate([rng.uniform(100, 450, n_plt), rng.uniform(2500, 4000, n_wbc),
                           rng.uniform(600, 1500, n_rbc)])
    inten = np.concatenate([rng.uniform(150, 200, n_plt), rng.uniform(150, 200, n_wbc),
                            rng.uniform(80, 130, n_rbc)])
    rp = pd.DataFrame({"area": area, "intensity_mean": inten})
    rp.to_parquet(os.path.join(exp_dir, f"{exp_id}_regionprops.parquet"))
    n = len(rp)
    emb = rng.standard_normal((n, D)).astype(np.float16)
    write_experiment_embeddings(exp_dir, exp_id, MV, emb, np.arange(n, dtype=np.int32),
                                dict(model_config_hash="h", ssl_backbone="t", latent_dim=D))


def _make_module(tmp_path, specs, pool_cap=600):
    rows = []
    for i, (eid, label, npl, nw, nr) in enumerate(specs):
        d = os.path.join(tmp_path, f"e{eid}")
        _write_store(d, eid, npl, nw, nr)
        rows.append(dict(experiment_id=int(eid), path=d, num_images=1,
                         sample_id=f"s{i}", class_label=label))
    data = pd.DataFrame(rows)

    dm = object.__new__(EmbeddingDataModule)
    dm.model_version = MV
    dm.cell_config = CellConfig(classes=CELLS, pool_cap=pool_cap)
    dm.n_cell_classes = 3
    dm.cell_weights = dm.cell_config.weights
    dm.pool_cap = pool_cap
    dm.load_workers = 2
    dm.uses_regionprops = True
    dm.task_type = "multiclass"
    dm.num_classes = 3
    dm.database_column = "class_label"
    dm.exp_embeddings, dm.class_local, dm.true_counts, dm.embedding_dim = {}, {}, {}, None

    data, ranges, labels = dm._finalize_ranges(data, "class_label", "multiclass")
    dm.data, dm.experiment_ranges, dm.experiment_labels = data, ranges, labels
    dm._exp_id_by_path = dict(zip(data["path"], data["experiment_id"].astype(int)))
    dm.length = sum(e - s for s, e in ranges.values())
    dm._rebuild_index()
    dm.augments = None
    dm.dataloader = dm  # inference shim: module is its own dataloader
    return dm


# --- cell_config parser ---
def test_cell_config_parse_and_classify(tmp_path):
    p = os.path.join(tmp_path, "cc.yaml")
    with open(p, "w") as f:
        f.write("pool_cap: 1000\ncells:\n"
                "  - {name: wbc, where: '(area > 2300) | (intensity_mean > 140)', weight: 20}\n"
                "  - {name: rbc, where: '*', weight: 1}\n")
    cfg = load_cell_config(p)
    assert cfg.names == ["wbc", "rbc"] and cfg.pool_cap == 1000
    df = pd.DataFrame({"area": [3000, 800, 400], "intensity_mean": [100, 100, 200]})
    codes = classify_cells(df, cfg)
    assert list(codes) == [0, 1, 0]  # big->wbc, small/dim->rbc, small/bright->wbc(int>140)


def test_catch_all_must_be_last(tmp_path):
    p = os.path.join(tmp_path, "bad.yaml")
    with open(p, "w") as f:
        f.write("cells:\n  - {name: rbc, where: '*', weight: 1}\n"
                "  - {name: wbc, where: 'area > 2300', weight: 5}\n")
    with pytest.raises(ValueError):
        load_cell_config(p)


# --- datamodule: capping + true counts ---
def test_caps_pool_keeps_true_counts(tmp_path):
    dm = _make_module(tmp_path, [("100", 0, 300, 500, 3000)], pool_cap=600)
    path = next(iter(dm.true_counts))
    tc = dm.true_counts[path]
    assert tc[PLT] == 300 and tc[WBC] == 500 and tc[RBC] == 3000   # TRUE counts (uncapped)
    cl = dm.class_local[path]
    assert len(cl[PLT]) == 300 and len(cl[WBC]) == 500             # rare classes kept whole
    assert len(cl[RBC]) == 600                                     # RBC capped to pool_cap
    assert dm.exp_embeddings[path].dtype == torch.float16          # fp16 resident


# --- sampler: weighted by true composition ---
def test_sampler_weighted_by_true_counts(tmp_path):
    dm = _make_module(tmp_path, [(str(1000 + i), i % 3, 300, 500, 3000) for i in range(9)],
                      pool_cap=600)
    samp = EmbeddingBatchSampler(
        dm.experiment_ranges, dm.experiment_labels, dm.class_local, dm.true_counts,
        dm.cell_weights, batch_size=3000, n_experiments=3, num_batches=5,
        task_type="multiclass", num_classes=3)
    # global -> class lookup
    g2c = {}
    for path, (start, _e) in dm.experiment_ranges.items():
        for ci, locs in dm.class_local[path].items():
            for li in locs:
                g2c[start + int(li)] = ci
    types = [g2c[g] for g in next(iter(samp))]
    wbc_frac = types.count(WBC) / len(types)
    # expected ∝ weight*true_count: 20*500 / (5*300 + 20*500 + 1*3000) = 10000/14500 ≈ 0.69
    assert 0.60 < wbc_frac < 0.78, f"WBC fraction {wbc_frac} off expected ~0.69"


def test_sampler_yields_full_num_batches(tmp_path):
    # The image BatchSampler would drain pools and stop at ~(smallest_class//exps_per_class)
    # batches; the embedding sampler must yield the FULL num_batches (resampling w/ replacement).
    dm = _make_module(tmp_path, [(str(2000 + i), i % 3, 300, 500, 3000) for i in range(9)],
                      pool_cap=600)
    samp = EmbeddingBatchSampler(
        dm.experiment_ranges, dm.experiment_labels, dm.class_local, dm.true_counts,
        dm.cell_weights, batch_size=300, n_experiments=3, num_batches=50,
        task_type="multiclass", num_classes=3)
    assert len(samp) == 50
    batches = list(samp)
    assert len(batches) == 50 and all(len(b) == 300 for b in batches)


def test_cross_experiment_pooled_bags(tmp_path):
    # Train-only mode: each bag pools cells from ALL experiments of one disease class.
    dm = _make_module(tmp_path, [(str(4000 + i), i % 3, 300, 500, 3000) for i in range(9)],
                      pool_cap=600)
    samp = EmbeddingBatchSampler(
        dm.experiment_ranges, dm.experiment_labels, dm.class_local, dm.true_counts,
        dm.cell_weights, batch_size=300, n_experiments=3, num_batches=5,
        task_type="multiclass", num_classes=3, cross_experiment=True)
    g2lab, g2exp = {}, {}
    for path, (start, end) in dm.experiment_ranges.items():
        for g in range(start, end):
            g2lab[g] = dm.experiment_labels[path]
            g2exp[g] = path
    batch = next(iter(samp))
    assert len(batch) == 300
    for b in range(3):  # each contiguous 100-cell bag:
        bag = batch[b * 100:(b + 1) * 100]
        assert len({g2lab[g] for g in bag}) == 1            # uniform disease label
        assert len({g2exp[g] for g in bag}) >= 2            # pooled across multiple patients


def test_no_drop_tiny_experiment(tmp_path):
    # WBC-poor experiment is NOT dropped; bag still full (weighted draw, with replacement).
    dm = _make_module(tmp_path, [("200", 0, 300, 500, 3000), ("201", 1, 5, 5, 50),
                                 ("202", 2, 300, 500, 3000)], pool_cap=600)
    assert len(dm.experiment_ranges) == 3
    samp = EmbeddingBatchSampler(
        dm.experiment_ranges, dm.experiment_labels, dm.class_local, dm.true_counts,
        dm.cell_weights, batch_size=300, n_experiments=3, num_batches=3,
        task_type="multiclass", num_classes=3)
    tiny = [p for p in dm.experiment_ranges if dm._exp_id_by_path[p] == 201][0]
    bag = samp._sample_cells_for_exp(tiny, 100)
    assert len(bag) == 100  # full bag despite only 60 cells total (with replacement)


def test_inference_building_blocks(tmp_path):
    # The post-training/CV/LOO inference path = pick_global_indices -> dataloader.get_items
    # -> model.forward_with_attention. Verify the embedding shim satisfies that contract.
    from remo_id.inference.mil import pick_global_indices_for_experiment
    dm = _make_module(tmp_path, [(str(500 + i), i % 3, 300, 500, 3000) for i in range(6)],
                      pool_cap=600)
    model = get_model("embedding-mil", embedding_dim=D, n_experiments=1, attn_dim=8,
                      pooling="attention", task_type="multiclass", num_classes=3, max_epochs=1).eval()
    path = next(iter(dm.experiment_ranges))
    idxs = pick_global_indices_for_experiment(dm.experiment_ranges, path, 100, seed=0)
    assert len(idxs) == 100
    x, y, eid = dm.dataloader.get_items(idxs)        # the shim
    assert x.shape == (100, D) and x.dtype == torch.float32
    with torch.no_grad():
        logits, attn, feats = model.forward_with_attention(x)   # n_experiments=1 -> one bag
    assert logits.shape == (1, 3)
    assert attn.shape[1] == 100


def test_collate_and_model_forward(tmp_path):
    dm = _make_module(tmp_path, [(str(300 + i), i % 3, 300, 500, 3000) for i in range(6)],
                      pool_cap=600)
    samp = EmbeddingBatchSampler(
        dm.experiment_ranges, dm.experiment_labels, dm.class_local, dm.true_counts,
        dm.cell_weights, batch_size=150, n_experiments=3, num_batches=2,
        task_type="multiclass", num_classes=3)
    items = [dm[i] for i in next(iter(samp))]
    x, y, exp_ids = dm.test_collate_fn(items)
    assert x.shape == (150, D) and x.dtype == torch.float32
    model = get_model("embedding-mil", embedding_dim=D, n_experiments=3, attn_dim=8,
                      pooling="additive", task_type="multiclass", num_classes=3, max_epochs=1)
    loss, logits, _ = model._forward_step((x, y, exp_ids))
    loss.backward()
    assert logits.shape == (3, 3) and torch.isfinite(loss)


def test_mixed_soft_bags(tmp_path):
    """Mixed bags (majority_min_pct<1.0) put ~60% majority-class cells + diluted off-class
    cells in each bag, and the embedding soft path turns the per-cell labels into a bag
    proportion target. Covers both the sampler mixing and the end-to-end soft forward."""
    # 2 experiments per disease class so each class has a pool to draw from.
    dm = _make_module(tmp_path, [(str(400 + i), i % 3, 300, 500, 3000) for i in range(6)],
                      pool_cap=600)
    samp = EmbeddingBatchSampler(
        dm.experiment_ranges, dm.experiment_labels, dm.class_local, dm.true_counts,
        dm.cell_weights, batch_size=150, n_experiments=3, num_batches=4,
        task_type="multiclass", num_classes=3,
        majority_min_pct=0.6, bag_label_mode="soft")
    spe = 150 // 3  # 50 cells/bag

    def label_of(g):
        for p, (s0, e0) in dm.experiment_ranges.items():
            if s0 <= g < e0:
                return dm.experiment_labels[p]

    import collections
    batch = next(iter(samp))
    assert len(batch) == 150
    for b in range(3):
        labs = [label_of(g) for g in batch[b * spe:(b + 1) * spe]]
        cnt = collections.Counter(labs)
        _maj, majn = cnt.most_common(1)[0]
        assert 0.5 <= majn / spe <= 0.7   # ~0.6 majority fraction
        assert len(cnt) >= 2              # genuinely mixed (off-class cells present)

    # End-to-end soft forward on the embedding path (the collate must keep per-cell labels).
    x, y, exp_ids = dm.test_collate_fn([dm[i] for i in batch])
    model = get_model("embedding-mil", embedding_dim=D, n_experiments=3, attn_dim=8,
                      pooling="transformer", task_type="multiclass", num_classes=3,
                      bag_label_mode="soft", max_epochs=1)
    loss, logits, bag_labels = model._forward_step((x, y, exp_ids))
    loss.backward()
    assert logits.shape == (3, 3) and torch.isfinite(loss)
    # Soft targets are proportion vectors (rows sum to 1), and at least one bag is mixed.
    assert torch.allclose(bag_labels.sum(dim=1), torch.ones(3), atol=1e-5)
    assert (bag_labels.max(dim=1).values < 0.99).any()


def test_mixed_bags_require_soft(tmp_path):
    """Hard labels + mixed bags is ill-defined (per-cell labels aren't uniform) -> reject."""
    dm = _make_module(tmp_path, [(str(500 + i), i % 3, 300, 500, 3000) for i in range(6)],
                      pool_cap=600)
    with pytest.raises(ValueError):
        EmbeddingBatchSampler(
            dm.experiment_ranges, dm.experiment_labels, dm.class_local, dm.true_counts,
            dm.cell_weights, batch_size=150, n_experiments=3, num_batches=1,
            task_type="multiclass", num_classes=3,
            majority_min_pct=0.6, bag_label_mode="hard")
