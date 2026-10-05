"""Manifest parsing/validation, manifest class configs, zarr image access,
embedding-store round trip and LOO epoch-checkpoint resolution. CPU only."""
import os

import numpy as np
import pandas as pd
import pytest
import torch

from remo_id.data.class_config import class_config_from_dict
from remo_id.data.embedding_store import (
    assert_single_version, emb_path, is_cached, read_experiment_embeddings,
    write_experiment_embeddings,
)
from remo_id.data.manifest import (
    assign_manifest_classes, is_manifest_class_config, load_manifest, manifest_class_config,
    manifest_classes,
)
from remo_id.data.zarr_io import get_image_from_zarr, get_zarr_path
from remo_id.workflows.loo_report import _find_epoch_checkpoint

from synthetic_data import make_synthetic_dataset, synthetic_frames, write_image_stack


def _write_csv(tmp_path, rows, name="m.csv"):
    path = os.path.join(tmp_path, name)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _row(eid, label, **kw):
    return dict(experiment_id=eid, path=f"exp_{eid}", num_images=10, label=label, **kw)


# --- load_manifest ---------------------------------------------------------
def test_load_manifest_resolves_paths_and_optional_columns(tmp_path):
    path = _write_csv(tmp_path, [_row(1, "a", sample_id="7"), _row(2, "b")])
    df = load_manifest(path)
    assert list(df["experiment_id"]) == ["1", "2"]
    assert df["path"].tolist() == [os.path.join(tmp_path, "exp_1"), os.path.join(tmp_path, "exp_2")]
    assert df["num_images"].dtype == np.int64
    assert df.loc[0, "sample_id"] == "7" and df.loc[1, "sample_id"] is None
    assert df["study_id"].isna().all() and df["site"].isna().all()


def test_load_manifest_keeps_absolute_paths_and_custom_label_column(tmp_path):
    rows = [dict(experiment_id=1, path="/data/exp_001", num_images=5, diagnosis="x")]
    df = load_manifest(_write_csv(tmp_path, rows), label_column="diagnosis")
    assert df.loc[0, "path"] == "/data/exp_001" and df.loc[0, "diagnosis"] == "x"


@pytest.mark.parametrize("rows, match", [
    ([dict(experiment_id=1, path="p", label="a")], "missing required"),
    ([_row(1, "a"), _row(1, "b")], "duplicate experiment_id"),
    ([_row("abc", "a")], "integer"),
    ([dict(experiment_id=1, path="p", num_images=0, label="a")], "num_images"),
    ([dict(experiment_id=1, path="p", num_images=2.5, label="a")], "num_images"),
])
def test_load_manifest_rejects_invalid(tmp_path, rows, match):
    with pytest.raises(ValueError, match=match):
        load_manifest(_write_csv(tmp_path, rows))


def test_empty_labels_are_nan(tmp_path):
    df = load_manifest(_write_csv(tmp_path, [_row(1, "a"), _row(2, None)]))
    assert df.loc[0, "label"] == "a" and pd.isna(df.loc[1, "label"])


# --- classes ---------------------------------------------------------------
def test_manifest_classes_order(tmp_path):
    df = load_manifest(_write_csv(tmp_path, [_row(i, lab) for i, lab in
                                             enumerate(["pos", "neg", "pos", "mid"])]))
    assert manifest_classes(df) == ["mid", "neg", "pos"]
    assert manifest_classes(df, classes=["pos", "neg", "mid"]) == ["pos", "neg", "mid"]
    with pytest.raises(ValueError, match="do not occur"):
        manifest_classes(df, classes=["pos", "other"])
    assert manifest_classes(df, classes=["pos", "other"], require_present=False) == ["pos", "other"]


def test_numeric_labels_sort_numerically(tmp_path):
    df = load_manifest(_write_csv(tmp_path, [_row(i, lab) for i, lab in enumerate([10, 2, 1])]))
    assert manifest_classes(df) == ["1", "2", "10"]


def test_manifest_class_config_binary_and_multiclass(tmp_path):
    path = _write_csv(tmp_path, [_row(1, "neg"), _row(2, "pos")])
    raw = manifest_class_config(path)
    assert is_manifest_class_config(raw["classes"])
    cfg = class_config_from_dict(raw)
    assert cfg.class_names == ["neg", "pos"] and cfg.task_type == "binary" and cfg.num_classes == 2

    path3 = _write_csv(tmp_path, [_row(1, "a"), _row(2, "b"), _row(3, "c")], "m3.csv")
    cfg3 = class_config_from_dict(manifest_class_config(path3))
    assert cfg3.task_type == "multiclass" and cfg3.num_classes == 3

    with pytest.raises(ValueError, match="at least two classes"):
        manifest_class_config(_write_csv(tmp_path, [_row(1, "a")], "m1.csv"))


def test_assign_manifest_classes(tmp_path):
    path = _write_csv(tmp_path, [_row(1, "neg"), _row(2, "pos"), _row(3, "pos"),
                                 _row(4, None), _row(5, "other")])
    defs = manifest_class_config(path, classes=["neg", "pos"])["classes"]
    df, names, unassigned = assign_manifest_classes(defs)
    assert names == ["neg", "pos"]
    assert dict(zip(df["experiment_id"], df["class_label"])) == {"1": 0, "2": 1, "3": 1}
    assert unassigned == {os.path.join(tmp_path, "exp_4"), os.path.join(tmp_path, "exp_5")}

    df, _, _ = assign_manifest_classes(defs, exclude_filters=[{"experiment_ids": [2]}])
    assert "2" not in set(df["experiment_id"])

    df, _, _ = assign_manifest_classes(defs, default_class_idx=0)
    assert dict(zip(df["experiment_id"], df["class_label"]))["5"] == 0

    with pytest.raises(ValueError, match="only support"):
        assign_manifest_classes(defs, exclude_filters=[{"study_id": "x"}])


def test_single_class_manifest_entry_is_binary_vs_rest(tmp_path):
    path = _write_csv(tmp_path, [_row(1, "neg"), _row(2, "pos"), _row(3, None)])
    defs = [{"name": "pos", "manifest": path, "column": "label", "value": "pos"}]
    df, _, unassigned = assign_manifest_classes(defs)
    assert dict(zip(df["experiment_id"], df["class_label"])) == {"1": 0, "2": 1}
    assert len(unassigned) == 1


# --- zarr image access + datamodule from manifest ---------------------------
def test_zarr_io_reads_frames(tmp_path):
    frames = synthetic_frames(5, 32, 0, np.random.default_rng(0))
    exp = os.path.join(tmp_path, "exp_1")
    write_image_stack(exp, frames)
    assert get_zarr_path(exp).endswith("exp_1.zip")
    np.testing.assert_array_equal(get_image_from_zarr(exp, 3), frames[:, :, 3])
    np.testing.assert_array_equal(get_image_from_zarr(exp, [0, 4]), frames[:, :, [0, 4]])
    assert get_image_from_zarr(exp).shape == (32, 32, 5)


def test_datamodule_from_manifest(tmp_path):
    from remo_id.data import RemoIDDataModule
    manifest = make_synthetic_dataset(str(tmp_path), n_per_class=3, n_images=6, size=32,
                                      replicates=2)
    dm = RemoIDDataModule(manifest=manifest, backbone_name="resnet18", crop_size=32, batch_size=4)
    assert dm.database is None
    assert dm.task_type == "binary" and dm.class_definitions[1]["name"] == "positive"
    assert len(dm.data) == 12 and sorted(dm.experiment_labels.values()) == [0] * 6 + [1] * 6
    x, y, eid = dm[0]
    assert x.shape == (1, 32, 32)
    train, val = dm.split_train_val(val_ratio=0.34, seed=0)
    # replicates of a sample never straddle the split
    assert not set(train.data["sample_id"]) & set(val.data["sample_id"])


# --- embedding store ---------------------------------------------------------
def test_embedding_store_round_trip(tmp_path):
    emb = np.random.default_rng(0).standard_normal((50, 8)).astype(np.float32)
    exp = str(tmp_path)
    assert not is_cached(exp, "7", "v1")
    path = write_experiment_embeddings(exp, "7", "v1", emb, provenance={"embedder": "toy"})
    assert path == emb_path(exp, "7", "v1") and path.endswith("7_emb__v1.zarr")
    assert is_cached(exp, "7", "v1")
    got, rp_row, attrs = read_experiment_embeddings(exp, "7", "v1")
    assert got.dtype == np.float16 and got.shape == (50, 8)
    np.testing.assert_allclose(got, emb.astype(np.float16))
    np.testing.assert_array_equal(rp_row, np.arange(50))
    assert attrs["embedder"] == "toy" and attrs["model_config_hash"] == "v1" and "emb_sha256" in attrs
    assert assert_single_version([attrs, attrs]) == "v1"
    with pytest.raises(ValueError, match="multiple model versions"):
        assert_single_version([attrs, {"model_config_hash": "other"}])
    with pytest.raises(ValueError):
        write_experiment_embeddings(exp, "8", "v1", emb, regionprops_row=np.arange(3))


# --- LOO epoch checkpoint resolution ------------------------------------------
def test_find_epoch_checkpoint_falls_back_to_matching_last_ckpt(tmp_path):
    fold = str(tmp_path)
    torch.save({"epoch": 49, "state_dict": {"w": torch.zeros(2)}}, os.path.join(fold, "last.ckpt"))
    assert _find_epoch_checkpoint(fold, 50) == os.path.join(fold, "last.ckpt")
    assert _find_epoch_checkpoint(fold, 49) is None
    torch.save({"epoch": 9}, os.path.join(fold, "epoch=10.ckpt"))
    assert _find_epoch_checkpoint(fold, 10) == os.path.join(fold, "epoch=10.ckpt")
    assert _find_epoch_checkpoint(fold, -1) == os.path.join(fold, "last.ckpt")
