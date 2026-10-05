"""End-to-end CLI smoke tests on synthetic data described by a CSV manifest:
one tiny CPU training run per architecture family, then inference from the
resulting checkpoint. No database, no GPU, no network (``--no-pretrained``)."""
import glob
import os

import pytest
from click.testing import CliRunner

from remo_id.cli import cli

from synthetic_data import make_synthetic_dataset, write_synthetic_embeddings

CELL_CONFIG = os.path.join(os.path.dirname(__file__), "..", "examples", "cell_configs",
                           "wbc_rbc_plt.yaml")
COMMON = ["--accelerator", "cpu", "--max_epochs", "1", "--num_batches", "5",
          "--n_experiments", "2", "--no-pretrained", "--seed", "0"]


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("synthetic"))
    manifest = make_synthetic_dataset(root, n_per_class=4, n_images=12, size=96)
    write_synthetic_embeddings(manifest, "toyv1", dim=16, n_cells=200)
    return manifest


def _run(args):
    result = CliRunner().invoke(cli, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


def _train(manifest, save_dir, model, *extra):
    _run(["train", "--manifest", manifest, "--model", model, "--save_dir", str(save_dir),
          *COMMON, *extra])
    run_dirs = glob.glob(os.path.join(str(save_dir), f"*_{model}"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    ckpt = os.path.join(run_dir, "best.ckpt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(run_dir, "last.ckpt")
    assert os.path.exists(ckpt)
    for sidecar in ("classes.yaml", "splits.yaml"):
        assert os.path.exists(os.path.join(run_dir, sidecar)), sidecar
    return run_dir, ckpt


@pytest.mark.parametrize("model, extra", [
    ("resnet18", ["--batch_size", "8", "--crop_size", "64"]),
    ("resnet18-mil", ["--batch_size", "8", "--crop_size", "64"]),
])
def test_train_and_infer_images(dataset, tmp_path, model, extra):
    run_dir, ckpt = _train(dataset, tmp_path / "ckpt", model, *extra)
    assert os.path.exists(os.path.join(run_dir, "predictions.npz"))
    out = tmp_path / "reports"
    _run(["infer", "--checkpoint", ckpt, "--manifest", dataset, "--bag_size", "4",
          "--output_path", str(out)])
    assert any(f.endswith(".pdf") for f in os.listdir(out))


@pytest.mark.parametrize("cell_config", [None, CELL_CONFIG])
def test_train_embedding_mil(dataset, tmp_path, cell_config):
    extra = ["--model_version", "toyv1", "--cells_per_bag", "16", "--pooling", "additive"]
    if cell_config:
        extra += ["--cell_config", cell_config]
    else:
        extra += ["--pool_cap", "100"]
    _train(dataset, tmp_path / "ckpt", "embedding-mil", *extra)


def test_loo_and_loo_predict_from_last_ckpt(dataset, tmp_path, monkeypatch):
    # Inside a SLURM job the CV/LOO path trains folds in-process instead of submitting.
    monkeypatch.setenv("SLURM_JOB_ID", os.environ.get("SLURM_JOB_ID", "0"))
    save_dir = tmp_path / "loo"
    _run(["train", "--manifest", dataset, "--model", "resnet18-mil", "--loo",
          "--save_dir", str(save_dir), "--batch_size", "8", "--crop_size", "64", *COMMON])
    cv_dirs = glob.glob(os.path.join(str(save_dir), "*_cv_resnet18-mil_loo"))
    assert len(cv_dirs) == 1
    cv_dir = cv_dirs[0]
    assert len(glob.glob(os.path.join(cv_dir, "repeat0_fold*", "last.ckpt"))) == 8
    # Trained without --save_epochs: epoch 1 resolves to each fold's last.ckpt.
    _run(["workflow", "loo-predict", "--cv_dir", cv_dir, "--manifest", dataset,
          "--epoch", "1", "--bag_size", "4", "--n_bags", "1"])
    assert os.path.exists(os.path.join(cv_dir, "predictions", "epoch_1.npz"))
