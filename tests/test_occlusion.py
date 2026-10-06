"""Cell occlusion: fill-model invariants on synthetic frames/masks, the figure selection
rules, and a tiny end-to-end CPU screen (background -> attribution -> screen -> report) through ``remo-id analysis occlusion`` on a synthetic MIL checkpoint."""
import glob
import os

import cv2
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from remo_id.cli import cli
from remo_id.interpret import occlusion as O
from remo_id.interpret.cell_occlusion import (recorded_experiments, select_example_frames,
                                              select_informative_cells)

from synthetic_data import cell_regionprops, make_synthetic_dataset, synthetic_frames


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)


def _frames(n=20, size=128, class_idx=1, seed=0):
    f, m = synthetic_frames(n, size, class_idx, np.random.default_rng(seed), return_masks=True)
    return np.moveaxis(f, 2, 0).copy(), np.moveaxis(m, 2, 0).copy()


def test_regionprops_match_masks():
    f, m = synthetic_frames(3, 64, 0, np.random.default_rng(1), return_masks=True)
    rp = cell_regionprops(f, m)
    for r in rp.itertuples():
        assert (m[:, :, r.image_id] == r.label).sum() == r.area
    # the frames do not depend on whether masks are returned
    f2 = synthetic_frames(3, 64, 0, np.random.default_rng(1))
    assert np.array_equal(f, f2)


def test_background_model_is_empty_slide():
    frames, masks = _frames()
    bg, frac = O.build_bg_model(frames, masks)
    assert bg.shape == frames.shape[1:] and bg.dtype == np.float32
    assert 0.0 <= frac < 1.0
    assert abs(float(np.median(bg)) - 225) < 3       # cells (~135) are excluded


def _isolated_label(mask):
    """A cell whose neighbourhood (closing + dilation + seam band) has no other cell."""
    reach = O.CLOSE_RADIUS + O.DILATE + 2 * 4 + 2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    for lab in np.unique(mask[mask > 0]):
        near = cv2.dilate((mask == lab).astype(np.uint8), k).astype(bool)
        if not ((mask > 0) & (mask != lab) & near).any():
            return int(lab)
    raise AssertionError("no isolated cell in the synthetic frame")


def test_occlude_fills_removed_cell_and_keeps_the_rest():
    frames, masks = _frames()
    bg, _ = O.build_bg_model(frames, masks)
    img, mask = frames[0], masks[0]
    lab = _isolated_label(mask)
    out = O.occlude(img, mask, [lab], bg, np.random.default_rng(5))
    core, keep = O.fill_core(mask, mask == lab)
    # kept cells are untouched (none is near the removed cell)
    assert np.array_equal(out[keep], img[keep])
    # nothing changes beyond the feathered seam of the filled region
    d = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 3)
    far = d > O.FEATHER + 1
    assert np.array_equal(out[far], img[far])
    # the removed cell is replaced by background-like pixels
    cell = mask == lab
    assert img[cell].mean() < 180
    assert abs(float(out[cell].mean()) - float(np.median(bg))) < 15
    # deterministic for a given generator seed; empty removal is the identity
    assert np.array_equal(out, O.occlude(img, mask, [lab], bg, np.random.default_rng(5)))
    assert np.array_equal(O.occlude(img, mask, [], bg, np.random.default_rng(5)), img)


def test_occlude_seam_only_touches_band_of_kept_neighbours():
    frames, masks = _frames(seed=3)
    bg, _ = O.build_bg_model(frames, masks)
    img, mask = frames[1], masks[1]
    labs = np.unique(mask[mask > 0])
    lab = int(labs[0])
    ff = O.frame_fill_model(img, bg, O.cell_free(mask), mask)
    out = O.occlude(img, mask, [lab], bg, np.random.default_rng(0), noise_params=ff)
    core, keep = O.fill_core(mask, mask == lab)
    band = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 3) \
        <= 2 * ff.edge_sigma + 2
    assert np.array_equal(out[keep & ~band], img[keep & ~band])


def test_select_informative_cells_quota_and_one_per_experiment():
    rng = np.random.default_rng(0)
    n = 60
    cells = pd.DataFrame({
        "experiment_id": [str(100 + i % 15) for i in range(n)],
        "image_id": np.arange(n), "label": 1, "y_true": [i % 3 for i in range(n)],
        "cell_type": rng.choice(["wbc", "rbc", "platelet"], n),
        "ex_d_own_minus_sham": -rng.uniform(0.01, 1, n),
        "ex_drop_own_vs_null": rng.uniform(-0.1, 1, n),
        "touches_border": rng.random(n) < 0.1, "frame_correct": rng.random(n) < 0.9})
    sel = select_informative_cells(cells, n=4, type_quota={"wbc": 2, "rbc": 2}, seed=1)
    assert (sel.groupby(["y_true", "experiment_id"]).size() == 1).all()
    assert (sel.ex_drop_own_vs_null > 0).all() and (sel.drop_own_minus_sham > 0).all()
    assert not sel.touches_border.any() and sel.frame_correct.all()
    for _, g in sel.groupby("y_true"):
        assert list(g.slot) == list(range(len(g)))
        assert len(g) <= 4
    again = select_informative_cells(cells, n=4, type_quota={"wbc": 2, "rbc": 2}, seed=1)
    pd.testing.assert_frame_equal(sel, again)


CELLS_YAML = "cells:\n  - {name: big, where: 'area > 100', weight: 5}\n" \
             "  - {name: small, where: '*', weight: 1}\n"


def _run(args):
    result = CliRunner().invoke(cli, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


def test_end_to_end_cpu_screen(tmp_path):
    manifest = make_synthetic_dataset(str(tmp_path / "data"), n_per_class=4, n_images=12,
                                      size=96, masks=True)
    save = tmp_path / "ckpt"
    _run(["train", "--manifest", manifest, "--model", "resnet18-mil", "--save_dir", str(save),
          "--accelerator", "cpu", "--max_epochs", "1", "--num_batches", "5",
          "--n_experiments", "2", "--batch_size", "8", "--crop_size", "64",
          "--no-pretrained", "--seed", "0", "--no-infer"])
    run_dir = glob.glob(str(save / "*_resnet18-mil"))[0]
    ckpt = os.path.join(run_dir, "last.ckpt")
    cells = tmp_path / "cells.yaml"
    cells.write_text(CELLS_YAML)
    work = tmp_path / "occ"
    common = ["--checkpoint", ckpt, "--work_dir", str(work), "--min_image_index", "0",
              "--n_frames", "6", "--device", "cpu", "--experiment_ids", "1",
              "--experiment_ids", "5"]
    _run(["analysis", "occlusion", "background", *common])
    _run(["analysis", "occlusion", "attribution", *common, "--cell_config", str(cells)])
    screen = ["--cell_config", str(cells), "--s1_frames", "3", "--n_masks", "8", "--p", "0.3",
              "--n_null", "2", "--n_random_per_type", "2", "--gpu_batch", "4", "--workers", "2",
              "--border_px", "2"]
    _run(["analysis", "occlusion", "screen", *common, *screen])
    _run(["analysis", "occlusion", "report", "--work_dir", str(work)])
    ex = recorded_experiments(str(work))
    assert sorted(ex.experiment_id) == ["1", "5"] and sorted(ex.y_true) == [0, 1]
    assert (ex.checkpoint == os.path.abspath(ckpt)).all()
    for e in ex.experiment_id:
        assert (work / "bg" / f"{e}.npy").exists()
        assert (work / "attribution" / f"{e}.parquet").exists()
        s1 = pd.read_parquet(work / "screen" / "stage1" / f"{e}.cells.parquet")
        assert np.isfinite(s1.s1_drop_own).all()
        s2 = pd.read_parquet(work / "screen" / "stage2" / f"{e}.parquet")
        for col in ("ex_d_own", "sham_d_own", "ex_d_own_minus_sham", "null_kind", "why"):
            assert col in s2.columns
        assert np.isfinite(s2.ex_d_own).all()
    rep = pd.read_parquet(work / "report" / "screen_cells.parquet")
    assert len(rep) and rep["class"].notna().all()
    assert (work / "report" / "screen_top_manifest.csv").exists()

    # same settings in a fresh work dir -> identical tables (seeded, deterministic)
    work2 = tmp_path / "occ2"
    common2 = [a if a != str(work) else str(work2) for a in common]
    for stage, extra in (("background", []), ("attribution", ["--cell_config", str(cells)]),
                         ("screen", screen)):
        _run(["analysis", "occlusion", stage, *common2, *extra])
    for f in sorted(glob.glob(str(work / "screen" / "stage*" / "*.parquet"))):
        g = f.replace(str(work), str(work2))
        pd.testing.assert_frame_equal(pd.read_parquet(f), pd.read_parquet(g))

    # rule-based frame choice runs on the screen outputs
    fr = select_example_frames(str(work), classes=[0, 1], p_range=(1e-6, 1 - 1e-6),
                               max_cover=1.0, n_ref=3)
    assert {"cand_rank", "lapvar_pct", "score", "p_true"} <= set(fr.columns)
