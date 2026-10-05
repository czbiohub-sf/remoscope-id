"""Synthetic Remoscope-style data for tests and examples.

Writes experiments in the layout remo-id reads (see ``remo_id/data/zarr_io.py``
and ``remo_id/data/manifest.py``): one directory per experiment holding a
``uint8 [H, W, N]`` zarr array inside a ``.zip`` (zarr ``ZipStore``), plus a
manifest CSV. Optionally adds per-cell embedding stores
(``remo_id/data/embedding_store.py``) and a ``*_regionprops.parquet`` with
``area`` / ``intensity_mean`` columns for cell_config gating, or (``masks=True``)
label masks ``*_masks.zarr`` with a regionprops table that describes exactly the
cells drawn in the images (the occlusion input format, see
``remo_id/interpret/cell_occlusion.py``).

Images are a bright background (~225, like brightfield frames) with dark
"cells"; the class shifts cell size so models have a learnable signal.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import zarr

from remo_id.data.embedding_store import write_experiment_embeddings


def write_image_stack(exp_dir: str, frames: np.ndarray) -> str:
    """Write ``frames`` (uint8 ``[H, W, N]``) as ``{exp_dir}/{basename}.zip``."""
    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, f"{os.path.basename(os.path.normpath(exp_dir))}.zip")
    store = zarr.storage.ZipStore(path, mode="w")
    h, w, _ = frames.shape
    arr = zarr.create_array(store=store, shape=frames.shape, dtype="uint8", chunks=(h, w, 1))
    arr[:] = frames
    store.close()
    return path


def synthetic_frames(n_images: int, size: int, class_idx: int, rng: np.random.Generator,
                     return_masks: bool = False):
    """``[size, size, n_images]`` uint8 frames with class-dependent cell radius.

    With ``return_masks`` also returns the int32 label masks of the drawn cells (same
    shape; label k = the k-th cell drawn in that frame, a later cell overwrites an
    earlier one where they overlap). The frames are identical either way.
    """
    yy, xx = np.mgrid[:size, :size]
    frames = rng.normal(225, 6, size=(size, size, n_images))
    masks = np.zeros(frames.shape, np.int32) if return_masks else None
    radius = 4 + 3 * class_idx
    for i in range(n_images):
        for k in range(max(4, size * size // 4000)):
            cy, cx = rng.integers(0, size, 2)
            disk = (yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2
            frames[disk, i] -= 90
            if return_masks:
                masks[disk, i] = k + 1
    frames = np.clip(frames, 0, 255).astype(np.uint8)
    return (frames, masks) if return_masks else frames


def cell_regionprops(frames: np.ndarray, masks: np.ndarray) -> pd.DataFrame:
    """Regionprops rows (occlusion format) of every labelled cell of every frame:
    image_id, label, area, centroid_y/x, bbox ty/tx/by/bx, intensity_mean."""
    rows = []
    for i in range(frames.shape[2]):
        m = masks[:, :, i]
        for lab in np.unique(m[m > 0]):
            ys, xs = np.nonzero(m == lab)
            rows.append(dict(image_id=i, label=int(lab), area=float(len(ys)),
                             centroid_y=ys.mean(), centroid_x=xs.mean(),
                             ty=int(ys.min()), tx=int(xs.min()),
                             by=int(ys.max()) + 1, bx=int(xs.max()) + 1,
                             intensity_mean=float(frames[ys, xs, i].mean())))
    return pd.DataFrame(rows)


def write_cell_masks(exp_dir: str, exp_id, frames: np.ndarray, masks: np.ndarray) -> None:
    """Write ``{exp_id}_masks.zarr`` (int32 ``[H, W, N]``) and the matching
    ``{exp_id}_regionprops.parquet`` into ``exp_dir``."""
    h, w, _ = masks.shape
    arr = zarr.create_array(store=os.path.join(exp_dir, f"{exp_id}_masks.zarr"),
                            shape=masks.shape, dtype="int32", chunks=(h, w, 1))
    arr[:] = masks
    cell_regionprops(frames, masks).to_parquet(
        os.path.join(exp_dir, f"{exp_id}_regionprops.parquet"))


def make_synthetic_dataset(root: str, classes=("negative", "positive"), n_per_class: int = 4,
                           n_images: int = 24, size: int = 96, replicates: int = 1,
                           seed: int = 0, masks: bool = False) -> str:
    """Write ``n_per_class`` samples per class (each with ``replicates`` experiments)
    under ``root`` and return the manifest path (``root/manifest.csv``).

    Paths in the manifest are relative to ``root``. ``masks`` also writes each
    experiment's cell label masks and regionprops (:func:`write_cell_masks`).
    """
    rng = np.random.default_rng(seed)
    rows, exp_id = [], 1
    for c, name in enumerate(classes):
        for s in range(n_per_class):
            for _ in range(replicates):
                rel = f"exp_{exp_id:03d}"
                frames, labels = synthetic_frames(n_images, size, c, rng, return_masks=True)
                write_image_stack(os.path.join(root, rel), frames)
                if masks:
                    write_cell_masks(os.path.join(root, rel), exp_id, frames, labels)
                rows.append(dict(experiment_id=exp_id, path=rel, num_images=n_images,
                                 label=name, sample_id=f"sample_{c}_{s}",
                                 study_id="synthetic", site="site_a"))
                exp_id += 1
    path = os.path.join(root, "manifest.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_synthetic_embeddings(manifest_path: str, model_version: str, dim: int = 16,
                               n_cells: int = 300, seed: int = 0) -> None:
    """Write an embedding store and a regionprops table for every manifest row.

    Cells are a mix of small/dim "rbc"-like, large "wbc"-like and small/bright
    "platelet"-like rows (matching ``examples/cell_configs/wbc_rbc_plt.yaml``);
    the embedding mean shifts with the class so the task is learnable. When the
    experiment already has a regionprops table (``make_synthetic_dataset(masks=True)``)
    there is one embedding per row of that table instead.
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(manifest_path)
    base = os.path.dirname(os.path.abspath(manifest_path))
    classes = sorted(df["label"].astype(str).unique())
    for _, row in df.iterrows():
        exp_dir = row["path"] if os.path.isabs(row["path"]) else os.path.join(base, row["path"])
        exp_id = str(row["experiment_id"])
        rp_path = os.path.join(exp_dir, f"{exp_id}_regionprops.parquet")
        if os.path.exists(rp_path):          # cells drawn in the images (masks=True)
            n = len(pd.read_parquet(rp_path, columns=["area"]))
        else:
            n = n_cells
            n_wbc, n_plt = n_cells // 10, n_cells // 10
            n_rbc = n_cells - n_wbc - n_plt
            area = np.concatenate([rng.uniform(600, 1500, n_rbc), rng.uniform(2500, 4000, n_wbc),
                                   rng.uniform(100, 450, n_plt)])
            inten = np.concatenate([rng.uniform(80, 130, n_rbc), rng.uniform(80, 130, n_wbc),
                                    rng.uniform(150, 200, n_plt)])
            pd.DataFrame({"area": area, "intensity_mean": inten}).to_parquet(rp_path)
        shift = classes.index(str(row["label"]))
        emb = rng.standard_normal((n, dim)) + shift
        write_experiment_embeddings(exp_dir, exp_id, model_version, emb,
                                    regionprops_row=np.arange(n))
