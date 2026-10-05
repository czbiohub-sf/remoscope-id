"""Read Remoscope image stacks stored as zipped zarr arrays.

Each experiment is a directory containing one image stack saved as a zarr array
inside a ``.zip`` file (zarr ``ZipStore``). The array is ``uint8`` with shape
``[H, W, N]``: frame ``i`` is ``array[:, :, i]``. Hidden files (``.*.zip``) are
ignored and the first other ``.zip`` in the directory is used.

Write a compatible stack with::

    import zarr
    store = zarr.storage.ZipStore("exp_001/exp_001.zip", mode="w")
    arr = zarr.create_array(store=store, shape=frames.shape, dtype="uint8",
                            chunks=(frames.shape[0], frames.shape[1], 1))
    arr[:] = frames          # uint8 [H, W, N]
    store.close()
"""
from __future__ import annotations

import os

import numpy as np
import zarr


def get_zarr_path(experiment_path: str) -> str:
    """Return the path of the image-stack ``.zip`` inside ``experiment_path``."""
    for fname in os.listdir(experiment_path):
        if fname.endswith(".zip") and not fname.startswith("."):
            return os.path.join(experiment_path, fname)
    raise FileNotFoundError(f"No .zip image stack found in {experiment_path}")


def get_zarr_file(experiment_path: str) -> zarr.Array:
    """Open the experiment's ``[H, W, N]`` image stack read-only."""
    return zarr.open(store=zarr.storage.ZipStore(get_zarr_path(experiment_path), mode="r"), mode="r")


def get_image_from_zarr(experiment_path: str, img_ids=None) -> np.ndarray:
    """Read frames from the experiment's image stack.

    ``img_ids`` may be ``None`` (all frames, ``[H, W, N]``), an int (one frame,
    ``[H, W]``) or a list of ints (``[H, W, len(img_ids)]``).
    """
    arr = get_zarr_file(experiment_path)
    if img_ids is None:
        return arr[:, :, :]
    return arr[:, :, img_ids]
