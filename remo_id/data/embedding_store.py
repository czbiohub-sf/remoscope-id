"""Per-experiment cell-embedding store read by ``--model embedding-mil``.

One zarr-v3 group per experiment and embedding model, colocated in the
experiment directory::

    {experiment_path}/{exp_id}_emb__{model_version}.zarr/
        embeddings       (N, D) float16   one row per cell (or any instance)
        regionprops_row  (N,)   int32     row of each embedding in the
                                          experiment's *regionprops.parquet
        (group .attrs)   provenance       must include ``emb_sha256`` (written
                                          automatically) and ``model_config_hash``

``regionprops_row`` links embeddings to per-cell morphology so a ``--cell_config``
can gate cells into classes; without a cell config it is not used. Any embedding
model works: produce an ``(N, D)`` array per experiment and call
:func:`write_experiment_embeddings`. Every experiment in a training run must
carry the same ``model_config_hash`` (embeddings from different models are never
mixed; see :func:`assert_single_version`).
"""
from __future__ import annotations

import hashlib
import math
import os

import numpy as np
import zarr


def emb_path(experiment_path: str, exp_id: str, model_version: str) -> str:
    """Colocated zarr path for one experiment's embeddings under a model version."""
    return os.path.join(experiment_path, f"{exp_id}_emb__{model_version}.zarr")


def _emb_sha256(emb: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(emb).tobytes()).hexdigest()


def write_experiment_embeddings(
    experiment_path: str,
    exp_id: str,
    model_version: str,
    emb: np.ndarray,
    regionprops_row: np.ndarray | None = None,
    provenance: dict | None = None,
    *,
    chunk_rows: int = 8192,
) -> str:
    """Write one experiment's ``(N, D)`` embeddings as a sharded zarr-v3 group.

    ``emb`` is stored as float16 and ``regionprops_row`` (default ``arange(N)``)
    as int32. ``provenance`` goes into the group ``.attrs`` together with
    ``emb_sha256``; ``model_config_hash`` defaults to ``model_version`` when not
    given. Returns the store path.
    """
    emb = np.ascontiguousarray(emb, dtype=np.float16)
    if emb.ndim != 2:
        raise ValueError(f"emb must be 2D (N, D); got shape {emb.shape}")
    n, d = emb.shape
    if regionprops_row is None:
        regionprops_row = np.arange(n)
    regionprops_row = np.ascontiguousarray(regionprops_row, dtype=np.int32)
    if regionprops_row.shape != (n,):
        raise ValueError(f"len(emb)={n} != len(regionprops_row)={regionprops_row.shape[0]}")

    attrs = {"model_config_hash": model_version, **(provenance or {})}
    attrs["emb_sha256"] = _emb_sha256(emb)

    path = emb_path(experiment_path, exp_id, model_version)
    crows = max(1, min(chunk_rows, n))
    srows = max(crows, math.ceil(n / crows) * crows)  # one shard -> one file
    grp = zarr.open_group(path, mode="w")
    a = grp.create_array(name="embeddings", shape=(n, d), dtype="float16",
                         chunks=(crows, d), shards=(srows, d), compressors=None)
    a[:] = emb
    r = grp.create_array(name="regionprops_row", shape=(n,), dtype="int32",
                         chunks=(crows,), shards=(srows,), compressors=None)
    r[:] = regionprops_row
    grp.attrs.update(attrs)
    return path


def read_experiment_embeddings(experiment_path: str, exp_id: str, model_version: str):
    """Return ``(emb, regionprops_row, attrs)``, reading the whole store.

    Whole sequential reads are the fast path on parallel file systems; filter
    rows in memory afterwards.
    """
    grp = zarr.open_group(emb_path(experiment_path, exp_id, model_version), mode="r")
    return grp["embeddings"][:], grp["regionprops_row"][:], dict(grp.attrs)


def is_cached(experiment_path: str, exp_id: str, model_version: str) -> bool:
    """True if a complete embedding store exists for this experiment + version."""
    path = emb_path(experiment_path, exp_id, model_version)
    if not os.path.isdir(path):
        return False
    try:
        grp = zarr.open_group(path, mode="r")
        return "embeddings" in grp and "regionprops_row" in grp and "emb_sha256" in grp.attrs
    except Exception:
        return False


def assert_single_version(metas) -> str:
    """Assert all provenance dicts share one ``model_config_hash``; return it."""
    hashes = {m.get("model_config_hash") for m in metas}
    if None in hashes:
        raise ValueError("Some embeddings are missing 'model_config_hash' provenance.")
    if len(hashes) != 1:
        raise ValueError(f"Embeddings span multiple model versions: {sorted(hashes)}")
    return next(iter(hashes))
