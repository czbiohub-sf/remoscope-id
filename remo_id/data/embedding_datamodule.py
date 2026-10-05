"""Data module that feeds precomputed per-cell embeddings to ``EmbeddingMIL``.

Subclasses :class:`RemoIDDataModule` to reuse experiment/label resolution
(``_prepare_data`` / ``_prepare_data_class_config``) and the sample_id-stratified split
decision (``_decide_train_val_paths``). It overrides the data-access half.

**Capped per-class pools + true counts.** At load, each cell is assigned a class via the
``cell_config`` gates (see :mod:`remo_id.data.cell_config`). Without a ``cell_config``
every embedding belongs to one class (weight 1) and no regionprops table is needed. Per experiment we keep at most
``pool_cap`` DISTINCT cells per class (random subset if over; rare classes kept whole) and
also record the **true, uncapped per-class counts**. A full cohort can be terabytes, so we
never hold all cells; the cap bounds RAM while the true counts let
:class:`EmbeddingBatchSampler` compose bags at ``weight_c * true_count_c`` — preserving
cross-sample composition even though abundant classes are capped. Pools are stored **fp16**
(cast to fp32 per item) to halve resident memory.

**Loading (Lustre).** Each experiment's embedding array is read WHOLE and sequentially
(``rows=None``) — selective row reads are slower because cell types scatter across every
chunk of the single-shard store — then classified + pooled in RAM, on a bounded thread pool.
The store format is documented in :mod:`remo_id.data.embedding_store`.
"""
from __future__ import annotations

import bisect
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from remo_id.data.cell_config import (
    CATCH_ALL, DEFAULT_POOL_CAP, CellClass, CellConfig, classify_cells, load_cell_config,
)
from remo_id.data.class_config import class_config_from_dict
from remo_id.data.datamodule import RemoIDDataModule, open_database
from remo_id.data.embedding_store import (
    assert_single_version,
    is_cached,
    read_experiment_embeddings,
)
from remo_id.data.manifest import is_manifest_class_config, manifest_class_config


def _default_load_workers() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    cpus = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 4)
    return max(1, min(cpus, 8))


def _find_regionprops_path(experiment_path: str) -> str:
    for fname in os.listdir(experiment_path):
        if fname.endswith("regionprops.parquet"):
            return os.path.join(experiment_path, fname)
    raise FileNotFoundError(f"No *regionprops.parquet in {experiment_path}")


class EmbeddingDataModule(RemoIDDataModule):
    """Per-experiment capped cell-class pools; instance = a ``[D]`` vector, bag = experiment."""

    def __init__(self,
                 model_version: str,
                 cell_config: "CellConfig | str | None" = None,
                 database_column: str = "class_label",
                 database_label=None,
                 label_contains: str | None = None,
                 study_id=None,
                 study_ids: list | None = None,
                 threshold=None,
                 threshold_mode: str = ">",
                 negative_study_ids: list | None = None,
                 negative_experiment_ids: list | None = None,
                 include_experiment_ids: list | None = None,
                 batch_size: int = 256,
                 task_type: str = "binary",
                 num_classes: int = 2,
                 class_definitions: list | None = None,
                 skip_null_labels: bool = True,
                 train: bool = True,
                 exclude_filters: list | None = None,
                 include_filters: list | None = None,
                 default_class_idx: int | None = None,
                 load_workers: int | None = None,
                 pool_cap: int | None = None,
                 manifest: str | None = None,
                 label_column: str = "label",
                 classes: list | None = None):
        self.model_version = model_version
        if cell_config is None:
            # No gating: every embedding is one class; regionprops are not read.
            self.cell_config = CellConfig(
                classes=[CellClass(name="all", where=CATCH_ALL, weight=1.0)],
                pool_cap=DEFAULT_POOL_CAP)
            self.uses_regionprops = False
        else:
            self.cell_config = load_cell_config(cell_config) if isinstance(cell_config, str) else cell_config
            self.uses_regionprops = True
        if pool_cap is not None:
            if pool_cap <= 0:
                raise ValueError(f"pool_cap must be positive, got {pool_cap}")
            self.cell_config.pool_cap = int(pool_cap)
        self.n_cell_classes = len(self.cell_config.classes)
        self.cell_weights = self.cell_config.weights          # [n_cell_classes]
        self.pool_cap = self.cell_config.pool_cap
        self.load_workers = load_workers or _default_load_workers()

        if manifest is not None:
            cfg = class_config_from_dict(manifest_class_config(manifest, label_column, classes))
            class_definitions, task_type, num_classes = cfg.class_definitions, cfg.task_type, cfg.num_classes
            database_column = "class_label"
        self.label_column = label_column
        uses_db = not is_manifest_class_config(class_definitions)
        self.database = open_database() if uses_db else None
        self.min_image_index = 0
        self.crop_size = None
        self.skip_null_labels = skip_null_labels
        self.database_label = database_label
        self.label_contains = label_contains
        self.database_column = database_column.lower()
        self.study_id = study_id
        self.study_ids = study_ids or []
        self.negative_study_ids = negative_study_ids
        self.negative_experiment_ids = negative_experiment_ids
        self.include_experiment_ids = include_experiment_ids
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.task_type = task_type
        self.num_classes = num_classes
        self.class_definitions = class_definitions
        self.exclude_filters = exclude_filters or []
        self.include_filters = include_filters or []
        self.default_class_idx = default_class_idx
        self.batch_size = batch_size
        self.backbone_name = "embedding"
        self.train = train
        self._null_label_paths = set()

        # In-RAM per-experiment data, keyed by experiment path.
        self.exp_embeddings: dict[str, torch.Tensor] = {}   # path -> [N_kept, D] fp16
        self.class_local: dict[str, dict] = {}              # path -> {class_idx: local row idx}
        self.true_counts: dict[str, np.ndarray] = {}        # path -> [n_cell_classes] true counts
        self.embedding_dim: int | None = None

        print(f"Preparing embedding data (cell classes: {self.cell_config.names}, "
              f"weights {list(self.cell_weights)}, pool_cap {self.pool_cap:,})...")
        self.data, self.experiment_ranges, self.experiment_labels = self._prepare_data(
            self.database, self.database_column, self.database_label, self.label_contains,
            self.study_id, self.study_ids, self.threshold, self.threshold_mode,
            self.negative_study_ids, self.negative_experiment_ids, self.include_experiment_ids,
            self.task_type, self.num_classes, self.class_definitions, self.skip_null_labels,
            self.exclude_filters, self.include_filters,
            self.default_class_idx,
        )

        self._exp_id_by_path = dict(zip(self.data["path"], self.data["experiment_id"].astype(int)))
        self.length = sum(e - s for s, e in self.experiment_ranges.values())
        self._rebuild_index()
        self._set_task_attrs(self.data)
        # Inference compatibility: the module is its own dataloader (get_items shim);
        # augments is unused (embeddings are precomputed) but train.py sets it.
        self.augments = None
        self.dataloader = self

    # ------------------------------------------------------------------ #
    # Load + classify + cap (overrides the image-zarr range builder)
    # ------------------------------------------------------------------ #
    def _load_one(self, exp_path: str, exp_id: str):
        if not is_cached(exp_path, exp_id, self.model_version):
            return None
        t0 = time.time()
        emb, rp_row, attrs = read_experiment_embeddings(exp_path, exp_id, self.model_version)
        t1 = time.time()
        if self.uses_regionprops:
            rp = pd.read_parquet(_find_regionprops_path(exp_path))
            meta = rp.iloc[rp_row].reset_index(drop=True)  # per-emb-row metadata
            codes = classify_cells(meta, self.cell_config)  # class idx per emb row, -1 if unmatched
        else:
            codes = np.zeros(emb.shape[0], dtype=np.int64)
        true_counts = np.array([(codes == i).sum() for i in range(self.n_cell_classes)], dtype=np.int64)

        keep_parts = []
        for i in range(self.n_cell_classes):
            idx_i = np.where(codes == i)[0]
            if len(idx_i) > self.pool_cap:
                idx_i = np.random.choice(idx_i, self.pool_cap, replace=False)
            keep_parts.append(idx_i)
        keep = np.sort(np.concatenate(keep_parts)) if keep_parts else np.array([], dtype=int)
        emb_kept = torch.from_numpy(np.ascontiguousarray(emb[keep])).to(torch.float16)
        kept_codes = codes[keep]
        t2 = time.time()
        stats = dict(
            read_s=t1 - t0, filter_s=t2 - t1, bytes_read=int(emb.nbytes),
            n_total=int(emb.shape[0]), n_kept=int(emb_kept.shape[0]),
            n_unmatched=int((codes == -1).sum()), true_counts=true_counts,
        )
        return emb_kept, kept_codes, true_counts, attrs, stats

    def _finalize_ranges(self, data, column, task_type):
        rows = [(str(r["path"]), str(r["experiment_id"])) for _, r in data.iterrows()]
        wall0 = time.time()
        loaded: dict[str, tuple] = {}
        metas = []
        with ThreadPoolExecutor(max_workers=self.load_workers) as pool:
            futures = {pool.submit(self._load_one, p, e): p for p, e in rows}
            for fut in futures:
                path = futures[fut]
                res = fut.result()
                if res is None:
                    continue
                emb_kept, kept_codes, true_counts, attrs, stats = res
                if emb_kept.shape[0] == 0:
                    continue
                loaded[path] = (emb_kept, kept_codes, true_counts, stats)
                metas.append(attrs)

        if not loaded:
            raise RuntimeError(
                f"No embedding stores found for model_version='{self.model_version}' "
                f"across {len(rows)} experiments.")
        assert_single_version(metas)

        experiment_ranges, experiment_labels = {}, {}
        global_idx = 0
        dims = set()
        for _, row in data.iterrows():
            path = str(row["path"])
            if path not in loaded:
                continue
            emb_kept, kept_codes, true_counts, _stats = loaded[path]
            n = emb_kept.shape[0]
            self.exp_embeddings[path] = emb_kept
            self.class_local[path] = {i: np.where(kept_codes == i)[0] for i in range(self.n_cell_classes)}
            self.true_counts[path] = true_counts
            dims.add(emb_kept.shape[1])
            experiment_labels[path] = int(row[column])
            experiment_ranges[path] = (global_idx, global_idx + n)
            global_idx += n

        if len(dims) != 1:
            raise ValueError(f"Embeddings span multiple dims: {dims}")
        self.embedding_dim = dims.pop()
        data = data[data["path"].isin(self.exp_embeddings)].reset_index(drop=True)
        self._log_load_summary(loaded, time.time() - wall0)
        return data, experiment_ranges, experiment_labels

    def _log_load_summary(self, loaded: dict, wall_s: float):
        n_exp = len(loaded)
        gb = 1024 ** 3
        total_read = sum(s["bytes_read"] for *_, s in loaded.values())
        n_total = sum(s["n_total"] for *_, s in loaded.values())
        n_kept = sum(s["n_kept"] for *_, s in loaded.values())
        resident = sum(t.element_size() * t.nelement() for t, *_ in loaded.values())
        read_s = sum(s["read_s"] for *_, s in loaded.values())
        filter_s = sum(s["filter_s"] for *_, s in loaded.values())
        true_tot = np.sum([s["true_counts"] for *_, s in loaded.values()], axis=0)
        kept_per_class = np.zeros(self.n_cell_classes, dtype=np.int64)
        for _emb, codes, *_ in loaded.values():
            for i in range(self.n_cell_classes):
                kept_per_class[i] += int((codes == i).sum())
        names = self.cell_config.names
        comp = " ".join(f"{names[i]} {true_tot[i]:,}(kept {kept_per_class[i]:,})"
                        for i in range(self.n_cell_classes))
        read_list = sorted(s["read_s"] for *_, s in loaded.values())
        print(
            f"[EmbeddingLoad] {n_exp} exps | dim={self.embedding_dim} | kept {n_kept:,} of "
            f"{n_total:,} cells | true {comp} | read {total_read/gb:.1f} GB in {read_s:.0f}s "
            f"aggregate ({total_read/gb/max(read_s,1e-9):.2f} GB/s) | resident {resident/gb:.1f} GB "
            f"(fp16) | wall {wall_s:.0f}s | workers={self.load_workers}")
        print(
            f"[EmbeddingLoad] per-exp read_s: min={read_list[0]:.1f} "
            f"p50={read_list[len(read_list)//2]:.1f} max={read_list[-1]:.1f} | "
            f"filter {filter_s:.0f}s aggregate | "
            f"parallel speedup {read_s/max(wall_s,1e-9):.2f}x at workers={self.load_workers}")

    # ------------------------------------------------------------------ #
    # Item access
    # ------------------------------------------------------------------ #
    def _rebuild_index(self):
        items = sorted(self.experiment_ranges.items(), key=lambda kv: kv[1][0])
        self._paths = [p for p, _ in items]
        self._starts = [s for _, (s, _e) in items]
        self._ends = [e for _, (_s, e) in items]

    def _locate(self, idx):
        i = bisect.bisect_right(self._starts, idx) - 1
        if i < 0 or idx >= self._ends[i]:
            raise IndexError(f"Index {idx} out of range.")
        return self._paths[i], idx - self._starts[i]

    def __getitem__(self, idx):
        path, local_row = self._locate(idx)
        emb = self.exp_embeddings[path][local_row].float()  # fp16 resident -> fp32 per item
        return emb, self.experiment_labels[path], self._exp_id_by_path[path]

    def sample_bag_indices(self, exp_path, bag_size, seed=None):
        """Global indices for one experiment's inference bag, sampled with the SAME
        cell_config weighting as training (weight_c * true_count_c, drawn from capped pools
        with replacement). Used by inference so eval bags match the training composition
        instead of being uniform over the kept pool (which is dominated by the capped class).
        """
        start, _end = self.experiment_ranges[exp_path]
        rng = np.random.default_rng(seed)
        pools = self.class_local[exp_path]
        probs = self.cell_weights * self.true_counts[exp_path].astype(np.float64)
        total = probs.sum()
        if total <= 0:  # no weighted cells — uniform over kept pool
            n_kept = sum(len(v) for v in pools.values())
            if n_kept == 0:
                return []
            return [start + int(i) for i in rng.integers(0, n_kept, size=bag_size)]
        n_per = rng.multinomial(bag_size, probs / total)
        out = []
        for ci in range(self.n_cell_classes):
            n_c = int(n_per[ci])
            pool = pools.get(ci)
            if n_c <= 0 or pool is None or len(pool) == 0:
                continue
            out.extend(int(start + li) for li in rng.choice(pool, size=n_c, replace=True))
        if len(out) < bag_size:  # top up if a weighted class had an empty pool
            nonempty = [i for i in range(self.n_cell_classes) if len(pools.get(i, [])) > 0]
            while len(out) < bag_size and nonempty:
                p = pools[nonempty[rng.integers(len(nonempty))]]
                out.append(int(start + p[rng.integers(len(p))]))
        return out[:bag_size]

    def get_items(self, idxs):
        """Inference shim mirroring RemoIDDataLoader.get_items: gather embeddings by global
        index, returning (x [B,D] f32, y [B] f32, exp_ids [B] i64). No augmentation.

        inference/mil.py calls ``ds.dataloader.get_items(idxs)``; the module is its own
        ``dataloader`` (self.dataloader = self), so post-training inference, CV, and the LOO
        workflow work without an image-zarr loader.
        """
        if len(idxs) == 0:
            raise ValueError("Empty batch.")
        embs, labels, exp_ids = [], [], []
        for i in idxs:
            path, local_row = self._locate(int(i))
            embs.append(self.exp_embeddings[path][local_row])
            labels.append(self.experiment_labels[path])
            exp_ids.append(self._exp_id_by_path[path])
        x = torch.stack(embs, dim=0).float()
        y = torch.tensor(labels, dtype=torch.float32)
        exp_ids = torch.tensor(exp_ids, dtype=torch.long)
        return x, y, exp_ids

    # ------------------------------------------------------------------ #
    # Splitting (share loaded pools + counts by reference)
    # ------------------------------------------------------------------ #
    def _set_task_attrs(self, data):
        col = self.database_column
        if self.task_type == "binary":
            self.pos_ids = data[data[col] == 1]["experiment_id"].tolist()
            self.neg_ids = data[data[col] == 0]["experiment_id"].tolist()
        elif self.task_type == "multiclass":
            self.class_experiment_ids = {
                c: data[data[col] == c]["experiment_id"].tolist() for c in range(self.num_classes)}
            self.pos_ids, self.neg_ids = [], []
        else:
            self.pos_ids, self.neg_ids = [], []

    def split_train_val(self, val_ratio=0.2, seed=None, split_by_sample_id=True):
        train_paths, val_paths = self._decide_train_val_paths(val_ratio, seed, split_by_sample_id)
        return self.split_from_paths(train_paths, val_paths)

    def split_from_paths(self, train_paths: set, val_paths: set):
        return (self._build_child(train_paths, is_train=True),
                self._build_child(val_paths, is_train=False))

    def _build_child(self, paths: set, is_train: bool) -> "EmbeddingDataModule":
        col = self.database_column
        new_data = self.data[self.data["path"].isin(paths)].reset_index(drop=True)
        ranges, labels = {}, {}
        global_idx = 0
        for _, row in new_data.iterrows():
            path = str(row["path"])
            if path not in self.exp_embeddings:
                continue
            n = self.exp_embeddings[path].shape[0]
            labels[path] = int(row[col])
            ranges[path] = (global_idx, global_idx + n)
            global_idx += n

        child = object.__new__(EmbeddingDataModule)
        child.model_version = self.model_version
        child.cell_config = self.cell_config
        child.n_cell_classes = self.n_cell_classes
        child.cell_weights = self.cell_weights
        child.pool_cap = self.pool_cap
        child.uses_regionprops = self.uses_regionprops
        child.load_workers = self.load_workers
        child.database_column = col
        child.database_label = self.database_label
        child.task_type = self.task_type
        child.num_classes = self.num_classes
        child.batch_size = self.batch_size
        child.backbone_name = self.backbone_name
        child.train = is_train
        child.data = new_data
        child.exp_embeddings = self.exp_embeddings   # shared by reference — no re-read
        child.class_local = self.class_local         # local indices are split-invariant
        child.true_counts = self.true_counts
        child.embedding_dim = self.embedding_dim
        child.experiment_ranges = ranges
        child.experiment_labels = labels
        child._exp_id_by_path = {p: self._exp_id_by_path[p] for p in ranges}
        child.length = sum(e - s for s, e in ranges.values())
        child._null_label_paths = self._null_label_paths & set(new_data["path"].tolist())
        child._rebuild_index()
        child._set_task_attrs(new_data)
        child.num_unique_patients = new_data["sample_id"].nunique()
        child.num_experiments = len(ranges)
        child.unique_patient_ids = new_data["sample_id"].unique().tolist()
        child.augments = None
        child.dataloader = child
        return child
