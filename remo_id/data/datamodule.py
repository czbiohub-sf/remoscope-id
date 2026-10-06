"""Data module and loader for remo-id."""

from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import random
import os
import bisect
import warnings
from enum import Enum
from typing import List

from remo_id.data.manifest import (
    assign_manifest_classes,
    is_manifest_class_config,
    manifest_class_config,
)
from remo_id.data.zarr_io import get_image_from_zarr, get_zarr_file


def open_database():
    """Connect to the Remoscope database (private ``remoscope-database`` package).

    Only the database-backed selection paths (``--study_id``, ``--class_config``
    entries with ``study_id``) need it; manifests do not.
    """
    try:
        from remoscope_database.database.database_manager import DatabaseManager
    except ImportError as e:
        raise ImportError(
            "Selecting experiments by study_id/database column needs the "
            "remoscope-database package, which is not publicly available. "
            "Describe your data with a CSV manifest instead (--manifest, see "
            "remo_id/data/manifest.py)."
        ) from e
    return DatabaseManager()


class SamplingMode(Enum):
    """
    Sampling strategies for selecting images from experiments.

    RANDOM: Uniform random sampling (default)
    START: First N images from experiment
    MIDDLE: Middle N images
    END: Last N images
    DISTRIBUTED: N/3 from start, middle, end
    """
    RANDOM = "random"
    START = "start"
    MIDDLE = "middle"
    END = "end"
    DISTRIBUTED = "distributed"


def sample_indices(total_images: int, n_samples: int, mode: SamplingMode, seed: int = None) -> List[int]:
    """
    Sample image indices from an experiment based on sampling mode.

    Args:
        total_images: Total number of images in the experiment
        n_samples: Number of samples to select
        mode: Sampling strategy to use
        seed: Random seed for reproducible sampling (only used for RANDOM mode)

    Returns:
        List of indices to sample
    """
    if n_samples >= total_images:
        return list(range(total_images))

    if mode == SamplingMode.RANDOM:
        if seed is not None:
            rng = random.Random(seed)
            return rng.sample(range(total_images), n_samples)
        return random.sample(range(total_images), n_samples)

    elif mode == SamplingMode.START:
        return list(range(n_samples))

    elif mode == SamplingMode.MIDDLE:
        mid = total_images // 2
        start = mid - n_samples // 2
        return list(range(start, start + n_samples))

    elif mode == SamplingMode.END:
        return list(range(total_images - n_samples, total_images))

    elif mode == SamplingMode.DISTRIBUTED:
        n_each = n_samples // 3
        remainder = n_samples - (n_each * 3)

        start_idx = list(range(n_each + (1 if remainder > 0 else 0)))
        mid = total_images // 2
        mid_start = mid - n_each // 2
        mid_idx = list(range(mid_start, mid_start + n_each + (1 if remainder > 1 else 0)))
        end_start = total_images - n_each
        end_idx = list(range(end_start, total_images))

        return start_idx + mid_idx + end_idx

    else:
        raise ValueError(f"Unknown sampling mode: {mode}")


class RemoIDDataModule(Dataset):
    """
    PyTorch Dataset over Remoscope experiments (bags of microscopy frames).

    Experiments and labels come from a CSV manifest (``manifest=``, see
    :mod:`remo_id.data.manifest`), a class config (``class_definitions=``, which may
    itself reference manifests), or the Remoscope database (``study_id`` /
    ``database_column``; needs the private remoscope-database package).

    Supports binary and multi-class classification.
    """

    def __init__(self,
                 database_column: str = "class_label",
                 database_label: str | None = None,
                 label_contains: str | None = None,
                 study_id: int | None = None,
                 study_ids: list | None = None,
                 threshold: float | int| None = None,
                 threshold_mode: str = ">",
                 negative_study_ids: list | None = None,
                 negative_experiment_ids: list | None = None,
                 include_experiment_ids: list | None = None,
                 batch_size: int = 256,
                 backbone_name: str = 'vit-b-16',
                 task_type: str = 'binary',
                 num_classes: int = 2,
                 class_definitions: list | None = None,
                 skip_null_labels: bool = False,
                 train: bool = True,
                 exclude_filters: list | None = None,
                 include_filters: list | None = None,
                 default_class_idx: int | None = None,
                 background_correction: str = 'none',
                 background_value: int = None,
                 intensity_jitter: float = None,
                 noise_std: float = None,
                 blur_sigma: float = None,
                 gradient_blur_sigma: float = None,
                 min_image_index: int = 0,
                 crop_size: int = None,
                 erase_num: str = None,
                 erase_scale: str = None,
                 manifest: str | None = None,
                 label_column: str = "label",
                 classes: list | None = None,
    ):
        if manifest is not None:
            # A manifest == a class config whose entries reference it.
            from remo_id.data.class_config import class_config_from_dict
            cfg = class_config_from_dict(manifest_class_config(manifest, label_column, classes))
            class_definitions, task_type, num_classes = cfg.class_definitions, cfg.task_type, cfg.num_classes
            database_column = "class_label"
        self.label_column = label_column
        uses_db = not is_manifest_class_config(class_definitions)
        self.database = open_database() if uses_db else None
        self.crop_size = crop_size
        self.skip_null_labels = skip_null_labels
        self.min_image_index = min_image_index
        self.database_label = database_label
        self.label_contains = label_contains
        self.database_column = database_column.lower()  # Normalize to lowercase for database compatibility
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
        self.backbone_name = backbone_name
        self.train = train

        # Intensity augmentation params
        self.background_correction = background_correction
        self.background_value = background_value
        self.intensity_jitter = intensity_jitter
        self.noise_std = noise_std
        self.blur_sigma = blur_sigma
        self.gradient_blur_sigma = gradient_blur_sigma
        # Random-erasing overrides: "MIN,MAX" strings (None = backbone defaults)
        self.erase_num = erase_num
        self.erase_scale = erase_scale

        # Initialize null label tracking (populated by _prepare_data for binary tasks)
        self._null_label_paths = set()

        print("Preparing data...")
        self.data, self.experiment_ranges, self.experiment_labels = self._prepare_data(
            self.database,
            self.database_column,
            self.database_label,
            self.label_contains,
            self.study_id,
            self.study_ids,
            self.threshold,
            self.threshold_mode,
            self.negative_study_ids,
            self.negative_experiment_ids,
            self.include_experiment_ids,
            self.task_type,
            self.num_classes,
            self.class_definitions,
            self.skip_null_labels,
            self.exclude_filters,
            self.include_filters,
            self.default_class_idx,
        )

        # Set task-specific attributes
        if task_type == "binary":
            self.neg_length = len(self.data[self.data[self.database_column] == 0])
            self.pos_length = len(self.data[self.data[self.database_column] == 1])
            self.pos_ids = self.data[self.data[self.database_column] == 1]['experiment_id'].tolist()
            self.neg_ids = self.data[self.data[self.database_column] == 0]['experiment_id'].tolist()
            print(f"Total experiments: {len(self.experiment_ranges)} | Positive: {self.pos_length} | Negative: {self.neg_length} | Source: {'study' if self.study_id is not None else 'experiment'}")
        elif task_type == "multiclass":
            self.class_experiment_ids = {}
            for c in range(num_classes):
                self.class_experiment_ids[c] = self.data[self.data[self.database_column] == c]['experiment_id'].tolist()
            class_counts = {c: len(ids) for c, ids in self.class_experiment_ids.items()}
            print(f"Total experiments: {len(self.experiment_ranges)} | Class counts: {class_counts} | Source: {'study' if self.study_id is not None else 'experiment'}")
            # Check for class imbalance warnings
            min_samples = 50
            for c, count in class_counts.items():
                if count < min_samples:
                    print(f"WARNING: Class {c} has only {count} experiments - consider merging or collecting more data")

        self.length = sum(e-s for s, e in self.experiment_ranges.values())
        self.dataloader = RemoIDDataLoader(
            self.data, database_column, batch_size, self.experiment_ranges,
            backbone_name=self.backbone_name, train=self.train,
            background_correction=self.background_correction,
            background_value=self.background_value, intensity_jitter=self.intensity_jitter,
            noise_std=self.noise_std, blur_sigma=self.blur_sigma,
            gradient_blur_sigma=self.gradient_blur_sigma,
            min_image_index=self.min_image_index,
            crop_size=self.crop_size,
            erase_num=self.erase_num, erase_scale=self.erase_scale,
        )

    def __len__(self):
        return self.length

    @property
    def null_label_paths(self) -> set:
        """Return set of experiment paths that originally had null labels."""
        return self._null_label_paths

    def has_null_label(self, path: str) -> bool:
        """Check if an experiment path originally had a null label."""
        return path in self._null_label_paths

    def _finalize_ranges(self, data, column, task_type):
        """Validate each experiment's zarr and build global image-index ranges.

        Returns ``(data, experiment_id_ranges, experiment_labels)`` with bad/unreadable
        experiments dropped. Shared tail of ``_prepare_data`` and
        ``_prepare_data_class_config``. (EmbeddingDataModule overrides this to build
        cell-index ranges over the embedding store instead.)
        """
        experiment_id_ranges = {}
        experiment_labels = {}
        global_idx = 0
        bad_experiments = []

        for _, row in data.iterrows():
            exp_path = row['path']
            if not os.path.exists(exp_path):
                print(exp_path)
                bad_experiments.append(exp_path)
                continue

            try:
                get_image_from_zarr(exp_path, 0)
            except Exception as e:
                print(f"Error reading zarr file at {exp_path}: {e}")
                bad_experiments.append(exp_path)
                continue

            num_images = row['num_images']
            effective_start = min(self.min_image_index, num_images)
            usable_images = num_images - effective_start
            if usable_images <= 0:
                print(f"Warning: Experiment {exp_path} has {num_images} images, fewer than min_image_index={self.min_image_index}. Excluding.")
                bad_experiments.append(exp_path)
                continue
            experiment_labels[exp_path] = int(row[column])
            experiment_id_ranges[exp_path] = (global_idx, global_idx + usable_images)
            global_idx += usable_images

        data = data[~data['path'].isin(bad_experiments)].reset_index(drop=True)
        print(f"Removed {len(bad_experiments)} experiments due to missing/invalid zarr files.")
        if self.min_image_index > 0:
            print(f"min_image_index={self.min_image_index}: skipping first {self.min_image_index} images per experiment")
        print(f"Final dataset size: {len(data)} experiments.")
        return data, experiment_id_ranges, experiment_labels

    def _prepare_data(
        self,
        database,
        column: str,
        label,
        label_contains: str | None = None,
        study_id: int | None = None,
        study_ids: list[str] | None = None,
        threshold: float | int | None = None,
        threshold_mode: str = ">",
        negative_study_ids: list[str] | None = None,
        negative_experiment_ids: list[str] | None = None,
        include_experiment_ids: list[str] | None = None,
        task_type: str = "binary",
        num_classes: int = 2,
        class_definitions: list | None = None,
        skip_null_labels: bool = False,
        exclude_filters: list | None = None,
        include_filters: list | None = None,
        default_class_idx: int | None = None,
    ):
        negative_study_ids = negative_study_ids or []
        negative_experiment_ids = negative_experiment_ids or []
        include_experiment_ids = include_experiment_ids or []
        study_ids = study_ids or []
        exclude_filters = exclude_filters or []
        include_filters = include_filters or []

        # ======== Manifest paths (no database) ========
        if is_manifest_class_config(class_definitions):
            return self._prepare_data_manifest(
                column, class_definitions, exclude_filters, include_filters,
                default_class_idx, task_type)

        experiment_table = database.experiment.get()

        # ======== Class config path (multi-study multiclass) ========
        if class_definitions and isinstance(class_definitions, list) and len(class_definitions) > 0 and isinstance(class_definitions[0], dict):
            return self._prepare_data_class_config(
                database, experiment_table, column, class_definitions,
                negative_study_ids, exclude_filters, include_filters,
                default_class_idx, task_type, num_classes, skip_null_labels,
            )

        # -------- Step 1: Gather all experiment IDs from various sources --------
        all_exp_ids = set()
        forced_negative_ids = set()  # IDs from negative_study_ids/negative_experiment_ids

        # Track which study tables have the column (for multi-study lookup)
        study_tables_with_column = {}  # study_id -> table
        study_exp_id_to_study = {}  # exp_id -> study_id (for label lookup)

        # From study table (main source - single study_id)
        study_table = None
        if study_id is not None:
            study_table = database.get_table_from_string(study_id).get()
            if column not in study_table.columns:
                raise ValueError(
                    f"study_id was provided ({study_id}) so column='{column}' must be in the study table. "
                    f"Study columns: {list(study_table.columns)}"
                )
            study_exp_df = study_table.select(["experiment_id"]).execute()
            all_exp_ids.update(str(eid) for eid in study_exp_df["experiment_id"])
        elif study_ids:
            # Multiple study_ids - check each for the column, fall back to experiment table
            for sid in study_ids:
                sid_table = database.get_table_from_string(sid).get()
                sid_exp_df = sid_table.select(["experiment_id"]).execute()
                for eid in sid_exp_df["experiment_id"]:
                    all_exp_ids.add(str(eid))
                    study_exp_id_to_study[str(eid)] = sid
                # Check if this study table has the column
                if column in sid_table.columns:
                    study_tables_with_column[sid] = sid_table
                    print(f"Study '{sid}' has column '{column}'")
                else:
                    print(f"Study '{sid}' does not have column '{column}', will use experiment table")
            # Verify column exists somewhere
            if not study_tables_with_column and column not in experiment_table.columns:
                raise ValueError(
                    f"Column '{column}' not found in any of the study tables ({study_ids}) "
                    f"or in the experiment table. Experiment columns: {list(experiment_table.columns)}"
                )
        else:
            if column not in experiment_table.columns:
                raise ValueError(
                    f"No study_id provided, so column='{column}' must be in the experiment table. "
                    f"Experiment columns: {list(experiment_table.columns)}"
                )

        # From include_experiment_ids
        all_exp_ids.update(str(eid) for eid in include_experiment_ids)

        # From negative_study_ids (mark as forced negative)
        for neg_study_id in negative_study_ids:
            neg_tbl = database.get_table_from_string(neg_study_id).get()
            if 'experiment_id' not in neg_tbl.columns:
                raise ValueError(f"Negative study {neg_study_id} has no 'experiment_id' column.")
            neg_ids = neg_tbl.select(["experiment_id"]).execute()["experiment_id"]
            for eid in neg_ids:
                forced_negative_ids.add(str(eid))
                all_exp_ids.add(str(eid))

        # From negative_experiment_ids (mark as forced negative)
        for neg_eid in negative_experiment_ids:
            forced_negative_ids.add(str(neg_eid))
            all_exp_ids.add(str(neg_eid))

        # -------- Step 2: Get base experiment info from experiment table --------
        exp_df = (
            experiment_table
            .select(['experiment_id', 'study_id', 'site', 'path', 'num_images'])
            .filter(experiment_table['experiment_id'].isin(list(all_exp_ids)))
            .execute()
        )
        exp_df['experiment_id'] = exp_df['experiment_id'].astype(str)

        # Look up sample_id from study tables (it lives there, not in experiment table)
        exp_df['sample_id'] = None
        # Collect all study tables we need to check
        study_tables_to_check = {}
        if study_id is not None and study_table is not None:
            study_tables_to_check[study_id] = study_table
        for sid in study_ids:
            if sid not in study_tables_to_check:
                study_tables_to_check[sid] = database.get_table_from_string(sid).get()
        for neg_sid in (negative_study_ids or []):
            if neg_sid not in study_tables_to_check:
                study_tables_to_check[neg_sid] = database.get_table_from_string(neg_sid).get()
        for sid, stbl in study_tables_to_check.items():
            stbl_col_names = [c.get_name() if hasattr(c, 'get_name') else str(c)
                              for c in stbl.columns]
            if 'sample_id' not in stbl_col_names:
                continue
            sid_sample_df = stbl.select(['experiment_id', 'sample_id']).execute()
            sid_sample_df['experiment_id'] = sid_sample_df['experiment_id'].astype(str)
            sample_map = dict(zip(sid_sample_df['experiment_id'], sid_sample_df['sample_id']))
            for idx in exp_df.index:
                eid = exp_df.at[idx, 'experiment_id']
                if eid in sample_map and exp_df.at[idx, 'sample_id'] is None:
                    exp_df.at[idx, 'sample_id'] = sample_map[eid]

        # -------- Step 3: Look up labels (per-experiment) --------
        # Initialize label column to None
        exp_df[column] = None
        self._null_label_paths = set()  # Track nulls ONLY from study lookup

        if study_id is not None:
            # Get labels from study table
            study_label_df = study_table.select(["experiment_id", column]).execute()
            study_label_df['experiment_id'] = study_label_df['experiment_id'].astype(str)
            label_map = dict(zip(study_label_df['experiment_id'], study_label_df[column]))

            for idx in exp_df.index:
                eid = str(exp_df.at[idx, 'experiment_id'])
                exp_path = exp_df.at[idx, 'path']

                if eid in forced_negative_ids:
                    # Forced negative - explicitly NOT null
                    # Label will be set to 0 during binary label conversion below
                    exp_df.at[idx, column] = "__FORCED_NEGATIVE__"
                elif eid in label_map:
                    val = label_map[eid]
                    if pd.isna(val):
                        self._null_label_paths.add(exp_path)
                    exp_df.at[idx, column] = val
                else:
                    # Not in study table (e.g., from include_experiment_ids) - treat as null
                    self._null_label_paths.add(exp_path)
        elif study_ids:
            # Multiple study_ids - check study tables first, fall back to experiment table
            # Build label maps for each study that has the column
            study_label_maps = {}
            for sid, sid_table in study_tables_with_column.items():
                sid_label_df = sid_table.select(["experiment_id", column]).execute()
                sid_label_df['experiment_id'] = sid_label_df['experiment_id'].astype(str)
                study_label_maps[sid] = dict(zip(sid_label_df['experiment_id'], sid_label_df[column]))

            # Get experiment table labels as fallback
            exp_label_map = {}
            if column in experiment_table.columns:
                exp_with_labels = (
                    experiment_table
                    .select(['experiment_id', column])
                    .filter(experiment_table['experiment_id'].isin(list(all_exp_ids)))
                    .execute()
                )
                exp_with_labels['experiment_id'] = exp_with_labels['experiment_id'].astype(str)
                exp_label_map = dict(zip(exp_with_labels['experiment_id'], exp_with_labels[column]))

            for idx in exp_df.index:
                eid = str(exp_df.at[idx, 'experiment_id'])
                exp_path = exp_df.at[idx, 'path']

                if eid in forced_negative_ids:
                    exp_df.at[idx, column] = "__FORCED_NEGATIVE__"
                else:
                    # Try to get label from the experiment's study table first
                    exp_study = study_exp_id_to_study.get(eid)
                    val = None
                    if exp_study and exp_study in study_label_maps:
                        val = study_label_maps[exp_study].get(eid)
                    # Fall back to experiment table
                    if val is None and eid in exp_label_map:
                        val = exp_label_map[eid]

                    if val is None or pd.isna(val):
                        self._null_label_paths.add(exp_path)
                    exp_df.at[idx, column] = val
        else:
            # No study_id - get labels from experiment table directly
            exp_with_labels = (
                experiment_table
                .select(['experiment_id', column])
                .filter(experiment_table['experiment_id'].isin(list(all_exp_ids)))
                .execute()
            )
            exp_with_labels['experiment_id'] = exp_with_labels['experiment_id'].astype(str)
            label_map = dict(zip(exp_with_labels['experiment_id'], exp_with_labels[column]))

            for idx in exp_df.index:
                eid = str(exp_df.at[idx, 'experiment_id'])
                exp_path = exp_df.at[idx, 'path']

                if eid in forced_negative_ids:
                    exp_df.at[idx, column] = "__FORCED_NEGATIVE__"
                elif eid in label_map:
                    val = label_map[eid]
                    if pd.isna(val):
                        self._null_label_paths.add(exp_path)
                    exp_df.at[idx, column] = val
                else:
                    self._null_label_paths.add(exp_path)

        data = exp_df.copy()

        # -------- Step 4: Handle skip_null_labels --------
        if skip_null_labels:
            null_mask = data['path'].isin(self._null_label_paths)
            null_count = null_mask.sum()
            if null_count > 0:
                print(f"Skipping {null_count} experiments with null values in '{column}'")
                data = data[~null_mask].reset_index(drop=True)

        # -------- Label main data based on task type --------
        if task_type == "multiclass":
            # Handle forced negatives - in multiclass, they become class 0
            forced_neg_mask = data[column] == "__FORCED_NEGATIVE__"

            # Preserve categorical values as class indices
            if class_definitions is not None:
                # Use class definitions to assign class indices
                # class_definitions format: ['study_id:column:value', 'study_id:column:value', 'study_id']
                data[column] = -1  # Default to invalid class
                for class_idx, class_def in enumerate(class_definitions):
                    parts = class_def.split(':')
                    if len(parts) == 1:
                        # Just study_id - all experiments from this study get this class
                        # Already handled via negative_study_ids mechanism
                        pass
                    elif len(parts) == 3:
                        # study_id:column:value format
                        def_study, def_col, def_val = parts
                        # This would require the column to match - for now, use simple approach
                        pass
                # Forced negatives get class 0
                data.loc[forced_neg_mask, column] = 0
                print(f"Multiclass: Using {len(class_definitions)} class definitions")
            else:
                # Auto-detect classes from unique values (excluding __FORCED_NEGATIVE__)
                non_forced = data[~forced_neg_mask][column]
                unique_vals = sorted(non_forced.dropna().unique())
                val_to_idx = {v: i for i, v in enumerate(unique_vals)}
                self._class_mapping = val_to_idx
                self._idx_to_class = {i: v for v, i in val_to_idx.items()}
                data[column] = data[column].map(val_to_idx)
                # Forced negatives get class 0
                data.loc[forced_neg_mask, column] = 0
                print(f"Multiclass: Auto-detected {len(unique_vals)} classes: {val_to_idx}")

            # Drop rows with invalid class
            data = data[data[column] >= 0].reset_index(drop=True)

        else:  # binary (default)
            # Null labels were already tracked during Step 3 (label lookup)
            if len(self._null_label_paths) > 0:
                print(f"Binary: Found {len(self._null_label_paths)} experiments with null labels (will be tracked)")

            # Handle forced negatives first - these are marked with __FORCED_NEGATIVE__
            forced_neg_mask = data[column] == "__FORCED_NEGATIVE__"

            if threshold is not None:
                # For threshold-based labels, forced negatives get 0
                col_vals = pd.to_numeric(data[column].where(~forced_neg_mask), errors="coerce")
                if threshold_mode.lower() in ("gt", ">", "above"):
                    data[column] = (col_vals > float(threshold)).astype(float)
                elif threshold_mode.lower() in ("lt", "<", "below"):
                    data[column] = (col_vals < float(threshold)).astype(float)
                elif threshold_mode.lower() in ("eq", "==", "equal"):
                    data[column] = (col_vals == float(threshold)).astype(float)
                else:
                    raise ValueError("threshold_mode must be one of: '>'/'gt'/'above' or '<'/'lt'/'below' or '=='/'eq'/'equal'")
                # Forced negatives and NaNs get 0
                data.loc[forced_neg_mask, column] = 0
                data[column] = data[column].fillna(0).astype(int)
            else:
                # For categorical labels, forced negatives get 0
                if label_contains is not None:
                    # Fuzzy match: positive if column value contains the string (case-insensitive)
                    target_lower = label_contains.lower()
                    vals = data[column].where(~forced_neg_mask).astype(str).str.strip().str.lower()
                    data[column] = vals.str.contains(target_lower, na=False).astype(int)
                    print(f"Binary: Using fuzzy match (contains '{label_contains}')")
                elif isinstance(label, bool):
                    vals = data[column].where(~forced_neg_mask).map(
                        lambda x: str(x).strip().lower() in {"1", "true", "t", "yes", "y"} if pd.notna(x) else False
                    )
                    target = label
                    data[column] = (vals == target).astype(int)
                else:
                    vals = data[column].where(~forced_neg_mask).astype(str).str.strip().str.lower()
                    target = str(label).strip().lower()
                    data[column] = (vals == target).astype(int)
                # Forced negatives explicitly set to 0
                data.loc[forced_neg_mask, column] = 0

        # -------- Build ranges / validate zarr --------
        return self._finalize_ranges(data, column, task_type)

    def _prepare_data_manifest(self, column, class_definitions, exclude_filters,
                               include_filters, default_class_idx, task_type):
        """Labels from manifest-backed class definitions (see remo_id.data.manifest)."""
        data, class_names, unassigned = assign_manifest_classes(
            class_definitions, exclude_filters=exclude_filters,
            include_filters=include_filters, default_class_idx=default_class_idx,
        )
        data = data.rename(columns={"class_label": column})
        if len(class_names) > 1:
            self._class_mapping = {name: idx for idx, name in enumerate(class_names)}
            self._idx_to_class = dict(enumerate(class_names))
        self._null_label_paths = set(unassigned)
        if unassigned:
            print(f"Manifest: skipping {len(unassigned)} experiments whose label is empty "
                  f"or not one of {class_names}")
        counts = data[column].value_counts().to_dict()
        print(f"Manifest: {len(data)} experiments | "
              + ", ".join(f"'{n}': {counts.get(i, 0)}" for i, n in enumerate(class_names)))
        return self._finalize_ranges(data, column, task_type)

    def _prepare_data_class_config(
        self,
        database,
        experiment_table,
        column: str,
        class_definitions: list[dict],
        negative_study_ids: list[str],
        exclude_filters: list[dict],
        include_filters: list[dict],
        default_class_idx: int | None,
        task_type: str,
        num_classes: int,
        skip_null_labels: bool,
    ):
        """Prepare data using YAML class config (multi-study multiclass).

        Each class_definitions entry is a dict with: name, study_id/experiment_ids, and optional filters.
        """
        # Experiment selection (which experiment → which class) lives in the
        # remoscope-database package (shared with other consumers of the database).
        from remoscope_database.class_config import resolve_class_assignment
        assignment = resolve_class_assignment(
            database,
            class_definitions,
            negative_study_ids=negative_study_ids,
            exclude_filters=exclude_filters,
            include_filters=include_filters,
            default_class_idx=default_class_idx,
        )
        class_assignment = assignment["class_assignment"]
        class_names = assignment["class_names"]
        study_ids_seen = assignment["study_ids_seen"]
        all_exp_ids = set(class_assignment.keys())

        # Store class mapping for later use
        if len(class_names) > 1:
            self._class_mapping = {name: idx for idx, name in enumerate(class_names)}
            self._idx_to_class = dict(enumerate(class_names))

        # -------- Step 2: Get base experiment info from experiment table --------
        exp_df = (
            experiment_table
            .select(['experiment_id', 'study_id', 'site', 'path', 'num_images'])
            .filter(experiment_table['experiment_id'].isin(list(all_exp_ids)))
            .execute()
        )
        exp_df['experiment_id'] = exp_df['experiment_id'].astype(str)

        # Look up sample_id from study tables
        exp_df['sample_id'] = None
        for sid in study_ids_seen:
            stbl = database.get_table_from_string(sid).get()
            stbl_col_names = [c.get_name() if hasattr(c, 'get_name') else str(c)
                              for c in stbl.columns]
            if 'sample_id' not in stbl_col_names:
                continue
            sid_sample_df = stbl.select(['experiment_id', 'sample_id']).execute()
            sid_sample_df['experiment_id'] = sid_sample_df['experiment_id'].astype(str)
            sample_map = dict(zip(sid_sample_df['experiment_id'], sid_sample_df['sample_id']))
            for idx in exp_df.index:
                eid = exp_df.at[idx, 'experiment_id']
                if eid in sample_map and exp_df.at[idx, 'sample_id'] is None:
                    exp_df.at[idx, 'sample_id'] = sample_map[eid]

        # -------- Step 3: Set labels directly from class_assignment --------
        exp_df[column] = exp_df['experiment_id'].map(class_assignment)

        # Track null labels (experiments that ended up without assignment)
        self._null_label_paths = set()
        null_mask = exp_df[column].isna()
        if null_mask.any():
            self._null_label_paths = set(exp_df.loc[null_mask, 'path'])
            if skip_null_labels:
                print(f"Skipping {null_mask.sum()} experiments with no class assignment")
                exp_df = exp_df[~null_mask].reset_index(drop=True)

        data = exp_df.copy()

        # Ensure integer labels
        data[column] = data[column].astype(int)

        print(f"Class config: Final dataset: {len(data)} experiments")
        for idx, name in enumerate(class_names):
            if len(class_names) == 1:
                # Binary: show pos/neg
                count_pos = len(data[data[column] == 1])
                count_neg = len(data[data[column] == 0])
                print(f"  '{name}' positive: {count_pos}, negative: {count_neg}")
                break
            else:
                count = len(data[data[column] == idx])
                print(f"  Class {idx} '{name}': {count} experiments")

        # -------- Build ranges / validate zarr --------
        return self._finalize_ranges(data, column, task_type)

    def __getitem__(self,idx):
        return self.dataloader.get_item(idx)

    def test_collate_fn(self, batch):
        x = torch.stack([b[0] for b in batch], dim=0)
        # Use appropriate dtype based on task type
        task_type = getattr(self, 'task_type', 'binary')
        if task_type == "multiclass":
            y = torch.tensor([b[1] for b in batch], dtype=torch.long)
        else:  # binary
            y = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        exp_ids = torch.tensor([b[2] for b in batch], dtype=torch.long)
        return x, y, exp_ids

    def collate_fn(self, batch):
        return self.dataloader.get_items(batch)

    def _decide_train_val_paths(self, val_ratio=0.2, seed=None, split_by_sample_id=True):
        """Decide which experiment paths go to train vs val (sample_id-stratified).

        Pure decision logic — no range/loader construction — so it can be shared by the
        image pipeline and the embedding pipeline. Returns ``(train_paths, val_paths)`` sets.
        """
        rng = random.Random(seed) if seed is not None else random.Random()

        # --- Step 1: Group experiments by sample_id ---
        sample_to_experiments = {}
        null_sample_count = 0
        for _, row in self.data.iterrows():
            exp_path = row['path']
            if split_by_sample_id:
                sample_id = row['sample_id']
                if pd.isna(sample_id) or sample_id is None:
                    sample_id = f"__unique_exp_{exp_path}"
                    null_sample_count += 1
            else:
                sample_id = f"__unique_exp_{exp_path}"
            sample_to_experiments.setdefault(sample_id, []).append(exp_path)

        if not split_by_sample_id:
            print("  Note: split_by_sample_id=False, each experiment treated independently")
        elif null_sample_count > 0:
            print(f"  Note: {null_sample_count} experiments have null sample_id (treated as unique patients)")

        # --- Step 2: Determine each sample's label (for stratification) ---
        # Use the majority label across all experiments in the sample
        sample_labels = {}
        for sample_id, exp_paths in sample_to_experiments.items():
            labels = [self.experiment_labels[p] for p in exp_paths if p in self.experiment_labels]
            if not labels:
                continue
            if self.task_type == "binary":
                # Majority vote (positive if any positive)
                sample_labels[sample_id] = 1 if any(l == 1 for l in labels) else 0
            else:  # multiclass
                from collections import Counter
                sample_labels[sample_id] = Counter(int(l) for l in labels).most_common(1)[0][0]

        # --- Step 3: Stratified split at sample level ---
        val_samples = set()
        train_samples = set()

        # Group samples by their label, then split each group proportionally
        label_to_samples = {}
        for sid, lbl in sample_labels.items():
            label_to_samples.setdefault(lbl, []).append(sid)

        for lbl, samples in label_to_samples.items():
            rng.shuffle(samples)
            n_val = max(1, int(len(samples) * val_ratio))  # At least 1 per class in val
            # But don't take all samples for val - leave at least 1 for train
            n_val = min(n_val, len(samples) - 1) if len(samples) > 1 else 0
            val_samples.update(samples[:n_val])
            train_samples.update(samples[n_val:])

        # If any class is completely missing from val (can happen with very small classes),
        # swap entire sample groups (not individual experiments) to fix it
        def _check_class_coverage(split_samples):
            classes = set()
            for sid in split_samples:
                for exp_path in sample_to_experiments[sid]:
                    if exp_path in self.experiment_labels:
                        classes.add(int(self.experiment_labels[exp_path]))
            return classes

        if self.task_type == "binary":
            needed = {0, 1}
        else:
            needed = set(range(self.num_classes))

        train_classes = _check_class_coverage(train_samples)
        val_classes = _check_class_coverage(val_samples)

        for missing_class in needed - val_classes:
            # Find a sample with this class in train and move the whole sample group
            for sid in list(train_samples):
                if sample_labels.get(sid) == missing_class and len(train_samples) > 1:
                    train_samples.discard(sid)
                    val_samples.add(sid)
                    break

        for missing_class in needed - train_classes:
            for sid in list(val_samples):
                if sample_labels.get(sid) == missing_class and len(val_samples) > 1:
                    val_samples.discard(sid)
                    train_samples.add(sid)
                    break

        # --- Step 4: Expand samples back to experiments ---
        val_experiments = set()
        train_experiments = set()
        for sid in val_samples:
            val_experiments.update(sample_to_experiments[sid])
        for sid in train_samples:
            train_experiments.update(sample_to_experiments[sid])

        print(f"Split by sample_id (patient-level, stratified) to prevent data leakage:")
        print(f"  Train: {len(train_samples)} unique patients, {len(train_experiments)} experiments")
        print(f"  Val:   {len(val_samples)} unique patients, {len(val_experiments)} experiments")

        return train_experiments, val_experiments

    def split_train_val(self, val_ratio=0.2, seed=None, split_by_sample_id=True):
        """Split data into train and validation sets.

        Uses stratified splitting by sample_id (patient) to prevent data leakage
        AND ensure both splits have representative class distributions.

        Experiments sharing a sample_id always stay in the same split.
        Experiments with null sample_id are treated as independent samples.

        Args:
            val_ratio: Fraction of samples to use for validation
            seed: Random seed for reproducible splits. If None, uses current random state.
            split_by_sample_id: If True, group by sample_id to prevent leakage. If False, treat each experiment independently.
        """
        train_experiments, val_experiments = self._decide_train_val_paths(
            val_ratio, seed, split_by_sample_id)

        min_idx = getattr(self, 'min_image_index', 0)

        def filter_experiments(exps):
            new_data = self.data[self.data['path'].isin(exps)].reset_index(drop=True)
            new_ranges = {}
            new_labels = {}
            global_idx = 0
            for _, row in new_data.iterrows():
                exp_path = row['path']
                num_images = row['num_images']
                usable_images = num_images - min(min_idx, num_images)
                if usable_images <= 0:
                    continue
                new_labels[exp_path] = int(row[self.database_column])
                new_ranges[exp_path] = (global_idx, global_idx + usable_images)
                global_idx += usable_images
            return new_data, new_ranges, new_labels

        train_data, train_ranges, train_labels = filter_experiments(train_experiments)
        val_data, val_ranges, val_labels = filter_experiments(val_experiments)

        # Create uninitialized instances
        train_module = object.__new__(RemoIDDataModule)
        val_module = object.__new__(RemoIDDataModule)

        # Manually assign attributes (avoid constructor)
        for module, data, ranges, labels, is_train in [
            (train_module, train_data, train_ranges, train_labels, True),
            (val_module, val_data, val_ranges, val_labels, False),
        ]:
            module.database_column = self.database_column
            module.database_label = self.database_label
            module.batch_size = self.batch_size
            module.data = data
            module.backbone_name = self.backbone_name
            module.experiment_ranges = ranges
            module.experiment_labels = labels
            module.task_type = self.task_type
            module.num_classes = self.num_classes
            module.length = sum(e - s for s, e in ranges.values())
            # Train split gets all augments; val split gets only normalization
            if is_train:
                dl_augment_kwargs = dict(
                    background_correction=self.background_correction,
                    background_value=self.background_value,
                    intensity_jitter=self.intensity_jitter,
                    noise_std=self.noise_std,
                    blur_sigma=self.blur_sigma,
                    gradient_blur_sigma=self.gradient_blur_sigma,
                    erase_num=getattr(self, 'erase_num', None),
                    erase_scale=getattr(self, 'erase_scale', None),
                )
            else:
                dl_augment_kwargs = dict(
                    background_correction=self.background_correction,
                    background_value=self.background_value,
                )
            module.dataloader = RemoIDDataLoader(
                data, self.database_column, self.batch_size, ranges,
                backbone_name=self.backbone_name, train=is_train,
                min_image_index=min_idx,
                crop_size=getattr(self, 'crop_size', None),
                **dl_augment_kwargs,
            )

            # Copy augmentation params for downstream use
            module.background_correction = self.background_correction
            module.background_value = self.background_value
            module.intensity_jitter = self.intensity_jitter
            module.noise_std = self.noise_std
            module.blur_sigma = self.blur_sigma
            module.gradient_blur_sigma = self.gradient_blur_sigma
            module.erase_num = getattr(self, 'erase_num', None)
            module.erase_scale = getattr(self, 'erase_scale', None)
            module.min_image_index = min_idx

            # Set task-specific attributes
            if self.task_type == "binary":
                module.pos_ids = data[data[self.database_column] == 1]['experiment_id'].tolist()
                module.neg_ids = data[data[self.database_column] == 0]['experiment_id'].tolist()
            elif self.task_type == "multiclass":
                module.class_experiment_ids = {}
                for c in range(self.num_classes):
                    module.class_experiment_ids[c] = data[data[self.database_column] == c]['experiment_id'].tolist()
                module.pos_ids = []
                module.neg_ids = []

            # Copy null label tracking - filter to only experiments in this split
            module._null_label_paths = self._null_label_paths & set(data['path'].tolist())

        # Store patient-level split statistics for logging
        # Calculate unique sample_ids from each split's data
        train_module.num_unique_patients = train_data['sample_id'].nunique()
        train_module.num_experiments = len(train_experiments)
        train_module.unique_patient_ids = train_data['sample_id'].unique().tolist()

        val_module.num_unique_patients = val_data['sample_id'].nunique()
        val_module.num_experiments = len(val_experiments)
        val_module.unique_patient_ids = val_data['sample_id'].unique().tolist()

        return train_module, val_module

    def split_from_paths(self, train_paths: set, val_paths: set):
        """Split data using externally-provided path sets (e.g. from k-fold CV).

        Same logic as the tail of split_train_val() but skips the stratified
        splitting step — the caller provides which experiments go where.

        Args:
            train_paths: Set of experiment paths for training
            val_paths: Set of experiment paths for validation

        Returns:
            (train_module, val_module) pair of RemoIDDataModule instances
        """
        min_idx = getattr(self, 'min_image_index', 0)

        def filter_experiments(exps):
            new_data = self.data[self.data['path'].isin(exps)].reset_index(drop=True)
            new_ranges = {}
            new_labels = {}
            global_idx = 0
            for _, row in new_data.iterrows():
                exp_path = row['path']
                num_images = row['num_images']
                usable_images = num_images - min(min_idx, num_images)
                if usable_images <= 0:
                    continue
                new_labels[exp_path] = int(row[self.database_column])
                new_ranges[exp_path] = (global_idx, global_idx + usable_images)
                global_idx += usable_images
            return new_data, new_ranges, new_labels

        train_data, train_ranges, train_labels = filter_experiments(train_paths)
        val_data, val_ranges, val_labels = filter_experiments(val_paths)

        train_module = object.__new__(RemoIDDataModule)
        val_module = object.__new__(RemoIDDataModule)

        for module, data, ranges, labels, is_train in [
            (train_module, train_data, train_ranges, train_labels, True),
            (val_module, val_data, val_ranges, val_labels, False),
        ]:
            module.database_column = self.database_column
            module.database_label = self.database_label
            module.batch_size = self.batch_size
            module.data = data
            module.backbone_name = self.backbone_name
            module.experiment_ranges = ranges
            module.experiment_labels = labels
            module.task_type = self.task_type
            module.num_classes = self.num_classes
            module.length = sum(e - s for s, e in ranges.values())

            # Train split gets all augments; val split gets only normalization
            if is_train:
                dl_augment_kwargs = dict(
                    background_correction=getattr(self, 'background_correction', 'none'),
                    background_value=getattr(self, 'background_value', None),
                    intensity_jitter=getattr(self, 'intensity_jitter', None),
                    noise_std=getattr(self, 'noise_std', None),
                    blur_sigma=getattr(self, 'blur_sigma', None),
                    gradient_blur_sigma=getattr(self, 'gradient_blur_sigma', None),
                    erase_num=getattr(self, 'erase_num', None),
                    erase_scale=getattr(self, 'erase_scale', None),
                )
            else:
                dl_augment_kwargs = dict(
                    background_correction=getattr(self, 'background_correction', 'none'),
                    background_value=getattr(self, 'background_value', None),
                )
            module.dataloader = RemoIDDataLoader(
                data, self.database_column, self.batch_size, ranges,
                backbone_name=self.backbone_name, train=is_train,
                min_image_index=min_idx,
                crop_size=getattr(self, 'crop_size', None),
                **dl_augment_kwargs,
            )

            # Copy augmentation params
            module.background_correction = getattr(self, 'background_correction', 'none')
            module.background_value = getattr(self, 'background_value', None)
            module.intensity_jitter = getattr(self, 'intensity_jitter', None)
            module.noise_std = getattr(self, 'noise_std', None)
            module.blur_sigma = getattr(self, 'blur_sigma', None)
            module.gradient_blur_sigma = getattr(self, 'gradient_blur_sigma', None)
            module.erase_num = getattr(self, 'erase_num', None)
            module.erase_scale = getattr(self, 'erase_scale', None)
            module.min_image_index = min_idx

            if self.task_type == "binary":
                module.pos_ids = data[data[self.database_column] == 1]['experiment_id'].tolist()
                module.neg_ids = data[data[self.database_column] == 0]['experiment_id'].tolist()
            elif self.task_type == "multiclass":
                module.class_experiment_ids = {}
                for c in range(self.num_classes):
                    module.class_experiment_ids[c] = data[data[self.database_column] == c]['experiment_id'].tolist()
                module.pos_ids = []
                module.neg_ids = []

            module._null_label_paths = self._null_label_paths & set(data['path'].tolist())

        train_module.num_unique_patients = train_data['sample_id'].nunique()
        train_module.num_experiments = len(train_paths)
        train_module.unique_patient_ids = train_data['sample_id'].unique().tolist()

        val_module.num_unique_patients = val_data['sample_id'].nunique()
        val_module.num_experiments = len(val_paths)
        val_module.unique_patient_ids = val_data['sample_id'].unique().tolist()

        return train_module, val_module


class MultiRandomErasing(torch.nn.Module):
    """Apply multiple small random erasures to simulate heavy masking."""

    def __init__(self, num_erases: tuple = (20, 50), scale: tuple = (0.001, 0.005),
                 ratio: tuple = (0.5, 2.0), value: float = 0.5):
        super().__init__()
        self.num_erases = num_erases  # (min, max) number of erasures
        self.scale = scale            # (min, max) fraction of image area per mask
        self.ratio = ratio            # aspect ratio range
        self.value = value            # fill value

    def forward(self, img):
        # Handle both single images [C, H, W] and batches [B, C, H, W]
        if img.dim() == 4:
            # Batch mode: apply erasing to each image independently
            for i in range(img.shape[0]):
                img[i] = self._erase_single(img[i])
            return img
        else:
            return self._erase_single(img)

    def _erase_single(self, img):
        # img shape: [C, H, W]
        _, h, w = img.shape
        img_area = h * w

        num = torch.randint(self.num_erases[0], self.num_erases[1] + 1, (1,)).item()

        for _ in range(num):
            area = img_area * torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
            aspect = torch.empty(1).uniform_(self.ratio[0], self.ratio[1]).item()

            eh = int(round((area * aspect) ** 0.5))
            ew = int(round((area / aspect) ** 0.5))

            if eh < h and ew < w:
                top = torch.randint(0, h - eh, (1,)).item()
                left = torch.randint(0, w - ew, (1,)).item()
                img[:, top:top+eh, left:left+ew] = self.value

        return img


class GaussianNoise(torch.nn.Module):
    """Add random Gaussian noise with std sampled from [0, max_std]."""

    def __init__(self, max_std: float = 0.05, p: float = 0.5):
        super().__init__()
        self.max_std = max_std
        self.p = p

    def forward(self, img):
        if img.dim() == 4:
            for i in range(img.shape[0]):
                img[i] = self._apply_single(img[i])
            return img
        return self._apply_single(img)

    def _apply_single(self, img):
        if torch.rand(1).item() > self.p:
            return img
        std = torch.empty(1).uniform_(0, self.max_std).item()
        noise = torch.randn_like(img) * std
        return (img + noise).clamp(0.0, 1.0)


class GradientBlur(torch.nn.Module):
    """Apply spatially-varying blur simulating sample tilt.

    Blends original with a fully blurred copy using a linear gradient mask.
    The gradient direction is random (0-360 degrees) or flat (uniform sigma).
    """

    def __init__(self, max_sigma: float = 2.0, p: float = 0.3):
        super().__init__()
        self.max_sigma = max_sigma
        self.p = p
        # Kernel size must be odd and >= 3*sigma
        self.kernel_size = max(3, int(np.ceil(max_sigma * 3)) | 1)  # ensure odd

    def forward(self, img):
        if img.dim() == 4:
            for i in range(img.shape[0]):
                img[i] = self._apply_single(img[i])
            return img
        return self._apply_single(img)

    def _apply_single(self, img):
        if torch.rand(1).item() > self.p:
            return img

        sigma = torch.empty(1).uniform_(0, self.max_sigma).item()
        if sigma < 0.1:
            return img  # negligible blur, skip
        blur = v2.GaussianBlur(kernel_size=self.kernel_size, sigma=sigma)
        blurred = blur(img)

        # Linear gradient mask along a random direction
        _, h, w = img.shape
        angle = torch.empty(1).uniform_(0, 2 * np.pi).item()

        # Create coordinate grids
        ys = torch.linspace(-1, 1, h, device=img.device)
        xs = torch.linspace(-1, 1, w, device=img.device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')

        # Project onto direction vector → gradient from 0 to 1
        proj = xx * np.cos(angle) + yy * np.sin(angle)
        mask = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
        mask = mask.unsqueeze(0)  # [1, H, W]

        return img * (1 - mask) + blurred * mask


class RemoIDDataLoader:
    """
    Efficient data loader for remo-id that handles zarr file access and augmentations.
    """

    def __init__(self, data: pd.DataFrame, column: str, batch_size: int, experiment_ranges: dict, backbone_name: str = 'vit-b-16', train: bool = True,
                 background_correction: str = 'none', background_value: int = None,
                 intensity_jitter: float = None,
                 noise_std: float = None, blur_sigma: float = None,
                 gradient_blur_sigma: float = None, min_image_index: int = 0,
                 crop_size: int = None, erase_num: str = None, erase_scale: str = None):
        self.augments = RemoIDDataLoader.prep_augments(
            backbone_name, train=train,
            intensity_jitter=intensity_jitter, noise_std=noise_std,
            blur_sigma=blur_sigma, gradient_blur_sigma=gradient_blur_sigma,
            crop_size=crop_size, erase_num=erase_num, erase_scale=erase_scale,
        )
        self.dataset = data
        self.column = column
        self.batch_size = batch_size
        self.zarr_cache = {}
        self.experiment_ranges = experiment_ranges
        self.label_by_path = dict(zip(self.dataset["path"], self.dataset[self.column]))
        self._min_image_index = min_image_index
        self.train = train
        self.background_correction = background_correction
        self.background_value = float(background_value if background_value is not None else 225)

        ranges = sorted([(s, e, p) for p, (s, e) in self.experiment_ranges.items()], key=lambda x: x[0])
        self._starts = [s for s,_,_ in ranges]
        self._ends   = [e for _,e,_ in ranges]
        self._paths  = [p for *_,p in ranges]

        # p99 flat-field background cache and pool
        self.p99_cache: dict[str, torch.Tensor | None] = {}
        self.p99_pool: list[torch.Tensor] = []
        self._p99_warned: set[str] = set()  # track warned paths
        if self.background_correction != 'none':
            self._load_p99_pool()

    def _get_p99(self, exp_path: str) -> torch.Tensor | None:
        """Load p99 background image for an experiment, with caching."""
        if exp_path in self.p99_cache:
            return self.p99_cache[exp_path]

        basename = os.path.basename(exp_path.rstrip('/'))
        p99_path = os.path.join(exp_path, f"{basename}background_p99.png")

        try:
            p99_np = np.array(Image.open(p99_path))  # uint8 [H, W]
            p99_tensor = torch.from_numpy(p99_np).float().unsqueeze(0)  # float32 [1, H, W]
            self.p99_cache[exp_path] = p99_tensor
            return p99_tensor
        except FileNotFoundError:
            if exp_path not in self._p99_warned:
                self._p99_warned.add(exp_path)
                warnings.warn(f"No p99 background found for {exp_path}, falling back to scalar normalization")
            self.p99_cache[exp_path] = None
            return None

    def _load_p99_pool(self):
        """Preload p99 backgrounds for all experiments and build the pool."""
        n_loaded = 0
        total = len(self._paths)
        for exp_path in self._paths:
            p99 = self._get_p99(exp_path)
            if p99 is not None:
                self.p99_pool.append(p99)
                n_loaded += 1
        print(f"Loaded {n_loaded}/{total} p99 backgrounds for flat-field correction")

    def _select_p99(self, exp_path: str) -> torch.Tensor | None:
        """Select a p99 background based on correction mode and train/val context."""
        # Val/inference always uses scalar normalization (no per-pixel correction)
        if not self.train:
            return None

        mode = self.background_correction
        if mode == 'none':
            return None
        elif mode == 'own':
            return self.p99_cache.get(exp_path)
        elif mode == 'random':
            if self.p99_pool:
                return random.choice(self.p99_pool)
            return self.p99_cache.get(exp_path)
        elif mode == 'blend':
            roll = random.random()
            if roll < 1/3:
                # Own p99
                return self.p99_cache.get(exp_path)
            elif roll < 2/3:
                # Random p99 from pool
                if self.p99_pool:
                    return random.choice(self.p99_pool)
                return self.p99_cache.get(exp_path)
            else:
                # Scalar normalization (no per-pixel correction)
                return None
        return None

    def _apply_flat_field(self, img: torch.Tensor, exp_path: str) -> torch.Tensor:
        """Apply background subtraction. Always runs — IS the uint8 → float32 normalization.

        Subtracts the per-pixel p99 background to remove dust and illumination
        pattern, then normalizes by scalar background_value. Falls back to
        scalar-only normalization when no p99 is available.

        Args:
            img: uint8 tensor [1, H, W] or [B, 1, H, W]
            exp_path: experiment path for p99 lookup

        Returns:
            float32 tensor with bg ~1.0
        """
        # Swap mode: subtract own background, add a random experiment's background
        if self.train and self.background_correction == 'swap':
            own_p99 = self.p99_cache.get(exp_path)
            if own_p99 is None:
                own_p99 = self._get_p99(exp_path)
            if own_p99 is not None and self.p99_pool:
                random_p99 = random.choice(self.p99_pool)
                return ((img.float() - own_p99 + random_p99) / self.background_value).clamp(min=0)
            elif own_p99 is not None:
                return ((img.float() - own_p99 + self.background_value) / self.background_value).clamp(min=0)
            return img.float() / self.background_value

        p99 = self._select_p99(exp_path)
        if p99 is not None:
            # Subtract per-pixel background (removes dust + illumination pattern),
            # add back uniform background, then scalar-normalize.
            # Clamp to avoid negative values from very dark pixels.
            return ((img.float() - p99 + self.background_value) / self.background_value).clamp(min=0)
        else:
            # Scalar normalization
            return img.float() / self.background_value

    def _get_zarr(self, path):
        if path not in self.zarr_cache:
            self.zarr_cache[path] = get_zarr_file(path)
        return self.zarr_cache[path]

    def get_item(self, idx: int):
        exp_path, local_idx = self._find_experiment_for_global_idx(idx)

        zarr_file = self._get_zarr(exp_path)
        img = zarr_file[:, :, local_idx]  # [H, W]
        img = torch.from_numpy(img) if not torch.is_tensor(img) else img
        img = img.unsqueeze(0)            # [1, H, W] - native grayscale
        img = self._apply_flat_field(img, exp_path)  # → float32 [1, H, W]

        if self.augments is not None:
            img = self.augments(img)

        label = self.label_by_path[exp_path]
        exp_id = int(self.dataset[self.dataset["path"] == exp_path]["experiment_id"].values[0])
        return img, label, exp_id

    def _find_experiment_for_global_idx(self, global_idx: int):
        i = bisect.bisect_right(self._starts, global_idx) - 1
        if i < 0 or global_idx >= self._ends[i]:
            raise ValueError(f"Index {global_idx} out of range.")
        local_idx = global_idx - self._starts[i] + self._min_image_index
        return self._paths[i], local_idx

    def get_items(self, idxs: list[int]):
        if len(idxs) == 0:
            raise ValueError("Empty batch.")

        # Map each idx -> (exp_path, local_idx) in the SAME order as idxs
        mapped = [self._find_experiment_for_global_idx(int(i)) for i in idxs]

        # Build contiguous runs of the same experiment, preserving order
        runs = []  # list of (exp_path, [local_idx, local_idx, ...] in order)
        cur_exp, cur_locals = mapped[0][0], [mapped[0][1]]
        for exp_path, local_idx in mapped[1:]:
            if exp_path == cur_exp:
                cur_locals.append(local_idx)
            else:
                runs.append((cur_exp, cur_locals))
                cur_exp, cur_locals = exp_path, [local_idx]
        runs.append((cur_exp, cur_locals))

        all_imgs = []
        all_labels = []
        all_exp_ids = []

        for exp_path, local_indices in runs:
            zarr_file = self._get_zarr(exp_path)

            # zarr slice preserves the order of local_indices
            imgs = zarr_file[:, :, local_indices]         # [H, W, B]
            imgs = torch.from_numpy(imgs) if not torch.is_tensor(imgs) else imgs
            imgs = imgs.permute(2, 0, 1)                  # [B, H, W]
            imgs = imgs.unsqueeze(1)                       # [B, 1, H, W] - native grayscale
            imgs = self._apply_flat_field(imgs, exp_path)  # → float32 [B, 1, H, W]

            if self.augments is not None:
                imgs = self.augments(imgs)

            label = self.dataset[self.dataset["path"] == exp_path][self.column].values[0]
            exp_id = int(self.dataset[self.dataset["path"] == exp_path]["experiment_id"].values[0])
            all_imgs.append(imgs)
            all_labels.extend([label] * len(local_indices))
            all_exp_ids.extend([exp_id] * len(local_indices))

        x = torch.cat(all_imgs, dim=0)  # [B, 1, H, W] in EXACT batch idx order
        y = torch.tensor(all_labels, dtype=torch.float32)
        exp_ids = torch.tensor(all_exp_ids, dtype=torch.long)
        return x, y, exp_ids

    @staticmethod
    def prep_augments(backbone_name: str = 'vit-b-16', train: bool = True,
                      intensity_jitter: float = None,
                      noise_std: float = None, blur_sigma: float = None,
                      gradient_blur_sigma: float = None,
                      crop_size: int = None,
                      erase_num: str = None, erase_scale: str = None):
        """
        Prepare augmentation pipeline for single-channel grayscale images.
        All backbones now expect [1, H, W] input.

        Normalization (uint8 → float32) is handled externally by _apply_flat_field()
        in RemoIDDataLoader, or by the caller for standalone usage. This pipeline
        expects float32 input and wraps it as tv_tensors.Image.

        Args:
            backbone_name: Name of the backbone model
            train: If True, use random augmentations; if False, use deterministic transforms
            intensity_jitter: If set, apply ColorJitter with this brightness/contrast range (train only)
            noise_std: If set, add Gaussian noise with max std (train only)
            blur_sigma: If set, apply random GaussianBlur with max sigma (train only)
            gradient_blur_sigma: If set, apply spatially-varying gradient blur (train only)
            erase_num: "MIN,MAX" override for MultiRandomErasing count (train only;
                None = backbone default)
            erase_scale: "MIN,MAX" override for MultiRandomErasing per-mask area
                fraction (train only; None = backbone default)
        """
        backbone = backbone_name.lower()

        def _pair(s, cast):
            return tuple(cast(x) for x in str(s).split(',')) if s is not None else None
        _erase_num = _pair(erase_num, int)
        _erase_scale = _pair(erase_scale, float)

        # --- Normalization block (shared by train and eval) ---
        # Wraps float32 tensor as tv_tensors.Image for torchvision v2 transforms.
        # Actual uint8→float32 normalization is done by _apply_flat_field() before this pipeline.
        norm_transforms = [v2.ToImage()]

        # --- Intensity augmentation block (train only) ---
        intensity_augments = []
        if train:
            if intensity_jitter is not None:
                intensity_augments.append(v2.ColorJitter(brightness=intensity_jitter, contrast=intensity_jitter))
            if noise_std is not None:
                intensity_augments.append(GaussianNoise(max_std=noise_std, p=0.5))
            if blur_sigma is not None:
                kernel_size = max(3, int(np.ceil(blur_sigma * 3)) | 1)
                intensity_augments.append(v2.RandomApply([v2.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, blur_sigma))], p=0.2))
            if gradient_blur_sigma is not None:
                intensity_augments.append(GradientBlur(max_sigma=gradient_blur_sigma, p=0.3))

        # --- Backbone-specific spatial transforms ---
        if backbone in ('vit-b-16', 'vit-b-16-1ch'):
            if train:
                spatial = [
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.Resize((512), interpolation=v2.InterpolationMode.BILINEAR),
                    v2.RandomCrop(224),
                ]
                post = [MultiRandomErasing(num_erases=_erase_num or (5, 10), scale=_erase_scale or (0.005, 0.03), ratio=(0.5, 2.0))]
            else:
                spatial = [
                    v2.Resize((512), interpolation=v2.InterpolationMode.BILINEAR),
                    v2.CenterCrop(224),  # deterministic eval crop
                ]
                post = []

        elif backbone in ('resnet18', 'resnet18-mil'):
            _crop = crop_size or 224
            if train:
                spatial = [v2.RandomHorizontalFlip(p=0.5), v2.RandomCrop(_crop)]
                post = [MultiRandomErasing(num_erases=_erase_num or (10, 25), scale=_erase_scale or (0.001, 0.01), ratio=(0.5, 2.0))]
            else:
                spatial = [v2.CenterCrop(_crop)]  # deterministic eval crop
                post = []

        elif backbone in ('resnet50', 'resnet50-mil'):
            if crop_size:
                if train:
                    spatial = [v2.RandomHorizontalFlip(p=0.5), v2.RandomCrop(crop_size)]
                    post = [MultiRandomErasing(num_erases=_erase_num or (10, 25), scale=_erase_scale or (0.001, 0.01), ratio=(0.5, 2.0))]
                else:
                    spatial = [v2.CenterCrop(crop_size)]  # deterministic eval crop
                    post = []
            else:
                if train:
                    spatial = [v2.RandomHorizontalFlip(p=0.5)]
                    post = [MultiRandomErasing(num_erases=_erase_num or (10, 25), scale=_erase_scale or (0.001, 0.01), ratio=(0.5, 2.0))]
                else:
                    spatial = []
                    post = []

        elif backbone in ('vit-l-16', 'vit-l-16-mil', 'vit-l-16-1ch'):
            _crop = crop_size or 512
            if train:
                spatial = [v2.RandomHorizontalFlip(p=0.5), v2.RandomCrop(_crop)]
                post = [MultiRandomErasing(num_erases=_erase_num or (5, 12), scale=_erase_scale or (0.005, 0.01), ratio=(0.5, 2.0))]
            else:
                spatial = [v2.CenterCrop(_crop)]  # deterministic eval crop
                post = []

        elif backbone in ('dinov3-vits16', 'dino-vits16', 'dino-vits16-mil'):
            _crop = crop_size or 768
            if train:
                spatial = [v2.RandomHorizontalFlip(p=0.5), v2.RandomCrop(_crop)]
                post = [MultiRandomErasing(num_erases=_erase_num or (5, 10), scale=_erase_scale or (0.005, 0.03), ratio=(0.5, 2.0))]
            else:
                spatial = [v2.CenterCrop(_crop)]  # deterministic eval crop
                post = []

        elif backbone in ('dino-vitb',):
            _crop = crop_size or 768
            if train:
                spatial = [v2.RandomHorizontalFlip(p=0.5), v2.RandomCrop(_crop)]
                post = [MultiRandomErasing(num_erases=_erase_num or (5, 10), scale=_erase_scale or (0.005, 0.03), ratio=(0.5, 2.0))]
            else:
                spatial = [v2.CenterCrop(_crop)]  # deterministic eval crop
                post = []

        elif backbone in ('microscopy-cnn', 'microscopy-cnn-mil', 'microscopycnn'):
            if train:
                spatial = [v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5)]
                post = [MultiRandomErasing(num_erases=_erase_num or (10, 25), scale=_erase_scale or (0.001, 0.005), ratio=(0.5, 2.0))]
            else:
                spatial = []
                post = []

        else:
            if train:
                spatial = [v2.RandomHorizontalFlip(p=0.5)]
                post = [MultiRandomErasing(num_erases=_erase_num or (10, 25), scale=_erase_scale or (0.001, 0.005), ratio=(0.5, 2.0))]
            else:
                spatial = []
                post = []

        # Assemble: normalization → spatial → intensity augments → erasing
        return v2.Compose(norm_transforms + spatial + intensity_augments + post)
