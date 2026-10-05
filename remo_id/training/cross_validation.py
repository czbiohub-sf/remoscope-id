"""K-Fold cross-validation splitting utilities.

Pure data splitting — no imports from inference, analysis, or reporting.
Orchestration (run_cross_validation, _predict_fold, etc.) lives in workflows/cv.py.
"""

import os
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Set


@dataclass
class CVConfig:
    n_folds: int = 10
    n_repeats: int = 5
    group_by: str = "sample_id"
    seed: int = 403


def create_cv_splits(
    data_module,
    config: CVConfig,
) -> List[List[Tuple[Set[str], Set[str]]]]:
    """Create stratified group k-fold splits.

    Returns nested list: repeats[folds[(train_paths, val_paths)]].
    Uses StratifiedGroupKFold with sample_id grouping.

    Args:
        data_module: RemoIDDataModule with loaded data
        config: CVConfig with n_folds, n_repeats, group_by, seed

    Returns:
        List of repeats, each containing a list of (train_paths, val_paths) tuples
    """
    from sklearn.model_selection import StratifiedGroupKFold

    df = data_module.data.copy()
    paths = df["path"].values
    labels = np.array([
        int(data_module.experiment_labels[p])
        for p in paths
    ])

    # Build groups from sample_id
    groups = []
    null_counter = 0
    for _, row in df.iterrows():
        sid = row.get(config.group_by)
        if pd.isna(sid) or sid is None:
            groups.append(f"__null_{null_counter}")
            null_counter += 1
        else:
            groups.append(str(sid))
    groups = np.array(groups)

    all_repeats = []
    for repeat_idx in range(config.n_repeats):
        repeat_seed = config.seed + repeat_idx * 1000
        sgkf = StratifiedGroupKFold(
            n_splits=config.n_folds,
            shuffle=True,
            random_state=repeat_seed,
        )

        folds = []
        for train_idx, val_idx in sgkf.split(paths, labels, groups):
            train_paths = set(paths[train_idx])
            val_paths = set(paths[val_idx])
            folds.append((train_paths, val_paths))
        all_repeats.append(folds)

    all_repeats = validate_cv_class_coverage(all_repeats, data_module)
    return all_repeats


def create_loo_splits(
    data_module,
) -> List[List[Tuple[Set[str], Set[str]]]]:
    """One fold per sample_id: val = {all experiments for a sample}, train = {rest}.

    LOO is deterministic so there is always exactly 1 repeat.
    Experiments with null sample_id are each treated as their own group.

    Returns:
        List containing a single repeat with N folds (one per unique sample).
        Same format as create_cv_splits(): List[List[Tuple[Set, Set]]].
    """
    df = data_module.data.copy()
    all_paths = set(df["path"].values)

    # Group paths by sample_id; null sample_ids become singleton groups
    sample_groups = {}
    null_counter = 0
    for _, row in df.iterrows():
        sid = row.get("sample_id")
        if pd.isna(sid) or sid is None:
            key = f"__null_{null_counter}"
            null_counter += 1
        else:
            key = str(sid)
        sample_groups.setdefault(key, set()).add(row["path"])

    folds = []
    for key in sorted(sample_groups.keys()):
        val_paths = sample_groups[key]
        train_paths = all_paths - val_paths
        folds.append((train_paths, val_paths))
    all_repeats = [folds]
    # Skip validate_cv_class_coverage for LOO: each fold intentionally holds
    # out a single sample_id which typically has only one class. The validator
    # would incorrectly pull extra sample_ids into val to cover missing classes.
    return all_repeats


def validate_cv_class_coverage(
    splits: List[List[Tuple[Set[str], Set[str]]]],
    data_module,
) -> List[List[Tuple[Set[str], Set[str]]]]:
    """Validate and fix class coverage in CV splits.

    Ensures every fold's val and train sets contain at least 1 experiment
    per class. If a class is missing from val, swaps a sample_id group
    containing that class from train to val (and vice versa).

    Raises ValueError if a class has fewer unique samples than folds,
    making full coverage impossible.
    """
    task_type = getattr(data_module, 'task_type', 'binary')
    num_classes = getattr(data_module, 'num_classes', 2)

    needed = {0, 1} if task_type == "binary" else set(range(num_classes))
    labels = data_module.experiment_labels  # path -> label

    # Build sample_id -> paths mapping
    df = data_module.data
    sample_to_paths = {}
    null_counter = 0
    path_to_sample = {}
    for _, row in df.iterrows():
        sid = row.get("sample_id")
        if pd.isna(sid) or sid is None:
            sid = f"__null_{null_counter}"
            null_counter += 1
        else:
            sid = str(sid)
        sample_to_paths.setdefault(sid, set()).add(row["path"])
        path_to_sample[row["path"]] = sid

    # Determine dominant class per sample_id
    sample_labels = {}
    for sid, paths in sample_to_paths.items():
        cls_counts = {}
        for p in paths:
            if p in labels:
                c = int(labels[p])
                cls_counts[c] = cls_counts.get(c, 0) + 1
        if cls_counts:
            sample_labels[sid] = max(cls_counts, key=cls_counts.get)

    def _get_classes(paths_set):
        classes = set()
        for p in paths_set:
            if p in labels:
                classes.add(int(labels[p]))
        return classes

    def _get_samples(paths_set):
        return {path_to_sample[p] for p in paths_set if p in path_to_sample}

    fixes = 0
    for repeat_idx, folds in enumerate(splits):
        for fold_idx, (train_paths, val_paths) in enumerate(folds):
            val_classes = _get_classes(val_paths)
            train_classes = _get_classes(train_paths)

            # Fix val missing classes
            for missing_class in needed - val_classes:
                train_samples = _get_samples(train_paths)
                moved = False
                for sid in sorted(train_samples):
                    if sample_labels.get(sid) == missing_class and len(train_samples) > 1:
                        paths_to_move = sample_to_paths[sid]
                        train_paths -= paths_to_move
                        val_paths |= paths_to_move
                        fixes += 1
                        moved = True
                        break
                if not moved:
                    raise ValueError(
                        f"Cannot fix class {missing_class} coverage in "
                        f"repeat{repeat_idx}_fold{fold_idx} val set — "
                        f"not enough samples with this class"
                    )

            # Fix train missing classes
            train_classes = _get_classes(train_paths)
            for missing_class in needed - train_classes:
                val_samples = _get_samples(val_paths)
                moved = False
                for sid in sorted(val_samples):
                    if sample_labels.get(sid) == missing_class and len(val_samples) > 1:
                        paths_to_move = sample_to_paths[sid]
                        val_paths -= paths_to_move
                        train_paths |= paths_to_move
                        fixes += 1
                        moved = True
                        break
                if not moved:
                    raise ValueError(
                        f"Cannot fix class {missing_class} coverage in "
                        f"repeat{repeat_idx}_fold{fold_idx} train set — "
                        f"not enough samples with this class"
                    )

            folds[fold_idx] = (train_paths, val_paths)

    if fixes > 0:
        print(f"CV class-coverage validation: {fixes} sample swaps to ensure all folds have all classes")

    return splits


def save_cv_splits(
    splits: List[List[Tuple[Set[str], Set[str]]]],
    data_module,
    save_path: str,
):
    """Save CV splits to YAML before training starts (enables SLURM resume).

    Each fold stores experiment_ids (not paths) for readability.
    """
    path_to_exp_id = dict(zip(
        data_module.data["path"],
        data_module.data["experiment_id"],
    ))

    data = {"repeats": []}
    for repeat_idx, folds in enumerate(splits):
        repeat_data = []
        for fold_idx, (train_paths, val_paths) in enumerate(folds):
            repeat_data.append({
                "fold": fold_idx,
                "train_experiment_ids": sorted(int(path_to_exp_id[p]) for p in train_paths if p in path_to_exp_id),
                "val_experiment_ids": sorted(int(path_to_exp_id[p]) for p in val_paths if p in path_to_exp_id),
            })
        data["repeats"].append(repeat_data)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    # Atomic write: temp file + rename to avoid corruption from parallel SLURM array tasks
    import tempfile
    dir_name = os.path.dirname(save_path) or "."
    with tempfile.NamedTemporaryFile(mode="w", dir=dir_name, suffix=".yaml", delete=False) as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        tmp_path = f.name
    os.replace(tmp_path, save_path)
    print(f"Saved CV splits ({len(splits)} repeats x {len(splits[0])} folds) to {save_path}")


def load_cv_splits(
    splits_path: str,
    data_module,
) -> List[List[Tuple[Set[str], Set[str]]]]:
    """Load CV splits from YAML, converting experiment_ids back to paths."""
    with open(splits_path) as f:
        data = yaml.safe_load(f)

    exp_id_to_path = dict(zip(
        data_module.data["experiment_id"].astype(str),
        data_module.data["path"],
    ))

    all_repeats = []
    for repeat_data in data["repeats"]:
        folds = []
        for fold_data in repeat_data:
            train_paths = {exp_id_to_path[str(eid)] for eid in fold_data["train_experiment_ids"] if str(eid) in exp_id_to_path}
            val_paths = {exp_id_to_path[str(eid)] for eid in fold_data["val_experiment_ids"] if str(eid) in exp_id_to_path}
            folds.append((train_paths, val_paths))
        all_repeats.append(folds)

    return all_repeats
