"""Parallel LOO prediction: one SLURM task per epoch.

Each task predicts all folds at a single epoch checkpoint, saving raw
logits/predictions to {cv_dir}/predictions/epoch_{N}.npz.

The aggregation step (loo-report) then loads these .npz files instead of
running GPU inference.
"""

import glob as glob_module
import os

import numpy as np
import torch

from remo_id.inference.core import (
    load_checkpoint,
    extract_checkpoint_metadata,
    set_inference_seeds,
)
from remo_id.inference.mil import _predict_experiments
from remo_id.training.cross_validation import load_cv_splits
from remo_id.workflows.loo_report import (
    _find_common_epochs,
    _find_epoch_checkpoint,
    _extract_training_bag_size,
)


def predict_epoch(
    cv_dir: str,
    epoch: int,
    class_config: str,
    seed: int = 403,
    n_bags: int = 5,
    bag_size_override: int = None,
    background_value: int = None,
    output_dir: str = None,
    scan_size: int = None,
    top_k: int = None,
    model_version: str = None,
    cell_config: str = None,
    load_workers: int = None,
    pool_cap: int = None,
    all_experiments: bool = False,
    fold_shard: int = None,
    fold_nshards: int = None,
) -> str:
    """Predict all folds at a single epoch, save results to .npz.

    When scan_size and top_k are set, uses two-stage hard-attention:
    scan scan_size images, select top_k by attention, re-aggregate.

    ``all_experiments`` makes every fold score EVERY experiment instead of only
    its held-out one, and adds an ``in_train`` flag per row. That turns the
    output from a LOO prediction set into a fold x experiment matrix, which is
    what you need to ask "does this experiment stay wrong even when the model
    trained on it?" -- these rows are memorization diagnostics, NOT held-out
    performance, so they must never be pooled into a metric.

    Because that multiplies the work by the number of folds, ``fold_shard``/``fold_nshards`` split the fold
    loop across SLURM array tasks; each shard writes its own npz suffixed with
    the shard index, to be concatenated afterwards.

    Returns path to saved .npz file.
    """
    from remo_id.inference.context import load_class_config
    from remo_id.data import RemoIDDataModule
    from remo_id.models import get_augmentation_backbone

    # Load class config
    class_definitions, class_names, num_classes, task_type, exclude_filters, include_filters, default_class_idx = (
        load_class_config(class_config)
    )

    # Extract training bag size
    training_bag_size = _extract_training_bag_size(cv_dir)
    if bag_size_override is not None:
        training_bag_size = bag_size_override

    # Extract model name from a checkpoint
    fold_dirs = sorted(glob_module.glob(os.path.join(cv_dir, "repeat*_fold*")))
    model_name = "resnet18-mil"
    for fd in fold_dirs:
        ckpts = glob_module.glob(os.path.join(fd, "*.ckpt"))
        for c in ckpts:
            try:
                meta = extract_checkpoint_metadata(c)
                model_name = meta.get("model_name", "resnet18-mil")
                break
            except Exception:
                continue
        break

    backbone_name = get_augmentation_backbone(model_name)

    # Build data module
    dm_kwargs = dict(
        database_column="class_label",
        database_label=None,
        study_id=None,
        threshold=None,
        threshold_mode="==",
        negative_study_ids=[],
        negative_experiment_ids=[],
        batch_size=1,
        backbone_name=backbone_name,
        train=False,
        task_type=task_type,
        num_classes=num_classes,
        class_definitions=class_definitions,
        exclude_filters=exclude_filters,
        include_filters=include_filters,
        default_class_idx=default_class_idx,
        skip_null_labels=True,
        background_correction='none',
    )
    if background_value is not None:
        dm_kwargs['background_value'] = background_value

    if model_name == "embedding-mil":
        from remo_id.data import EmbeddingDataModule
        if not model_version:
            raise ValueError("embedding-mil LOO requires --model_version")
        data_module = EmbeddingDataModule(
            model_version=model_version, cell_config=cell_config, pool_cap=pool_cap,
            database_column="class_label", task_type=task_type, num_classes=num_classes,
            class_definitions=class_definitions, exclude_filters=exclude_filters,
            include_filters=include_filters, default_class_idx=default_class_idx,
            batch_size=1, skip_null_labels=True, load_workers=load_workers,
        )
    else:
        data_module = RemoIDDataModule(**dm_kwargs)

    # Load CV splits
    splits_path = os.path.join(cv_dir, "cv_splits.yaml")
    all_repeats = load_cv_splits(splits_path, data_module)
    fold_val_paths = {}
    for repeat_idx, folds in enumerate(all_repeats):
        for fold_idx, (train_paths, val_paths) in enumerate(folds):
            label = f"repeat{repeat_idx}_fold{fold_idx}"
            fold_val_paths[label] = val_paths

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_inference_seeds(seed)

    # Predict each fold
    all_y_true = []
    all_y_probs = []
    all_y_pred = []
    all_exp_ids = []
    all_study_ids = []
    all_folds = []
    all_repeats_idx = []
    all_in_train = []

    if fold_shard is not None:
        if fold_nshards is None:
            raise ValueError("fold_shard requires fold_nshards")
        fold_dirs = fold_dirs[fold_shard::fold_nshards]
        print(f"  shard {fold_shard}/{fold_nshards}: {len(fold_dirs)} folds "
              f"({[os.path.basename(f) for f in fold_dirs]})")

    for fd in fold_dirs:
        fold_label = os.path.basename(fd)
        val_paths = fold_val_paths.get(fold_label, set())
        if not val_paths:
            print(f"  Warning: no val paths for {fold_label}, skipping")
            continue

        ckpt_path = _find_epoch_checkpoint(fd, epoch)
        if ckpt_path is None:
            print(f"  Warning: no epoch={epoch} checkpoint in {fd}")
            continue

        print(f"  {fold_label}: loading {os.path.basename(ckpt_path)}")

        ckpt_meta = extract_checkpoint_metadata(ckpt_path)
        mn = ckpt_meta.get("model_name", model_name)

        model, device_used, _ = load_checkpoint(
            ckpt_path, mn,
            task_type=task_type, num_classes=num_classes,
            n_experiments=1, device=device,
        )

        # Build filtered data module for the experiments this fold will score.
        # Default: its held-out set. all_experiments: the whole cohort, so the
        # fold also re-scores what it trained on.
        all_paths = set(data_module.data["path"].values)
        target_paths = all_paths if all_experiments else val_paths
        other_paths = all_paths - target_paths
        _, target_module = data_module.split_from_paths(other_paths, target_paths)

        df = target_module.data.copy()
        df["experiment_id"] = df["experiment_id"].astype(str)
        # Held-out membership is by PATH (what cv_splits.yaml stores), resolved
        # to experiment_id here so each output row can say whether this fold
        # trained on it.
        val_eids = set(df.loc[df["path"].isin(val_paths), "experiment_id"])

        results, _, _, _, _, _ = _predict_experiments(
            model=model, ds=target_module, df=df,
            database_column=target_module.database_column,
            task_type=task_type, num_classes=num_classes,
            bag_size=training_bag_size, seed=seed, device=device_used,
            pred_threshold=0.5,
            class_names=class_names,
            n_bags=n_bags,
            scan_size=scan_size,
            top_k=top_k,
        )

        valid = [r for r in results if not np.isnan(r.y_prob) and not r.has_null_label]

        # Parse fold indices
        parts = fold_label.replace("repeat", "").replace("fold", " ").split("_")
        try:
            repeat_idx = int(parts[0])
            fold_idx = int(parts[1].strip())
        except (ValueError, IndexError):
            repeat_idx, fold_idx = 0, 0

        for r in valid:
            all_y_true.append(r.y_true)
            all_y_probs.append(r.y_probs_all)
            all_y_pred.append(r.y_pred)
            all_exp_ids.append(r.experiment_id)
            all_study_ids.append(getattr(r, "study_id", ""))
            all_folds.append(fold_idx)
            all_repeats_idx.append(repeat_idx)
            all_in_train.append(0 if r.experiment_id in val_eids else 1)

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Incremental save after each fold (crash-safe)
        pred_dir = output_dir if output_dir else os.path.join(cv_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        epoch_label = {-2: "best", -1: "last"}.get(epoch, str(epoch))
        shard_suffix = f"_shard{fold_shard}" if fold_shard is not None else ""
        out_path = os.path.join(pred_dir, f"epoch_{epoch_label}{shard_suffix}.npz")
        np.savez(
            out_path,
            in_train=np.array(all_in_train, dtype=int),
            y_true=np.array(all_y_true, dtype=int),
            y_probs=np.array(all_y_probs, dtype=float),
            y_pred=np.array(all_y_pred, dtype=int),
            experiment_ids=np.array(all_exp_ids, dtype=str),
            study_ids=np.array(all_study_ids, dtype=str),
            folds=np.array(all_folds, dtype=int),
            repeats=np.array(all_repeats_idx, dtype=int),
            class_names=np.array(class_names, dtype=str),
            epoch=np.array(epoch),
        )

    n_exps = len(all_y_true)
    n_folds = len(set(all_folds))
    print(f"  Saved {n_exps} predictions from {n_folds} folds to {out_path}")
    return out_path


def aggregate_epoch_from_embeddings(
    cv_dir: str,
    epoch: int,
    class_config: str,
    seed: int = 403,
    n_bags: int = 5,
    bag_size: int = 500,
    output_dir: str = None,
    embeddings_dir: str = None,
) -> str:
    """Pool LOO predictions at a fixed epoch by SAMPLING cached per-image
    embeddings and running ONLY the aggregator head -- no backbone encode.

    Numerically identical (model in eval mode) to ``predict_epoch``'s
    encode-then-aggregate path: it rebuilds each fold's held-out target module
    exactly as predict_epoch does (so the per-experiment iteration index ``i``
    and ``experiment_ranges`` match), reproduces the same per-bag seeds
    ``seed + 1009*i + 7919*bag_idx`` and the same local image indices via
    ``pick_global_indices_for_experiment``, then feeds the matching cached
    post-projection embeddings to ``model._aggregate`` and averages over n_bags.

    Reads embeddings from ``{cv_dir}/embeddings/epoch_{N}/{exp_id}.embeddings.npz``
    (override with ``embeddings_dir``). Writes ``{cv_dir}/predictions/epoch_{N}.npz``
    with the SAME schema/dtypes as ``predict_epoch``. Returns the .npz path.
    """
    from remo_id.inference.context import load_class_config
    from remo_id.data import RemoIDDataModule
    from remo_id.models import get_augmentation_backbone
    from remo_id.inference.mil import pick_global_indices_for_experiment

    class_definitions, class_names, num_classes, task_type, exclude_filters, include_filters, default_class_idx = (
        load_class_config(class_config)
    )

    epoch_label = {-2: "best", -1: "last"}.get(epoch, str(epoch))
    emb_dir = embeddings_dir or os.path.join(cv_dir, "embeddings", f"epoch_{epoch_label}")
    if not os.path.isdir(emb_dir):
        raise FileNotFoundError(
            f"No embeddings dir for epoch {epoch}: {emb_dir}. Run the embedding "
            f"pass (workflow loo-per-image --output_dir {emb_dir}) first."
        )

    # Model name from any checkpoint
    fold_dirs = sorted(glob_module.glob(os.path.join(cv_dir, "repeat*_fold*")))
    model_name = "resnet18-mil"
    for fd in fold_dirs:
        for c in glob_module.glob(os.path.join(fd, "*.ckpt")):
            try:
                model_name = extract_checkpoint_metadata(c).get("model_name", "resnet18-mil")
                break
            except Exception:
                continue
        break
    backbone_name = get_augmentation_backbone(model_name)

    # Build data module + splits IDENTICALLY to predict_epoch so iteration order
    # (the `i` in the bag seed) and experiment_ranges line up exactly.
    data_module = RemoIDDataModule(
        database_column="class_label", database_label=None, study_id=None,
        threshold=None, threshold_mode="==", negative_study_ids=[], negative_experiment_ids=[],
        batch_size=1, backbone_name=backbone_name, train=False,
        task_type=task_type, num_classes=num_classes,
        class_definitions=class_definitions, exclude_filters=exclude_filters,
        include_filters=include_filters, default_class_idx=default_class_idx,
        skip_null_labels=True, background_correction='none',
    )
    splits_path = os.path.join(cv_dir, "cv_splits.yaml")
    all_repeats = load_cv_splits(splits_path, data_module)
    fold_val_paths = {}
    for repeat_idx, folds in enumerate(all_repeats):
        for fi, (train_paths, val_paths) in enumerate(folds):
            fold_val_paths[f"repeat{repeat_idx}_fold{fi}"] = val_paths

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_inference_seeds(seed)

    all_y_true, all_y_probs, all_y_pred = [], [], []
    all_exp_ids, all_study_ids, all_folds, all_repeats_idx = [], [], [], []
    out_path = None
    n_missing = 0

    for fd in fold_dirs:
        fold_label = os.path.basename(fd)
        val_paths = fold_val_paths.get(fold_label, set())
        if not val_paths:
            continue

        ckpt_path = _find_epoch_checkpoint(fd, epoch)
        if ckpt_path is None:
            print(f"  Warning: no epoch={epoch} checkpoint in {fd}")
            continue

        ckpt_meta = extract_checkpoint_metadata(ckpt_path)
        mn = ckpt_meta.get("model_name", model_name)
        model, device_used, _ = load_checkpoint(
            ckpt_path, mn, task_type=task_type, num_classes=num_classes,
            n_experiments=1, device=device,
        )
        model.eval()

        all_paths = set(data_module.data["path"].values)
        other_paths = all_paths - val_paths
        _, target_module = data_module.split_from_paths(other_paths, val_paths)
        df = target_module.data.copy()
        df["experiment_id"] = df["experiment_id"].astype(str)

        parts = fold_label.replace("repeat", "").replace("fold", " ").split("_")
        try:
            repeat_idx = int(parts[0])
            fold_idx = int(parts[1].strip())
        except (ValueError, IndexError):
            repeat_idx, fold_idx = 0, 0

        for i, row in df.iterrows():
            exp_id = str(row["experiment_id"])
            exp_path = str(row["path"])
            if "class_label" not in row:
                continue
            if bool(row.get("_has_null_label", False)):
                continue
            y_true = int(row["class_label"])
            study_id = str(row.get("study_id", ""))
            if exp_path not in target_module.experiment_ranges:
                continue
            start, end = target_module.experiment_ranges[exp_path]

            emb_file = os.path.join(emb_dir, f"{exp_id}.embeddings.npz")
            if not os.path.exists(emb_file):
                print(f"  Warning: missing embeddings for exp {exp_id} ({fold_label})")
                n_missing += 1
                continue
            feats_all = torch.from_numpy(
                np.load(emb_file, allow_pickle=True)["embeddings"].astype(np.float32)
            )

            bag_logits_list = []
            for bag_idx in range(n_bags):
                bag_seed = int(seed + 1009 * i + 7919 * bag_idx)
                set_inference_seeds(bag_seed)
                idxs = pick_global_indices_for_experiment(
                    target_module.experiment_ranges, exp_path, bag_size, bag_seed,
                    sample_mode="random", sample_percent=100.0,
                )
                if len(idxs) == 0:
                    continue
                local = [g - start for g in idxs]
                chunk = feats_all[local].unsqueeze(0).to(device_used)  # [1, bag, D]
                with torch.no_grad():
                    bag_logits, _ = model._aggregate(chunk)
                bag_logits_list.append(bag_logits)

            if not bag_logits_list:
                continue
            avg_logits = torch.stack(bag_logits_list).mean(dim=0)

            if task_type == "binary":
                prob = float(torch.sigmoid(avg_logits[0]).item())
                y_pred = int(prob >= 0.5)
                y_probs_all = [1.0 - prob, prob]
            else:  # multiclass
                logits_0 = avg_logits[0] if avg_logits.dim() > 1 else avg_logits
                probs_t = torch.softmax(logits_0, dim=-1)
                y_pred = int(probs_t.argmax().item())
                y_probs_all = probs_t.cpu().tolist()

            all_y_true.append(y_true)
            all_y_probs.append(y_probs_all)
            all_y_pred.append(y_pred)
            all_exp_ids.append(exp_id)
            all_study_ids.append(study_id)
            all_folds.append(fold_idx)
            all_repeats_idx.append(repeat_idx)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Incremental save after each fold (crash-safe), same schema as predict_epoch
        pred_dir = output_dir if output_dir else os.path.join(cv_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        out_path = os.path.join(pred_dir, f"epoch_{epoch_label}.npz")
        np.savez(
            out_path,
            y_true=np.array(all_y_true, dtype=int),
            y_probs=np.array(all_y_probs, dtype=float),
            y_pred=np.array(all_y_pred, dtype=int),
            experiment_ids=np.array(all_exp_ids, dtype=str),
            study_ids=np.array(all_study_ids, dtype=str),
            folds=np.array(all_folds, dtype=int),
            repeats=np.array(all_repeats_idx, dtype=int),
            class_names=np.array(class_names, dtype=str),
            epoch=np.array(epoch),
        )

    n_exps = len(all_y_true)
    n_folds = len(set(all_folds))
    print(f"  [aggregate] epoch {epoch}: {n_exps} preds from {n_folds} folds "
          f"(bag {bag_size}, {n_bags} bags, {n_missing} missing) -> {out_path}")
    return out_path


def submit_predict_array(
    cv_dir: str,
    class_config: str,
    n_gpus: int = None,
    seed: int = 403,
    n_bags: int = 5,
    bag_size: int = None,
    background_value: int = None,
    model_version: str = None,
    cell_config: str = None,
    epochs: list = None,
    partition: str = "gpu",
) -> str:
    """Generate and submit SLURM array job: one task per epoch.

    ``epochs`` restricts the array to these epochs (default: every epoch with a
    checkpoint in all folds, plus last/best). ``n_gpus=None`` runs all epochs
    concurrently. Returns the job ID.
    """
    import subprocess

    if epochs is None:
        epochs = _find_common_epochs(cv_dir, include_best_last=True)
        print(f"Found {len(epochs)} common epochs: {epochs}")
    else:
        epochs = [int(e) for e in epochs]
        print(f"Predicting epochs: {epochs}")
    n_epochs = len(epochs)
    n_gpus = min(n_gpus or n_epochs, n_epochs)

    # Write epoch list for array tasks to read
    epoch_list_path = os.path.join(cv_dir, "predictions", "epoch_list.txt")
    os.makedirs(os.path.dirname(epoch_list_path), exist_ok=True)
    with open(epoch_list_path, "w") as f:
        for e in epochs:
            f.write(f"{e}\n")

    bag_size_arg = f"--bag_size {bag_size}" if bag_size else ""
    bg_arg = f"--background_value {background_value}" if background_value else ""
    # Embedding-MIL loads all cell embeddings into RAM (~tens of GB) -> bump mem.
    emb_args, mem = "", "64G"
    if model_version:
        emb_args = f"--model_version {model_version}"
        if cell_config:
            emb_args += f" --cell_config {os.path.abspath(cell_config)}"
        mem = "128G"

    script_path = os.path.join(cv_dir, "predictions", "predict_array.sh")
    script = f"""#!/bin/bash
#SBATCH --job-name=loo-predict
#SBATCH --array=0-{n_epochs - 1}%{n_gpus}
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH --partition={partition}
#SBATCH --gpus-per-node=1
#SBATCH --output={cv_dir}/predictions/slurm-%A_%a.out

env | grep "^SLURM" | sort
nvidia-smi

export PYTHONUNBUFFERED=1

# Read epoch for this array task
EPOCH=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {epoch_list_path})
echo "Array task $SLURM_ARRAY_TASK_ID: predicting epoch $EPOCH"

python -m remo_id.cli workflow loo-predict \\
    --cv_dir {cv_dir} \\
    --class_config {os.path.abspath(class_config)} \\
    --epoch $EPOCH \\
    --seed {seed} \\
    --n_bags {n_bags} \\
    {bag_size_arg} {bg_arg} {emb_args}
"""

    os.makedirs(os.path.join(cv_dir, "predictions"), exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script)

    print(f"Submitting prediction array: {n_epochs} epochs across {n_gpus} GPUs")
    print(f"  Script: {script_path}")

    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"  Submitted batch job {job_id}")
        return job_id
    else:
        print(f"  sbatch failed: {result.stderr}")
        return None


def submit_bag_size_sweep(
    cv_dir: str,
    class_config: str,
    epoch: int,
    bag_sizes: list[int],
    seeds: list[int],
    n_gpus: int = None,
    partition: str = "gpu",
) -> str:
    """Submit SLURM array job for bag size sweep: one task per (bag_size, seed).

    Returns job ID.
    """
    import subprocess

    sweep_dir = os.path.join(cv_dir, "bag_size_sweep")
    os.makedirs(os.path.join(sweep_dir, "slurm-outputs"), exist_ok=True)

    # Write parameter file
    param_file = os.path.join(sweep_dir, "params.txt")
    with open(param_file, "w") as f:
        for bs in bag_sizes:
            for s in seeds:
                f.write(f"{bs},{s}\n")

    n_tasks = len(bag_sizes) * len(seeds)
    n_gpus = min(n_gpus or n_tasks, n_tasks)

    script_path = os.path.join(sweep_dir, "sweep_array.sh")
    script = f"""#!/bin/bash
#SBATCH --job-name=bag-sweep
#SBATCH --array=0-{n_tasks - 1}%{n_gpus}
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition={partition}
#SBATCH --gpus-per-node=1
#SBATCH --output={sweep_dir}/slurm-outputs/%A_%a.out

env | grep "^SLURM" | sort
nvidia-smi

export PYTHONUNBUFFERED=1

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {param_file})
BAG_SIZE=$(echo $LINE | cut -d, -f1)
SEED=$(echo $LINE | cut -d, -f2)

echo "Task $SLURM_ARRAY_TASK_ID: bag_size=$BAG_SIZE, seed=$SEED, epoch={epoch}"

python -m remo_id.cli workflow loo-predict \\
    --cv_dir {cv_dir} \\
    --class_config {os.path.abspath(class_config)} \\
    --epoch {epoch} \\
    --bag_size $BAG_SIZE \\
    --seed $SEED \\
    --n_bags 1 \\
    --output_dir {sweep_dir}/bs${{BAG_SIZE}}_seed${{SEED}}
"""
    with open(script_path, "w") as f:
        f.write(script)

    print(f"Submitting bag size sweep: {n_tasks} tasks ({len(bag_sizes)} sizes x {len(seeds)} seeds) across {n_gpus} GPUs")
    print(f"  Bag sizes: {bag_sizes}")
    print(f"  Seeds: {seeds}")
    print(f"  Epoch: {epoch}")
    print(f"  Script: {script_path}")

    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"  Submitted batch job {job_id}")
        return job_id
    else:
        print(f"  sbatch failed: {result.stderr}")
        return None


def aggregate_bag_size_sweep(cv_dir: str, class_names: list[str]) -> dict:
    """Load bag size sweep results and compute metrics per (bag_size, seed).

    Returns dict with 'bag_sizes', 'seeds', 'aucs' (seeds x bag_sizes array),
    'f1s', and per-result details.
    """
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.preprocessing import label_binarize

    sweep_dir = os.path.join(cv_dir, "bag_size_sweep")
    if not os.path.exists(sweep_dir):
        return None

    # Discover all (bag_size, seed) directories
    results = {}
    for d in sorted(os.listdir(sweep_dir)):
        if not d.startswith("bs"):
            continue
        parts = d.split("_seed")
        if len(parts) != 2:
            continue
        bs = int(parts[0].replace("bs", ""))
        seed = int(parts[1])

        # Find npz file
        npz_files = glob_module.glob(os.path.join(sweep_dir, d, "epoch_*.npz"))
        if not npz_files:
            continue

        data = np.load(npz_files[0], allow_pickle=True)
        results[(bs, seed)] = {
            "y_true": data["y_true"],
            "y_probs": data["y_probs"],
        }

    if not results:
        return None

    bag_sizes = sorted(set(bs for bs, _ in results.keys()))
    seeds = sorted(set(s for _, s in results.keys()))
    num_classes = len(class_names)

    aucs = np.full((len(seeds), len(bag_sizes)), np.nan)
    f1s = np.full((len(seeds), len(bag_sizes)), np.nan)

    for si, seed in enumerate(seeds):
        for bi, bs in enumerate(bag_sizes):
            if (bs, seed) not in results:
                continue
            y_true = results[(bs, seed)]["y_true"]
            y_probs = results[(bs, seed)]["y_probs"]

            try:
                yb = label_binarize(y_true, classes=list(range(num_classes)))
                if num_classes == 2:
                    yb = np.column_stack([1 - yb, yb])
                aucs[si, bi] = roc_auc_score(yb, y_probs, average="macro", multi_class="ovr")
            except Exception:
                pass

            try:
                y_pred = y_probs.argmax(axis=1)
                f1s[si, bi] = f1_score(y_true, y_pred, average="macro")
            except Exception:
                pass

    return {
        "bag_sizes": bag_sizes,
        "seeds": seeds,
        "aucs": aucs,
        "f1s": f1s,
    }


def aggregate_topk_experiment(cv_dir: str, class_names: list[str], k_values: list[int] = None) -> dict:
    """Aggregate top-K confident predictions from many single-image runs.

    Each subdirectory in {cv_dir}/topk_experiment/seed{N}/ contains an epoch_*.npz
    with bag_size=1 predictions. For each experiment, we collect all single-image
    predictions across seeds, then select the top-K by max class probability
    and average those for the final prediction.

    Returns dict with k_values, aucs, f1s arrays.
    """
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.preprocessing import label_binarize
    from collections import defaultdict

    if k_values is None:
        k_values = [1, 3, 5, 10, 25, 50]

    topk_dir = os.path.join(cv_dir, "topk_experiment")
    if not os.path.exists(topk_dir):
        return None

    # Collect all predictions per experiment across seeds
    exp_predictions = defaultdict(list)  # exp_id -> [(y_true, y_probs, max_prob)]
    n_seeds_loaded = 0

    for d in sorted(os.listdir(topk_dir)):
        if not d.startswith("seed"):
            continue
        npz_files = glob_module.glob(os.path.join(topk_dir, d, "epoch_*.npz"))
        if not npz_files:
            continue

        data = np.load(npz_files[0], allow_pickle=True)
        exp_ids = data["experiment_ids"]
        y_true = data["y_true"]
        y_probs = data["y_probs"]
        n_seeds_loaded += 1

        for i, eid in enumerate(exp_ids):
            max_prob = float(np.max(y_probs[i]))
            exp_predictions[eid].append({
                "y_true": int(y_true[i]),
                "y_probs": y_probs[i].copy(),
                "max_prob": max_prob,
            })

    if not exp_predictions:
        return None

    print(f"Loaded {n_seeds_loaded} seeds, {len(exp_predictions)} experiments")
    n_preds = [len(v) for v in exp_predictions.values()]
    print(f"Predictions per experiment: min={min(n_preds)}, max={max(n_preds)}, mean={np.mean(n_preds):.1f}")

    num_classes = len(class_names)
    # Filter k_values to those <= available predictions
    max_k = min(n_preds)
    k_values = [k for k in k_values if k <= max_k]
    if not k_values:
        k_values = [max_k]

    # Also compute "bottom-K" and "random-K" for comparison
    strategies = ["top", "bottom", "random"]
    results = {}

    for strategy in strategies:
        aucs = np.full(len(k_values), np.nan)
        f1s = np.full(len(k_values), np.nan)

        for ki, k in enumerate(k_values):
            all_yt = []
            all_yp = []

            for eid, preds in exp_predictions.items():
                if strategy == "top":
                    selected = sorted(preds, key=lambda x: x["max_prob"], reverse=True)[:k]
                elif strategy == "bottom":
                    selected = sorted(preds, key=lambda x: x["max_prob"])[:k]
                else:  # random
                    rng = np.random.RandomState(42)
                    idx = rng.choice(len(preds), min(k, len(preds)), replace=False)
                    selected = [preds[i] for i in idx]

                avg_probs = np.mean([s["y_probs"] for s in selected], axis=0)
                all_yt.append(selected[0]["y_true"])
                all_yp.append(avg_probs)

            all_yt = np.array(all_yt, dtype=int)
            all_yp = np.array(all_yp, dtype=float)

            try:
                yb = label_binarize(all_yt, classes=list(range(num_classes)))
                if num_classes == 2:
                    yb = np.column_stack([1 - yb, yb])
                aucs[ki] = roc_auc_score(yb, all_yp, average="macro", multi_class="ovr")
            except Exception:
                pass

            try:
                y_pred = all_yp.argmax(axis=1)
                f1s[ki] = f1_score(all_yt, y_pred, average="macro")
            except Exception:
                pass

        results[strategy] = {"aucs": aucs, "f1s": f1s}

    return {
        "k_values": k_values,
        "strategies": results,
        "n_seeds": n_seeds_loaded,
        "n_experiments": len(exp_predictions),
    }


def aggregate_two_stage_experiment(cv_dir: str, class_names: list[str]) -> dict:
    """Aggregate two-stage hard-attention experiment results.

    Reads from {cv_dir}/two_stage_experiment/scan{S}_top{K}_seed{SEED}/.
    Returns dict with configs and metrics.
    """
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.preprocessing import label_binarize

    exp_dir = os.path.join(cv_dir, "two_stage_experiment")
    if not os.path.exists(exp_dir):
        return None

    num_classes = len(class_names)
    # Discover all (scan, topk, seed) directories
    raw = {}
    for d in sorted(os.listdir(exp_dir)):
        if not d.startswith("scan"):
            continue
        try:
            parts = d.split("_")
            scan = int(parts[0].replace("scan", ""))
            topk = int(parts[1].replace("top", ""))
            seed = int(parts[2].replace("seed", ""))
        except (IndexError, ValueError):
            continue

        npz_files = glob_module.glob(os.path.join(exp_dir, d, "epoch_*.npz"))
        if not npz_files:
            continue

        data = np.load(npz_files[0], allow_pickle=True)
        raw[(scan, topk, seed)] = {
            "y_true": data["y_true"],
            "y_probs": data["y_probs"],
        }

    if not raw:
        return None

    configs = sorted(set((s, k) for s, k, _ in raw.keys()))
    seeds = sorted(set(sd for _, _, sd in raw.keys()))

    rows = []
    for scan, topk in configs:
        aucs = []
        f1s = []
        for seed in seeds:
            if (scan, topk, seed) not in raw:
                continue
            y_true = raw[(scan, topk, seed)]["y_true"]
            y_probs = raw[(scan, topk, seed)]["y_probs"]

            try:
                yb = label_binarize(y_true, classes=list(range(num_classes)))
                if num_classes == 2:
                    yb = np.column_stack([1 - yb, yb])
                aucs.append(roc_auc_score(yb, y_probs, average="macro", multi_class="ovr"))
            except Exception:
                pass

            try:
                y_pred = y_probs.argmax(axis=1)
                f1s.append(f1_score(y_true, y_pred, average="macro"))
            except Exception:
                pass

        rows.append({
            "scan_size": scan,
            "top_k": topk,
            "auc_mean": np.mean(aucs) if aucs else np.nan,
            "auc_std": np.std(aucs) if aucs else np.nan,
            "f1_mean": np.mean(f1s) if f1s else np.nan,
            "f1_std": np.std(f1s) if f1s else np.nan,
            "n_seeds": len(aucs),
        })

    return {"rows": rows, "configs": configs, "seeds": seeds}


def predict_per_image_fold(
    cv_dir: str,
    fold_idx: int,
    epoch: int,
    class_config: str,
    seed: int = 403,
    save_embeddings: bool = True,
    all_experiments: bool = False,
    output_dir: str = None,
    exp_shard: int = 0,
    exp_nshards: int = 1,
    encode_chunk_size: int = 256,
) -> str:
    """Encode ALL images for a fold's held-out experiments, predict each individually.

    Saves per-experiment CSVs: image_id, prob_0, prob_1, ..., prob_N
    When save_embeddings is True, also writes a companion
    {experiment_id}.embeddings.npz holding the per-image backbone features
    (post-projection, the input to _aggregate) plus image_id / y_true / probs,
    so downstream bag-size analyses can resample without re-encoding on GPU.
    Returns output directory path.
    """
    from remo_id.inference.context import load_class_config
    from remo_id.data import RemoIDDataModule
    from remo_id.models import get_augmentation_backbone
    import torch

    class_definitions, class_names, num_classes, task_type, exclude_filters, include_filters, default_class_idx = (
        load_class_config(class_config)
    )

    fold_dirs = sorted(glob_module.glob(os.path.join(cv_dir, "repeat*_fold*")))
    model_name = "resnet18-mil"
    for fd in fold_dirs:
        for c in glob_module.glob(os.path.join(fd, "*.ckpt")):
            try:
                meta = extract_checkpoint_metadata(c)
                model_name = meta.get("model_name", "resnet18-mil")
                break
            except Exception:
                continue
        break

    backbone_name = get_augmentation_backbone(model_name)
    data_module = RemoIDDataModule(
        database_column="class_label", database_label=None, study_id=None,
        threshold=None, threshold_mode="==", negative_study_ids=[], negative_experiment_ids=[],
        batch_size=1, backbone_name=backbone_name, train=False,
        task_type=task_type, num_classes=num_classes,
        class_definitions=class_definitions, exclude_filters=exclude_filters,
        include_filters=include_filters, default_class_idx=default_class_idx,
        skip_null_labels=True, background_correction='none',
    )

    from remo_id.training.cross_validation import load_cv_splits
    splits_path = os.path.join(cv_dir, "cv_splits.yaml")
    all_repeats = load_cv_splits(splits_path, data_module)
    fold_val_paths = {}
    for repeat_idx, folds in enumerate(all_repeats):
        for fi, (train_paths, val_paths) in enumerate(folds):
            fold_val_paths[f"repeat{repeat_idx}_fold{fi}"] = val_paths

    fd = fold_dirs[fold_idx]
    fold_label = os.path.basename(fd)
    val_paths = fold_val_paths.get(fold_label, set())
    if not val_paths and not all_experiments:
        # With --all_experiments we target EVERY experiment in the config, so an
        # empty fold val set is fine (e.g. a config whose experiments aren't in
        # cv_splits, like the indeterminate-overlay encode).
        print(f"No val paths for {fold_label}")
        return None

    ckpt_path = _find_epoch_checkpoint(fd, epoch)
    if ckpt_path is None:
        print(f"No epoch={epoch} checkpoint in {fd}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_inference_seeds(seed)

    ckpt_meta = extract_checkpoint_metadata(ckpt_path)
    mn = ckpt_meta.get("model_name", model_name)

    model, device, _ = load_checkpoint(
        ckpt_path, mn, task_type=task_type, num_classes=num_classes,
        n_experiments=1, device=device,
    )
    model.eval()

    all_paths = set(data_module.data["path"].values)
    # all_experiments: run this single fold's model over EVERY experiment (not just its
    # held-out val set) — for a single-model embeddings file usable by the gallery etc.
    target_paths = all_paths if all_experiments else val_paths
    other_paths = all_paths - target_paths
    train_arg = other_paths if other_paths else target_paths
    _, target_module = data_module.split_from_paths(train_arg, target_paths)
    df = target_module.data.copy().reset_index(drop=True)

    # Optional experiment sharding for parallel array jobs: each task processes a
    # disjoint round-robin slice (exp_idx % exp_nshards == exp_shard). Order is
    # deterministic across tasks (same data_module build), so slices are disjoint.
    if exp_nshards and exp_nshards > 1:
        df = df.iloc[exp_shard::exp_nshards].copy()
        print(f"exp shard {exp_shard}/{exp_nshards}: {len(df)} experiments this task")

    out_dir = output_dir or os.path.join(cv_dir, "per_image_predictions")
    os.makedirs(out_dir, exist_ok=True)

    # Per-image encoding is independent (eval mode), so chunk size only affects
    # GPU batching/throughput, never the embeddings. Larger = faster at inference.
    max_chunk_size = encode_chunk_size

    for _, row in df.iterrows():
        exp_id = str(row["experiment_id"])
        exp_path = str(row["path"])
        y_true = int(row.get("class_label", 0))

        if exp_path not in target_module.experiment_ranges:
            continue
        start, end = target_module.experiment_ranges[exp_path]
        n_images = end - start
        if n_images == 0:
            continue

        csv_path = os.path.join(out_dir, f"{exp_id}.csv")
        emb_path = os.path.join(out_dir, f"{exp_id}.embeddings.npz")
        if os.path.exists(csv_path) and (not save_embeddings or os.path.exists(emb_path)):
            print(f"  Skipping {exp_id} (already exists)")
            continue

        all_indices = list(range(start, end))

        # Per-image inference: encode all, then classify individually.
        # For mi-Net (linear pooling): classifier(feats) directly.
        # For attention/mean/max/topk: _aggregate with bag_size=1.
        is_minet = getattr(model, 'pooling', 'attention') == 'linear'
        all_probs = []
        all_feats = [] if save_embeddings else None
        with torch.no_grad():
            for cs in range(0, n_images, max_chunk_size):
                ce = min(cs + max_chunk_size, n_images)
                # Serial reads only: concurrent zarr reads on the shared handle
                # corrupt ~1% of pixels (verified: parallel vs serial diff 0.0128
                # vs serial-vs-serial 1e-5). See reference-loo-embed-aggregate.
                x, _, _ = target_module.dataloader.get_items(all_indices[cs:ce])
                x = x.to(device)
                feats = model._encode_instances(x)
                if hasattr(model, 'projection') and model.projection is not None:
                    feats = model.projection(feats)
                # post-projection per-image embeddings (input to _aggregate), float32
                # so a later cache→sample→aggregate pass is bit-for-bit identical.
                if save_embeddings:
                    all_feats.append(feats.cpu().numpy().astype(np.float32))
                if is_minet:
                    logits = model.classifier(feats)
                else:
                    logits_list = []
                    for i in range(feats.shape[0]):
                        bag_logits, _ = model._aggregate(feats[i:i+1].unsqueeze(0))
                        logits_list.append(bag_logits)
                    logits = torch.cat(logits_list, dim=0)
                if task_type == "binary" and (logits.dim() == 1 or logits.shape[-1] == 1):
                    p = torch.sigmoid(logits.view(-1, 1))
                    probs = torch.cat([1 - p, p], dim=-1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
                del x, feats, logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        all_probs = np.concatenate(all_probs, axis=0)  # [N_images, num_classes]

        # Save CSV
        import pandas as pd
        cols = {f"prob_{c}": all_probs[:, c] for c in range(num_classes)}
        csv_df = pd.DataFrame({"image_id": range(n_images), **cols})
        csv_df["y_true"] = y_true
        csv_df["experiment_id"] = exp_id
        csv_df.to_csv(csv_path, index=False)

        # Save companion embeddings (post-projection per-image features)
        if save_embeddings:
            embeddings = np.concatenate(all_feats, axis=0)  # [N_images, D] float32
            np.savez_compressed(
                emb_path,
                embeddings=embeddings,
                image_id=np.arange(n_images),
                probs=all_probs.astype(np.float32),
                y_true=np.int64(y_true),
                experiment_id=exp_id,
                fold_idx=np.int64(fold_idx),
                class_names=np.array(class_names),
                epoch=np.int64(epoch),
            )

        print(f"  {fold_label}: exp={exp_id} class={class_names[y_true]} "
              f"n_images={n_images}" + (f" (+embeddings {embeddings.shape})"
                                        if save_embeddings else ""))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_dir


def submit_per_image_array(
    cv_dir: str,
    class_config: str,
    epoch: int,
    seed: int = 403,
    n_gpus: int = None,
    partition: str = "gpu",
) -> str:
    """Submit SLURM array job for per-image predictions, one task per fold."""
    import subprocess

    fold_dirs = sorted(glob_module.glob(os.path.join(cv_dir, "repeat*_fold*")))
    n_folds = len(fold_dirs)
    n_gpus = min(n_gpus or n_folds, n_folds)

    out_dir = os.path.join(cv_dir, "per_image_predictions")
    os.makedirs(os.path.join(out_dir, "slurm-outputs"), exist_ok=True)

    script_path = os.path.join(out_dir, "predict_array.sh")
    script = f"""#!/bin/bash
#SBATCH --job-name=per-img-pred
#SBATCH --array=0-{n_folds - 1}%{n_gpus}
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition={partition}
#SBATCH --gpus-per-node=1
#SBATCH --output={out_dir}/slurm-outputs/%A_%a.out

env | grep "^SLURM" | sort
nvidia-smi

export PYTHONUNBUFFERED=1

echo "Array task $SLURM_ARRAY_TASK_ID: fold $SLURM_ARRAY_TASK_ID"

python -m remo_id.cli workflow loo-per-image \\
    --cv_dir {cv_dir} \\
    --class_config {os.path.abspath(class_config)} \\
    --epoch {epoch} \\
    --fold_idx $SLURM_ARRAY_TASK_ID \\
    --seed {seed}
"""

    with open(script_path, "w") as f:
        f.write(script)

    print(f"Submitting per-image prediction array: {n_folds} folds across {n_gpus} GPUs")
    print(f"  Script: {script_path}")

    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"  Submitted batch job {job_id}")
        return job_id
    else:
        print(f"  sbatch failed: {result.stderr}")
        return None


def predict_bag_variance_fold(
    cv_dir: str,
    fold_idx: int,
    epoch: int,
    class_config: str,
    seed: int = 403,
    n_shuffles: int = 5,
    bag_sizes: list = None,
) -> str:
    """Encode ALL images for a single fold's held-out experiments, sweep bag sizes.

    Saves per-fold results to {cv_dir}/bag_variance/fold_{fold_idx}.npz.
    Returns path to saved file.
    """
    from remo_id.inference.context import load_class_config
    from remo_id.data import RemoIDDataModule
    from remo_id.models import get_augmentation_backbone
    import torch

    if bag_sizes is None:
        bag_sizes = [1, 5, 10, 20, 50, 100, 200, 500, 1000]

    class_definitions, class_names, num_classes, task_type, exclude_filters, include_filters, default_class_idx = (
        load_class_config(class_config)
    )

    # Extract model name
    fold_dirs = sorted(glob_module.glob(os.path.join(cv_dir, "repeat*_fold*")))
    model_name = "resnet18-mil"
    for fd in fold_dirs:
        for c in glob_module.glob(os.path.join(fd, "*.ckpt")):
            try:
                meta = extract_checkpoint_metadata(c)
                model_name = meta.get("model_name", "resnet18-mil")
                break
            except Exception:
                continue
        break

    backbone_name = get_augmentation_backbone(model_name)
    data_module = RemoIDDataModule(
        database_column="class_label", database_label=None, study_id=None,
        threshold=None, threshold_mode="==", negative_study_ids=[], negative_experiment_ids=[],
        batch_size=1, backbone_name=backbone_name, train=False,
        task_type=task_type, num_classes=num_classes,
        class_definitions=class_definitions, exclude_filters=exclude_filters,
        include_filters=include_filters, default_class_idx=default_class_idx,
        skip_null_labels=True, background_correction='none',
    )

    # Load splits and find this fold's val paths
    from remo_id.training.cross_validation import load_cv_splits
    splits_path = os.path.join(cv_dir, "cv_splits.yaml")
    all_repeats = load_cv_splits(splits_path, data_module)
    fold_val_paths = {}
    for repeat_idx, folds in enumerate(all_repeats):
        for fi, (train_paths, val_paths) in enumerate(folds):
            fold_val_paths[f"repeat{repeat_idx}_fold{fi}"] = val_paths

    fd = fold_dirs[fold_idx]
    fold_label = os.path.basename(fd)
    val_paths = fold_val_paths.get(fold_label, set())
    if not val_paths:
        print(f"No val paths for {fold_label}")
        return None

    ckpt_path = _find_epoch_checkpoint(fd, epoch)
    if ckpt_path is None:
        print(f"No epoch={epoch} checkpoint in {fd}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_inference_seeds(seed)

    ckpt_meta = extract_checkpoint_metadata(ckpt_path)
    mn = ckpt_meta.get("model_name", model_name)

    model, device, _ = load_checkpoint(
        ckpt_path, mn, task_type=task_type, num_classes=num_classes,
        n_experiments=1, device=device,
    )
    model.eval()

    # Build filtered data module for held-out experiments
    all_paths = set(data_module.data["path"].values)
    other_paths = all_paths - val_paths
    _, target_module = data_module.split_from_paths(other_paths, val_paths)
    df = target_module.data.copy()

    max_chunk_size = 64
    # Collect: list of (y_true, bag_size, true_class_prob)
    records = []

    for _, row in df.iterrows():
        exp_id = str(row["experiment_id"])
        exp_path = str(row["path"])
        y_true = int(row.get("class_label", 0))
        if y_true < 0 or y_true >= num_classes:
            continue
        if exp_path not in target_module.experiment_ranges:
            continue

        start, end = target_module.experiment_ranges[exp_path]
        n_images = end - start
        if n_images == 0:
            continue

        all_indices = list(range(start, end))

        # Reuse cached per-image embeddings (post-projection) from the per-image
        # inference pass if available -- skips the expensive backbone encode, so
        # many shuffles/seeds become cheap. Falls back to GPU encoding otherwise.
        emb_path = os.path.join(cv_dir, "per_image_predictions",
                                f"{exp_id}.embeddings.npz")
        all_feats = None
        if os.path.exists(emb_path):
            cached = np.load(emb_path, allow_pickle=True)["embeddings"]
            if cached.shape[0] == n_images:
                all_feats = torch.from_numpy(cached.astype(np.float32))
                print(f"  {fold_label}: exp={exp_id} using cached embeddings "
                      f"{tuple(all_feats.shape)} (no encode)")
            else:
                print(f"  {fold_label}: exp={exp_id} cached embeddings size "
                      f"{cached.shape[0]} != {n_images}; re-encoding")

        if all_feats is None:
            # Encode all images once (post-projection features)
            with torch.no_grad():
                feats_chunks = []
                for cs in range(0, n_images, max_chunk_size):
                    ce = min(cs + max_chunk_size, n_images)
                    x, _, _ = target_module.dataloader.get_items(all_indices[cs:ce])
                    x = x.to(device)
                    feats = model._encode_instances(x).cpu()
                    feats_chunks.append(feats)
                    del x
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                all_feats = torch.cat(feats_chunks, dim=0)
                if hasattr(model, 'projection') and model.projection is not None:
                    all_feats = model.projection(all_feats.to(device)).cpu()

        # Sweep bag sizes with multiple shuffles
        rng = np.random.RandomState(seed + hash(exp_id) % 10000)
        for _ in range(n_shuffles):
            perm = rng.permutation(len(all_feats))
            shuffled = all_feats[perm]
            for bs in bag_sizes:
                for chunk_start in range(0, len(shuffled), bs):
                    chunk = shuffled[chunk_start:chunk_start + bs]
                    if len(chunk) == 0:
                        continue
                    with torch.no_grad():
                        chunk_feats = chunk.unsqueeze(0).to(device)
                        logits, _ = model._aggregate(chunk_feats)
                        if logits.dim() > 1:
                            probs = torch.softmax(logits[0], dim=-1)
                        else:
                            probs = torch.softmax(logits, dim=-1)
                        true_class_prob = float(probs[y_true].cpu())
                    records.append((y_true, bs, true_class_prob, exp_id))

        print(f"  {fold_label}: exp={exp_id} class={class_names[y_true]} "
              f"n_images={n_images} records={len(records)}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save
    out_dir = os.path.join(cv_dir, "bag_variance")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"fold_{fold_idx}.npz")
    if records:
        arr = np.array(records, dtype=[("y_true", int), ("bag_size", int),
                                       ("prob", float), ("experiment_id", "U20")])
        np.savez(out_path, records=arr, class_names=np.array(class_names),
                 bag_sizes=np.array(bag_sizes))
    else:
        np.savez(out_path, records=np.array([]), class_names=np.array(class_names),
                 bag_sizes=np.array(bag_sizes))

    print(f"  Saved {len(records)} records to {out_path}")
    return out_path


def submit_bag_variance_array(
    cv_dir: str,
    class_config: str,
    epoch: int,
    seed: int = 403,
    n_shuffles: int = 5,
    n_gpus: int = None,
    partition: str = "gpu",
) -> str:
    """Submit SLURM array job: one task per fold for bag variance analysis.

    Returns job ID.
    """
    import subprocess

    fold_dirs = sorted(glob_module.glob(os.path.join(cv_dir, "repeat*_fold*")))
    n_folds = len(fold_dirs)
    n_gpus = min(n_gpus or n_folds, n_folds)

    out_dir = os.path.join(cv_dir, "bag_variance")
    os.makedirs(os.path.join(out_dir, "slurm-outputs"), exist_ok=True)

    script_path = os.path.join(out_dir, "variance_array.sh")
    script = f"""#!/bin/bash
#SBATCH --job-name=bag-variance
#SBATCH --array=0-{n_folds - 1}%{n_gpus}
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition={partition}
#SBATCH --gpus-per-node=1
#SBATCH --output={out_dir}/slurm-outputs/%A_%a.out

env | grep "^SLURM" | sort
nvidia-smi

export PYTHONUNBUFFERED=1

echo "Array task $SLURM_ARRAY_TASK_ID: fold $SLURM_ARRAY_TASK_ID"

python -m remo_id.cli workflow loo-bag-variance \\
    --cv_dir {cv_dir} \\
    --class_config {os.path.abspath(class_config)} \\
    --epoch {epoch} \\
    --fold_idx $SLURM_ARRAY_TASK_ID \\
    --seed {seed} \\
    --n_shuffles {n_shuffles}
"""

    with open(script_path, "w") as f:
        f.write(script)

    print(f"Submitting bag variance array: {n_folds} folds across {n_gpus} GPUs")
    print(f"  Script: {script_path}")

    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"  Submitted batch job {job_id}")
        return job_id
    else:
        print(f"  sbatch failed: {result.stderr}")
        return None


def load_bag_variance_results(cv_dir: str) -> dict:
    """Load and aggregate bag variance results from all fold .npz files.

    Returns {class_idx: {bag_size: [true_class_prob, ...]}} or None if not found.
    """
    var_dir = os.path.join(cv_dir, "bag_variance")
    if not os.path.exists(var_dir):
        return None

    npz_files = sorted(glob_module.glob(os.path.join(var_dir, "fold_*.npz")))
    if not npz_files:
        return None

    # Load first file for metadata
    first = np.load(npz_files[0], allow_pickle=True)
    class_names = list(first["class_names"])
    bag_sizes = list(first["bag_sizes"])
    num_classes = len(class_names)

    per_class_probs = {ci: {bs: [] for bs in bag_sizes} for ci in range(num_classes)}

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        records = data["records"]
        if records.size == 0:
            continue
        # records is a structured array; access by field name (experiment_id
        # field added later, so positional unpacking is avoided).
        for rec in records:
            y_true = int(rec["y_true"])
            bs = int(rec["bag_size"])
            if y_true < num_classes and bs in per_class_probs.get(y_true, {}):
                per_class_probs[y_true][bs].append(float(rec["prob"]))

    # Check we got data
    total = sum(len(v) for ci in per_class_probs for v in per_class_probs[ci].values())
    if total == 0:
        return None

    return per_class_probs


def predict_position_sweep_fold(
    cv_dir: str,
    fold_idx: int,
    epoch: int,
    class_config: str,
    seed: int = 403,
    bag_size: int = 500,
    n_draws: int = 10,
    window_step: int = 500,
    window_max: int = 20000,
    output_dir: str = None,
) -> str:
    """Sweep IMAGE POSITION for one fold's held-out experiments.

    For each held-out experiment, the cached per-image embeddings (ordered by
    acquisition index 0..N-1) are windowed two ways -- ``first N`` = indices
    [0, N) and ``last N`` = indices [n-N, n) -- for N in
    ``range(window_step, window_max+1, window_step)``. Within each window region
    we draw ``n_draws`` random bags of ``bag_size`` images (without replacement,
    or the whole region when it is smaller than bag_size), run ONLY the
    aggregator head (``model._aggregate``), and record the softmax probability
    vector of each draw. The largest window (whole movie) is the per-experiment
    full reference.

    Mirrors ``predict_bag_variance_fold``: same cached-embedding reuse, same
    held-out fold resolution from ``cv_splits.yaml`` (the npz files predate the
    ``fold_idx`` field, so it is NOT trusted), same checkpoint per fold.

    Saves per-fold results to ``{cv_dir}/position_sweep/bag{bag_size}/fold_{idx}.npz``.
    Returns the saved path.
    """
    from remo_id.inference.context import load_class_config
    from remo_id.data import RemoIDDataModule
    from remo_id.models import get_augmentation_backbone
    import torch

    class_definitions, class_names, num_classes, task_type, exclude_filters, include_filters, default_class_idx = (
        load_class_config(class_config)
    )

    window_sizes = list(range(window_step, window_max + 1, window_step))

    # Model name from any checkpoint
    fold_dirs = sorted(glob_module.glob(os.path.join(cv_dir, "repeat*_fold*")))
    model_name = "resnet18-mil"
    for fd in fold_dirs:
        for c in glob_module.glob(os.path.join(fd, "*.ckpt")):
            try:
                model_name = extract_checkpoint_metadata(c).get("model_name", "resnet18-mil")
                break
            except Exception:
                continue
        break

    backbone_name = get_augmentation_backbone(model_name)
    data_module = RemoIDDataModule(
        database_column="class_label", database_label=None, study_id=None,
        threshold=None, threshold_mode="==", negative_study_ids=[], negative_experiment_ids=[],
        batch_size=1, backbone_name=backbone_name, train=False,
        task_type=task_type, num_classes=num_classes,
        class_definitions=class_definitions, exclude_filters=exclude_filters,
        include_filters=include_filters, default_class_idx=default_class_idx,
        skip_null_labels=True, background_correction='none',
    )

    splits_path = os.path.join(cv_dir, "cv_splits.yaml")
    all_repeats = load_cv_splits(splits_path, data_module)
    fold_val_paths = {}
    for repeat_idx, folds in enumerate(all_repeats):
        for fi, (train_paths, val_paths) in enumerate(folds):
            fold_val_paths[f"repeat{repeat_idx}_fold{fi}"] = val_paths

    fd = fold_dirs[fold_idx]
    fold_label = os.path.basename(fd)
    val_paths = fold_val_paths.get(fold_label, set())
    if not val_paths:
        print(f"No val paths for {fold_label}")
        return None

    ckpt_path = _find_epoch_checkpoint(fd, epoch)
    if ckpt_path is None:
        print(f"No epoch={epoch} checkpoint in {fd}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_inference_seeds(seed)

    ckpt_meta = extract_checkpoint_metadata(ckpt_path)
    mn = ckpt_meta.get("model_name", model_name)
    model, device, _ = load_checkpoint(
        ckpt_path, mn, task_type=task_type, num_classes=num_classes,
        n_experiments=1, device=device,
    )
    model.eval()

    all_paths = set(data_module.data["path"].values)
    other_paths = all_paths - val_paths
    _, target_module = data_module.split_from_paths(other_paths, val_paths)
    df = target_module.data.copy()

    max_chunk_size = 64
    # records: one row per (exp, direction, N, draw)
    rec_exp, rec_ytrue, rec_dir, rec_N, rec_draw = [], [], [], [], []
    rec_probs = []

    for _, row in df.iterrows():
        exp_id = str(row["experiment_id"])
        exp_path = str(row["path"])
        y_true = int(row.get("class_label", 0))
        if y_true < 0 or y_true >= num_classes:
            continue
        if exp_path not in target_module.experiment_ranges:
            continue
        start, end = target_module.experiment_ranges[exp_path]
        n_images = end - start
        if n_images == 0:
            continue
        all_indices = list(range(start, end))

        # Reuse cached per-image embeddings (post-projection) when present.
        emb_path = os.path.join(cv_dir, "per_image_predictions",
                                f"{exp_id}.embeddings.npz")
        all_feats = None
        if os.path.exists(emb_path):
            cached = np.load(emb_path, allow_pickle=True)["embeddings"]
            if cached.shape[0] == n_images:
                all_feats = torch.from_numpy(cached.astype(np.float32))
                print(f"  {fold_label}: exp={exp_id} cached embeddings "
                      f"{tuple(all_feats.shape)} (no encode)")
            else:
                print(f"  {fold_label}: exp={exp_id} cached size "
                      f"{cached.shape[0]} != {n_images}; re-encoding")

        if all_feats is None:
            with torch.no_grad():
                feats_chunks = []
                for cs in range(0, n_images, max_chunk_size):
                    ce = min(cs + max_chunk_size, n_images)
                    x, _, _ = target_module.dataloader.get_items(all_indices[cs:ce])
                    x = x.to(device)
                    feats = model._encode_instances(x).cpu()
                    feats_chunks.append(feats)
                    del x
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                all_feats = torch.cat(feats_chunks, dim=0)
                if hasattr(model, 'projection') and model.projection is not None:
                    all_feats = model.projection(all_feats.to(device)).cpu()

        # Effective windows for this experiment: drop N > n_images, but always
        # include n_images itself (the whole-movie reference). Each direction's
        # largest window therefore equals the full bag.
        eff = sorted({min(N, n_images) for N in window_sizes} | {n_images})

        for direction in ("first", "last"):
            for N in eff:
                if direction == "first":
                    region = np.arange(0, N)
                else:
                    region = np.arange(n_images - N, n_images)
                k = min(bag_size, N)
                for draw in range(n_draws):
                    rng = np.random.RandomState(
                        (seed + 1009 * hash(exp_id) % 100000
                         + 131 * (0 if direction == "first" else 1)
                         + 17 * N + draw) % (2**31 - 1)
                    )
                    pick = rng.choice(region, size=k, replace=False)
                    chunk = all_feats[pick].unsqueeze(0).to(device)  # [1, k, D]
                    with torch.no_grad():
                        logits, _ = model._aggregate(chunk)
                        logits0 = logits[0] if logits.dim() > 1 else logits
                        probs = torch.softmax(logits0, dim=-1).cpu().numpy()
                    rec_exp.append(exp_id)
                    rec_ytrue.append(y_true)
                    rec_dir.append(direction)
                    rec_N.append(int(N))
                    rec_draw.append(draw)
                    rec_probs.append(probs.astype(np.float32))

        print(f"  {fold_label}: exp={exp_id} class={class_names[y_true]} "
              f"n_images={n_images} windows={len(eff)} rows so far={len(rec_exp)}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_root = output_dir or os.path.join(cv_dir, "position_sweep", f"bag{bag_size}")
    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, f"fold_{fold_idx}.npz")
    np.savez(
        out_path,
        experiment_id=np.array(rec_exp, dtype="U20"),
        y_true=np.array(rec_ytrue, dtype=int),
        direction=np.array(rec_dir, dtype="U8"),
        N=np.array(rec_N, dtype=int),
        draw=np.array(rec_draw, dtype=int),
        probs=(np.stack(rec_probs) if rec_probs
               else np.zeros((0, num_classes), dtype=np.float32)),
        class_names=np.array(class_names),
        bag_size=np.int64(bag_size),
        window_sizes=np.array(window_sizes),
    )
    print(f"  Saved {len(rec_exp)} rows to {out_path}")
    return out_path


def submit_position_sweep_array(
    cv_dir: str,
    class_config: str,
    epoch: int,
    seed: int = 403,
    bag_size: int = 500,
    n_draws: int = 10,
    window_step: int = 500,
    window_max: int = 20000,
    n_gpus: int = None,
    partition: str = "gpu",
) -> str:
    """Submit SLURM array job: one task per fold for the position sweep."""
    import subprocess

    fold_dirs = sorted(glob_module.glob(os.path.join(cv_dir, "repeat*_fold*")))
    n_folds = len(fold_dirs)
    n_gpus = min(n_gpus or n_folds, n_folds)

    out_dir = os.path.join(cv_dir, "position_sweep", f"bag{bag_size}")
    os.makedirs(os.path.join(out_dir, "slurm-outputs"), exist_ok=True)

    script_path = os.path.join(out_dir, "position_array.sh")
    script = f"""#!/bin/bash
#SBATCH --job-name=pos-sweep
#SBATCH --array=0-{n_folds - 1}%{n_gpus}
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition={partition}
#SBATCH --gpus-per-node=1
#SBATCH --output={out_dir}/slurm-outputs/%A_%a.out

env | grep "^SLURM" | sort
nvidia-smi

export PYTHONUNBUFFERED=1

echo "Array task $SLURM_ARRAY_TASK_ID: fold $SLURM_ARRAY_TASK_ID"

python -m remo_id.cli workflow loo-position-sweep \\
    --cv_dir {cv_dir} \\
    --class_config {os.path.abspath(class_config)} \\
    --epoch {epoch} \\
    --fold_idx $SLURM_ARRAY_TASK_ID \\
    --seed {seed} \\
    --bag_size {bag_size} \\
    --n_draws {n_draws} \\
    --window_step {window_step} \\
    --window_max {window_max}
"""
    with open(script_path, "w") as f:
        f.write(script)

    print(f"Submitting position sweep: {n_folds} folds across {n_gpus} GPUs (bag {bag_size})")
    print(f"  Script: {script_path}")

    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"  Submitted batch job {job_id}")
        return job_id
    else:
        print(f"  sbatch failed: {result.stderr}")
        return None


def make_position_sweep_report(
    cv_dir: str,
    bag_size: int = 500,
    output: str = None,
    exclude_classes: list = None,
) -> str:
    """Pool position-sweep fold npz files, score each window, and plot.

    For every (direction, N) the per-experiment prediction is the mean softmax
    over the ``n_draws`` random bags. We report, pooled across all held-out
    experiments:

    * ``variability``  -- std of P(true class) across the draws, averaged over
      experiments (how stable a single bag is at that window).
    * ``corr_to_full`` -- agreement of the window's argmax with the whole-movie
      argmax (fraction of experiments), plus mean L1 distance between the window
      mean-prob vector and the whole-movie mean-prob vector.
    * ``corr_to_true`` -- accuracy (argmax vs y_true) and macro one-vs-rest AUC
      across experiments.

    Writes ``position_sweep_metrics.csv`` and ``position_sweep.png`` under
    ``{cv_dir}/analysis/position_sweep/bag{bag_size}/``. Returns the output dir.
    """
    import pandas as pd
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score, accuracy_score
    from remo_id.reports import style
    import matplotlib.pyplot as plt

    sweep_dir = os.path.join(cv_dir, "position_sweep", f"bag{bag_size}")
    npz_files = sorted(glob_module.glob(os.path.join(sweep_dir, "fold_*.npz")))
    if not npz_files:
        print(f"No position-sweep results in {sweep_dir}. Run loo-position-sweep first.")
        return None

    class_names = None
    grid = set()  # regular window grid (multiples of window_step)
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # dir->exp->N->[probvecs]
    ytrue_of = {}
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        if d["probs"].shape[0] == 0:
            continue
        if class_names is None:
            class_names = list(d["class_names"])
        grid.update(int(x) for x in d["window_sizes"])
        for i in range(d["probs"].shape[0]):
            direction = str(d["direction"][i])
            exp = str(d["experiment_id"][i])
            N = int(d["N"][i])
            raw[direction][exp][N].append(d["probs"][i])
            ytrue_of[exp] = int(d["y_true"][i])

    num_classes = len(class_names)
    excl_idx = {class_names.index(c) for c in (exclude_classes or []) if c in class_names}
    suffix = ("_excl_" + "_".join(sorted(exclude_classes))) if exclude_classes else ""

    # Per-experiment full reference = largest N available for that exp/direction.
    rows = []
    # collect per (direction, N) pooled arrays
    pooled = defaultdict(lambda: defaultdict(lambda: {"yt": [], "mean": [],
                                                      "std": [], "argmatch_full": [],
                                                      "l1_full": []}))
    for direction in raw:
        for exp in raw[direction]:
            yt = ytrue_of[exp]
            if yt in excl_idx:
                continue
            Ns = sorted(raw[direction][exp].keys())
            full_N = Ns[-1]
            full_mean = np.mean(raw[direction][exp][full_N], axis=0)
            full_arg = int(np.argmax(full_mean))
            for N in Ns:
                # Pool only regular-grid windows; each experiment's exact image
                # count (an off-grid N) is kept solely as its full reference.
                if N not in grid:
                    continue
                draws = np.stack(raw[direction][exp][N])  # [n_draws, C]
                mean_prob = draws.mean(axis=0)
                std_true = float(draws[:, yt].std())
                p = pooled[direction][N]
                p["yt"].append(yt)
                p["mean"].append(mean_prob)
                p["std"].append(std_true)
                p["argmatch_full"].append(int(int(np.argmax(mean_prob)) == full_arg))
                p["l1_full"].append(float(np.abs(mean_prob - full_mean).sum()))

    # Aggregate pooled -> metrics per (direction, N)
    for direction in pooled:
        for N in sorted(pooled[direction].keys()):
            p = pooled[direction][N]
            yt = np.array(p["yt"], dtype=int)
            mean = np.array(p["mean"], dtype=float)
            y_pred = mean.argmax(axis=1)
            acc = accuracy_score(yt, y_pred)
            # Macro one-vs-rest AUC over the classes actually present (so it is
            # well defined when a class -- e.g. healthy -- has been excluded).
            present = [c for c in range(num_classes) if (yt == c).any() and (yt != c).any()]
            try:
                aucs = [roc_auc_score((yt == c).astype(int), mean[:, c]) for c in present]
                auc = float(np.mean(aucs)) if aucs else np.nan
            except Exception:
                auc = np.nan
            rows.append({
                "direction": direction,
                "N": N,
                "n_experiments": len(yt),
                "macro_auc": auc,
                "accuracy": acc,
                "variability_std_true": float(np.mean(p["std"])),
                "argmatch_full": float(np.mean(p["argmatch_full"])),
                "l1_to_full": float(np.mean(p["l1_full"])),
            })

    df = pd.DataFrame(rows).sort_values(["direction", "N"]).reset_index(drop=True)
    out = output or os.path.join(cv_dir, "analysis", "position_sweep", f"bag{bag_size}")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"position_sweep_metrics{suffix}.csv"), index=False)

    # ---- Figure: 2x2 panels vs N, first (solid) vs last (dashed) ----
    style.apply()
    fig, axes = style.figure(ncols=2, nrows=2, width="double", aspect=0.7)
    axes = np.atleast_1d(axes).ravel()
    panels = [
        ("macro_auc", "Macro AUC (vs true label)"),
        ("accuracy", "Accuracy (vs true label)"),
        ("variability_std_true", "Variability  (std of P[true] across bags)"),
        ("argmatch_full", "Argmax agreement with whole movie"),
    ]
    dir_style = {"first": dict(color=style.WONG[2], ls="-", marker="o"),
                 "last": dict(color=style.ACCENT, ls="--", marker="s")}
    for ax, (col, title) in zip(axes, panels):
        for direction in ("first", "last"):
            sub = df[df["direction"] == direction]
            if sub.empty:
                continue
            ax.plot(sub["N"], sub[col], label=f"{direction} N",
                    markersize=3, lw=1.2, **dir_style[direction])
        ax.set_xlabel("Number of images in window (N)")
        ax.set_ylabel(title, fontsize=7)
        style.range_frame(ax)
    axes[0].legend(frameon=False, fontsize=7)
    style.panel_labels(axes)
    excl_note = f", excl {', '.join(exclude_classes)}" if exclude_classes else ""
    fig.suptitle(
        f"Position sweep — resnet50-mil LOO (bag {bag_size}, "
        f"{df['n_experiments'].max()} exps{excl_note})",
        fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"position_sweep{suffix}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Position sweep report ({bag_size=}, excl={exclude_classes}):")
    print(df.to_string(index=False))
    print(f"\nSaved to {out}")
    return out


def make_position_sweep_per_class_report(
    cv_dir: str,
    bag_size: int = 500,
    output: str = None,
) -> str:
    """Per-class breakdown of the position sweep (same three axes, split by
    true class). Pools the same fold npz files as ``make_position_sweep_report``
    but reports, for each class c and each (direction, N):

    * ``auc_ovr``        -- one-vs-rest AUC for class c (label = y_true==c,
      score = window mean P(c)), over ALL pooled experiments.
    * ``recall``         -- fraction of class-c experiments whose window argmax
      equals c, and ``mean_P_true`` = mean window P(c) over class-c exps.
    * ``argmatch_full``  -- argmax agreement with the whole-movie call, averaged
      over class-c experiments (+ ``l1_to_full``).
    * ``variability``    -- std of P(c) across the draws, averaged over class-c
      experiments.

    Writes ``position_sweep_per_class.csv`` and ``position_sweep_per_class.png``
    (rows = class, cols = AUC / agreement-with-full / variability; first solid,
    last dashed) under the same output dir. Returns the output dir.
    """
    import pandas as pd
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score
    from remo_id.reports import style
    import matplotlib.pyplot as plt

    sweep_dir = os.path.join(cv_dir, "position_sweep", f"bag{bag_size}")
    npz_files = sorted(glob_module.glob(os.path.join(sweep_dir, "fold_*.npz")))
    if not npz_files:
        print(f"No position-sweep results in {sweep_dir}. Run loo-position-sweep first.")
        return None

    class_names = None
    grid = set()
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # dir->exp->N->[probvecs]
    ytrue_of = {}
    for f in npz_files:
        d = np.load(f, allow_pickle=True)
        if d["probs"].shape[0] == 0:
            continue
        if class_names is None:
            class_names = list(d["class_names"])
        grid.update(int(x) for x in d["window_sizes"])
        for i in range(d["probs"].shape[0]):
            direction = str(d["direction"][i])
            exp = str(d["experiment_id"][i])
            N = int(d["N"][i])
            raw[direction][exp][N].append(d["probs"][i])
            ytrue_of[exp] = int(d["y_true"][i])

    num_classes = len(class_names)

    # Per (direction, N): collect per-experiment (yt, mean prob, std P(true),
    # argmatch-to-full, l1-to-full), then slice by class.
    pooled = defaultdict(lambda: defaultdict(lambda: {
        "yt": [], "mean": [], "std_true": [], "argmatch_full": [], "l1_full": []}))
    for direction in raw:
        for exp in raw[direction]:
            Ns = sorted(raw[direction][exp].keys())
            full_N = Ns[-1]
            full_mean = np.mean(raw[direction][exp][full_N], axis=0)
            full_arg = int(np.argmax(full_mean))
            yt = ytrue_of[exp]
            for N in Ns:
                if N not in grid:
                    continue
                draws = np.stack(raw[direction][exp][N])
                mean_prob = draws.mean(axis=0)
                p = pooled[direction][N]
                p["yt"].append(yt)
                p["mean"].append(mean_prob)
                p["std_true"].append(float(draws[:, yt].std()))
                p["argmatch_full"].append(int(int(np.argmax(mean_prob)) == full_arg))
                p["l1_full"].append(float(np.abs(mean_prob - full_mean).sum()))

    rows = []
    for direction in pooled:
        for N in sorted(pooled[direction].keys()):
            p = pooled[direction][N]
            yt = np.array(p["yt"], dtype=int)
            mean = np.array(p["mean"], dtype=float)
            std_true = np.array(p["std_true"], dtype=float)
            argm = np.array(p["argmatch_full"], dtype=float)
            l1 = np.array(p["l1_full"], dtype=float)
            pred = mean.argmax(axis=1)
            for c in range(num_classes):
                cmask = yt == c
                n_c = int(cmask.sum())
                try:
                    auc = roc_auc_score((yt == c).astype(int), mean[:, c])
                except Exception:
                    auc = np.nan
                rows.append({
                    "direction": direction, "N": N, "class": class_names[c],
                    "class_idx": c, "n_class_exps": n_c,
                    "auc_ovr": auc,
                    "recall": float(np.mean(pred[cmask] == c)) if n_c else np.nan,
                    "mean_P_true": float(np.mean(mean[cmask, c])) if n_c else np.nan,
                    "argmatch_full": float(np.mean(argm[cmask])) if n_c else np.nan,
                    "l1_to_full": float(np.mean(l1[cmask])) if n_c else np.nan,
                    "variability_std_true": float(np.mean(std_true[cmask])) if n_c else np.nan,
                })

    df = pd.DataFrame(rows).sort_values(["class_idx", "direction", "N"]).reset_index(drop=True)
    out = output or os.path.join(cv_dir, "analysis", "position_sweep", f"bag{bag_size}")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "position_sweep_per_class.csv"), index=False)

    # ---- Figure: rows = class, cols = {OvR AUC, agreement-with-full, variability} ----
    style.apply()
    metric_cols = [
        ("auc_ovr", "One-vs-rest AUC"),
        ("argmatch_full", "Argmax agreement w/ whole movie"),
        ("variability_std_true", "Variability (std P[true])"),
    ]
    fig, axes = style.figure(ncols=3, nrows=num_classes, width="single", aspect=0.8)
    axes = np.atleast_2d(axes)
    dir_style = {"first": dict(color=style.WONG[2], ls="-", marker="o"),
                 "last": dict(color=style.ACCENT, ls="--", marker="s")}
    for ri, c in enumerate(range(num_classes)):
        for ci, (col, title) in enumerate(metric_cols):
            ax = axes[ri, ci]
            for direction in ("first", "last"):
                sub = df[(df["direction"] == direction) & (df["class_idx"] == c)]
                if sub.empty:
                    continue
                ax.plot(sub["N"], sub[col], label=f"{direction} N",
                        markersize=2.5, lw=1.1, **dir_style[direction])
            if ri == num_classes - 1:
                ax.set_xlabel("Images in window (N)")
            if ri == 0:
                ax.set_title(title, fontsize=7)
            if ci == 0:
                n_c = int(df[df["class_idx"] == c]["n_class_exps"].max())
                ax.set_ylabel(f"{class_names[c]}  (n={n_c})", fontsize=7)
            style.range_frame(ax)
    axes[0, 0].legend(frameon=False, fontsize=6)
    style.panel_labels(axes)
    fig.suptitle(
        f"Position sweep per class — resnet50-mil LOO (bag {bag_size})", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "position_sweep_per_class.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Position sweep PER-CLASS report ({bag_size=}):")
    # Print a compact view at a few Ns
    show = df[df["N"].isin([500, 2000, 5000, 10000, 20000])]
    print(show.to_string(index=False))
    print(f"\nSaved to {out}")
    return out


def load_epoch_predictions(cv_dir: str) -> dict:
    """Load all saved epoch prediction .npz files.

    Returns {epoch: {"y_true": ..., "y_probs": ..., "y_pred": ...,
                      "experiment_ids": ..., "study_ids": ...,
                      "folds": ..., "repeats": ...}}
    """
    pred_dir = os.path.join(cv_dir, "predictions")
    results = {}

    for npz_path in sorted(glob_module.glob(os.path.join(pred_dir, "epoch_*.npz"))):
        data = np.load(npz_path, allow_pickle=True)
        epoch = int(data["epoch"])
        results[epoch] = {
            "y_true": data["y_true"],
            "y_probs": data["y_probs"],
            "y_pred": data["y_pred"],
            "experiment_ids": data["experiment_ids"],
            "study_ids": data["study_ids"],
            "folds": data["folds"],
            "repeats": data["repeats"],
        }
        if "class_names" in data:
            results[epoch]["class_names"] = list(data["class_names"])

    return results
