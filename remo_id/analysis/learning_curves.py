"""Learning curve analysis — performance vs training set size.

Trains models on increasing fractions of the data, evaluates on held-out
validation, and plots metric trajectories. Designed for SLURM array execution.
"""

import os
import numpy as np
import pandas as pd


def run_learning_curve(
    data_module,
    fractions: list = None,
    n_seeds: int = 3,
    base_seed: int = 403,
    train_kwargs: dict = None,
) -> pd.DataFrame:
    """Run learning curve experiment: train on subsets, evaluate on fixed val.

    Args:
        data_module: Full (unsplit) RemoIDDataModule
        fractions: List of training fraction values (e.g. [0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
        n_seeds: Number of random seeds per fraction
        base_seed: Base seed for reproducibility
        train_kwargs: Dict of training kwargs for run_single_fold()

    Returns:
        DataFrame with columns: fraction, seed, metric_name, metric_value
    """
    from remo_id.training.train import run_single_fold, build_model_kwargs_from_config
    from remo_id.training.config import TrainingConfig
    from remo_id.models import get_model
    from lightning.pytorch import seed_everything
    import random

    if fractions is None:
        fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    train_kwargs = train_kwargs or {}
    model_name = train_kwargs.get("model", "resnet18")
    task_type = train_kwargs.get("task_type", "binary")
    num_classes = train_kwargs.get("num_classes", 2)
    save_dir = train_kwargs.get("save_dir", "./learning_curve")

    # First, create a fixed train/val split
    full_train, val_module = data_module.split_train_val(
        val_ratio=0.2, split_by_sample_id=True
    )

    all_train_paths = list(full_train.experiment_ranges.keys())
    results = []

    # Group training paths by label for stratified subsampling
    from collections import defaultdict
    label_groups = defaultdict(list)
    for path in all_train_paths:
        label = full_train.experiment_labels[path]
        label_groups[label].append(path)

    for frac in fractions:
        for seed_idx in range(n_seeds):
            seed = base_seed + seed_idx * 100
            seed_everything(seed, workers=True)

            # Stratified subsample: preserve class proportions, min 1 per class
            n_select = max(2, int(len(all_train_paths) * frac))
            rng = random.Random(seed)
            selected_paths = set()
            remaining = n_select
            for label in sorted(label_groups.keys()):
                paths = label_groups[label][:]
                n_class = max(1, round(len(paths) / len(all_train_paths) * n_select))
                n_class = min(n_class, len(paths), remaining)
                rng.shuffle(paths)
                selected_paths.update(paths[:n_class])
                remaining -= n_class
            # Fill any remainder from the full pool (due to rounding)
            if len(selected_paths) < n_select:
                unselected = [p for p in all_train_paths if p not in selected_paths]
                rng.shuffle(unselected)
                selected_paths.update(unselected[:n_select - len(selected_paths)])

            # Create subsample data module
            sub_train, _ = data_module.split_from_paths(selected_paths, set())
            # Reuse the fixed val module

            fold_label = f"frac{frac:.2f}_seed{seed}"
            fold_dir = os.path.join(save_dir, "learning_curve", fold_label)
            os.makedirs(fold_dir, exist_ok=True)

            # Build TrainingConfig and model kwargs
            lc_config = TrainingConfig(
                model_name=model_name,
                task_type=task_type,
                num_classes=num_classes,
                batch_size=train_kwargs.get("batch_size", 200),
                n_experiments=train_kwargs.get("n_experiments", 8),
                num_batches=train_kwargs.get("num_batches", 100),
                lr=train_kwargs.get("lr", 1e-4),
                lr_schedule=train_kwargs.get("lr_schedule", "cosine"),
                warmup_epochs=train_kwargs.get("warmup_epochs", 5),
                max_epochs=train_kwargs.get("max_epochs", 100),
                weight_decay=train_kwargs.get("weight_decay", 0.01),
                attn_dim=train_kwargs.get("attn_dim", 128),
                pretrained=train_kwargs.get("pretrained", True),
                dropout=train_kwargs.get("dropout", 0.3),
                finetune_layers=train_kwargs.get("finetune_layers", "layer3,layer4"),
                num_heads=train_kwargs.get("num_heads", 1),
                projection_dim=train_kwargs.get("projection_dim"),
                pooling=train_kwargs.get("pooling", "attention"),
                classifier_hidden=train_kwargs.get("classifier_hidden"),
                save_dir=fold_dir,
                seed=seed,
                early_stopping_patience=train_kwargs.get("early_stopping_patience", 15),
                precision=train_kwargs.get("precision", "auto"),
            )

            model_kwargs = build_model_kwargs_from_config(lc_config)
            model_instance = get_model(model_name, **model_kwargs)
            model_cls = model_instance.__class__

            try:
                _, trainer, _, _ = run_single_fold(
                    config=lc_config,
                    model_instance=model_instance,
                    model_cls=model_cls,
                    model_kwargs=model_kwargs,
                    train_module=sub_train,
                    val_module=val_module,
                    fold_label=fold_label,
                )

                best_loss = float(trainer.checkpoint_callback.best_model_score or np.nan)
                results.append({
                    "fraction": frac,
                    "n_experiments": n_select,
                    "seed": seed,
                    "best_val_loss": best_loss,
                })
                print(f"  {fold_label}: val_loss={best_loss:.4f} ({n_select} experiments)")

            except Exception as e:
                print(f"  {fold_label}: FAILED - {e}")
                results.append({
                    "fraction": frac,
                    "n_experiments": n_select,
                    "seed": seed,
                    "best_val_loss": np.nan,
                })

    return pd.DataFrame(results)


def generate_learning_curve_sbatch(
    save_dir: str,
    fractions: list,
    n_seeds: int,
    train_cmd: str,
    partition: str = "gpu",
) -> str:
    """Generate sbatch --array script for learning curve jobs.

    Args:
        save_dir: Output directory
        fractions: List of data fractions
        n_seeds: Seeds per fraction
        train_cmd: Base remo-id train command
        partition: SLURM partition

    Returns:
        Path to generated script
    """
    total_jobs = len(fractions) * n_seeds
    fractions_str = ",".join(f"{f:.2f}" for f in fractions)
    script_path = os.path.join(save_dir, "learning_curve_array.sh")

    script = f"""#!/bin/bash
#SBATCH --job-name=remo-id-lc
#SBATCH --array=0-{total_jobs - 1}
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=14
#SBATCH --mem=240G
#SBATCH --partition={partition}
#SBATCH --gpus=1
#SBATCH --output={save_dir}/slurm-outputs/%A_%a.out

FRACTIONS=({fractions_str})
N_FRACTIONS={len(fractions)}

FRAC_IDX=$((SLURM_ARRAY_TASK_ID / {n_seeds}))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % {n_seeds}))
FRACTION=${{FRACTIONS[$FRAC_IDX]}}
SEED=$((403 + SEED_IDX * 100))

echo "Running fraction=$FRACTION seed=$SEED (array task $SLURM_ARRAY_TASK_ID)"

{train_cmd} \\
    --learning_curve_fraction $FRACTION \\
    --seed $SEED \\
    --save_dir {save_dir}/learning_curve/frac${{FRACTION}}_seed${{SEED}}
"""

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "slurm-outputs"), exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script)

    print(f"Generated learning curve sbatch script: {script_path}")
    print(f"  Total jobs: {total_jobs} ({len(fractions)} fractions x {n_seeds} seeds)")
    print(f"  Submit with: sbatch {script_path}")

    return script_path
