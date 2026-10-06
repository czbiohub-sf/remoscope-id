"""Training logic for remo-id models."""

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback

import os
import dataclasses
import torch
import yaml
from torch.utils.data import DataLoader

from remo_id.models import get_model, is_mil_model, get_augmentation_backbone
from remo_id.data import RemoIDDataModule, BatchSampler, CrossExperimentBatchSampler, CrossExperimentCollateFn
from remo_id.training.config import TrainingConfig, default_save_dir


def percent_seen_per_sample(n_imgs, num_samples, bag_size, num_rounds):
    """Estimate the % of one sample's images touched by the sampler over a span.

    Models each batch ("round") as drawing ``bag_size`` images spread across the
    ``num_samples`` samples in the pool, so a given sample contributes
    ``bag_size / num_samples`` images per round on average. For drawing that many
    images without replacement within a round, the probability a specific image
    of a sample with ``n_imgs`` images is missed is ``1 - imgs_per_round/n_imgs``;
    assuming rounds are independent, the miss probability compounds over
    ``num_rounds``, so the fraction seen at least once is
    ``1 - miss_prob**num_rounds``.
    """
    imgs_per_round = bag_size / num_samples
    miss_prob = 1 - imgs_per_round / n_imgs
    total = 1 - miss_prob ** num_rounds
    return 100 * total


def estimate_image_coverage(experiment_ranges, batch_size, batches_per_epoch, max_epochs):
    """Aggregate ``percent_seen_per_sample`` over real experiment sizes.

    ``batches_per_epoch`` must be the ACTUAL batches yielded per epoch, i.e.
    ``len(train_sampler)`` — NOT the configured ``num_batches``. The default
    ``BatchSampler`` samples experiments without replacement and ends an epoch
    when a pool exhausts, so it often yields far fewer than ``num_batches``;
    ``CrossExperimentBatchSampler`` samples with replacement and yields exactly
    ``num_batches``. Both are captured by ``len(sampler)``.

    ``num_samples`` is the experiment-pool size, so ``batch_size/num_samples`` is
    the expected per-experiment image draw per batch. Returns total/seen image
    counts and percentages for one epoch (``batches_per_epoch`` rounds) and the
    full run (``batches_per_epoch * max_epochs``).
    """
    sizes = [e - s for s, e in experiment_ranges.values()]
    num_samples = len(sizes)
    total_imgs = sum(sizes)
    rounds_epoch = batches_per_epoch
    rounds_run = batches_per_epoch * max_epochs
    seen_epoch = sum(percent_seen_per_sample(n, num_samples, batch_size, rounds_epoch) / 100 * n for n in sizes)
    seen_run = sum(percent_seen_per_sample(n, num_samples, batch_size, rounds_run) / 100 * n for n in sizes)
    return {
        "num_experiments": num_samples,
        "total_images": total_imgs,
        "batches_per_epoch": batches_per_epoch,
        "seen_per_epoch": seen_epoch,
        "pct_per_epoch": 100 * seen_epoch / total_imgs if total_imgs else 0.0,
        "seen_per_run": seen_run,
        "pct_per_run": 100 * seen_run / total_imgs if total_imgs else 0.0,
    }


class SaveEpochsCheckpoint(Callback):
    """Checkpoint at an explicit, possibly non-periodic list of epochs.

    Unlike ModelCheckpoint(every_n_epochs=N), this supports arbitrary epoch
    lists and epoch 0 (the post-init / pretrained-weights snapshot, before any
    training). ``epoch=0.ckpt`` = initial weights; ``epoch=k.ckpt`` = weights
    after k full training epochs.
    """

    def __init__(self, dirpath, epochs):
        super().__init__()
        self.dirpath = dirpath
        self.epochs = set(epochs)

    def on_fit_start(self, trainer, pl_module):
        if 0 in self.epochs:
            os.makedirs(self.dirpath, exist_ok=True)
            trainer.save_checkpoint(os.path.join(self.dirpath, "epoch=0.ckpt"))

    def on_train_epoch_end(self, trainer, pl_module):
        completed = trainer.current_epoch + 1  # epochs fully trained
        if completed in self.epochs:
            os.makedirs(self.dirpath, exist_ok=True)
            trainer.save_checkpoint(
                os.path.join(self.dirpath, f"epoch={completed}.ckpt"))


class SaveLastCheckpoint(Callback):
    """Always persist the most recent epoch as ``last.ckpt``.

    Lightning's ``ModelCheckpoint(save_last=True)`` only refreshes ``last.ckpt``
    on epochs where the monitored top-k checkpoint is also saved (see
    ``ModelCheckpoint.on_train_epoch_end``: ``last`` is written only when
    ``_last_global_step_saved == trainer.global_step``). When the best epoch
    precedes the final epoch, ``last.ckpt`` therefore ends up holding the *best*
    weights rather than the final-epoch weights. LOO analysis requires
    ``last.ckpt`` to be the same (final) epoch across all folds, so we write it
    explicitly at the end of every training epoch.
    """

    def __init__(self, dirpath):
        super().__init__()
        self.dirpath = dirpath
        self.last_model_path = os.path.join(dirpath, "last.ckpt")

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        os.makedirs(self.dirpath, exist_ok=True)
        trainer.save_checkpoint(self.last_model_path)


def get_num_workers():
    """Get num_workers from SLURM_CPUS_PER_TASK or fallback to defaults."""
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        total = int(slurm_cpus)
        available = max(total - 2, 2)
        train_workers = max(int(available * 0.8), 1)
        val_workers = max(available - train_workers, 1)
        print(f"Detected SLURM CPUs: {total}, using train_workers={train_workers}, val_workers={val_workers}")
        return train_workers, val_workers
    print("No SLURM environment detected, using default workers: train=4, val=2")
    return 4, 2


def coerce_database_label(x):
    """Interpret string labels like 'true/false/1/0' as Python booleans."""
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return x


def auto_detect_precision(precision: str) -> str:
    """Resolve 'auto' precision based on GPU compute capability."""
    if precision != "auto":
        return precision
    if not torch.cuda.is_available():
        return "32"
    capability = torch.cuda.get_device_capability()
    if capability >= (8, 0):  # Ampere+ (A100, H100, RTX 30xx/40xx)
        resolved = "bf16-mixed"
    else:  # V100, RTX 20xx, older
        resolved = "16-mixed"
    print(f"Auto-detected GPU {torch.cuda.get_device_name()} "
          f"(compute {capability[0]}.{capability[1]}) → precision={resolved}")
    return resolved



def build_model_kwargs(
    model: str,
    task_type: str,
    num_classes: int,
    n_experiments: int,
    n_pos: int,
    lr: float,
    lr_schedule: str,
    warmup_epochs: int,
    max_epochs: int,
    weight_decay: float,
    attn_dim: int = 128,
    pretrained: bool = True,
    dropout: float = 0.3,
    finetune_layers: str = "layer3,layer4",
    num_heads: int = 1,
    projection_dim: int = None,
    pooling: str = "attention",
    classifier_hidden: int = None,
    top_k: int = None,
    transformer_layers: int = 2,
    encode_chunk_size: int = None,
    bag_label_mode: str = "hard",
    attn_entropy_weight: float = 0.0,
    patch_pool: bool = False,
    embedding_dim: int = None,
    dino_weights: str = None,
) -> dict:
    """Build model kwargs dict from training parameters."""
    class_weights = torch.tensor([(n_experiments - n_pos) / n_pos]) if task_type == "binary" else None

    model_kwargs = dict(
        class_weights=class_weights, task_type=task_type, num_classes=num_classes,
        lr=lr, lr_schedule=lr_schedule, warmup_epochs=warmup_epochs, max_epochs=max_epochs,
        weight_decay=weight_decay,
    )

    # Embedding-MIL has no image backbone: omit pretrained/finetune/encode_chunk/patch_pool.
    if model == "embedding-mil":
        model_kwargs.update(dict(
            embedding_dim=embedding_dim, n_experiments=n_experiments, attn_dim=attn_dim,
            dropout=dropout, num_heads=num_heads, projection_dim=projection_dim,
            pooling=pooling, classifier_hidden=classifier_hidden, top_k=top_k,
            transformer_layers=transformer_layers,
            bag_label_mode=bag_label_mode, attn_entropy_weight=attn_entropy_weight,
        ))
        return model_kwargs

    if is_mil_model(model):
        effective_finetune_layers = finetune_layers
        if 'dino' in model:
            effective_finetune_layers = finetune_layers.replace("layer3", "block10").replace("layer4", "block11")
        elif 'vit' in model:
            effective_finetune_layers = finetune_layers.replace("layer3", "layer-2").replace("layer4", "layer-1")

        model_kwargs.update(dict(
            n_experiments=n_experiments, attn_dim=attn_dim, pretrained=pretrained,
            dropout=dropout, finetune_layers=effective_finetune_layers, num_heads=num_heads,
            projection_dim=projection_dim,
        ))
        if '-mil' in model:
            model_kwargs.update(dict(
                pooling=pooling, classifier_hidden=classifier_hidden,
                top_k=top_k, encode_chunk_size=encode_chunk_size,
                bag_label_mode=bag_label_mode,
                attn_entropy_weight=attn_entropy_weight,
            ))
            if 'dino' in model:
                model_kwargs['patch_pool'] = patch_pool
                model_kwargs['dino_weights'] = dino_weights
        else:
            model_kwargs['classifier_hidden'] = classifier_hidden
    else:
        if 'dino' in model:
            model_kwargs.update(dict(
                pretrained=pretrained, finetune_layers=finetune_layers,
                patch_pool=patch_pool, dino_weights=dino_weights,
            ))
        elif 'resnet' in model:
            model_kwargs.update(dict(
                pretrained=pretrained, finetune_layers=finetune_layers,
            ))
        elif 'vit' in model:
            # Translate ResNet-style layer names to ViT layer indices for instance models
            effective_finetune_layers = finetune_layers.replace("layer3", "layer-2").replace("layer4", "layer-1")
            model_kwargs['finetune_layers'] = effective_finetune_layers

    return model_kwargs


def build_model_kwargs_from_config(config: TrainingConfig) -> dict:
    """Build model kwargs dict from a TrainingConfig."""
    return build_model_kwargs(
        model=config.model_name,
        task_type=config.task_type,
        num_classes=config.num_classes,
        n_experiments=config.n_experiments,
        n_pos=config.n_pos,
        lr=config.lr,
        lr_schedule=config.lr_schedule,
        warmup_epochs=config.warmup_epochs,
        max_epochs=config.max_epochs,
        weight_decay=config.weight_decay,
        attn_dim=config.attn_dim,
        pretrained=config.pretrained,
        dropout=config.dropout,
        finetune_layers=config.finetune_layers,
        num_heads=config.num_heads,
        projection_dim=config.projection_dim,
        pooling=config.pooling,
        classifier_hidden=config.classifier_hidden,
        top_k=config.top_k,
        transformer_layers=config.transformer_layers,
        encode_chunk_size=config.encode_chunk_size,
        bag_label_mode=config.bag_label_mode,
        attn_entropy_weight=config.attn_entropy_weight,
        patch_pool=config.patch_pool,
        embedding_dim=config.embedding_dim,
        dino_weights=config.dino_weights,
    )


def build_data_module(
    model: str,
    database_column: str = "class_label",
    database_label=None,
    label_contains: str = None,
    study_id: str = None,
    study_ids: list = None,
    threshold: float = None,
    threshold_mode: str = "==",
    negative_study_ids: list = None,
    negative_experiment_ids: list = None,
    batch_size: int = 200,
    task_type: str = "binary",
    num_classes: int = 2,
    class_definitions: list = None,
    skip_null_labels: bool = True,
    exclude_filters: list = None,
    include_filters: list = None,
    default_class: str = None,
    background_correction: str = 'none',
    background_value: int = None,
    intensity_jitter: float = None,
    noise_std: float = None,
    blur_sigma: float = None,
    gradient_blur_sigma: float = None,
    erase_num: str = None,
    erase_scale: str = None,
    min_image_index: int = 0,
    crop_size: int = None,
    model_version: str = None,
    cell_config: str = None,
    load_workers: int = None,
    pool_cap: int = None,
    manifest: str = None,
    label_column: str = "label",
    classes: list = None,
) -> 'RemoIDDataModule':
    """Build the data module for training.

    Shared between run_training() and run_cross_validation(). For model=="embedding-mil"
    returns an EmbeddingDataModule (precomputed cell embeddings); otherwise RemoIDDataModule.
    ``manifest`` selects experiments from a CSV instead of the database (it is turned
    into the equivalent class config, see :mod:`remo_id.data.manifest`).
    """
    default_class_idx = None
    if class_definitions and default_class is not None:
        class_names = list(dict.fromkeys(d["name"] for d in class_definitions))
        default_class_idx = class_names.index(default_class)

    if model == "embedding-mil":
        from remo_id.data.embedding_datamodule import EmbeddingDataModule
        if not model_version:
            raise ValueError("--model_version is required for --model embedding-mil")
        return EmbeddingDataModule(
            model_version=model_version,
            cell_config=cell_config,
            pool_cap=pool_cap,
            manifest=manifest,
            label_column=label_column,
            classes=classes,
            database_column=database_column,
            database_label=database_label,
            label_contains=label_contains,
            study_id=study_id,
            study_ids=study_ids,
            threshold=threshold,
            threshold_mode=threshold_mode,
            negative_study_ids=negative_study_ids or [],
            negative_experiment_ids=negative_experiment_ids or [],
            batch_size=batch_size,
            task_type=task_type,
            num_classes=num_classes,
            class_definitions=class_definitions,
            skip_null_labels=skip_null_labels,
            exclude_filters=exclude_filters,
            include_filters=include_filters,
            default_class_idx=default_class_idx,
            load_workers=load_workers,
        )

    return RemoIDDataModule(
        manifest=manifest,
        label_column=label_column,
        classes=classes,
        database_column=database_column,
        database_label=database_label,
        label_contains=label_contains,
        study_id=study_id,
        study_ids=study_ids,
        threshold=threshold,
        threshold_mode=threshold_mode,
        negative_study_ids=negative_study_ids or [],
        negative_experiment_ids=negative_experiment_ids or [],
        backbone_name=get_augmentation_backbone(model),
        batch_size=batch_size,
        task_type=task_type,
        num_classes=num_classes,
        class_definitions=class_definitions,
        skip_null_labels=skip_null_labels,
        exclude_filters=exclude_filters,
        include_filters=include_filters,
        default_class_idx=default_class_idx,
        background_correction=background_correction,
        background_value=background_value,
        intensity_jitter=intensity_jitter,
        noise_std=noise_std,
        blur_sigma=blur_sigma,
        gradient_blur_sigma=gradient_blur_sigma,
        erase_num=erase_num,
        erase_scale=erase_scale,
        min_image_index=min_image_index,
        crop_size=crop_size,
    )


def _permute_labels_by_sample_id(data_module, seed, save_dir=None):
    """Negative-control: randomly permute class labels across patients (sample_id).

    Balance-preserving (permutes the multiset of per-sample labels, so overall class
    counts are unchanged). All experiments of a sample receive one shared new label.
    Mutates ``data_module.data[database_column]`` and ``data_module.experiment_labels``
    in place; the split/sampler/prediction code all read from these, so the scramble
    propagates everywhere. Expected result of training on these labels: chance-level
    validation performance. Writes an audit map to ``label_permutation.json``.
    """
    import json
    import random
    import pandas as pd
    from collections import Counter

    df = data_module.data
    col = data_module.database_column

    sample_to_rows = {}
    for i, row in df.iterrows():
        sid = row.get('sample_id')
        if sid is None or (isinstance(sid, float) and pd.isna(sid)) or str(sid).strip() in ('', 'nan', 'None'):
            sid = f"__unique__{row['path']}"
        sample_to_rows.setdefault(str(sid), []).append(i)

    samples = list(sample_to_rows.keys())
    old_labels = {
        sid: Counter(int(df.at[i, col]) for i in idxs).most_common(1)[0][0]
        for sid, idxs in sample_to_rows.items()
    }

    label_list = [old_labels[s] for s in samples]
    permuted = label_list[:]
    random.Random(seed).shuffle(permuted)
    new_labels = dict(zip(samples, permuted))

    for sid, idxs in sample_to_rows.items():
        lbl = int(new_labels[sid])
        for i in idxs:
            df.at[i, col] = lbl
            p = df.at[i, 'path']
            if p in data_module.experiment_labels:
                data_module.experiment_labels[p] = lbl

    print(f"[permute_labels] Permuted labels across {len(samples)} samples (seed={seed}). "
          f"Class balance preserved: {dict(Counter(label_list))} -> {dict(Counter(permuted))}")

    if save_dir:
        audit = {sid: [int(old_labels[sid]), int(new_labels[sid])] for sid in samples}
        with open(os.path.join(save_dir, 'label_permutation.json'), 'w') as f:
            json.dump({'seed': seed, 'database_column': col, 'permutation': audit}, f, indent=2)

    return data_module


def run_training(
    database_column: str = "class_label",
    database_label: str = None,
    label_contains: str = None,
    study_id: str = None,
    study_ids: list = None,
    threshold: float = None,
    threshold_mode: str = "==",
    negative_study_ids: list = None,
    negative_experiment_ids: list = None,
    skip_null_labels: bool = True,
    val_ratio: float = 0.2,
    batch_size: int = 200,
    n_experiments: int = 8,
    num_batches: int = 100,
    model: str = "resnet18",
    task_type: str = "binary",
    num_classes: int = 2,
    class_definitions: list = None,
    project: str = "remo-id",
    save_dir: str = None,
    max_epochs: int = 100,
    min_epochs: int = None,
    log_every_n_steps: int = 50,
    accelerator: str = "auto",
    devices: int = 1,
    seed: int = 403,
    val_seeds: str = None,
    lr: float = 1e-4,
    lr_schedule: str = "cosine",
    warmup_epochs: int = 5,
    attn_dim: int = 128,
    pretrained: bool = True,
    dropout: float = 0.3,
    weight_decay: float = 0.01,
    finetune_layers: str = "layer3,layer4",
    early_stopping_patience: int = 15,
    num_heads: int = 1,
    accumulate_grad_batches: int = 1,
    cross_experiment_sampling: bool = False,
    sample_aware_sampling: bool = False,
    sample_aware_mode: str = "single",
    max_experiments_per_bag: int = None,
    majority_min_pct: float = 1.0,
    bag_label_mode: str = "hard",
    attn_entropy_weight: float = 0.0,
    projection_dim: int = None,
    pooling: str = "attention",
    classifier_hidden: int = None,
    top_k: int = None,
    transformer_layers: int = 2,
    encode_chunk_size: int = None,
    split_by_sample_id: bool = True,
    val_experiment_ids: list = None,
    exclude_filters: list = None,
    include_filters: list = None,
    default_class: str = None,
    class_config_raw: dict = None,
    precision: str = "auto",
    compile: bool = False,
    deterministic: bool = True,
    run_infer: bool = True,
    ckpt_path: str = None,
    background_correction: str = 'none',
    background_value: int = None,
    intensity_jitter: float = None,
    noise_std: float = None,
    blur_sigma: float = None,
    gradient_blur_sigma: float = None,
    erase_num: str = None,
    erase_scale: str = None,
    min_image_index: int = 0,
    save_every_n_epochs: int = None,
    save_epochs: str = None,
    patch_pool: bool = False,
    crop_size: int = None,
    permute_labels: bool = False,
    permute_labels_seed: int = 404,
    model_version: str = None,
    cells_per_bag: int = None,
    cell_config: str = None,
    load_workers: int = None,
    pool_cap: int = None,
    manifest: str = None,
    label_column: str = "label",
    classes: list = None,
    dino_weights: str = None,
):
    """Run training with the given configuration."""
    save_dir = save_dir or default_save_dir()
    negative_study_ids = negative_study_ids or []
    negative_experiment_ids = negative_experiment_ids or []

    if not deterministic:
        torch.backends.cudnn.benchmark = True

    seed_everything(seed=seed, workers=True)

    # save_dir should already be a timestamped directory created by the CLI
    os.makedirs(save_dir, exist_ok=True)

    # Embedding-MIL counts cells, not images. --cells_per_bag (if given) sets the per-bag
    # cell count; batch_size is then cells_per_bag * n_experiments so the sampler's
    # samples_per_experiment == cells_per_bag.
    if model == "embedding-mil" and cells_per_bag:
        batch_size = cells_per_bag * n_experiments

    samples_per_experiment = batch_size // n_experiments
    n_pos = n_experiments // 2
    print(f"Batch config: batch_size={batch_size}, n_experiments={n_experiments}, "
          f"samples_per_experiment={samples_per_experiment}, n_pos={n_pos}")

    db_label = coerce_database_label(database_label)

    # Validate bag-mixing + label-mode combinations
    if task_type == "binary":
        # For binary, majority_min_pct is the minimum fraction of positive-experiment
        # images in a positive bag (see CrossExperimentBatchSampler). Negative bags
        # stay 100% negative and the bag label remains any-positive.
        if not (0.0 < majority_min_pct <= 1.0 + 1e-6):
            raise ValueError(
                f"--majority_min_pct must be in (0.0, 1.0] for binary tasks, got {majority_min_pct}"
            )
        if bag_label_mode != "hard":
            raise ValueError(
                "--bag_label_mode=soft is multiclass-only; binary MIL uses hard labels."
            )
    if bag_label_mode == "soft" and task_type != "multiclass":
        raise ValueError("--bag_label_mode=soft requires --task_type=multiclass")
    if bag_label_mode == "soft" and majority_min_pct >= 0.999:
        print(
            "WARN: --bag_label_mode=soft with --majority_min_pct=1.0 is degenerate "
            "(proportion target is one-hot — numerically identical to hard mode)."
        )
    if attn_entropy_weight > 0 and bag_label_mode != "soft":
        print(
            "WARN: --attn_entropy_weight>0 has no effect unless --bag_label_mode=soft"
        )
    if attn_entropy_weight > 0 and pooling not in ("attention", "topk"):
        print(
            f"WARN: --attn_entropy_weight>0 has no effect with --pooling={pooling} "
            "(only active for attention/topk pooling)"
        )

    config = TrainingConfig(
        model_name=model,
        pretrained=pretrained,
        finetune_layers=finetune_layers,
        dino_weights=dino_weights,
        task_type=task_type,
        num_classes=num_classes,
        database_column=database_column,
        database_label=db_label,
        study_id=study_id,
        negative_study_ids=negative_study_ids,
        negative_experiment_ids=negative_experiment_ids,
        skip_null_labels=skip_null_labels,
        class_definitions=class_definitions,
        class_config_raw=class_config_raw,
        manifest=manifest,
        label_column=label_column,
        classes=classes,
        batch_size=batch_size,
        n_experiments=n_experiments,
        num_batches=num_batches,
        cross_experiment_sampling=cross_experiment_sampling,
        sample_aware_sampling=sample_aware_sampling,
        sample_aware_mode=sample_aware_mode,
        max_experiments_per_bag=max_experiments_per_bag,
        majority_min_pct=majority_min_pct,
        bag_label_mode=bag_label_mode,
        attn_entropy_weight=attn_entropy_weight,
        lr=lr,
        lr_schedule=lr_schedule,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        weight_decay=weight_decay,
        attn_dim=attn_dim,
        dropout=dropout,
        num_heads=num_heads,
        projection_dim=projection_dim,
        pooling=pooling,
        classifier_hidden=classifier_hidden,
        top_k=top_k,
        transformer_layers=transformer_layers,
        encode_chunk_size=encode_chunk_size,
        save_dir=save_dir,
        project=project,
        accelerator=accelerator,
        devices=devices,
        seed=seed,
        log_every_n_steps=log_every_n_steps,
        early_stopping_patience=early_stopping_patience,
        accumulate_grad_batches=accumulate_grad_batches,
        precision=precision,
        compile=compile,
        deterministic=deterministic,
        run_infer=run_infer,
        background_correction=background_correction,
        background_value=background_value,
        intensity_jitter=intensity_jitter,
        noise_std=noise_std,
        blur_sigma=blur_sigma,
        gradient_blur_sigma=gradient_blur_sigma,
        erase_num=erase_num,
        erase_scale=erase_scale,
        min_image_index=min_image_index,
        save_every_n_epochs=save_every_n_epochs,
        save_epochs=save_epochs,
        patch_pool=patch_pool,
        crop_size=crop_size,
        permute_labels=permute_labels,
        permute_labels_seed=permute_labels_seed,
        model_version=model_version,
        cells_per_bag=cells_per_bag,
        cell_config=cell_config,
        load_workers=load_workers,
        pool_cap=pool_cap,
    )

    full_data_module = build_data_module(
        model=model, database_column=database_column, database_label=db_label,
        label_contains=label_contains, study_id=study_id, study_ids=study_ids,
        threshold=threshold, threshold_mode=threshold_mode,
        negative_study_ids=negative_study_ids,
        negative_experiment_ids=negative_experiment_ids,
        batch_size=batch_size, task_type=task_type, num_classes=num_classes,
        class_definitions=class_definitions,
        skip_null_labels=skip_null_labels, exclude_filters=exclude_filters,
        include_filters=include_filters, default_class=default_class,
        background_correction=background_correction,
        background_value=background_value, intensity_jitter=intensity_jitter,
        noise_std=noise_std, blur_sigma=blur_sigma,
        gradient_blur_sigma=gradient_blur_sigma,
        erase_num=erase_num,
        erase_scale=erase_scale,
        min_image_index=min_image_index,
        crop_size=crop_size,
        model_version=model_version,
        cell_config=cell_config,
        load_workers=load_workers,
        pool_cap=pool_cap,
        manifest=manifest,
        label_column=label_column,
        classes=classes,
    )

    if permute_labels:
        print(f"[permute_labels] ENABLED — this is a negative-control run. Labels will be "
              f"randomly permuted across patients (seed={permute_labels_seed}).")
        _permute_labels_by_sample_id(full_data_module, permute_labels_seed, save_dir=save_dir)

    # Embedding-MIL resolves its feature dim from the loaded store; build kwargs AFTER.
    if model == "embedding-mil":
        config.embedding_dim = full_data_module.embedding_dim
    model_kwargs = build_model_kwargs_from_config(config)

    model_instance = get_model(model, **model_kwargs)
    model_cls = model_instance.__class__

    if compile:
        print(f"Compiling model with torch.compile()...")
        model_instance = torch.compile(model_instance)

    # Multi-seed validation: the (expensive) embedding store is loaded ONCE above; here we
    # run several train/val splits in-process. Each seed re-splits cheaply (child modules
    # reuse the resident embedding pools by reference) and trains a FRESH model into its own
    # save_dir subfolder — a fast variability estimate across held-out sets without
    # reloading data per run. Mutually exclusive with --val_experiment_ids.
    if val_seeds:
        if val_experiment_ids:
            raise ValueError("--val_seeds cannot be combined with --val_experiment_ids")
        seeds = [int(s) for s in str(val_seeds).replace(",", " ").split()]
        print(f"[val_seeds] Running {len(seeds)} held-out splits (data loaded once): {seeds}")
        results = []
        for s in seeds:
            train_module, val_module = full_data_module.split_train_val(
                val_ratio=val_ratio, seed=s, split_by_sample_id=split_by_sample_id)
            seed_config = dataclasses.replace(
                config, seed=s, save_dir=os.path.join(save_dir, f"seed_{s}"))
            res = run_single_fold(
                config=seed_config,
                model_instance=None,  # fresh weights per seed
                model_cls=model_cls,
                model_kwargs=model_kwargs,
                train_module=train_module,
                val_module=val_module,
                fold_label=f"seed{s}",
                ckpt_path=ckpt_path,
            )
            results.append(res)
        # Each seed's artifacts are saved to its own subdir; return the last for the caller.
        return results[-1]

    if val_experiment_ids:
        # Use explicit experiment IDs for validation split
        val_experiment_ids = [str(eid) for eid in val_experiment_ids]
        df = full_data_module.data
        val_paths = set(df[df['experiment_id'].astype(str).isin(val_experiment_ids)]['path'])
        train_paths = set(df['path']) - val_paths
        matched = len(val_paths)
        print(f"Using --val_experiment_ids: {matched} val, {len(train_paths)} train")
        if matched < len(val_experiment_ids):
            missing = set(val_experiment_ids) - set(df['experiment_id'].astype(str))
            print(f"WARNING: {len(missing)} val_experiment_ids not found in data: {missing}")
        train_module, val_module = full_data_module.split_from_paths(train_paths, val_paths)
    else:
        train_module, val_module = full_data_module.split_train_val(val_ratio=val_ratio, seed=seed, split_by_sample_id=split_by_sample_id)

    model_instance, trainer, train_module, val_module = run_single_fold(
        config=config,
        model_instance=model_instance,
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        train_module=train_module,
        val_module=val_module,
        ckpt_path=ckpt_path,
    )

    return model_instance, trainer, train_module, val_module


def run_single_fold(
    config: TrainingConfig,
    model_instance,
    model_cls,
    model_kwargs: dict,
    train_module,
    val_module,
    fold_label: str = None,
    # Legacy individual params — kept for backwards compat with callers that pass them
    model_name: str = None,
    batch_size: int = None,
    n_experiments: int = None,
    n_pos: int = None,
    num_batches: int = None,
    task_type: str = None,
    num_classes: int = None,
    cross_experiment_sampling: bool = None,
    sample_aware_sampling: bool = None,
    sample_aware_mode: str = None,
    max_experiments_per_bag: int = None,
    save_dir: str = None,
    project: str = None,
    max_epochs: int = None,
    min_epochs: int = None,
    log_every_n_steps: int = None,
    accelerator: str = None,
    devices: int = None,
    seed: int = None,
    lr: float = None,
    lr_schedule: str = None,
    warmup_epochs: int = None,
    finetune_layers: str = None,
    pretrained: bool = None,
    weight_decay: float = None,
    early_stopping_patience: int = None,
    accumulate_grad_batches: int = None,
    precision: str = None,
    compile: bool = None,
    deterministic: bool = None,
    attn_dim: int = None,
    dropout: float = None,
    num_heads: int = None,
    projection_dim: int = None,
    pooling: str = None,
    classifier_hidden: int = None,
    database_column: str = None,
    database_label=None,
    study_id: str = None,
    negative_study_ids: list = None,
    skip_null_labels: bool = None,
    negative_experiment_ids: list = None,
    class_definitions: list = None,
    class_config_raw: dict = None,
    run_infer: bool = None,
    ckpt_path: str = None,
):
    """Train a single fold. Extracted from run_training() so k-fold CV can call it.

    Args:
        config: TrainingConfig with all training parameters
        model_instance: Instantiated model (or None to create from model_kwargs)
        model_cls: Model class for checkpoint loading
        model_kwargs: kwargs used to create model (for re-instantiation)
        train_module: RemoIDDataModule for training
        val_module: RemoIDDataModule for validation
        fold_label: Optional label for logging (e.g. "fold_0")

    Returns:
        (model_instance, trainer, train_module, val_module)
    """
    # Resolve values: config takes precedence, legacy params are fallbacks
    # (legacy params remain for backwards compat with callers like workflows/cv.py)
    _model_name = model_name if model_name is not None else config.model_name
    _batch_size = batch_size if batch_size is not None else config.batch_size
    _n_experiments = n_experiments if n_experiments is not None else config.n_experiments
    _n_pos = n_pos if n_pos is not None else config.n_pos
    _num_batches = num_batches if num_batches is not None else config.num_batches
    _task_type = task_type if task_type is not None else config.task_type
    _num_classes = num_classes if num_classes is not None else config.num_classes
    _cross_experiment_sampling = cross_experiment_sampling if cross_experiment_sampling is not None else config.cross_experiment_sampling
    _sample_aware_sampling = sample_aware_sampling if sample_aware_sampling is not None else config.sample_aware_sampling
    _sample_aware_mode = sample_aware_mode if sample_aware_mode is not None else config.sample_aware_mode
    _max_experiments_per_bag = max_experiments_per_bag if max_experiments_per_bag is not None else config.max_experiments_per_bag
    _save_dir = save_dir if save_dir is not None else config.save_dir
    _project = project if project is not None else config.project
    _max_epochs = max_epochs if max_epochs is not None else config.max_epochs
    _min_epochs = min_epochs if min_epochs is not None else config.min_epochs
    _log_every_n_steps = log_every_n_steps if log_every_n_steps is not None else config.log_every_n_steps
    _accelerator = accelerator if accelerator is not None else config.accelerator
    _devices = devices if devices is not None else config.devices
    _seed = seed if seed is not None else config.seed
    _lr = lr if lr is not None else config.lr
    _lr_schedule = lr_schedule if lr_schedule is not None else config.lr_schedule
    _warmup_epochs = warmup_epochs if warmup_epochs is not None else config.warmup_epochs
    _finetune_layers = finetune_layers if finetune_layers is not None else config.finetune_layers
    _pretrained = pretrained if pretrained is not None else config.pretrained
    _weight_decay = weight_decay if weight_decay is not None else config.weight_decay
    _early_stopping_patience = early_stopping_patience if early_stopping_patience is not None else config.early_stopping_patience
    _accumulate_grad_batches = accumulate_grad_batches if accumulate_grad_batches is not None else config.accumulate_grad_batches
    _precision = precision if precision is not None else config.precision
    _compile = compile if compile is not None else config.compile
    _deterministic = deterministic if deterministic is not None else config.deterministic
    _attn_dim = attn_dim if attn_dim is not None else config.attn_dim
    _dropout = dropout if dropout is not None else config.dropout
    _num_heads = num_heads if num_heads is not None else config.num_heads
    _projection_dim = projection_dim if projection_dim is not None else config.projection_dim
    _pooling = pooling if pooling is not None else config.pooling
    _classifier_hidden = classifier_hidden if classifier_hidden is not None else config.classifier_hidden
    _majority_min_pct = config.majority_min_pct
    _bag_label_mode = config.bag_label_mode
    _attn_entropy_weight = config.attn_entropy_weight
    _database_column = database_column if database_column is not None else config.database_column
    _database_label = database_label if database_label is not None else config.database_label
    _study_id = study_id if study_id is not None else config.study_id
    _negative_study_ids = negative_study_ids if negative_study_ids is not None else config.negative_study_ids or []
    _skip_null_labels = skip_null_labels if skip_null_labels is not None else config.skip_null_labels
    _negative_experiment_ids = negative_experiment_ids if negative_experiment_ids is not None else config.negative_experiment_ids or []
    _class_definitions = class_definitions if class_definitions is not None else config.class_definitions
    _class_config_raw = class_config_raw if class_config_raw is not None else config.class_config_raw
    _run_infer = run_infer if run_infer is not None else config.run_infer

    samples_per_experiment = _batch_size // _n_experiments
    num_workers_train, num_workers_val = get_num_workers()
    # Embedding-MIL samples cell vectors from RAM (no zarr decode / augmentation), so
    # worker processes add only fork/copy-on-write cost over the large resident arrays.
    if _model_name == "embedding-mil":
        num_workers_train = num_workers_val = 0

    val_exp_ids = val_module.data['experiment_id'].tolist()
    train_exp_ids = train_module.data['experiment_id'].tolist()
    prefix = f"[{fold_label}] " if fold_label else ""
    print(f"{prefix}Validation experiment IDs ({len(val_exp_ids)}): {val_exp_ids}")
    print(f"{prefix}Training experiment IDs ({len(train_exp_ids)}): {train_exp_ids}")

    # If no model_instance provided, create one fresh
    if model_instance is None:
        model_instance = get_model(_model_name, **model_kwargs)
        model_cls = model_instance.__class__

    if _compile:
        model_instance = torch.compile(model_instance)

    # Mixed-bag construction (majority_min_pct < 1.0) or soft LLP labels require the
    # CrossExperimentBatchSampler + CrossExperimentCollateFn path even when CES is off.
    _needs_mixed_path = _majority_min_pct < 1.0 or _bag_label_mode == "soft"
    _use_cross_exp_sampler = _cross_experiment_sampling or _needs_mixed_path

    # Choose sampler
    _is_embedding = _model_name == "embedding-mil"
    if _sample_aware_sampling and (_is_embedding or _use_cross_exp_sampler):
        raise ValueError(
            "--sample_aware_sampling is only supported with the default per-experiment "
            "BatchSampler (not cross-experiment/mixed-bag/embedding sampling)"
        )
    if _is_embedding:
        from remo_id.data import EmbeddingBatchSampler
        _ce = "cross-experiment (class-pooled bags)" if _cross_experiment_sampling else "per-experiment bags"
        print(f"{prefix}Embedding bag: cell classes {train_module.cell_config.names}, "
              f"weights {list(train_module.cell_weights)}; cells/bag={_batch_size // _n_experiments}; "
              f"train sampling = {_ce}")
        train_sampler = EmbeddingBatchSampler(
            train_module.experiment_ranges, train_module.experiment_labels,
            train_module.class_local, train_module.true_counts, train_module.cell_weights,
            batch_size=_batch_size, n_experiments=_n_experiments, n_pos=_n_pos,
            num_batches=_num_batches, task_type=_task_type, num_classes=_num_classes,
            is_validation=False, cross_experiment=_cross_experiment_sampling,
            majority_min_pct=_majority_min_pct, bag_label_mode=_bag_label_mode,
        )
    elif _use_cross_exp_sampler:
        train_sampler = CrossExperimentBatchSampler(
            train_module.experiment_ranges, train_module.experiment_labels,
            bag_size=_batch_size // _n_experiments, n_bags=_n_experiments,
            num_batches=_num_batches, max_experiments_per_bag=_max_experiments_per_bag,
            task_type=_task_type, num_classes=_num_classes,
            majority_min_pct=_majority_min_pct,
            cross_experiment=_cross_experiment_sampling,
            is_validation=False,
        )
    else:
        # Sample-aware: pool by sample_id so technical replicates of one sample
        # count as a single pool entry (one random replicate per epoch).
        _exp_sample_ids = None
        if _sample_aware_sampling:
            _exp_sample_ids = dict(zip(train_module.data['path'],
                                       train_module.data['sample_id']))
        train_sampler = BatchSampler(
            train_module.experiment_ranges, train_module.experiment_labels,
            batch_size=_batch_size, n_experiments=_n_experiments, n_pos=_n_pos,
            num_batches=_num_batches, task_type=_task_type, num_classes=_num_classes,
            is_validation=False, exp_sample_ids=_exp_sample_ids,
            sample_aware_mode=_sample_aware_mode,
        )

    # Estimate how much of each experiment's image pool the sampler actually
    # touches. Uses len(train_sampler) (real batches/epoch) — the default
    # BatchSampler samples experiments without replacement and may yield far
    # fewer than _num_batches per epoch (see estimate_image_coverage).
    batches_per_epoch = len(train_sampler)
    coverage = estimate_image_coverage(
        train_module.experiment_ranges, _batch_size, batches_per_epoch, _max_epochs)
    print(f"{prefix}Image coverage estimate: {coverage['total_images']:,} train images "
          f"across {coverage['num_experiments']} experiments "
          f"({batches_per_epoch} batches/epoch)")
    print(f"{prefix}  per epoch: ~{coverage['seen_per_epoch']:,.0f} unique images seen "
          f"({coverage['pct_per_epoch']:.2f}%)")
    print(f"{prefix}  per run ({_max_epochs} epochs): ~{coverage['seen_per_run']:,.0f} "
          f"unique images seen ({coverage['pct_per_run']:.2f}%)")

    val_n_experiments = min(_n_experiments, len(val_module.experiment_ranges))
    # Adjust val batch_size to be divisible by val_n_experiments (needed for LOO
    # where the held-out sample may have fewer experiments than n_experiments)
    val_batch_size = _batch_size - (_batch_size % val_n_experiments) if val_n_experiments > 0 else _batch_size
    # Clamp n_pos to actual positive count so a lone negative experiment still gets selected
    if _task_type == "binary":
        val_n_pos_actual = sum(1 for v in val_module.experiment_labels.values() if v == 1)
        val_n_pos = min(_n_pos, val_n_pos_actual, val_n_experiments)
    else:
        val_n_pos = _n_pos
    if _is_embedding:
        # Embedding val sampler pads to the model's n_experiments (its with-replacement
        # selection repeats experiments when val has fewer), keeping a whole number of
        # equal bags so _forward_step reshapes by self.n_experiments cleanly. Use the full
        # _batch_size / _n_experiments (always divisible: _batch_size = cells_per_bag * n).
        val_sampler = EmbeddingBatchSampler(
            val_module.experiment_ranges, val_module.experiment_labels,
            val_module.class_local, val_module.true_counts, val_module.cell_weights,
            batch_size=_batch_size, n_experiments=_n_experiments, n_pos=_n_pos,
            num_batches=max(1, _num_batches // 5), task_type=_task_type, num_classes=_num_classes,
            is_validation=True,
        )
    else:
        val_sampler = BatchSampler(
            val_module.experiment_ranges, val_module.experiment_labels,
            batch_size=val_batch_size, n_experiments=val_n_experiments, n_pos=val_n_pos,
            num_batches=_num_batches // 5, task_type=_task_type, num_classes=_num_classes,
            is_validation=True,
        )

    if _is_embedding:
        # Embedding path always uses its own collate: it returns per-cell embeddings and
        # each cell's experiment disease label, which is exactly what _forward_step's soft
        # path needs (it computes the bag class-proportion from those per-cell labels).
        # Mixing happens in EmbeddingBatchSampler, not the collate.
        train_collate = train_module.test_collate_fn
    elif _use_cross_exp_sampler:
        train_collate = CrossExperimentCollateFn(
            n_bags=_n_experiments, task_type=_task_type, bag_label_mode=_bag_label_mode,
        )
    else:
        train_collate = train_module.test_collate_fn

    train_loader = DataLoader(
        train_module, batch_sampler=train_sampler,
        num_workers=num_workers_train, collate_fn=train_collate,
        pin_memory=True,
        persistent_workers=num_workers_train > 0,
        prefetch_factor=2 if num_workers_train > 0 else None,
    )
    val_loader = DataLoader(
        val_module, batch_sampler=val_sampler,
        num_workers=num_workers_val, collate_fn=val_module.test_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers_val > 0,
        prefetch_factor=2 if num_workers_val > 0 else None,
    )

    fold_save_dir = os.path.join(_save_dir, fold_label) if fold_label else _save_dir
    import wandb
    # Finish any existing wandb run so each fold gets its own
    if wandb.run is not None:
        wandb.finish()

    # Build descriptive wandb run name
    slurm_id = os.environ.get('SLURM_JOB_ID', '')
    _is_mil = is_mil_model(_model_name)
    pooling_str = f"_{_pooling}" if _is_mil else ""
    if _is_mil and _pooling == "attention" and _num_heads > 1:
        pooling_str = f"_{_pooling}{_num_heads}head"
    mil_str = "_mil" if _is_mil else ""
    backbone_str = _model_name.replace("-mil", "")
    wandb_run_name = f"{backbone_str}{mil_str}_{_task_type}{pooling_str}"
    if slurm_id:
        wandb_run_name = f"{wandb_run_name}_{slurm_id}"

    wandb_id_file = os.path.join(fold_save_dir, 'wandb_run_id.txt')
    try:
        if ckpt_path and os.path.exists(wandb_id_file):
            run_id = open(wandb_id_file).read().strip()
            print(f"Resuming W&B run: {run_id}")
            logger = WandbLogger(save_dir=fold_save_dir, project=_project, id=run_id, resume="allow")
        else:
            logger = WandbLogger(save_dir=fold_save_dir, project=_project, name=wandb_run_name)
            # Trigger wandb.init so we can save the run ID for future resumes
            _ = logger.experiment
            os.makedirs(fold_save_dir, exist_ok=True)
            with open(wandb_id_file, 'w') as f:
                f.write(wandb.run.id)
    except wandb.errors.UsageError as e:
        print(f"WARNING: W&B login failed ({e}). Switching to offline mode.")
        os.environ["WANDB_MODE"] = "offline"
        logger = WandbLogger(save_dir=fold_save_dir, project=_project, name=wandb_run_name)
        _ = logger.experiment
        os.makedirs(fold_save_dir, exist_ok=True)

    # Log hyperparameters
    hyperparams = {
        "model": {
            "name": getattr(model_instance, "name", _model_name),
            "backbone": model_instance.backbone_name,
            "finetune_layers": _finetune_layers,
            "pretrained": _pretrained,
        },
        "training": {
            "lr": _lr, "lr_schedule": _lr_schedule, "warmup_epochs": _warmup_epochs,
            "max_epochs": _max_epochs, "weight_decay": _weight_decay,
            "early_stopping_patience": _early_stopping_patience, "seed": _seed,
            "accumulate_grad_batches": _accumulate_grad_batches,
            "precision": _precision, "compile": _compile, "deterministic": _deterministic,
        },
        "data": {
            "database_column": _database_column, "database_label": _database_label,
            "study_id": _study_id, "task_type": _task_type, "num_classes": _num_classes,
            "skip_null_labels": _skip_null_labels,
            "negative_study_ids": _negative_study_ids,
            "negative_experiment_ids": _negative_experiment_ids,
            "class_definitions": _class_definitions,
            "min_image_index": config.min_image_index,
            "background_correction": config.background_correction,
            "background_value": config.background_value,
        },
        "batching": {
            "batch_size": _batch_size, "n_experiments": _n_experiments,
            "samples_per_experiment": samples_per_experiment, "n_pos": _n_pos,
            "cross_experiment_sampling": _cross_experiment_sampling,
            "sample_aware_sampling": _sample_aware_sampling,
            "sample_aware_mode": _sample_aware_mode,
            "max_experiments_per_bag": _max_experiments_per_bag,
        },
        "dataset_stats": {
            "num_train_images": len(train_loader.dataset),
            "num_val_images": len(val_loader.dataset),
            "train_experiments": getattr(train_module, 'num_experiments', None),
            "val_experiments": getattr(val_module, 'num_experiments', None),
            "train_unique_patients": getattr(train_module, 'num_unique_patients', None),
            "val_unique_patients": getattr(val_module, 'num_unique_patients', None),
            "est_images_seen_per_epoch": coverage["seen_per_epoch"],
            "est_pct_seen_per_epoch": coverage["pct_per_epoch"],
            "est_images_seen_per_run": coverage["seen_per_run"],
            "est_pct_seen_per_run": coverage["pct_per_run"],
        },
    }

    if is_mil_model(_model_name):
        hyperparams["mil"] = {
            "attn_dim": _attn_dim, "dropout": _dropout, "num_heads": _num_heads,
            "projection_dim": _projection_dim, "pooling": _pooling,
            "classifier_hidden": _classifier_hidden,
        }
    if fold_label:
        hyperparams["fold"] = fold_label

    splits_data = _build_splits_yaml(_task_type, train_module, val_module, _class_definitions)
    hyperparams["splits"] = splits_data
    logger.log_hyperparams(hyperparams)

    callbacks = [
        ModelCheckpoint(
            dirpath=fold_save_dir,
            filename="best",
            monitor="validation/loss", mode="min", save_top_k=1,
            save_last=False, verbose=True,
        ),
        # last.ckpt is saved explicitly (final epoch) rather than via
        # ModelCheckpoint(save_last=True), which is monitor-gated and would
        # leave last.ckpt holding the best (not final) weights — see
        # SaveLastCheckpoint docstring.
        SaveLastCheckpoint(dirpath=fold_save_dir),
    ]
    if _early_stopping_patience > 0:
        callbacks.append(EarlyStopping(
            monitor="validation/loss", patience=_early_stopping_patience,
            mode="min", verbose=True,
        ))

    _save_every_n_epochs = config.save_every_n_epochs
    if _save_every_n_epochs is not None:
        callbacks.append(ModelCheckpoint(
            dirpath=fold_save_dir,
            filename="epoch={epoch}",
            every_n_epochs=_save_every_n_epochs,
            save_top_k=-1,
            verbose=True,
        ))

    _save_epochs = config.save_epochs
    if _save_epochs is not None:
        epoch_list = sorted({int(e) for e in str(_save_epochs).split(",") if e.strip()})
        callbacks.append(SaveEpochsCheckpoint(dirpath=fold_save_dir, epochs=epoch_list))

    resolved_precision = auto_detect_precision(_precision)

    trainer = Trainer(
        default_root_dir=fold_save_dir,
        enable_checkpointing=True,
        deterministic=_deterministic, max_epochs=_max_epochs, min_epochs=_min_epochs,
        accelerator=_accelerator, devices=_devices, logger=logger,
        enable_progress_bar=False, log_every_n_steps=_log_every_n_steps,
        callbacks=callbacks, accumulate_grad_batches=_accumulate_grad_batches,
        precision=resolved_precision,
    )

    if _accumulate_grad_batches > 1:
        print(f"{prefix}Gradient accumulation: {_accumulate_grad_batches} batches, "
              f"effective batch size = {_batch_size * _accumulate_grad_batches} images per weight update")

    trainer.fit(model_instance, train_loader, val_loader, ckpt_path=ckpt_path)

    # Load best checkpoint (fall back to last.ckpt if validation/loss was never logged,
    # e.g. LOO mode where the single val experiment can't form a valid batch)
    ckpt_callback = next(c for c in callbacks if isinstance(c, ModelCheckpoint))
    last_callback = next(c for c in callbacks if isinstance(c, SaveLastCheckpoint))
    last_ckpt_path = last_callback.last_model_path if os.path.exists(last_callback.last_model_path) else ""
    best_ckpt_path = ckpt_callback.best_model_path or last_ckpt_path
    if best_ckpt_path:
        label = "best" if ckpt_callback.best_model_path else "last"
        print(f"{prefix}Loading {label} checkpoint from: {best_ckpt_path}")
        model_instance = model_cls.load_from_checkpoint(best_ckpt_path)

    # Save metadata
    _save_training_metadata(
        best_ckpt_path=best_ckpt_path,
        class_config_raw=_class_config_raw,
        database_column=_database_column,
        database_label=_database_label,
        study_id=_study_id,
        negative_study_ids=_negative_study_ids,
        task_type=_task_type,
        num_classes=_num_classes,
        class_definitions=_class_definitions,
        splits_data=splits_data,
    )

    # Save predictions alongside checkpoint
    if _run_infer:
        class_names = None
        if _class_definitions:
            raw_names = [d['name'] for d in _class_definitions if isinstance(d, dict) and 'name' in d]
            class_names = list(dict.fromkeys(raw_names))

        try:
            save_post_training_predictions(
                model_instance=model_instance,
                train_module=train_module,
                val_module=val_module,
                best_ckpt_path=best_ckpt_path,
                model_name=_model_name,
                database_column=_database_column,
                task_type=_task_type,
                num_classes=_num_classes,
                # Evaluate at the bag size the model was trained on, not the full
                # per-step batch_size (which mismatches the trained bag).
                bag_size=_batch_size // _n_experiments,
                seed=_seed,
                class_names=class_names,
                crop_size=config.crop_size,
            )
        except Exception as e:
            print(f"{prefix}Warning: Failed to save post-training predictions: {e}")

    return model_instance, trainer, train_module, val_module


def _save_training_metadata(
    best_ckpt_path: str,
    class_config_raw: dict,
    database_column: str,
    database_label,
    study_id: str,
    negative_study_ids: list,
    task_type: str,
    num_classes: int,
    class_definitions: list,
    splits_data: dict,
):
    """Save classes.yaml and splits.yaml alongside the best checkpoint."""
    if not best_ckpt_path:
        return

    save_dir = os.path.dirname(best_ckpt_path)

    # --- classes.yaml ---
    if class_config_raw:
        classes_data = class_config_raw
    else:
        classes_data = _build_classes_yaml(
            database_column=database_column,
            database_label=database_label,
            study_id=study_id,
            negative_study_ids=negative_study_ids,
            task_type=task_type,
            class_definitions=class_definitions,
            num_classes=num_classes,
        )

    classes_path = os.path.join(save_dir, "classes.yaml")
    with open(classes_path, 'w') as f:
        yaml.dump(classes_data, f, default_flow_style=False, sort_keys=False)
    print(f"Saved class config: {classes_path}")

    splits_path = os.path.join(save_dir, "splits.yaml")
    with open(splits_path, 'w') as f:
        yaml.dump(splits_data, f, default_flow_style=False, sort_keys=False)
    print(f"Saved train/val splits: {splits_path}")


def _build_classes_yaml(database_column, database_label, study_id, negative_study_ids,
                        task_type, class_definitions, num_classes=2):
    """Generate a classes.yaml equivalent from CLI args."""
    if class_definitions:
        # Already have structured class definitions
        return {"classes": class_definitions}

    classes_data = {}

    if task_type == "binary":
        cls_entry = {"name": f"{database_label}+", "study_id": study_id,
                     "column": database_column, "value": str(database_label)}
        classes_data["classes"] = [cls_entry]
        if negative_study_ids:
            classes_data["negative_studies"] = list(negative_study_ids)
    elif task_type == "multiclass":
        classes_data["classes"] = [
            {"name": f"class_{i}", "study_id": study_id, "column": database_column}
            for i in range(num_classes)
        ]

    return classes_data


def _to_native_list(ids):
    """Convert experiment IDs to native Python types for YAML serialization."""
    return [int(x) if isinstance(x, (int, float)) else str(x) for x in ids]


def _build_splits_yaml(task_type, train_module, val_module, class_definitions):
    """Build the splits.yaml from train/val data modules."""
    if task_type == "binary":
        train_split = {}
        val_split = {}

        if hasattr(train_module, 'pos_ids') and hasattr(train_module, 'neg_ids'):
            train_split["positive"] = _to_native_list(train_module.pos_ids)
            train_split["negative"] = _to_native_list(train_module.neg_ids)
        else:
            train_split["experiment_ids"] = _to_native_list(train_module.data['experiment_id'].tolist())

        if hasattr(val_module, 'pos_ids') and hasattr(val_module, 'neg_ids'):
            val_split["positive"] = _to_native_list(val_module.pos_ids)
            val_split["negative"] = _to_native_list(val_module.neg_ids)
        else:
            val_split["experiment_ids"] = _to_native_list(val_module.data['experiment_id'].tolist())

        return {"train": train_split, "val": val_split}

    elif task_type == "multiclass":
        train_split = {}
        val_split = {}

        if hasattr(train_module, 'class_experiment_ids'):
            for class_name, exp_ids in train_module.class_experiment_ids.items():
                train_split[str(class_name)] = _to_native_list(exp_ids)
        else:
            train_split["experiment_ids"] = _to_native_list(train_module.data['experiment_id'].tolist())

        if hasattr(val_module, 'class_experiment_ids'):
            for class_name, exp_ids in val_module.class_experiment_ids.items():
                val_split[str(class_name)] = _to_native_list(exp_ids)
        else:
            val_split["experiment_ids"] = _to_native_list(val_module.data['experiment_id'].tolist())

        return {"train": train_split, "val": val_split}


def save_post_training_predictions(
    model_instance,
    train_module,
    val_module,
    best_ckpt_path: str,
    model_name: str,
    database_column: str,
    task_type: str = "binary",
    num_classes: int = 2,
    bag_size: int = 200,
    seed: int = 403,
    class_names: list = None,
    crop_size: int = None,
):
    """Run inference on train+val splits and save predictions alongside checkpoint.

    Called automatically after training. Saves predictions.npz, .metadata.json,
    and optionally .attention.npz and .embeddings.npz.
    """
    if not best_ckpt_path:
        print("No best checkpoint path — skipping prediction saving")
        return

    from remo_id.inference.results import save_predictions
    from remo_id.inference.core import get_augmentations, set_inference_seeds

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_instance.eval()
    model_instance.to(device)
    set_inference_seeds(seed)

    backbone_name = getattr(model_instance, 'backbone_name', get_augmentation_backbone(model_name))
    val_augments = get_augmentations(backbone_name, train=False, crop_size=crop_size)

    # Ensure both modules use validation augmentations
    train_module.dataloader.augments = val_augments
    val_module.dataloader.augments = val_augments

    is_mil = hasattr(model_instance, 'forward_with_attention')
    all_results = []
    attention_weights = {}
    attention_indices = {}
    embeddings = {}
    per_image_features = {}

    for split_name, module in [("train", train_module), ("val", val_module)]:
        df = module.data.copy()
        df["experiment_id"] = df["experiment_id"].astype(str)
        df["_has_null_label"] = False  # training data has known labels
        df = df.sort_values("experiment_id").reset_index(drop=True)

        if len(df) == 0:
            continue

        print(f"\nSaving {split_name} predictions ({len(df)} experiments)...")

        if is_mil:
            from remo_id.inference.mil import _predict_experiments
            results, _, _, _, _, captured = _predict_experiments(
                model=model_instance, ds=module, df=df,
                database_column=database_column, task_type=task_type,
                num_classes=num_classes, bag_size=bag_size, seed=seed,
                device=device, class_names=class_names,
                capture_embeddings=True, capture_features=True,
                n_top_attention=0, n_bottom_attention=0,
            )
        else:
            from remo_id.inference.instance import _predict_experiments_instance
            # Instance models need y_true column
            df["y_true"] = df[database_column]
            in_channels = getattr(model_instance, 'in_channels', 1)
            results, _, _, _ = _predict_experiments_instance(
                model=model_instance, df=df,
                database_column=database_column, task_type=task_type,
                num_classes=num_classes, bag_size=bag_size, seed=seed,
                device=device, augments=val_augments, in_channels=in_channels,
                class_names=class_names,
            )

        # Tag results with split
        for r in results:
            r.split = split_name

        # Collect embeddings, features, attention from MIL results
        if is_mil:
            for r in results:
                if r.feat_emb is not None:
                    embeddings[f"exp_{r.experiment_id}"] = r.feat_emb
            if captured:
                per_image_features.update(captured.get('features', {}))
                attention_weights.update(captured.get('attention_weights', {}))
                attention_indices.update(captured.get('attention_indices', {}))

        all_results.extend(results)

    if not all_results:
        print("No predictions to save")
        return

    # Build metadata
    metadata = {
        "task_type": task_type,
        "num_classes": num_classes,
        "class_names": class_names,
        "model_name": model_name,
        "seed": seed,
        "bag_size": bag_size,
        "is_mil": is_mil,
        "backbone_name": backbone_name,
        "crop_size": crop_size,
    }

    save_predictions(
        checkpoint_path=best_ckpt_path,
        results=all_results,
        metadata=metadata,
        attention_weights=attention_weights if attention_weights else None,
        attention_indices=attention_indices if attention_indices else None,
        embeddings=embeddings if embeddings else None,
        per_image_features=per_image_features if per_image_features else None,
    )
