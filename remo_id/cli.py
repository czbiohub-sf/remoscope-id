"""Click-based CLI for remo-id."""

import os

import click
import sys
import yaml

from remo_id.models import list_models

MODEL_CHOICES = list_models()


@click.group()
@click.version_option(package_name="remo-id")
def cli():
    """remo-id: Deep learning for microscopy image classification.

    Supports binary and multi-class classification of Remoscope
    blood-microscopy experiments. Describe your data with a CSV
    manifest (--manifest); see remo_id/data/manifest.py for the format.

    Use 'remo-id COMMAND --help' for more information on each command.
    """
    pass


# --- Shared option groups ---
def _labeling_options(f):
    """Options shared between train, infer, gallery, and sample commands."""
    f = click.option("--database_column", default=None, help="Column to label from (study table if --study_id set, else experiment table)")(f)
    f = click.option("--database_label", default=None, help="Categorical label to treat as positive (e.g. 'true')")(f)
    f = click.option("--study_id", default=None, help="Study ID for label lookup")(f)
    f = click.option("--threshold", type=float, default=None, help="Threshold for numeric label columns")(f)
    f = click.option("--threshold_mode", type=click.Choice([">", "<", "=="]), default="==", help="Threshold direction")(f)
    f = click.option("--negative_study_ids", multiple=True, default=[], help="Study IDs to merge as guaranteed negatives")(f)
    f = click.option("--negative_experiment_ids", multiple=True, type=int, default=[], help="Experiment IDs to merge as guaranteed negatives")(f)
    f = click.option("--skip_null_labels/--no-skip_null_labels", default=True, help="Exclude experiments with null label values (default: True)")(f)
    return f


def _task_options(f):
    """Task type options shared between train and infer."""
    f = click.option("--task_type", type=click.Choice(["binary", "multiclass"]), default="binary", help="Task type (inferred from --manifest/--class_config)")(f)
    f = click.option("--num_classes", type=int, default=2, help="Number of classes for multiclass mode")(f)
    return f


def _manifest_options(f):
    """CSV manifest data source (see remo_id/data/manifest.py)."""
    f = click.option("--classes", default=None, help="Manifest: comma-separated class names in index order (default: sorted unique labels). With two classes the second is the positive class.")(f)
    f = click.option("--label_column", default="label", show_default=True, help="Manifest: column holding the label")(f)
    f = click.option("--manifest", type=click.Path(exists=True, dir_okay=False), default=None, help="CSV manifest with experiment_id, path, num_images, label [, sample_id, study_id, site]. Replaces --study_id/--class_config/--database_column.")(f)
    return f


def _class_source_options(f):
    """--class_config or --manifest (one is required)."""
    f = _manifest_options(f)
    f = click.option("--class_config", type=click.Path(exists=True), default=None, help="Path to YAML class config (or use --manifest)")(f)
    return f


def _parse_classes(classes):
    """'A,B,C' -> ['A', 'B', 'C'] (None stays None)."""
    if classes is None or isinstance(classes, (list, tuple)):
        return list(classes) if classes is not None else None
    return [c.strip() for c in classes.split(",") if c.strip()]


def _manifest_class_config(kwargs, default_classes=None) -> dict:
    """Class-config dict for a classification --manifest (rejects conflicting options).

    ``default_classes`` (e.g. a checkpoint's class names) is used when --classes
    is not given; those classes need not all occur in the manifest.
    """
    from remo_id.data.class_config import validate_class_config_conflicts
    from remo_id.data.manifest import manifest_class_config
    validate_class_config_conflicts(kwargs, source="--manifest")
    classes = _parse_classes(kwargs.get("classes"))
    return manifest_class_config(
        kwargs["manifest"], kwargs.get("label_column", "label"), classes or default_classes,
        require_present=classes is not None or default_classes is None)


def _resolve_class_config_path(kwargs, out_dir) -> str:
    """Return --class_config, or write the class config implied by --manifest to
    ``{out_dir}/manifest_classes.yaml`` and return that path."""
    if kwargs.get("manifest"):
        raw = _manifest_class_config(kwargs)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "manifest_classes.yaml")
        # Atomic: parallel array tasks sharing out_dir write the same file.
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
        os.replace(tmp, path)
        print(f"Class config for --manifest: {path}")
        return path
    if not kwargs.get("class_config"):
        raise click.UsageError("Provide --class_config or --manifest.")
    return kwargs["class_config"]


@cli.command()
@_labeling_options
@_task_options
@_manifest_options
@click.option("--class_config", type=click.Path(exists=True), default=None, help="Path to YAML class configuration file for multi-study multiclass")
@click.option("--label_contains", default=None, help="Fuzzy match: positive if column value contains this string (case-insensitive)")
@click.option("--study_ids", multiple=True, default=[], help="Filter to experiments from these studies")
@click.option("--val_ratio", type=float, default=0.2, help="Fraction of experiments for validation")
@click.option("--batch_size", type=int, default=200, help="Total images per batch")
@click.option("--n_experiments", type=int, default=8, help="Number of experiment bags per batch (MIL models)")
@click.option("--num_batches", type=int, default=100, help="Number of batches per training epoch")
@click.option("--model", type=click.Choice(MODEL_CHOICES), default="resnet18", help="Model architecture")
@click.option("--project", default="remo-id", help="W&B project name")
@click.option("--save_dir", default=None, help="Directory to save checkpoints (default: $REMO_ID_SAVE_DIR or ./checkpoints)")
@click.option("--max_epochs", type=int, default=100, help="Maximum training epochs")
@click.option("--min_epochs", type=int, default=None, help="Minimum epochs before early stopping can trigger")
@click.option("--log_every_n_steps", type=int, default=50, help="Log training metrics every N steps")
@click.option("--accelerator", default="auto", help="Training accelerator (auto, gpu, cpu)")
@click.option("--devices", type=int, default=1, help="Number of devices")
@click.option("--seed", type=int, default=403, help="Random seed")
@click.option("--val_seeds", type=str, default=None, help="Comma-separated seeds (e.g. '403,404,405') to run several held-out val splits in ONE job, reusing the loaded embedding store (mutually exclusive with --val_experiment_ids). Each seed trains a fresh model into save_dir/seed_<s>/")
@click.option("--lr", type=float, default=1e-4, help="Learning rate")
@click.option("--lr_schedule", type=click.Choice(["none", "cosine"]), default="cosine", help="Learning rate schedule")
@click.option("--warmup_epochs", type=int, default=5, help="Number of warmup epochs for LR schedule")
@click.option("--attn_dim", type=int, default=128, help="Attention hidden dimension (MIL models)")
@click.option("--pretrained/--no-pretrained", default=True, help="Use pretrained backbone weights (ImageNet; DINOv3 for dino-* models)")
@click.option("--dino_weights", type=click.Path(exists=True, dir_okay=False), default=None, help="dino-* models: DINOv3 checkpoint file (Meta DINOv3 License; not bundled). Default: $REMO_ID_DINO_WEIGHTS")
@click.option("--dropout", type=float, default=0.3, help="Dropout rate (MIL models)")
@click.option("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
@click.option("--finetune_layers", type=str, default="layer3,layer4", help="Backbone layers to fine-tune ('all'=full fine-tune, 'none'=freeze all, or comma-separated layer names)")
@click.option("--early_stopping_patience", type=int, default=15, help="Early stopping patience in epochs")
@click.option("--num_heads", type=int, default=1, help="Number of attention heads (MIL models)")
@click.option("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients over N batches")
@click.option("--cross_experiment_sampling", is_flag=True, default=False, help="Mix images from multiple experiments per bag")
@click.option("--sample_aware_sampling", is_flag=True, default=False, help="Pool training draws by sample_id: each epoch every sample is drawn once via one random technical replicate; the sample's other replicates sit out until the pool refreshes")
@click.option("--sample_aware_mode", type=click.Choice(["single", "pooled", "pooled_random"]), default="single", help="With --sample_aware_sampling: 'single'=one random replicate per bag; 'pooled'=even split across ALL replicates (constant bag composition, which a model can memorize; prefer pooled_random); 'pooled_random'=random replicate subset + Dirichlet proportions per bag (within-bag technical augmentation, no constant fingerprint)")
@click.option("--max_experiments_per_bag", type=int, default=None, help="Max experiments to mix per bag (with --cross_experiment_sampling)")
@click.option("--majority_min_pct", type=float, default=1.0, help="Min fraction of majority/positive-experiment images per bag. 1.0=single-class bags; multiclass: 0.6 mixes ~40%% from other classes; binary: 0.5 = positive bags >=50%% positive-experiment images (negative bags stay 100%% negative). Triggers the mixed sampler.")
@click.option("--bag_label_mode", type=click.Choice(["hard", "soft"]), default="hard", help="'hard'=CE on majority class; 'soft'=LLP soft-CE on proportion vector (multiclass only)")
@click.option("--attn_entropy_weight", type=float, default=0.0, help="Weight for attention-entropy regularizer (soft mode + attention/topk pooling only)")
@click.option("--projection_dim", type=int, default=None, help="Projection dimension to reduce features before aggregation (MIL models)")
@click.option("--pooling", type=click.Choice(["attention", "mean", "max", "linear", "topk", "transformer", "additive"]), default="attention", help="Pooling method for MIL aggregation (transformer = self-attention over cells; additive = exact per-cell signed contributions)")
@click.option("--top_k", type=int, default=None, help="Top-k instances for topk pooling (required with --pooling topk)")
@click.option("--transformer_layers", type=int, default=2, help="Number of self-attention encoder layers for --pooling transformer")
@click.option("--classifier_hidden", type=int, default=None, help="Classifier hidden dim (0=linear only, None=default, N=custom)")
@click.option("--encode_chunk_size", type=int, default=None, help="Chunk size for backbone encoding to limit GPU memory (MIL models). Images are encoded in chunks then reassembled for attention.")
@click.option("--patch_pool/--no-patch_pool", default=False, help="Use attention-pooled patch tokens instead of CLS token (DINO models only)")
@click.option("--split_by_sample_id/--no-split_by_sample_id", default=True, help="Split train/val by sample_id to prevent leakage (default: True)")
@click.option("--precision", type=click.Choice(["auto", "32", "16-mixed", "bf16-mixed"]), default="auto", help="Training precision (auto detects GPU capability)")
@click.option("--compile/--no-compile", default=False, help="Use torch.compile for model optimization")
@click.option("--deterministic/--no-deterministic", default=True, help="Deterministic training (disable for speed with cudnn.benchmark)")
@click.option("--infer/--no-infer", default=True, help="Save post-training predictions, embeddings, attention, and per-image features")
@click.option("--analysis/--no-analysis", default=False, help="Run repeatability analysis: log figures to wandb, save metrics.json")
@click.option("--analysis_bag_size", type=int, default=None, help="Bag size for analysis inference (default: training batch_size)")
@click.option("--analysis_n_runs", type=int, default=5, help="Number of resampling passes for repeatability")
@click.option("--n_folds", type=int, default=1, help="Number of CV folds (1=no CV, default)")
@click.option("--n_repeats", type=int, default=1, help="Number of CV repeats (default 1)")
@click.option("--n_gpus", type=int, default=None, help="Max GPUs for parallel CV (default: one per fold)")
@click.option("--partition", default="gpu", show_default=True, help="SLURM partition for the CV array job (CV/LOO submitted from outside SLURM)")
@click.option("--loo/--no-loo", default=False, help="Leave-one-out CV (1 fold per experiment, no early stopping)")
@click.option("--save_every_n_epochs", type=int, default=None, help="Save a checkpoint every N epochs (in addition to best checkpoint)")
@click.option("--save_epochs", type=str, default=None, help="Comma-separated explicit epochs to checkpoint, e.g. '0,1,2,3,5,10,15,25,35,50'. Epoch 0 = post-init/pretrained snapshot. Coexists with --save_every_n_epochs.")
@click.option("--resume/--no-resume", default=False, help="Resume from last.ckpt in save_dir if it exists")
@click.option("--prepare_only/--no-prepare_only", default=False, help="Create CV dir + splits only, do not submit jobs (for external job orchestration)")
@click.option("--crop_size", type=int, default=None, help="Override default random crop size for augmentation (backbone-specific default if not set)")
@click.option("--background_correction", type=click.Choice(["none", "own", "random", "blend", "swap"]), default="none", help="Flat-field correction mode using p99 background images")
@click.option("--background_value", type=int, default=None, help="Background pixel value for normalization (default 225). Scalar divisor for none mode, fallbacks, and val/inference")
@click.option("--intensity_jitter", type=float, default=None, help="ColorJitter brightness/contrast range for batch effect reduction")
@click.option("--noise_std", type=float, default=None, help="Max std for random Gaussian noise augmentation")
@click.option("--blur_sigma", type=float, default=None, help="Max sigma for random Gaussian blur augmentation")
@click.option("--gradient_blur_sigma", type=float, default=None, help="Max sigma for spatially-varying gradient blur (simulates sample tilt)")
@click.option("--erase_num", type=str, default=None, help="Random-erasing count override as 'MIN,MAX' (e.g. '0,50'; default: backbone-specific, resnet50=10,25)")
@click.option("--erase_scale", type=str, default=None, help="Random-erasing per-mask area fraction override as 'MIN,MAX' (e.g. '0.001,0.05'; default: backbone-specific, resnet50=0.001,0.01)")
@click.option("--min_image_index", type=int, default=0, help="Skip images below this index per experiment (e.g. 1000 to exclude early warm-up images)")
@click.option("--val_experiment_ids", multiple=True, type=str, default=[], help="Hold out these experiments for validation (overrides --val_ratio)")
@click.option("--permute_labels/--no-permute_labels", default=False, help="Negative-control sanity check: randomly permute class labels across patients (sample_id) before splitting. Balance-preserving. Expected result: val performance collapses to chance.")
@click.option("--permute_labels_seed", type=int, default=404, help="Seed for the label permutation (orthogonal to --seed) so the shuffle is reproducible.")
@click.option("--model_version", type=str, default=None, help="Embedding-store version tag (required for --model embedding-mil). Reads {exp_id}_emb__<version>.zarr.")
@click.option("--cells_per_bag", type=int, default=None, help="Embedding-MIL: cells sampled per bag. Sets batch_size = cells_per_bag * n_experiments (default: batch_size // n_experiments).")
@click.option("--cell_config", type=str, default=None, help="Embedding-MIL: cell_config YAML defining cell classes (regionprops gates) + per-class sampling weights + pool_cap (see examples/cell_configs/). Optional: without it all embeddings form one pool and no regionprops are needed.")
@click.option("--pool_cap", type=int, default=None, help="Embedding-MIL: max distinct embeddings kept per cell class per experiment (default: cell_config pool_cap, else 200000)")
@click.option("--load_workers", type=int, default=None, help="Embedding-MIL: thread-pool size for the parallel store load. Bounds RAM (~workers * full-array size); lower it if loading OOMs. Default min(SLURM_CPUS, 8).")
def train(**kwargs):
    """Train a classification model.

    \b
    Examples:
        remo-id train --manifest manifest.csv --model resnet18-mil
        remo-id train --manifest manifest.csv --classes negative,positive --model resnet18-mil --loo
        remo-id train --manifest manifest.csv --model embedding-mil --model_version my_embedder
        remo-id train --class_config classes.yaml --model resnet18-mil --n_folds 10 --n_repeats 5
    """
    import torch

    run_infer = kwargs.pop('infer', True)
    run_analysis = kwargs.pop('analysis', False)
    analysis_bag_size = kwargs.pop('analysis_bag_size', None)
    analysis_n_runs = kwargs.pop('analysis_n_runs', 5)
    n_folds = kwargs.pop('n_folds', 1)
    n_repeats = kwargs.pop('n_repeats', 1)
    n_gpus = kwargs.pop('n_gpus', None)
    loo = kwargs.pop('loo', False)
    resume = kwargs.pop('resume', False)
    partition = kwargs.pop('partition', 'gpu')

    from remo_id.training.config import default_save_dir
    kwargs['save_dir'] = kwargs['save_dir'] or default_save_dir()

    kwargs['run_infer'] = run_infer
    class_config_path = kwargs.pop('class_config', None)

    kwargs['study_ids'] = list(kwargs['study_ids'])
    kwargs['negative_study_ids'] = list(kwargs['negative_study_ids'])
    kwargs['negative_experiment_ids'] = list(kwargs['negative_experiment_ids'])
    kwargs['val_experiment_ids'] = list(kwargs['val_experiment_ids'])
    kwargs['class_definitions'] = None
    kwargs['classes'] = _parse_classes(kwargs.get('classes'))

    manifest = kwargs.get('manifest')
    if manifest is not None and class_config_path is not None:
        raise click.UsageError("--manifest and --class_config are mutually exclusive.")
    if manifest is None and (kwargs['classes'] or kwargs['label_column'] != 'label'):
        raise click.UsageError("--classes/--label_column require --manifest.")

    if manifest is not None:
        kwargs['manifest'] = os.path.abspath(manifest)
        kwargs['exclude_filters'] = []
        kwargs['include_filters'] = []
        kwargs['default_class'] = None
        from remo_id.data.class_config import class_config_from_dict
        cfg = class_config_from_dict(_manifest_class_config(kwargs))
        kwargs['task_type'] = cfg.task_type
        kwargs['num_classes'] = cfg.num_classes
        kwargs['database_column'] = 'class_label'
        kwargs['database_label'] = None
        kwargs['class_definitions'] = cfg.class_definitions
        kwargs['class_config_raw'] = cfg.raw_config
        print(f"Manifest {manifest}: classes {cfg.class_names} "
              f"(task_type={cfg.task_type}, num_classes={cfg.num_classes})")
    # Load and validate class_config YAML if provided
    elif class_config_path is not None:
        from remo_id.data.class_config import parse_class_config, validate_class_config_conflicts
        validate_class_config_conflicts(kwargs)

        cfg = parse_class_config(class_config_path)

        # Validate class_config structure
        for i, cls in enumerate(cfg.class_definitions):
            if 'name' not in cls:
                raise click.UsageError(f"Class entry {i} missing required 'name' field.")
            if 'study_id' not in cls and 'experiment_ids' not in cls and 'manifest' not in cls:
                raise click.UsageError(f"Class entry {i} ('{cls['name']}') must have 'study_id', 'experiment_ids' or 'manifest'.")

        # Validate default_class references an existing class name
        if cfg.default_class is not None and cfg.default_class not in cfg.class_names:
            raise click.UsageError(
                f"default_class '{cfg.default_class}' not found in class names: {cfg.class_names}"
            )

        kwargs['task_type'] = cfg.task_type
        kwargs['num_classes'] = cfg.num_classes
        if cfg.task_type == 'binary' and len(cfg.class_names) == 1:
            kwargs['negative_study_ids'] = cfg.negative_study_ids

        # Use synthetic column name
        kwargs['database_column'] = 'class_label'
        kwargs['database_label'] = None
        kwargs['study_id'] = None
        kwargs['study_ids'] = []

        # Pass parsed config to training
        kwargs['class_definitions'] = cfg.class_definitions
        kwargs['exclude_filters'] = cfg.exclude_filters
        kwargs['include_filters'] = cfg.include_filters
        kwargs['default_class'] = cfg.default_class
        kwargs['class_config_raw'] = cfg.raw_config

        print(f"Loaded class config from {class_config_path}")
        print(f"  Classes: {cfg.class_names}")
        print(f"  Task type: {kwargs['task_type']}, num_classes: {kwargs['num_classes']}")
    else:
        # Validate that database_column is provided when not using class_config
        if not kwargs.get('database_column'):
            raise click.UsageError("Provide --manifest, --class_config, or --database_column.")
        kwargs['exclude_filters'] = []
        kwargs['include_filters'] = []
        kwargs['default_class'] = None
        kwargs['class_config_raw'] = None

    # Validate topk pooling requires --top_k
    if kwargs.get('pooling') == 'topk' and kwargs.get('top_k') is None:
        raise click.UsageError("--pooling topk requires --top_k to be set")

    from remo_id.training import run_training
    from remo_id.training.train import build_data_module, coerce_database_label

    if torch.cuda.is_available():
        print("CUDA is available. GPU memory summary before training:")
        print(torch.cuda.memory_summary())
    else:
        print("CUDA is not available. Training will be on CPU.")

    prepare_only = kwargs.pop('prepare_only', False)

    if n_folds > 1 or loo:
        # K-fold or leave-one-out cross-validation mode
        from remo_id.workflows.cv import run_cross_validation, prepare_cv_dir
        from remo_id.training.cross_validation import CVConfig

        db_label = coerce_database_label(kwargs.get('database_label'))
        data_module = build_data_module(
            model=kwargs['model'],
            database_column=kwargs['database_column'],
            database_label=db_label,
            label_contains=kwargs.get('label_contains'),
            study_id=kwargs.get('study_id'),
            study_ids=kwargs.get('study_ids'),
            threshold=kwargs.get('threshold'),
            threshold_mode=kwargs.get('threshold_mode', '=='),
            negative_study_ids=kwargs.get('negative_study_ids', []),
            negative_experiment_ids=kwargs.get('negative_experiment_ids', []),
            batch_size=kwargs.get('batch_size', 200),
            task_type=kwargs.get('task_type', 'binary'),
            num_classes=kwargs.get('num_classes', 2),
            class_definitions=kwargs.get('class_definitions'),
            skip_null_labels=kwargs.get('skip_null_labels', True),
            exclude_filters=kwargs.get('exclude_filters', []),
            include_filters=kwargs.get('include_filters', []),
            default_class=kwargs.get('default_class'),
            background_correction=kwargs.get('background_correction', 'none'),
            background_value=kwargs.get('background_value'),
            intensity_jitter=kwargs.get('intensity_jitter'),
            noise_std=kwargs.get('noise_std'),
            blur_sigma=kwargs.get('blur_sigma'),
            gradient_blur_sigma=kwargs.get('gradient_blur_sigma'),
            min_image_index=kwargs.get('min_image_index', 0),
            crop_size=kwargs.get('crop_size'),
            model_version=kwargs.get('model_version'),
            cell_config=kwargs.get('cell_config'),
            load_workers=kwargs.get('load_workers'),
            pool_cap=kwargs.get('pool_cap'),
        )

        if loo:
            # Count unique sample_ids (null sample_ids count as individual samples)
            sample_ids = data_module.data["sample_id"]
            n_unique = sample_ids.nunique(dropna=True) + sample_ids.isna().sum()
            n_folds = int(n_unique)
            n_repeats = 1
            kwargs['early_stopping_patience'] = 0
            n_exp = len(data_module.data)
            print(f"LOO mode: {n_exp} experiments, {n_folds} unique samples → {n_folds} folds, early stopping disabled")

        cv_config = CVConfig(
            n_folds=n_folds,
            n_repeats=n_repeats,
            seed=kwargs.get('seed', 403),
        )

        # On login node (no SLURM_JOB_ID): prepare CV dir + splits, submit array job
        # Inside SLURM job: run folds assigned to this array task
        in_slurm = os.environ.get('SLURM_JOB_ID') is not None
        if not in_slurm:
            cv_dir = prepare_cv_dir(
                data_module=data_module,
                cv_config=cv_config,
                save_dir=kwargs['save_dir'],
                model_name=kwargs['model'],
                loo=loo,
            )
            if prepare_only:
                print(f"N_FOLDS: {n_folds}")
                return
            _submit_cv_array_job(cv_dir, n_folds, n_repeats, kwargs, loo, class_config_path, n_gpus,
                                 partition=partition)
            return

        cv_results = run_cross_validation(
            data_module=data_module,
            cv_config=cv_config,
            save_dir=kwargs['save_dir'],
            train_kwargs=kwargs,
            loo=loo,
        )
        if loo:
            print(f"\nLeave-one-out CV complete: {n_folds} folds")
        else:
            print(f"\nCross-validation complete: {n_folds} folds x {n_repeats} repeats")
    else:
        # Standard single-split training — create named directory before GPU work
        slurm_id = os.environ.get('SLURM_JOB_ID')
        if slurm_id:
            subdir = f"{slurm_id}_{kwargs['model']}"
        else:
            from datetime import datetime
            subdir = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{kwargs['model']}"
        kwargs['save_dir'] = os.path.join(kwargs['save_dir'], subdir)
        os.makedirs(kwargs['save_dir'], exist_ok=True)
        print(f"Save directory: {kwargs['save_dir']}")

        ckpt_path = None
        if resume:
            last_ckpt = os.path.join(kwargs['save_dir'], 'last.ckpt')
            if os.path.exists(last_ckpt):
                print(f"Resuming from: {last_ckpt}")
                ckpt_path = last_ckpt
            else:
                print(f"No last.ckpt in {kwargs['save_dir']} — starting fresh")
        kwargs['ckpt_path'] = ckpt_path

        model, trainer, train_module, val_module = run_training(**kwargs)

        if torch.cuda.is_available():
            print(torch.cuda.memory_summary())

        if run_analysis:
            from remo_id.analysis import run_post_training_analysis
            from lightning.pytorch.callbacks import ModelCheckpoint

            ckpt_callbacks = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)]
            best_ckpt = ckpt_callbacks[0].best_model_path if ckpt_callbacks else None

            analysis_class_names = None
            if kwargs.get('class_definitions'):
                raw_names = [d['name'] for d in kwargs['class_definitions'] if isinstance(d, dict) and 'name' in d]
                analysis_class_names = list(dict.fromkeys(raw_names))

            run_post_training_analysis(
                model=model, train_module=train_module, val_module=val_module,
                task_type=kwargs.get('task_type', 'binary'),
                num_classes=kwargs.get('num_classes', 2),
                class_names=analysis_class_names,
                n_images=analysis_bag_size or kwargs.get('batch_size', 200),
                n_runs=analysis_n_runs,
                checkpoint_path=best_ckpt,
                crop_size=kwargs.get('crop_size'),
            )

        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


def _submit_cv_array_job(cv_dir: str, n_folds: int, n_repeats: int, kwargs: dict, loo: bool,
                         class_config_path: str = None, n_gpus: int = None, partition: str = "gpu"):
    """Generate and submit an sbatch array job for parallel CV training."""
    import subprocess

    total_folds = n_folds * n_repeats
    if n_gpus is None:
        n_gpus = total_folds
    n_gpus = min(n_gpus, total_folds)

    # Keys set by class_config parsing — skip these if class_config is used
    class_config_derived = {
        'database_column', 'database_label', 'study_id', 'study_ids',
        'task_type', 'num_classes', 'negative_study_ids', 'negative_experiment_ids',
    }
    # Keys that are internal / not CLI options
    skip_keys = {
        'save_dir', 'class_definitions', 'exclude_filters', 'include_filters', 'default_class',
        'class_config_raw', 'run_infer', 'cv_dir',
    }
    if class_config_path:
        skip_keys.update(class_config_derived)
    elif kwargs.get('manifest'):
        # Labels come from the manifest; task_type/num_classes are re-derived.
        skip_keys.update(class_config_derived - {'task_type', 'num_classes'})

    # Toggle flags: --flag/--no-flag
    toggle_flags = {
        'pretrained', 'split_by_sample_id', 'compile', 'deterministic',
        'skip_null_labels',
    }
    # Simple flags: --flag only (is_flag=True), skip if False
    simple_flags = {'cross_experiment_sampling', 'sample_aware_sampling'}

    train_args = ["--no-analysis"]
    if class_config_path:
        train_args.append(f"--class_config {os.path.abspath(class_config_path)}")

    for key, value in kwargs.items():
        if key in skip_keys or value is None:
            continue
        cli_key = f"--{key}"
        if key == 'classes':
            train_args.append(f"{cli_key} {','.join(value)}")
        elif key in toggle_flags:
            if value:
                train_args.append(cli_key)
            else:
                train_args.append(f"--no-{key}")
        elif key in simple_flags:
            if value:
                train_args.append(cli_key)
        elif isinstance(value, (list, tuple)):
            for v in value:
                train_args.append(f"{cli_key} {v}")
        elif isinstance(value, bool):
            if value:
                train_args.append(cli_key)
        elif value == 0 or value:
            train_args.append(f"{cli_key} {value}")

    train_cmd = " \\\n    ".join(train_args)

    script_path = os.path.join(cv_dir, "cv_array.sh")
    script = f"""#!/bin/bash
#SBATCH --job-name=remo-id-cv
#SBATCH --array=0-{n_gpus - 1}
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=14
#SBATCH --mem=128G
#SBATCH --partition={partition}
#SBATCH --gpus-per-node=1
#SBATCH --output={cv_dir}/slurm-outputs/%A_%a.out

# Total: {total_folds} folds ({n_repeats} repeats x {n_folds} folds) across {n_gpus} GPUs
# Folds distributed round-robin by SLURM_ARRAY_TASK_ID

env | grep "^SLURM" | sort
nvidia-smi

# W&B is configured through its environment variables (WANDB_API_KEY,
# WANDB_ENTITY, WANDB_BASE_URL, WANDB_MODE, ...), inherited from the submitting shell.

echo "Array task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"

python -m remo_id.cli train \\
    {train_cmd} \\
    --n_folds {n_folds} \\
    --n_repeats {n_repeats} \\
    --save_dir {cv_dir}{"  --loo" if loo else ""}
"""

    os.makedirs(os.path.join(cv_dir, "slurm-outputs"), exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script)

    folds_per_gpu = total_folds // n_gpus
    extra = f"-{folds_per_gpu + 1}" if total_folds % n_gpus else ""
    print(f"\nSubmitting CV array job: {total_folds} folds across {n_gpus} GPUs (~{folds_per_gpu}{extra} folds/GPU)")
    print(f"  Output dir: {cv_dir}")
    print(f"  Script: {script_path}")

    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  {result.stdout.strip()}")
    else:
        print(f"  sbatch failed: {result.stderr.strip()}")
        sys.exit(1)


@cli.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True), help="Path to model checkpoint")
@click.option("--model_name", type=click.Choice(MODEL_CHOICES), default="resnet18", help="Model architecture")
@click.option("--model_type", type=click.Choice(["instance", "mil", "auto"]), default="auto", help="Model type (auto-detect if not specified)")
@click.option("--study_id", default=None, help="Study ID to run inference on (database)")
@click.option("--class_config", type=click.Path(exists=True), default=None, help="Path to YAML class config (same as training, e.g. the checkpoint's classes.yaml)")
@_manifest_options
@click.option("--database_column", default=None, help="Column for labeling (with --study_id)")
@click.option("--database_label", default=None, help="Categorical label to treat as positive")
@click.option("--threshold", type=float, default=None, help="Threshold for labeling")
@click.option("--threshold_mode", type=click.Choice([">", "<", "=="]), default="==", help="Threshold direction")
@click.option("--task_type", type=click.Choice(["binary", "multiclass"]), default="binary", help="Task type")
@click.option("--num_classes", type=int, default=2, help="Number of classes for multiclass")
@click.option("--experiment_ids", multiple=True, type=str, default=[], help="Specific experiment IDs to evaluate")
@click.option("--experiment_ids_file", type=click.Path(exists=True), default=None, help="File with experiment IDs (one per line)")
@click.option("--bag_size", type=int, default=200, help="Number of images per experiment/bag")
@click.option("--image_thresh", type=float, default=0.5, help="Image-level threshold for positive prediction")
@click.option("--exp_pos_pct", type=float, default=None, help="Experiment positive if %% positive images >= this")
@click.option("--pred_threshold", type=float, default=0.5, help="Prediction threshold for MIL models")
@click.option("--output_path", default="./reports/", help="Output directory for reports")
@click.option("--seed", type=int, default=403, help="Random seed")
@click.option("--negative_study_ids", multiple=True, default=[], help="Study IDs to merge as negatives")
@click.option("--negative_experiment_ids", multiple=True, type=int, default=[], help="Experiment IDs to merge as negatives")
@click.option("--skip_null_labels/--no-skip_null_labels", default=True, help="Exclude experiments with null label values (default: True)")
@click.option("--gradcam_max_experiments", type=int, default=8, help="Max experiments for Grad-CAM visualization")
@click.option("--sample_mode", type=click.Choice(["random", "first", "last"]), default="random", help="Sampling mode for image selection")
@click.option("--sample_percent", type=float, default=100.0, help="Percentage of experiment to sample from (with first/last mode)")
@click.option("--signed_gradcam", is_flag=True, default=False, help="Use signed Grad-CAM (red=for, blue=against)")
@click.option("--normalize_global", is_flag=True, default=False, help="Normalize GradCAM globally across all images")
@click.option("--scan_size", type=int, default=None, help="Two-stage: scan this many images first (MIL only)")
@click.option("--top_k", type=int, default=None, help="Two-stage: select top-k instances by attention for final prediction (MIL only)")
def infer(**kwargs):
    """Run inference on a trained model.

    \b
    Examples:
        remo-id infer --checkpoint model.ckpt --manifest test_manifest.csv
        remo-id infer --checkpoint model.ckpt --class_config checkpoints/run/classes.yaml
    """
    # Convert tuple options to lists
    kwargs['experiment_ids'] = list(kwargs['experiment_ids'])
    kwargs['negative_study_ids'] = list(kwargs['negative_study_ids'])
    kwargs['negative_experiment_ids'] = list(kwargs['negative_experiment_ids'])

    # Auto-detect task_type, num_classes, model_name from checkpoint
    from remo_id.inference.core import auto_detect_from_checkpoint
    ckpt_meta = auto_detect_from_checkpoint(kwargs['checkpoint'])

    auto_detected = []
    if kwargs.get('model_name') == 'resnet18' and ckpt_meta.get('model_name'):
        kwargs['model_name'] = ckpt_meta['model_name']
        auto_detected.append(f"model_name={ckpt_meta['model_name']}")
    if kwargs.get('task_type') == 'binary' and ckpt_meta.get('task_type') != 'binary':
        kwargs['task_type'] = ckpt_meta['task_type']
        auto_detected.append(f"task_type={ckpt_meta['task_type']}")
    if kwargs.get('num_classes') == 2 and ckpt_meta.get('num_classes', 2) != 2:
        kwargs['num_classes'] = ckpt_meta['num_classes']
        auto_detected.append(f"num_classes={ckpt_meta['num_classes']}")

    if auto_detected:
        class_names = ckpt_meta.get('class_names')
        classes_str = f", classes={class_names}" if class_names else ""
        print(f"Auto-detected from checkpoint: {', '.join(auto_detected)}{classes_str}")

    # Handle --class_config / --manifest (classification) inference
    class_config_path = kwargs.pop('class_config', None)
    manifest = kwargs.pop('manifest', None)
    label_column = kwargs.pop('label_column', 'label')
    classes = kwargs.pop('classes', None)
    if manifest is not None and class_config_path is not None:
        raise click.UsageError("--manifest and --class_config are mutually exclusive.")
    if manifest is not None or class_config_path is not None:
        from remo_id.data.class_config import class_config_from_dict, parse_class_config
        if manifest is not None:
            # Keep the training class order: default to the checkpoint's classes.yaml.
            sidecar = os.path.join(os.path.dirname(kwargs['checkpoint']), "classes.yaml")
            trained_classes = None
            if os.path.exists(sidecar):
                trained_classes = parse_class_config(sidecar).class_names
                trained_classes = trained_classes if len(trained_classes) > 1 else None
            cfg = class_config_from_dict(_manifest_class_config(
                {**kwargs, 'manifest': manifest, 'label_column': label_column, 'classes': classes},
                default_classes=trained_classes))
            class_config_path = manifest
        else:
            cfg = parse_class_config(class_config_path)

        kwargs['task_type'] = cfg.task_type
        kwargs['num_classes'] = cfg.num_classes
        if cfg.task_type == 'binary' and len(cfg.class_names) == 1:
            kwargs['negative_study_ids'] = cfg.negative_study_ids

        kwargs['database_column'] = 'class_label'
        kwargs['database_label'] = None
        kwargs['study_id'] = None
        kwargs['class_definitions'] = cfg.class_definitions
        kwargs['exclude_filters'] = cfg.exclude_filters
        kwargs['include_filters'] = cfg.include_filters
        if cfg.default_class is not None:
            kwargs['default_class_idx'] = cfg.class_names.index(cfg.default_class)
        else:
            kwargs['default_class_idx'] = None
        kwargs['class_names'] = cfg.class_names

        print(f"Loaded class config from {class_config_path}")
        print(f"  Classes: {cfg.class_names}")
        print(f"  Task type: {kwargs['task_type']}, num_classes: {kwargs['num_classes']}")
    else:
        kwargs['class_definitions'] = None
        kwargs['exclude_filters'] = []
        kwargs['include_filters'] = []
        kwargs['default_class_idx'] = None
        kwargs['class_names'] = None
        # Require study_id when not using class_config/manifest
        if not kwargs.get('study_id') or not kwargs.get('database_column'):
            raise click.UsageError("Provide --manifest, --class_config, or --study_id with --database_column.")

    # Map threshold/threshold_mode to label_threshold/label_threshold_mode
    if 'threshold' in kwargs:
        kwargs['label_threshold'] = kwargs.pop('threshold')
    if 'threshold_mode' in kwargs:
        kwargs['label_threshold_mode'] = kwargs.pop('threshold_mode')
    # Generate out_pdf path from output_path
    kwargs['out_pdf'] = os.path.join(kwargs.get('output_path', './reports/'), 'inference_report.pdf')

    from remo_id.inference import run_inference
    run_inference(**kwargs)


@cli.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True), help="Path to model checkpoint")
@click.option("--experiment_ids", multiple=True, type=int, default=[], help="Experiment IDs to analyze (auto-resolved from class_config if omitted)")
@click.option("--class_config", type=click.Path(exists=True), default=None, help="Class config YAML (mutually exclusive with --study_id/--database_column)")
@_manifest_options
@click.option("--study_id", default=None, help="Study ID for label lookup")
@click.option("--database_column", default=None, help="Column for labeling")
@click.option("--database_label", default=None, help="Positive label value")
@click.option("--threshold", type=float, default=None, help="Threshold for numeric labels")
@click.option("--threshold_mode", type=click.Choice([">", "<", "=="]), default="==", help="Threshold direction")
@click.option("--negative_study_ids", multiple=True, default=[], help="Study IDs to merge as negatives")
@click.option("--negative_experiment_ids", multiple=True, type=int, default=[], help="Experiment IDs to merge as negatives")
@click.option("--max_experiments_per_class", type=int, default=None, help="Max experiments per class (default: all)")
@click.option("--model_name", type=click.Choice(MODEL_CHOICES), default="resnet18", help="Model architecture")
@click.option("--mode", type=click.Choice(["crops", "whole"]), default="crops", help="Gallery mode")
@click.option("--num_crops", type=int, default=50, help="Total number of crops to show (crops mode)")
@click.option("--num_images", type=int, default=100, help="Number of images to scan per experiment")
@click.option("--crop_size", type=int, default=120, help="Size of crop in pixels (crops mode)")
@click.option("--top_images", type=int, default=4, help="Number of top images per experiment (whole mode)")
@click.option("--bottom_images", type=int, default=0, help="Number of bottom-activated images per experiment (whole mode)")
@click.option("--output", type=click.Path(), default="gallery.pdf", help="Output PDF path")
@click.option("--seed", type=int, default=403, help="Random seed")
@click.option("--top_images_per_experiment", type=int, default=10, help="Top-activated images per experiment (crops mode)")
@click.option("--min_peak_distance", type=int, default=30, help="Minimum distance between peaks in pixels (crops mode)")
@click.option("--peak_significance", type=float, default=0.7, help="Significance threshold for peaks (0-1)")
@click.option("--max_peaks_per_image", type=int, default=3, help="Maximum peaks per image (crops mode)")
@click.option("--normalize_global", is_flag=True, default=False, help="Normalize activations globally (crops mode)")
@click.option("--positive_only", is_flag=True, default=False, help="Only show positive gradients (crops mode)")
@click.option("--heatmap_opacity", type=float, default=0.2, help="Heatmap opacity (0-1)")
@click.option("--minimal", is_flag=True, default=False, help="Minimal output: just images + GradCAM overlays, no section pages or text (whole mode)")
@click.option("--gradcam_layer", type=click.Choice(["layer2", "layer3", "layer4"]), default="layer4", help="ResNet block for Grad-CAM: layer4 (7x7, coarse/default), layer3 (14x14), layer2 (28x28, sharpest)")
def gallery(**kwargs):
    """Generate high-activation crop gallery from Grad-CAM.

    \b
    Examples:
        remo-id gallery --checkpoint model.ckpt --manifest manifest.csv \\
            --experiment_ids 1 --experiment_ids 2

        remo-id gallery --checkpoint model.ckpt --class_config classes.yaml \\
            --max_experiments_per_class 5 --mode whole
    """
    kwargs['experiment_ids'] = list(kwargs['experiment_ids'])
    kwargs['negative_study_ids'] = list(kwargs['negative_study_ids'])
    kwargs['negative_experiment_ids'] = list(kwargs['negative_experiment_ids'])

    if kwargs.get('manifest'):
        kwargs['class_config'] = _resolve_class_config_path(
            kwargs, os.path.dirname(os.path.abspath(kwargs['output'])))
    for key in ('manifest', 'label_column', 'classes'):
        kwargs.pop(key, None)
    class_config = kwargs.get('class_config')

    # Validate mutual exclusivity
    if class_config:
        from remo_id.data.class_config import validate_class_config_conflicts
        validate_class_config_conflicts(kwargs)

    # Auto-detect task_type, num_classes, model_name from checkpoint
    from remo_id.inference.core import auto_detect_from_checkpoint
    ckpt_meta = auto_detect_from_checkpoint(kwargs['checkpoint'])

    auto_detected = []
    if kwargs.get('model_name') == 'resnet18' and ckpt_meta.get('model_name'):
        kwargs['model_name'] = ckpt_meta['model_name']
        auto_detected.append(f"model_name={ckpt_meta['model_name']}")
    if ckpt_meta.get('task_type') and ckpt_meta['task_type'] != 'binary':
        kwargs['task_type'] = ckpt_meta['task_type']
        auto_detected.append(f"task_type={ckpt_meta['task_type']}")
    if ckpt_meta.get('num_classes', 2) != 2:
        kwargs['num_classes'] = ckpt_meta['num_classes']
        auto_detected.append(f"num_classes={ckpt_meta['num_classes']}")

    # Try to auto-discover class_config from checkpoint sidecar
    if not class_config and not kwargs.get('study_id') and not kwargs.get('experiment_ids'):
        ckpt_dir = os.path.dirname(kwargs['checkpoint'])
        sidecar = os.path.join(ckpt_dir, "classes.yaml")
        if os.path.exists(sidecar):
            class_config = sidecar
            kwargs['class_config'] = sidecar
            print(f"Auto-discovered class config: {sidecar}")

    if auto_detected:
        class_names = ckpt_meta.get('class_names')
        classes_str = f", classes={class_names}" if class_names else ""
        print(f"Auto-detected from checkpoint: {', '.join(auto_detected)}{classes_str}")

    # Validate that we have some way to find experiments
    if not class_config and not kwargs.get('experiment_ids') and not kwargs.get('study_id'):
        raise click.UsageError(
            "Must provide one of: --class_config, --experiment_ids, or --study_id"
        )

    mode = kwargs.pop('mode', 'crops')
    if mode == 'whole':
        from remo_id.interpret.gallery import run_whole_image_gallery
        if kwargs.get('num_images') == 100:
            kwargs['num_images'] = 1000
        run_whole_image_gallery(**kwargs)
    else:
        from remo_id.interpret.gallery import run_gallery
        run_gallery(**kwargs)


@cli.command()
@_class_source_options
@click.option("--model", type=click.Choice(MODEL_CHOICES), default="resnet18", help="Model architecture (determines augmentations)")
@click.option("--num_images", type=int, default=3, help="Number of images to sample per class")
@click.option("--num_augments", type=int, default=5, help="Number of augmented versions per image")
@click.option("--output_dir", type=click.Path(), default="./results/augment_samples", help="Directory to save output figures")
@click.option("--seed", type=int, default=403, help="Random seed")
@click.option("--background_correction", type=click.Choice(["none", "own", "random", "blend", "swap"]), default="none", help="Flat-field correction mode using p99 background images")
@click.option("--background_value", type=int, default=None, help="Background pixel value for normalization (default 225)")
@click.option("--intensity_jitter", type=float, default=None, help="ColorJitter brightness/contrast range")
@click.option("--noise_std", type=float, default=None, help="Max std for random Gaussian noise augmentation")
@click.option("--blur_sigma", type=float, default=None, help="Max sigma for random Gaussian blur augmentation")
@click.option("--gradient_blur_sigma", type=float, default=None, help="Max sigma for spatially-varying gradient blur")
def sample(**kwargs):
    """Visualize augmented images from the data pipeline per class.

    \b
    Each row shows one image with the original plus augmented versions.
    Rows are ordered by study_id, then class name.

    \b
    Example:
        remo-id sample --manifest manifest.csv --model resnet18-mil
    """
    from remo_id.data.sample_viz import run_sample_visualization
    kwargs['class_config'] = _resolve_class_config_path(kwargs, kwargs['output_dir'])
    for key in ('manifest', 'label_column', 'classes'):
        kwargs.pop(key)
    run_sample_visualization(**kwargs)


# ---------------------------------------------------------------------------
# analysis command group
# ---------------------------------------------------------------------------


def resolve_output_dir(output, checkpoint_path, subcommand):
    """Derive output directory: explicit --output wins, else {checkpoint_dir}/analysis/{subcommand}/."""
    if output:
        return output
    from pathlib import Path
    return str(Path(checkpoint_path).parent / "analysis" / subcommand)


@cli.group("analysis")
def analysis_group():
    """Generate figures and analysis from a trained model checkpoint.

    \b
    Examples:
        remo-id analysis performance --checkpoint run/best.ckpt \\
            --class_config run/classes.yaml --splits run/splits.yaml
        remo-id analysis all --checkpoint run/best.ckpt \\
            --class_config run/classes.yaml --splits run/splits.yaml
    """
    pass


def _pa_common_options(f):
    """Options shared across all analysis subcommands."""
    f = click.option("--checkpoint", required=True, type=click.Path(exists=True),
                     help="Path to model checkpoint (.ckpt)")(f)
    f = click.option("--class_config", required=True, type=click.Path(exists=True),
                     help="Path to YAML class config (same file used at training)")(f)
    f = click.option("--splits", required=True, type=click.Path(exists=True),
                     help="Path to splits YAML saved alongside the checkpoint")(f)
    f = click.option("--split", default="val", show_default=True,
                     help="Which split to evaluate (train/val/all)")(f)
    f = click.option("--output", default=None,
                     type=click.Path(), help="Output directory (default: {checkpoint_dir}/analysis/{subcommand}/)")(f)
    f = click.option("--bag_size", type=int, default=200, show_default=True,
                     help="Number of images per bag")(f)
    f = click.option("--seed", type=int, default=403, show_default=True,
                     help="Random seed")(f)
    f = click.option("--batch_size", type=int, default=256, show_default=True,
                     help="DataModule batch size")(f)
    f = click.option("--bootstrap_n", type=int, default=1000, show_default=True,
                     help="Bootstrap iterations for confidence intervals")(f)
    return f


def _resolve_split_experiment_ids(splits_data, split):
    from remo_id.inference.context import resolve_split_experiment_ids
    return resolve_split_experiment_ids(splits_data, split)

def _load_class_config(class_config_path):
    from remo_id.inference.context import load_class_config
    return load_class_config(class_config_path)

def _build_analysis_context(checkpoint, class_config_path, splits_path, split_name,
                         bag_size, seed, batch_size, **kwargs):
    from remo_id.inference.context import build_analysis_context
    return build_analysis_context(checkpoint, class_config_path, splits_path, split_name,
                               bag_size, seed, batch_size, **kwargs)

def _build_context_from_predictions(checkpoint, split_name):
    from remo_id.inference.context import build_context_from_predictions
    return build_context_from_predictions(checkpoint, split_name)

def _build_context_from_cv_predictions(cv_dir, class_config_path, use_last=False):
    from remo_id.inference.context import build_context_from_cv_predictions
    return build_context_from_cv_predictions(cv_dir, class_config_path, use_last=use_last)


def _pa_common_options_flexible(f):
    """Like _pa_common_options but checkpoint/splits are optional (for CV mode)."""
    f = click.option("--checkpoint", default=None, type=click.Path(exists=True),
                     help="Path to model checkpoint (.ckpt)")(f)
    f = click.option("--class_config", required=True, type=click.Path(exists=True),
                     help="Path to YAML class config (same file used at training)")(f)
    f = click.option("--splits", default=None, type=click.Path(exists=True),
                     help="Path to splits YAML saved alongside the checkpoint")(f)
    f = click.option("--split", default="val", show_default=True,
                     help="Which split to evaluate (train/val/all)")(f)
    f = click.option("--output", default=None,
                     type=click.Path(), help="Output directory")(f)
    f = click.option("--bag_size", type=int, default=200, show_default=True,
                     help="Number of images per bag")(f)
    f = click.option("--seed", type=int, default=403, show_default=True,
                     help="Random seed")(f)
    f = click.option("--batch_size", type=int, default=256, show_default=True,
                     help="DataModule batch size")(f)
    f = click.option("--bootstrap_n", type=int, default=1000, show_default=True,
                     help="Bootstrap iterations for confidence intervals")(f)
    return f


def _cv_aware_options(f):
    """Options for commands supporting both --checkpoint and --cv_dir modes."""
    f = click.option("--cv_dir", default=None, type=click.Path(exists=True),
                     help="CV directory (mutually exclusive with --checkpoint)")(f)
    f = click.option("--use_last", is_flag=True, default=False,
                     help="Use last.ckpt predictions (only with --cv_dir)")(f)
    return f


def _validate_analysis_mode(kwargs):
    """Validate --checkpoint vs --cv_dir mutual exclusivity.

    Returns "cv" or "checkpoint".
    """
    has_cv = kwargs.get("cv_dir") is not None
    has_ckpt = kwargs.get("checkpoint") is not None

    if has_cv and has_ckpt:
        raise click.UsageError("--cv_dir and --checkpoint are mutually exclusive.")
    if not has_cv and not has_ckpt:
        raise click.UsageError("Either --checkpoint or --cv_dir is required.")
    if not has_cv and kwargs.get("use_last"):
        raise click.UsageError("--use_last can only be used with --cv_dir.")
    if has_cv and kwargs.get("splits") is not None:
        raise click.UsageError("--splits is not used with --cv_dir.")

    if has_cv:
        return "cv"

    # Checkpoint mode: validate that splits is present (required for GPU fallback)
    if kwargs.get("splits") is None:
        # Still allow if saved predictions exist (no splits needed)
        ctx = _build_context_from_predictions(kwargs["checkpoint"], kwargs.get("split", "val"))
        if ctx is None:
            raise click.UsageError(
                "--splits is required when using --checkpoint without saved predictions."
            )
    return "checkpoint"


def resolve_output_dir_cv(output, cv_dir, subcommand):
    """Derive output directory for CV mode."""
    if output:
        return output
    return os.path.join(cv_dir, "analysis", subcommand)


def _save_figures(figs: list, subdir: str):
    """Save a list of (name, fig) tuples as PNG into subdir."""
    import matplotlib.pyplot as plt
    os.makedirs(subdir, exist_ok=True)
    for name, fig in figs:
        fig.savefig(os.path.join(subdir, f"{name}.png"))
        plt.close(fig)


# --- performance ---

@analysis_group.command()
@_pa_common_options_flexible
@_cv_aware_options
def performance(**kwargs):
    """ROC, PR, confusion matrix, calibration, probability distributions + metrics CSV.

    Supports both --checkpoint (single model) and --cv_dir (pooled CV predictions).
    """
    from remo_id.reports import figures as plots

    mode = _validate_analysis_mode(kwargs)

    if mode == "cv":
        out = resolve_output_dir_cv(kwargs["output"], kwargs["cv_dir"], "performance")
        ctx = _build_context_from_cv_predictions(
            kwargs["cv_dir"], kwargs["class_config"], use_last=kwargs["use_last"],
        )
        print(f"Using pooled CV predictions from {kwargs['cv_dir']}")
    else:
        out = resolve_output_dir(kwargs["output"], kwargs["checkpoint"], "performance")
        # Try loading saved predictions (no GPU needed)
        ctx = _build_context_from_predictions(kwargs["checkpoint"], kwargs["split"])
        if ctx is not None:
            print("Using saved predictions (no GPU inference needed)")
            from remo_id.analysis.performance import run_performance
            run_performance(
                checkpoint_path=kwargs["checkpoint"],
                output=out, split=kwargs["split"],
                bootstrap_n=kwargs["bootstrap_n"],
            )
        else:
            ctx = _build_analysis_context(
                checkpoint=kwargs["checkpoint"],
                class_config_path=kwargs["class_config"],
                splits_path=kwargs["splits"],
                split_name=kwargs["split"],
                bag_size=kwargs["bag_size"],
                seed=kwargs["seed"],
                batch_size=kwargs["batch_size"],
            )

    bootstrap_n = kwargs["bootstrap_n"]
    y_true, y_pred, y_probs = ctx["y_true"], ctx["y_pred"], ctx["y_probs"]
    class_names = ctx["class_names"]
    task_type = ctx.get("task_type", "multiclass")

    figs = [
        ("roc_curves",        plots.plot_roc_curves(y_true, y_probs, class_names, bootstrap_n=bootstrap_n, task_type=task_type)),
        ("pr_curves",         plots.plot_pr_curves(y_true, y_probs, class_names, bootstrap_n=bootstrap_n, task_type=task_type)),
        ("confusion_matrix",  plots.plot_confusion_matrix(y_true, y_pred, class_names)),
        ("calibration",       plots.plot_calibration(y_true, y_probs, class_names, task_type=task_type)),
        ("prob_distributions",plots.plot_prob_distributions(y_true, y_probs, class_names, task_type=task_type)),
    ]

    # K-fold variability plot (CV mode only)
    if mode == "cv" and ctx.get("per_fold"):
        from remo_id.workflows.cv import aggregate_cv_results
        pooled = (y_true, y_probs, y_pred)
        cv_results = aggregate_cv_results(
            ctx["per_fold"], pooled, class_names,
            bootstrap_n=bootstrap_n, task_type=task_type,
        )
        if cv_results["per_fold_metrics"]:
            figs.append(("kfold_variability",
                         plots.plot_kfold_variability(cv_results["per_fold_metrics"], class_names)))

    _save_figures(figs, out)

    metrics_df = plots.metrics_table_with_ci(y_true, y_probs, class_names, n=bootstrap_n, task_type=task_type)
    metrics_df.to_csv(os.path.join(out, "metrics.csv"), index=False)
    print(metrics_df.to_string(index=False))
    print(f"Saved performance figures to {out}")


# --- bag-size ---

@analysis_group.command("bag-size")
@_pa_common_options
@click.option("--bag_sizes", default="1,3,5,10,25,50,100,200,400,700,1000",
              help="Comma-separated bag sizes to sweep")
@click.option("--seeds", default="403,42,7,123,999",
              help="Comma-separated seeds for variability estimate")
def bag_size(**kwargs):
    """Sweep bag sizes and sampling positions to find the plateau."""
    from remo_id.reports import figures as plots
    from remo_id.inference.mil import _predict_experiments
    from remo_id.inference.core import (
        set_inference_seeds,
    )
    from sklearn.metrics import roc_auc_score
    import numpy as np

    out = resolve_output_dir(kwargs["output"], kwargs["checkpoint"], "bag_size")
    bag_sizes = [int(x) for x in kwargs["bag_sizes"].split(",")]
    seeds = [int(x) for x in kwargs["seeds"].split(",")]
    class_names, num_classes, task_type = None, None, None  # resolved below

    # Load context once to get model and data setup
    base_ctx = _build_analysis_context(
        checkpoint=kwargs["checkpoint"],
        class_config_path=kwargs["class_config"],
        splits_path=kwargs["splits"],
        split_name=kwargs["split"],
        bag_size=bag_sizes[0],
        seed=seeds[0],
        batch_size=kwargs["batch_size"],
    )
    class_names = base_ctx["class_names"]
    model       = base_ctx["model"]
    ds          = base_ctx["ds"]
    df          = base_ctx["df"]
    task_type   = base_ctx["task_type"]
    num_classes = base_ctx["num_classes"]
    device      = base_ctx["device"]

    # Sweep bag_size × seed
    aucs_by_seed = np.full((len(seeds), len(bag_sizes)), np.nan)
    for si, seed in enumerate(seeds):
        for bi, bs in enumerate(bag_sizes):
            set_inference_seeds(seed)
            results, *_ = _predict_experiments(
                model=model, ds=ds, df=df, database_column="class_label",
                task_type=task_type, num_classes=num_classes,
                bag_size=bs, seed=seed, device=device,
                class_names=class_names,
            )
            valid = [r for r in results if not r.has_null_label and not np.isnan(r.y_prob)]
            if not valid:
                continue
            yt = np.array([r.y_true for r in valid], dtype=int)
            yp = np.array([r.y_probs_all for r in valid], dtype=float)
            try:
                from sklearn.preprocessing import label_binarize
                yb = label_binarize(yt, classes=list(range(num_classes)))
                aucs_by_seed[si, bi] = roc_auc_score(yb, yp, average="macro", multi_class="ovr")
            except Exception:
                pass
            print(f"  bag_size={bs}, seed={seed}: macro-AUC={aucs_by_seed[si, bi]:.4f}")

    # Sampling position sweep (fixed bag_size = largest)
    fixed_bs = bag_sizes[-1]
    position_labels = [
        f"first {fixed_bs}",
        "first 10%",
        "first 25%",
        "random",
        "last 25%",
        "last 10%",
    ]
    position_modes = [
        ("start", 100.0),
        ("first", 10.0),
        ("first", 25.0),
        ("random", 100.0),
        ("last", 25.0),
        ("last", 10.0),
    ]
    pos_aucs = np.full((len(position_labels), len(seeds)), np.nan)
    for si, seed in enumerate(seeds):
        for pi, (mode, pct) in enumerate(position_modes):
            set_inference_seeds(seed)
            results, *_ = _predict_experiments(
                model=model, ds=ds, df=df, database_column="class_label",
                task_type=task_type, num_classes=num_classes,
                bag_size=fixed_bs, seed=seed, device=device,
                sample_mode=mode, sample_percent=pct,
                class_names=class_names,
            )
            valid = [r for r in results if not r.has_null_label and not np.isnan(r.y_prob)]
            if not valid:
                continue
            yt = np.array([r.y_true for r in valid], dtype=int)
            yp = np.array([r.y_probs_all for r in valid], dtype=float)
            try:
                from sklearn.preprocessing import label_binarize
                yb = label_binarize(yt, classes=list(range(num_classes)))
                pos_aucs[pi, si] = roc_auc_score(yb, yp, average="macro", multi_class="ovr")
            except Exception:
                pass

    figs = [
        ("bag_size_sweep",    plots.plot_bag_size_sweep(bag_sizes, aucs_by_seed)),
        ("sampling_position", plots.plot_sampling_position(position_labels, pos_aucs, bag_size=fixed_bs)),
    ]
    _save_figures(figs, out)
    print(f"Saved bag-size figures to {out}")


# --- interpret ---

@analysis_group.command()
@_pa_common_options
@click.option("--n_top", type=int, default=8, show_default=True,
              help="Top-attention frames per class")
@click.option("--n_repeats", type=int, default=1, show_default=True,
              help="Independent bags per experiment for UMAP (>1 shows variability)")
def interpret(**kwargs):
    """Top-attention grid, UMAP embedding, attention entropy."""
    from remo_id.reports import figures as plots
    from remo_id.inference.mil import get_instance_overlay
    from remo_id.interpret import tensor_to_rgb_numpy
    import torch
    import numpy as np

    out = resolve_output_dir(kwargs["output"], kwargs["checkpoint"], "interpret")
    n_top = kwargs["n_top"]
    n_repeats = kwargs["n_repeats"]
    ctx = _build_analysis_context(
        checkpoint=kwargs["checkpoint"],
        class_config_path=kwargs["class_config"],
        splits_path=kwargs["splits"],
        split_name=kwargs["split"],
        bag_size=kwargs["bag_size"],
        seed=kwargs["seed"],
        batch_size=kwargs["batch_size"],
        n_top_attention=n_top,
        capture_embeddings=(n_repeats <= 1),
    )
    class_names    = ctx["class_names"]
    attention_data = ctx["attention_pages_data"]
    valid          = ctx["valid"]

    # Collect top-attention frames per true class with Grad-CAM/attention overlays
    model = ctx["model"]
    device = ctx["device"]
    frames_by_class = [[] for _ in class_names]
    for page, result in zip(attention_data, ctx["results"]):
        if result.has_null_label or np.isnan(result.y_prob):
            continue
        c = int(result.y_true)
        if c >= len(class_names):
            continue
        x_vis = page["x_vis"]
        attn_vis = page["attn_vis"]
        bag_max = page.get("bag_attn_max", float(np.max(attn_vis)))
        bag_min = page.get("bag_attn_min", float(np.min(attn_vis)))
        for j in range(min(n_top, x_vis.shape[0])):
            x1 = x_vis[j:j+1]
            if device:
                x1 = x1.to(device)
            try:
                img_np = get_instance_overlay(model, x1, image_weight=0.9)
            except Exception:
                img_np = tensor_to_rgb_numpy(x_vis[j])
            if device:
                del x1
                torch.cuda.empty_cache()
            frames_by_class[c].append({
                "image": img_np,
                "raw_attn": float(attn_vis[j]),
                "bag_attn_max": bag_max,
                "bag_attn_min": bag_min,
                "experiment_id": page["exp_id"],
            })

    # Attention entropy per class
    entropy_by_class = [
        [r.attn_entropy for r in valid if int(r.y_true) == c and r.attn_entropy is not None]
        for c in range(len(class_names))
    ]

    figs = [
        ("attention_entropy", plots.plot_attention_entropy(entropy_by_class, class_names)),
    ]
    if any(len(frames) > 0 for frames in frames_by_class):
        figs.append(("top_attention_grid",
                     plots.plot_top_attention_grid(frames_by_class, class_names,
                                                   n_top=n_top)))

    # UMAP
    if n_repeats > 1:
        from remo_id.inference.mil import _predict_experiments_multi_embed
        embeddings, emb_labels, emb_exp_ids = _predict_experiments_multi_embed(
            model=ctx["model"], ds=ctx["ds"], df=ctx["df"],
            database_column=ctx["database_column"],
            task_type=ctx["task_type"], num_classes=ctx["num_classes"],
            bag_size=kwargs["bag_size"], seed=kwargs["seed"], device=ctx["device"],
            n_repeats=n_repeats, class_names=class_names,
        )
        if len(embeddings) > 5:
            figs.append(("umap", plots.plot_umap(
                embeddings, emb_labels, class_names, experiment_ids=emb_exp_ids)))
    else:
        embeddings = [r.feat_emb for r in valid if r.feat_emb is not None]
        emb_labels = [int(r.y_true) for r in valid if r.feat_emb is not None]
        if len(embeddings) > 5:
            figs.append(("umap", plots.plot_umap(embeddings, emb_labels, class_names)))

    _save_figures(figs, out)
    print(f"Saved interpret figures to {out}")


# --- robustness ---

@analysis_group.command()
@_pa_common_options
@click.option("--seeds", default="403,42,7,123,999",
              help="Comma-separated seeds to assess variability")
def robustness(**kwargs):
    """Seed variability, threshold sensitivity, operating points."""
    from remo_id.reports import figures as plots
    from remo_id.inference.mil import _predict_experiments
    from remo_id.inference.core import set_inference_seeds
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import label_binarize
    import numpy as np

    out = resolve_output_dir(kwargs["output"], kwargs["checkpoint"], "robustness")
    seeds = [int(x) for x in kwargs["seeds"].split(",")]

    base_ctx = _build_analysis_context(
        checkpoint=kwargs["checkpoint"],
        class_config_path=kwargs["class_config"],
        splits_path=kwargs["splits"],
        split_name=kwargs["split"],
        bag_size=kwargs["bag_size"],
        seed=seeds[0],
        batch_size=kwargs["batch_size"],
    )
    class_names = base_ctx["class_names"]
    model       = base_ctx["model"]
    ds          = base_ctx["ds"]
    df          = base_ctx["df"]
    task_type   = base_ctx["task_type"]
    num_classes = base_ctx["num_classes"]
    device      = base_ctx["device"]

    metrics_by_seed = {"macro_auc": [], "macro_ap": []}
    for seed in seeds:
        set_inference_seeds(seed)
        results, *_ = _predict_experiments(
            model=model, ds=ds, df=df, database_column="class_label",
            task_type=task_type, num_classes=num_classes,
            bag_size=kwargs["bag_size"], seed=seed, device=device,
            class_names=class_names,
        )
        valid = [r for r in results if not r.has_null_label and not np.isnan(r.y_prob)]
        if not valid:
            continue
        yt = np.array([r.y_true for r in valid], dtype=int)
        yp = np.array([r.y_probs_all for r in valid], dtype=float)
        try:
            yb = label_binarize(yt, classes=list(range(num_classes)))
            metrics_by_seed["macro_auc"].append(
                roc_auc_score(yb, yp, average="macro", multi_class="ovr")
            )
            metrics_by_seed["macro_ap"].append(
                average_precision_score(yb, yp, average="macro")
            )
        except Exception:
            pass
        print(f"  seed={seed}: macro-AUC={metrics_by_seed['macro_auc'][-1]:.4f}")

    y_true  = base_ctx["y_true"]
    y_probs = base_ctx["y_probs"]

    figs = [
        ("seed_variability",       plots.plot_seed_variability(metrics_by_seed)),
        ("threshold_sensitivity",  plots.plot_threshold_sensitivity(y_true, y_probs, class_names)),
        ("operating_points",       plots.plot_operating_points(y_true, y_probs, class_names)),
    ]
    _save_figures(figs, out)
    print(f"Saved robustness figures to {out}")


# --- baselines ---

@analysis_group.command()
@_pa_common_options
def baselines(**kwargs):
    """Compare fine-tuned vs pretrained features with linear probes and majority baseline."""
    import torch
    from remo_id.analysis.baselines import extract_features, run_baselines, create_pretrained_model

    out = resolve_output_dir(kwargs["output"], kwargs["checkpoint"], "baselines")
    os.makedirs(out, exist_ok=True)

    ctx = _build_analysis_context(
        checkpoint=kwargs["checkpoint"],
        class_config_path=kwargs["class_config"],
        splits_path=kwargs["splits"],
        split_name="train",
        bag_size=kwargs["bag_size"],
        seed=kwargs["seed"],
        batch_size=kwargs["batch_size"],
    )

    # Extract features from train split
    with open(kwargs["splits"]) as f:
        splits_data = yaml.safe_load(f)
    train_ids = _resolve_split_experiment_ids(splits_data, "train")
    val_ids = _resolve_split_experiment_ids(splits_data, "val")

    # Build separate data modules for train and val
    from remo_id.data import RemoIDDataModule
    class_definitions, class_names, num_classes, task_type, exclude_filters, include_filters, default_class_idx = (
        _load_class_config(kwargs["class_config"])
    )

    def _build_ds(experiment_ids):
        return RemoIDDataModule(
            database_column="class_label", database_label=None,
            study_id=None, threshold=None, threshold_mode="==",
            negative_study_ids=[], negative_experiment_ids=[],
            include_experiment_ids=experiment_ids,
            batch_size=kwargs["batch_size"],
            backbone_name=ctx["model"].backbone_name,
            train=False, task_type=task_type, num_classes=num_classes,
            class_definitions=class_definitions,
            exclude_filters=exclude_filters,
            include_filters=include_filters,
            default_class_idx=default_class_idx,
            skip_null_labels=True,
        )

    print("Extracting fine-tuned features (train)...")
    train_ds = _build_ds(train_ids)
    X_train, y_train, _ = extract_features(ctx["model"], train_ds, bag_size=kwargs["bag_size"], seed=kwargs["seed"])
    print(f"  Train: {X_train.shape[0]} experiments, {X_train.shape[1]} features")

    print("Extracting fine-tuned features (val)...")
    val_ds = _build_ds(val_ids)
    X_val, y_val, _ = extract_features(ctx["model"], val_ds, bag_size=kwargs["bag_size"], seed=kwargs["seed"])
    print(f"  Val: {X_val.shape[0]} experiments, {X_val.shape[1]} features")

    # Extract pretrained (ImageNet) features
    print("Creating pretrained model...")
    pretrained_model = create_pretrained_model(kwargs["checkpoint"])

    print("Extracting pretrained features (train)...")
    train_ds_pt = _build_ds(train_ids)
    X_train_pt, _, _ = extract_features(pretrained_model, train_ds_pt, bag_size=kwargs["bag_size"], seed=kwargs["seed"])
    print(f"  Train: {X_train_pt.shape[0]} experiments, {X_train_pt.shape[1]} features")

    print("Extracting pretrained features (val)...")
    val_ds_pt = _build_ds(val_ids)
    X_val_pt, _, _ = extract_features(pretrained_model, val_ds_pt, bag_size=kwargs["bag_size"], seed=kwargs["seed"])
    print(f"  Val: {X_val_pt.shape[0]} experiments, {X_val_pt.shape[1]} features")

    del pretrained_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Training baselines...")
    results = run_baselines(
        X_train, y_train, X_val, y_val,
        task_type=task_type, num_classes=num_classes,
        X_train_pretrained=X_train_pt, X_val_pretrained=X_val_pt,
    )
    results.to_csv(os.path.join(out, "baseline_results.csv"), index=False)
    print(results.to_string(index=False))
    print(f"\nSaved baseline results to {out}")


# --- stats-test ---

@analysis_group.command("stats-test")
@click.option("--checkpoint_a", required=True, type=click.Path(exists=True), help="Checkpoint A")
@click.option("--checkpoint_b", required=True, type=click.Path(exists=True), help="Checkpoint B")
@click.option("--class_config", required=True, type=click.Path(exists=True))
@click.option("--splits", required=True, type=click.Path(exists=True))
@click.option("--split", default="val", show_default=True)
@click.option("--output", default=None, type=click.Path())
@click.option("--bag_size", type=int, default=200, show_default=True)
@click.option("--seed", type=int, default=403, show_default=True)
@click.option("--batch_size", type=int, default=256, show_default=True)
def stats_test(**kwargs):
    """Statistical significance tests comparing two checkpoints."""
    from remo_id.analysis.statistical_tests import mcnemar_test, delong_test, bootstrap_paired_diff
    from sklearn.metrics import roc_auc_score
    import json

    out = kwargs["output"] or os.path.join(
        os.path.dirname(kwargs["checkpoint_a"]), "analysis", "stats_test"
    )
    os.makedirs(out, exist_ok=True)

    ctx_a = _build_analysis_context(
        checkpoint=kwargs["checkpoint_a"],
        class_config_path=kwargs["class_config"],
        splits_path=kwargs["splits"],
        split_name=kwargs["split"],
        bag_size=kwargs["bag_size"],
        seed=kwargs["seed"],
        batch_size=kwargs["batch_size"],
    )
    ctx_b = _build_analysis_context(
        checkpoint=kwargs["checkpoint_b"],
        class_config_path=kwargs["class_config"],
        splits_path=kwargs["splits"],
        split_name=kwargs["split"],
        bag_size=kwargs["bag_size"],
        seed=kwargs["seed"],
        batch_size=kwargs["batch_size"],
    )

    y_true = ctx_a["y_true"]
    y_pred_a = ctx_a["y_pred"]
    y_pred_b = ctx_b["y_pred"]
    y_probs_a = ctx_a["y_probs"]
    y_probs_b = ctx_b["y_probs"]

    results = {}

    # McNemar
    mcnemar = mcnemar_test(y_true, y_pred_a, y_pred_b)
    results["mcnemar"] = mcnemar
    print(f"McNemar: statistic={mcnemar['statistic']:.3f}, p={mcnemar['p_value']:.4f}")

    # DeLong (per-class for multiclass, or direct for binary)
    num_classes = ctx_a["num_classes"]
    for c in range(num_classes):
        y_bin = (y_true == c).astype(int)
        try:
            delong = delong_test(y_bin, y_probs_a[:, c], y_probs_b[:, c])
            results[f"delong_class_{c}"] = delong
            print(f"DeLong (class {c}): AUC_A={delong['auc_a']:.3f}, AUC_B={delong['auc_b']:.3f}, p={delong['p_value']:.4f}")
        except Exception as e:
            print(f"DeLong (class {c}): failed - {e}")

    # Bootstrap paired AUC diff (macro)
    def macro_auc(yt, yp):
        from sklearn.preprocessing import label_binarize
        yb = label_binarize(yt, classes=list(range(num_classes)))
        if num_classes == 2:
            yb = np.column_stack([1 - yb, yb])
        return roc_auc_score(yb, yp, average="macro", multi_class="ovr")

    import numpy as np
    boot = bootstrap_paired_diff(y_true, y_probs_a, y_probs_b, macro_auc)
    results["bootstrap_macro_auc"] = boot
    print(f"Bootstrap AUC diff: {boot['observed_diff']:.4f} [{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}], p={boot['p_value']:.4f}")

    with open(os.path.join(out, "stats_test_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {out}")


# --- ablation ---

@analysis_group.command()
@click.option("--checkpoints", multiple=True, required=True,
              help="Name:path pairs, e.g. 'ViT-L:/path/to/best.ckpt'")
@click.option("--class_config", required=True, type=click.Path(exists=True))
@click.option("--splits", required=True, type=click.Path(exists=True))
@click.option("--split", default="val", show_default=True)
@click.option("--output", default="./figures", type=click.Path())
@click.option("--bag_size", type=int, default=200, show_default=True)
@click.option("--seed", type=int, default=403, show_default=True)
@click.option("--batch_size", type=int, default=256, show_default=True)
@click.option("--bootstrap_n", type=int, default=1000, show_default=True)
def ablation(**kwargs):
    """Compare multiple checkpoints (model ablation)."""
    from remo_id.reports import figures as plots
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import label_binarize
    import numpy as np

    out = os.path.join(kwargs.get("output") or "./ablation_figures", "ablation")
    os.makedirs(out, exist_ok=True)

    model_names, auc_vals, ap_vals = [], [], []
    for ckpt_spec in kwargs["checkpoints"]:
        name, ckpt_path = ckpt_spec.split(":", 1)
        model_names.append(name)
        ctx = _build_analysis_context(
            checkpoint=ckpt_path,
            class_config_path=kwargs["class_config"],
            splits_path=kwargs["splits"],
            split_name=kwargs["split"],
            bag_size=kwargs["bag_size"],
            seed=kwargs["seed"],
            batch_size=kwargs["batch_size"],
        )
        y_true  = ctx["y_true"]
        y_probs = ctx["y_probs"]
        num_classes = ctx["num_classes"]
        try:
            yb = label_binarize(y_true, classes=list(range(num_classes)))
            auc_vals.append(roc_auc_score(yb, y_probs, average="macro", multi_class="ovr"))
            ap_vals.append(average_precision_score(yb, y_probs, average="macro"))
        except Exception as e:
            print(f"  {name}: metric computation failed: {e}")
            auc_vals.append(np.nan)
            ap_vals.append(np.nan)
        print(f"  {name}: macro-AUC={auc_vals[-1]:.4f}")

    metrics_by_seed = {
        f"{n} AUC": [v] for n, v in zip(model_names, auc_vals)
    }
    fig = plots.plot_seed_variability(metrics_by_seed)
    fig.get_axes()[0].set_title("Model ablation — macro-AUC")
    _save_figures([("ablation_auc", fig)], out)

    import pandas as pd
    ablation_df = pd.DataFrame({
        "model": model_names,
        "macro_auc": auc_vals,
        "macro_ap": ap_vals,
    })
    ablation_df.to_csv(os.path.join(out, "ablation_table.csv"), index=False)
    print(ablation_df.to_string(index=False))
    print(f"Saved ablation figures to {out}")


# --- error-analysis ---

@analysis_group.command("error-analysis")
@_pa_common_options_flexible
@_cv_aware_options
def error_analysis(**kwargs):
    """Misclassification analysis with confidence and study breakdowns.

    Supports both --checkpoint (single model) and --cv_dir (pooled CV predictions).
    """
    from remo_id.reports.figures import (
        plot_error_confidence, plot_error_by_study, plot_confusion_errors,
    )

    mode = _validate_analysis_mode(kwargs)

    if mode == "cv":
        from remo_id.analysis.error_analysis import identify_misclassifications_from_arrays
        out = resolve_output_dir_cv(kwargs["output"], kwargs["cv_dir"], "error_analysis")
        os.makedirs(out, exist_ok=True)

        ctx = _build_context_from_cv_predictions(
            kwargs["cv_dir"], kwargs["class_config"], use_last=kwargs["use_last"],
        )
        print(f"Using pooled CV predictions from {kwargs['cv_dir']}")

        y_true, y_pred, y_probs = ctx["y_true"], ctx["y_pred"], ctx["y_probs"]
        class_names = ctx["class_names"]

        error_df = identify_misclassifications_from_arrays(
            y_true, y_pred, y_probs, class_names,
            experiment_ids=ctx.get("experiment_ids"),
            study_ids=ctx.get("study_ids"),
        )
    else:
        from remo_id.analysis.error_analysis import identify_misclassifications
        out = resolve_output_dir(kwargs["output"], kwargs["checkpoint"], "error_analysis")
        os.makedirs(out, exist_ok=True)

        ctx = _build_analysis_context(
            checkpoint=kwargs["checkpoint"],
            class_config_path=kwargs["class_config"],
            splits_path=kwargs["splits"],
            split_name=kwargs["split"],
            bag_size=kwargs["bag_size"],
            seed=kwargs["seed"],
            batch_size=kwargs["batch_size"],
        )

        y_true, y_pred, y_probs = ctx["y_true"], ctx["y_pred"], ctx["y_probs"]
        class_names = ctx["class_names"]
        results = ctx["valid"]

        error_df = identify_misclassifications(results, class_names)

    error_df.to_csv(os.path.join(out, "misclassifications.csv"), index=False)

    n_errors = (~error_df["correct"]).sum()
    print(f"Total: {len(error_df)} experiments, {n_errors} misclassified ({n_errors/len(error_df)*100:.1f}%)")

    figs = [
        ("error_confidence", plot_error_confidence(y_true, y_pred, y_probs, class_names)),
        ("error_by_study", plot_error_by_study(error_df, class_names)),
        ("error_transitions", plot_confusion_errors(error_df, class_names)),
    ]
    _save_figures(figs, out)
    print(f"Saved error analysis to {out}")


# --- learning-curves ---

@analysis_group.command("learning-curves")
@click.option("--class_config", required=True, type=click.Path(exists=True))
@click.option("--model", type=click.Choice(MODEL_CHOICES), default="resnet18-mil")
@click.option("--fractions", default="0.1,0.2,0.4,0.6,0.8,1.0",
              help="Comma-separated training fractions")
@click.option("--n_seeds", type=int, default=3, show_default=True)
@click.option("--output", default=None, type=click.Path())
@click.option("--generate_sbatch", is_flag=True, default=False,
              help="Generate sbatch --array script instead of training directly")
@click.option("--partition", default="gpu", show_default=True, help="SLURM partition")
def learning_curves(**kwargs):
    """Training size vs performance sweep (learning curve)."""
    from remo_id.analysis.learning_curves import generate_learning_curve_sbatch

    fractions = [float(x) for x in kwargs["fractions"].split(",")]
    out = kwargs["output"] or "./learning_curves"
    os.makedirs(out, exist_ok=True)

    if kwargs["generate_sbatch"]:
        train_cmd = (
            f"remo-id train --class_config {kwargs['class_config']} "
            f"--model {kwargs['model']}"
        )
        generate_learning_curve_sbatch(
            save_dir=out,
            fractions=fractions,
            n_seeds=kwargs["n_seeds"],
            train_cmd=train_cmd,
            partition=kwargs["partition"],
        )
    else:
        import subprocess
        train_cmd = (
            f"remo-id train --class_config {kwargs['class_config']} "
            f"--model {kwargs['model']}"
        )
        script_path = generate_learning_curve_sbatch(
            save_dir=out,
            fractions=fractions,
            n_seeds=kwargs["n_seeds"],
            train_cmd=train_cmd,
            partition=kwargs["partition"],
        )
        print(f"Submitting learning curve array job...")
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"sbatch submission failed: {result.stderr.strip()}")
            print(f"You can submit manually: sbatch {script_path}")


# --- cv-performance ---

@analysis_group.command("cv-performance")
@click.option("--cv_dir", required=True, type=click.Path(exists=True),
              help="Path to CV output directory (contains repeat*_fold* subdirs)")
@click.option("--class_config", required=True, type=click.Path(exists=True),
              help="Path to YAML class config")
@click.option("--output", default=None, type=click.Path(),
              help="Output directory (default: {cv_dir}/analysis/cv_performance/)")
@click.option("--bag_size", type=int, default=200, show_default=True,
              help="Number of images per bag")
@click.option("--seed", type=int, default=403, show_default=True,
              help="Random seed")
@click.option("--batch_size", type=int, default=256, show_default=True,
              help="DataModule batch size")
@click.option("--bootstrap_n", type=int, default=1000, show_default=True,
              help="Bootstrap iterations for confidence intervals")
@click.option("--n_bags", type=int, default=5, show_default=True,
              help="Number of independently sampled bags per experiment (averaged for prediction)")
@click.option("--use_last", is_flag=True, default=False,
              help="Use last.ckpt instead of best.ckpt for predictions")
@click.option("--patient_level", type=click.Choice(["none", "first", "combined"]),
              default="none", show_default=True,
              help="Aggregate to patient (sample_id) level: "
                   "'first' keeps only the first replicate per patient, "
                   "'combined' averages probabilities across replicates")
@click.option("--model_version", type=str, default=None,
              help="Embedding-MIL: embedding-store version tag (required for embedding-mil cv_dirs)")
@click.option("--cell_config", type=str, default=None,
              help="Embedding-MIL: cell_config YAML used at training (omit if trained without one)")
@click.option("--pool_cap", type=int, default=None,
              help="Embedding-MIL: --pool_cap used at training")
@click.option("--load_workers", type=int, default=None,
              help="Embedding-MIL: thread-pool size for the parallel store load")
def cv_performance(**kwargs):
    """Pool held-out CV fold predictions and report metrics with 95% bootstrap CIs.

    SLURM array aware: when SLURM_ARRAY_TASK_ID is set, only predicts
    assigned folds and saves predictions to disk. When all predictions
    exist (no array context or aggregation run), generates figures and metrics.
    """
    import numpy as np
    import glob as glob_module
    from remo_id.reports import figures as plots
    from remo_id.data import RemoIDDataModule
    from remo_id.inference.core import extract_checkpoint_metadata
    from remo_id.models import get_augmentation_backbone
    from remo_id.workflows.cv import (
        collect_cv_predictions_by_fold, aggregate_cv_results,
        load_cached_cv_predictions, aggregate_to_patient_level,
    )

    cv_dir = kwargs["cv_dir"]
    patient_level = kwargs["patient_level"]
    default_subdir = "cv_performance_last" if kwargs["use_last"] else "cv_performance"
    if patient_level != "none":
        default_subdir += f"_patient_{patient_level}"
    out = kwargs["output"] or os.path.join(cv_dir, "analysis", default_subdir)
    os.makedirs(out, exist_ok=True)

    is_array_task = os.environ.get('SLURM_ARRAY_TASK_ID') is not None

    # Load class config
    class_definitions, class_names, num_classes, task_type, exclude_filters, include_filters, default_class_idx = (
        _load_class_config(kwargs["class_config"])
    )

    # Find first fold checkpoint to determine model architecture
    fold_dirs = sorted(glob_module.glob(os.path.join(cv_dir, "repeat*_fold*")))
    if not fold_dirs:
        raise click.UsageError(f"No fold directories found in {cv_dir}")
    first_ckpt = None
    for fd in fold_dirs:
        all_ckpts = sorted(glob_module.glob(os.path.join(fd, "*.ckpt")))
        ckpts = [c for c in all_ckpts if "last" not in os.path.basename(c)]
        if not ckpts:
            ckpts = [c for c in all_ckpts if "last" in os.path.basename(c)]
        if ckpts:
            first_ckpt = ckpts[0]
            break
    if not first_ckpt:
        raise click.UsageError(f"No checkpoints found in {cv_dir}")

    ckpt_meta = extract_checkpoint_metadata(first_ckpt)
    model_name = ckpt_meta.get("model_name", "resnet18")
    backbone_name = get_augmentation_backbone(model_name)

    # Build data module with all experiments
    if model_name == "embedding-mil":
        from remo_id.data import EmbeddingDataModule
        if not kwargs.get("model_version"):
            raise click.UsageError("embedding-mil cv-performance requires --model_version")
        ds = EmbeddingDataModule(
            model_version=kwargs["model_version"], cell_config=kwargs["cell_config"],
            pool_cap=kwargs["pool_cap"],
            database_column="class_label", task_type=task_type, num_classes=num_classes,
            class_definitions=class_definitions, exclude_filters=exclude_filters,
            include_filters=include_filters, default_class_idx=default_class_idx,
            batch_size=kwargs["batch_size"], skip_null_labels=True,
            load_workers=kwargs.get("load_workers"),
        )
    else:
        ds = RemoIDDataModule(
            database_column="class_label",
            database_label=None,
            study_id=None,
            threshold=None,
            threshold_mode="==",
            negative_study_ids=[],
            negative_experiment_ids=[],
            batch_size=kwargs["batch_size"],
            backbone_name=backbone_name,
            train=False,
            task_type=task_type,
            num_classes=num_classes,
            class_definitions=class_definitions,
            exclude_filters=exclude_filters,
            include_filters=include_filters,
            default_class_idx=default_class_idx,
            skip_null_labels=True,
        )

    if is_array_task:
        # Phase 1: predict assigned folds only, save to disk
        print(f"Array task: predicting assigned folds from {cv_dir}...")
        collect_cv_predictions_by_fold(
            cv_dir, ds, class_names,
            bag_size=kwargs["bag_size"], seed=kwargs["seed"],
            n_bags=kwargs["n_bags"], use_last=kwargs["use_last"],
        )
        print("Fold predictions saved. Aggregation will run separately.")
        return

    # Non-array: try loading cached predictions first, fall back to full inference
    pooled, per_fold, missing = load_cached_cv_predictions(cv_dir, ds, use_last=kwargs["use_last"])
    if missing:
        print(f"{len(missing)} folds missing cached predictions, running inference...")
        pooled, per_fold = collect_cv_predictions_by_fold(
            cv_dir, ds, class_names,
            bag_size=kwargs["bag_size"], seed=kwargs["seed"],
            n_bags=kwargs["n_bags"], use_last=kwargs["use_last"],
        )

    y_true, y_probs, y_pred, experiment_ids, sample_ids = pooled

    if len(per_fold) == 0:
        raise RuntimeError(
            "No fold predictions collected. Check that checkpoints exist in fold directories."
        )

    print(f"Pooled: {len(y_true)} experiments from {len(per_fold)} folds")

    # Apply patient-level aggregation if requested
    if patient_level != "none":
        if sample_ids is None:
            raise click.UsageError(
                "Patient-level aggregation requires sample_id in predictions. "
                "Re-run CV predictions to include sample_id."
            )
        n_before = len(y_true)
        y_true, y_probs, y_pred, sample_ids = aggregate_to_patient_level(
            y_true, y_probs, y_pred, experiment_ids, sample_ids, mode=patient_level,
        )
        print(f"Patient-level ({patient_level}): {n_before} experiments → {len(y_true)} patients")

        # Also aggregate per-fold predictions for fold-level metrics
        for fold_data in per_fold:
            if "sample_ids" in fold_data and fold_data["sample_ids"] is not None:
                ft, fp, fpred, fsids = aggregate_to_patient_level(
                    fold_data["y_true"], fold_data["y_probs"], fold_data["y_pred"],
                    fold_data["experiment_ids"], fold_data["sample_ids"],
                    mode=patient_level,
                )
                fold_data["y_true"] = ft
                fold_data["y_probs"] = fp
                fold_data["y_pred"] = fpred

    pooled = (y_true, y_probs, y_pred)

    # Aggregate metrics
    bootstrap_n = kwargs["bootstrap_n"]
    results = aggregate_cv_results(per_fold, pooled, class_names, bootstrap_n=bootstrap_n, task_type=task_type)

    # Youden-J optimal confusion matrix
    from remo_id.analysis.optimal_thresholds import find_youden_thresholds
    youden = find_youden_thresholds(y_true, y_probs, class_names)
    youden_thresholds = np.array([r["optimal_threshold"] for r in youden])
    y_pred_youden = (y_probs - youden_thresholds).argmax(axis=1)

    # Generate figures from pooled predictions
    figs = [
        ("roc_curves",         plots.plot_roc_curves(y_true, y_probs, class_names, bootstrap_n=bootstrap_n, task_type=task_type)),
        ("pr_curves",          plots.plot_pr_curves(y_true, y_probs, class_names, bootstrap_n=bootstrap_n, task_type=task_type)),
        ("confusion_matrix",   plots.plot_confusion_matrix(y_true, y_pred, class_names)),
        ("confusion_matrix_youden", plots.plot_confusion_matrix(
            y_true, y_pred_youden, class_names,
            title="Confusion Matrix (Youden-J thresholds)")),
        ("calibration",        plots.plot_calibration(y_true, y_probs, class_names, task_type=task_type)),
        ("prob_distributions", plots.plot_prob_distributions(y_true, y_probs, class_names, task_type=task_type)),
    ]

    # K-fold variability box plots
    if results["per_fold_metrics"]:
        figs.append(("kfold_variability",
                     plots.plot_kfold_variability(results["per_fold_metrics"], class_names)))

    _save_figures(figs, out)

    # Save metrics CSV with bootstrap CIs
    results["pooled_metrics_df"].to_csv(os.path.join(out, "metrics.csv"), index=False)
    print(results["pooled_metrics_df"].to_string(index=False))

    # Save per-fold metrics CSV
    import pandas as pd
    fold_rows = []
    for i, fold_data in enumerate(per_fold):
        row = {"repeat": fold_data["repeat"], "fold": fold_data["fold"]}
        for metric_name, values in results["per_fold_metrics"].items():
            row[metric_name] = values[i] if i < len(values) else np.nan
        fold_rows.append(row)
    pd.DataFrame(fold_rows).to_csv(os.path.join(out, "per_fold_metrics.csv"), index=False)

    print(f"Saved CV performance analysis to {out}")


# --- cv-baselines ---

@analysis_group.command("cv-baselines")
@click.option("--cv_dir", required=True, type=click.Path(exists=True),
              help="Path to CV output directory (contains repeat*_fold* subdirs)")
@click.option("--class_config", required=True, type=click.Path(exists=True),
              help="Path to YAML class config")
@click.option("--output", default=None, type=click.Path(),
              help="Output directory (default: {cv_dir}/analysis/cv_baselines/)")
@click.option("--bag_size", type=int, default=200, show_default=True,
              help="Number of images per bag")
@click.option("--seed", type=int, default=403, show_default=True,
              help="Random seed")
@click.option("--batch_size", type=int, default=256, show_default=True,
              help="DataModule batch size")
@click.option("--bootstrap_n", type=int, default=1000, show_default=True,
              help="Bootstrap iterations for confidence intervals")
def cv_baselines(**kwargs):
    """Run sklearn baselines across all CV folds and pool predictions for fair comparison.

    Extracts features from each fold's checkpoint, trains sklearn classifiers
    (LogisticRegression, DummyClassifier) on fold train splits, predicts on fold
    val splits, then pools predictions identically to cv-performance.

    Pretrained (ImageNet) features are extracted once for all experiments and
    partitioned per fold to avoid redundant computation.
    """
    import pandas as pd
    import glob as glob_module
    from remo_id.data import RemoIDDataModule
    from remo_id.inference.core import extract_checkpoint_metadata
    from remo_id.models import get_augmentation_backbone
    from remo_id.analysis.baselines import (
        extract_features, create_pretrained_model, predict_fold_baselines,
    )
    from remo_id.workflows.cv import (
        _save_fold_predictions, _deduplicate_pooled_predictions,
        aggregate_cv_results,
    )

    cv_dir = kwargs["cv_dir"]
    out = kwargs["output"] or os.path.join(cv_dir, "analysis", "cv_baselines")
    os.makedirs(out, exist_ok=True)

    # Load class config
    class_definitions, class_names, num_classes, task_type, exclude_filters, include_filters, default_class_idx = (
        _load_class_config(kwargs["class_config"])
    )

    # Find first fold checkpoint to determine architecture
    fold_dirs = sorted(glob_module.glob(os.path.join(cv_dir, "repeat*_fold*")))
    if not fold_dirs:
        raise click.UsageError(f"No fold directories found in {cv_dir}")
    first_ckpt = None
    for fd in fold_dirs:
        all_ckpts = sorted(glob_module.glob(os.path.join(fd, "*.ckpt")))
        ckpts = [c for c in all_ckpts if "last" not in os.path.basename(c)]
        if not ckpts:
            ckpts = [c for c in all_ckpts if "last" in os.path.basename(c)]
        if ckpts:
            first_ckpt = ckpts[0]
            break
    if not first_ckpt:
        raise click.UsageError(f"No checkpoints found in {cv_dir}")

    ckpt_meta = extract_checkpoint_metadata(first_ckpt)
    model_name = ckpt_meta.get("model_name", "resnet18")
    backbone_name = get_augmentation_backbone(model_name)

    # Build full data module with all experiments
    ds = RemoIDDataModule(
        database_column="class_label",
        database_label=None,
        study_id=None,
        threshold=None,
        threshold_mode="==",
        negative_study_ids=[],
        negative_experiment_ids=[],
        batch_size=kwargs["batch_size"],
        backbone_name=backbone_name,
        train=False,
        task_type=task_type,
        num_classes=num_classes,
        class_definitions=class_definitions,
        exclude_filters=exclude_filters,
        include_filters=include_filters,
        default_class_idx=default_class_idx,
        skip_null_labels=True,
    )

    # Load CV splits
    splits_path = os.path.join(cv_dir, "cv_splits.yaml")
    if not os.path.exists(splits_path):
        raise click.UsageError(f"No cv_splits.yaml found in {cv_dir}")
    with open(splits_path) as f:
        splits_data = yaml.safe_load(f)

    # One-time pretrained feature extraction (ImageNet weights are fold-independent)
    print("Extracting pretrained features for all experiments (one-time)...")
    pretrained_model = create_pretrained_model(first_ckpt)
    pretrained_features = {}
    for pool in ("mean", "max"):
        X_pt, y_pt, ids_pt = extract_features(
            pretrained_model, ds, bag_size=kwargs["bag_size"],
            seed=kwargs["seed"], pooling=pool,
        )
        pretrained_features[pool] = (X_pt, y_pt, ids_pt)
        print(f"  Pretrained {pool}-pool: {X_pt.shape[0]} experiments, {X_pt.shape[1]} features")
    del pretrained_model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Iterate over folds
    all_fold_results = {}  # {baseline_name: [fold_dicts]}

    for repeat_idx, repeat_data in enumerate(splits_data["repeats"]):
        for fold_data in repeat_data:
            fold_idx = fold_data["fold"]
            fold_label = f"repeat{repeat_idx}_fold{fold_idx}"
            fold_dir = os.path.join(cv_dir, fold_label)

            # Find fold checkpoint
            all_ckpts = sorted(glob_module.glob(os.path.join(fold_dir, "*.ckpt")))
            ckpts = [c for c in all_ckpts if "last" not in os.path.basename(c)]
            if not ckpts:
                ckpts = [c for c in all_ckpts if "last" in os.path.basename(c)]
            if not ckpts:
                print(f"Warning: no checkpoint for {fold_label}, skipping")
                continue
            ckpt_path = ckpts[0]

            train_ids = [str(eid) for eid in fold_data["train_experiment_ids"]]
            val_ids = [str(eid) for eid in fold_data["val_experiment_ids"]]

            print(f"\n{fold_label}: {len(train_ids)} train, {len(val_ids)} val experiments")

            fold_results = predict_fold_baselines(
                checkpoint_path=ckpt_path,
                data_module=ds,
                train_experiment_ids=train_ids,
                val_experiment_ids=val_ids,
                class_definitions=class_definitions,
                exclude_filters=exclude_filters,
                include_filters=include_filters,
                default_class_idx=default_class_idx,
                task_type=task_type,
                num_classes=num_classes,
                backbone_name=backbone_name,
                bag_size=kwargs["bag_size"],
                batch_size=kwargs["batch_size"],
                seed=kwargs["seed"],
                pretrained_features=pretrained_features,
            )

            # Collect per-baseline predictions
            for baseline_name, (y_true, y_probs, y_pred, exp_ids) in fold_results.items():
                if baseline_name not in all_fold_results:
                    all_fold_results[baseline_name] = []

                # Save fold predictions
                baseline_dir = os.path.join(out, baseline_name.replace(" ", "_").replace("+", ""))
                fold_out = os.path.join(baseline_dir, fold_label)
                os.makedirs(fold_out, exist_ok=True)
                _save_fold_predictions(
                    fold_out, repeat_idx, fold_idx,
                    y_true, y_probs, y_pred,
                    experiment_ids=exp_ids,
                )

                all_fold_results[baseline_name].append({
                    "repeat": repeat_idx,
                    "fold": fold_idx,
                    "y_true": y_true,
                    "y_probs": y_probs,
                    "y_pred": y_pred,
                    "experiment_ids": exp_ids,
                })

    # Pool and compute metrics per baseline
    comparison_rows = []
    for baseline_name, per_fold in all_fold_results.items():
        print(f"\n{'='*60}")
        print(f"Pooling: {baseline_name} ({len(per_fold)} folds)")
        print(f"{'='*60}")

        y_true, y_probs, y_pred, _, _, _ = _deduplicate_pooled_predictions(per_fold)
        pooled = (y_true, y_probs, y_pred)

        results = aggregate_cv_results(per_fold, pooled, class_names, bootstrap_n=kwargs["bootstrap_n"])

        # Save per-baseline metrics
        baseline_dir = os.path.join(out, baseline_name.replace(" ", "_").replace("+", ""))
        os.makedirs(baseline_dir, exist_ok=True)
        results["pooled_metrics_df"].to_csv(os.path.join(baseline_dir, "metrics.csv"), index=False)

        # Extract summary row for comparison table
        metrics_df = results["pooled_metrics_df"]
        row = {"model": baseline_name}
        for _, mrow in metrics_df.iterrows():
            cls = mrow.get("class", "")
            for col in ("auc", "ap", "f1@best", "precision@best", "recall@best"):
                if col in mrow:
                    row[f"{col}_{cls}"] = mrow[col]
        comparison_rows.append(row)

    # Load MIL model's pooled metrics if available
    mil_metrics_path = os.path.join(cv_dir, "analysis", "cv_performance", "metrics.csv")
    if os.path.exists(mil_metrics_path):
        print(f"\nLoading MIL model metrics from {mil_metrics_path}")
        mil_df = pd.read_csv(mil_metrics_path)
        row = {"model": f"MIL ({model_name})"}
        for _, mrow in mil_df.iterrows():
            cls = mrow.get("class", "")
            for col in ("auc", "ap", "f1@best", "precision@best", "recall@best"):
                if col in mrow:
                    row[f"{col}_{cls}"] = mrow[col]
        comparison_rows.append(row)

    # Save comparison table
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(out, "comparison.csv"), index=False)
    print(f"\n{'='*60}")
    print("Model Comparison (pooled CV predictions with bootstrap CIs)")
    print(f"{'='*60}")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved comparison to {os.path.join(out, 'comparison.csv')}")


# --- optimal-thresholds ---

@analysis_group.command("optimal-thresholds")
@_pa_common_options_flexible
@_cv_aware_options
@click.option("--min_sensitivity", type=float, default=0.95, show_default=True,
              help="Default minimum sensitivity for all classes")
@click.option("--sensitivity_targets", type=str, default=None,
              help="Per-class overrides, comma-separated name:value pairs (e.g. 'disease:0.99,healthy:0.90')")
@click.option("--target_class", type=str, default=None,
              help="Class name highlighted in the ROC operating-point figures (figures are skipped without it)")
@click.option("--dual", is_flag=True, default=False,
              help="Also emit the dual operating-point table (Youden-J + high-sensitivity rows per class)")
@click.option("--per_fold", is_flag=True, default=False,
              help="Pick thresholds on each CV fold's held-out preds (CV-only); "
                   "reports per-fold distribution + pooled metrics when each "
                   "fold's threshold is applied to its own held-out data.")
def optimal_thresholds(**kwargs):
    """Find optimal decision thresholds per class given minimum sensitivity constraints.

    Supports both --checkpoint (single model) and --cv_dir (pooled CV predictions).
    With --dual, additionally emits a combined Youden-J + rule-out table.
    With --per_fold (CV-only), picks the threshold on each fold's held-out preds
    and reports the per-fold distribution + deployment-equivalent pooled metrics.
    """
    import json
    from remo_id.analysis.optimal_thresholds import (
        find_optimal_thresholds as _find,
        build_sensitivity_targets,
        dual_operating_points_table,
        find_thresholds_per_fold,
        summarize_per_fold_thresholds,
        pooled_metrics_with_per_fold_thresholds,
    )
    from remo_id.reports import figures as plots

    mode = _validate_analysis_mode(kwargs)

    if mode == "cv":
        out = resolve_output_dir_cv(kwargs["output"], kwargs["cv_dir"], "optimal-thresholds")
        ctx = _build_context_from_cv_predictions(
            kwargs["cv_dir"], kwargs["class_config"], use_last=kwargs["use_last"],
        )
        print(f"Using pooled CV predictions from {kwargs['cv_dir']}")
    else:
        out = resolve_output_dir(kwargs["output"], kwargs["checkpoint"], "optimal-thresholds")
        # Try saved predictions first
        ctx = _build_context_from_predictions(kwargs["checkpoint"], kwargs["split"])
        if ctx is not None:
            print("Using saved predictions (no GPU inference needed)")
        else:
            ctx = _build_analysis_context(
                checkpoint=kwargs["checkpoint"],
                class_config_path=kwargs["class_config"],
                splits_path=kwargs["splits"],
                split_name=kwargs["split"],
                bag_size=kwargs["bag_size"],
                seed=kwargs["seed"],
                batch_size=kwargs["batch_size"],
            )

    y_true, y_probs = ctx["y_true"], ctx["y_probs"]
    class_names = ctx["class_names"]

    targets = build_sensitivity_targets(
        class_names, kwargs["min_sensitivity"], kwargs["sensitivity_targets"]
    )
    results = _find(y_true, y_probs, class_names, targets)

    # Save JSON
    os.makedirs(out, exist_ok=True)
    json_path = os.path.join(out, "optimal_thresholds.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save figure
    import matplotlib.pyplot as plt
    target_class = kwargs.get("target_class")
    if target_class is not None:
        fig = plots.plot_optimal_thresholds(y_true, y_probs, class_names, results,
                                            target_class=target_class)
        fig.savefig(os.path.join(out, "optimal_thresholds.png"))
        plt.close(fig)
    else:
        print("Skipping ROC operating-point figures (pass --target_class to draw them)")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Class':<20} {'Target':>8} {'Threshold':>10} {'Sens':>8} {'Spec':>8} {'Prec':>8} {'F1':>8} {'NPV':>8}")
    print(f"{'-'*80}")
    for r in results:
        warn = " (!)" if "warning" in r else ""
        print(f"{r['class_name']:<20} {r['target_sensitivity']:>8.2f} {r['optimal_threshold']:>10.3f} "
              f"{r['achieved_sensitivity']:>8.3f} {r['achieved_specificity']:>8.3f} "
              f"{r['achieved_precision']:>8.3f} {r['achieved_f1']:>8.3f} {r['achieved_npv']:>8.3f}{warn}")
    print(f"{'='*80}")

    for r in results:
        if "warning" in r:
            print(f"WARNING: {r['warning']}")

    if kwargs.get("dual"):
        from remo_id.analysis.optimal_thresholds import find_youden_thresholds
        dual_csv = os.path.join(out, "dual_operating_points.csv")
        dual_df = dual_operating_points_table(
            y_true, y_probs, class_names, targets, output_csv=dual_csv,
        )
        print("\nDual operating points (Youden-J + rule-out):")
        print(dual_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        if target_class is not None:
            youden = find_youden_thresholds(y_true, y_probs, class_names)
            dual_fig = plots.plot_dual_operating_points(
                y_true, y_probs, class_names, youden, results, target_class=target_class,
            )
            dual_fig.savefig(os.path.join(out, "dual_operating_points.png"))
            plt.close(dual_fig)

    if kwargs.get("per_fold"):
        if mode != "cv":
            raise click.UsageError("--per_fold requires --cv_dir (needs fold structure).")
        per_fold = ctx.get("per_fold")
        if not per_fold:
            raise click.UsageError("No per-fold predictions found in cv_dir.")

        per_fold_df = find_thresholds_per_fold(
            per_fold, class_names, rule="sens_constrained",
            sensitivity_targets=targets,
        )
        summary_df = summarize_per_fold_thresholds(per_fold_df)
        deployed_df = pooled_metrics_with_per_fold_thresholds(
            per_fold, class_names, per_fold_df,
        )

        per_fold_csv = os.path.join(out, "per_fold_thresholds.csv")
        summary_csv = os.path.join(out, "per_fold_thresholds_summary.csv")
        deployed_csv = os.path.join(out, "per_fold_thresholds_deployed_pooled.csv")
        per_fold_df.to_csv(per_fold_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        deployed_df.to_csv(deployed_csv, index=False)

        if target_class is not None:
            pf_fig = plots.plot_per_fold_thresholds(per_fold_df, target_class=target_class)
            pf_fig.savefig(os.path.join(out, "per_fold_thresholds.png"))
            plt.close(pf_fig)

        print("\nPer-fold threshold stability (median [IQR]):")
        for _, r in summary_df.iterrows():
            print(
                f"  {r['class']:<20}  n={int(r['n_folds']):>2}  "
                f"thr={r['threshold_median']:.3f} [{r['threshold_q25']:.3f}–{r['threshold_q75']:.3f}]  "
                f"sens={r['sens_median']:.3f} [{r['sens_q25']:.3f}–{r['sens_q75']:.3f}]  "
                f"spec={r['spec_median']:.3f} [{r['spec_q25']:.3f}–{r['spec_q75']:.3f}]"
            )

        print("\nPooled held-out metrics when each fold uses its own threshold "
              "(deployment-equivalent):")
        print(deployed_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nSaved optimal thresholds to {out}")


# --- pairwise ---

@analysis_group.command("pairwise")
@_pa_common_options_flexible
@_cv_aware_options
@click.option("--class_positive", required=True, type=str,
              help="Positive class name (e.g. 'disease')")
@click.option("--class_negative", required=True, type=str,
              help="Negative class name (e.g. 'control')")
@click.option("--sensitivity_target", type=float, default=0.95, show_default=True,
              help="Sensitivity target for the rule-out operating point")
def pairwise(**kwargs):
    """Pairwise (one-vs-one) discrimination between two classes.

    Filters pooled predictions to the two named classes, renormalizes
    probabilities, and reports binary metrics with 95% bootstrap CIs.
    Produces a pairwise metrics table, pairwise ROC, and 2x2 confusion matrix.
    """
    from remo_id.analysis.pairwise import run_pairwise

    mode = _validate_analysis_mode(kwargs)
    subdir = f"pairwise_{kwargs['class_positive']}_vs_{kwargs['class_negative']}"

    if mode == "cv":
        out = resolve_output_dir_cv(kwargs["output"], kwargs["cv_dir"], subdir)
        ctx = _build_context_from_cv_predictions(
            kwargs["cv_dir"], kwargs["class_config"], use_last=kwargs["use_last"],
        )
        print(f"Using pooled CV predictions from {kwargs['cv_dir']}")
    else:
        out = resolve_output_dir(kwargs["output"], kwargs["checkpoint"], subdir)
        ctx = _build_context_from_predictions(kwargs["checkpoint"], kwargs["split"])
        if ctx is None:
            ctx = _build_analysis_context(
                checkpoint=kwargs["checkpoint"],
                class_config_path=kwargs["class_config"],
                splits_path=kwargs["splits"],
                split_name=kwargs["split"],
                bag_size=kwargs["bag_size"],
                seed=kwargs["seed"],
                batch_size=kwargs["batch_size"],
            )

    run_pairwise(
        y_true=ctx["y_true"],
        y_probs=ctx["y_probs"],
        class_names=ctx["class_names"],
        class_positive=kwargs["class_positive"],
        class_negative=kwargs["class_negative"],
        output_dir=out,
        bootstrap_n=kwargs["bootstrap_n"],
        sensitivity_target=kwargs["sensitivity_target"],
    )


# --- cascade ---

@analysis_group.command("cascade")
@click.option("--models_yaml", default=None, type=click.Path(exists=True),
              help="YAML listing (checkpoint, class_config, predictions_kind) per cascade node. "
                   "Alternative to repeated --model.")
@click.option("--model", "model_args", multiple=True,
              help="Cascade model entry 'checkpoint,class_config,kind[,predictions_dir]'. "
                   "Repeatable. kind is one of single|cv|loo.")
@click.option("--split", default="val", show_default=True,
              help="Split to evaluate (applied to single-run predictions; cv/loo are implicitly val).")
@click.option("--output", default=None, type=click.Path(),
              help="Output dir. Default: {models_yaml.parent}/analysis/cascade/{name}/ "
                   "or {first_checkpoint.parent}/analysis/cascade/{name}/ if no yaml.")
@click.option("--bootstrap_n", type=int, default=1000, show_default=True,
              help="Bootstrap iterations for leaf-level CIs.")
@click.option("--print_tree", is_flag=True, default=False,
              help="Infer the cascade tree from the class_configs, print it, and exit.")
@click.option("--target_sensitivity", "target_sensitivity", multiple=True,
              help="Per-class sensitivity for the sensitivity-routing walk as NAME=VALUE "
                   "(repeatable or comma-separated, e.g. 'disease=0.99').")
@click.option("--default_sensitivity", type=float, default=0.95, show_default=True,
              help="Sensitivity for classes not listed in --target_sensitivity.")
def cascade(models_yaml, model_args, split, output, bootstrap_n, print_tree,
            target_sensitivity, default_sensitivity):
    """End-to-end cascade evaluation over an auto-inferred tree of trained classifiers.

    \b
    The tree is derived from each model's class_config: model B is the child of
    model A under route class C iff B's in-scope experiment set equals exactly
    the experiments A labels as class C. Leaves are the classes of the deepest
    nodes. Requires strict coverage — every in-scope experiment must have
    predictions at the correct level.
    """
    from remo_id.analysis.cascade import run_cascade

    targets = {}
    for item in ",".join(target_sensitivity).split(","):
        if not item.strip():
            continue
        name, sep, value = item.rpartition("=")
        if not sep or not name.strip():
            raise click.UsageError(f"--target_sensitivity expects NAME=VALUE, got '{item}'")
        targets[name.strip()] = float(value)

    run_cascade(
        models_yaml=models_yaml,
        cli_models=list(model_args),
        split=split,
        output=output,
        bootstrap_n=bootstrap_n,
        print_tree=print_tree,
        target_sensitivity=targets,
        default_sensitivity=default_sensitivity,
    )


# --- moe ---

@analysis_group.command("moe")
@click.option("--cv_dir_a", required=True, type=click.Path(exists=True),
              help="Binary gate CV directory (condition of interest vs rest)")
@click.option("--cv_dir_b", required=True, type=click.Path(exists=True),
              help="Expert multiclass CV directory (classes within the gate-negative set)")
@click.option("--class_config_b", required=True, type=click.Path(exists=True),
              help="Class YAML for the expert model (defines expert class names)")
@click.option("--gate_class_name", required=True, type=str,
              help="Name of the gate-positive class in the combined output (e.g. 'infected')")
@click.option("--output", required=True, type=click.Path(),
              help="Output directory")
@click.option("--bootstrap_n", type=int, default=1000, show_default=True,
              help="Bootstrap iterations for CIs")
@click.option("--use_last", is_flag=True, default=False,
              help="Use last.ckpt predictions")
def moe(cv_dir_a, cv_dir_b, class_config_b, gate_class_name, output, bootstrap_n, use_last):
    """Combine a binary gate (Model A) + multiclass expert (Model B) into one MOE output.

    Inner-joins pooled CV predictions by experiment_id. Combined probs:
    P(gate+) = P_A(pos); P(class_i) = P_A(neg) * P_B(i). Writes a per-experiment
    CSV; metrics are only computed if run programmatically with y_true_combined.
    """
    from remo_id.analysis.moe import run_moe

    _, expert_class_names, _, _, _, _, _ = _load_class_config(class_config_b)
    run_moe(
        cv_dir_a=cv_dir_a,
        cv_dir_b=cv_dir_b,
        expert_b_class_names=expert_class_names,
        gate_class_name=gate_class_name,
        output_dir=output,
        bootstrap_n=bootstrap_n,
        use_last=use_last,
    )


# --- compare-configs ---

@analysis_group.command("compare-configs")
@click.option("--cv_dir", "cv_dirs", multiple=True, required=True, type=click.Path(exists=True),
              help="CV directory (repeatable; order matches --class_config/--label)")
@click.option("--class_config", "class_configs", multiple=True, required=True,
              type=click.Path(exists=True),
              help="Class YAML per cv_dir (repeatable, same order)")
@click.option("--label", "labels", multiple=True, default=None,
              help="Human-readable label per cv_dir (repeatable). Defaults to basename.")
@click.option("--output", required=True, type=click.Path(),
              help="Output CSV path for the comparison table")
@click.option("--bootstrap_n", type=int, default=1000, show_default=True,
              help="Bootstrap iterations for 95%% CIs")
@click.option("--use_last", is_flag=True, default=False,
              help="Use last.ckpt predictions")
def compare_configs(cv_dirs, class_configs, labels, output, bootstrap_n, use_last):
    """Cross-sweep table: aggregate macro AUC/F1/error/fold std across N cv_dirs."""
    from remo_id.analysis.configuration_comparison import compare_configurations

    if len(cv_dirs) != len(class_configs):
        raise click.UsageError(
            f"Got {len(cv_dirs)} --cv_dir and {len(class_configs)} --class_config; "
            "these must be the same length and in matching order."
        )
    if labels and len(labels) != len(cv_dirs):
        raise click.UsageError(
            f"Got {len(labels)} --label but {len(cv_dirs)} --cv_dir; "
            "--label is optional but must match length when provided."
        )

    configs = []
    for i, (cd, cc) in enumerate(zip(cv_dirs, class_configs)):
        _, class_names, _, task_type, _, _, _ = _load_class_config(cc)
        configs.append({
            "cv_dir": cd,
            "class_names": class_names,
            "task_type": task_type,
            "label": labels[i] if labels else None,
            "use_last": use_last,
        })

    compare_configurations(configs, output_path=output, bootstrap_n=bootstrap_n)


# --- all ---

@analysis_group.command("all")
@_pa_common_options_flexible
@_cv_aware_options
@click.option("--seeds", default="403,42,7,123,999",
              help="Comma-separated seeds for robustness/bag-size sweeps")
@click.option("--bag_sizes", default="1,3,5,10,25,50,100,200,400,700,1000",
              help="Comma-separated bag sizes for sweep")
@click.option("--n_top", type=int, default=8, show_default=True,
              help="Top-attention frames per class for interpret")
@click.option("--target_class", type=str, default=None,
              help="Class name highlighted in optimal-thresholds figures")
@click.pass_context
def all_analyses(ctx, **kwargs):
    """Run all analyses: performance, bag-size, interpret, robustness, baselines, error-analysis.

    With --cv_dir, only runs commands that work on pooled predictions (performance,
    error-analysis, optimal-thresholds). GPU-only commands are skipped.
    """
    import traceback

    mode = _validate_analysis_mode(kwargs)

    # GPU-only commands that can't work with --cv_dir
    gpu_only_commands = {"bag-size", "interpret", "robustness", "baselines"}

    all_commands = [
        ("performance",    {}),
        ("bag-size",       {"bag_sizes": kwargs["bag_sizes"], "seeds": kwargs["seeds"]}),
        ("interpret",      {"n_top": kwargs["n_top"]}),
        ("robustness",     {"seeds": kwargs["seeds"]}),
        ("baselines",      {}),
        ("error-analysis", {}),
        ("optimal-thresholds", {"target_class": kwargs["target_class"]}),
    ]

    if mode == "cv":
        common = {k: kwargs[k] for k in
                  ("class_config", "split",
                   "output", "bag_size", "seed", "batch_size", "bootstrap_n",
                   "cv_dir", "use_last")}
        # checkpoint/splits not used in CV mode but commands expect the keys
        common["checkpoint"] = None
        common["splits"] = None
    else:
        common = {k: kwargs[k] for k in
                  ("checkpoint", "class_config", "splits", "split",
                   "output", "bag_size", "seed", "batch_size", "bootstrap_n")}
        # CV options not used in checkpoint mode but cv-aware commands expect them
        common["cv_dir"] = None
        common["use_last"] = False

    failed = []
    skipped = []
    for cmd, extra in all_commands:
        if mode == "cv" and cmd in gpu_only_commands:
            print(f"\n{'='*60}\nSkipping: analysis {cmd} (requires GPU, not supported with --cv_dir)\n{'='*60}")
            skipped.append(cmd)
            continue
        print(f"\n{'='*60}\nRunning: analysis {cmd}\n{'='*60}")
        try:
            ctx.invoke(analysis_group.commands[cmd], **{**common, **extra})
        except Exception as e:
            print(f"FAILED: analysis {cmd}: {e}")
            traceback.print_exc()
            failed.append(cmd)

    if skipped:
        print(f"\nSkipped {len(skipped)} GPU-only command(s): {', '.join(skipped)}")
    if failed:
        print(f"\n{'='*60}")
        print(f"WARNING: {len(failed)} subcommand(s) failed: {', '.join(failed)}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}\nAll analyses completed successfully.\n{'='*60}")


# ---------------------------------------------------------------------------
# analysis occlusion: cell occlusion screen
# ---------------------------------------------------------------------------


@analysis_group.group("occlusion")
def occlusion_group():
    """Cell occlusion screen: which cells a MIL model's prediction depends on.

    Scores the given experiments with ONE --checkpoint. For held-out explanations of a
    CV run, call each stage once per fold with that fold's checkpoint and its held-out
    --experiment_ids (folds can share one --work_dir). Needs per-experiment label masks
    and regionprops next to the image stack (see remo_id/interpret/cell_occlusion.py).
    Run the stages in order; each resumes per experiment.

    \b
    Example:
        remo-id analysis occlusion background --checkpoint run/last.ckpt --work_dir occ/
        remo-id analysis occlusion attribution --checkpoint run/last.ckpt --work_dir occ/ \\
            --cell_config cells.yaml
        remo-id analysis occlusion screen --checkpoint run/last.ckpt --work_dir occ/ \\
            --cell_config cells.yaml
        remo-id analysis occlusion report --work_dir occ/
    """
    pass


def _occlusion_options(f):
    """Model, experiments, frame sampling and work dir of the per-experiment stages."""
    opts = [
        click.option("--checkpoint", type=click.Path(exists=True), default=None,
                     help="Image MIL checkpoint that scores every experiment (background "
                          "uses it only to find classes.yaml)"),
        click.option("--experiment_ids", multiple=True, default=(),
                     help="Experiments to explain (repeatable; default: every experiment "
                          "of the class config / manifest)"),
        click.option("--work_dir", required=True, type=click.Path(),
                     help="Stage outputs (bg/, attribution/, screen/, experiments/)"),
        click.option("--seed", type=int, default=403, show_default=True,
                     help="Seed of the frame subset and of every fill / mask draw"),
        click.option("--n_frames", type=int, default=500, show_default=True,
                     help="Frames screened per experiment"),
        click.option("--min_image_index", type=int, default=1000, show_default=True,
                     help="Skip frames below this index (run warm-up)"),
        click.option("--device", default=None, help="torch device (default: cuda if available)"),
        click.option("--overwrite", is_flag=True, help="Recompute finished experiments"),
    ]
    f = _class_source_options(f)
    for o in reversed(opts):
        f = o(f)
    return f


def _cell_config_option(f):
    return click.option("--cell_config", required=True, type=click.Path(exists=True),
                        help="cell_config YAML typing cells by regionprops gates "
                             "(examples/cell_configs/wbc_rbc_plt.yaml)")(f)


def _occlusion_setup(kwargs, need_checkpoint=True, **settings):
    """(OcclusionConfig, experiments table) for a stage's options. The class config
    defaults to classes.yaml next to the checkpoint."""
    from remo_id.interpret.cell_occlusion import OcclusionConfig, resolve_experiments
    ckpt = kwargs.get("checkpoint")
    if need_checkpoint and not ckpt:
        raise click.UsageError("--checkpoint is required for this stage.")
    work_dir = os.path.abspath(kwargs["work_dir"])
    if kwargs.get("manifest") or kwargs.get("class_config"):
        class_config = _resolve_class_config_path(kwargs, work_dir)
    elif ckpt and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(ckpt)), "classes.yaml")):
        class_config = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "classes.yaml")
    else:
        raise click.UsageError("Provide --class_config or --manifest.")
    cfg = OcclusionConfig(work_dir=work_dir, checkpoint=ckpt, cell_config=kwargs.get("cell_config"),
                          device=kwargs["device"], seed=kwargs["seed"], n_frames=kwargs["n_frames"],
                          min_image_index=kwargs["min_image_index"], **settings)
    return cfg, resolve_experiments(class_config, list(kwargs["experiment_ids"]))


@occlusion_group.command("background")
@_occlusion_options
def occlusion_background(**kwargs):
    """Empty-slide background model per experiment (CPU) -> {work_dir}/bg/."""
    from remo_id.interpret.cell_occlusion import run_background
    cfg, ex = _occlusion_setup(kwargs, need_checkpoint=False)
    run_background(cfg, ex, kwargs["overwrite"])


@occlusion_group.command("attribution")
@_occlusion_options
@_cell_config_option
@click.option("--batch", type=int, default=8, show_default=True, help="Frames per Grad-CAM batch")
def occlusion_attribution(**kwargs):
    """Per-cell signed Grad-CAM of the true-class log-odds (GPU) -> {work_dir}/attribution/."""
    from remo_id.interpret.cell_occlusion import run_attribution
    cfg, ex = _occlusion_setup(kwargs)
    run_attribution(cfg, ex, kwargs["batch"], kwargs["overwrite"])


@occlusion_group.command("screen")
@_occlusion_options
@_cell_config_option
@click.option("--s1_frames", type=int, default=150, show_default=True,
              help="Stage 1: frames screened with random multi-cell masks")
@click.option("--n_masks", type=int, default=1000, show_default=True,
              help="Stage 1: random masks per frame")
@click.option("--p", type=float, default=0.01, show_default=True,
              help="Stage 1: probability that a mask removes a cell")
@click.option("--lam", type=float, default=0.1, show_default=True, help="Stage 1: ridge strength")
@click.option("--n_null", type=int, default=4, show_default=True,
              help="Stage 2: equal-area nulls per removed cell")
@click.option("--n_random_per_type", type=int, default=25, show_default=True,
              help="Stage 2: random cells per cell type")
@click.option("--border_px", type=int, default=48, show_default=True,
              help="Cells within this distance of the frame edge count as touching the border")
@click.option("--gpu_batch", type=int, default=48, show_default=True,
              help="Fixed encoder batch (baseline and occluded frames share it)")
@click.option("--workers", type=int, default=None,
              help="Fill threads (default: SLURM_CPUS_PER_TASK - 1, else 7)")
@click.option("--null_cell_type", default=None,
              help="Cell type of the equal-area nulls (default: the cell config's catch-all class)")
@click.option("--focus_cell_type", default=None,
              help="Cell type given extra random candidates (default: highest-weight class)")
@click.option("--random_cell_types", default=None,
              help="Comma-separated strata of the random sample, in draw order "
                   "(default: every cell-config class in config order)")
@click.option("--contrast", default=None,
              help="POS,NEG class indices of the contrast target logit[POS]-logit[NEG] "
                   "(default: last,first)")
def occlusion_screen(**kwargs):
    """Stage-1 random-subset and stage-2 exact single-cell removal (GPU) -> {work_dir}/screen/."""
    from remo_id.interpret.cell_occlusion import run_screen
    cfg, ex = _occlusion_setup(
        kwargs, s1_frames=kwargs["s1_frames"], n_masks=kwargs["n_masks"], p=kwargs["p"],
        lam=kwargs["lam"], n_null=kwargs["n_null"], n_random_per_type=kwargs["n_random_per_type"],
        border_px=kwargs["border_px"], gpu_batch=kwargs["gpu_batch"],
        workers=kwargs["workers"] or max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) - 1),
        null_cell_type=kwargs["null_cell_type"], focus_cell_type=kwargs["focus_cell_type"],
        random_cell_types=_parse_classes(kwargs["random_cell_types"]),
        contrast=tuple(int(c) for c in kwargs["contrast"].split(",")) if kwargs["contrast"] else None)
    run_screen(cfg, ex, kwargs["overwrite"])


@occlusion_group.command("report")
@click.option("--work_dir", required=True, type=click.Path(exists=True),
              help="Work dir of the screen")
@click.option("--output_dir", default=None, type=click.Path(),
              help="Report tables (default {work_dir}/report)")
@click.option("--class_names", default=None,
              help="Comma-separated display names for the `class` column, in class index "
                   "order (default: class-config names)")
def occlusion_report(**kwargs):
    """Stage-2 cell table and ranked top cells per class (CPU)."""
    from remo_id.interpret.cell_occlusion import run_report
    run_report(kwargs["work_dir"], kwargs["output_dir"] or os.path.join(kwargs["work_dir"], "report"),
               _parse_classes(kwargs["class_names"]))


# ---------------------------------------------------------------------------
# report command group
# ---------------------------------------------------------------------------


@cli.group("report")
def report_group():
    """Generate reports from saved predictions and analysis. No GPU required.

    \b
    Examples:
        remo-id report infer --checkpoint best.ckpt
        remo-id report performance --checkpoint best.ckpt
    """
    pass


@report_group.command("infer")
@click.option("--checkpoint", required=True, type=click.Path(exists=True),
              help="Path to model checkpoint (reads saved predictions)")
@click.option("--output", default=None, type=click.Path(),
              help="Output PDF path (default: {checkpoint_dir}/inference_report.pdf)")
def report_infer(**kwargs):
    """Generate inference PDF from saved predictions (no GPU needed).

    Reads predictions.npz and .metadata.json next to the checkpoint.
    Generates metrics box, strip plot, confusion matrix, and stats table.
    """
    import numpy as np
    from remo_id.inference.results import load_predictions

    checkpoint = kwargs["checkpoint"]
    try:
        df, meta = load_predictions(checkpoint)
    except FileNotFoundError:
        print(f"No saved predictions found for {checkpoint}")
        print("Run 'remo-id infer' first to generate predictions.")
        return

    task_type = meta.get("task_type", "binary") if meta else "binary"
    num_classes = meta.get("num_classes", 2) if meta else 2
    class_names = meta.get("class_names") if meta else None

    # Determine output path
    output = kwargs["output"]
    if not output:
        from pathlib import Path
        output = str(Path(checkpoint).parent / "inference_report.pdf")

    # Build arrays
    has_null = df["has_null_label"].to_numpy() if "has_null_label" in df.columns else np.zeros(len(df), dtype=bool)
    valid_df_filtered = df[~has_null].dropna(subset=["y_prob"])

    y_true = valid_df_filtered["y_true"].to_numpy(dtype=int)
    y_pred = valid_df_filtered["y_pred"].to_numpy(dtype=int)
    y_prob = valid_df_filtered["y_prob"].to_numpy(dtype=float)

    # Compute metrics
    from remo_id.analysis.metrics import (
        compute_binary_metrics, compute_multiclass_metrics,
    )

    if task_type == "binary":
        metrics = compute_binary_metrics(y_true, y_prob)
    else:
        metrics = compute_multiclass_metrics(y_true, y_pred, num_classes)

    # Generate PDF
    from reportlab.pdfgen import canvas as rl_canvas
    from remo_id.reports.pdf import add_png_page
    from remo_id.reports.pdf.summary import (
        build_summary_first_page, make_confusion_matrix_heatmap,
    )

    experiment_ids = df["experiment_id"].tolist()

    c = rl_canvas.Canvas(output)

    # Summary page
    build_summary_first_page(
        c=c,
        title="Inference Report",
        subtitle=f"Checkpoint: {os.path.basename(checkpoint)} | Task: {task_type}",
        metrics=metrics,
        experiment_ids=experiment_ids,
        y_true=df["y_true"].to_numpy(),
        y_prob=df["y_prob"].to_numpy(),
        has_null_labels=has_null,
        task_type=task_type,
        n_experiments=len(df),
        n_null=int(has_null.sum()),
        class_names=class_names,
        y_pred=df["y_pred"].to_numpy() if "y_pred" in df.columns else None,
    )

    # Confusion matrix page
    if task_type in ("binary", "multiclass") and class_names and len(valid_df_filtered) > 0:
        from sklearn.metrics import confusion_matrix
        n_cls = num_classes if task_type == "multiclass" else 2
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
        cm_png = make_confusion_matrix_heatmap(cm, class_names[:n_cls])
        add_png_page(c, "Confusion Matrix", cm_png)

    c.save()
    print(f"Inference report saved to {output}")


@report_group.command("performance")
@click.option("--checkpoint", required=True, type=click.Path(exists=True),
              help="Path to model checkpoint")
@click.option("--split", default="val", show_default=True,
              help="Which split to evaluate")
@click.option("--output", default=None, type=click.Path(),
              help="Output directory")
@click.option("--bootstrap_n", type=int, default=1000, show_default=True,
              help="Bootstrap iterations")
def report_performance(**kwargs):
    """Generate performance figures from saved predictions (no GPU needed).

    Reads predictions.npz saved next to the checkpoint.
    Generates ROC curves, PR curves, confusion matrix, calibration, metrics CSV.
    Equivalent to 'remo-id analysis performance' but requires saved predictions (no GPU fallback).
    """
    checkpoint = kwargs["checkpoint"]
    split = kwargs["split"]
    bootstrap_n = kwargs["bootstrap_n"]

    ctx = _build_context_from_predictions(checkpoint, split)
    if ctx is None:
        print(f"No saved predictions found for {checkpoint}")
        print("Run 'remo-id infer' or 'remo-id train' first.")
        return

    out = kwargs["output"] or resolve_output_dir(None, checkpoint, "performance")
    os.makedirs(out, exist_ok=True)

    # Generate figures (same logic as analysis performance)
    from remo_id.reports import figures as plots
    y_true, y_pred, y_probs = ctx["y_true"], ctx["y_pred"], ctx["y_probs"]
    class_names = ctx["class_names"]
    task_type = ctx.get("task_type", "multiclass")

    figs = [
        ("roc_curves",        plots.plot_roc_curves(y_true, y_probs, class_names, bootstrap_n=bootstrap_n, task_type=task_type)),
        ("pr_curves",         plots.plot_pr_curves(y_true, y_probs, class_names, bootstrap_n=bootstrap_n, task_type=task_type)),
        ("confusion_matrix",  plots.plot_confusion_matrix(y_true, y_pred, class_names)),
        ("calibration",       plots.plot_calibration(y_true, y_probs, class_names, task_type=task_type)),
        ("prob_distributions",plots.plot_prob_distributions(y_true, y_probs, class_names, task_type=task_type)),
    ]
    _save_figures(figs, out)

    metrics_df = plots.metrics_table_with_ci(y_true, y_probs, class_names, n=bootstrap_n, task_type=task_type)
    metrics_df.to_csv(os.path.join(out, "metrics.csv"), index=False)
    print(metrics_df.to_string(index=False))
    print(f"Performance report saved to {out}")


# ---------------------------------------------------------------------------
# workflow command group
# ---------------------------------------------------------------------------


@cli.group("workflow")
def workflow_group():
    """Cross-layer workflows that orchestrate training, inference, analysis, and reporting.

    \b
    These commands typically require GPU and run multi-stage pipelines.

    Examples:
        remo-id workflow robustness --checkpoint best.ckpt
        remo-id workflow cv --manifest manifest.csv --model resnet18-mil --n_folds 10
    """
    pass


@workflow_group.command("robustness")
@click.option("--checkpoint", required=True, type=click.Path(exists=True),
              help="Path to model checkpoint")
@click.option("--output", default=None, type=click.Path(),
              help="Output directory")
@click.option("--seeds", default="42,123,456,789,1011",
              help="Comma-separated seeds for multi-seed analysis")
@click.option("--bag_size", type=int, default=200, help="Images per bag")
@click.option("--split", default="val", help="Split to evaluate (train/val/all)")
def workflow_robustness(**kwargs):
    """Multi-seed robustness analysis: run inference at multiple seeds and measure variability."""
    from remo_id.workflows.robustness import run_robustness
    seeds = [int(s) for s in kwargs["seeds"].split(",")]
    run_robustness(
        checkpoint_path=kwargs["checkpoint"],
        output=kwargs["output"],
        seeds=seeds,
        bag_size=kwargs["bag_size"],
        split=kwargs["split"],
    )


@workflow_group.command("cv")
@_class_source_options
@click.option("--model", type=click.Choice(MODEL_CHOICES), default="resnet18-mil",
              help="Model architecture")
@click.option("--n_folds", type=int, default=10, help="Number of CV folds")
@click.option("--n_repeats", type=int, default=1, help="Number of CV repeats")
@click.option("--save_dir", default=None,
              help="Base directory for fold checkpoints (default: $REMO_ID_SAVE_DIR or ./checkpoints)")
@click.option("--seed", type=int, default=403, help="Random seed")
@click.option("--batch_size", type=int, default=200, help="Batch size")
@click.option("--max_epochs", type=int, default=100, help="Maximum training epochs")
@click.option("--lr", type=float, default=1e-4, help="Learning rate")
@click.option("--n_experiments", type=int, default=8, help="Experiment bags per batch")
def workflow_cv(**kwargs):
    """K-fold cross-validation: train, infer, pool predictions, analyze."""
    from remo_id.workflows.cv import run_cross_validation
    from remo_id.training.config import default_save_dir
    from remo_id.training.cross_validation import CVConfig
    from remo_id.data import RemoIDDataModule
    from remo_id.data.class_config import class_config_from_dict, parse_class_config
    from remo_id.models import get_augmentation_backbone

    if kwargs["manifest"] and kwargs["class_config"]:
        raise click.UsageError("--manifest and --class_config are mutually exclusive.")
    if kwargs["manifest"]:
        cfg = class_config_from_dict(_manifest_class_config(kwargs))
    elif kwargs["class_config"]:
        cfg = parse_class_config(kwargs["class_config"])
    else:
        raise click.UsageError("Provide --class_config or --manifest.")
    default_class_idx = cfg.class_names.index(cfg.default_class) if cfg.default_class in cfg.class_names else None

    model_name = kwargs["model"]
    backbone_name = get_augmentation_backbone(model_name)

    data_module = RemoIDDataModule(
        database_column="class_label",
        batch_size=kwargs["batch_size"],
        backbone_name=backbone_name,
        task_type=cfg.task_type,
        num_classes=cfg.num_classes,
        class_definitions=cfg.class_definitions,
        exclude_filters=cfg.exclude_filters,
        include_filters=cfg.include_filters,
        default_class_idx=default_class_idx,
        negative_study_ids=cfg.negative_study_ids,
        skip_null_labels=True,
    )

    cv_config = CVConfig(
        n_folds=kwargs["n_folds"],
        n_repeats=kwargs["n_repeats"],
        seed=kwargs["seed"],
    )

    train_kwargs = {
        "model": model_name,
        "task_type": cfg.task_type,
        "num_classes": cfg.num_classes,
        "batch_size": kwargs["batch_size"],
        "max_epochs": kwargs["max_epochs"],
        "lr": kwargs["lr"],
        "n_experiments": kwargs["n_experiments"],
        "database_column": "class_label",
        "class_definitions": cfg.class_definitions,
        "class_config_raw": cfg.raw_config,
        "manifest": os.path.abspath(kwargs["manifest"]) if kwargs["manifest"] else None,
        "label_column": kwargs["label_column"],
        "classes": _parse_classes(kwargs["classes"]),
    }

    run_cross_validation(
        data_module=data_module,
        cv_config=cv_config,
        save_dir=kwargs["save_dir"] or default_save_dir(),
        train_kwargs=train_kwargs,
    )


@workflow_group.command("loo-report")
@click.option("--cv_dir", required=True, type=click.Path(exists=True),
              help="Path to LOO CV directory (contains repeat*_fold* subdirs)")
@_class_source_options
@click.option("--output", default=None, type=click.Path(),
              help="Output directory (default: {cv_dir}/analysis/loo_report/)")
@click.option("--seed", type=int, default=403, help="Random seed")
@click.option("--bootstrap_n", type=int, default=1000, help="Bootstrap iterations for CIs")
@click.option("--n_bags", type=int, default=5, help="Number of bags to average per experiment")
@click.option("--pairwise", multiple=True, help="Pairwise pairs as 'pos:neg' (e.g. 'disease:control')")
@click.option("--min_sensitivity", type=float, default=0.95, help="Default minimum sensitivity for all classes")
@click.option("--sensitivity_targets", type=str, default=None,
              help="Per-class sensitivity targets, e.g. 'disease:0.95,healthy:0.90'")
@click.option("--target_class", type=str, default=None,
              help="Class name highlighted in the dual operating-point ROC figure (skipped without it)")
@click.option("--bag_sizes", default=None, help="Comma-separated sizes for bag size sweep")
@click.option("--epoch", type=int, default=None, help="Force specific epoch (skip sweep). Use -1 for last.ckpt, -2 for best.ckpt")
@click.option("--bag_size", type=int, default=None, help="Override inference bag size (default: training bag size)")
@click.option("--error_consistency/--no-error_consistency", default=False, help="Run error consistency analysis (GPU-intensive, default off)")
@click.option("--no_regenerate", is_flag=True, default=False,
              help="Refuse to re-encode/predict on GPU; only read existing predictions/epoch_*.npz (guards against regenerating at the wrong bag size)")
def workflow_loo_report(**kwargs):
    """LOO CV comprehensive report: epoch sweep, performance tables, figures, error analysis."""
    from remo_id.workflows.loo_report import run_loo_report
    run_loo_report(
        cv_dir=kwargs["cv_dir"],
        class_config=_resolve_class_config_path(kwargs, kwargs["cv_dir"]),
        output=kwargs["output"],
        seed=kwargs["seed"],
        bootstrap_n=kwargs["bootstrap_n"],
        n_bags=kwargs["n_bags"],
        pairwise_pairs=list(kwargs["pairwise"]) if kwargs["pairwise"] else None,
        min_sensitivity=kwargs["min_sensitivity"],
        sensitivity_targets_str=kwargs["sensitivity_targets"],
        bag_sizes_str=kwargs["bag_sizes"],
        forced_epoch=kwargs["epoch"],
        bag_size_override=kwargs["bag_size"],
        error_consistency=kwargs["error_consistency"],
        no_regenerate=kwargs["no_regenerate"],
        target_class=kwargs["target_class"],
    )


@workflow_group.command("loo-predict")
@click.option("--cv_dir", required=True, type=click.Path(exists=True),
              help="Path to LOO CV directory")
@_class_source_options
@click.option("--epoch", type=int, required=True, help="Epoch number to predict")
@click.option("--seed", type=int, default=403, help="Random seed")
@click.option("--n_bags", type=int, default=5, help="Number of bags per experiment")
@click.option("--bag_size", type=int, default=None, help="Override inference bag size")
@click.option("--background_value", type=int, default=None, help="Background pixel value")
@click.option("--output_dir", type=click.Path(), default=None, help="Custom output directory for predictions")
@click.option("--scan_size", type=int, default=None, help="Two-stage: scan this many images for attention selection")
@click.option("--top_k", type=int, default=None, help="Two-stage: select top-K instances by attention from scan")
@click.option("--model_version", type=str, default=None, help="Embedding-MIL: embedding-store version tag (required for embedding-mil LOO)")
@click.option("--cell_config", type=str, default=None, help="Embedding-MIL: cell_config YAML used at training (omit if trained without one)")
@click.option("--pool_cap", type=int, default=None, help="Embedding-MIL: --pool_cap used at training")
@click.option("--load_workers", type=int, default=None, help="Embedding-MIL: store-load thread count (bounds RAM; default min(SLURM_CPUS, 8); lower it if loading runs out of memory)")
@click.option("--all_experiments", is_flag=True, default=False,
              help="Score EVERY experiment with every fold, not just the fold's "
                   "held-out one, and record an in_train flag. Diagnostic only "
                   "(train-set rows are memorization, not performance) and n_folds x "
                   "the work — shard it with --fold_shard/--fold_nshards.")
@click.option("--fold_shard", type=int, default=None,
              help="Process folds [shard::nshards] only; output npz gets a _shard{N} suffix")
@click.option("--fold_nshards", type=int, default=None, help="Total number of fold shards")
def workflow_loo_predict(**kwargs):
    """Predict all folds at a single epoch (one SLURM array task)."""
    from remo_id.workflows.loo_predict import predict_epoch
    predict_epoch(
        cv_dir=kwargs["cv_dir"],
        epoch=kwargs["epoch"],
        all_experiments=kwargs["all_experiments"],
        fold_shard=kwargs["fold_shard"],
        fold_nshards=kwargs["fold_nshards"],
        class_config=_resolve_class_config_path(kwargs, kwargs["cv_dir"]),
        seed=kwargs["seed"],
        n_bags=kwargs["n_bags"],
        bag_size_override=kwargs["bag_size"],
        background_value=kwargs["background_value"],
        output_dir=kwargs["output_dir"],
        scan_size=kwargs["scan_size"],
        top_k=kwargs["top_k"],
        model_version=kwargs["model_version"],
        cell_config=kwargs["cell_config"],
        pool_cap=kwargs["pool_cap"],
        load_workers=kwargs["load_workers"],
    )


@workflow_group.command("loo-aggregate")
@click.option("--cv_dir", required=True, type=click.Path(exists=True),
              help="Path to LOO CV directory")
@_class_source_options
@click.option("--epoch", type=int, required=True, help="Epoch number to aggregate")
@click.option("--seed", type=int, default=403, help="Random seed")
@click.option("--n_bags", type=int, default=5, help="Number of bags per experiment")
@click.option("--bag_size", type=int, default=500, help="Bag size to sample from cached embeddings")
@click.option("--output_dir", type=click.Path(), default=None, help="Custom output dir for predictions npz")
@click.option("--embeddings_dir", type=click.Path(), default=None,
              help="Cached embeddings dir (default: {cv_dir}/embeddings/epoch_{N})")
def workflow_loo_aggregate(**kwargs):
    """Pool predictions at one epoch by sampling CACHED embeddings -> aggregator head (no encode)."""
    from remo_id.workflows.loo_predict import aggregate_epoch_from_embeddings
    aggregate_epoch_from_embeddings(
        cv_dir=kwargs["cv_dir"],
        epoch=kwargs["epoch"],
        class_config=_resolve_class_config_path(kwargs, kwargs["cv_dir"]),
        seed=kwargs["seed"],
        n_bags=kwargs["n_bags"],
        bag_size=kwargs["bag_size"],
        output_dir=kwargs["output_dir"],
        embeddings_dir=kwargs["embeddings_dir"],
    )


@workflow_group.command("loo-predict-submit")
@click.option("--cv_dir", required=True, type=click.Path(exists=True),
              help="Path to LOO CV directory")
@_class_source_options
@click.option("--epochs", type=str, default=None,
              help="Comma-separated epochs to predict (e.g. '50'; default: every epoch checkpointed in all folds, plus last/best)")
@click.option("--n_gpus", type=int, default=None, help="Max concurrent GPU jobs (default: one per epoch)")
@click.option("--partition", default="gpu", show_default=True, help="SLURM partition")
@click.option("--seed", type=int, default=403, help="Random seed")
@click.option("--n_bags", type=int, default=5, help="Number of bags per experiment")
@click.option("--bag_size", type=int, default=None, help="Override inference bag size")
@click.option("--background_value", type=int, default=None, help="Background pixel value")
@click.option("--model_version", type=str, default=None, help="Embedding-MIL: embedding-store version tag (required for embedding-mil LOO)")
@click.option("--cell_config", type=str, default=None, help="Embedding-MIL: cell_config YAML used at training (omit if trained without one)")
def workflow_loo_predict_submit(**kwargs):
    """Submit SLURM array job: one GPU per epoch, all folds predicted in parallel."""
    from remo_id.workflows.loo_predict import submit_predict_array
    submit_predict_array(
        cv_dir=kwargs["cv_dir"],
        class_config=_resolve_class_config_path(kwargs, kwargs["cv_dir"]),
        n_gpus=kwargs["n_gpus"],
        seed=kwargs["seed"],
        n_bags=kwargs["n_bags"],
        bag_size=kwargs["bag_size"],
        background_value=kwargs["background_value"],
        model_version=kwargs["model_version"],
        cell_config=kwargs["cell_config"],
        epochs=[int(e) for e in kwargs["epochs"].split(",") if e.strip()] if kwargs["epochs"] else None,
        partition=kwargs["partition"],
    )


@workflow_group.command("loo-per-image")
@click.option("--cv_dir", required=True, type=click.Path(exists=True),
              help="Path to LOO CV directory")
@_class_source_options
@click.option("--epoch", type=int, required=True, help="Epoch number")
@click.option("--fold_idx", type=int, required=True, help="Fold index to process")
@click.option("--seed", type=int, default=403, help="Random seed")
@click.option("--save_embeddings/--no-save_embeddings", default=True,
              help="Also save per-image embeddings npz (default True)")
@click.option("--all_experiments/--no-all_experiments", default=False,
              help="Run this fold's model over ALL experiments (not just its held-out val set)")
@click.option("--output_dir", type=click.Path(), default=None,
              help="Output dir for per-image CSV/npz (default: {cv_dir}/per_image_predictions)")
@click.option("--exp_shard", type=int, default=0, help="This task's experiment shard id (for array parallelism)")
@click.option("--exp_nshards", type=int, default=1, help="Total experiment shards; each task does exp_idx %% exp_nshards == exp_shard")
@click.option("--encode_chunk_size", type=int, default=256, help="GPU batch size for per-image encoding (inference only; does not change embeddings)")
def workflow_loo_per_image(**kwargs):
    """Predict every image individually for one fold (held-out, or --all_experiments)."""
    from remo_id.workflows.loo_predict import predict_per_image_fold
    predict_per_image_fold(
        cv_dir=kwargs["cv_dir"],
        fold_idx=kwargs["fold_idx"],
        epoch=kwargs["epoch"],
        class_config=_resolve_class_config_path(kwargs, kwargs["cv_dir"]),
        seed=kwargs["seed"],
        save_embeddings=kwargs["save_embeddings"],
        all_experiments=kwargs["all_experiments"],
        output_dir=kwargs["output_dir"],
        exp_shard=kwargs["exp_shard"],
        exp_nshards=kwargs["exp_nshards"],
        encode_chunk_size=kwargs["encode_chunk_size"],
    )


@workflow_group.command("loo-bag-variance")
@click.option("--cv_dir", required=True, type=click.Path(exists=True),
              help="Path to LOO CV directory")
@_class_source_options
@click.option("--epoch", type=int, required=True, help="Epoch number")
@click.option("--fold_idx", type=int, required=True, help="Fold index to process")
@click.option("--seed", type=int, default=403, help="Random seed")
@click.option("--n_shuffles", type=int, default=5, help="Number of shuffle repetitions")
def workflow_loo_bag_variance(**kwargs):
    """Encode all images for one fold's held-out experiments, sweep bag sizes."""
    from remo_id.workflows.loo_predict import predict_bag_variance_fold
    predict_bag_variance_fold(
        cv_dir=kwargs["cv_dir"],
        fold_idx=kwargs["fold_idx"],
        epoch=kwargs["epoch"],
        class_config=_resolve_class_config_path(kwargs, kwargs["cv_dir"]),
        seed=kwargs["seed"],
        n_shuffles=kwargs["n_shuffles"],
    )


@workflow_group.command("loo-position-sweep")
@click.option("--cv_dir", required=True, type=click.Path(exists=True),
              help="Path to LOO CV directory")
@_class_source_options
@click.option("--epoch", type=int, required=True, help="Epoch number")
@click.option("--fold_idx", type=int, default=None,
              help="Fold index to process (omit to submit a SLURM array over all folds)")
@click.option("--seed", type=int, default=403, help="Random seed")
@click.option("--bag_size", type=int, default=500, help="Bag size drawn from each window")
@click.option("--n_draws", type=int, default=10, help="Random bag draws per window")
@click.option("--window_step", type=int, default=500, help="Window size step")
@click.option("--window_max", type=int, default=20000, help="Largest window size")
@click.option("--n_gpus", type=int, default=None, help="Max parallel GPUs (submit mode; default: one per fold)")
@click.option("--partition", default="gpu", show_default=True, help="SLURM partition (submit mode)")
def workflow_loo_position_sweep(**kwargs):
    """Sweep image POSITION (first-N / last-N) for LOO held-out experiments.

    With --fold_idx, processes one fold. Without it, submits a SLURM array
    (one task per fold). Aggregate/plot afterward with loo-position-report.
    """
    from remo_id.workflows.loo_predict import (
        predict_position_sweep_fold, submit_position_sweep_array,
    )
    common = dict(
        cv_dir=kwargs["cv_dir"], class_config=_resolve_class_config_path(kwargs, kwargs["cv_dir"]),
        epoch=kwargs["epoch"], seed=kwargs["seed"], bag_size=kwargs["bag_size"],
        n_draws=kwargs["n_draws"], window_step=kwargs["window_step"],
        window_max=kwargs["window_max"],
    )
    if kwargs["fold_idx"] is None:
        submit_position_sweep_array(n_gpus=kwargs["n_gpus"], partition=kwargs["partition"], **common)
    else:
        predict_position_sweep_fold(fold_idx=kwargs["fold_idx"], **common)


@workflow_group.command("loo-position-report")
@click.option("--cv_dir", required=True, type=click.Path(exists=True),
              help="Path to LOO CV directory")
@click.option("--bag_size", type=int, default=500, help="Bag size of the sweep to report")
@click.option("--output", default=None, type=click.Path(), help="Output directory")
@click.option("--per_class", is_flag=True, default=False,
              help="Also emit the per-class breakdown (AUC/agreement/variability per true class)")
@click.option("--exclude_class", multiple=True,
              help="Class name(s) to drop from the pooled report (repeatable, e.g. --exclude_class healthy)")
def workflow_loo_position_report(**kwargs):
    """Pool position-sweep folds, score windows, and plot first-N vs last-N."""
    from remo_id.workflows.loo_predict import (
        make_position_sweep_report, make_position_sweep_per_class_report,
    )
    exclude = list(kwargs["exclude_class"]) or None
    make_position_sweep_report(
        cv_dir=kwargs["cv_dir"], bag_size=kwargs["bag_size"], output=kwargs["output"],
        exclude_classes=exclude,
    )
    if kwargs["per_class"]:
        make_position_sweep_per_class_report(
            cv_dir=kwargs["cv_dir"], bag_size=kwargs["bag_size"], output=kwargs["output"],
        )


@workflow_group.command("loo-bag-sweep")
@click.option("--cv_dir", required=True, type=click.Path(exists=True),
              help="Path to LOO CV directory")
@_class_source_options
@click.option("--epoch", type=int, required=True, help="Epoch to evaluate")
@click.option("--bag_sizes", default="1,5,25,50,100,200,500",
              help="Comma-separated bag sizes to sweep")
@click.option("--seeds", default="42,123,456,789,1011",
              help="Comma-separated seeds for repeats")
@click.option("--n_gpus", type=int, default=None, help="Max parallel GPUs (default: one per task)")
@click.option("--partition", default="gpu", show_default=True, help="SLURM partition")
def workflow_loo_bag_sweep(**kwargs):
    """Submit parallel bag size sweep: one GPU per (bag_size, seed) combo."""
    from remo_id.workflows.loo_predict import submit_bag_size_sweep
    bag_sizes = [int(x) for x in kwargs["bag_sizes"].split(",")]
    seeds = [int(x) for x in kwargs["seeds"].split(",")]
    submit_bag_size_sweep(
        cv_dir=kwargs["cv_dir"],
        class_config=_resolve_class_config_path(kwargs, kwargs["cv_dir"]),
        epoch=kwargs["epoch"],
        bag_sizes=bag_sizes,
        seeds=seeds,
        n_gpus=kwargs["n_gpus"],
        partition=kwargs["partition"],
    )


@workflow_group.command("loo-bag-sweep-report")
@click.option("--cv_dir", required=True, type=click.Path(exists=True),
              help="Path to LOO CV directory")
@_class_source_options
@click.option("--output", default=None, type=click.Path(),
              help="Output directory (default: {cv_dir}/analysis/bag_size_sweep/)")
def workflow_loo_bag_sweep_report(**kwargs):
    """Aggregate bag size sweep results and generate figures."""
    from remo_id.workflows.loo_predict import aggregate_bag_size_sweep
    from remo_id.inference.context import load_class_config
    from remo_id.reports import figures as plots
    import matplotlib.pyplot as plt
    import numpy as np

    _, class_names, *_ = load_class_config(_resolve_class_config_path(kwargs, kwargs["cv_dir"]))

    result = aggregate_bag_size_sweep(kwargs["cv_dir"], class_names)
    if result is None:
        print("No bag size sweep results found. Run 'loo-bag-sweep' first.")
        return

    out = kwargs["output"] or os.path.join(kwargs["cv_dir"], "analysis", "bag_size_sweep")
    os.makedirs(out, exist_ok=True)

    bag_sizes = result["bag_sizes"]
    aucs = result["aucs"]
    f1s = result["f1s"]

    # Save CSV
    import pandas as pd
    rows = []
    for si, seed in enumerate(result["seeds"]):
        for bi, bs in enumerate(bag_sizes):
            rows.append({"bag_size": bs, "seed": seed, "macro_auc": aucs[si, bi], "macro_f1": f1s[si, bi]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "bag_size_sweep.csv"), index=False)

    # Summary table
    print(f"\nBag Size Sweep Results ({len(result['seeds'])} seeds):")
    print(f"{'Bag Size':>10} {'AUC mean':>10} {'AUC std':>10} {'F1 mean':>10} {'F1 std':>10}")
    for bi, bs in enumerate(bag_sizes):
        auc_vals = aucs[:, bi]
        f1_vals = f1s[:, bi]
        print(f"{bs:>10} {np.nanmean(auc_vals):>10.4f} {np.nanstd(auc_vals):>10.4f} {np.nanmean(f1_vals):>10.4f} {np.nanstd(f1_vals):>10.4f}")

    # Plot
    fig = plots.plot_bag_size_sweep(bag_sizes, aucs.tolist())
    fig.savefig(os.path.join(out, "bag_size_sweep.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved results to {out}")


@workflow_group.command("loo-topk-report")
@click.option("--cv_dir", required=True, type=click.Path(exists=True),
              help="Path to LOO CV directory")
@_class_source_options
@click.option("--k_values", default="1,3,5,10,25,50",
              help="Comma-separated K values for top-K selection")
@click.option("--output", default=None, type=click.Path(),
              help="Output directory")
def workflow_loo_topk_report(**kwargs):
    """Aggregate top-K confident single-image predictions."""
    from remo_id.workflows.loo_predict import aggregate_topk_experiment
    from remo_id.inference.context import load_class_config
    import pandas as pd

    _, class_names, *_ = load_class_config(_resolve_class_config_path(kwargs, kwargs["cv_dir"]))
    k_values = [int(x) for x in kwargs["k_values"].split(",")]

    result = aggregate_topk_experiment(kwargs["cv_dir"], class_names, k_values)
    if result is None:
        print("No top-K experiment results found. Run the prediction jobs first.")
        return

    out = kwargs["output"] or os.path.join(kwargs["cv_dir"], "analysis", "topk_experiment")
    os.makedirs(out, exist_ok=True)

    k_values = result["k_values"]
    print(f"\nTop-K Selection Results ({result['n_seeds']} seeds, {result['n_experiments']} experiments):")
    print(f"{'K':>5} {'top AUC':>10} {'random AUC':>10} {'bottom AUC':>10} {'top F1':>10} {'random F1':>10} {'bottom F1':>10}")
    rows = []
    for ki, k in enumerate(k_values):
        top = result["strategies"]["top"]
        rand = result["strategies"]["random"]
        bot = result["strategies"]["bottom"]
        print(f"{k:>5} {top['aucs'][ki]:>10.4f} {rand['aucs'][ki]:>10.4f} {bot['aucs'][ki]:>10.4f} {top['f1s'][ki]:>10.4f} {rand['f1s'][ki]:>10.4f} {bot['f1s'][ki]:>10.4f}")
        rows.append({"k": k,
                      "top_auc": top["aucs"][ki], "random_auc": rand["aucs"][ki], "bottom_auc": bot["aucs"][ki],
                      "top_f1": top["f1s"][ki], "random_f1": rand["f1s"][ki], "bottom_f1": bot["f1s"][ki]})

    pd.DataFrame(rows).to_csv(os.path.join(out, "topk_results.csv"), index=False)
    print(f"\nSaved to {out}")


@workflow_group.command("loo-two-stage-report")
@click.option("--cv_dir", required=True, type=click.Path(exists=True),
              help="Path to LOO CV directory")
@_class_source_options
@click.option("--output", default=None, type=click.Path(),
              help="Output directory")
def workflow_loo_two_stage_report(**kwargs):
    """Aggregate two-stage hard-attention experiment results."""
    from remo_id.workflows.loo_predict import aggregate_two_stage_experiment
    from remo_id.inference.context import load_class_config
    import pandas as pd

    _, class_names, *_ = load_class_config(_resolve_class_config_path(kwargs, kwargs["cv_dir"]))

    result = aggregate_two_stage_experiment(kwargs["cv_dir"], class_names)
    if result is None:
        print("No two-stage experiment results found.")
        return

    out = kwargs["output"] or os.path.join(kwargs["cv_dir"], "analysis", "two_stage_experiment")
    os.makedirs(out, exist_ok=True)

    print(f"\nTwo-Stage Hard-Attention Results ({len(result['seeds'])} seeds):")
    print(f"{'scan':>6} {'top_k':>6} {'AUC mean':>10} {'AUC std':>10} {'F1 mean':>10} {'F1 std':>10}")
    for row in result["rows"]:
        print(f"{row['scan_size']:>6} {row['top_k']:>6} {row['auc_mean']:>10.4f} {row['auc_std']:>10.4f} {row['f1_mean']:>10.4f} {row['f1_std']:>10.4f}")

    pd.DataFrame(result["rows"]).to_csv(os.path.join(out, "two_stage_results.csv"), index=False)
    print(f"\nSaved to {out}")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
