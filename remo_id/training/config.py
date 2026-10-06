from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any


def default_save_dir() -> str:
    """Checkpoint root: ``$REMO_ID_SAVE_DIR`` or ``./checkpoints``."""
    return os.environ.get("REMO_ID_SAVE_DIR", "./checkpoints")


@dataclass
class TrainingConfig:
    # Model
    model_name: str = "resnet18"
    pretrained: bool = True
    finetune_layers: str = "layer3,layer4"
    dino_weights: str = None       # DINOv3 checkpoint path (dino-* models; else $REMO_ID_DINO_WEIGHTS)

    # Task
    task_type: str = "binary"
    num_classes: int = 2

    # Data metadata (for wandb logging + checkpoint metadata)
    database_column: str = None
    database_label: Any = None
    study_id: str = None
    negative_study_ids: list = field(default_factory=list)
    negative_experiment_ids: list = field(default_factory=list)
    skip_null_labels: bool = True
    class_definitions: list = None
    class_config_raw: dict = None
    val_experiment_ids: list = field(default_factory=list)
    manifest: str = None           # CSV manifest (see remo_id.data.manifest)
    label_column: str = "label"
    classes: list = None

    # Batching
    batch_size: int = 200
    n_experiments: int = 8
    num_batches: int = 100
    cross_experiment_sampling: bool = False
    sample_aware_sampling: bool = False
    sample_aware_mode: str = "single"
    max_experiments_per_bag: int = None
    majority_min_pct: float = 1.0
    bag_label_mode: str = "hard"
    attn_entropy_weight: float = 0.0

    # Optimizer
    lr: float = 1e-4
    lr_schedule: str = "cosine"
    warmup_epochs: int = 5
    max_epochs: int = 100
    min_epochs: int = None
    weight_decay: float = 0.01

    # MIL architecture
    attn_dim: int = 128
    dropout: float = 0.3
    num_heads: int = 1
    projection_dim: int = None
    pooling: str = "attention"
    classifier_hidden: int = None
    top_k: int = None
    transformer_layers: int = 2
    encode_chunk_size: int = None
    patch_pool: bool = False

    # Embedding-MIL (model_name == "embedding-mil"): consume precomputed cell embeddings
    model_version: str = None      # embedding-store version tag (required for embedding-mil)
    embedding_dim: int = None      # auto-resolved from the store if None
    cells_per_bag: int = None      # cells sampled per bag; defaults to batch_size // n_experiments
    cell_config: str = None        # path to cell_config YAML (class gates + per-class sampling weights + pool_cap)
    load_workers: int = None       # thread pool size for the parallel store load
    pool_cap: int = None           # per-class distinct-cell cap (overrides cell_config; default 200k)

    # Trainer
    save_dir: str = field(default_factory=default_save_dir)
    project: str = "remo-id"
    accelerator: str = "auto"
    devices: int = 1
    seed: int = 403
    log_every_n_steps: int = 50
    early_stopping_patience: int = 15
    accumulate_grad_batches: int = 1
    precision: str = "auto"
    compile: bool = False
    deterministic: bool = True
    save_every_n_epochs: int = None
    save_epochs: str = None

    # Data filtering
    min_image_index: int = 0

    # Sanity-check controls
    permute_labels: bool = False
    permute_labels_seed: int = 404

    # Augmentation
    crop_size: int = None
    background_correction: str = 'none'
    background_value: int = None
    intensity_jitter: float = None
    noise_std: float = None
    blur_sigma: float = None
    gradient_blur_sigma: float = None
    erase_num: str = None
    erase_scale: str = None

    # Post-training
    run_infer: bool = True

    @property
    def n_pos(self) -> int:
        return self.n_experiments // 2

    @property
    def samples_per_experiment(self) -> int:
        return self.batch_size // self.n_experiments

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        """Create from dict, ignoring unknown keys."""
        import dataclasses
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})
