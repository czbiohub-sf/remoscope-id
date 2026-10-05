"""Sample visualization utilities for inspecting the data pipeline."""

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from remo_id.data.class_config import parse_class_config
from remo_id.data.datamodule import RemoIDDataModule, RemoIDDataLoader
from remo_id.data.zarr_io import get_zarr_file
from remo_id.models import get_augmentation_backbone
from remo_id.reports.style import apply

apply()


def _show_image(ax, img_tensor):
    """Display a [C, H, W] tensor on a matplotlib axis."""
    ax.axis('off')
    if img_tensor.shape[0] == 1:
        ax.imshow(img_tensor.squeeze(0).cpu().numpy(), cmap='gray')
    else:
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img_np)


def _load_p99(exp_path: str, cache: dict) -> torch.Tensor | None:
    """Load p99 background for an experiment path, with caching."""
    if exp_path in cache:
        return cache[exp_path]

    basename = os.path.basename(exp_path.rstrip('/'))
    p99_path = os.path.join(exp_path, f"{basename}background_p99.png")

    try:
        p99_np = np.array(Image.open(p99_path))  # uint8 [H, W]
        p99_tensor = torch.from_numpy(p99_np).float().unsqueeze(0)  # float32 [1, H, W]
        cache[exp_path] = p99_tensor
        return p99_tensor
    except FileNotFoundError:
        cache[exp_path] = None
        return None


def _apply_flat_field(img: torch.Tensor, exp_path: str, mode: str,
                      bg_value: float, p99_cache: dict, p99_pool: list,
                      train: bool) -> torch.Tensor:
    """Apply flat-field correction for sample visualization.

    Mirrors RemoIDDataLoader._apply_flat_field + _select_p99 logic.
    """
    p99 = None
    if train and mode != 'none':
        if mode == 'own':
            p99 = _load_p99(exp_path, p99_cache)
        elif mode == 'random':
            if p99_pool:
                p99 = random.choice(p99_pool)
            else:
                p99 = _load_p99(exp_path, p99_cache)
        elif mode == 'blend':
            roll = random.random()
            if roll < 1/3:
                p99 = _load_p99(exp_path, p99_cache)
            elif roll < 2/3:
                if p99_pool:
                    p99 = random.choice(p99_pool)
                else:
                    p99 = _load_p99(exp_path, p99_cache)
            # else: scalar (p99 stays None)
        elif mode == 'swap':
            own_p99 = _load_p99(exp_path, p99_cache)
            if own_p99 is not None and p99_pool:
                random_p99 = random.choice(p99_pool)
                return ((img.float() - own_p99 + random_p99) / bg_value).clamp(min=0)
            elif own_p99 is not None:
                return ((img.float() - own_p99 + bg_value) / bg_value).clamp(min=0)
            return img.float() / bg_value

    if p99 is not None:
        return ((img.float() - p99 + bg_value) / bg_value).clamp(min=0)
    else:
        return img.float() / bg_value


def run_sample_visualization(
    class_config: str,
    model: str = "resnet18",
    num_images: int = 3,
    num_augments: int = 5,
    output_dir: str = "./results/augment_samples",
    seed: int = 403,
    background_correction: str = 'none',
    background_value: int = None,
    intensity_jitter: float = None,
    noise_std: float = None,
    blur_sigma: float = None,
    gradient_blur_sigma: float = None,
):
    """
    Visualize augmented samples from each class defined in a class config.

    Produces a grid where each row is one image, showing the original plus
    multiple augmented versions. Rows are ordered by study_id, then class name.

    Args:
        class_config: Path to YAML class config file
        model: Model architecture (determines augmentation pipeline)
        num_images: Number of images to sample per class
        num_augments: Number of augmented versions per image
        output_dir: Directory to save output figures
        seed: Random seed for reproducibility
        background_correction: Flat-field correction mode (none/own/random/blend)
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Parse class config
    cfg = parse_class_config(class_config)
    backbone = get_augmentation_backbone(model)

    print(f"Setting up data module for {model}...")
    print(f"  Classes: {cfg.class_names}")
    print(f"  Task type: {cfg.task_type}")
    print(f"  Background correction: {background_correction}")

    # Compute default_class_idx
    default_class_idx = None
    if cfg.default_class and cfg.task_type == 'multiclass':
        default_class_idx = cfg.class_names.index(cfg.default_class)

    # Create data module to get experiment metadata
    data_module = RemoIDDataModule(
        database_column='class_idx',
        database_label=None,
        backbone_name=backbone,
        batch_size=200,
        task_type=cfg.task_type,
        num_classes=cfg.num_classes,
        class_definitions=cfg.class_definitions,
        skip_null_labels=True,
        exclude_filters=cfg.exclude_filters,
        include_filters=cfg.include_filters,
        default_class_idx=default_class_idx,
        negative_study_ids=cfg.negative_study_ids or [],
        negative_experiment_ids=[],
        train=False,
    )

    # Get augmentation pipelines (normalization is applied manually below)
    augment_kwargs = dict(
        intensity_jitter=intensity_jitter, noise_std=noise_std,
        blur_sigma=blur_sigma, gradient_blur_sigma=gradient_blur_sigma,
    )
    train_augment = RemoIDDataLoader.prep_augments(backbone, train=True, **augment_kwargs)
    eval_augment = RemoIDDataLoader.prep_augments(backbone, train=False, **augment_kwargs)

    # Scalar divisor for normalization (matches _apply_flat_field scalar mode)
    bg_value = float(background_value if background_value is not None else 225)

    data = data_module.data

    # Build class index -> name mapping
    if len(cfg.class_names) > 1:
        idx_to_name = dict(enumerate(cfg.class_names))
    else:
        idx_to_name = {0: "negative", 1: cfg.class_names[0]}

    # Sample images per class
    rows = []  # (class_name, study_id, zarr_path, image_index)
    rng = random.Random(seed)

    for class_idx in sorted(data['class_idx'].unique()):
        class_data = data[data['class_idx'] == class_idx]
        class_name = idx_to_name.get(int(class_idx), str(class_idx))

        n_sample = min(num_images, len(class_data))
        sampled = class_data.sample(n=n_sample, random_state=seed)

        for _, row in sampled.iterrows():
            img_idx = rng.randint(0, max(int(row['num_images']) - 1, 0))
            rows.append((class_name, row['study_id'], row['path'], img_idx))

    # Sort by study_id, then class_name
    rows.sort(key=lambda r: (r[1], r[0]))

    # Preload p99 pool if needed for random/blend modes
    p99_cache = {}
    p99_pool = []
    if background_correction in ('random', 'blend', 'swap'):
        unique_paths = set(r[2] for r in rows)
        for exp_path in unique_paths:
            p99 = _load_p99(exp_path, p99_cache)
            if p99 is not None:
                p99_pool.append(p99)
        n_found = sum(1 for v in p99_cache.values() if v is not None)
        print(f"  Loaded {n_found}/{len(unique_paths)} p99 backgrounds for flat-field correction")

    # Build figure
    n_rows = len(rows)
    n_cols = 1 + 1 + num_augments  # label + original + augments

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2 * n_cols, 2.2 * n_rows),
        gridspec_kw={'width_ratios': [1.2] + [1] * (n_cols - 1)},
    )

    if n_rows == 1:
        axes = [list(axes)]
    else:
        axes = [list(row) for row in axes]

    mode_label = f"bg={background_correction}"
    fig.suptitle(f'Sample Augmentations — {model} ({mode_label})', fontsize=14)

    zarr_cache = {}

    for row_idx, (class_name, study_id, zarr_path, img_idx) in enumerate(rows):
        # Label cell
        label_ax = axes[row_idx][0]
        label_ax.axis('off')
        label_ax.text(
            0.5, 0.5, f'{class_name}\n{study_id}',
            fontsize=7, ha='center', va='center', wrap=True,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
        )

        # Load raw image
        if zarr_path not in zarr_cache:
            zarr_cache[zarr_path] = get_zarr_file(zarr_path)
        zarr_file = zarr_cache[zarr_path]
        img_raw = zarr_file[:, :, img_idx]
        img_uint8 = torch.from_numpy(np.array(img_raw)).unsqueeze(0)  # [1, H, W] uint8

        # Original (eval = always scalar normalization)
        img_eval = _apply_flat_field(img_uint8, zarr_path, background_correction,
                                     bg_value, p99_cache, p99_pool, train=False)
        img_eval = eval_augment(img_eval)
        _show_image(axes[row_idx][1], img_eval)
        if row_idx == 0:
            axes[row_idx][1].set_title('original', fontsize=8)

        # Augmented versions (train = uses selected correction mode)
        for aug_idx in range(num_augments):
            img_norm = _apply_flat_field(img_uint8.clone(), zarr_path, background_correction,
                                         bg_value, p99_cache, p99_pool, train=True)
            img_aug = train_augment(img_norm)
            _show_image(axes[row_idx][2 + aug_idx], img_aug)
            if row_idx == 0:
                axes[row_idx][2 + aug_idx].set_title(f'aug {aug_idx + 1}', fontsize=8)

    plt.tight_layout()
    suffix = f"_{background_correction}" if background_correction != 'none' else ''
    output_path = os.path.join(output_dir, f'sample_{model.replace("-", "_")}{suffix}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved to {output_path}')
