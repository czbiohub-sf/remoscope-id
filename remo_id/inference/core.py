"""Core inference utilities shared between instance and MIL inference and the Grad-CAM gallery."""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from remo_id.models import get_augmentation_backbone, get_model, is_mil_model
from remo_id.data import RemoIDDataLoader


# Seeds for multiple inference runs (used by post-training analysis)
INFERENCE_SEEDS = [42, 123, 456, 789, 1011]


def set_inference_seeds(seed: int):
    """Set random seeds for reproducible inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_scalar(s: str):
    """Parse string to appropriate scalar type (bool, int, float, or str)."""
    sl = s.strip().lower()
    if sl in {"true", "t", "1", "yes", "y"}:
        return True
    if sl in {"false", "f", "0", "no", "n"}:
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def resolve_experiment_ids(
    experiment_ids: list = None,
    experiment_ids_file: str = None,
) -> List[str]:
    """Collect experiment IDs from a list and/or file, deduplicated and ordered."""
    ids = []
    if experiment_ids_file:
        with open(experiment_ids_file, "r") as f:
            ids.extend([ln.strip() for ln in f if ln.strip()])
    if experiment_ids:
        ids.extend([str(x).strip() for x in experiment_ids if str(x).strip()])
    return list(dict.fromkeys(ids))  # deduplicate preserving order


def label_from_row(
    row: pd.Series,
    database_column: str,
    label_mode: str,
    database_label: Optional[str],
    label_threshold: Optional[float],
    task_type: str = "binary",
    class_mapping: Optional[Dict] = None,
):
    """Extract label from row based on task type and label mode."""
    v = row[database_column]

    if task_type == "multiclass":
        if pd.isna(v):
            return -1
        if class_mapping is not None:
            return class_mapping.get(v, -1)
        return int(v)

    else:  # binary
        if pd.isna(v):
            return 0
        if label_mode == "eq":
            v_str = str(v).strip().lower()
            pos_str = str(database_label).strip().lower()
            return int(v_str == pos_str)
        elif label_mode == "gt":
            return int(float(v) > float(label_threshold))
        elif label_mode == "lt":
            return int(float(v) < float(label_threshold))
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")


def query_experiments_with_labels(
    study_id: str,
    database_column: str,
    experiment_ids: List[str] = None,
) -> pd.DataFrame:
    """Query experiments from database, joining study table for labels.

    Returns DataFrame with columns: experiment_id, sample_id, path, num_images,
    plus the label column from the study table.
    """
    from remo_id.data.datamodule import open_database
    dm = open_database()

    exp_tbl = dm.experiment.get()
    exp_sel = exp_tbl.select(["experiment_id", "path", "num_images"])
    if experiment_ids:
        exp_sel = exp_sel.filter(
            exp_tbl["experiment_id"].cast("string").isin([str(x) for x in experiment_ids])
        )

    if study_id:
        study_tbl = dm.studies[study_id].get()
        cols = ["experiment_id"]
        col_names = [c.get_name() if hasattr(c, 'get_name') else str(c) for c in study_tbl.columns]
        if "study_id" in col_names:
            cols.append("study_id")
        if "sample_id" in col_names:
            cols.append("sample_id")
        cols.append(database_column)
        joined = exp_sel.inner_join(
            study_tbl.select(cols),
            predicates=["experiment_id"],
        )
        df = joined.execute()
    else:
        df = exp_sel.execute()
        if database_column not in df.columns:
            df[database_column] = None

    # Ensure sample_id column exists
    if "sample_id" not in df.columns:
        df["sample_id"] = None

    if experiment_ids:
        df = df[df["experiment_id"].astype(str).isin([str(x) for x in experiment_ids])].reset_index(drop=True)

    return df


def resolve_negative_studies(
    df: pd.DataFrame,
    database_column: str,
    negative_study_ids: List[str] = None,
    negative_experiment_ids: List[int] = None,
) -> pd.DataFrame:
    """Add forced-negative experiments from negative studies/experiment IDs.

    Adds '_is_forced_negative' column to the DataFrame. Negative study experiments
    are appended; negative experiment IDs already in the DataFrame are marked in-place.
    """
    from remo_id.data.datamodule import open_database
    dm = open_database()

    df = df.copy()
    df["_is_forced_negative"] = False

    for neg_study_id in (negative_study_ids or []):
        try:
            neg_study_tbl = dm.studies[neg_study_id].get()
            neg_study_cols = [c.get_name() if hasattr(c, 'get_name') else str(c) for c in neg_study_tbl.columns]
            neg_exp_ids = neg_study_tbl.select(["experiment_id"]).execute()["experiment_id"].astype(str).tolist()
            exp_tbl = dm.experiment.get()
            neg_exp_df = exp_tbl.select(["experiment_id", "path", "num_images"]).execute()
            # Get sample_id from study table if available
            if "sample_id" in neg_study_cols:
                sid_df = neg_study_tbl.select(["experiment_id", "sample_id"]).execute()
                sid_df["experiment_id"] = sid_df["experiment_id"].astype(str)
                neg_exp_df["experiment_id"] = neg_exp_df["experiment_id"].astype(str)
                neg_exp_df = neg_exp_df.merge(sid_df, on="experiment_id", how="left")
            else:
                neg_exp_df["sample_id"] = None
            neg_exp_df = neg_exp_df[neg_exp_df["experiment_id"].astype(str).isin(neg_exp_ids)].copy()
            if len(neg_exp_df) > 0:
                neg_exp_df[database_column] = None
                neg_exp_df["_is_forced_negative"] = True
                existing_ids = set(df["experiment_id"].astype(str))
                neg_exp_df = neg_exp_df[~neg_exp_df["experiment_id"].astype(str).isin(existing_ids)]
                df = pd.concat([df, neg_exp_df], ignore_index=True)
                print(f"Added {len(neg_exp_df)} experiments from negative study '{neg_study_id}'")
        except Exception as e:
            print(f"Warning: Could not load negative study '{neg_study_id}': {e}")

    for neg_exp_id in (negative_experiment_ids or []):
        if str(neg_exp_id) in df["experiment_id"].astype(str).values:
            df.loc[df["experiment_id"].astype(str) == str(neg_exp_id), "_is_forced_negative"] = True

    return df


def load_checkpoint(
    checkpoint_path: str,
    model_name: str = "resnet18",
    task_type: str = "binary",
    num_classes: int = 2,
    n_experiments: int = 1,
    device: str = None,
):
    """
    Load a model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model_name: Name of the model architecture
        task_type: Task type (binary or multiclass)
        num_classes: Number of classes for multiclass
        n_experiments: Number of experiments (for MIL models)
        device: Device to load model on (default: auto-detect)

    Returns:
        Tuple of (model, device)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint first to detect input channels
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # Detect input channels from checkpoint
    in_channels = _detect_in_channels(state, model_name)
    print(f"Loading model {model_name} from {checkpoint_path}")
    print(f"Detected input channels: {in_channels}")

    # Extract hyperparameters from checkpoint if available
    hp = ckpt.get("hyper_parameters", {})

    # Embedding-MIL has no image backbone: rebuild from saved hparams (embedding_dim etc.),
    # n_experiments=1 for inference. No in_channels/pretrained/finetune_layers (not accepted).
    if model_name == "embedding-mil":
        kwargs = {
            "task_type": task_type, "num_classes": num_classes,
            "n_experiments": n_experiments,
            "embedding_dim": hp.get("embedding_dim"),
            "attn_dim": hp.get("attn_dim", _detect_attn_dim(state)),
        }
        for k in ("projection_dim", "pooling", "classifier_hidden", "num_heads", "top_k", "dropout"):
            if k in hp:
                kwargs[k] = hp[k]
        model = get_model(model_name, **kwargs)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"Warning: missing keys: {missing}, unexpected keys: {unexpected}")
        model.eval()
        model.to(device)
        return model, device, in_channels

    # Build model kwargs
    kwargs = {
        "task_type": task_type,
        "num_classes": num_classes,
        "in_channels": in_channels,
    }

    # Pass pretrained/finetune_layers for models that support them
    if is_mil_model(model_name) or 'resnet' in model_name.lower() or 'dino' in model_name.lower():
        kwargs["pretrained"] = hp.get("pretrained", True)
        kwargs["finetune_layers"] = hp.get("finetune_layers", "layer3,layer4")
    if 'dino' in model_name.lower():
        # The checkpoint's state_dict replaces every backbone weight below, so the
        # (separately licensed) DINOv3 pretrained weights are not needed here.
        kwargs["pretrained"] = False

    # Add MIL-specific kwargs
    if is_mil_model(model_name):
        kwargs["n_experiments"] = n_experiments
        kwargs["attn_dim"] = _detect_attn_dim(state)
        print(f"Detected attn_dim: {kwargs['attn_dim']}")

        # Capacity-control params
        if "projection_dim" in hp:
            kwargs["projection_dim"] = hp["projection_dim"]
        if "pooling" in hp:
            kwargs["pooling"] = hp["pooling"]
        if "classifier_hidden" in hp:
            kwargs["classifier_hidden"] = hp["classifier_hidden"]
        if "num_heads" in hp:
            kwargs["num_heads"] = hp["num_heads"]
        if "top_k" in hp:
            kwargs["top_k"] = hp["top_k"]

    model = get_model(model_name, **kwargs)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"Warning: missing keys: {missing}, unexpected keys: {unexpected}")
    model.eval()
    model.to(device)

    return model, device, in_channels


def _detect_in_channels(state_dict: dict, model_name: str) -> int:
    """
    Detect the number of input channels from a checkpoint state dict.

    Args:
        state_dict: Model state dict
        model_name: Model architecture name

    Returns:
        Number of input channels (1 or 3)
    """
    # Common keys for first conv layer across architectures
    conv_keys = [
        # ResNet patterns (instance and MIL)
        "model.conv1.weight",
        "backbone.conv1.weight",
        "encoder.conv1.weight",
        "resnet_backbone.conv1.weight",
        # ViT patterns
        "model.conv_proj.weight",
        "backbone.conv_proj.weight",
        "encoder.conv_proj.weight",
        # MicroscopyCNN patterns
        "backbone.encoder.0.weight",
        "encoder.0.weight",
    ]

    for key in conv_keys:
        if key in state_dict:
            weight = state_dict[key]
            if weight.dim() >= 2:
                # Conv weight shape: [out_channels, in_channels, ...]
                return weight.shape[1]

    # Default to 1 channel (grayscale) if not found
    return 1


def _detect_attn_dim(state_dict: dict) -> int:
    """Detect the attention hidden dimension from a MIL checkpoint state dict."""
    # attn_V.weight has shape [attn_dim, feat_dim]
    if "attn_V.weight" in state_dict:
        return state_dict["attn_V.weight"].shape[0]
    return 128


def auto_detect_from_checkpoint(checkpoint_path: str, **overrides) -> dict:
    """Extract model_name, task_type, num_classes from checkpoint metadata.

    Overrides take precedence over auto-detected values.
    Returns dict with detected/overridden values.
    """
    meta = extract_checkpoint_metadata(checkpoint_path)
    detected = {
        "model_name": meta.get("model_name"),
        "task_type": meta.get("task_type", "binary"),
        "num_classes": meta.get("num_classes", 2),
        "class_names": meta.get("class_names"),
    }
    detected.update(overrides)
    return detected


def extract_background_value(checkpoint_path: str) -> float:
    """Extract background_value from checkpoint hyperparameters.

    Returns the training background_value if stored, else 225.0 (new default).
    Old checkpoints trained with /255 normalization won't have this key.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hp = ckpt.get("hyper_parameters", {})
    val = hp.get("background_value", None)
    if val is not None:
        return float(val)
    # Check nested data section (logged in hyperparams["data"])
    # Not available via hyper_parameters, so fall back to default
    return 225.0


def extract_checkpoint_metadata(checkpoint_path: str) -> dict:
    """Extract metadata from checkpoint without loading the model.

    Returns dict with keys:
        task_type, num_classes, class_names, model_name,
        database_column, database_label, class_definitions
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hp = ckpt.get("hyper_parameters", {})

    task_type = hp.get("task_type", "binary")
    num_classes = hp.get("num_classes", 2)
    model_name = hp.get("model_name", "")
    if not model_name and "embedding_dim" in hp:
        # EmbeddingMIL has no image backbone; embedding_dim uniquely identifies it.
        model_name = "embedding-mil"
    if not model_name:
        # Reconstruct model_name from backbone + pooling (older checkpoints)
        backbone = hp.get("backbone", "")
        is_mil = "pooling" in hp
        if not backbone:
            # Infer backbone from state_dict keys
            state_keys = list(ckpt.get("state_dict", {}).keys())
            if any(k.startswith("dino.") for k in state_keys):
                backbone = "dino-vits16"
            elif any(k.startswith("backbone.layer") for k in state_keys):
                backbone = "resnet18"
            elif any(k.startswith("vit.") for k in state_keys):
                backbone = "vit-l-16"
        if backbone and is_mil:
            model_name = f"{backbone}-mil"
        elif backbone:
            model_name = backbone
        if not model_name:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has no 'model_name' in hyper_parameters. "
                "This checkpoint predates the model_name convention and is no longer supported."
            )

    database_column = hp.get("database_column", None)
    database_label = hp.get("database_label", None)
    class_definitions = hp.get("class_definitions", None)

    # Extract class_names with fallback chain
    class_names = None

    # Try class_definitions first
    if class_definitions and isinstance(class_definitions, list):
        try:
            raw_names = [d["name"] for d in class_definitions if isinstance(d, dict) and "name" in d]
            # Deduplicate preserving order (same logic as cli.py)
            class_names = list(dict.fromkeys(raw_names))
        except (KeyError, TypeError):
            pass

    # Fallback: try class_config
    if class_names is None:
        class_config = hp.get("class_config_raw", None)
        if class_config and isinstance(class_config, dict):
            classes = class_config.get("classes", [])
            if classes:
                try:
                    raw_names = [c["name"] for c in classes if isinstance(c, dict) and "name" in c]
                    class_names = list(dict.fromkeys(raw_names))
                except (KeyError, TypeError):
                    pass

    # Final fallback: generic names
    if class_names is None and task_type == "multiclass":
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Read crop_size and backbone_name from sibling metadata.json if available
    crop_size = None
    backbone_name = None
    meta_json = os.path.join(os.path.dirname(checkpoint_path), "metadata.json")
    if os.path.exists(meta_json):
        with open(meta_json) as f:
            meta = json.load(f)
        crop_size = meta.get("crop_size", None)
        backbone_name = meta.get("backbone_name", None)
    if backbone_name is None:
        backbone_name = get_augmentation_backbone(model_name)

    return {
        "task_type": task_type,
        "num_classes": num_classes,
        "class_names": class_names,
        "model_name": model_name,
        "database_column": database_column,
        "database_label": database_label,
        "class_definitions": class_definitions,
        "crop_size": crop_size,
        "backbone_name": backbone_name,
    }


def detect_model_type(checkpoint_path: str) -> str:
    """
    Auto-detect if checkpoint is for an instance or MIL model.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        "instance" or "mil"
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "hyper_parameters" in ckpt:
            hp = ckpt["hyper_parameters"]
            model_name = hp.get("model_name", "")
            if is_mil_model(model_name):
                return "mil"
        # Check state dict keys for MIL-specific layers
        state = ckpt.get("state_dict", ckpt)
        keys = list(state.keys())
        mil_indicators = ["attn_V", "attn_w", "attention."]
        for key in keys:
            for indicator in mil_indicators:
                if indicator in key:
                    return "mil"
    except Exception:
        pass

    return "instance"


def get_augmentations(backbone_name: str, train: bool = False, crop_size: int = None):
    """
    Get data augmentation pipeline for a given backbone.

    Args:
        backbone_name: Name of the backbone model
        train: If True, use training augmentations; else validation
        crop_size: Override default crop size (backbone-specific default if None)

    Returns:
        Augmentation transform
    """
    return RemoIDDataLoader.prep_augments(backbone_name=backbone_name, train=train, crop_size=crop_size)


def preprocess(img_np: np.ndarray, augments=None, target_channels: int = 1, background_value: float = 225.0) -> torch.Tensor:
    """
    Preprocess a single image for inference.

    Args:
        img_np: Input image as numpy array [H,W] or [H,W,1] or [H,W,3]
        augments: Optional augmentation transform
        target_channels: Number of channels the model expects (1 or 3)
        background_value: Scalar divisor for uint8→float32 normalization (default 225)

    Returns:
        Preprocessed tensor [1, C, H, W]
    """
    # Handle single-channel
    if img_np.ndim == 3 and img_np.shape[-1] == 1:
        img_np = img_np[..., 0]

    x = torch.from_numpy(np.array(img_np))

    # ensure CHW
    if x.ndim == 2:
        x = x.unsqueeze(0)  # [1,H,W]
    elif x.ndim == 3 and x.shape[-1] in (1, 3):  # HWC
        x = x.permute(2, 0, 1)  # CHW

    # Replicate channels if needed
    current_channels = x.shape[0]
    if current_channels == 1 and target_channels == 3:
        x = x.repeat(3, 1, 1)  # [3, H, W]
    elif current_channels == 3 and target_channels == 1:
        x = x.mean(dim=0, keepdim=True)  # [1, H, W]

    # Normalize uint8 → float32 (matches _apply_flat_field scalar mode)
    x = x.float() / float(background_value)

    x = x.unsqueeze(0)  # [1, C, H, W]

    if augments is not None:
        x = augments(x)

    return x


def preprocess_batch(images_np: List[np.ndarray], augments=None, target_channels: int = 1, background_value: float = 225.0) -> torch.Tensor:
    """
    Preprocess a batch of images for inference.

    Args:
        images_np: List of input images as numpy arrays
        augments: Optional augmentation transform
        target_channels: Number of channels the model expects (1 or 3)
        background_value: Scalar divisor for uint8→float32 normalization (default 225)

    Returns:
        Preprocessed tensor [B, C, H, W]
    """
    tensors = []
    for img_np in images_np:
        if img_np.ndim == 3 and img_np.shape[-1] == 1:
            img_np = img_np[..., 0]

        x = torch.from_numpy(np.array(img_np))

        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim == 3 and x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1)

        # Replicate channels if needed
        current_channels = x.shape[0]
        if current_channels == 1 and target_channels == 3:
            x = x.repeat(3, 1, 1)  # [3, H, W]
        elif current_channels == 3 and target_channels == 1:
            x = x.mean(dim=0, keepdim=True)  # [1, H, W]

        tensors.append(x)

    batch = torch.stack(tensors, dim=0)  # [B, C, H, W]

    # Normalize uint8 → float32 (matches _apply_flat_field scalar mode)
    batch = batch.float() / float(background_value)

    if augments is not None:
        batch = augments(batch)

    return batch


@torch.no_grad()
def predict_proba(model, x: torch.Tensor, task_type: str = "binary"):
    """
    Predict based on task type.

    Args:
        model: The model to use for prediction
        x: Input tensor [1, C, H, W]
        task_type: Task type (binary or multiclass)

    Returns:
        - binary: probability (0-1) via sigmoid
        - multiclass: (class_index, max_prob, all_probs)
    """
    logits = model(x)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    if task_type == "binary":
        logits = logits.squeeze()
        prob = torch.sigmoid(logits).item()
        return prob
    elif task_type == "multiclass":
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        pred_class = int(probs.argmax(dim=-1).item())
        max_prob = float(probs.max().item())
        return pred_class, max_prob, probs.squeeze().cpu().numpy()
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


@torch.no_grad()
def predict_proba_batch(model, x: torch.Tensor, task_type: str = "binary", batch_size: int = 32):
    """
    Batch prediction for efficiency.

    Args:
        model: The model to use for prediction
        x: Input tensor [N, C, H, W]
        task_type: Task type (binary or multiclass)
        batch_size: Batch size for processing

    Returns:
        List of predictions
    """
    results = []
    n_samples = x.shape[0]

    for i in range(0, n_samples, batch_size):
        batch = x[i:i+batch_size]
        logits = model(batch)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if task_type == "binary":
            probs = torch.sigmoid(logits.squeeze(-1))
            results.extend(probs.cpu().tolist())
        elif task_type == "multiclass":
            probs = torch.softmax(logits, dim=-1)
            pred_classes = probs.argmax(dim=-1)
            max_probs = probs.max(dim=-1).values
            for j in range(len(batch)):
                results.append((
                    int(pred_classes[j].item()),
                    float(max_probs[j].item()),
                    probs[j].cpu().numpy()
                ))

    return results


def resolve_experiments_from_class_config(class_config_path: str, experiment_ids: List[str] = None) -> Tuple[pd.DataFrame, list, int, str]:
    """Resolve experiments and labels from a class_config YAML.

    Returns:
        (df, class_names, num_classes, task_type) where df has columns:
        experiment_id, sample_id, study_id, path, num_images, y
    """
    from remo_id.inference.context import load_class_config

    class_definitions, class_names, num_classes, task_type, exclude_filters, include_filters, default_class_idx = (
        load_class_config(class_config_path)
    )
    df = resolve_experiments_from_class_definitions(
        class_definitions, task_type, num_classes, exclude_filters=exclude_filters,
        include_filters=include_filters, default_class_idx=default_class_idx,
        experiment_ids=experiment_ids,
    )
    return df, class_names, num_classes, task_type


def resolve_experiments_from_class_definitions(
    class_definitions: list,
    task_type: str,
    num_classes: int,
    exclude_filters: list = None,
    include_filters: list = None,
    default_class_idx: int = None,
    experiment_ids: List[str] = None,
) -> pd.DataFrame:
    """Resolve parsed class definitions (database- or manifest-backed) to a
    DataFrame with experiment_id, sample_id, study_id, path, num_images, y."""
    from remo_id.data import RemoIDDataModule

    ds = RemoIDDataModule(
        database_column="class_label",
        include_experiment_ids=experiment_ids or [],
        batch_size=1,
        backbone_name="resnet",
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

    df = ds.data.copy()
    df["experiment_id"] = df["experiment_id"].astype(str)
    if experiment_ids:
        df = df[df["experiment_id"].isin([str(x) for x in experiment_ids])].copy()
    df["y"] = df["class_label"].astype(int)
    for col in ("sample_id", "study_id"):
        if col not in df.columns:
            df[col] = None
    return df.reset_index(drop=True)


def get_experiment_metadata(experiment_id: str, study_id: str) -> dict:
    """Fetch all columns from the study table for an experiment.

    Args:
        experiment_id: Experiment ID to look up
        study_id: Study ID for the study table

    Returns:
        Dict of column_name -> value for this experiment
    """
    from remo_id.data.datamodule import open_database
    dm = open_database()

    try:
        study_tbl = dm.studies[study_id].get()
        row = study_tbl.filter(
            study_tbl["experiment_id"].cast("string") == str(experiment_id)
        ).execute()
        if len(row) == 0:
            return {}
        return row.iloc[0].to_dict()
    except Exception:
        return {}


def aggregate_image_predictions_binary(
    image_probs: List[float],
    image_thresh: float,
    exp_pos_pct: Optional[float] = None
) -> Tuple[int, int, float, float]:
    """
    Aggregate image-level predictions to experiment level for binary classification.

    Args:
        image_probs: List of image-level probabilities
        image_thresh: Threshold for positive prediction
        exp_pos_pct: Experiment positive if %positive >= this (0-100)

    Returns:
        Tuple of (exp_pred, n_pos_images, pct_pos_images, mean_prob)
    """
    probs = np.array(image_probs, dtype=float)
    pos = (probs > image_thresh).astype(int)
    n_pos = int(pos.sum())
    n = len(pos)
    pct = (n_pos / max(1, n)) * 100.0
    mean_p = float(probs.mean()) if n > 0 else 0.0

    if exp_pos_pct is not None:
        exp_pred = int(pct >= exp_pos_pct)
    else:
        exp_pred = int(n_pos >= 1)

    return exp_pred, n_pos, pct, mean_p


def aggregate_image_predictions_multiclass(
    image_preds: List[Tuple[int, float, np.ndarray]],
    num_classes: int
) -> Tuple[int, float, dict]:
    """
    Aggregate image-level predictions to experiment level for multiclass.

    Args:
        image_preds: List of (pred_class, max_prob, class_probs) tuples
        num_classes: Number of classes

    Returns:
        Tuple of (exp_pred, mean_confidence, class_counts)
    """
    from collections import Counter

    if len(image_preds) == 0:
        return 0, 0.0, {}

    pred_classes = [p[0] for p in image_preds]
    confidences = [p[1] for p in image_preds]

    # Mode: most common prediction
    class_counts = Counter(pred_classes)
    exp_pred = class_counts.most_common(1)[0][0]

    # Mean confidence for the winning class
    winning_confidences = [c for p, c in zip(pred_classes, confidences) if p == exp_pred]
    mean_confidence = float(np.mean(winning_confidences)) if winning_confidences else 0.0

    return exp_pred, mean_confidence, dict(class_counts)
