"""Baseline classifiers for comparing fine-tuned vs pretrained features.

Three baselines:
1. Pretrained + Linear: LogisticRegression on fresh pretrained (ImageNet) features
2. Linear: LogisticRegression on fine-tuned checkpoint features
3. MajorityClass: DummyClassifier (chance performance)
"""

import numpy as np
import pandas as pd
import torch
from typing import Tuple


def extract_features(
    model,
    data_module,
    bag_size: int = 200,
    seed: int = 403,
    device: str = None,
    pooling: str = "mean",
    crop_size: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract pooled backbone features from each experiment.

    For MIL models: uses _encode_instances() and pools.
    For instance models: uses backbone features and pools.

    Args:
        model: Trained Lightning model
        data_module: RemoIDDataModule
        bag_size: Number of images per experiment
        seed: Random seed for reproducible sampling
        device: Torch device
        pooling: Pooling method - "mean" or "max"

    Returns:
        (X, y, experiment_ids) where X is [n_experiments, feat_dim],
        y is [n_experiments], experiment_ids is [n_experiments]
    """
    import random
    from remo_id.inference.core import get_augmentations

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    backbone_name = getattr(model, "backbone_name", "resnet18")
    val_augments = get_augmentations(backbone_name, train=False, crop_size=crop_size)
    data_module.dataloader.augments = val_augments

    # Build path -> experiment_id lookup
    path_to_exp_id = dict(zip(
        data_module.data["path"].values,
        data_module.data["experiment_id"].astype(str).values,
    ))

    X_list = []
    y_list = []
    exp_id_list = []

    exp_ranges = data_module.experiment_ranges
    exp_labels = data_module.experiment_labels

    for path, (start_idx, end_idx) in exp_ranges.items():
        n_available = end_idx - start_idx
        label = exp_labels[path]

        rng = random.Random(seed + hash(str(path)) % 10000)
        n_pick = min(bag_size, n_available)
        local_indices = rng.sample(range(n_available), n_pick)
        global_indices = [start_idx + i for i in local_indices]

        images, _, _ = data_module.dataloader.get_items(global_indices)
        images = images.to(device)

        with torch.no_grad():
            if hasattr(model, "_encode_instances"):
                # MIL model — get instance features
                features = model._encode_instances(images)  # [N, feat_dim]
            elif hasattr(model, "backbone"):
                # Instance model with backbone attribute
                features = model.backbone(images)
                if features.dim() > 2:
                    features = features.mean(dim=[-2, -1])  # Global avg pool
            else:
                # Fallback: try forward hooks
                features = images  # Skip if no good extraction point
                continue

            # Pool across images to get experiment-level feature
            if pooling == "max":
                feat_pooled = features.max(dim=0).values.cpu().numpy()
            else:
                feat_pooled = features.mean(dim=0).cpu().numpy()

        X_list.append(feat_pooled)
        y_list.append(int(label))
        exp_id_list.append(path_to_exp_id.get(path, str(path)))

        del images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.array(X_list), np.array(y_list), np.array(exp_id_list)


def create_pretrained_model(checkpoint_path: str, device: str = None):
    """Create a fresh pretrained model (ImageNet weights) matching a checkpoint's architecture.

    Uses hyper_parameters from the checkpoint to reconstruct the same architecture,
    but with pretrained=True and no fine-tuned weights loaded.

    Args:
        checkpoint_path: Path to trained checkpoint (used to read architecture hparams)
        device: Torch device

    Returns:
        Fresh pretrained model in eval mode
    """
    from remo_id.models import get_model, is_mil_model
    from remo_id.inference.core import extract_checkpoint_metadata, _detect_in_channels, _detect_attn_dim

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hp = ckpt.get("hyper_parameters", {})
    state = ckpt.get("state_dict", ckpt)

    meta = extract_checkpoint_metadata(checkpoint_path)
    model_name = meta.get("model_name", "resnet18")
    in_channels = _detect_in_channels(state, model_name)

    kwargs = {
        "task_type": hp.get("task_type", "binary"),
        "num_classes": hp.get("num_classes", 2),
        "in_channels": in_channels,
        "pretrained": True,
    }

    if is_mil_model(model_name) or "resnet" in model_name.lower() or "dino" in model_name.lower():
        kwargs["finetune_layers"] = hp.get("finetune_layers", "layer3,layer4")
    if "dino" in model_name.lower():
        kwargs["dino_weights"] = hp.get("dino_weights")

    if is_mil_model(model_name):
        kwargs["n_experiments"] = hp.get("n_experiments", 8)
        kwargs["attn_dim"] = _detect_attn_dim(state)
        if "projection_dim" in hp:
            kwargs["projection_dim"] = hp["projection_dim"]
        if "pooling" in hp:
            kwargs["pooling"] = hp["pooling"]
        if "classifier_hidden" in hp:
            kwargs["classifier_hidden"] = hp["classifier_hidden"]

    model = get_model(model_name, **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    return model


def run_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task_type: str = "multiclass",
    num_classes: int = 2,
    X_train_pretrained: np.ndarray = None,
    X_val_pretrained: np.ndarray = None,
) -> pd.DataFrame:
    """Train baselines: Pretrained+Linear, Linear (fine-tuned features), MajorityClass.

    Args:
        X_train, y_train: Fine-tuned features and labels (training split)
        X_val, y_val: Fine-tuned features and labels (validation split)
        task_type: "binary" or "multiclass"
        num_classes: Number of classes
        X_train_pretrained, X_val_pretrained: Pretrained (ImageNet) features (optional)

    Returns:
        DataFrame with model, accuracy, balanced_accuracy, f1_macro, auc_macro
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.preprocessing import StandardScaler, label_binarize
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    )

    labels = list(range(num_classes))

    def _eval_classifier(name, clf, X_tr, X_va):
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        clf.fit(X_tr_s, y_train)
        y_pred = clf.predict(X_va_s)

        acc = accuracy_score(y_val, y_pred)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0, labels=labels)

        auc_val = np.nan
        try:
            if hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(X_va_s)
                if num_classes == 2:
                    auc_val = roc_auc_score(y_val, y_proba[:, 1])
                else:
                    yb = label_binarize(y_val, classes=labels)
                    auc_val = roc_auc_score(yb, y_proba, average="macro", multi_class="ovr")
        except Exception:
            pass

        return {
            "model": name,
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "f1_macro": float(f1),
            "auc_macro": float(auc_val),
        }

    rows = []

    # Pretrained + Linear (only if pretrained features provided)
    if X_train_pretrained is not None and X_val_pretrained is not None:
        rows.append(_eval_classifier(
            "Pretrained + Linear",
            LogisticRegression(max_iter=1000, random_state=42),
            X_train_pretrained, X_val_pretrained,
        ))

    # Linear (fine-tuned features)
    rows.append(_eval_classifier(
        "Linear",
        LogisticRegression(max_iter=1000, random_state=42),
        X_train, X_val,
    ))

    # MajorityClass
    rows.append(_eval_classifier(
        "MajorityClass",
        DummyClassifier(strategy="most_frequent"),
        X_train, X_val,
    ))

    return pd.DataFrame(rows)


def predict_fold_baselines(
    checkpoint_path: str,
    data_module,
    train_experiment_ids: list,
    val_experiment_ids: list,
    class_definitions: list,
    exclude_filters: list,
    include_filters: list = None,
    default_class_idx=None,
    task_type: str = "binary",
    num_classes: int = 2,
    backbone_name: str = "resnet18",
    bag_size: int = 200,
    batch_size: int = 256,
    seed: int = 403,
    pretrained_features: dict = None,
) -> dict:
    """Run all sklearn baselines for one CV fold.

    Args:
        checkpoint_path: Path to this fold's checkpoint
        data_module: Full RemoIDDataModule (used to build filtered subsets)
        train_experiment_ids: Experiment IDs for this fold's train split
        val_experiment_ids: Experiment IDs for this fold's val split
        class_definitions: Class definitions from class config
        exclude_filters: Exclude filters from class config
        default_class_idx: Default class index
        task_type: "binary" or "multiclass"
        num_classes: Number of classes
        backbone_name: Backbone name for data module
        bag_size: Images per experiment for feature extraction
        batch_size: DataModule batch size
        seed: Random seed
        pretrained_features: Pre-extracted pretrained features as
            {"mean": (X, y, ids), "max": (X, y, ids)} to avoid redundant extraction

    Returns:
        {baseline_name: (y_true, y_probs, y_pred, experiment_ids)}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.preprocessing import StandardScaler
    from remo_id.inference.core import load_checkpoint, extract_checkpoint_metadata
    from remo_id.data import RemoIDDataModule

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _build_ds(experiment_ids):
        return RemoIDDataModule(
            database_column="class_label", database_label=None,
            study_id=None, threshold=None, threshold_mode="==",
            negative_study_ids=[], negative_experiment_ids=[],
            include_experiment_ids=experiment_ids,
            batch_size=batch_size,
            backbone_name=backbone_name,
            train=False, task_type=task_type, num_classes=num_classes,
            class_definitions=class_definitions,
            exclude_filters=exclude_filters,
            include_filters=include_filters or [],
            default_class_idx=default_class_idx,
            skip_null_labels=True,
        )

    # Load finetuned model
    ckpt_meta = extract_checkpoint_metadata(checkpoint_path)
    model_name = ckpt_meta.get("model_name", "resnet18")
    model, device, _ = load_checkpoint(
        checkpoint_path, model_name,
        task_type=task_type, num_classes=num_classes,
        n_experiments=1, device=device,
    )

    # Extract finetuned features (both poolings)
    train_ds = _build_ds(train_experiment_ids)
    val_ds = _build_ds(val_experiment_ids)

    ft_features = {}
    for pool in ("mean", "max"):
        X_tr, y_tr, ids_tr = extract_features(model, train_ds, bag_size=bag_size, seed=seed, pooling=pool)
        X_va, y_va, ids_va = extract_features(model, val_ds, bag_size=bag_size, seed=seed, pooling=pool)
        ft_features[pool] = {
            "X_train": X_tr, "y_train": y_tr, "ids_train": ids_tr,
            "X_val": X_va, "y_val": y_va, "ids_val": ids_va,
        }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Pretrained features: partition from pre-extracted or extract fresh
    if pretrained_features is not None:
        pt_features = {}
        for pool in ("mean", "max"):
            X_all, y_all, ids_all = pretrained_features[pool]
            train_set = set(str(e) for e in train_experiment_ids)
            val_set = set(str(e) for e in val_experiment_ids)

            train_mask = np.array([str(eid) in train_set for eid in ids_all])
            val_mask = np.array([str(eid) in val_set for eid in ids_all])

            pt_features[pool] = {
                "X_train": X_all[train_mask], "y_train": y_all[train_mask], "ids_train": ids_all[train_mask],
                "X_val": X_all[val_mask], "y_val": y_all[val_mask], "ids_val": ids_all[val_mask],
            }
    else:
        pretrained_model = create_pretrained_model(checkpoint_path, device=device)
        train_ds_pt = _build_ds(train_experiment_ids)
        val_ds_pt = _build_ds(val_experiment_ids)
        pt_features = {}
        for pool in ("mean", "max"):
            X_tr, y_tr, ids_tr = extract_features(pretrained_model, train_ds_pt, bag_size=bag_size, seed=seed, pooling=pool)
            X_va, y_va, ids_va = extract_features(pretrained_model, val_ds_pt, bag_size=bag_size, seed=seed, pooling=pool)
            pt_features[pool] = {
                "X_train": X_tr, "y_train": y_tr, "ids_train": ids_tr,
                "X_val": X_va, "y_val": y_va, "ids_val": ids_va,
            }
        del pretrained_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Define baselines
    baselines = {
        "Finetuned Mean-pool + Linear": ("ft", "mean"),
        "Finetuned Max-pool + Linear": ("ft", "max"),
        "Pretrained Mean-pool + Linear": ("pt", "mean"),
        "Pretrained Max-pool + Linear": ("pt", "max"),
        "MajorityClass": ("majority", None),
    }

    results = {}
    labels = list(range(num_classes))

    for name, (feat_type, pool) in baselines.items():
        if feat_type == "majority":
            # Use finetuned mean features just for labels
            f = ft_features["mean"]
            clf = DummyClassifier(strategy="most_frequent")
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(f["X_train"])
            X_va_s = scaler.transform(f["X_val"])
            clf.fit(X_tr_s, f["y_train"])
            y_pred = clf.predict(X_va_s)
            y_probs = clf.predict_proba(X_va_s)
            results[name] = (f["y_val"], y_probs, y_pred, f["ids_val"])
        else:
            f = ft_features[pool] if feat_type == "ft" else pt_features[pool]
            clf = LogisticRegression(max_iter=1000, random_state=42)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(f["X_train"])
            X_va_s = scaler.transform(f["X_val"])
            clf.fit(X_tr_s, f["y_train"])
            y_pred = clf.predict(X_va_s)
            y_probs = clf.predict_proba(X_va_s)
            results[name] = (f["y_val"], y_probs, y_pred, f["ids_val"])

    return results
