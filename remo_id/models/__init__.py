"""Model definitions for remo-id."""

from remo_id.models.base import BaseModel
from remo_id.models.instance import (
    ResNetModel,
    ViTModel,
    DINOv3Model,
    MicroscopyCNNModel,
    MicroscopyEncoder,
    PatchGatedAttention,
    create_single_channel_resnet,
    BaseInstanceModel,
)
from remo_id.models.mil import (
    ResNetAttentionMIL,
    ViTLAttentionMIL,
    DINOv3AttentionMIL,
    MicroscopyCNNAttentionMIL,
    BaseMILModel,
)
from remo_id.models.embedding_mil import EmbeddingMIL

# Model registry for easy lookup
MODEL_REGISTRY = {
    # Instance-level models
    "resnet18": ResNetModel,
    "resnet50": ResNetModel,
    "vit-b-16": ViTModel,
    "vit-l-16": ViTModel,
    "microscopy-cnn": MicroscopyCNNModel,
    "dino-vits16": DINOv3Model,
    "dino-vitsplus": DINOv3Model,
    "dino-vitb": DINOv3Model,
    # MIL models with attention
    "resnet18-mil": ResNetAttentionMIL,
    "resnet50-mil": ResNetAttentionMIL,
    "vit-l-16-mil": ViTLAttentionMIL,
    "dino-vits16-mil": DINOv3AttentionMIL,
    "dino-vitsplus-mil": DINOv3AttentionMIL,
    "dino-vitb-mil": DINOv3AttentionMIL,
    "microscopy-cnn-mil": MicroscopyCNNAttentionMIL,
    # MIL over precomputed cell embeddings (no backbone)
    "embedding-mil": EmbeddingMIL,
}

# Models that need backbone parameter specified
_BACKBONE_MODELS = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "resnet18-mil": "resnet18",
    "resnet50-mil": "resnet50",
    "vit-b-16": "vit-b-16",
    "vit-l-16": "vit-l-16",
    "dino-vits16": "dino-vits16",
    "dino-vitsplus": "dino-vitsplus",
    "dino-vitb": "dino-vitb",
    "dino-vits16-mil": "dino-vits16",
    "dino-vitsplus-mil": "dino-vitsplus",
    "dino-vitb-mil": "dino-vitb",
}

# Maps model names to their backbone type for augmentation selection
_AUGMENTATION_BACKBONE_MAP = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "vit-b-16": "vit-b-16",
    "vit-l-16": "vit-l-16",
    "microscopy-cnn": "microscopy-cnn",
    "dino-vits16": "dino-vits16",
    "dino-vitsplus": "dino-vits16",
    "dino-vitb": "dino-vitb",
    "resnet18-mil": "resnet18",
    "resnet50-mil": "resnet50",
    "vit-l-16-mil": "vit-l-16",
    "dino-vits16-mil": "dino-vits16",
    "dino-vitsplus-mil": "dino-vits16",
    "dino-vitb-mil": "dino-vitb",
    "microscopy-cnn-mil": "microscopy-cnn",
    "embedding-mil": "embedding",  # no image augmentation; value unused
}


def get_model(name: str, **kwargs):
    """
    Get a model by name from the registry.

    Args:
        name: Model name (e.g., 'resnet18', 'vit-l-16-mil')
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        Instantiated model
    """
    name_lower = name.lower()
    if name_lower not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[name_lower]

    # Add backbone parameter for models that need it
    if name_lower in _BACKBONE_MODELS and "backbone" not in kwargs:
        kwargs["backbone"] = _BACKBONE_MODELS[name_lower]

    return model_cls(**kwargs)


def get_augmentation_backbone(model_name: str) -> str:
    """Map model name to backbone name for augmentation selection."""
    return _AUGMENTATION_BACKBONE_MAP.get(model_name.lower(), model_name.lower())


def is_mil_model(name: str) -> bool:
    """Check if a model name refers to a MIL (Multiple Instance Learning) model."""
    name_lower = name.lower()
    return "-mil" in name_lower


def list_models() -> list[str]:
    """Return list of available model names."""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    # Base classes
    "BaseModel",
    "BaseInstanceModel",
    "BaseMILModel",
    # Instance models
    "ResNetModel",
    "ViTModel",
    "MicroscopyCNNModel",
    "DINOv3Model",
    "MicroscopyEncoder",
    # MIL models
    "ResNetAttentionMIL",
    "ViTLAttentionMIL",
    "DINOv3AttentionMIL",
    "MicroscopyCNNAttentionMIL",
    "EmbeddingMIL",
    # Helpers
    "create_single_channel_resnet",
    # Registry functions
    "MODEL_REGISTRY",
    "get_model",
    "get_augmentation_backbone",
    "is_mil_model",
    "list_models",
]
