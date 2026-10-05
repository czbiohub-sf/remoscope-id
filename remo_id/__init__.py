"""
remo-id: Deep learning for microscopy image classification.

Multiple-instance learning framework for classifying Remoscope blood-microscopy
experiments (bags of frames or of precomputed cell embeddings). Supports binary
and multi-class classification.
"""

__version__ = "1.0.0"

# Re-export main public API
from remo_id.models import get_model, is_mil_model, MODEL_REGISTRY
from remo_id.data import RemoIDDataModule, RemoIDDataLoader, BatchSampler
from remo_id.inference import run_inference, load_checkpoint, detect_model_type

__all__ = [
    "__version__",
    "get_model",
    "is_mil_model",
    "MODEL_REGISTRY",
    "RemoIDDataModule",
    "RemoIDDataLoader",
    "BatchSampler",
    "run_inference",
    "load_checkpoint",
    "detect_model_type",
]
