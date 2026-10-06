"""Inference utilities for remo-id."""

from remo_id.inference.core import (
    INFERENCE_SEEDS,
    load_checkpoint,
    detect_model_type,
    get_augmentations,
    label_from_row,
    parse_scalar,
    preprocess,
    preprocess_batch,
    predict_proba,
    predict_proba_batch,
    resolve_experiment_ids,
    set_inference_seeds,
)
from remo_id.inference.results import (
    ExperimentResult,
    save_predictions,
    load_predictions,
    load_attention,
    load_embeddings,
    has_saved_predictions,
    results_to_arrays,
    dataframe_to_results,
)

__all__ = [
    "load_checkpoint",
    "detect_model_type",
    "get_augmentations",
    "preprocess",
    "preprocess_batch",
    "predict_proba",
    "predict_proba_batch",
    "run_inference",
    # Results serialization
    "ExperimentResult",
    "save_predictions",
    "load_predictions",
    "load_attention",
    "load_embeddings",
    "has_saved_predictions",
    "results_to_arrays",
    "dataframe_to_results",
]


def run_inference(
    checkpoint: str,
    study_id: str = None,
    model_type: str = "auto",
    **kwargs
):
    """Unified inference entry point.

    Args:
        checkpoint: Path to model checkpoint
        study_id: Study ID to run inference on (None when using class_config)
        model_type: "instance", "mil", or "auto" (auto-detect)
        **kwargs: Additional arguments for inference

    Returns:
        Inference results
    """
    if model_type == "auto":
        # Check model_name kwarg first (set by CLI auto-detection from metadata)
        model_name = kwargs.get("model_name", "")
        from remo_id.models import is_mil_model
        if model_name and is_mil_model(model_name):
            model_type = "mil"
        else:
            model_type = detect_model_type(checkpoint)

    if model_type == "mil":
        from remo_id.inference.mil import run_mil_inference
        return run_mil_inference(checkpoint=checkpoint, study_id=study_id, **kwargs)
    else:
        from remo_id.inference.instance import run_instance_inference
        return run_instance_inference(checkpoint=checkpoint, study_id=study_id, **kwargs)
