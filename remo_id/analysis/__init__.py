"""Analysis module for remo-id."""

from remo_id.analysis.post_training import run_post_training_analysis
from remo_id.analysis.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    binconf,
    precision_recall_f1,
)
from remo_id.analysis.results import (
    save_metrics,
    load_metrics,
    save_curves,
    load_curves,
    analysis_output_dir,
)


__all__ = [
    "run_post_training_analysis",
    "compute_binary_metrics",
    "compute_multiclass_metrics",
    "binconf",
    "precision_recall_f1",
    # Results serialization
    "save_metrics",
    "load_metrics",
    "save_curves",
    "load_curves",
    "analysis_output_dir",
]
