"""Save and load analysis output (metrics, curves, matrices).

Analysis results are saved to ``{checkpoint_dir}/analysis/{subcommand}/``.

Formats:
    metrics.json      — scalar metrics with optional bootstrap CIs
    roc_curves.npz    — fpr/tpr arrays per class
    pr_curves.npz     — precision/recall arrays
    confusion_matrix.npz
    calibration.npz
"""

import json
import os
from typing import Any, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Metrics (JSON)
# ---------------------------------------------------------------------------

def save_metrics(path: str, metrics: Dict[str, Any]):
    """Save metrics dict to JSON.

    Handles numpy types and ndarray conversion automatically.

    Args:
        path: Output JSON file path.
        metrics: Dict of metric names to values. Values can be scalars,
            lists, or nested dicts. numpy arrays are converted to lists.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = _make_json_serializable(metrics)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved metrics: {path}")


def load_metrics(path: str) -> Dict[str, Any]:
    """Load metrics dict from JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No metrics file at {path}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Curves / arrays (NPZ)
# ---------------------------------------------------------------------------

def save_curves(path: str, **arrays):
    """Save curve arrays to compressed NPZ.

    Args:
        path: Output NPZ file path.
        **arrays: Named numpy arrays to save (e.g., fpr=fpr, tpr=tpr).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {k: np.asarray(v) for k, v in arrays.items()}
    np.savez_compressed(path, **data)
    print(f"Saved curves: {path}")


def load_curves(path: str) -> Dict[str, np.ndarray]:
    """Load curve arrays from NPZ.

    Returns:
        Dict mapping array names to numpy arrays.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No curves file at {path}")
    return dict(np.load(path, allow_pickle=False))


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def analysis_output_dir(checkpoint_path: str, subcommand: str) -> str:
    """Compute default analysis output directory for a subcommand.

    Convention: ``{checkpoint_dir}/analysis/{subcommand}/``
    """
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    out = os.path.join(ckpt_dir, "analysis", subcommand)
    os.makedirs(out, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj
