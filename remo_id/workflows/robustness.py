"""Multi-seed robustness workflow.

Runs inference at multiple seeds to measure prediction stability,
then computes variability metrics and generates figures.

Usage:
    remo-id workflow robustness --checkpoint best.ckpt
"""

import os
from typing import List, Optional

import numpy as np

from remo_id.analysis.results import save_metrics, analysis_output_dir


def _resolve_checkpoint_files(checkpoint_path, class_config=None, splits=None):
    """Auto-discover classes.yaml and splits.yaml in the checkpoint directory.

    Args:
        checkpoint_path: Path to model checkpoint (.ckpt)
        class_config: Explicit class config path (skips discovery if provided)
        splits: Explicit splits path (skips discovery if provided)

    Returns:
        (class_config_path, splits_path) tuple

    Raises:
        FileNotFoundError: If auto-discovery fails for either file
    """
    ckpt_dir = os.path.dirname(checkpoint_path)

    if class_config is None:
        candidate = os.path.join(ckpt_dir, "classes.yaml")
        if os.path.exists(candidate):
            class_config = candidate
        else:
            raise FileNotFoundError(
                f"No classes.yaml found in {ckpt_dir}. "
                "Provide --class_config explicitly."
            )

    if splits is None:
        candidate = os.path.join(ckpt_dir, "splits.yaml")
        if os.path.exists(candidate):
            splits = candidate
        else:
            raise FileNotFoundError(
                f"No splits.yaml found in {ckpt_dir}. "
                "Provide --splits explicitly."
            )

    return class_config, splits


def _save_figures(figs: list, output_dir: str):
    """Save a list of (name, fig) tuples as PNG into output_dir."""
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    for name, fig in figs:
        fig.savefig(os.path.join(output_dir, f"{name}.png"))
        plt.close(fig)


ROBUSTNESS_SEEDS = [42, 123, 456, 789, 1011]


def run_robustness(
    checkpoint_path: str,
    output: Optional[str] = None,
    seeds: Optional[List[int]] = None,
    bag_size: int = 200,
    split: str = "val",
    class_config: Optional[str] = None,
    splits: Optional[str] = None,
    batch_size: int = 256,
):
    """Run multi-seed robustness analysis.

    For each seed, runs inference on the specified split and collects predictions.
    Then computes per-experiment prediction variance across seeds.

    Args:
        checkpoint_path: Path to model checkpoint.
        output: Output directory (default: ``{ckpt_dir}/analysis/robustness/``).
        seeds: List of random seeds (default: ROBUSTNESS_SEEDS).
        bag_size: Images per bag for inference.
        split: Which split to evaluate ("train", "val", or "all").
        class_config: Path to class config YAML (auto-discovered if None).
        splits: Path to splits YAML (auto-discovered if None).
        batch_size: DataModule batch size.
    """
    from remo_id.inference.context import build_analysis_context
    from remo_id.inference.mil import _predict_experiments
    from remo_id.inference.core import set_inference_seeds
    from remo_id.reports import figures as plots
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import label_binarize

    seeds = seeds or ROBUSTNESS_SEEDS
    out = output or analysis_output_dir(checkpoint_path, "robustness")
    os.makedirs(out, exist_ok=True)

    class_config, splits_path = _resolve_checkpoint_files(
        checkpoint_path, class_config, splits
    )

    base_ctx = build_analysis_context(
        checkpoint=checkpoint_path,
        class_config_path=class_config,
        splits_path=splits_path,
        split_name=split,
        bag_size=bag_size,
        seed=seeds[0],
        batch_size=batch_size,
    )
    class_names = base_ctx["class_names"]
    model = base_ctx["model"]
    ds = base_ctx["ds"]
    df = base_ctx["df"]
    task_type = base_ctx["task_type"]
    num_classes = base_ctx["num_classes"]
    device = base_ctx["device"]

    metrics_by_seed = {"macro_auc": [], "macro_ap": []}
    for seed in seeds:
        set_inference_seeds(seed)
        results, *_ = _predict_experiments(
            model=model, ds=ds, df=df, database_column="class_label",
            task_type=task_type, num_classes=num_classes,
            bag_size=bag_size, seed=seed, device=device,
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

    y_true = base_ctx["y_true"]
    y_probs = base_ctx["y_probs"]

    figs = [
        ("seed_variability", plots.plot_seed_variability(metrics_by_seed)),
        ("threshold_sensitivity", plots.plot_threshold_sensitivity(y_true, y_probs, class_names)),
        ("operating_points", plots.plot_operating_points(y_true, y_probs, class_names)),
    ]
    _save_figures(figs, out)

    # Save seed metrics
    metrics = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "checkpoint": checkpoint_path,
        "split": split,
        "macro_auc_mean": float(np.mean(metrics_by_seed["macro_auc"])) if metrics_by_seed["macro_auc"] else None,
        "macro_auc_std": float(np.std(metrics_by_seed["macro_auc"])) if metrics_by_seed["macro_auc"] else None,
        "macro_ap_mean": float(np.mean(metrics_by_seed["macro_ap"])) if metrics_by_seed["macro_ap"] else None,
        "macro_ap_std": float(np.std(metrics_by_seed["macro_ap"])) if metrics_by_seed["macro_ap"] else None,
    }
    save_metrics(os.path.join(out, "seed_metrics.json"), metrics)
    print(f"\nSaved robustness analysis to {out}")
