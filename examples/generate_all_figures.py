#!/usr/bin/env python
"""Generate all figures from a cohesive dummy dataset for visual iteration.

Usage (from the repository root):
    python examples/generate_all_figures.py [output_dir]

Default output: examples/figure_output/
"""

import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The dummy dataset lives with the tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from dummy_dataset import DummyDataset

from remo_id.reports import style

from remo_id.reports.figures.classification import (
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrix,
    plot_calibration,
    plot_prob_distributions,
    metrics_table_with_ci,
)
from remo_id.reports.figures.comparison import (
    plot_seed_variability,
    plot_threshold_sensitivity,
    plot_operating_points,
    plot_kfold_variability,
    plot_baseline_comparison,
    plot_experiment_probabilities,
    plot_roc_binary_splits,
    plot_repeatability_hist,
    plot_optimal_thresholds,
    plot_dual_operating_points,
)
from remo_id.reports.figures.data import (
    plot_bag_size_sweep,
    plot_sampling_position,
    plot_learning_curve,
    plot_class_distribution,
    plot_error_confidence,
    plot_error_by_study,
    plot_confusion_errors,
)
from remo_id.reports.figures.interpretation import (
    plot_top_attention_grid,
    plot_umap,
    plot_attention_entropy,
)


def save(fig, outdir: Path, name: str):
    """Save figure and close.

    Uses the style-sheet savefig defaults (450 dpi) so the gallery reflects
    publication-quality output.
    """
    png = outdir / f"{name}.png"
    fig.savefig(png)                       # 450 dpi via rcParams["savefig.dpi"]
    plt.close(fig)
    return png


def _run_paper_analyses(ds, outdir, run):
    """Pairwise, dual operating points, MOE, per-fold thresholds and
    configuration comparison. Each produces its own subdirectory
    under ``outdir``."""
    from remo_id.analysis.optimal_thresholds import (
        find_youden_thresholds, find_optimal_thresholds,
        build_sensitivity_targets, dual_operating_points_table,
    )

    # --- Pairwise ------------------------------------------------------------
    pairwise_dir = outdir / "pairwise_disease_vs_healthy"
    pairwise_dir.mkdir(exist_ok=True)
    run("pairwise_disease_vs_healthy", lambda: _pairwise_wrapper(
        ds, "disease+", "healthy", pairwise_dir,
    ))

    # --- Dual operating points -----------------------------------------------
    targets = build_sensitivity_targets(ds.CLASS_NAMES, 0.95, None)
    run("dual_operating_points_table", lambda: dual_operating_points_table(
        ds.y_true, ds.y_probs, ds.CLASS_NAMES, targets,
    ))
    youden = find_youden_thresholds(ds.y_true, ds.y_probs, ds.CLASS_NAMES)
    highsens = find_optimal_thresholds(ds.y_true, ds.y_probs, ds.CLASS_NAMES, targets)
    run("dual_operating_points", lambda: plot_dual_operating_points(
        ds.y_true, ds.y_probs, ds.CLASS_NAMES, youden, highsens,
        target_class=ds.CLASS_NAMES[0],
    ))

    # --- MOE -----------------------------------------------------------------
    moe_dir = outdir / "moe"
    moe_dir.mkdir(exist_ok=True)
    run("moe_combined_predictions", lambda: _moe_demo(ds, moe_dir))

    # --- Per-fold threshold stability (deployment-equivalent) ----------------
    run("per_fold_thresholds", lambda: _per_fold_thresholds_demo(ds))

    # --- Configuration comparison ---------------------------------------------
    run("compare_configs_table", lambda: _compare_configs_demo())


def _pairwise_wrapper(ds, class_positive: str, class_negative: str, out_dir):
    """Run pairwise end-to-end; return a thumbnail of the ROC for the gallery."""
    import matplotlib.pyplot as plt
    from remo_id.analysis.pairwise import run_pairwise

    run_pairwise(
        ds.y_true, ds.y_probs, ds.CLASS_NAMES,
        class_positive=class_positive, class_negative=class_negative,
        output_dir=str(out_dir), bootstrap_n=200,
    )
    fig, ax = style.figure(width="single", aspect=1.0)
    ax.imshow(plt.imread(str(out_dir / "roc_curve.png")))
    ax.axis("off")
    ax.set_title(f"Pairwise {class_positive} vs {class_negative}")
    return fig


def _moe_demo(ds, out_dir):
    """Fabricate a two-model MOE scenario using the dummy dataset.

    Model A (binary gate): infection+ vs not-infection, using dummy probs.
    Model B (expert): disease+/healthy/other 3-class, renormalized from ds.y_probs
    excluding the infection+ column. Produces combined 4-class output.
    """
    import numpy as np
    import pandas as pd
    from remo_id.analysis.moe import combine_moe_probs
    from remo_id.analysis.metrics import compute_multiclass_metrics
    import matplotlib.pyplot as plt

    MAL_IDX = ds.CLASS_NAMES.index("infection+")
    expert_idx = [i for i in range(ds.n_classes) if i != MAL_IDX]
    expert_names = [ds.CLASS_NAMES[i] for i in expert_idx]

    p_mal = ds.y_probs[:, MAL_IDX]
    p_a = np.column_stack([1 - p_mal, p_mal])

    p_b = ds.y_probs[:, expert_idx]
    p_b = p_b / p_b.sum(axis=1, keepdims=True).clip(min=1e-12)

    combined = combine_moe_probs(p_a, p_b, gate_positive_idx=1)
    y_pred = combined.argmax(axis=1)

    combined_class_names = ["infection+"] + expert_names
    y_true_remapped = np.array([
        0 if c == MAL_IDX else 1 + expert_idx.index(c) for c in ds.y_true
    ])

    metrics = compute_multiclass_metrics(
        y_true_remapped, y_pred,
        num_classes=len(combined_class_names),
        y_probs=combined,
    )

    out_dir.mkdir(exist_ok=True)
    rows = {"experiment_id": ds.experiment_ids, "y_pred": y_pred,
            "y_true_remapped": y_true_remapped}
    for k, name in enumerate(combined_class_names):
        rows[f"p_{name}"] = combined[:, k]
    pd.DataFrame(rows).to_csv(out_dir / "combined_predictions.csv", index=False)

    fig, ax = style.figure(width="single", aspect=0.9)
    cm = metrics["confusion_matrix"]
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap="Greys", vmin=0, vmax=1)
    n = len(combined_class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(combined_class_names, rotation=45, ha="right")
    ax.set_yticklabels(combined_class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"MOE 4-class CM (F1_macro={metrics['f1_macro']:.2f})")
    for i in range(n):
        for j in range(n):
            c = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", color=c)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _per_fold_thresholds_demo(ds):
    """Fabricate 10 CV folds and run per-fold threshold selection."""
    import numpy as np
    from remo_id.analysis.optimal_thresholds import (
        find_thresholds_per_fold, build_sensitivity_targets,
    )
    from remo_id.reports.figures import plot_per_fold_thresholds

    rng = np.random.default_rng(123)
    n_per_fold = max(20, ds.n_experiments // 3)
    per_fold = []
    for f in range(10):
        y_true = rng.choice(ds.n_classes, size=n_per_fold,
                            p=np.array(ds.CLASS_DIST) / sum(ds.CLASS_DIST))
        y_probs = np.zeros((n_per_fold, ds.n_classes))
        for i, c in enumerate(y_true):
            logits = rng.normal(size=ds.n_classes) * 0.45
            logits[c] += 1.4 + rng.normal(0, 0.35)
            y_probs[i] = np.exp(logits) / np.exp(logits).sum()
        per_fold.append({
            "repeat": 0, "fold": f,
            "y_true": y_true, "y_probs": y_probs,
            "y_pred": y_probs.argmax(1),
        })

    targets = build_sensitivity_targets(ds.CLASS_NAMES, 0.90, None)
    per_fold_df = find_thresholds_per_fold(
        per_fold, ds.CLASS_NAMES, rule="sens_constrained",
        sensitivity_targets=targets,
    )
    return plot_per_fold_thresholds(per_fold_df, target_class=ds.CLASS_NAMES[0])


def _compare_configs_demo():
    """Fabricate a configuration-comparison table with 6 example configs."""
    import pandas as pd
    rng_rows = [
        ("three_class", "multiclass", 3, 120, 0.95, 0.88),
        ("binary", "binary", 2, 120, 0.93, 0.86),
        ("three_class_one_scope", "multiclass", 3, 60, 0.91, 0.82),
        ("binary_one_scope", "binary", 2, 60, 0.90, 0.81),
        ("binary_other_condition", "binary", 2, 300, 0.84, 0.78),
        ("four_class", "multiclass", 4, 420, 0.83, 0.55),
    ]
    rows = [
        {"config": name, "task": task, "n_classes": k, "n_experiments": n, "n_folds": 10,
         "macro_auc": f"{auc:.3f} [{auc - 0.04:.3f}\u2013{min(auc + 0.03, 1):.3f}]",
         "macro_f1": f"{f1:.3f} [{f1 - 0.08:.3f}\u2013{min(f1 + 0.06, 1):.3f}]",
         "weighted_f1": f"{f1 + 0.02:.3f} [{f1 - 0.06:.3f}\u2013{min(f1 + 0.07, 1):.3f}]",
         "error_rate": f"{1 - f1 - 0.05:.3f}", "fold_f1_std": "0.150"}
        for name, task, k, n, auc, f1 in rng_rows
    ]
    return pd.DataFrame(rows)


def main(output_dir: str | None = None):
    outdir = Path(output_dir) if output_dir else Path(__file__).parent / "figure_output"
    outdir.mkdir(parents=True, exist_ok=True)

    ds = DummyDataset(seed=42)
    saved = []
    errors = []

    def run(name, fn):
        try:
            result = fn()
            if isinstance(result, plt.Figure):
                p = save(result, outdir, name)
                saved.append(name)
                print(f"  OK  {name} -> {p.name}")
            else:
                # metrics_table_with_ci returns DataFrame
                csv_path = outdir / f"{name}.csv"
                result.to_csv(csv_path, index=False)
                saved.append(name)
                print(f"  OK  {name} -> {csv_path.name}")
        except Exception as e:
            errors.append((name, e))
            print(f"  FAIL {name}: {e}")

    print(f"Generating figures to {outdir}/\n")

    # === Classification (multiclass) ===
    run("roc_curves_multiclass", lambda: plot_roc_curves(
        ds.y_true, ds.y_probs, ds.CLASS_NAMES, bootstrap_n=200))
    run("pr_curves_multiclass", lambda: plot_pr_curves(
        ds.y_true, ds.y_probs, ds.CLASS_NAMES, bootstrap_n=200))
    run("confusion_matrix", lambda: plot_confusion_matrix(
        ds.y_true, ds.y_pred, ds.CLASS_NAMES))
    run("calibration_multiclass", lambda: plot_calibration(
        ds.y_true, ds.y_probs, ds.CLASS_NAMES))
    run("prob_distributions_multiclass", lambda: plot_prob_distributions(
        ds.y_true, ds.y_probs, ds.CLASS_NAMES))
    run("metrics_table_multiclass", lambda: metrics_table_with_ci(
        ds.y_true, ds.y_probs, ds.CLASS_NAMES, n=200))

    # === Classification (binary) ===
    run("roc_curves_binary", lambda: plot_roc_curves(
        ds.y_true_binary, ds.y_probs_binary, ds.binary_class_names,
        bootstrap_n=200, task_type="binary"))
    run("pr_curves_binary", lambda: plot_pr_curves(
        ds.y_true_binary, ds.y_probs_binary, ds.binary_class_names,
        bootstrap_n=200, task_type="binary"))
    run("confusion_matrix_binary", lambda: plot_confusion_matrix(
        ds.y_true_binary, ds.y_pred_binary, ds.binary_class_names))
    run("calibration_binary", lambda: plot_calibration(
        ds.y_true_binary, ds.y_probs_binary, ds.binary_class_names,
        task_type="binary"))
    run("prob_distributions_binary", lambda: plot_prob_distributions(
        ds.y_true_binary, ds.y_probs_binary, ds.binary_class_names,
        task_type="binary"))
    run("metrics_table_binary", lambda: metrics_table_with_ci(
        ds.y_true_binary, ds.y_probs_binary, ds.binary_class_names,
        n=200, task_type="binary"))

    # === Comparison ===
    run("seed_variability", lambda: plot_seed_variability(ds.metrics_by_seed))
    run("threshold_sensitivity", lambda: plot_threshold_sensitivity(
        ds.y_true, ds.y_probs, ds.CLASS_NAMES))
    run("operating_points", lambda: plot_operating_points(
        ds.y_true, ds.y_probs, ds.CLASS_NAMES))
    run("kfold_variability", lambda: plot_kfold_variability(
        ds.fold_metrics, ds.CLASS_NAMES))
    run("baseline_comparison", lambda: plot_baseline_comparison(ds.baseline_df))
    run("optimal_thresholds", lambda: plot_optimal_thresholds(
        ds.y_true, ds.y_probs, ds.CLASS_NAMES, ds.optimal_results,
        target_class=ds.CLASS_NAMES[0]))

    # === Binary post-training plots ===
    binary_splits = [ds.splits[i] for i in range(len(ds.splits)) if ds._binary_mask[i]]
    binary_studies = [ds.study_ids[i] for i in range(len(ds.study_ids)) if ds._binary_mask[i]]

    run("experiment_probabilities", lambda: plot_experiment_probabilities(
        ds.mean_probs, ds.std_probs, ds.y_true_binary,
        binary_splits, binary_studies,
        class_names=ds.binary_class_names, model_type="resnet18-mil",
        n_runs=5, n_images=200))

    # ROC binary splits: use all vs val-only
    val_mask = [s == "val" for s in binary_splits]
    import numpy as np
    val_mask_arr = np.array(val_mask)
    run("roc_binary_splits", lambda: plot_roc_binary_splits(
        ds.y_true_binary, ds.y_probs_binary[:, 1],
        ds.y_true_binary[val_mask_arr], ds.y_probs_binary[val_mask_arr, 1],
        model_type="resnet18-mil", n_runs=5, n_images=200))

    run("repeatability_hist", lambda: plot_repeatability_hist(
        ds.std_probs, model_type="resnet18-mil", n_runs=5, n_images=200))

    # === Data ===
    run("bag_size_sweep", lambda: plot_bag_size_sweep(ds.bag_sizes, ds.aucs_by_seed))
    run("sampling_position", lambda: plot_sampling_position(
        ds.positions, ds.position_aucs, bag_size=200))
    run("learning_curve", lambda: plot_learning_curve(ds.learning_curve_df))
    run("class_distribution", lambda: plot_class_distribution(
        ds.imbalance, ds.CLASS_NAMES))

    # === Error analysis ===
    run("error_confidence", lambda: plot_error_confidence(
        ds.y_true, ds.y_pred, ds.y_probs, ds.CLASS_NAMES))
    run("error_by_study", lambda: plot_error_by_study(ds.error_df, ds.CLASS_NAMES))
    run("confusion_errors", lambda: plot_confusion_errors(ds.error_df, ds.CLASS_NAMES))

    # === Interpretation ===
    run("top_attention_grid", lambda: plot_top_attention_grid(
        ds.frames_by_class, ds.CLASS_NAMES))
    run("umap", lambda: plot_umap(
        ds.embeddings, ds.embedding_labels, ds.CLASS_NAMES))
    run("attention_entropy", lambda: plot_attention_entropy(
        ds.entropy_by_class, ds.CLASS_NAMES))

    # === Threshold / pairwise / MOE analyses ===
    _run_paper_analyses(ds, outdir, run)

    # === Summary ===
    print(f"\n{'='*50}")
    print(f"Generated: {len(saved)}/{len(saved) + len(errors)}")
    if errors:
        print(f"\nFailed ({len(errors)}):")
        for name, e in errors:
            print(f"  {name}: {e}")
    else:
        print("All figures generated successfully!")
    print(f"\nOutput directory: {outdir}")

    return len(errors) == 0


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else None
    success = main(output)
    sys.exit(0 if success else 1)
