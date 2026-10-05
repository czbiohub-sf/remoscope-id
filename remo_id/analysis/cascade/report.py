"""Orchestrator: parse input, infer tree, load predictions, check coverage,
walk paths, compute metrics, write artifacts and figures, print summary."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from remo_id.analysis.cascade.coverage import verify_strict_coverage
from remo_id.analysis.cascade.figures import (
    plot_cascade_tree, plot_per_depth_accuracy,
)
from remo_id.analysis.cascade.loader import ModelPredictions, load_model_predictions
from remo_id.analysis.cascade.metrics import (
    error_attribution, leaf_level_metrics, per_depth_accuracy, per_node_metrics,
)
from remo_id.analysis.cascade.models import parse_cascade_models
from remo_id.analysis.cascade.probabilities import compute_leaf_probabilities
from remo_id.analysis.cascade.routing import (
    walk_ground_truth_path, walk_predicted_path,
)
from remo_id.analysis.cascade.scope import compute_model_scopes
from remo_id.analysis.cascade.tree import infer_tree, render_tree_ascii


def run_cascade(
    models_yaml: str | None,
    cli_models: list[str],
    split: str = "val",
    output: str | None = None,
    bootstrap_n: int = 1000,
    print_tree: bool = False,
    target_sensitivity: dict | None = None,
    default_sensitivity: float = 0.95,
) -> dict:
    """End-to-end cascade analysis. Returns a summary dict and writes
    artifacts to the resolved output directory.

    The sensitivity-routing walk thresholds each node's target class at
    ``target_sensitivity[class_name]`` (default ``default_sensitivity``).
    """
    cascade = parse_cascade_models(
        yaml_path=models_yaml,
        cli_model_args=cli_models,
        cli_split=split,
        cli_bootstrap_n=bootstrap_n,
    )
    for m in cascade.models:
        m.split = cascade.split

    print(f"\n[1/6] Computing model scopes from {len(cascade.models)} class_configs…")
    scopes = compute_model_scopes(cascade.models)

    print("[2/6] Inferring cascade tree…")
    tree = infer_tree(scopes, cascade.models)
    tree_ascii = render_tree_ascii(
        tree, counts={nid: len(scopes[nid].scope) for nid in scopes},
    )
    print("\n" + tree_ascii + "\n")

    if print_tree:
        return {"name": cascade.name, "tree": tree_ascii, "leaves": tree.leaves}

    print("[3/6] Loading predictions per node…")
    model_preds: dict[str, ModelPredictions] = {}
    for m in cascade.models:
        preds = load_model_predictions(m, split=cascade.split)
        print(f"    {m.short_name}: {len(preds.df)} preds ({preds.source_info})")
        model_preds[m.id] = preds
        _check_class_name_alignment(m.id, preds.class_names, scopes[m.id].class_names)

    print("[4/6] Verifying strict coverage…")
    coverage_rows = []
    for nid in scopes:
        report = verify_strict_coverage(
            model_id=nid, scope=scopes[nid], preds=model_preds[nid],
            split=cascade.split,
        )
        coverage_rows.append({
            "node_id": nid,
            "short_name": cascade.models[_index_of(cascade.models, nid)].short_name,
            "in_scope": report.in_scope,
            "found": report.found,
            "missing": len(report.missing),
        })

    print("[5/6] Walking predicted & ground-truth paths…")
    pred_walk = walk_predicted_path(tree, model_preds)
    gt_walk = walk_ground_truth_path(tree, model_preds)

    print("[6/8] Computing metrics (Youden-J routing)…")
    per_node = per_node_metrics(tree, model_preds, pred_walk)
    leaf = leaf_level_metrics(
        pred_walk, gt_walk, canonical_leaves=tree.leaves,
        bootstrap_n=cascade.bootstrap_n,
    )
    depth_acc = per_depth_accuracy(pred_walk, gt_walk)
    err_attr = error_attribution(pred_walk, gt_walk)

    print("[7/8] Sensitivity-threshold routing walk…")
    sens_preds = _apply_sensitivity_thresholds(
        model_preds, tree, target_sensitivity or {}, default_sensitivity)
    sens_walk = walk_predicted_path(tree, sens_preds)
    sens_leaf = leaf_level_metrics(
        sens_walk, gt_walk, canonical_leaves=tree.leaves,
        bootstrap_n=cascade.bootstrap_n,
    )

    print("[8/8] Computing leaf probabilities for figures…")
    leaf_y_true, leaf_y_probs, _, leaf_names, _ = (
        compute_leaf_probabilities(tree, model_preds, gt_walk)
    )

    # Apply display name mapping to all leaf-level outputs
    dn = cascade.display_names or {}
    if dn:
        leaf = _rename_leaf_dict(leaf, dn)
        sens_leaf = _rename_leaf_dict(sens_leaf, dn) if sens_leaf else None
        leaf_names = [dn.get(n, n) for n in leaf_names] if leaf_names else None

    out_dir = _resolve_output_dir(output, cascade)
    os.makedirs(out_dir, exist_ok=True)
    _write_outputs(
        out_dir=out_dir, tree=tree, tree_ascii=tree_ascii,
        pred_walk=pred_walk, gt_walk=gt_walk,
        per_node=per_node, leaf=leaf, depth_acc=depth_acc,
        err_attr=err_attr, coverage=coverage_rows,
        model_preds=model_preds,
        sens_leaf=sens_leaf,
        leaf_y_true=leaf_y_true, leaf_y_probs=leaf_y_probs,
        leaf_names=leaf_names,
        bootstrap_n=cascade.bootstrap_n,
    )

    _print_summary(cascade.name, leaf, depth_acc, err_attr, out_dir)

    return {
        "name": cascade.name,
        "output": out_dir,
        "leaf_metrics": leaf,
        "per_depth": depth_acc.to_dict(orient="records"),
    }


def _apply_sensitivity_thresholds(
    model_preds: dict[str, ModelPredictions],
    tree,
    target_sensitivity: dict,
    default_sensitivity: float = 0.95,
) -> dict[str, ModelPredictions]:
    """Create a copy of model_preds with sensitivity-optimized thresholds.

    For each node, the sensitivity target class is the class that routes
    deeper into the tree (i.e. has a child node, not a leaf). This ensures
    we maximise sensitivity for the class that needs further classification.
    If all children are leaves, the first class listed in
    ``target_sensitivity`` is used (else class 0). The required sensitivity is
    ``target_sensitivity.get(class_name, default_sensitivity)``.
    """
    import copy
    import numpy as np
    from sklearn.metrics import roc_curve

    out: dict[str, ModelPredictions] = {}
    for nid, preds in model_preds.items():
        new_preds = copy.deepcopy(preds)
        if "y_probs_all" not in new_preds.df.columns:
            out[nid] = new_preds
            continue

        y_true = new_preds.df["y_true"].astype(int).to_numpy()
        y_probs = np.asarray(new_preds.df["y_probs_all"].tolist(), dtype=float)

        # Determine the target class for sensitivity optimisation
        target_idx = _find_sensitivity_target(nid, preds.class_names, tree, target_sensitivity)
        target_name = preds.class_names[target_idx]
        target_sens = float(target_sensitivity.get(target_name, default_sensitivity))

        p_target = y_probs[:, target_idx]
        y_bin = (y_true == target_idx).astype(int)
        fpr, tpr, thresholds = roc_curve(y_bin, p_target)

        meets = tpr >= target_sens
        if meets.any():
            # First index meeting target = highest threshold = best specificity
            idx = np.where(meets)[0][0]
        else:
            idx = 0
        thr = float(thresholds[idx])

        # Binary: threshold on target class probability
        if len(preds.class_names) == 2:
            other_idx = 1 - target_idx
            new_preds.df["y_pred"] = np.where(p_target >= thr, target_idx, other_idx)
        else:
            # Multiclass: only adjust the target class threshold, keep argmax for others
            y_pred = y_probs.argmax(axis=1)
            y_pred[p_target >= thr] = target_idx
            new_preds.df["y_pred"] = y_pred

        achieved_sens = float(tpr[idx])
        achieved_spec = float(1.0 - fpr[idx])
        print(
            f"    Sensitivity target ({preds.model.short_name}): "
            f"{target_name}>={thr:.3f} → sens={achieved_sens:.3f}, spec={achieved_spec:.3f}"
        )
        out[nid] = new_preds
    return out


def _find_sensitivity_target(
    node_id: str, class_names: list[str], tree, target_sensitivity: dict | None = None,
) -> int:
    """Find the class index to target for sensitivity optimisation.

    Priority: class that routes to a child node (not a leaf) > first class
    named in ``target_sensitivity`` > class index 0.
    """
    node = tree.nodes_by_id.get(node_id)
    if node is not None:
        for i, name in enumerate(class_names):
            target = node.children.get(name)
            if target is not None and not isinstance(target, str):
                return i
    for name in target_sensitivity or {}:
        if name in class_names:
            return class_names.index(name)
    return 0


def _rename_leaf_dict(d: dict, mapping: dict[str, str]) -> dict:
    """Rename leaf names in a leaf_level_metrics dict."""
    out = dict(d)
    if "leaves" in out:
        out["leaves"] = [mapping.get(n, n) for n in out["leaves"]]
    return out


def _check_class_name_alignment(
    model_id: str, preds_names: list[str], scope_names: list[str],
) -> None:
    if list(preds_names) != list(scope_names):
        print(
            f"  WARNING: class_name drift for {model_id}: "
            f"predictions={preds_names} vs class_config={scope_names}. "
            f"Using class_config ordering."
        )


def _index_of(models, node_id: str) -> int:
    for i, m in enumerate(models):
        if m.id == node_id:
            return i
    raise KeyError(node_id)


def _resolve_output_dir(output: str | None, cascade) -> str:
    if output:
        return output
    if cascade.source_path:
        return str(Path(cascade.source_path).parent / "analysis" / "cascade" / cascade.name)
    return str(Path(cascade.models[0].checkpoint).parent / "analysis" / "cascade" / cascade.name)


def _write_outputs(
    out_dir: str, tree, tree_ascii: str,
    pred_walk: pd.DataFrame, gt_walk: pd.DataFrame,
    per_node: dict, leaf: dict, depth_acc: pd.DataFrame,
    err_attr: pd.DataFrame, coverage: list[dict],
    model_preds: dict,
    sens_leaf: dict | None = None,
    leaf_y_true=None, leaf_y_probs=None, leaf_names=None,
    bootstrap_n: int = 1000,
) -> None:
    with open(os.path.join(out_dir, "inferred_tree.txt"), "w") as f:
        f.write(tree_ascii + "\n")

    pd.DataFrame(coverage).to_csv(
        os.path.join(out_dir, "coverage_report.csv"), index=False,
    )

    node_rows = []
    for nid, m in per_node.items():
        for view, vals in (("all", m["all"]), ("routed_only", m["routed_only"])):
            row = {"node_id": nid, "view": view}
            row.update({k: v for k, v in vals.items() if not isinstance(v, list)})
            node_rows.append(row)
    pd.DataFrame(node_rows).to_csv(
        os.path.join(out_dir, "per_node_metrics.csv"), index=False,
    )

    with open(os.path.join(out_dir, "leaf_metrics.json"), "w") as f:
        json.dump(leaf, f, indent=2, default=_json_default)

    depth_acc.to_csv(os.path.join(out_dir, "per_depth_accuracy.csv"), index=False)
    err_attr.to_csv(os.path.join(out_dir, "error_attribution.csv"), index=False)

    routing = pred_walk.merge(gt_walk, on="experiment_id", how="inner")
    try:
        routing.to_parquet(os.path.join(out_dir, "routing_table.parquet"), index=False)
    except ImportError:
        routing.to_csv(os.path.join(out_dir, "routing_table.csv"), index=False)

    routed_counts = {}
    routed_accs = {}
    for nid in tree.nodes_by_id:
        view = per_node[nid]["routed_only"]
        routed_counts[nid] = per_node[nid]["n_routed"]
        routed_accs[nid] = view.get("accuracy", float("nan"))
    figs_dir = os.path.join(out_dir, "figures")
    os.makedirs(figs_dir, exist_ok=True)

    fig = plot_cascade_tree(tree, routed_counts, routed_accs)
    fig.savefig(os.path.join(figs_dir, "cascade_tree.png"), dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)

    if not depth_acc.empty:
        fig = plot_per_depth_accuracy(depth_acc)
        fig.savefig(os.path.join(figs_dir, "per_depth_accuracy.png"), dpi=200)
        plt.close(fig)

    _leaf_confusion_plot(leaf, figs_dir, title="Leaf Confusion (Youden-J)")
    _leaf_performance_table(leaf, os.path.join(out_dir, "table_youden.csv"), bootstrap_n)

    if sens_leaf is not None:
        _leaf_confusion_plot(
            sens_leaf, figs_dir,
            title="Leaf Confusion (sensitivity)",
            filename="confusion_sensitivity.png",
        )
        _leaf_performance_table(sens_leaf, os.path.join(out_dir, "table_sensitivity.csv"), bootstrap_n)

    if leaf_y_true is not None and leaf_y_probs is not None:
        _leaf_probability_figures(
            figs_dir, leaf_y_true, leaf_y_probs, leaf_names, bootstrap_n,
        )


def _leaf_performance_table(
    leaf: dict, output_path: str, bootstrap_n: int = 1000,
) -> None:
    """Build a per-class performance table from the routing walk confusion matrix,
    with bootstrap CIs.  Mirrors the format of per-node LOO table_youden.csv."""
    import numpy as np

    cm = np.asarray(leaf["confusion_matrix"], dtype=int)
    leaves = leaf["leaves"]
    n_classes = len(leaves)
    n_total = int(cm.sum())

    # Reconstruct y_true, y_pred from confusion matrix for bootstrapping
    y_true_list, y_pred_list = [], []
    for i in range(n_classes):
        for j in range(n_classes):
            y_true_list.extend([i] * cm[i, j])
            y_pred_list.extend([j] * cm[i, j])
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    def _per_class_metrics(yt, yp):
        rows = []
        for c in range(n_classes):
            tp = int(((yp == c) & (yt == c)).sum())
            fp = int(((yp == c) & (yt != c)).sum())
            fn = int(((yp != c) & (yt == c)).sum())
            tn = int(((yp != c) & (yt != c)).sum())
            sens = tp / max(1, tp + fn)
            spec = tn / max(1, tn + fp)
            ppv = tp / max(1, tp + fp)
            npv = tn / max(1, tn + fn)
            f1 = 2 * ppv * sens / max(1e-12, ppv + sens)
            acc = (tp + tn) / max(1, tp + fp + fn + tn)
            rows.append({
                "support": tp + fn, "f1": f1, "accuracy": acc,
                "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
            })
        return rows

    point = _per_class_metrics(y_true, y_pred)

    # Bootstrap CIs
    rng = np.random.default_rng(0)
    n = len(y_true)
    boot = {k: [[] for _ in range(n_classes)] for k in ["f1", "accuracy", "sensitivity", "specificity", "ppv", "npv"]}
    for _ in range(bootstrap_n):
        idx = rng.integers(0, n, size=n)
        bm = _per_class_metrics(y_true[idx], y_pred[idx])
        for c in range(n_classes):
            for k in boot:
                boot[k][c].append(bm[c][k])

    def fmt(val, lo, hi):
        return f"{val:.3f} [{lo:.3f}\u2013{hi:.3f}]"

    rows = []
    for c in range(n_classes):
        row = {"Class": leaves[c], "Support": point[c]["support"]}
        for k in ["f1", "accuracy", "sensitivity", "specificity", "ppv", "npv"]:
            lo = float(np.percentile(boot[k][c], 2.5))
            hi = float(np.percentile(boot[k][c], 97.5))
            row[k.upper() if k != "ppv" else "PPV" if k == "ppv" else k] = fmt(point[c][k], lo, hi)
        rows.append(row)

    # Macro row
    macro = {"Class": "macro", "Support": n_total}
    for k in ["f1", "accuracy", "sensitivity", "specificity", "ppv", "npv"]:
        val = np.mean([point[c][k] for c in range(n_classes)])
        boot_macro = [np.mean([boot[k][c][i] for c in range(n_classes)]) for i in range(bootstrap_n)]
        lo = float(np.percentile(boot_macro, 2.5))
        hi = float(np.percentile(boot_macro, 97.5))
        macro[k.upper() if k != "ppv" else "PPV" if k == "ppv" else k] = fmt(val, lo, hi)
    rows.append(macro)

    df = pd.DataFrame(rows)
    # Normalise column names
    df.columns = [c.replace("f1", "F1").replace("accuracy", "Accuracy")
                   .replace("sensitivity", "Sensitivity").replace("specificity", "Specificity")
                   .replace("ppv", "PPV").replace("npv", "NPV") for c in df.columns]
    df.to_csv(output_path, index=False)
    print(f"  Saved {output_path}")


def _leaf_confusion_plot(
    leaf: dict, figs_dir: str,
    title: str = "Leaf-level confusion matrix",
    filename: str = "leaf_confusion_matrix.png",
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from remo_id.reports import style

    style.apply()
    cm = np.asarray(leaf["confusion_matrix"], dtype=int)
    leaves = leaf["leaves"]
    n = len(leaves)
    fig, ax = style.figure(width="double", aspect=1.0)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sums
    im = ax.imshow(cm_norm, cmap="Greys", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    style.thin_colorbar(cbar)
    style.strip_imshow_frame(ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(leaves, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(leaves, fontsize=6)
    ax.set_xlabel("Predicted leaf", fontsize=7)
    ax.set_ylabel("True leaf", fontsize=7)
    ax.set_title(title, fontsize=7)
    for i in range(n):
        for j in range(n):
            text_color = "white" if cm_norm[i, j] > 0.55 else style.FG
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", color=text_color, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, filename), dpi=200)
    plt.close(fig)


def _leaf_probability_figures(
    figs_dir: str,
    y_true, y_probs, leaf_names: list[str],
    bootstrap_n: int,
) -> None:
    """ROC, PR, calibration, probability distributions from leaf probability
    vectors (chain-rule scores).  These are threshold-independent ranking
    metrics so the leaf probabilities are valid scores."""
    import matplotlib.pyplot as plt
    import numpy as np

    from remo_id.reports.figures.classification import (
        plot_roc_curves,
        plot_pr_curves,
        plot_calibration,
        plot_prob_distributions,
    )

    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    n_classes = len(leaf_names)
    task = "binary" if n_classes == 2 else "multiclass"

    valid = y_true >= 0
    if valid.sum() < len(y_true):
        print(f"  Excluding {(~valid).sum()} unmapped experiments from figures")
        y_true = y_true[valid]
        y_probs = y_probs[valid]

    if len(y_true) == 0:
        return

    plots = [
        ("roc_curves.png", lambda: plot_roc_curves(y_true, y_probs, leaf_names, bootstrap_n=bootstrap_n, task_type=task)),
        ("pr_curves.png", lambda: plot_pr_curves(y_true, y_probs, leaf_names, bootstrap_n=bootstrap_n, task_type=task)),
        ("calibration.png", lambda: plot_calibration(y_true, y_probs, leaf_names, task_type=task)),
        ("prob_distributions.png", lambda: plot_prob_distributions(y_true, y_probs, leaf_names, task_type=task)),
    ]

    for fname, plot_fn in plots:
        try:
            fig = plot_fn()
            fig.savefig(os.path.join(figs_dir, fname), dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"  Warning: could not generate {fname}: {e}")


def _json_default(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")


def _print_summary(
    name: str, leaf: dict, depth_acc: pd.DataFrame,
    err_attr: pd.DataFrame, out_dir: str,
) -> None:
    print(f"\n=== Cascade '{name}' summary ===")
    pe = leaf["point_estimates"]
    ci = leaf.get("ci_95", {})
    def fmt(k):
        if k in ci and ci[k]:
            lo, hi = ci[k]
            return f"{pe[k]:.3f}  [95% CI {lo:.3f}, {hi:.3f}]"
        return f"{pe[k]:.3f}"
    print(f"Leaves               : {leaf['leaves']}")
    print(f"Evaluated experiments: {leaf['n']}")
    print(f"Cascade rejected     : {leaf['n_cascade_rejected']} ({100 * leaf['cascade_rejection_rate']:.1f}%)")
    print(f"Leaf accuracy        : {fmt('accuracy')}")
    print(f"Balanced accuracy    : {fmt('balanced_accuracy')}")
    print(f"Macro F1             : {fmt('f1_macro')}")

    if not depth_acc.empty:
        print("\nPer-depth accuracy:")
        for _, r in depth_acc.iterrows():
            print(f"  depth {int(r['depth'])}: acc={r['accuracy']:.3f}  (n={int(r['n'])})")

    if not err_attr.empty:
        wrong = err_attr[~err_attr["correct"]]
        if not wrong.empty:
            top = (
                wrong.groupby("first_error_node").size()
                .sort_values(ascending=False).head(5)
            )
            print("\nTop error-attribution nodes:")
            for node, count in top.items():
                print(f"  {node}: {int(count)} misclassifications")

    print(f"\nOutputs written to: {out_dir}")
