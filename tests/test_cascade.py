"""Tests for the cascade analysis pipeline.

Exercises tree inference, path walks, coverage validation, and metrics
against in-memory fixtures — no database, no torch.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from _helpers_cascade import (  # noqa: E402
    build_three_node_cascade, fake_predictions, fake_scope,
)

from remo_id.analysis.cascade.coverage import verify_strict_coverage
from remo_id.analysis.cascade.metrics import (
    error_attribution, leaf_level_metrics, per_depth_accuracy, per_node_metrics,
)
from remo_id.analysis.cascade.routing import (
    walk_ground_truth_path, walk_predicted_path,
)
from remo_id.analysis.cascade.tree import infer_tree, render_tree_ascii


# ---------------------------------------------------------------------------
# Tree inference
# ---------------------------------------------------------------------------

def test_infer_tree_happy_path():
    f = build_three_node_cascade()
    tree = infer_tree(f["scopes"], f["models"])
    assert tree.root.id == "root"
    assert set(tree.leaves) == {"healthy", "disease", "control"}
    healthy_target = tree.root.children["healthy"]
    sick_target = tree.root.children["sick"]
    assert isinstance(healthy_target, str) and healthy_target == "healthy"
    assert sick_target.id == "child"
    assert sick_target.children["disease"] == "disease"
    assert sick_target.children["control"] == "control"
    assert tree.node_depth["root"] == 0
    assert tree.node_depth["child"] == 1


def test_infer_tree_rejects_ambiguous_parent():
    f = build_three_node_cascade()
    f["scopes"]["child"].class_experiments["disease"] = set()
    f["scopes"]["child"].scope = set(f["sick"])
    ambiguous_model = f["models"][0].__class__(
        checkpoint="other", class_config="other.yaml",
        predictions_kind="single", predictions_dir="/tmp",
    )
    ambiguous_scope = fake_scope(
        "other", ["a", "b"],
        {"a": set(f["sick"]), "b": set()},  # same scope as root.sick
    )
    scopes = {**f["scopes"], "other": ambiguous_scope}
    models = f["models"] + [ambiguous_model]
    with pytest.raises(ValueError, match="Ambiguous parent"):
        infer_tree(scopes, models)


def test_infer_tree_rejects_disconnected():
    f = build_three_node_cascade()
    orphan_model = f["models"][0].__class__(
        checkpoint="orphan", class_config="orphan.yaml",
        predictions_kind="single", predictions_dir="/tmp",
    )
    orphan_scope = fake_scope(
        "orphan", ["p", "q"],
        {"p": {"zzz0"}, "q": {"zzz1"}},  # not a subset of any other
    )
    scopes = {**f["scopes"], "orphan": orphan_scope}
    with pytest.raises(ValueError, match="exactly one root|not in root"):
        infer_tree(scopes, f["models"] + [orphan_model])


def test_render_tree_ascii_contains_all_leaves():
    f = build_three_node_cascade()
    tree = infer_tree(f["scopes"], f["models"])
    s = render_tree_ascii(tree, counts={"root": 10, "child": 5})
    assert "healthy" in s and "disease" in s and "control" in s
    assert "n=10" in s and "n=5" in s


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _perfect_predictions(f):
    root_gt = {eid: 0 for eid in f["healthy"]} | {eid: 1 for eid in f["sick"]}
    child_gt = {eid: 0 for eid in f["disease"]} | {eid: 1 for eid in f["control"]}
    root_preds = fake_predictions(f["models"][0], root_gt, root_gt, ["healthy", "sick"])
    child_preds = fake_predictions(f["models"][1], child_gt, child_gt, ["disease", "control"])
    return {"root": root_preds, "child": child_preds}


def test_gt_walk_reaches_expected_leaves():
    f = build_three_node_cascade()
    tree = infer_tree(f["scopes"], f["models"])
    model_preds = _perfect_predictions(f)
    gt = walk_ground_truth_path(tree, model_preds).set_index("experiment_id")
    for eid in f["healthy"]:
        assert gt.loc[eid, "gt_terminal_leaf"] == "healthy"
    for eid in f["disease"]:
        assert gt.loc[eid, "gt_terminal_leaf"] == "disease"
    for eid in f["control"]:
        assert gt.loc[eid, "gt_terminal_leaf"] == "control"


def test_pred_walk_flags_false_positive_rejection():
    f = build_three_node_cascade()
    tree = infer_tree(f["scopes"], f["models"])
    root_gt = {eid: 0 for eid in f["healthy"]} | {eid: 1 for eid in f["sick"]}
    root_pred = dict(root_gt)
    root_pred["e0"] = 1  # false positive: healthy experiment predicted as sick
    child_gt = {eid: 0 for eid in f["disease"]} | {eid: 1 for eid in f["control"]}
    model_preds = {
        "root": fake_predictions(f["models"][0], root_gt, root_pred, ["healthy", "sick"]),
        "child": fake_predictions(f["models"][1], child_gt, child_gt, ["disease", "control"]),
    }
    pred = walk_predicted_path(tree, model_preds).set_index("experiment_id")
    assert pred.loc["e0", "pred_cascade_rejected"] is True or pred.loc["e0", "pred_cascade_rejected"] == True  # noqa: E712
    assert pred.loc["e0", "pred_terminal_leaf"].startswith("unresolved_at_")
    assert pred.loc["e0", "pred_rejection_node"] == "child"


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_strict_coverage_errors_on_missing_experiment():
    f = build_three_node_cascade()
    root_gt = {eid: 0 for eid in f["healthy"]} | {eid: 1 for eid in f["sick"]}
    preds = fake_predictions(f["models"][0], root_gt, root_gt, ["healthy", "sick"])
    preds.df = preds.df[preds.df["experiment_id"] != "e3"].reset_index(drop=True)
    with pytest.raises(ValueError, match="Strict coverage failed"):
        verify_strict_coverage(
            model_id="root", scope=f["scopes"]["root"],
            preds=preds, split="val",
        )


def test_strict_coverage_passes_when_complete():
    f = build_three_node_cascade()
    root_gt = {eid: 0 for eid in f["healthy"]} | {eid: 1 for eid in f["sick"]}
    preds = fake_predictions(f["models"][0], root_gt, root_gt, ["healthy", "sick"])
    report = verify_strict_coverage(
        model_id="root", scope=f["scopes"]["root"], preds=preds, split="val",
    )
    assert report.missing == set()
    assert report.found == 10


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_leaf_confusion_matches_expected_perfect():
    f = build_three_node_cascade()
    tree = infer_tree(f["scopes"], f["models"])
    model_preds = _perfect_predictions(f)
    pred_walk = walk_predicted_path(tree, model_preds)
    gt_walk = walk_ground_truth_path(tree, model_preds)
    leaf = leaf_level_metrics(pred_walk, gt_walk, tree.leaves, bootstrap_n=20)
    assert leaf["point_estimates"]["accuracy"] == pytest.approx(1.0)
    assert leaf["point_estimates"]["balanced_accuracy"] == pytest.approx(1.0)
    assert leaf["n_cascade_rejected"] == 0
    assert set(leaf["leaves"]) == {"healthy", "disease", "control"}


def test_per_depth_accuracy_monotone_on_perfect_data():
    f = build_three_node_cascade()
    tree = infer_tree(f["scopes"], f["models"])
    model_preds = _perfect_predictions(f)
    pred_walk = walk_predicted_path(tree, model_preds)
    gt_walk = walk_ground_truth_path(tree, model_preds)
    depth_df = per_depth_accuracy(pred_walk, gt_walk)
    assert (depth_df["accuracy"] == 1.0).all()
    assert depth_df.loc[depth_df["depth"] == 0, "n"].iloc[0] == 10
    assert depth_df.loc[depth_df["depth"] == 1, "n"].iloc[0] == 5


def test_error_attribution_pinpoints_first_divergence():
    f = build_three_node_cascade()
    tree = infer_tree(f["scopes"], f["models"])
    root_gt = {eid: 0 for eid in f["healthy"]} | {eid: 1 for eid in f["sick"]}
    child_gt = {eid: 0 for eid in f["disease"]} | {eid: 1 for eid in f["control"]}
    child_pred = dict(child_gt)
    child_pred["e5"] = 1   # disease → control: error at depth 1
    root_pred = dict(root_gt)
    root_pred["e0"] = 1    # healthy → sick: error at depth 0
    model_preds = {
        "root": fake_predictions(f["models"][0], root_gt, root_pred, ["healthy", "sick"]),
        "child": fake_predictions(f["models"][1], child_gt, child_pred, ["disease", "control"]),
    }
    pred_walk = walk_predicted_path(tree, model_preds)
    gt_walk = walk_ground_truth_path(tree, model_preds)
    err = error_attribution(pred_walk, gt_walk).set_index("experiment_id")
    assert err.loc["e0", "first_error_depth"] == 0
    assert err.loc["e0", "first_error_node"] == "root"
    assert err.loc["e5", "first_error_depth"] == 1
    assert err.loc["e5", "first_error_node"] == "child"
    for eid in f["control"]:
        assert bool(err.loc[eid, "correct"])


def test_per_node_metrics_all_and_routed_views():
    f = build_three_node_cascade()
    tree = infer_tree(f["scopes"], f["models"])
    model_preds = _perfect_predictions(f)
    pred_walk = walk_predicted_path(tree, model_preds)
    pn = per_node_metrics(tree, model_preds, pred_walk)
    assert pn["root"]["n_all"] == 10
    assert pn["root"]["n_routed"] == 10
    assert pn["child"]["n_all"] == 5
    assert pn["child"]["n_routed"] == 5
    assert pn["root"]["all"]["accuracy"] == pytest.approx(1.0)
    assert pn["child"]["all"]["accuracy"] == pytest.approx(1.0)
