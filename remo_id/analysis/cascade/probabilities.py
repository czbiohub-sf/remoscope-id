"""Compute leaf-level probability vectors via chain-rule propagation
through the cascade tree.

For each experiment, we walk the tree from root to leaves, multiplying
the incoming probability by each node's softmax output at each branch:

    P(leaf) = prod_{node on path to leaf} P_node(class leading to leaf)

Experiments missing from a child node (e.g. healthy experiments that were
not in that child's training set) get their incoming probability split
uniformly across all leaves reachable from that child.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from remo_id.analysis.cascade.loader import ModelPredictions
from remo_id.analysis.cascade.tree import CascadeNode, CascadeTree


def compute_leaf_probabilities(
    tree: CascadeTree,
    model_preds: dict[str, ModelPredictions],
    gt_walk: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Compute leaf-level probability vectors via chain rule.

    Returns
    -------
    y_true : 1-D int array [N]
        True leaf class index for each experiment.
    y_probs : 2-D float array [N, L]
        Leaf probability vector for each experiment (sums to ~1).
    y_pred : 1-D int array [N]
        Predicted leaf class index (argmax of y_probs).
    leaf_names : list[str]
        Leaf names in column order (matches tree.leaves).
    experiment_ids : list[str]
        Experiment IDs in row order.
    """
    leaves = tree.leaves
    n_leaves = len(leaves)
    leaf_to_idx = {leaf: i for i, leaf in enumerate(leaves)}

    # Build per-node probability lookups: node_id -> {eid -> prob_vector}
    node_probs: dict[str, dict[str, np.ndarray]] = {}
    for node_id, preds in model_preds.items():
        eid_col = preds.df["experiment_id"].astype(str).to_numpy()
        if "y_probs_all" in preds.df.columns:
            probs = np.asarray(preds.df["y_probs_all"].tolist(), dtype=float)
        else:
            # Fallback: one-hot from y_pred
            n_classes = len(preds.class_names)
            y_pred = preds.df["y_pred"].astype(int).to_numpy()
            probs = np.zeros((len(y_pred), n_classes))
            probs[np.arange(len(y_pred)), y_pred] = 1.0
        node_probs[node_id] = {
            eid_col[i]: probs[i] for i in range(len(eid_col))
        }

    # Cache reachable leaves per node
    reachable_cache: dict[str, list[str]] = {}

    # Get all experiment IDs from root
    root_eids = sorted(model_preds[tree.root.id].df["experiment_id"].astype(str))
    n_experiments = len(root_eids)

    # Compute leaf probability matrix
    y_probs = np.zeros((n_experiments, n_leaves))
    for i, eid in enumerate(root_eids):
        _propagate(
            tree.root, eid, 1.0, node_probs, leaf_to_idx,
            reachable_cache, y_probs[i],
        )

    y_pred = y_probs.argmax(axis=1)

    # Get ground truth leaf indices from gt_walk
    gt_leaf_map = dict(zip(
        gt_walk["experiment_id"].astype(str),
        gt_walk["gt_terminal_leaf"].astype(str),
    ))
    y_true = np.array([
        leaf_to_idx.get(gt_leaf_map.get(eid, ""), -1) for eid in root_eids
    ], dtype=int)

    return y_true, y_probs, y_pred, leaves, root_eids


def _propagate(
    node: CascadeNode,
    eid: str,
    incoming: float,
    node_probs: dict[str, dict[str, np.ndarray]],
    leaf_to_idx: dict[str, int],
    reachable_cache: dict[str, list[str]],
    out: np.ndarray,
) -> None:
    """Recursively propagate probability through the cascade tree."""
    probs = node_probs.get(node.id, {}).get(eid)

    if probs is not None:
        for c, class_name in enumerate(node.class_names):
            p = incoming * float(probs[c])
            target = node.children[class_name]
            if isinstance(target, str):
                out[leaf_to_idx[target]] += p
            else:
                _propagate(
                    target, eid, p, node_probs, leaf_to_idx,
                    reachable_cache, out,
                )
    else:
        # Missing at this node: distribute uniformly across reachable leaves
        reachable = _reachable_leaves(node, reachable_cache)
        if reachable:
            p = incoming / len(reachable)
            for leaf in reachable:
                out[leaf_to_idx[leaf]] += p


def _reachable_leaves(
    node: CascadeNode, cache: dict[str, list[str]]
) -> list[str]:
    """All terminal leaf labels reachable from this node (cached)."""
    if node.id in cache:
        return cache[node.id]
    result: list[str] = []
    for class_name in node.class_names:
        target = node.children[class_name]
        if isinstance(target, str):
            result.append(target)
        else:
            result.extend(_reachable_leaves(target, cache))
    cache[node.id] = result
    return result
