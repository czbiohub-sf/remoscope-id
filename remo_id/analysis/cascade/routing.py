"""Walk each experiment through the cascade using its predictions (for the
cascade's own output) and its true labels (for ground truth)."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from remo_id.analysis.cascade.loader import ModelPredictions
from remo_id.analysis.cascade.tree import CascadeNode, CascadeTree


def _build_node_lookups(
    tree: CascadeTree, model_preds: dict[str, ModelPredictions]
) -> dict[str, dict[str, tuple[int, int]]]:
    """Per node, map experiment_id -> (y_true, y_pred) for O(1) lookup."""
    out: dict[str, dict[str, tuple[int, int]]] = {}
    for node_id, preds in model_preds.items():
        eid_col = preds.df["experiment_id"].astype(str).to_numpy()
        yt = preds.df["y_true"].astype(int).to_numpy()
        yp = preds.df["y_pred"].astype(int).to_numpy()
        out[node_id] = {eid_col[i]: (int(yt[i]), int(yp[i])) for i in range(len(eid_col))}
    return out


def walk_ground_truth_path(
    tree: CascadeTree, model_preds: dict[str, ModelPredictions]
) -> pd.DataFrame:
    """Descend using y_true at each node. Under strict coverage, every
    experiment in a node's in-scope set must be present at that node, so
    the walk always reaches a leaf.
    """
    lookups = _build_node_lookups(tree, model_preds)
    rows: list[dict] = []
    root_ids = set(model_preds[tree.root.id].df["experiment_id"].astype(str))
    for eid in sorted(root_ids):
        rows.append(_walk_one(tree.root, eid, lookups, use_pred=False))
    return _assemble_df(rows, prefix="gt_")


def walk_predicted_path(
    tree: CascadeTree, model_preds: dict[str, ModelPredictions]
) -> pd.DataFrame:
    """Descend using y_pred at each node. If routing lands on a child node
    that doesn't have this experiment (false-positive-at-parent), flag as
    cascade_rejected and terminate at ``unresolved_at_<node_id>``.
    """
    lookups = _build_node_lookups(tree, model_preds)
    rows: list[dict] = []
    root_ids = set(model_preds[tree.root.id].df["experiment_id"].astype(str))
    for eid in sorted(root_ids):
        rows.append(_walk_one(tree.root, eid, lookups, use_pred=True))
    return _assemble_df(rows, prefix="pred_")


def _walk_one(
    start: CascadeNode,
    eid: str,
    lookups: dict[str, dict[str, tuple[int, int]]],
    use_pred: bool,
) -> dict:
    path: list[tuple[str, str]] = []
    node: Optional[CascadeNode] = start
    terminal_leaf: Optional[str] = None
    cascade_rejected = False
    rejection_node: Optional[str] = None

    while node is not None:
        node_lookup = lookups.get(node.id, {})
        if eid not in node_lookup:
            cascade_rejected = True
            rejection_node = node.id
            terminal_leaf = f"unresolved_at_{node.id}"
            break
        y_true, y_pred = node_lookup[eid]
        idx = y_pred if use_pred else y_true
        if idx < 0 or idx >= len(node.class_names):
            cascade_rejected = True
            rejection_node = node.id
            terminal_leaf = f"invalid_class_at_{node.id}"
            break
        class_name = node.class_names[idx]
        path.append((node.id, class_name))
        target = node.children.get(class_name)
        if target is None:
            raise RuntimeError(
                f"Tree inconsistency: node {node.id} has no route for class {class_name!r}."
            )
        if isinstance(target, str):
            terminal_leaf = target
            node = None
        else:
            node = target

    return {
        "experiment_id": eid,
        "path": path,
        "terminal_leaf": terminal_leaf,
        "cascade_rejected": cascade_rejected,
        "rejection_node": rejection_node,
        "terminal_depth": len(path),
    }


def _assemble_df(rows: list[dict], prefix: str) -> pd.DataFrame:
    max_depth = max((len(r["path"]) for r in rows), default=0)

    records: list[dict] = []
    for r in rows:
        rec = {
            "experiment_id": r["experiment_id"],
            f"{prefix}terminal_leaf": r["terminal_leaf"],
            f"{prefix}terminal_depth": r["terminal_depth"],
            f"{prefix}cascade_rejected": r["cascade_rejected"],
            f"{prefix}rejection_node": r["rejection_node"],
        }
        for d in range(max_depth):
            if d < len(r["path"]):
                nid, cls = r["path"][d]
                rec[f"{prefix}depth_{d}_node"] = nid
                rec[f"{prefix}depth_{d}_class"] = cls
            else:
                rec[f"{prefix}depth_{d}_node"] = None
                rec[f"{prefix}depth_{d}_class"] = None
        records.append(rec)
    return pd.DataFrame.from_records(records)
