"""Cascade-specific figures. Leaf-level confusion matrix and per-class
metrics tables reuse the existing classification plots."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from remo_id.analysis.cascade.tree import CascadeNode, CascadeTree
from remo_id.reports import style


def plot_cascade_tree(
    tree: CascadeTree,
    node_routed_counts: Optional[dict[str, int]] = None,
    node_routed_accuracy: Optional[dict[str, float]] = None,
) -> plt.Figure:
    """Static tree diagram: nodes as boxes with n + accuracy, edges
    labeled by class name. Layout by depth (x) and in-order (y)."""
    style.apply()

    positions: dict[str, tuple[float, float]] = {}
    leaf_positions: dict[tuple[str, str], tuple[float, float]] = {}
    _layout(tree.root, 0, [0.0], positions, leaf_positions)

    depth_xs: dict[int, float] = {}
    for node_id, (_, y) in positions.items():
        depth = tree.node_depth[node_id]
        depth_xs[depth] = depth * 1.6

    fig, ax = style.figure(width="double", aspect=0.55)
    ax.set_axis_off()

    for node_id, (_y_idx, y_val) in positions.items():
        depth = tree.node_depth[node_id]
        x = depth_xs[depth]
        node = tree.nodes_by_id[node_id]
        annots = [node.model.short_name]
        if node_routed_counts and node_id in node_routed_counts:
            annots.append(f"n={node_routed_counts[node_id]}")
        if node_routed_accuracy and node_id in node_routed_accuracy:
            annots.append(f"acc={node_routed_accuracy[node_id]:.2f}")
        box_label = "\n".join(annots)
        ax.text(
            x, y_val, box_label,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=style.FG, linewidth=1.0),
            fontsize=7, color=style.FG,
        )
        positions[node_id] = (x, y_val)

    max_depth = max(tree.node_depth.values())
    for (parent_id, class_name), (_, y_leaf) in leaf_positions.items():
        parent_x, parent_y = positions[parent_id]
        leaf_x = (max_depth + 1) * 1.6
        ax.annotate(
            "",
            xy=(leaf_x - 0.25, y_leaf), xytext=(parent_x + 0.25, parent_y),
            arrowprops=dict(arrowstyle="->", color=style.MUTE, lw=0.9),
        )
        ax.text(
            (parent_x + leaf_x) / 2, (parent_y + y_leaf) / 2 + 0.05,
            class_name, ha="center", va="bottom",
            fontsize=7, color=style.MUTE,
        )
        leaf_label = tree.nodes_by_id[parent_id].children[class_name]
        ax.text(
            leaf_x, y_leaf, leaf_label,
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=style.MUTE_FAINT,
                      edgecolor=style.MUTE, linewidth=0.8),
            fontsize=7, color=style.FG,
        )

    for node_id, node in tree.nodes_by_id.items():
        parent_x, parent_y = positions[node_id]
        for class_name, target in node.children.items():
            if not isinstance(target, CascadeNode):
                continue
            child_x, child_y = positions[target.id]
            ax.annotate(
                "",
                xy=(child_x - 0.25, child_y),
                xytext=(parent_x + 0.25, parent_y),
                arrowprops=dict(arrowstyle="->", color=style.FG, lw=1.0),
            )
            ax.text(
                (parent_x + child_x) / 2, (parent_y + child_y) / 2 + 0.05,
                class_name, ha="center", va="bottom",
                fontsize=7, color=style.FG,
            )

    ax.set_title("Cascade decision tree")
    ax.set_xlim(-0.8, (max_depth + 1) * 1.6 + 1.4)
    all_ys = [y for _, y in positions.values()] + [y for _, y in leaf_positions.values()]
    if all_ys:
        ax.set_ylim(min(all_ys) - 0.6, max(all_ys) + 0.6)
    fig.tight_layout()
    return fig


def _layout(
    node: CascadeNode,
    _depth: int,
    y_cursor: list[float],
    positions: dict[str, tuple[float, float]],
    leaf_positions: dict[tuple[str, str], tuple[float, float]],
) -> float:
    child_ys: list[float] = []
    for class_name, target in node.children.items():
        if isinstance(target, CascadeNode):
            child_y = _layout(target, _depth + 1, y_cursor, positions, leaf_positions)
        else:
            child_y = y_cursor[0]
            leaf_positions[(node.id, class_name)] = (0.0, child_y)
            y_cursor[0] -= 1.0
        child_ys.append(child_y)
    y = sum(child_ys) / len(child_ys) if child_ys else y_cursor[0]
    positions[node.id] = (0.0, y)
    return y


def plot_per_depth_accuracy(per_depth_df: pd.DataFrame) -> plt.Figure:
    """Per-depth accuracy bar chart. Expects columns: depth, n, accuracy."""
    style.apply()
    fig, ax = style.figure(width="single", aspect=0.75)
    depths = per_depth_df["depth"].to_numpy()
    accs = per_depth_df["accuracy"].to_numpy()
    ns = per_depth_df["n"].to_numpy()
    ax.bar(depths, accs, color=style.FG, width=0.55)
    for d, a, n in zip(depths, accs, ns):
        ax.text(d, a + 0.01, f"n={int(n)}", ha="center", va="bottom",
                fontsize=7, color=style.MUTE)
    ax.set_xlabel("Tree depth")
    ax.set_ylabel("Accuracy at depth")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(depths)
    ax.set_title("Per-depth routing accuracy")
    fig.tight_layout()
    return fig
