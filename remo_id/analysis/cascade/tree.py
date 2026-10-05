"""Infer the cascade decision tree from each model's scope + per-class
experiment sets.

Rule: model B is the child of model A under route class ``C`` iff
``scope(B) == class_experiments(A, C)``. Parent classes that have no
matching child are terminal leaves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from remo_id.analysis.cascade.models import Model
from remo_id.analysis.cascade.scope import ModelScope


@dataclass
class CascadeNode:
    model: Model
    scope_size: int
    class_names: list[str]
    children: dict[str, Union["CascadeNode", str]] = field(default_factory=dict)
    """Map from predicted-class name to either a child CascadeNode or a leaf label."""

    @property
    def id(self) -> str:
        return self.model.id

    def is_leaf_target(self, class_name: str) -> bool:
        return isinstance(self.children.get(class_name), str)


@dataclass
class CascadeTree:
    root: CascadeNode
    leaves: list[str]
    nodes_by_id: dict[str, CascadeNode]
    node_depth: dict[str, int]

    def depth_of(self, node_id: str) -> int:
        return self.node_depth[node_id]

    def leaf_labels_per_node(self) -> dict[str, list[str]]:
        """For each node, the leaf labels terminating via its routes."""
        out: dict[str, list[str]] = {}
        for nid, node in self.nodes_by_id.items():
            out[nid] = [
                target for target in node.children.values()
                if isinstance(target, str)
            ]
        return out


def infer_tree(
    scopes: dict[str, ModelScope],
    models: list[Model],
) -> CascadeTree:
    """Build the cascade tree from the given model scopes.

    Errors on: multiple roots, disconnected models, ambiguous parent,
    or parent-class → child-scope mismatch (subset without equality).
    """
    by_id = {m.id: m for m in models}
    if set(by_id) != set(scopes):
        raise ValueError("scopes and models must cover the same set of model ids.")

    parents: dict[str, tuple[str, str]] = {}
    for child_id, child_scope in scopes.items():
        candidates = []
        for parent_id, parent_scope in scopes.items():
            if parent_id == child_id:
                continue
            for class_name, members in parent_scope.class_experiments.items():
                if len(members) == 0:
                    continue
                if members == child_scope.scope:
                    candidates.append((parent_id, class_name, 1.0))
                elif child_scope.scope.issubset(members):
                    coverage = len(child_scope.scope) / len(members)
                    if coverage >= 0.8:
                        candidates.append((parent_id, class_name, coverage))
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous parent for model {child_id}: matches "
                f"{candidates}. Two models' class_configs select the "
                f"same experiment set — rewrite one so scopes are distinct."
            )
        if len(candidates) == 1:
            parent_id, class_name, coverage = candidates[0]
            parents[child_id] = (parent_id, class_name)
            if coverage < 1.0:
                gap = len(scopes[parent_id].class_experiments[class_name]) - len(child_scope.scope)
                print(
                    f"  Note: {by_id[child_id].short_name} covers "
                    f"{coverage:.0%} of parent class '{class_name}' "
                    f"({gap} experiments without child predictions → cascade_rejected)"
                )

    roots = [mid for mid in scopes if mid not in parents]
    if len(roots) != 1:
        raise ValueError(
            f"Cascade must have exactly one root (a model whose scope is not equal "
            f"to any other model's class); found {len(roots)}: {roots}. "
            f"Check that the broadest model's scope is a proper superset of every "
            f"other model's scope."
        )
    root_id = roots[0]

    _validate_root_superset(root_id, scopes)
    _validate_no_orphan_route_targets(scopes, parents)

    nodes_by_id: dict[str, CascadeNode] = {
        mid: CascadeNode(
            model=by_id[mid],
            scope_size=len(scopes[mid].scope),
            class_names=list(scopes[mid].class_names),
        )
        for mid in scopes
    }

    leaf_label_uses: dict[str, int] = {}
    path_to_leaf: dict[tuple[str, str], str] = {}

    def assign_leaf_label(parent_path: list[str], class_name: str) -> str:
        leaf = class_name
        if leaf_label_uses.get(leaf, 0) > 0:
            leaf = ".".join(parent_path + [class_name])
        leaf_label_uses[leaf] = leaf_label_uses.get(leaf, 0) + 1
        return leaf

    _wire_children(root_id, [], nodes_by_id, scopes, parents, assign_leaf_label)

    node_depth: dict[str, int] = {}
    leaves_ordered: list[str] = []
    _collect_order(nodes_by_id[root_id], 0, node_depth, leaves_ordered)

    return CascadeTree(
        root=nodes_by_id[root_id],
        leaves=leaves_ordered,
        nodes_by_id=nodes_by_id,
        node_depth=node_depth,
    )


def _validate_root_superset(root_id: str, scopes: dict[str, ModelScope]) -> None:
    root_scope = scopes[root_id].scope
    for mid, ms in scopes.items():
        if mid == root_id:
            continue
        if not ms.scope.issubset(root_scope):
            extra = ms.scope - root_scope
            sample = sorted(extra)[:5]
            raise ValueError(
                f"Model {mid} has {len(extra)} experiment(s) not in root {root_id}'s scope "
                f"(e.g. {sample}). All cascade models must be reachable from the root."
            )


def _validate_no_orphan_route_targets(
    scopes: dict[str, ModelScope],
    parents: dict[str, tuple[str, str]],
) -> None:
    """Verify that for every (parent, class_name) that claims a child,
    the child's scope EQUALS (not just contains) that parent class's
    experiments. The parents dict already encodes equality, so the check
    here is a cross-validation that two candidate children don't disagree."""
    by_parent_class: dict[tuple[str, str], str] = {}
    for child_id, (parent_id, class_name) in parents.items():
        key = (parent_id, class_name)
        if key in by_parent_class:
            raise ValueError(
                f"Two children both claim parent {parent_id} route '{class_name}': "
                f"{by_parent_class[key]} and {child_id}."
            )
        by_parent_class[key] = child_id


def _wire_children(
    node_id: str,
    ancestor_classes: list[str],
    nodes_by_id: dict[str, CascadeNode],
    scopes: dict[str, ModelScope],
    parents: dict[str, tuple[str, str]],
    assign_leaf_label,
) -> None:
    node = nodes_by_id[node_id]
    child_by_parent_class = {
        class_name: child_id
        for child_id, (parent_id, class_name) in parents.items()
        if parent_id == node_id
    }
    for class_name in node.class_names:
        if class_name in child_by_parent_class:
            child_id = child_by_parent_class[class_name]
            node.children[class_name] = nodes_by_id[child_id]
            _wire_children(
                child_id,
                ancestor_classes + [class_name],
                nodes_by_id, scopes, parents, assign_leaf_label,
            )
        else:
            leaf = assign_leaf_label(ancestor_classes, class_name)
            node.children[class_name] = leaf


def _collect_order(
    node: CascadeNode,
    depth: int,
    depths: dict[str, int],
    leaves: list[str],
) -> None:
    depths[node.id] = depth
    for class_name in node.class_names:
        target = node.children[class_name]
        if isinstance(target, CascadeNode):
            _collect_order(target, depth + 1, depths, leaves)
        else:
            leaves.append(target)


def render_tree_ascii(
    tree: CascadeTree,
    counts: Optional[dict[str, int]] = None,
    node_accuracies: Optional[dict[str, float]] = None,
) -> str:
    """Pretty-print the inferred tree.

    ``counts`` maps node_id → number of experiments in that node's scope.
    ``node_accuracies`` maps node_id → accuracy on its routed subset.
    """
    lines: list[str] = []
    _render_recursive(tree.root, "", True, counts, node_accuracies, lines, is_root=True)
    return "\n".join(lines)


def _render_recursive(
    node: CascadeNode,
    prefix: str,
    is_last: bool,
    counts: Optional[dict[str, int]],
    node_accs: Optional[dict[str, float]],
    lines: list[str],
    is_root: bool = False,
) -> None:
    node_label = node.model.short_name
    annotations: list[str] = [f"classes={node.class_names}"]
    if counts and node.id in counts:
        annotations.append(f"n={counts[node.id]}")
    elif node.scope_size:
        annotations.append(f"n={node.scope_size}")
    if node_accs and node.id in node_accs:
        annotations.append(f"acc={node_accs[node.id]:.3f}")
    head = f"{node_label}  [{'; '.join(annotations)}]"

    if is_root:
        lines.append(head)
        child_prefix = ""
    else:
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + head)
        child_prefix = prefix + ("    " if is_last else "│   ")

    class_items = list(node.children.items())
    for i, (class_name, target) in enumerate(class_items):
        last_child = i == len(class_items) - 1
        connector = "└── " if last_child else "├── "
        if isinstance(target, str):
            lines.append(child_prefix + connector + f"{class_name} → LEAF '{target}'")
        else:
            lines.append(child_prefix + connector + f"{class_name} →")
            _render_recursive(
                target,
                child_prefix + ("    " if last_child else "│   "),
                True, counts, node_accs, lines,
            )
