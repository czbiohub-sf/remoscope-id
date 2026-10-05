"""Strict coverage validation: every experiment a node's class_config
considers in-scope (intersected with the chosen split) must appear in that
node's predictions. Errors name the first missing ids so the user can fix
the upstream run.
"""

from __future__ import annotations

from dataclasses import dataclass

from remo_id.analysis.cascade.loader import ModelPredictions
from remo_id.analysis.cascade.scope import ModelScope


@dataclass
class CoverageReport:
    model_id: str
    in_scope: int
    found: int
    missing: set[str]


def verify_strict_coverage(
    model_id: str,
    scope: ModelScope,
    preds: ModelPredictions,
    split: str,
    max_listed: int = 10,
) -> CoverageReport:
    """Check that ``scope ∩ predicted_split_experiments`` is fully covered.

    The split intersection is derived from the predictions themselves: an
    experiment is 'in the split' for this node iff it appears in the
    filtered predictions df. This matches the user's rule: "every
    experiment inside the class_config has predictions at the correct level".
    """
    pred_ids = set(preds.df["experiment_id"].astype(str))
    missing = scope.scope - pred_ids

    if missing:
        listed = sorted(missing)[:max_listed]
        suffix = "" if len(missing) <= max_listed else f" … (+{len(missing) - max_listed} more)"
        raise ValueError(
            f"Strict coverage failed for node {model_id} (split={split!r}): "
            f"{len(missing)} experiment(s) are in the class_config scope "
            f"({len(scope.scope)} total) but missing from predictions "
            f"({len(pred_ids)} found). Missing ids: {listed}{suffix}. "
            f"Re-run inference for this node or fix the class_config / split."
        )

    return CoverageReport(
        model_id=model_id,
        in_scope=len(scope.scope),
        found=len(pred_ids & scope.scope),
        missing=set(),
    )
