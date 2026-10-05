"""Synthetic fixtures for cascade tests.

Builds the in-memory counterparts of Model / ModelScope / ModelPredictions
so tests can exercise tree inference, coverage, routing, and metrics
without touching the real database or checkpoint machinery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from remo_id.analysis.cascade.loader import ModelPredictions
from remo_id.analysis.cascade.models import Model
from remo_id.analysis.cascade.scope import ModelScope


class _FakeClassConfig:
    def __init__(self, class_names):
        self.class_names = list(class_names)


def fake_model(checkpoint: str) -> Model:
    return Model(
        checkpoint=checkpoint,
        class_config=f"{checkpoint}.yaml",
        predictions_kind="single",
        predictions_dir="/tmp",
    )


def fake_scope(
    model_id: str, class_names: list[str], class_experiments: dict[str, set[str]],
) -> ModelScope:
    scope = set().union(*class_experiments.values()) if class_experiments else set()
    return ModelScope(
        model_id=model_id,
        class_names=list(class_names),
        scope=scope,
        class_experiments={k: set(v) for k, v in class_experiments.items()},
        class_config_result=_FakeClassConfig(class_names),
    )


def fake_predictions(
    model: Model,
    ground_truth: dict[str, int],
    predicted: dict[str, int],
    class_names: list[str],
) -> ModelPredictions:
    ids = sorted(set(ground_truth) | set(predicted))
    n_classes = len(class_names)
    rows = []
    for eid in ids:
        yt = ground_truth.get(eid)
        yp = predicted.get(eid)
        if yt is None or yp is None:
            continue
        probs = np.full(n_classes, 0.05 / max(1, n_classes - 1))
        probs[yp] = 0.95
        rows.append({
            "experiment_id": eid, "y_true": int(yt), "y_pred": int(yp),
            "y_probs_all": list(probs), "split": "val",
        })
    df = pd.DataFrame(rows)
    return ModelPredictions(
        model=model, df=df, class_names=class_names,
        source_info=f"fake:{model.checkpoint}",
    )


def build_three_node_cascade():
    """A canonical 3-node cascade used by several tests.

    Root (ckpt=root): classes ['healthy', 'sick']
      - healthy experiments: {e0, e1, e2, e3, e4}  → LEAF
      - sick experiments:    {e5, e6, e7, e8, e9}  → child 'disease_vs_control'

    Child (ckpt=child): classes ['disease', 'control'] (scope = sick set)
      - disease experiments: {e5, e6, e7}   → LEAF
      - control experiments:    {e8, e9}       → LEAF
    """
    root = Model(checkpoint="root", class_config="root.yaml",
                 predictions_kind="single", predictions_dir="/tmp")
    child = Model(checkpoint="child", class_config="child.yaml",
                  predictions_kind="single", predictions_dir="/tmp")
    healthy = {"e0", "e1", "e2", "e3", "e4"}
    sick = {"e5", "e6", "e7", "e8", "e9"}
    disease = {"e5", "e6", "e7"}
    control = {"e8", "e9"}
    root_scope = fake_scope(
        "root", ["healthy", "sick"],
        {"healthy": healthy, "sick": sick},
    )
    child_scope = fake_scope(
        "child", ["disease", "control"],
        {"disease": disease, "control": control},
    )
    return {
        "models": [root, child],
        "scopes": {"root": root_scope, "child": child_scope},
        "healthy": healthy, "sick": sick, "disease": disease, "control": control,
    }
