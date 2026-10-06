"""Input parsing for cascade analysis.

Users supply one (checkpoint, class_config, predictions_kind) triple per
trained model — either via a YAML or via repeated ``--model`` CLI args.
The tree structure is inferred later from the configs + database.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


VALID_KINDS = ("single", "cv", "loo")


@dataclass
class Model:
    checkpoint: str
    class_config: str
    predictions_kind: str
    predictions_dir: Optional[str] = None
    split: str = "val"
    epoch: Optional[int] = None
    report_dir: Optional[str] = None

    @property
    def id(self) -> str:
        """Stable identifier used as dict key and in reports."""
        return self.checkpoint

    @property
    def short_name(self) -> str:
        """Human-friendly label for tree rendering."""
        return os.path.basename(os.path.dirname(self.checkpoint)) or os.path.basename(self.checkpoint)

    def resolved_predictions_dir(self) -> str:
        if self.predictions_dir:
            return self.predictions_dir
        return str(Path(self.checkpoint).parent)


@dataclass
class CascadeModels:
    name: str
    split: str
    bootstrap_n: int
    models: list[Model] = field(default_factory=list)
    source_path: Optional[str] = None
    display_names: Optional[dict[str, str]] = None


def parse_cascade_models(
    yaml_path: Optional[str],
    cli_model_args: list[str],
    cli_split: str = "val",
    cli_bootstrap_n: int = 1000,
    cli_name: str = "cascade",
) -> CascadeModels:
    """Parse cascade input from a YAML file and/or CLI ``--model`` args.

    CLI form: each arg is ``"checkpoint,class_config,kind[,predictions_dir]"``.
    YAML form: see module docstring for ``run_cascade``.

    Validates kinds, file existence, and that at least two models are present.
    """
    import yaml

    if not yaml_path and not cli_model_args:
        raise ValueError(
            "Provide either --models_yaml or at least one --model argument."
        )

    name = cli_name
    split = cli_split
    bootstrap_n = cli_bootstrap_n
    models: list[Model] = []
    source_path = None

    display_names = None

    if yaml_path:
        source_path = yaml_path
        with open(yaml_path) as f:
            raw = yaml.safe_load(f) or {}
        name = raw.get("name", cli_name)
        split = raw.get("split", cli_split)
        bootstrap_n = int(raw.get("bootstrap_n", cli_bootstrap_n))
        display_names = raw.get("display_names")
        for entry in raw.get("models", []):
            models.append(_model_from_dict(entry, default_split=split))

    for arg in cli_model_args:
        models.append(_model_from_cli_arg(arg, default_split=split))

    if len(models) < 2:
        raise ValueError(
            f"Cascade requires at least 2 models; got {len(models)}."
        )

    seen: set[str] = set()
    for m in models:
        if m.id in seen:
            raise ValueError(f"Duplicate checkpoint in cascade: {m.id}")
        seen.add(m.id)
        _validate_model(m)

    return CascadeModels(
        name=name,
        split=split,
        bootstrap_n=bootstrap_n,
        models=models,
        source_path=source_path,
        display_names=display_names,
    )


def _model_from_dict(entry: dict, default_split: str) -> Model:
    missing = [k for k in ("checkpoint", "class_config", "predictions_kind") if k not in entry]
    if missing:
        raise ValueError(f"Model entry missing required keys: {missing}. Got: {entry}")
    return Model(
        checkpoint=entry["checkpoint"],
        class_config=entry["class_config"],
        predictions_kind=entry["predictions_kind"],
        predictions_dir=entry.get("predictions_dir"),
        split=entry.get("split", default_split),
        epoch=entry.get("epoch"),
        report_dir=entry.get("report_dir"),
    )


def _model_from_cli_arg(arg: str, default_split: str) -> Model:
    parts = [p.strip() for p in arg.split(",")]
    if len(parts) < 3 or len(parts) > 4:
        raise ValueError(
            f"--model must be 'checkpoint,class_config,kind[,predictions_dir]'; got: {arg!r}"
        )
    return Model(
        checkpoint=parts[0],
        class_config=parts[1],
        predictions_kind=parts[2],
        predictions_dir=parts[3] if len(parts) == 4 else None,
        split=default_split,
    )


def _validate_model(m: Model) -> None:
    if m.predictions_kind not in VALID_KINDS:
        raise ValueError(
            f"predictions_kind must be one of {VALID_KINDS}; got {m.predictions_kind!r} for {m.id}"
        )
    if not os.path.exists(m.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {m.checkpoint}")
    if not os.path.exists(m.class_config):
        raise FileNotFoundError(f"Class config not found: {m.class_config}")
    pd_dir = m.resolved_predictions_dir()
    if not os.path.isdir(pd_dir):
        raise FileNotFoundError(f"Predictions dir not found for {m.id}: {pd_dir}")
