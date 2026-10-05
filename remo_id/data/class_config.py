from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ClassConfigResult:
    class_definitions: list
    class_names: list
    task_type: str
    num_classes: int
    negative_study_ids: list
    exclude_filters: list
    include_filters: list
    default_class: str | None
    raw_config: dict


def parse_class_config(path: str) -> ClassConfigResult:
    """Load and validate a class config YAML file."""
    import yaml

    with open(path, 'r') as f:
        class_config = yaml.safe_load(f)
    return class_config_from_dict(class_config)


def class_config_from_dict(class_config: dict) -> ClassConfigResult:
    """Validate an already-loaded class config (e.g. built from a manifest)."""
    if 'classes' not in class_config or not class_config['classes']:
        raise ValueError("class_config YAML must have a non-empty 'classes' list.")

    class_names = list(dict.fromkeys(c['name'] for c in class_config['classes']))
    num_classes = len(class_names)

    if num_classes <= 2:
        task_type = 'binary'
        negative_study_ids = class_config.get('negative_studies', []) if num_classes == 1 else []
        num_classes = 2
    else:
        task_type = 'multiclass'
        negative_study_ids = []

    exclude_filters = class_config.get('exclude', [])
    include_filters = class_config.get('include', [])
    default_class = class_config.get('default_class')

    return ClassConfigResult(
        class_definitions=class_config['classes'],
        class_names=class_names,
        task_type=task_type,
        num_classes=num_classes,
        negative_study_ids=negative_study_ids,
        exclude_filters=exclude_filters,
        include_filters=include_filters,
        default_class=default_class,
        raw_config=class_config,
    )


def validate_class_config_conflicts(kwargs: dict, source: str = "--class_config") -> None:
    """Raise click.UsageError if class_config/manifest combined with conflicting options."""
    import click
    conflicts = []
    if source != "--class_config":
        for key in ('class_config', 'study_ids', 'label_contains'):
            if kwargs.get(key):
                conflicts.append(f'--{key}')
    if kwargs.get('database_column'):
        conflicts.append('--database_column')
    if kwargs.get('study_id'):
        conflicts.append('--study_id')
    if kwargs.get('database_label'):
        conflicts.append('--database_label')
    if kwargs.get('negative_study_ids'):
        conflicts.append('--negative_study_ids')
    if kwargs.get('negative_experiment_ids'):
        conflicts.append('--negative_experiment_ids')
    if kwargs.get('class_definitions'):
        conflicts.append('--class_definitions')
    if conflicts:
        raise click.UsageError(
            f"{source} cannot be used with {', '.join(conflicts)}. "
            f"All labeling is defined by {source}."
        )
