"""Unit tests for the mixed-bag extensions to CrossExperimentBatchSampler
and CrossExperimentCollateFn.

Covers:
1. Parity with single-class bags when majority_min_pct=1.0
2. Proportion distribution under majority_min_pct=0.6
3. Single-experiment mode (cross_experiment=False) within a class slot
4. Soft collate preserves per-image labels with proportions matching inputs
"""

import random
from collections import Counter

import torch

from remo_id.data.sampler import (
    CrossExperimentBatchSampler,
    CrossExperimentCollateFn,
)


def _make_multiclass_ranges(num_classes=4, exps_per_class=8, images_per_exp=500):
    """Build synthetic (exp_ranges, exp_labels) for multiclass tests."""
    exp_ranges = {}
    exp_labels = {}
    idx = 0
    for c in range(num_classes):
        for i in range(exps_per_class):
            exp_id = f"class{c}_exp{i}"
            exp_ranges[exp_id] = (idx, idx + images_per_exp)
            exp_labels[exp_id] = c
            idx += images_per_exp
    return exp_ranges, exp_labels


def _experiment_of(idx, exp_ranges):
    for exp_id, (start, end) in exp_ranges.items():
        if start <= idx < end:
            return exp_id
    raise KeyError(idx)


def test_parity_single_class_at_majority_min_pct_one():
    """majority_min_pct=1.0 → every bag is single-class (matches prior behavior)."""
    random.seed(0)
    exp_ranges, exp_labels = _make_multiclass_ranges()
    bag_size = 200
    n_bags = 4
    sampler = CrossExperimentBatchSampler(
        exp_ranges, exp_labels,
        bag_size=bag_size, n_bags=n_bags, num_batches=50,
        task_type="multiclass", num_classes=4,
        majority_min_pct=1.0, cross_experiment=True,
    )
    bags_checked = 0
    for batch in sampler:
        assert len(batch) == bag_size * n_bags
        for b in range(n_bags):
            bag = batch[b * bag_size : (b + 1) * bag_size]
            classes = {exp_labels[_experiment_of(i, exp_ranges)] for i in bag}
            assert len(classes) == 1, f"Mixed classes in single-class bag: {classes}"
            bags_checked += 1
    assert bags_checked > 0
    print(f"Parity test: {bags_checked} bags, all single-class ✓")


def test_majority_distribution_at_min_pct_point_six():
    """With majority_min_pct=0.6, the majority class count per bag is in [120, 200]
    and filler classes share the remainder roughly uniformly."""
    random.seed(1)
    exp_ranges, exp_labels = _make_multiclass_ranges()
    bag_size = 200
    n_bags = 4
    num_batches = 500
    sampler = CrossExperimentBatchSampler(
        exp_ranges, exp_labels,
        bag_size=bag_size, n_bags=n_bags, num_batches=num_batches,
        task_type="multiclass", num_classes=4,
        majority_min_pct=0.6, cross_experiment=True,
    )
    majority_counts = []
    filler_class_totals = Counter()
    bags = 0
    for batch in sampler:
        for b in range(n_bags):
            bag = batch[b * bag_size : (b + 1) * bag_size]
            class_counter = Counter(exp_labels[_experiment_of(i, exp_ranges)] for i in bag)
            majority_class, maj_count = class_counter.most_common(1)[0]
            majority_counts.append(maj_count)
            assert 120 <= maj_count <= bag_size, \
                f"Majority count {maj_count} outside [120, {bag_size}]"
            for c, n in class_counter.items():
                if c != majority_class:
                    filler_class_totals[c] += n
            bags += 1
    assert bags == n_bags * num_batches
    avg_maj = sum(majority_counts) / len(majority_counts)
    # Expected avg majority count = bag_size * (1 + 0.6) / 2 = 160
    assert 140 <= avg_maj <= 175, f"Average majority count {avg_maj:.1f} outside [140, 175]"
    # Filler classes should be roughly uniform across the 3 non-majority classes
    filler_values = list(filler_class_totals.values())
    if filler_values:
        max_v, min_v = max(filler_values), min(filler_values)
        # Ratio no worse than ~1.3 for uniform filler across 3 classes
        assert max_v / max(min_v, 1) < 1.3, \
            f"Filler class counts too skewed: {filler_class_totals}"
    print(f"Distribution test: avg_maj={avg_maj:.1f}, filler={dict(filler_class_totals)} ✓")


def test_single_experiment_mode_within_class_slot():
    """cross_experiment=False → majority slot uses exactly one experiment; each
    filler class slot uses exactly one experiment for its share."""
    random.seed(2)
    exp_ranges, exp_labels = _make_multiclass_ranges()
    bag_size = 200
    n_bags = 4
    sampler = CrossExperimentBatchSampler(
        exp_ranges, exp_labels,
        bag_size=bag_size, n_bags=n_bags, num_batches=50,
        task_type="multiclass", num_classes=4,
        majority_min_pct=0.7, cross_experiment=False,
    )
    bags = 0
    for batch in sampler:
        for b in range(n_bags):
            bag = batch[b * bag_size : (b + 1) * bag_size]
            per_class_exps = {}
            for i in bag:
                exp = _experiment_of(i, exp_ranges)
                c = exp_labels[exp]
                per_class_exps.setdefault(c, set()).add(exp)
            for c, exps in per_class_exps.items():
                assert len(exps) == 1, \
                    f"cross_experiment=False but class {c} drew from {exps}"
            bags += 1
    assert bags > 0
    print(f"Single-experiment test: {bags} bags, one exp per class slot ✓")


def test_soft_collate_produces_correct_proportions():
    """CrossExperimentCollateFn with bag_label_mode='soft' preserves raw per-image
    labels so the downstream model can compute a correct proportion vector."""
    bag_size = 10
    n_bags = 2
    num_classes = 4
    # Bag 0: all class 0. Bag 1: 6 class 2 + 3 class 1 + 1 class 3.
    batch = []
    # Bag 0
    for _ in range(bag_size):
        img = torch.zeros(1, 8, 8)
        batch.append((img, 0, 0))
    # Bag 1
    bag1_labels = [2] * 6 + [1] * 3 + [3] * 1
    for label in bag1_labels:
        img = torch.zeros(1, 8, 8)
        batch.append((img, label, 1))

    collate = CrossExperimentCollateFn(
        n_bags=n_bags, task_type="multiclass", bag_label_mode="soft"
    )
    x, y, exp_ids = collate(batch)
    assert y.shape == (n_bags * bag_size,)
    assert y.dtype == torch.long
    y_reshaped = y.view(n_bags, bag_size)
    # Compute proportion vector as model would
    props = torch.zeros(n_bags, num_classes)
    for c in range(num_classes):
        props[:, c] = (y_reshaped == c).float().mean(dim=1)
    assert torch.allclose(
        props[0], torch.tensor([1.0, 0.0, 0.0, 0.0])
    ), f"Bag 0 proportions wrong: {props[0].tolist()}"
    assert torch.allclose(
        props[1], torch.tensor([0.0, 0.3, 0.6, 0.1])
    ), f"Bag 1 proportions wrong: {props[1].tolist()}"
    print("Soft collate proportion test: proportions correct ✓")


def test_hard_collate_replicates_majority_under_mixed_bag():
    """CrossExperimentCollateFn with bag_label_mode='hard' collapses mixed bags
    to the majority class (idea 6)."""
    bag_size = 10
    n_bags = 1
    # Bag of 6 class 2 + 3 class 1 + 1 class 3 → majority = 2
    batch = []
    bag_labels = [2] * 6 + [1] * 3 + [3] * 1
    for label in bag_labels:
        img = torch.zeros(1, 8, 8)
        batch.append((img, label, 0))

    collate = CrossExperimentCollateFn(
        n_bags=n_bags, task_type="multiclass", bag_label_mode="hard"
    )
    x, y, exp_ids = collate(batch)
    assert y.shape == (bag_size,)
    assert (y == 2).all(), f"Hard collate did not collapse to majority: {y.tolist()}"
    # Assertion in _forward_step must pass: std across bag = 0
    std = y.view(n_bags, bag_size).float().std(dim=1)
    assert (std == 0).all()
    print("Hard collate mixed-bag test: majority replicated ✓")


def test_binary_mixed_bags_allowed_soft_rejected():
    """Binary accepts majority_min_pct<1.0 (min positive-experiment fraction in
    positive bags) but refuses bag_label_mode='soft'."""
    exp_ranges = {"p0": (0, 100), "n0": (100, 200)}
    exp_labels = {"p0": 1, "n0": 0}
    CrossExperimentBatchSampler(
        exp_ranges, exp_labels,
        bag_size=50, n_bags=2, num_batches=1,
        task_type="binary", num_classes=2,
        majority_min_pct=0.6,
    )
    try:
        CrossExperimentCollateFn(n_bags=2, task_type="binary", bag_label_mode="soft")
        assert False, "Expected ValueError for binary + bag_label_mode=soft"
    except ValueError:
        pass


if __name__ == "__main__":
    test_parity_single_class_at_majority_min_pct_one()
    test_majority_distribution_at_min_pct_point_six()
    test_single_experiment_mode_within_class_slot()
    test_soft_collate_produces_correct_proportions()
    test_hard_collate_replicates_majority_under_mixed_bag()
    test_binary_mixed_bags_allowed_soft_rejected()
    print("\nAll mixed-bag sampler tests passed.")
