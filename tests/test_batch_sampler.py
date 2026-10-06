#!/usr/bin/env python
"""
Test script for BatchSampler bug fixes.

Tests:
1. Pool-based sampling ensures each experiment is seen ~once per epoch
2. Short experiments are filtered and warned about
3. __len__ returns accurate batch count
4. Works for binary and multiclass tasks
"""

from collections import Counter
from remo_id.data import BatchSampler


def create_mock_data(n_pos=20, n_neg=30, images_per_exp=50, short_exp_count=5, short_images=10):
    """Create mock experiment ranges and labels for testing."""
    exp_ranges = {}
    exp_labels = {}
    global_idx = 0

    # Create positive experiments
    for i in range(n_pos):
        exp_id = f"pos_exp_{i}"
        if i < short_exp_count:
            # Some short experiments
            n_images = short_images
        else:
            n_images = images_per_exp
        exp_ranges[exp_id] = (global_idx, global_idx + n_images)
        exp_labels[exp_id] = 1
        global_idx += n_images

    # Create negative experiments
    for i in range(n_neg):
        exp_id = f"neg_exp_{i}"
        if i < short_exp_count:
            # Some short experiments
            n_images = short_images
        else:
            n_images = images_per_exp
        exp_ranges[exp_id] = (global_idx, global_idx + n_images)
        exp_labels[exp_id] = 0
        global_idx += n_images

    return exp_ranges, exp_labels


def create_multiclass_data(num_classes=4, exps_per_class=15, images_per_exp=50, short_per_class=2, short_images=10):
    """Create mock multiclass data."""
    exp_ranges = {}
    exp_labels = {}
    global_idx = 0

    for c in range(num_classes):
        for i in range(exps_per_class):
            exp_id = f"class{c}_exp_{i}"
            if i < short_per_class:
                n_images = short_images
            else:
                n_images = images_per_exp
            exp_ranges[exp_id] = (global_idx, global_idx + n_images)
            exp_labels[exp_id] = c
            global_idx += n_images

    return exp_ranges, exp_labels


def test_binary_pool_sampling():
    """Test that pool-based sampling sees each experiment ~once per epoch."""
    print("\n" + "="*60)
    print("TEST: Binary pool-based sampling")
    print("="*60)

    # Create data with no short experiments for clean test
    exp_ranges, exp_labels = create_mock_data(n_pos=20, n_neg=40, images_per_exp=50, short_exp_count=0)

    sampler = BatchSampler(
        exp_ranges=exp_ranges,
        exp_labels=exp_labels,
        batch_size=200,  # 8 experiments * 25 images
        n_experiments=8,
        n_pos=4,
        num_batches=1000,  # More than we can yield
        task_type="binary"
    )

    # Count how many times each experiment is sampled in one epoch
    exp_counter = Counter()
    batch_count = 0

    for batch_idxs in sampler:
        batch_count += 1
        # Figure out which experiments are in this batch
        for idx in batch_idxs:
            for exp_id, (start, end) in exp_ranges.items():
                if start <= idx < end:
                    exp_counter[exp_id] += 1
                    break

    # With 20 pos and 40 neg, n_pos=4, n_neg=4:
    # Max batches from pos: 20 // 4 = 5
    # Max batches from neg: 40 // 4 = 10
    # Pool-limited by positive: 5 batches
    expected_batches = min(20 // 4, 40 // 4)

    print(f"Batches yielded: {batch_count}")
    print(f"Expected batches (pool-limited): {expected_batches}")
    print(f"len(sampler): {len(sampler)}")

    # Check that len() matches actual yield count
    assert batch_count == len(sampler), f"len() mismatch: {len(sampler)} vs actual {batch_count}"
    assert batch_count == expected_batches, f"Expected {expected_batches} batches, got {batch_count}"

    # Each experiment should be sampled exactly once (or 0 times if pool ran out)
    pos_counts = [exp_counter[f"pos_exp_{i}"] for i in range(20)]
    neg_counts = [exp_counter[f"neg_exp_{i}"] for i in range(40)]

    # All 20 positives should be used exactly once (5 batches * 4 pos = 20)
    assert all(c == 25 for c in pos_counts), f"Positive experiments not sampled evenly: {pos_counts}"

    # Only 20 negatives should be used (5 batches * 4 neg = 20)
    neg_used = sum(1 for c in neg_counts if c > 0)
    neg_unused = sum(1 for c in neg_counts if c == 0)
    print(f"Negative experiments used: {neg_used}, unused: {neg_unused}")
    assert neg_used == 20, f"Expected 20 negatives used, got {neg_used}"

    print("✓ Pool-based sampling works correctly for binary")


def test_binary_short_experiment_filtering():
    """Test that short experiments are filtered out."""
    print("\n" + "="*60)
    print("TEST: Short experiment filtering")
    print("="*60)

    # Create data with some short experiments
    exp_ranges, exp_labels = create_mock_data(
        n_pos=20, n_neg=30,
        images_per_exp=50,
        short_exp_count=5,  # 5 short pos, 5 short neg
        short_images=10     # Only 10 images
    )

    # With batch_size=200, n_experiments=8: samples_per_exp = 25
    # Experiments with only 10 images should be filtered
    sampler = BatchSampler(
        exp_ranges=exp_ranges,
        exp_labels=exp_labels,
        batch_size=200,
        n_experiments=8,
        n_pos=4,
        num_batches=1000,
        task_type="binary"
    )

    # Check filtering worked
    # Original: 20 pos, 30 neg
    # After filtering: 15 pos (20-5), 25 neg (30-5)
    print(f"Positive experiments after filtering: {len(sampler.pos_experiments)}")
    print(f"Negative experiments after filtering: {len(sampler.neg_experiments)}")

    assert len(sampler.pos_experiments) == 15, f"Expected 15 pos, got {len(sampler.pos_experiments)}"
    assert len(sampler.neg_experiments) == 25, f"Expected 25 neg, got {len(sampler.neg_experiments)}"

    # Verify no short experiments in pools
    for exp in sampler.pos_experiments:
        start, end = exp_ranges[exp]
        assert (end - start) >= 25, f"Short experiment {exp} not filtered"

    for exp in sampler.neg_experiments:
        start, end = exp_ranges[exp]
        assert (end - start) >= 25, f"Short experiment {exp} not filtered"

    print("✓ Short experiment filtering works correctly")


def test_multiclass_pool_sampling():
    """Test pool-based sampling for multiclass."""
    print("\n" + "="*60)
    print("TEST: Multiclass pool-based sampling")
    print("="*60)

    exp_ranges, exp_labels = create_multiclass_data(
        num_classes=4,
        exps_per_class=12,  # 12 per class, no short
        images_per_exp=50,
        short_per_class=0
    )

    # n_experiments=8, so 2 per class
    sampler = BatchSampler(
        exp_ranges=exp_ranges,
        exp_labels=exp_labels,
        batch_size=200,
        n_experiments=8,
        n_pos=4,  # Not used for multiclass
        num_batches=1000,
        task_type="multiclass",
        num_classes=4
    )

    batch_count = 0
    exp_counter = Counter()

    for batch_idxs in sampler:
        batch_count += 1
        for idx in batch_idxs:
            for exp_id, (start, end) in exp_ranges.items():
                if start <= idx < end:
                    exp_counter[exp_id] += 1
                    break

    # With 12 per class and 2 per class per batch: 6 batches
    expected_batches = 12 // 2  # exps_per_class // exps_per_class_per_batch

    print(f"Batches yielded: {batch_count}")
    print(f"Expected batches: {expected_batches}")
    print(f"len(sampler): {len(sampler)}")

    assert batch_count == len(sampler), f"len() mismatch"
    assert batch_count == expected_batches, f"Expected {expected_batches} batches, got {batch_count}"

    print("✓ Multiclass pool-based sampling works correctly")


def test_no_duplicate_experiments_within_batch():
    """Test that within a single batch, no experiment is duplicated (unless pool is tiny)."""
    print("\n" + "="*60)
    print("TEST: No duplicate experiments within batch")
    print("="*60)

    exp_ranges, exp_labels = create_mock_data(n_pos=20, n_neg=30, images_per_exp=50, short_exp_count=0)

    sampler = BatchSampler(
        exp_ranges=exp_ranges,
        exp_labels=exp_labels,
        batch_size=200,
        n_experiments=8,
        n_pos=4,
        num_batches=100,
        task_type="binary"
    )

    for batch_idxs in sampler:
        # Find experiments in this batch
        batch_exps = set()
        for idx in batch_idxs:
            for exp_id, (start, end) in exp_ranges.items():
                if start <= idx < end:
                    batch_exps.add(exp_id)
                    break

        # Should have exactly n_experiments unique experiments
        assert len(batch_exps) == 8, f"Expected 8 unique experiments, got {len(batch_exps)}"

    print("✓ No duplicate experiments within batches")


def test_epoch_reproducibility():
    """Test that different epochs can have different orderings."""
    print("\n" + "="*60)
    print("TEST: Epoch variation (pools reshuffled each epoch)")
    print("="*60)

    exp_ranges, exp_labels = create_mock_data(n_pos=16, n_neg=16, images_per_exp=50, short_exp_count=0)

    sampler = BatchSampler(
        exp_ranges=exp_ranges,
        exp_labels=exp_labels,
        batch_size=200,
        n_experiments=8,
        n_pos=4,
        num_batches=100,
        task_type="binary"
    )

    def get_epoch_order():
        """Get the order of experiments across all batches in one epoch."""
        order = []
        for batch_idxs in sampler:
            batch_exps = []
            for idx in batch_idxs:
                for exp_id, (start, end) in exp_ranges.items():
                    if start <= idx < end:
                        if exp_id not in batch_exps:
                            batch_exps.append(exp_id)
                        break
            order.extend(batch_exps)
        return order

    epoch1_order = get_epoch_order()
    epoch2_order = get_epoch_order()

    print(f"Epoch 1 first 5 experiments: {epoch1_order[:5]}")
    print(f"Epoch 2 first 5 experiments: {epoch2_order[:5]}")

    # Orders should be different (with high probability)
    # Both should have same experiments but different order
    assert set(epoch1_order) == set(epoch2_order), "Different experiments in different epochs!"
    # Note: There's a small chance they could be the same, but very unlikely
    if epoch1_order == epoch2_order:
        print("⚠ Warning: Epochs had same order (rare but possible)")
    else:
        print("✓ Different epochs have different orderings")


def test_len_accuracy():
    """Test that __len__ accurately predicts batch count."""
    print("\n" + "="*60)
    print("TEST: __len__ accuracy")
    print("="*60)

    test_cases = [
        # (n_pos, n_neg, n_experiments, n_pos_per_batch, num_batches_requested)
        (20, 40, 8, 4, 1000),  # Limited by pos: 20//4 = 5
        (40, 20, 8, 4, 1000),  # Limited by neg: 20//4 = 5
        (20, 20, 8, 4, 1000),  # Balanced: 20//4 = 5
        (20, 20, 8, 4, 3),     # Limited by num_batches: 3
        (100, 100, 8, 4, 10),  # Limited by num_batches: 10
    ]

    for n_pos, n_neg, n_exp, n_pos_batch, num_batches in test_cases:
        exp_ranges, exp_labels = create_mock_data(n_pos=n_pos, n_neg=n_neg, images_per_exp=50, short_exp_count=0)

        sampler = BatchSampler(
            exp_ranges=exp_ranges,
            exp_labels=exp_labels,
            batch_size=n_exp * 25,
            n_experiments=n_exp,
            n_pos=n_pos_batch,
            num_batches=num_batches,
            task_type="binary"
        )

        actual_batches = sum(1 for _ in sampler)
        predicted_len = len(sampler)

        status = "✓" if actual_batches == predicted_len else "✗"
        print(f"{status} pos={n_pos}, neg={n_neg}, n_exp={n_exp}, n_pos={n_pos_batch}, req={num_batches} -> len()={predicted_len}, actual={actual_batches}")

        assert actual_batches == predicted_len, f"len() mismatch: predicted {predicted_len}, actual {actual_batches}"

    print("✓ __len__ accuracy verified")


def test_augmentation_separation():
    """Test that train and val augmentations are different."""
    print("\n" + "="*60)
    print("TEST: Train/Val augmentation separation")
    print("="*60)

    from remo_id.data import RemoIDDataLoader

    # Test for different backbones
    backbones = ['vit-b-16', 'resnet18', 'vit-l-16', 'microscopy-cnn']

    for backbone in backbones:
        train_aug = RemoIDDataLoader.prep_augments(backbone, train=True)
        val_aug = RemoIDDataLoader.prep_augments(backbone, train=False)

        # Convert to string representations to compare
        train_str = str(train_aug)
        val_str = str(val_aug)

        has_random_train = 'Random' in train_str
        has_random_val = 'Random' in val_str

        print(f"{backbone}:")
        print(f"  Train has random: {has_random_train}")
        print(f"  Val has random: {has_random_val}")

        assert has_random_train, f"{backbone} train should have random augmentations"
        assert not has_random_val, f"{backbone} val should NOT have random augmentations"

    print("✓ Train/Val augmentation separation verified")


if __name__ == "__main__":
    print("BatchSampler Bug Fix Tests")
    print("="*60)

    # Run all tests
    test_binary_pool_sampling()
    test_binary_short_experiment_filtering()
    test_multiclass_pool_sampling()
    test_no_duplicate_experiments_within_batch()
    test_epoch_reproducibility()
    test_len_accuracy()
    test_augmentation_separation()

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
