"""Batch sampling utilities for balanced training."""

from torch.utils.data import Sampler
import math
import random

import numpy as np


class CrossExperimentBatchSampler(Sampler):
    """
    Samples bags that mix images from multiple experiments.

    For each bag:
    1. Sample k ~ Uniform(1, max_experiments_per_bag)
    2. Select k experiments (ensuring correct label for the bag)
    3. Distribute bag_size images across k experiments
    4. Bag label = 1 if any positive experiment included, else 0

    This prevents experiment memorization by forcing the model to learn
    features that generalize across experiments rather than memorizing
    experiment-specific patterns.

    For multiclass tasks, bags can additionally be mixed across classes via
    ``majority_min_pct``: each bag samples p_maj ~ Uniform(majority_min_pct, 1.0)
    of the majority class, with the remainder drawn uniformly from OTHER classes.
    When majority_min_pct == 1.0 bags are single-class (original behavior).

    IMPORTANT: Use CrossExperimentCollateFn with this sampler to correctly
    compute bag-level labels from the mixed experiment images.

    Args:
        exp_ranges: Dict mapping experiment path to (start_idx, end_idx)
        exp_labels: Dict mapping experiment path to label (0 or 1)
        bag_size: Number of images per bag (default 200)
        n_bags: Number of bags per batch (default 4)
        num_batches: Number of batches per epoch (default 100)
        max_experiments_per_bag: Maximum experiments to mix per bag (default: all available)
        task_type: Task type ('binary' or 'multiclass')
        num_classes: Number of classes for multiclass
        majority_min_pct: Minimum fraction of the majority class per bag (multiclass only).
            Each bag samples p_maj ~ Uniform(majority_min_pct, 1.0); the rest of the bag
            is drawn uniformly from OTHER classes. Default 1.0 = single-class bags.
        cross_experiment: If True, majority and filler slots may pull from multiple
            experiments per class; if False, each slot uses a single experiment.
            Default True (matches the prior CES behavior).
        is_validation: If True, force majority_min_pct=1.0 regardless of constructor value.
    """

    def __init__(self, exp_ranges, exp_labels, bag_size=200, n_bags=4,
                 num_batches=100, max_experiments_per_bag=None,
                 task_type="binary", num_classes=2,
                 majority_min_pct=1.0, cross_experiment=True, is_validation=False):
        self.exp_ranges = exp_ranges
        self.exp_labels = exp_labels
        self.bag_size = bag_size
        self.n_bags = n_bags
        self.num_batches = num_batches
        self.task_type = task_type
        self.num_classes = num_classes
        self.cross_experiment = cross_experiment
        self.is_validation = is_validation

        if is_validation:
            majority_min_pct = 1.0

        if task_type == "binary":
            # majority_min_pct is the minimum fraction of images drawn from
            # positive experiments in a POSITIVE bag (images split evenly across
            # experiments, so this equals n_pos / k). 1.0 = pure-positive bags;
            # 0.5 = positive bags are at least half positive-experiment images.
            # Negative bags are always 100% negative.
            if not (0.0 < majority_min_pct <= 1.0 + 1e-6):
                raise ValueError(
                    f"majority_min_pct must be in (0.0, 1.0] for binary tasks, "
                    f"got {majority_min_pct}"
                )

        if task_type == "multiclass":
            min_allowed = 1.0 / num_classes
            if not (min_allowed - 1e-6 <= majority_min_pct <= 1.0 + 1e-6):
                raise ValueError(
                    f"majority_min_pct must be in [{min_allowed:.4f}, 1.0] "
                    f"for num_classes={num_classes}, got {majority_min_pct}"
                )

        self.majority_min_pct = majority_min_pct

        if task_type == "binary":
            # Separate experiments by label
            self.pos_experiments = [e for e in exp_ranges if exp_labels[e] == 1]
            self.neg_experiments = [e for e in exp_ranges if exp_labels[e] == 0]

            if len(self.pos_experiments) == 0:
                raise ValueError("CrossExperimentBatchSampler requires at least 1 positive experiment")
            if len(self.neg_experiments) == 0:
                raise ValueError("CrossExperimentBatchSampler requires at least 1 negative experiment")

            # Max experiments to mix per bag (default: all available)
            total_experiments = len(self.pos_experiments) + len(self.neg_experiments)
            self.max_experiments = max_experiments_per_bag or total_experiments

            print(f"CrossExperimentBatchSampler: {len(self.pos_experiments)} positive, "
                  f"{len(self.neg_experiments)} negative experiments, "
                  f"max_experiments_per_bag={self.max_experiments}")

        elif task_type == "multiclass":
            # Group experiments by class
            self.class_experiments = {}
            for c in range(num_classes):
                self.class_experiments[c] = [e for e in exp_ranges if exp_labels[e] == c]
            total_experiments = sum(len(exps) for exps in self.class_experiments.values())
            self.max_experiments = max_experiments_per_bag or total_experiments

            class_counts = {c: len(exps) for c, exps in self.class_experiments.items()}
            print(f"CrossExperimentBatchSampler (multiclass): class counts={class_counts}, "
                  f"max_experiments_per_bag={self.max_experiments}")

    def _sample_images_from_experiments(self, selected_experiments):
        """Distribute bag_size images across selected experiments."""
        return self._sample_n_from_experiments(selected_experiments, self.bag_size)

    def _sample_n_from_experiments(self, selected_experiments, n_total):
        """Distribute n_total images across selected experiments."""
        n_selected = len(selected_experiments)
        images_per_exp = n_total // n_selected
        remainder = n_total % n_selected

        bag_idxs = []
        for i, exp in enumerate(selected_experiments):
            start, end = self.exp_ranges[exp]
            n_avail = end - start
            n_images = images_per_exp + (1 if i < remainder else 0)
            if n_images == 0:
                continue

            if n_avail >= n_images:
                idxs = random.sample(range(start, end), n_images)
            else:
                # Not enough images, sample with replacement
                idxs = random.choices(range(start, end), k=n_images)
            bag_idxs.extend(idxs)

        return bag_idxs

    def _pick_experiments_for_class(self, class_id):
        """Pick 1 (or up to max_experiments) experiments from the given class's pool."""
        pool = self.class_experiments.get(class_id, [])
        if not pool:
            return []
        if self.cross_experiment:
            k = random.randint(1, min(self.max_experiments, len(pool)))
            return random.sample(pool, k)
        return [random.choice(pool)]

    def _uniform_partition(self, n_total, n_parts):
        """Partition n_total into n_parts roughly-uniformly. Returns a list of sizes."""
        if n_parts <= 0 or n_total <= 0:
            return []
        base = n_total // n_parts
        rem = n_total % n_parts
        parts = [base + (1 if i < rem else 0) for i in range(n_parts)]
        return parts

    def _build_mixed_multiclass_bag(self):
        """Build a single mixed multiclass bag. Returns (bag_idxs, majority_class)."""
        target_class = random.randint(0, self.num_classes - 1)
        if not self.class_experiments.get(target_class):
            # Fall back to any non-empty class
            non_empty = [c for c, exps in self.class_experiments.items() if exps]
            if not non_empty:
                return [], 0
            target_class = random.choice(non_empty)

        p_maj = random.uniform(self.majority_min_pct, 1.0)
        n_maj = max(1, round(self.bag_size * p_maj))
        n_maj = min(n_maj, self.bag_size)
        n_other = self.bag_size - n_maj

        maj_exps = self._pick_experiments_for_class(target_class)
        bag_idxs = self._sample_n_from_experiments(maj_exps, n_maj) if maj_exps else []

        if n_other > 0:
            other_classes = [c for c in range(self.num_classes)
                             if c != target_class and self.class_experiments.get(c)]
            if other_classes:
                counts = self._uniform_partition(n_other, len(other_classes))
                for c, n_c in zip(other_classes, counts):
                    if n_c <= 0:
                        continue
                    exps = self._pick_experiments_for_class(c)
                    if exps:
                        bag_idxs.extend(self._sample_n_from_experiments(exps, n_c))
            # If no other classes available, bag stays majority-only (short of bag_size)
            # Pad with extra majority samples to keep bag_size consistent
            if len(bag_idxs) < self.bag_size and maj_exps:
                shortfall = self.bag_size - len(bag_idxs)
                bag_idxs.extend(self._sample_n_from_experiments(maj_exps, shortfall))

        random.shuffle(bag_idxs)
        return bag_idxs, target_class

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_idxs = []
            bag_labels = []

            for _ in range(self.n_bags):
                if self.task_type == "binary":
                    # Decide bag label (balanced 50/50)
                    is_positive_bag = random.random() < 0.5

                    # Sample k ~ Uniform(1, max_experiments)
                    k = random.randint(1, self.max_experiments)

                    if is_positive_bag:
                        # Enforce the minimum positive-image fraction: images are
                        # split evenly across the k experiments, so the fraction
                        # from positive experiments equals n_pos / k. Require
                        # n_pos / k >= majority_min_pct (and at least 1 positive).
                        max_pos = min(k, len(self.pos_experiments))
                        min_pos = max(1, math.ceil(self.majority_min_pct * k))
                        min_pos = min(min_pos, max_pos)
                        n_pos = random.randint(min_pos, max_pos)
                        n_neg = k - n_pos

                        # Sample experiments
                        selected_pos = random.sample(self.pos_experiments, n_pos)
                        n_neg_actual = min(n_neg, len(self.neg_experiments))
                        selected_neg = random.sample(self.neg_experiments, n_neg_actual) if n_neg_actual > 0 else []
                        selected = selected_pos + selected_neg
                        bag_labels.append(1)  # Positive bag
                    else:
                        # Only negative experiments for negative bag
                        k_actual = min(k, len(self.neg_experiments))
                        selected = random.sample(self.neg_experiments, k_actual)
                        bag_labels.append(0)  # Negative bag

                    bag_idxs = self._sample_images_from_experiments(selected)
                    batch_idxs.extend(bag_idxs)

                elif self.task_type == "multiclass":
                    # For multiclass, each bag has a majority class. When
                    # majority_min_pct < 1.0 the bag is mixed: p_maj of images come
                    # from the majority class and the remainder is drawn uniformly
                    # from other classes. When majority_min_pct == 1.0 the bag is
                    # single-class (original behavior).
                    bag_idxs, target_class = self._build_mixed_multiclass_bag()
                    if bag_idxs:
                        batch_idxs.extend(bag_idxs)
                        bag_labels.append(target_class)

            # Note: bag_labels are computed here but not stored on the sampler.
            # The CrossExperimentCollateFn owns label computation from per-image labels.
            yield batch_idxs

    def __len__(self):
        return self.num_batches


class CrossExperimentCollateFn:
    """
    Collate function for CrossExperimentBatchSampler.

    This collate function computes bag-level labels from the per-image labels.
    For binary classification: bag_label = 1 if ANY image in the bag is positive.
    For multiclass (hard mode): bag_label = mode (most common class) of images in bag.
    For multiclass (soft mode): per-image labels are preserved so the model can
        compute the proportion vector downstream (LLP training).

    In hard mode, each image gets the bag-level label. In soft mode (multiclass only)
    each image keeps its true class label so the forward step can compute proportions.

    Args:
        n_bags: Number of bags per batch
        task_type: 'binary' or 'multiclass'
        bag_label_mode: 'hard' (replicate majority) or 'soft' (preserve per-image labels,
            multiclass only). Default 'hard'.
    """

    def __init__(self, n_bags, task_type="binary", bag_label_mode="hard"):
        if bag_label_mode not in ("hard", "soft"):
            raise ValueError(f"bag_label_mode must be 'hard' or 'soft', got {bag_label_mode}")
        if bag_label_mode == "soft" and task_type != "multiclass":
            raise ValueError("bag_label_mode='soft' is only supported for multiclass tasks")
        self.n_bags = n_bags
        self.task_type = task_type
        self.bag_label_mode = bag_label_mode

    def __call__(self, batch):
        import torch

        # batch is a list of (img, label, exp_id) tuples
        x = torch.stack([b[0] for b in batch], dim=0)
        exp_ids = torch.tensor([b[2] for b in batch], dtype=torch.long)
        original_labels = [b[1] for b in batch]

        # Compute bag-level labels from original per-image labels
        bag_size = len(batch) // self.n_bags
        y_list = []

        for bag_idx in range(self.n_bags):
            start = bag_idx * bag_size
            end = start + bag_size
            bag_labels_raw = original_labels[start:end]

            if self.task_type == "multiclass" and self.bag_label_mode == "soft":
                # Preserve raw per-image labels; the model computes proportions downstream
                y_list.extend(bag_labels_raw)
                continue

            if self.task_type == "binary":
                # Bag is positive if ANY image is positive
                bag_label = 1 if any(l == 1 for l in bag_labels_raw) else 0
            elif self.task_type == "multiclass":
                # Bag label = most common class (mode)
                from collections import Counter
                counter = Counter(bag_labels_raw)
                bag_label = counter.most_common(1)[0][0]

            # Replicate bag label for all images in bag
            y_list.extend([bag_label] * bag_size)

        # Handle any remainder images (shouldn't happen with correct bag_size)
        while len(y_list) < len(batch):
            y_list.append(y_list[-1] if y_list else 0)

        if self.task_type == "multiclass":
            y = torch.tensor(y_list, dtype=torch.long)
        else:
            y = torch.tensor(y_list, dtype=torch.float32)

        return x, y, exp_ids


class BatchSampler(Sampler):
    """
    Custom batch sampler for balanced sampling across experiments.

    Supports binary and multiclass tasks with different sampling strategies
    for each.

    Sampling pools are built from "units". By default each experiment is its own
    unit (original behavior). With ``exp_sample_ids`` (sample-aware sampling),
    all experiments sharing a sample_id form ONE unit: each pool draw picks one
    technical replicate at random and retires the whole sample until the pool
    refreshes (next epoch). This stops samples with more replicates from being
    oversampled. Null sample_ids stay independent units.

    Args:
        is_validation: If True, relaxes batch requirements to allow smaller batches.
            For binary: requires only 1 pos + 1 neg instead of n_pos + n_neg.
            This ensures validation can run even with small validation sets.
        exp_sample_ids: Optional dict mapping experiment key -> sample_id.
            When provided, pools are per-sample units as described above.
        sample_aware_mode: How a drawn sample's bag is built (sample-aware only).
            'single' (default): one random technical replicate supplies the bag.
            'pooled': the bag is split evenly across ALL of the sample's
            replicates. WARNING: the constant per-sample composition proved
            memorizable (every bag of a sample is statistically identical) —
            prefer 'pooled_random'.
            'pooled_random': per bag, a random nonempty subset of the sample's
            replicates with Dirichlet-random proportions. Same-sample bags vary
            from pure single-session (matches eval bags) to any cross-session
            blend — within-bag technical augmentation without a constant,
            memorizable fingerprint. Bag biology is unchanged (one sample).
    """

    def __init__(self, exp_ranges, exp_labels, batch_size=256, n_experiments=8, n_pos=4, num_batches=200,
                 task_type="binary", num_classes=2, is_validation=False, exp_sample_ids=None,
                 sample_aware_mode="single"):
        if sample_aware_mode not in ("single", "pooled", "pooled_random"):
            raise ValueError(f"sample_aware_mode must be 'single', 'pooled' or 'pooled_random', got {sample_aware_mode}")
        self.sample_aware_mode = sample_aware_mode
        self.exp_ranges = exp_ranges
        self.exp_labels = exp_labels
        self.batch_size = batch_size
        self.n_experiments = n_experiments
        self.n_pos = n_pos
        self.num_batches = num_batches
        self.task_type = task_type
        self.num_classes = num_classes
        self.is_validation = is_validation
        self.exp_sample_ids = exp_sample_ids

        assert batch_size % n_experiments == 0

        samples_per_exp = batch_size // n_experiments

        # Helper to check if experiment has enough images (overridable: the embedding
        # sampler keeps every experiment and backfills short bags instead of dropping).
        def has_enough_images(exp):
            return self._keep_experiment(exp, samples_per_exp)

        # Helper to get experiment image count
        def get_image_count(exp):
            start, end = exp_ranges[exp]
            return end - start

        if task_type == "binary":
            assert 0 <= n_pos <= n_experiments
            # Separate experiments by label
            all_pos = [e for e in exp_ranges if exp_labels[e] == 1]
            all_neg = [e for e in exp_ranges if exp_labels[e] == 0]

            # Filter to experiments with enough images
            self.pos_experiments = [e for e in all_pos if has_enough_images(e)]
            self.neg_experiments = [e for e in all_neg if has_enough_images(e)]

            # Warn about filtered experiments
            filtered_pos = len(all_pos) - len(self.pos_experiments)
            filtered_neg = len(all_neg) - len(self.neg_experiments)
            if filtered_pos > 0 or filtered_neg > 0:
                print(f"BatchSampler: Filtered {filtered_pos} positive and {filtered_neg} negative experiments "
                      f"with fewer than {samples_per_exp} images")
                print(f"BatchSampler: Using {len(self.pos_experiments)} positive and {len(self.neg_experiments)} negative experiments")

            self.pos_units = self._sample_units(self.pos_experiments)
            self.neg_units = self._sample_units(self.neg_experiments)
            if exp_sample_ids is not None:
                print(f"BatchSampler (sample-aware): {len(self.pos_units)} positive samples "
                      f"({len(self.pos_experiments)} experiments), {len(self.neg_units)} negative samples "
                      f"({len(self.neg_experiments)} experiments)")

        elif task_type == "multiclass":
            # Group experiments by class
            self.class_experiments = {}
            total_filtered = 0
            for c in range(num_classes):
                all_class = [e for e in exp_ranges if exp_labels[e] == c]
                self.class_experiments[c] = [e for e in all_class if has_enough_images(e)]
                filtered = len(all_class) - len(self.class_experiments[c])
                if filtered > 0:
                    total_filtered += filtered
                    print(f"BatchSampler: Filtered {filtered} class {c} experiments with fewer than {samples_per_exp} images")
            if total_filtered > 0:
                class_counts = {c: len(exps) for c, exps in self.class_experiments.items()}
                print(f"BatchSampler: Using class counts: {class_counts}")
            self.class_units = {c: self._sample_units(exps)
                                for c, exps in self.class_experiments.items()}
            if exp_sample_ids is not None:
                unit_counts = {c: len(u) for c, u in self.class_units.items()}
                print(f"BatchSampler (sample-aware): samples per class: {unit_counts}")
            # Calculate experiments per class per batch for balanced sampling
            self.exps_per_class = max(1, n_experiments // num_classes)

    def _sample_bag_for_unit(self, unit, samples_per_exp):
        """Build one bag's indices from a popped pool unit.

        'single' mode (and any singleton unit): one random replicate supplies
        the whole bag. 'pooled' mode: the bag is split near-evenly across ALL
        of the sample's replicates, so each bag mixes acquisition sessions.
        """
        if self.sample_aware_mode == "single" or len(unit) == 1:
            return self._sample_cells_for_exp(random.choice(unit), samples_per_exp)
        if self.sample_aware_mode == "pooled_random":
            # Random subset of replicates + Dirichlet-random proportions: the
            # bag's session composition varies draw to draw (incl. pure
            # single-session bags), so no constant fingerprint exists.
            k = random.randint(1, len(unit))
            exps = random.sample(unit, k)
            if k == 1:
                return self._sample_cells_for_exp(exps[0], samples_per_exp)
            counts = np.random.multinomial(samples_per_exp,
                                           np.random.dirichlet(np.ones(k)))
            idxs = []
            for e, n_e in zip(exps, counts):
                if n_e > 0:
                    idxs.extend(self._sample_cells_for_exp(e, int(n_e)))
            return idxs
        # 'pooled': fixed even split across ALL replicates
        exps = unit[:]
        random.shuffle(exps)  # rotate which replicate absorbs the remainder
        base, rem = divmod(samples_per_exp, len(exps))
        idxs = []
        for i, e in enumerate(exps):
            k = base + (1 if i < rem else 0)
            if k > 0:
                idxs.extend(self._sample_cells_for_exp(e, k))
        return idxs

    def _sample_units(self, exps):
        """Group experiments into pool units.

        Default: one unit per experiment. Sample-aware (exp_sample_ids set):
        one unit per sample_id holding all of that sample's technical
        replicates; experiments with a null sample_id stay independent units.
        """
        if self.exp_sample_ids is None:
            return [[e] for e in exps]
        groups = {}
        for e in exps:
            sid = self.exp_sample_ids.get(e)
            if sid is None or (isinstance(sid, float) and math.isnan(sid)):
                sid = f"__unique_exp_{e}"
            groups.setdefault(sid, []).append(e)
        return list(groups.values())

    def __iter__(self):
        """
        Pool-based sampling: each experiment is seen approximately once per epoch.
        Experiments are popped from shuffled pools without replacement within an epoch.
        The epoch ends when any pool is exhausted (or num_batches is reached).

        In validation mode, requirements are relaxed to allow smaller batches.
        """
        samples_per_exp = self.batch_size // self.n_experiments
        batches_yielded = 0

        if self.task_type == "binary":
            num_neg = self.n_experiments - self.n_pos

            # Create fresh pools each epoch. Pools hold units (samples when
            # sample-aware, else singleton experiments); popping a unit picks
            # one random replicate and retires the unit for the epoch.
            pos_pool = self.pos_units[:]
            neg_pool = self.neg_units[:]
            random.shuffle(pos_pool)
            random.shuffle(neg_pool)

            # For validation, relax requirements — allow only pos OR only neg
            # (e.g. LOO with a single experiment that is all-positive or all-negative)
            if self.is_validation:
                min_pos = 1 if len(self.pos_units) > 0 else 0
                min_neg = 1 if len(self.neg_units) > 0 else 0
            else:
                min_pos = self.n_pos
                min_neg = num_neg

            # Yield batches until a pool is exhausted or num_batches reached
            while len(pos_pool) >= min_pos and len(neg_pool) >= min_neg and batches_yielded < self.num_batches:
                # For validation, use available experiments up to the target
                actual_n_pos = min(len(pos_pool), self.n_pos)
                actual_n_neg = min(len(neg_pool), num_neg)

                selected_pos = [pos_pool.pop() for _ in range(actual_n_pos)]
                selected_neg = [neg_pool.pop() for _ in range(actual_n_neg)]

                # For validation with MIL models, we need exactly n_experiments
                # Pad with repeated experiments if needed
                if self.is_validation:
                    total_needed = self.n_experiments
                    selected_exps = selected_pos + selected_neg
                    if len(selected_exps) < total_needed and len(selected_exps) > 0:
                        # Pad by repeating experiments
                        while len(selected_exps) < total_needed:
                            selected_exps.append(random.choice(selected_pos + selected_neg))
                else:
                    selected_exps = selected_pos + selected_neg

                random.shuffle(selected_exps)  # shuffles bag order

                batch_idxs = []
                for unit in selected_exps:
                    batch_idxs.extend(self._sample_bag_for_unit(unit, samples_per_exp))

                batches_yielded += 1
                yield batch_idxs

        elif self.task_type == "multiclass":
            # Create fresh pools for each class (units: samples when
            # sample-aware, else singleton experiments)
            class_pools = {}
            for c in range(self.num_classes):
                class_pools[c] = self.class_units[c][:]
                random.shuffle(class_pools[c])

            # Yield batches until any class pool is exhausted or num_batches reached
            while batches_yielded < self.num_batches:
                selected_exps = []
                remaining = self.n_experiments
                can_continue = True

                # Randomize class iteration order so all classes get
                # represented over multiple batches (important when
                # n_experiments < num_classes)
                class_order = list(range(self.num_classes))
                random.shuffle(class_order)

                for i, c in enumerate(class_order):
                    if remaining <= 0:
                        break
                    # Last class in iteration gets whatever remains
                    is_last = (i == len(class_order) - 1)
                    n_from_class = remaining if is_last else min(self.exps_per_class, remaining)
                    pool = class_pools[c]

                    if len(pool) == 0:
                        continue

                    if self.is_validation:
                        # For validation, use what's available up to target
                        actual_n = min(len(pool), n_from_class)
                        if actual_n == 0:
                            continue
                    else:
                        if len(pool) < n_from_class:
                            # Pool exhausted for this class, end epoch
                            can_continue = False
                            break
                        actual_n = n_from_class

                    selected = [pool.pop() for _ in range(actual_n)]
                    selected_exps.extend(selected)
                    remaining -= actual_n

                if not can_continue or len(selected_exps) == 0:
                    break

                # For validation with MIL models, we need exactly n_experiments
                # Pad with repeated experiments if needed
                if self.is_validation and len(selected_exps) < self.n_experiments and len(selected_exps) > 0:
                    while len(selected_exps) < self.n_experiments:
                        # Pick from any available class (wrapped as a singleton unit)
                        all_available = []
                        for c in range(self.num_classes):
                            all_available.extend(self.class_experiments[c])
                        if all_available:
                            selected_exps.append([random.choice(all_available)])
                        else:
                            break

                random.shuffle(selected_exps)

                batch_idxs = []
                for unit in selected_exps:
                    batch_idxs.extend(self._sample_bag_for_unit(unit, samples_per_exp))

                batches_yielded += 1
                yield batch_idxs

    def _keep_experiment(self, exp, samples_per_exp):
        """Whether to retain an experiment given the per-bag sample count.

        Default (image MIL): drop experiments with fewer images than a bag needs.
        Overridden by EmbeddingBatchSampler to keep every experiment (no drops).
        """
        start, end = self.exp_ranges[exp]
        return (end - start) >= samples_per_exp

    def _sample_cells_for_exp(self, exp, samples_per_exp):
        """Pick global indices for one bag from one experiment.

        Default: uniform random sample of the experiment's range (with replacement only
        if the range is smaller than the bag). Overridden by EmbeddingBatchSampler for
        cell-type-ratio composition with backfill.
        """
        start, end = self.exp_ranges[exp]
        n_avail = end - start
        if n_avail >= samples_per_exp:
            return random.sample(range(start, end), samples_per_exp)
        return random.choices(range(start, end), k=samples_per_exp)

    def __len__(self):
        """Return the actual number of batches that will be yielded per epoch."""
        if self.task_type == "binary":
            num_neg = self.n_experiments - self.n_pos

            if self.is_validation:
                # For validation, we can form batches with smaller sizes
                # Allow batches with only pos OR only neg (e.g. LOO)
                if len(self.pos_units) == 0 and len(self.neg_units) == 0:
                    return 0
                # Estimate batches: use ceiling division since we use variable batch sizes
                max_from_pos = len(self.pos_units)  # Each batch uses 1 to n_pos
                max_from_neg = len(self.neg_units)  # Each batch uses 1 to num_neg
                # Conservative estimate: assume we use max per batch
                pool_limited = min(
                    max_from_pos // max(1, self.n_pos),
                    max_from_neg // max(1, num_neg)
                )
                # But we can always form at least 1 batch if we have 1 of each
                pool_limited = max(1, pool_limited)
            else:
                # Limited by whichever pool runs out first
                max_from_pos = len(self.pos_units) // self.n_pos if self.n_pos > 0 else float('inf')
                max_from_neg = len(self.neg_units) // num_neg if num_neg > 0 else float('inf')
                pool_limited = min(max_from_pos, max_from_neg)

            return min(self.num_batches, int(pool_limited))

        elif self.task_type == "multiclass":
            if self.is_validation:
                # For validation, we can form batches with whatever is available
                # As long as we have at least 1 experiment total
                total_exps = sum(len(self.class_units[c]) for c in range(self.num_classes))
                if total_exps == 0:
                    return 0
                # Estimate: we use up to n_experiments per batch
                pool_limited = max(1, total_exps // max(1, self.n_experiments))
            else:
                # Limited by smallest class pool
                min_batches = float('inf')
                for c in range(self.num_classes):
                    n_from_class = self.exps_per_class if c < self.num_classes - 1 else (self.n_experiments - self.exps_per_class * (self.num_classes - 1))
                    if n_from_class > 0 and len(self.class_units[c]) > 0:
                        max_batches_from_class = len(self.class_units[c]) // n_from_class
                        min_batches = min(min_batches, max_batches_from_class)
                pool_limited = min_batches if min_batches != float('inf') else 0

            return min(self.num_batches, int(pool_limited))

        return self.num_batches


class EmbeddingBatchSampler(BatchSampler):
    """BatchSampler for precomputed cell embeddings with weighted-by-true-composition bags.

    Two differences from the image BatchSampler:
      1. **No experiment is ever dropped** (``_keep_experiment`` always True).
      2. Each bag is composed by **weighted sampling over cell classes**. The per-class
         slot count is a multinomial draw with probability ``p_c ∝ weight_c * true_count_c``
         — the *true* (uncapped) per-experiment class counts, so cross-sample composition
         survives while ``weight_c`` enriches scarce-but-informative classes. The actual
         embeddings for each class's slots are drawn WITH replacement from that class's
         capped in-RAM pool (the cap only limits within-class morphological diversity, not
         composition). A class with ``weight_c * true_count_c == 0`` simply gets no slots.

    Inputs (from EmbeddingDataModule, shared by reference across train/val children):
      - ``class_local[exp] = {class_idx: LOCAL row idx array}`` into the experiment's block
        (global index = range_start + local; local indexing is split-invariant);
      - ``true_counts[exp] = [n_cell_classes] uncapped counts``;
      - ``weights = [n_cell_classes]`` per-class enrichment weights.
    """

    def __init__(self, exp_ranges, exp_labels, class_local, true_counts, weights,
                 batch_size, n_experiments, n_pos=4, num_batches=200,
                 task_type="binary", num_classes=2, is_validation=False,
                 cross_experiment=False, majority_min_pct=1.0, bag_label_mode="hard"):
        self.class_local = class_local
        self.true_counts = true_counts
        self.cell_weights = np.asarray(weights, dtype=np.float64)
        self.n_cell_classes = len(self.cell_weights)
        self.cross_experiment = cross_experiment
        # Mixed bags: each bag is majority_min_pct cells from a majority disease class +
        # the remainder spread across the OTHER disease classes, so every cell keeps its
        # own experiment's disease label and _forward_step's soft path reads the bag's
        # class proportion. Dilutes per-patient signatures (anti-memorization).
        self.majority_min_pct = majority_min_pct
        self.bag_label_mode = bag_label_mode
        if majority_min_pct < 1.0 and bag_label_mode != "soft":
            raise ValueError(
                "Embedding mixed bags (majority_min_pct < 1.0) require --bag_label_mode soft: "
                "a mixed bag's per-cell disease labels are not uniform, so a hard bag label "
                "is ill-defined. Use soft (LLP proportion) labels."
            )
        super().__init__(exp_ranges, exp_labels, batch_size=batch_size,
                         n_experiments=n_experiments, n_pos=n_pos, num_batches=num_batches,
                         task_type=task_type, num_classes=num_classes,
                         is_validation=is_validation)
        if self.cross_experiment:
            self._build_class_pools()

    def _disease_class_experiments(self):
        """{disease_class_idx: [experiment keys]} for the split this sampler sees."""
        if self.task_type == "multiclass":
            return {c: list(self.class_experiments.get(c, [])) for c in range(self.num_classes)}
        if self.task_type == "binary":
            return {0: list(self.neg_experiments), 1: list(self.pos_experiments)}

    def _build_class_pools(self):
        """Pool, per disease class, every train experiment's cells by cell type into one
        global-index array, plus the class's TRUE (uncapped) per-cell-type counts.

        Cross-experiment bags then draw from these class-wide pools, so a bag mixes cells
        from MANY patients of one class — breaking per-patient/per-experiment memorization.
        Composition still follows weight_t * true_count_t (true class composition).
        """
        self._pool_global = {}   # disease_class -> {cell_type: np.ndarray of global idx}
        self._pool_counts = {}   # disease_class -> np.ndarray[n_cell_classes] true counts
        for c, exps in self._disease_class_experiments().items():
            per_type = {}
            counts = np.zeros(self.n_cell_classes, dtype=np.float64)
            for t in range(self.n_cell_classes):
                arrs = []
                for e in exps:
                    start, _e = self.exp_ranges[e]
                    loc = self.class_local[e].get(t)
                    if loc is not None and len(loc):
                        arrs.append(start + loc)
                per_type[t] = np.concatenate(arrs) if arrs else np.empty(0, dtype=np.int64)
                counts[t] = sum(float(self.true_counts[e][t]) for e in exps)
            self._pool_global[c] = per_type
            self._pool_counts[c] = counts

    def _keep_experiment(self, exp, samples_per_exp):
        return True  # never drop

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        """Yield exactly ``num_batches`` bags/epoch, sampling experiments WITH replacement.

        Unlike the image BatchSampler (which drains experiment pools without replacement and
        ends the epoch early), embedding bags are cheap resamples of in-RAM cells — we want
        many fresh draws per epoch, so each batch independently picks a balanced set of
        experiments and composes cells for each.
        """
        spe = self.batch_size // self.n_experiments
        for _ in range(self.num_batches):
            if self.majority_min_pct < 1.0:
                # Mixed bags: each of n_experiments bags is a majority class + diluted
                # off-class cells. Per-cell labels stay heterogeneous so soft labels work.
                classes = self._bag_class_assignment()
                batch = []
                for c in classes:
                    batch.extend(self._mixed_bag(c, spe))
                if not batch:
                    break
                yield batch
            elif self.cross_experiment:
                # Each bag = cells pooled across ALL train experiments of one disease class.
                classes = self._bag_class_assignment()
                batch = []
                for c in classes:
                    batch.extend(self._pooled_class_bag(c, spe))
                if not batch:
                    break
                yield batch
            else:
                exps = self._select_experiments_for_batch()
                if not exps:
                    break
                batch = []
                for e in exps:
                    batch.extend(self._sample_cells_for_exp(e, spe))
                yield batch

    def _disease_classes(self):
        """List of disease-class indices for this task."""
        if self.task_type == "binary":
            return [0, 1]
        return list(range(self.num_classes))

    def _exps_of_class(self, c):
        """Experiment keys belonging to disease class c (split this sampler sees)."""
        if self.task_type == "multiclass":
            return list(self.class_experiments.get(c, []))
        return list(self.pos_experiments if c == 1 else self.neg_experiments)

    @staticmethod
    def _even_split(n, k):
        """Split n into k near-even nonnegative ints summing to n."""
        if k <= 0:
            return []
        base, rem = divmod(n, k)
        return [base + (1 if i < rem else 0) for i in range(k)]

    def _mixed_bag(self, c, spe):
        """One mixed bag: ~majority_min_pct cells from a random class-c experiment, the
        rest spread across one random experiment per OTHER disease class. Each cell keeps
        its source experiment's disease label (set in __getitem__), so the bag carries a
        soft class-proportion target. Returns up to spe global cell indices."""
        n_maj = min(spe, int(round(spe * self.majority_min_pct)))
        cells = []
        maj_exps = self._exps_of_class(c)
        if maj_exps and n_maj > 0:
            e = random.choice(maj_exps)
            cells.extend(self._sample_cells_for_exp(e, n_maj))
        others = [oc for oc in self._disease_classes()
                  if oc != c and self._exps_of_class(oc)]
        n_min = spe - len(cells)
        if others and n_min > 0:
            for oc, k in zip(others, self._even_split(n_min, len(others))):
                if k <= 0:
                    continue
                e = random.choice(self._exps_of_class(oc))
                cells.extend(self._sample_cells_for_exp(e, k))
        # Top up from the majority class if short (e.g. no other classes present, or
        # an off-class experiment had an empty pool).
        while len(cells) < spe and maj_exps:
            e = random.choice(maj_exps)
            add = self._sample_cells_for_exp(e, spe - len(cells))
            if not add:
                break
            cells.extend(add)
        return cells[:spe]

    def _bag_class_assignment(self):
        """Assign the batch's n_experiments bags to disease classes, balanced."""
        if self.task_type == "binary":
            classes = [1] * self.n_pos + [0] * (self.n_experiments - self.n_pos)
        elif self.task_type == "multiclass":
            per = [self.n_experiments // self.num_classes] * self.num_classes
            for i in range(self.n_experiments % self.num_classes):
                per[i] += 1
            classes = [c for c in range(self.num_classes) for _ in range(per[c])]
        else:
            classes = [0] * self.n_experiments
        random.shuffle(classes)
        return classes

    def _pooled_class_bag(self, c, k):
        """Draw k cells from disease-class c's pooled cells, cell-type weighted by the
        class's true composition. All cells share disease label c (uniform bag label)."""
        counts = self._pool_counts[c]
        probs = self.cell_weights * counts
        total = probs.sum()
        if total <= 0:
            return []
        n_per = np.random.multinomial(k, probs / total)
        out = []
        for t in range(self.n_cell_classes):
            n_t = int(n_per[t])
            pool = self._pool_global[c].get(t)
            if n_t <= 0 or pool is None or len(pool) == 0:
                continue
            out.extend(int(g) for g in np.random.choice(pool, size=n_t, replace=True))
        # Top up if a weighted cell type had an empty pool (degenerate guard).
        if len(out) < k:
            nonempty = [t for t in range(self.n_cell_classes) if len(self._pool_global[c].get(t, [])) > 0]
            while len(out) < k and nonempty:
                t = nonempty[np.random.randint(len(nonempty))]
                pool = self._pool_global[c][t]
                out.append(int(pool[np.random.randint(len(pool))]))
        return out[:k]

    def _select_experiments_for_batch(self):
        """Pick n_experiments experiments, class-balanced, with replacement across batches."""
        if self.task_type == "binary":
            pos, neg = self.pos_experiments, self.neg_experiments
            sel = [random.choice(pos) for _ in range(self.n_pos)] if pos else []
            n_neg = self.n_experiments - len(sel)
            sel += [random.choice(neg) for _ in range(n_neg)] if neg else []
            fill = pos or neg
            while len(sel) < self.n_experiments and fill:
                sel.append(random.choice(fill))
        elif self.task_type == "multiclass":
            per = [self.n_experiments // self.num_classes] * self.num_classes
            for i in range(self.n_experiments % self.num_classes):
                per[i] += 1
            sel = []
            for c in range(self.num_classes):
                pool = self.class_experiments.get(c, [])
                if pool:
                    sel += [random.choice(pool) for _ in range(per[c])]
            allpool = [e for c in range(self.num_classes) for e in self.class_experiments.get(c, [])]
            while len(sel) < self.n_experiments and allpool:
                sel.append(random.choice(allpool))
            random.shuffle(sel)
        return sel

    def _class_probs(self, exp):
        probs = self.cell_weights * self.true_counts[exp].astype(np.float64)
        total = probs.sum()
        if total <= 0:  # no cells in any weighted class — fall back to kept-pool sizes
            sizes = np.array([len(self.class_local[exp].get(i, []))
                              for i in range(self.n_cell_classes)], dtype=np.float64)
            total = sizes.sum()
            if total <= 0:
                return None
            return sizes / total
        return probs / total

    def _sample_cells_for_exp(self, exp, samples_per_exp):
        start, _end = self.exp_ranges[exp]
        pools = self.class_local[exp]
        probs = self._class_probs(exp)
        if probs is None:
            return []  # experiment has no kept cells (shouldn't reach: it'd be dropped)
        # Per-class slot counts: a single multinomial draw -> sums to samples_per_exp.
        n_per = np.random.multinomial(samples_per_exp, probs)
        out = []
        for ci in range(self.n_cell_classes):
            n_c = int(n_per[ci])
            if n_c <= 0:
                continue
            pool = pools.get(ci)
            if pool is None or len(pool) == 0:
                continue  # weighted but unkept (impossible when true_count>0); skip + top up
            sel = np.random.choice(pool, size=n_c, replace=True)  # with replacement
            out.extend(int(start + li) for li in sel)
        # Top up if a class with slots had an empty pool (degenerate guard).
        if len(out) < samples_per_exp:
            nonempty = [i for i in range(self.n_cell_classes) if len(pools.get(i, [])) > 0]
            while len(out) < samples_per_exp and nonempty:
                ci = nonempty[np.random.randint(len(nonempty))]
                li = pools[ci][np.random.randint(len(pools[ci]))]
                out.append(int(start + li))
        return out[:samples_per_exp]
