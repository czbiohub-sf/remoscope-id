"""Sample-aware BatchSampler: one replicate per sample per epoch, uniform sample
usage regardless of replicate count, null sample_ids treated as independent.

Synthetic cohort with uneven technical replication:
  class 0: 12 samples -> 4 singletons + 8 duplicates = 20 exps,
           plus 2 null-sample_id experiments (unique samples)
  class 1: 22 samples -> 4 replicates each          = 88 exps
  class 2: 13 samples -> 5 singletons + 8 duplicates = 21 exps
"""
from collections import Counter, defaultdict

import pytest

from remo_id.data import BatchSampler

BATCH_SIZE, N_EXPERIMENTS = 200, 4
SAMPLES_PER_EXP = BATCH_SIZE // N_EXPERIMENTS
N_EPOCHS = 3000


def build_cohort():
    exp_ranges, exp_labels, exp_sample_ids = {}, {}, {}
    gidx = [0]

    def add(exp, label, sid, n_images=1000):
        exp_ranges[exp] = (gidx[0], gidx[0] + n_images)
        gidx[0] += n_images
        exp_labels[exp] = label
        exp_sample_ids[exp] = sid

    for i in range(4):
        add(f"a_s{i}_r0", 0, f"a_{i}")
    for i in range(4, 12):
        for r in range(2):
            add(f"a_s{i}_r{r}", 0, f"a_{i}")
    add("a_null_1", 0, None)
    add("a_null_2", 0, float("nan"))
    for i in range(22):
        for r in range(4):
            add(f"b_s{i}_r{r}", 1, f"b_{i}")
    for i in range(5):
        add(f"c_s{i}_r0", 2, f"c_{i}")
    for i in range(5, 13):
        for r in range(2):
            add(f"c_s{i}_r{r}", 2, f"c_{i}")
    return exp_ranges, exp_labels, exp_sample_ids


def exp_to_sample(exp, exp_sample_ids):
    sid = exp_sample_ids.get(exp)
    if sid is None or (isinstance(sid, float) and sid != sid):
        return f"__unique_exp_{exp}"
    return sid


def recover_exps(batch_idxs, exp_ranges):
    """Map yielded global indices back to the experiments they came from."""
    exps = []
    for i in range(0, len(batch_idxs), SAMPLES_PER_EXP):
        idx = batch_idxs[i]
        for e, (s, t) in exp_ranges.items():
            if s <= idx < t:
                exps.append(e)
                break
    return exps


def _sampler(exp_ranges, exp_labels, exp_sample_ids=None):
    return BatchSampler(exp_ranges, exp_labels, batch_size=BATCH_SIZE,
                        n_experiments=N_EXPERIMENTS, num_batches=100,
                        task_type="multiclass", num_classes=3,
                        exp_sample_ids=exp_sample_ids)


@pytest.fixture(scope="module")
def draws():
    """Run the sample-aware sampler for many epochs and tally draws."""
    exp_ranges, exp_labels, exp_sample_ids = build_cohort()
    s = _sampler(exp_ranges, exp_labels, exp_sample_ids)
    sample_draws, replicate_draws = Counter(), Counter()
    repeated_in_epoch = []
    for _ in range(N_EPOCHS):
        seen = set()
        for batch in s:
            for e in recover_exps(batch, exp_ranges):
                sid = exp_to_sample(e, exp_sample_ids)
                if sid in seen:
                    repeated_in_epoch.append(sid)
                seen.add(sid)
                sample_draws[sid] += 1
                replicate_draws[e] += 1
    return dict(sampler=s, exp_ranges=exp_ranges, exp_sample_ids=exp_sample_ids,
                sample_draws=sample_draws, replicate_draws=replicate_draws,
                repeated_in_epoch=repeated_in_epoch)


def test_units_are_samples_with_null_ids_independent(draws):
    unit_counts = {c: len(u) for c, u in draws["sampler"].class_units.items()}
    assert unit_counts == {0: 14, 1: 22, 2: 13}


def test_sample_never_drawn_twice_per_epoch(draws):
    assert draws["repeated_in_epoch"] == []


def test_samples_drawn_uniformly_within_class(draws):
    by_class = defaultdict(list)
    for sid, n in draws["sample_draws"].items():
        cls = "a" if sid.startswith(("a_", "__unique_exp_a")) else sid[0]
        by_class[cls].append(n)
    assert set(by_class) == {"a", "b", "c"}
    for cls, counts in by_class.items():
        spread = (max(counts) - min(counts)) / (sum(counts) / len(counts))
        assert spread < 0.10, f"class {cls}: sample draw counts not uniform: {sorted(counts)}"


def test_replicates_drawn_uniformly_within_sample(draws):
    per_sample = defaultdict(list)
    for e in draws["exp_ranges"]:
        per_sample[exp_to_sample(e, draws["exp_sample_ids"])].append(e)
    worst = 0.0
    for exps in per_sample.values():
        if len(exps) < 2:
            continue
        counts = [draws["replicate_draws"][e] for e in exps]
        worst = max(worst, (max(counts) - min(counts)) / (sum(counts) / len(counts)))
    assert worst < 0.20


def test_null_sample_id_experiments_are_drawn(draws):
    for e in ("a_null_1", "a_null_2"):
        assert draws["sample_draws"][f"__unique_exp_{e}"] > 0


def test_default_path_pools_experiments():
    exp_ranges, exp_labels, _ = build_cohort()
    s0 = _sampler(exp_ranges, exp_labels)
    assert {c: len(u) for c, u in s0.class_units.items()} == {0: 22, 1: 88, 2: 21}
    assert all(len(u) == 1 for units in s0.class_units.values() for u in units)
