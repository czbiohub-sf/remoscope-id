#!/usr/bin/env python
"""Tests for remo_id/analysis/learning_curves.py (generate_learning_curve_sbatch only)"""

import os
import tempfile
from remo_id.analysis.learning_curves import generate_learning_curve_sbatch


def test_total_jobs():
    fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    n_seeds = 3
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_learning_curve_sbatch(tmpdir, fractions, n_seeds, "remo-id train --model resnet18")
        with open(path) as f:
            content = f.read()
        total = len(fractions) * n_seeds
        assert f"--array=0-{total - 1}" in content
    print("  \u2713 total jobs = len(fractions) * n_seeds")


def test_array_index_math():
    fractions = [0.1, 0.5, 1.0]
    n_seeds = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_learning_curve_sbatch(tmpdir, fractions, n_seeds, "remo-id train")
        with open(path) as f:
            content = f.read()
        assert f"FRAC_IDX=$((SLURM_ARRAY_TASK_ID / {n_seeds}))" in content
        assert f"SEED_IDX=$((SLURM_ARRAY_TASK_ID % {n_seeds}))" in content
    print("  \u2713 array index mapping: FRAC_IDX and SEED_IDX correct")


def test_seed_calculation():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_learning_curve_sbatch(tmpdir, [0.5], 3, "remo-id train")
        with open(path) as f:
            content = f.read()
        assert "SEED=$((403 + SEED_IDX * 100))" in content
    print("  \u2713 seed calculation: 403 + seed_idx * 100")


def test_script_file_and_directives():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_learning_curve_sbatch(tmpdir, [0.2, 0.8], 2, "remo-id train --model vit")
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "#SBATCH --job-name=" in content
        assert "#SBATCH --time=" in content
        assert "#SBATCH --gpus=1" in content
        assert "remo-id train --model vit" in content
    print("  \u2713 script exists with expected SBATCH directives")


if __name__ == "__main__":
    print("test_learning_curves.py")
    test_total_jobs()
    test_array_index_math()
    test_seed_calculation()
    test_script_file_and_directives()
    print("All tests passed.")
