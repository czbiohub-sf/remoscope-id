"""Cohesive dummy dataset for figure visual iteration.

Single class that constructs a 4-class, 60-experiment dataset and lazily
derives all data formats needed by every figure function.
"""

import numpy as np
import pandas as pd
from functools import cached_property


class DummyDataset:
    """Reproducible synthetic dataset covering all figure function inputs.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_experiments : int
        Total number of experiments (default 60).
    """

    CLASS_NAMES = ["disease+", "healthy", "infection+", "other"]
    CLASS_DIST = [18, 18, 14, 10]  # slight imbalance
    STUDY_IDS_BY_CLASS = ["study_a", "study_b", "study_c", "study_d"]

    def __init__(self, seed: int = 42, n_experiments: int = 60):
        self.seed = seed
        self.n_experiments = n_experiments
        self.n_classes = len(self.CLASS_NAMES)
        self.rng = np.random.default_rng(seed)

        # Build experiment table
        self._build_experiments()

    def _build_experiments(self):
        """Construct experiment metadata with sample_id replicates."""
        rng = self.rng

        # Assign class indices
        class_idx = []
        for ci, count in enumerate(self.CLASS_DIST):
            class_idx.extend([ci] * count)
        class_idx = np.array(class_idx[:self.n_experiments])

        # Assign study_ids
        study_ids = [self.STUDY_IDS_BY_CLASS[c] for c in class_idx]

        # Assign sample_ids with replicates (~80% grouped, 1-4 replicates)
        sample_ids = []
        sample_counter = 0
        i = 0
        while i < self.n_experiments:
            if rng.random() < 0.8:
                n_reps = int(rng.integers(2, 5))  # 2-4 replicates
            else:
                n_reps = 1
            n_reps = min(n_reps, self.n_experiments - i)
            for _ in range(n_reps):
                sample_ids.append(f"S{sample_counter:03d}")
            sample_counter += 1
            i += n_reps

        sample_ids = sample_ids[:self.n_experiments]

        # Train/val split by sample_id (80/20)
        unique_samples = list(dict.fromkeys(sample_ids))  # preserve order
        rng.shuffle(unique_samples)
        n_val = max(1, int(len(unique_samples) * 0.2))
        val_samples = set(unique_samples[:n_val])
        splits = ["val" if s in val_samples else "train" for s in sample_ids]

        # num_images per experiment
        num_images = rng.integers(30, 200, size=self.n_experiments)

        self.experiment_ids = np.arange(1000, 1000 + self.n_experiments)
        self.class_idx = class_idx
        self.sample_ids = sample_ids
        self.study_id_list = study_ids
        self.split_list = splits
        self.num_images = num_images

    # ------------------------------------------------------------------
    # Multiclass classification outputs
    # ------------------------------------------------------------------

    @cached_property
    def y_true(self) -> np.ndarray:
        """[N] int class indices."""
        return self.class_idx.copy()

    @cached_property
    def y_probs(self) -> np.ndarray:
        """[N, C] softmax probabilities with ~75% accuracy."""
        rng = np.random.default_rng(self.seed + 1)
        n = self.n_experiments
        C = self.n_classes
        # Start with noise, then boost the true class
        raw = rng.dirichlet(np.ones(C) * 0.5, size=n)
        for i in range(n):
            raw[i, self.y_true[i]] += rng.uniform(1.0, 3.0)
        # Normalize to sum to 1
        probs = raw / raw.sum(axis=1, keepdims=True)
        return probs

    @cached_property
    def y_pred(self) -> np.ndarray:
        """[N] argmax predictions."""
        return self.y_probs.argmax(axis=1)

    # ------------------------------------------------------------------
    # Binary classification view (first 2 classes)
    # ------------------------------------------------------------------

    @cached_property
    def _binary_mask(self) -> np.ndarray:
        return self.y_true < 2

    @cached_property
    def y_true_binary(self) -> np.ndarray:
        """[M] binary labels (0 or 1)."""
        return self.y_true[self._binary_mask]

    @cached_property
    def y_probs_binary(self) -> np.ndarray:
        """[M, 2] binary probabilities."""
        p = self.y_probs[self._binary_mask][:, :2]
        return p / p.sum(axis=1, keepdims=True)

    @cached_property
    def y_pred_binary(self) -> np.ndarray:
        return self.y_probs_binary.argmax(axis=1)

    @cached_property
    def binary_class_names(self) -> list:
        return self.CLASS_NAMES[:2]

    # ------------------------------------------------------------------
    # Metadata lists
    # ------------------------------------------------------------------

    @cached_property
    def splits(self) -> list:
        return list(self.split_list)

    @cached_property
    def study_ids(self) -> list:
        return list(self.study_id_list)

    # ------------------------------------------------------------------
    # Seed variability (comparison.py)
    # ------------------------------------------------------------------

    @cached_property
    def metrics_by_seed(self) -> dict:
        rng = np.random.default_rng(self.seed + 10)
        return {
            "accuracy": (rng.normal(0.82, 0.03, size=5)).tolist(),
            "f1_macro": (rng.normal(0.78, 0.04, size=5)).tolist(),
            "auc_macro": (rng.normal(0.90, 0.02, size=5)).tolist(),
            "precision_macro": (rng.normal(0.80, 0.03, size=5)).tolist(),
            "recall_macro": (rng.normal(0.79, 0.04, size=5)).tolist(),
        }

    # ------------------------------------------------------------------
    # K-fold variability (comparison.py)
    # ------------------------------------------------------------------

    @cached_property
    def fold_metrics(self) -> dict:
        rng = np.random.default_rng(self.seed + 11)
        n_folds = 10
        return {
            "accuracy": (rng.normal(0.80, 0.05, size=n_folds)).tolist(),
            "f1_macro": (rng.normal(0.76, 0.06, size=n_folds)).tolist(),
            "auc_macro": (rng.normal(0.88, 0.04, size=n_folds)).tolist(),
        }

    # ------------------------------------------------------------------
    # Bag-size sweep (data.py)
    # ------------------------------------------------------------------

    @cached_property
    def bag_sizes(self) -> list:
        return [10, 25, 50, 100, 200, 400]

    @cached_property
    def aucs_by_seed(self) -> np.ndarray:
        """[3 seeds, 6 bag sizes] — AUC increases with bag size."""
        rng = np.random.default_rng(self.seed + 12)
        base = np.array([0.72, 0.80, 0.86, 0.90, 0.92, 0.93])
        return base[None, :] + rng.normal(0, 0.02, size=(3, 6))

    # ------------------------------------------------------------------
    # Sampling position (data.py)
    # ------------------------------------------------------------------

    @cached_property
    def positions(self) -> list:
        return ["random", "start", "middle", "end", "distributed"]

    @cached_property
    def position_aucs(self) -> np.ndarray:
        """[5 positions, 3 seeds]."""
        rng = np.random.default_rng(self.seed + 13)
        base = np.array([0.91, 0.85, 0.87, 0.84, 0.90])
        return base[:, None] + rng.normal(0, 0.02, size=(5, 3))

    # ------------------------------------------------------------------
    # Learning curve (data.py)
    # ------------------------------------------------------------------

    @cached_property
    def learning_curve_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + 14)
        fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        rows = []
        for f in fractions:
            for seed in range(3):
                loss = 0.8 - 0.5 * f + rng.normal(0, 0.03)
                rows.append({"fraction": f, "seed": seed, "best_val_loss": loss})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Dataset statistics (data.py)
    # ------------------------------------------------------------------

    @cached_property
    def stats_df(self) -> pd.DataFrame:
        rows = []
        for ci, name in enumerate(self.CLASS_NAMES):
            mask = self.y_true == ci
            n_exp = int(mask.sum())
            n_img = int(self.num_images[mask].sum())
            rows.append({"class": name, "n_experiments": n_exp, "n_images": n_img})
        return pd.DataFrame(rows)

    @cached_property
    def imbalance(self) -> dict:
        counts = {}
        for ci, name in enumerate(self.CLASS_NAMES):
            counts[name] = int((self.y_true == ci).sum())
        max_c = max(counts.values())
        min_c = min(counts.values())
        return {
            "class_counts": counts,
            "imbalance_ratio": max_c / max(1, min_c),
        }

    @cached_property
    def replicate_df(self) -> pd.DataFrame:
        from collections import Counter
        counts = Counter(self.sample_ids)
        return pd.DataFrame([
            {"sample_id": sid, "n_experiments": c}
            for sid, c in counts.items()
        ])

    # ------------------------------------------------------------------
    # Baseline comparison (comparison.py)
    # ------------------------------------------------------------------

    @cached_property
    def baseline_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "model": ["Fine-tuned MIL", "Pretrained+Linear", "Linear", "MajorityClass"],
            "auc_macro": [0.92, 0.84, 0.72, 0.50],
            "f1_macro": [0.78, 0.68, 0.55, 0.10],
        })

    # ------------------------------------------------------------------
    # Error analysis (data.py)
    # ------------------------------------------------------------------

    @cached_property
    def error_df(self) -> pd.DataFrame:
        correct = self.y_true == self.y_pred
        rows = []
        for i in range(self.n_experiments):
            rows.append({
                "experiment_id": int(self.experiment_ids[i]),
                "study_id": self.study_ids[i],
                "true_class_idx": int(self.y_true[i]),
                "predicted_class_idx": int(self.y_pred[i]),
                "correct": bool(correct[i]),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Experiment probabilities / repeatability (comparison.py)
    # ------------------------------------------------------------------

    @cached_property
    def mean_probs(self) -> np.ndarray:
        """[N] mean positive probability per experiment (binary view)."""
        return self.y_probs_binary[:, 1]

    @cached_property
    def std_probs(self) -> np.ndarray:
        """[N] std of positive probability across runs."""
        rng = np.random.default_rng(self.seed + 20)
        return rng.uniform(0.01, 0.08, size=len(self.y_true_binary))

    # ------------------------------------------------------------------
    # Attention grid (interpretation.py)
    # ------------------------------------------------------------------

    @cached_property
    def frames_by_class(self) -> list:
        """list[list[dict]] — 10 frames per class with 32x32 random images."""
        rng = np.random.default_rng(self.seed + 40)
        result = []
        for ci in range(self.n_classes):
            frames = []
            bag_min = rng.uniform(0, 0.1)
            bag_max = rng.uniform(0.8, 1.0)
            for j in range(10):
                frames.append({
                    "image": rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8),
                    "raw_attn": float(rng.uniform(bag_min, bag_max)),
                    "bag_attn_min": float(bag_min),
                    "bag_attn_max": float(bag_max),
                    "experiment_id": f"exp_{ci}_{j}",
                })
            result.append(frames)
        return result

    # ------------------------------------------------------------------
    # UMAP embeddings (interpretation.py)
    # ------------------------------------------------------------------

    @cached_property
    def embeddings(self) -> np.ndarray:
        """[N, 64] clustered embeddings for visible UMAP separation."""
        rng = np.random.default_rng(self.seed + 50)
        dim = 64
        centroids = rng.normal(0, 3, size=(self.n_classes, dim))
        emb = np.zeros((self.n_experiments, dim))
        for i in range(self.n_experiments):
            emb[i] = centroids[self.y_true[i]] + rng.normal(0, 0.5, size=dim)
        return emb

    @cached_property
    def embedding_labels(self) -> np.ndarray:
        return self.y_true.copy()

    # ------------------------------------------------------------------
    # Attention entropy (interpretation.py)
    # ------------------------------------------------------------------

    @cached_property
    def entropy_by_class(self) -> list:
        """list[list[float]] — entropy values per class."""
        rng = np.random.default_rng(self.seed + 60)
        result = []
        for ci in range(self.n_classes):
            n = int((self.y_true == ci).sum())
            base = 2.0 + ci * 0.3  # different entropy levels per class
            result.append((rng.normal(base, 0.4, size=n)).tolist())
        return result

    # ------------------------------------------------------------------
    # Optimal thresholds (comparison.py)
    # ------------------------------------------------------------------

    @cached_property
    def optimal_results(self) -> list:
        """list[dict] for plot_optimal_thresholds."""
        return [
            {
                "target_sensitivity": 0.90,
                "optimal_threshold": 0.35,
                "achieved_sensitivity": 0.91,
                "achieved_specificity": 0.72,
            },
            {
                "target_sensitivity": 0.90,
                "optimal_threshold": 0.30,
                "achieved_sensitivity": 0.92,
                "achieved_specificity": 0.68,
            },
            {
                "target_sensitivity": 0.90,
                "optimal_threshold": 0.40,
                "achieved_sensitivity": 0.88,
                "achieved_specificity": 0.75,
                "warning": "Target sensitivity not achieved",
            },
            {
                "target_sensitivity": 0.90,
                "optimal_threshold": 0.32,
                "achieved_sensitivity": 0.90,
                "achieved_specificity": 0.70,
            },
        ]
