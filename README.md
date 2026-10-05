# remo-id

**remo-id** trains and evaluates deep-learning classifiers on Remoscope blood-microscopy
data. One Remoscope acquisition (an *experiment*) is a stack of tens of thousands of
brightfield frames from a single blood sample. remo-id predicts a label for each experiment:
binary or multiclass. It supports per-image classifiers and multiple-instance
learning (MIL) models that treat each experiment as a bag of images or of precomputed
per-cell embeddings.

This repository is the code companion to the publication below.

> **TITLE TBD.** Authors TBD. *medRxiv* (link TBD). DOI TBD.

[`examples/sepsis_paper/`](examples/sepsis_paper/) shows how remo-id was configured and run for
the sepsis study (class config, training and LOO evaluation commands, the cell occlusion
screen, figure and table builders); use it as a template for your own data. The rest of the
repository is a general framework you can point at your own data.

Developed at the [Chan Zuckerberg Biohub San Francisco](https://www.czbiohub.org/sf/) by the
Bioengineering group.

## Contents

- [What it does](#what-it-does)
- [Installation](#installation)
- [Quickstart on synthetic data](#quickstart-on-synthetic-data)
- [Using your own data](#using-your-own-data)
- [Models](#models)
- [Training](#training)
- [Cross-validation](#cross-validation)
- [Inference, analysis and reports](#inference-analysis-and-reports)
  - [Occlusion analysis](#occlusion-analysis)
- [Running on a SLURM cluster](#running-on-a-slurm-cluster)
- [Data availability](#data-availability)
- [Repository layout](#repository-layout)
- [Tests](#tests)
- [Ownership](#ownership)
- [License and citation](#license-and-citation)

## What it does

- **Tasks:** binary and multiclass labels per experiment.
- **Instance models:** a CNN or vision transformer classifies single images. An experiment's
  prediction aggregates its images.
- **MIL models:** a backbone encodes a bag of images from one experiment, and a pooling head
  (gated attention, mean, max, linear/mi-Net, top-k, transformer or additive) predicts the
  experiment label.
- **Embedding-MIL:** the same pooling heads trained directly on precomputed per-cell
  embeddings from any embedding model. There is no image backbone, so it scales to millions
  of cells per experiment.
- **Data handling:**
  - Train/validation splits are grouped by sample, so technical replicates of the same
    blood sample never straddle the split.
  - Bags can be class-balanced, can mix experiments, can be sampled sample-aware, and can
    use soft labels.
  - Augmentation: microscopy-specific random crops at native resolution, flat-field
    correction, intensity jitter, noise, blur and random erasing.
- **Evaluation:**
  - k-fold and leave-one-out cross-validation with pooled held-out predictions.
  - Metrics with bootstrap confidence intervals, operating-point and threshold analysis,
    pairwise (one-vs-one) analysis and statistical tests.
  - Grad-CAM and attention interpretation, a cell occlusion analysis, and PDF/PNG reports.
- **Tracking:** experiments are tracked with [Weights & Biases](https://wandb.ai).

## Installation

Requirements:
- Linux.
- Python 3.12.
- A CUDA GPU for training. CPU works for tests and small runs (`--accelerator auto` picks the GPU when one is available).

[uv](https://docs.astral.sh/uv/) is recommended. `uv.lock` pins the exact dependency versions
used for the paper.

```bash
git clone https://github.com/czbiohub-sf/remo-id-public.git
cd remo-id-public
uv sync --extra dev            # creates .venv from uv.lock (includes pytest)
source .venv/bin/activate
remo-id --help
```

To use pip instead, run `pip install -e ".[dev]"`. The dependencies are declared in
`pyproject.toml`. If you need a specific CUDA build, install PyTorch first from
<https://pytorch.org>.

**W&B logging.** Runs log to Weights & Biases. Configure it with the standard environment
variables (`WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_BASE_URL` for a private server). Set
`WANDB_MODE=offline` or `WANDB_MODE=disabled` to run without an account.

## Quickstart on synthetic data

```bash
# 12 tiny synthetic experiments (2 classes) + manifest.csv + per-cell embedding stores
python examples/make_synthetic_data.py /tmp/remo_synthetic --embeddings

# MIL on images (ResNet-18 backbone, gated-attention pooling)
WANDB_MODE=offline remo-id train --manifest /tmp/remo_synthetic/manifest.csv \
    --model resnet18-mil --batch_size 16 --n_experiments 2 --crop_size 64 \
    --max_epochs 2 --no-pretrained --save_dir /tmp/remo_runs/mil

# MIL on precomputed embeddings
WANDB_MODE=offline remo-id train --manifest /tmp/remo_synthetic/manifest.csv \
    --model embedding-mil --model_version synthetic_v1 --cells_per_bag 32 \
    --n_experiments 2 --max_epochs 2 --save_dir /tmp/remo_runs/emb
```

Each run writes a `<run_id>_<model>/` folder inside `--save_dir` containing `best.ckpt` and `last.ckpt`. It also writes
`best.classes.yaml` (class definitions) and `best.splits.yaml` (train/validation experiment
IDs), plus post-training predictions. The default `--save_dir` is `$REMO_ID_SAVE_DIR`, or
`./checkpoints` if that is unset.

## Using your own data

### Image data layout

Each experiment is a directory holding one image stack: a `uint8` zarr array of shape
`[H, W, N]` stored in a `.zip` file (zarr `ZipStore`). Frame `i` is `array[:, :, i]`. You can
optionally add a flat-field background image `<dirname>background_p99.png` next to it.

```python
import zarr
store = zarr.storage.ZipStore("exp_001/exp_001.zip", mode="w")
arr = zarr.create_array(store=store, shape=frames.shape, dtype="uint8",
                        chunks=(frames.shape[0], frames.shape[1], 1))
arr[:] = frames            # uint8 [H, W, N]
store.close()
```

### Manifest CSV

A manifest lists the experiments and their labels. See
[`examples/manifest_example.csv`](examples/manifest_example.csv).

| column | required | meaning |
|---|---|---|
| `experiment_id` | yes | unique integer ID |
| `path` | yes | experiment directory (relative paths are resolved against the manifest's folder) |
| `num_images` | yes | number of frames in the stack |
| `label` | yes | class name (`--label_column` selects another column) |
| `sample_id` | no | blood-sample ID: experiments with the same `sample_id` always land in the same split |
| `study_id`, `site` | no | grouping metadata carried into reports |

Pass the manifest with `--manifest manifest.csv`.
- **Class order:** classes are sorted by name. Set the order explicitly with
  `--classes neg,pos`; in binary tasks the second class is the positive one.
- **Task type:** two classes give a binary task, more give a multiclass task.

The class definitions are saved with the checkpoint (`*.classes.yaml`). Every downstream
command (`infer`, `analysis`, `workflow`) can then reuse them through `--class_config`.

### Bring your own embeddings (embedding-MIL)

`--model embedding-mil` reads one store per experiment, colocated in the experiment
directory: `{experiment_path}/{experiment_id}_emb__{model_version}.zarr`. Any model that
gives an `(N, D)` array per experiment works:

```python
from remo_id.data.embedding_store import write_experiment_embeddings

for exp_id, exp_path in experiments:
    emb = my_encoder(cells_of(exp_path))            # (N, D) array, any embedding model
    write_experiment_embeddings(exp_path, str(exp_id), "my_model_v1", emb)
```

Then train with `--model embedding-mil --model_version my_model_v1 --cells_per_bag 500`.

Optionally, `--cell_config cfg.yaml` groups cells into classes (for example WBC / RBC /
platelet) and samples bags with per-class weights.
- The classes are defined by gates on a per-cell `*regionprops.parquet` table in the
  experiment directory.
- Each embedding's row in that table is given by `regionprops_row`.
- See [`examples/cell_configs/wbc_rbc_plt.yaml`](examples/cell_configs/wbc_rbc_plt.yaml).

Without a cell config, all embeddings form a single pool (capped at `--pool_cap`). Never mix
embeddings from different models in one run: the loader checks that every store has the
same `model_config_hash`.

### Remoscope database (CZ Biohub internal)

Inside CZ Biohub, data can also be selected straight from the internal Remoscope database
with `--class_config classes.yaml` or with `--study_id/--database_column/--database_label`.
This needs the private `remoscope-database` package, which is not a dependency of this
repository. With a manifest you never need it.

## Models

| `--model` | type | backbone |
|---|---|---|
| `resnet18`, `resnet50` | instance | torchvision ResNet (ImageNet-pretrained by default) |
| `vit-b-16`, `vit-l-16` | instance | torchvision ViT |
| `microscopy-cnn` | instance | small CNN trained from scratch |
| `dino-vits16`, `dino-vitsplus`, `dino-vitb` | instance | Meta DINOv3 ViT |
| `resnet18-mil`, `resnet50-mil` | MIL | ResNet |
| `vit-l-16-mil` | MIL | ViT-L/16 |
| `microscopy-cnn-mil` | MIL | small CNN |
| `dino-vits16-mil`, `dino-vitsplus-mil`, `dino-vitb-mil` | MIL | DINOv3 |
| `embedding-mil` | MIL on embeddings | none (precomputed embeddings) |

The MIL pooling head is chosen with `--pooling`:

| `--pooling` | what it does |
|---|---|
| `attention` | gated multi-head attention (default) |
| `mean` | average the instance features, then classify |
| `max` | max-pool the instance features, then classify |
| `linear` | classify each instance and average the logits (mi-Net) |
| `topk` | attention over the top-k instances (`--top_k`) |
| `transformer` | self-attention encoder over instances with a CLS readout (`--transformer_layers`) |
| `additive` | per-instance signed contributions summed exactly (interpretable) |

**DINOv3 weights** are released by Meta under the DINOv3 License and are not redistributed
here.
- Download a checkpoint from Meta and pass it with `--dino_weights /path/to/weights.pth`, or
  set `REMO_ID_DINO_WEIGHTS`.
- The DINOv3 code is fetched with `torch.hub` from GitHub. To use a local clone instead,
  set `REMO_ID_DINO_REPO`.
- Use `--no-pretrained` to train from random weights.
- Fine-tune DINO blocks with `--finetune_layers block10,block11`; the ResNet default is
  `layer3,layer4`.

## Training

`remo-id train --help` lists every option. The most important ones:

| option | default | meaning |
|---|---|---|
| `--batch_size` | 200 | images (or cells) per training step |
| `--n_experiments` | 8 | bags per step for MIL; bag size = `batch_size // n_experiments` |
| `--accumulate_grad_batches` | 1 | gradient accumulation |
| `--num_batches` | 100 | steps per epoch |
| `--max_epochs` | 100 | training epochs |
| `--lr`, `--weight_decay`, `--dropout` | 1e-4, 0.01, 0.3 | AdamW and regularisation |
| `--lr_schedule`, `--warmup_epochs` | cosine, 5 | learning-rate schedule |
| `--finetune_layers` | `layer3,layer4` | backbone layers to train (`all`, `none`, or a list) |
| `--crop_size` | backbone default | random-crop size at native resolution (no resizing) |
| `--early_stopping_patience` | 15 | 0 disables early stopping |
| `--save_epochs` | none | extra epochs to checkpoint, e.g. `0,10,25,50` |
| `--val_ratio` / `--val_experiment_ids` | 0.2 | random validation split, or an explicit one |
| `--cross_experiment_sampling` | off | mix images from several experiments per bag |
| `--sample_aware_sampling` | off | draw one technical replicate per sample per epoch |
| `--majority_min_pct`, `--bag_label_mode` | 1.0, hard | mixed bags with hard or soft (proportion) labels |
| `--intensity_jitter`, `--noise_std`, `--blur_sigma`, `--background_correction` | off | augmentations |
| `--precision` | auto | bf16 on Ampere+, fp16 on older GPUs, fp32 on CPU |
| `--permute_labels` | off | negative control: shuffle labels across samples (performance should drop to chance) |

Example: a three-class attention-MIL with a ResNet-50 backbone.

```bash
remo-id train --manifest manifest.csv --classes healthy,icu,sepsis \
    --model resnet50-mil --pooling attention \
    --batch_size 200 --n_experiments 4 --accumulate_grad_batches 3 \
    --max_epochs 50 --early_stopping_patience 0
```

## Cross-validation

- **k-fold:** `--n_folds K --n_repeats R` runs stratified, sample-grouped k-fold.
  `remo-id workflow cv` runs the whole pipeline: train, infer, pool, analyze.
- **Leave-one-out:** `--loo` makes every experiment (or sample) its own validation fold.
  Early stopping is turned off.
  - `remo-id workflow loo-predict --epoch N --bag_size B` predicts every fold with the same
    fixed-epoch checkpoint. Use the same epoch for all folds; a run trained without
    `--save_epochs` uses each fold's `last.ckpt`.
  - `remo-id workflow loo-report` turns the pooled predictions into tables and figures.
- On a SLURM cluster, CV folds and LOO prediction epochs run as array jobs. Set the
  partition with `--partition`.

## Inference, analysis and reports

```bash
remo-id infer --checkpoint run/best.ckpt --manifest new_data.csv --bag_size 200 --output_path preds/
remo-id analysis all --checkpoint run/best.ckpt --class_config run/best.classes.yaml \
    --splits run/best.splits.yaml
remo-id analysis cv-performance --cv_dir run/<timestamp>_cv_resnet50-mil_5fold_1repeat/ \
    --class_config run/<timestamp>_cv_resnet50-mil_5fold_1repeat/repeat0_fold0/best.classes.yaml
remo-id report performance --checkpoint run/best.ckpt     # CPU-only, from saved predictions
```

- **Model settings:** `infer` reads the model name, task type and class count from the
  checkpoint.
- **Analysis subcommands:** `performance`, `cv-performance`, `bag-size`, `interpret`,
  `robustness`, `baselines`, `cv-baselines`, `stats-test`, `ablation`, `error-analysis`,
  `learning-curves`, `optimal-thresholds`, `pairwise`, `cascade`, `moe`, `compare-configs`.
  Each has a `--help`.
- **Galleries:** `remo-id gallery` renders Grad-CAM crop galleries
  (`remo_id/interpret/gallery.py`).
- **Augmentation preview:** `remo-id sample` previews the augmented training images.

### Occlusion analysis

`remo-id analysis occlusion` finds the individual cells a MIL model's prediction depends on.
It removes cells from real frames, fills them with the experiment's empty-slide background,
and measures the change in the true-class log-odds. Every stage scores the given experiments
with ONE `--checkpoint`. The experiments come from `--manifest` or `--class_config`
(default: `classes.yaml` next to the checkpoint), optionally restricted with repeated
`--experiment_ids`. Explain experiments with a model that did not train on them, e.g. the
validation split of a run. For a cross-validation run, call each stage once per fold with that
fold's checkpoint and held-out `--experiment_ids`; all folds can share one `--work_dir`
([`examples/sepsis_paper/occlusion_fold.py`](examples/sepsis_paper/occlusion_fold.py) does this
for a LOO run, one SLURM array task per fold).

| stage | does | output in `--work_dir` |
|---|---|---|
| `background` | empty-slide background model per experiment (CPU) | `bg/` |
| `attribution` | per-cell signed Grad-CAM of the true-class log-odds (ResNet MIL) | `attribution/` |
| `screen` | stage 1: random multi-cell removals + ridge estimate of every cell's effect; stage 2: exact single-cell removals with a sham (the cell's own pixels restored) and equal-area nulls | `screen/stage1/`, `screen/stage2/` |
| `report` | per-cell table and the top cells per class (remove minus sham, beating the null) | `report/screen_cells.parquet`, `report/screen_top_manifest.csv` |

```bash
OCC="--checkpoint run/<run_id>_resnet50-mil/last.ckpt --manifest manifest.csv \
     --experiment_ids 7 --experiment_ids 9 --work_dir occlusion/"
remo-id analysis occlusion background $OCC                           # CPU
remo-id analysis occlusion attribution $OCC --cell_config cells.yaml   # GPU
remo-id analysis occlusion screen $OCC --cell_config cells.yaml        # GPU
remo-id analysis occlusion report --work_dir occlusion/              # CPU
```

The screen needs the background and attribution outputs. Every stage records each
experiment in `{work_dir}/experiments/` and resumes per experiment (`--overwrite`
recomputes). Every random draw is seeded (`--seed`), so a rerun gives the same cells and
rankings; the GPU type and the number of BLAS threads can change the last floating-point
digits. Use one GPU type for all `screen` runs whose outputs you compare or pool.

**Data requirements.** Besides the image stack, each experiment directory needs:
- `<dirname>masks.zarr` (or the only `*masks.zarr`): an integer zarr array `[H, W, N]`, the
  same shape as the image stack. Frame `i`'s label image is `array[:, :, i]`, 0 is
  background and `k > 0` is cell `k` of that frame.
- `<dirname>regionprops.parquet` (or the only `*regionprops.parquet`): one row per cell with
  `image_id`, `label` (its mask label in that frame), `area`, `centroid_y`, `centroid_x`, the
  bounding box `ty, tx, by, bx` (skimage `bbox` order), and the columns your
  `--cell_config` gates use.
- The model must score full frames so the masks align with its input. Image MIL backbones
  with global pooling do this.

`python examples/make_synthetic_data.py <dir> --masks` writes data in this format. The
building blocks are reusable on their own: the fill model in `remo_id/interpret/occlusion.py`,
the screen and the rule-based cell and frame selection in
`remo_id/interpret/cell_occlusion.py`, and per-class Grad-CAM with a class-colour tint in
`remo_id/interpret/gradcam.py` (`frame_class_gradcams`) and `overlays.py`
(`class_tint_overlay`).

## Running on a SLURM cluster

Nothing in remo-id requires SLURM. [`scripts/`](scripts/) contains example wrappers:
- `submit_cmd.sh`: GPU.
- `submit_preempted.sh`: preemptible GPU with requeue.
- `submit_cpu_cmd.sh`: CPU.
- `generate_hp_sweep.py` + `run_sweep_row.py`: hyperparameter-sweep array generator.

Adjust the partitions and resources for your cluster:

```bash
sbatch scripts/submit_cmd.sh remo-id train --manifest manifest.csv --model resnet18-mil
```

## Data availability

The Remoscope image data and the cohort table of the sepsis study will be released at
**(dataset link TBD)**. The clinical metadata is not public. [`examples/sepsis_paper/README.md`](examples/sepsis_paper/README.md)
lists which outputs need it.

## Repository layout

| path | contents |
|---|---|
| `remo_id/models/` | instance, MIL and embedding-MIL Lightning modules, model registry |
| `remo_id/data/` | datamodules, manifest and zarr readers, embedding store, bag samplers, class and cell configs |
| `remo_id/training/` | training entry point, config, k-fold / LOO splitting |
| `remo_id/inference/` | checkpoint loading, bag prediction, prediction files |
| `remo_id/analysis/` | metrics, CV pooling, thresholds, statistical tests, baselines, cascade |
| `remo_id/interpret/` | Grad-CAM, ViT attention rollout, overlays, Grad-CAM galleries, cell occlusion analysis |
| `remo_id/reports/` | figure style, figures, tables, PDF reports |
| `remo_id/workflows/` | multi-step CV / LOO pipelines (SLURM array aware) |
| `remo_id/cli.py` | the `remo-id` command-line interface |
| `examples/` | synthetic-data generator, example manifest, class and cell configs |
| `examples/sepsis_paper/` | worked example: how the sepsis study was configured and run (template for your own data) |
| `scripts/` | example SLURM wrappers and the sweep generator |
| `tests/` | pytest suite (CPU, synthetic data) |

## Tests

```bash
uv sync --extra dev
pytest -q
```

The suite runs on CPU with synthetic data, including end-to-end training and inference for
image MIL, instance models and embedding-MIL. It needs no private packages.

## Ownership

- **Owner:** Bioengineering group, Chan Zuckerberg Biohub San Francisco
  (group leader: **TBD**).
- **Maintainer:** Nicholas Bratvold ([@NicholasBratvold](https://github.com/NicholasBratvold)).
- **Possible heirs:** TBD.

## License and citation

Released under the [MIT License](LICENSE). Please read our [Code of Conduct](CODE_OF_CONDUCT.md)
and report security issues as described in [SECURITY.md](SECURITY.md).

If you use this code, please cite the publication above (citation TBD).
