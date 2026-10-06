# Remo-ID

**Remo-ID** trains and evaluates deep-learning classifiers on Remoscope
data. One Remoscope acquisition (an *experiment*) is 20,000
brightfield frames from a single blood sample. Remo-ID predicts a label for each experiment:
binary or multiclass. It supports per-image classifiers and multiple-instance
learning (MIL) models that treat each experiment as a bag of images or of precomputed
per-cell embeddings.

This repository was used in the publication below. See [`examples/sepsis_paper/`](examples/sepsis_paper/)

> **Rapid Sepsis Diagnosis in Critically Ill Adults with Remoscope Label-Free Whole-Blood Imaging** Nicholas Bratvold, Ilakkiyan Jeyakumar, Michelle Khoo, Gregory Suematsu, Olivia Chao, Tal Shwartzman, Andrew Willmore, Julia Hutchinson, Carolyn S. Calfee, Michael A. Matthay, Natasha Spottiswoode, Charles R. Langelier* and Paul M. Lebel* *medRxiv* (link TBD). DOI TBD.

Developed at the [Chan Zuckerberg Biohub San Francisco](https://www.czbiohub.org/sf/) by the
Bioengineering group.

## Installation

Python 3.12; a CUDA GPU for training (CPU works for tests and small runs). Only tested on Linux.
`uv.lock` pins the dependency versions.

```bash
git clone https://github.com/czbiohub-sf/remoscope-id.git && cd remoscope-id
uv sync --extra dev
source .venv/bin/activate
remo-id --help
```

Runs log to [Weights & Biases](https://wandb.ai) via the standard `WANDB_*` environment
variables; `WANDB_MODE=disabled` turns it off.

## Quickstart

```bash
python examples/make_synthetic_data.py /tmp/remo_synthetic --embeddings

# MIL on images
remo-id train --manifest /tmp/remo_synthetic/manifest.csv --model resnet18-mil \
    --batch_size 16 --n_experiments 2 --crop_size 64 --max_epochs 2 --no-pretrained

# MIL on precomputed embeddings
remo-id train --manifest /tmp/remo_synthetic/manifest.csv --model embedding-mil \
    --model_version synthetic_v1 --cells_per_bag 32 --n_experiments 2 --max_epochs 2
```

Each run writes `<save_dir>/<run_id>_<model>/` with `best.ckpt`, `last.ckpt`, the class
definitions, the train/val splits and the post-training predictions. `--save_dir` defaults to
`$REMO_ID_SAVE_DIR` or `./checkpoints`.

## Data

**Images.** One directory per experiment holding a `uint8` zarr array `[H, W, N]` in a `.zip`
(zarr `ZipStore`). An optional `<dirname>background_p99.png` enables flat-field correction
to remove background batch effects acquisition to acquisition.

**Manifest.** A CSV listing the experiments ([example](examples/manifest_example.csv)),
passed with `--manifest`:

| column | required | meaning |
|---|---|---|
| `experiment_id` | yes | unique acquisition ID |
| `path` | yes | experiment directory (relative to the manifest) |
| `num_images` | yes | frames in the stack |
| `label` | yes | class name |
| `sample_id` | no | experiments with the same `sample_id` stay in the same split |
| any other column | no | kept as metadata (e.g. `study_id`, `scope`, clinical variables) |

Classes are sorted by name unless set with `--classes neg,pos` (in binary tasks the second is
positive). The class definitions are saved with the checkpoint (`*.classes.yaml`) and reused
by every downstream command via `--class_config`.

**Precomputed embeddings.** `--model embedding-mil` reads
`{path}/{experiment_id}_emb__{model_version}.zarr`. Write one from any `(N, D)` array:

```python
from remo_id.data.embedding_store import write_experiment_embeddings
write_experiment_embeddings(exp_path, str(exp_id), "my_model_v1", emb)
```

An optional `--cell_config` ([example](examples/cell_configs/wbc_rbc_plt.yaml)) gates cells into
classes using a per-cell `*regionprops.parquet` and weights them in each bag.

**Remoscope database.** Data can also be selected from the internal Remoscope database with
`--class_config` or `--study_id/--database_column/--database_label`. This needs the private
`remoscope-database` package, which is not a dependency of this repository.

It is strongly recommended to set up your own database to store your data. The manifest.csv was created as a workaround but organizing data into an opinionated and structured format allows for data processing pipelines, training, and analysis to be streamlined and requires no management of various floating csv files. It is simple to implement, will save many hours, and is very well documented online.

## Models

Here are some options, add more as needed.

| `--model` | type |
|---|---|
| `resnet18`, `resnet50`, `vit-b-16`, `vit-l-16`, `microscopy-cnn`, `dino-vits16`, `dino-vitsplus`, `dino-vitb` | instance |
| `resnet18-mil`, `resnet50-mil`, `vit-l-16-mil`, `microscopy-cnn-mil`, `dino-vits16-mil`, `dino-vitsplus-mil`, `dino-vitb-mil` | MIL on images |
| `embedding-mil` | MIL on embeddings |

MIL pooling (`--pooling`): `attention` (default), `mean`, `max`, `linear` (mi-Net), `topk`,
`transformer`, `additive`.

DINOv3 weights are not redistributed: download them from Meta and pass `--dino_weights`
(or set `REMO_ID_DINO_WEIGHTS`).

## Usage

`remo-id <command> --help` lists every option.

```bash
# train (bag size = batch_size / n_experiments)
remo-id train --manifest manifest.csv --model resnet50-mil \
    --batch_size 200 --n_experiments 4 --accumulate_grad_batches 3 --max_epochs 50

# cross-validation: --n_folds K, or leave-one-out with --loo
remo-id train --manifest manifest.csv --model resnet50-mil --loo
remo-id workflow loo-predict --cv_dir <cv_dir> --manifest manifest.csv --epoch 50 --bag_size 500

# inference and analysis
remo-id infer --checkpoint run/best.ckpt --manifest new_data.csv --bag_size 200 --output_path preds/
remo-id analysis all --checkpoint run/best.ckpt --class_config run/best.classes.yaml \
    --splits run/best.splits.yaml
```

- `remo-id analysis`: performance with bootstrap CIs, CV pooling, thresholds, pairwise,
  statistical tests, baselines, interpretation, and more.
- `remo-id analysis occlusion {background,attribution,screen,report}`: finds the cells a MIL
  model's prediction depends on by removing them from real frames. Needs per-frame cell masks
  and a regionprops table; see `remo_id/interpret/cell_occlusion.py`.
- `remo-id gallery`: Grad-CAM galleries. `remo-id sample`: preview augmented images.
- [`scripts/`](scripts/) has example SLURM wrappers; CV and LOO steps can run as array jobs
  (`--partition`).

## Tests

```bash
pytest -q
```

CPU only, synthetic data, no private packages.

## Data availability

The Remoscope image data and the manifest of the sepsis study will be released at
**(dataset link TBD)**.

## Ownership and license

Owned by the Bioengineering group, Chan Zuckerberg Biohub San Francisco. Maintainer: Nicholas
Bratvold ([@NicholasBratvold](https://github.com/NicholasBratvold)).

[MIT License](LICENSE) · [Code of Conduct](CODE_OF_CONDUCT.md) · [Security](SECURITY.md)

Cite if you want: link TBD.
