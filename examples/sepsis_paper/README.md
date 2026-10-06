# Example: sepsis study workflow

This folder shows how remo-id was configured and run for the sepsis study that accompanies

> **Rapid Sepsis Diagnosis in Critically Ill Adults with Remoscope Label-Free Whole-Blood Imaging** Nicholas Bratvold, Ilakkiyan Jeyakumar, Michelle Khoo, Gregory Suematsu, Olivia Chao, Tal Shwartzman, Andrew Willmore, Julia Hutchinson, Carolyn S. Calfee, Michael A. Matthay, Natasha Spottiswoode, Charles R. Langelier* and Paul M. Lebel*. medRxiv (link TBD), DOI TBD.

It holds the class and cell configs, the training and leave-one-out (LOO) evaluation
commands, the cell occlusion screen and the figure and table builders that generated the
figures in the publication. Use it as a template for your own data.

The study's model is a resnet50 attention multiple-instance-learning (MIL) classifier. It
treats each Remoscope acquisition (experiment) as a bag of full-resolution brightfield frames
and classifies it as Sepsis, ICU control or Healthy. It was evaluated by patient-grouped LOO
cross-validation: 47 folds, one patient per fold, 129 experiments. The analysis use the held-out
predictions made by one fixed-epoch checkpoint of each fold.

## Data availability

The image data and the cohort data are distributed with the data release (link TBD).

## Layout

| Path | What it does |
|---|---|
| `configs/sepsis_icu_healthy.yaml` | class config used for training and every analysis (3 classes) |
| `configs/sepsis_icu_healthy_ohmu.yaml` | the same classes restricted to the lfm-ohmu scope|
| `configs/wbc_rbc_plt.yaml` | cell typing (WBC / RBC / platelet regionprops gates) |
| `common.py` | shared helpers; importing it registers the study's class colours |
| `make_paper_results.py` | one command for every figure and table |
| `figures_paper.py` | the output builders called by `make_paper_results.py` |
| `configs/clinical_tables.yaml` | spec of the clinical outputs |
| `build_manifest.py` | writes the manifest from the database (the only script that reads it) |
| `umap_images.py` | per-image UMAP |
| `infer_indeterminate.py` | LOO-ensemble predictions for the patients without an adjudicated label|
| `occlusion_fold.py` | runs one `remo-id analysis occlusion` stage on one LOO fold's held-out experiments with that fold's checkpoint (one SLURM array task per fold) |
| `run_all.sh` | the whole pipeline as sbatch commands, in stages |

## Setup

```bash
git clone https://github.com/czbiohub-sf/remoscope-id.git && cd remoscope-id
uv sync --extra dev --extra examples
source .venv/bin/activate
```

The commands below select data with the class config, which needs the internal
`remoscope-database` package. Without remoscope-database, pass the released manifest with
`--manifest manifest.csv --classes ICU-ctrl,healthy,sepsis` in place of `--class_config $CFG`
(the classes in the class config's order, which is the trained models' class order). The
example scripts read the manifest directly (`--manifest`); `run_all.sh` does both with `NO_DB=1`.

Run every command from the repository root. All compute goes through SLURM. The example
wrappers live in `scripts/submit_cmd.sh` (GPU) and `scripts/submit_cpu_cmd.sh` (CPU); override
resources on the `sbatch` command line. `run_all.sh` contains every command below,
ready to submit.

## How the study pipeline was run

Variables used below: `CFG=examples/sepsis_paper/configs/sepsis_icu_healthy.yaml`, `CV=<the LOO folder
created by training>`.

### 1. Train (47 LOO folds)

```bash
python -m remo_id.cli train --class_config $CFG --model resnet50-mil --pooling attention \
  --batch_size 200 --n_experiments 4 --accumulate_grad_batches 3 \
  --lr 1e-4 --lr_schedule cosine --warmup_epochs 5 \
  --max_epochs 50 --early_stopping_patience 0 \
  --num_heads 1 --attn_dim 128 --dropout 0.3 --weight_decay 0.01 \
  --finetune_layers layer3,layer4 --pretrained \
  --no-infer --loo --save_dir <SAVE_DIR>
```

Run this from a submit host, outside a SLURM job. It only reads metadata: it writes
`<SAVE_DIR>/<timestamp>_cv_resnet50-mil_loo/` (this is `$CV`) with `cv_splits.yaml`, then submits
one GPU array task per fold. Each fold trains for 50 epochs with no early stopping, and its
`last.ckpt` is the epoch-50 model. Each task is run with a >80 GB GPU, 14 CPUs and 128 GB RAM.

### 2. Held-out predictions and embeddings at epoch 50 (GPU)

Every step uses `--epoch 50` and the same seed. Training used bag 50; every analysis uses
bag 500. The bag size is always passed explicitly, because checkpoints do not store it.

| Step | Command (per array task) | Array | Output |
|---|---|---|---|
| pooled predictions | `remo-id workflow loo-predict --cv_dir $CV --class_config $CFG --epoch 50 --bag_size 500 --n_bags 5 --seed 403` | - | `$CV/predictions/epoch_50.npz` |
| every experiment through every fold (misclassifications_all_folds) | `remo-id workflow loo-predict --cv_dir $CV --class_config $CFG --epoch 50 --bag_size 500 --n_bags 5 --seed 403 --all_experiments --fold_shard $T --fold_nshards 47 --output_dir $CV/predictions_all_experiments` | 47, ~4 h each | `epoch_50_shard*.npz` |
| per-image held-out embeddings (bag_size_variability) | `remo-id workflow loo-per-image --cv_dir $CV --class_config $CFG --epoch 50 --fold_idx $T --seed 403` | 47 | `$CV/per_image_predictions/` |
| fold-0 per-image embeddings of every lfm-ohmu experiment | `remo-id workflow loo-per-image --cv_dir $CV --class_config examples/sepsis_paper/configs/sepsis_icu_healthy_ohmu.yaml --epoch 50 --fold_idx 0 --all_experiments --output_dir $CV/per_image_fold0_all --exp_shard $T --exp_nshards 8 --seed 403` | 8 | `$CV/per_image_fold0_all/` |
| indeterminate-patient ensemble (needs the manifest's clinical columns) | `python examples/sepsis_paper/infer_indeterminate.py --cv_dir $CV --manifest manifest.csv --epoch 50 --bag_size 500 --repeats 5 --fold_index $T`, then the same with `--aggregate` (CPU) | 47 | `$CV/analysis/indeterminate/ensemble_probs_epoch50.npz` |

Resources: one GPU per task. Give the predict, all-experiments and per-image arrays 80 GB GPUs,
`--mem=128G` and `--time=12:00:00`. With `--manifest`, the fold-0 per-image step embeds every
experiment; `umap_images.py` keeps the lfm-ohmu ones.

Optional: `remo-id workflow loo-report --cv_dir $CV --class_config $CFG --epoch 50 --bag_size 500
--no_regenerate` writes the package's standard LOO summary from the predictions.

### 3. Manifest and UMAP (CPU)

```bash
python examples/sepsis_paper/build_manifest.py --class_config $CFG --output manifest.csv   # needs the database; otherwise use the released manifest
python examples/sepsis_paper/umap_images.py --embeddings_dir $CV/per_image_fold0_all --manifest manifest.csv \
  --output_dir $CV/full_umap_ohmu                                       # 32 CPUs, --mem=192G
```

UMAP runs multithreaded without a fixed `random_state`, so a rerun is not guarunteed to produce the same result. It will be very similiar.

### 4. Cell occlusion screen

`remo-id analysis occlusion` scores experiments with one checkpoint
([root README](../../README.md#occlusion-analysis)). To explain every experiment by the fold
model that held it out, `occlusion_fold.py` runs each stage once per fold: array task K
reads `repeat0_foldK/splits.yaml` (the `val` experiments) and the fold's epoch-50 checkpoint
and calls the stage with `--checkpoint <fold ckpt> --experiment_ids <held-out ids>`. All folds
write to one work dir, `$CV/occlusion`, and every stage resumes per experiment. The stages
read the fold's `classes.yaml`; without the database, add `--manifest manifest.csv --classes
ICU-ctrl,healthy,sepsis` to each stage command.

| Step | Command (per array task) | Resources |
|---|---|---|
| background models | `python examples/sepsis_paper/occlusion_fold.py --cv_dir $CV --epoch 50 background --work_dir $CV/occlusion` | array 0-46, CPU, 32 GB |
| per-cell Grad-CAM | `... occlusion_fold.py --cv_dir $CV --epoch 50 attribution --work_dir $CV/occlusion --cell_config examples/sepsis_paper/configs/wbc_rbc_plt.yaml` | array 0-46, GPU, 32 GB |
| screen | `... occlusion_fold.py --cv_dir $CV --epoch 50 screen --work_dir $CV/occlusion --cell_config examples/sepsis_paper/configs/wbc_rbc_plt.yaml --random_cell_types wbc,rbc,platelet` | array 0-46, GPU (one GPU type for every fold keeps the numerics consistent), 12 CPUs, 48 GB, up to 8 h |
| report | `remo-id analysis occlusion report --work_dir $CV/occlusion --class_names "ICU control,Healthy,Sepsis"` | CPU, 64 GB |

The screen needs the background and Grad-CAM steps. `run_all.sh occlusion` submits the chain.

### 5. Figures and tables

```bash
sbatch --mem=32G --time=02:00:00 scripts/submit_cmd.sh python examples/sepsis_paper/make_paper_results.py \
  --cv_dir $CV --manifest manifest.csv --classes ICU-ctrl,healthy,sepsis --epoch 50 \
  --output results --static_dir <static_assets>
```

The job uses a GPU for one figure, which re-aggregates bags from the cached
per-image embeddings. If you
already have that step's `bag_size_variability_per_experiment.csv`, pass `--bagvar_csv`; with a
filled `--occlusion_cache` the job then runs on the CPU wrapper. Each output step is independent:
a failing step is reported in the summary and does not stop the rest.

The clinical outputs (demographics, laboratory values, clinical diagnoses, the
clinical-variable comparisons and the error tables with diagnoses) are built when the
manifest's clinical columns, as named in `configs/clinical_tables.yaml`, hold data. Without
them they are skipped with a one-line message, and everything else is built from the
prediction files, the occlusion screen and the manifest.

#### Cell occlusion figure

`cell_occlusion_interpretability.png` is built from the occlusion screen (`--occlusion_dir`,
default `$CV/occlusion`). There is no hand-picked selection: the builder in
`figures_paper.py` chooses everything by rule, with seeded tie-breaking (`--seed`):

- **Panel a, informative cells per class.** Rows Sepsis, ICU control and Healthy. Each row
  holds the 10 cells whose single removal lowers the held-out model's true-class log-odds
  most beyond the sham (remove minus sham). A cell qualifies if its removal also beats the
  equal-area null, it does not touch the frame border and its frame is classified correctly.
  Each row takes at most one cell per experiment: the top 5 white cells and the top 5 red
  cells, with free slots going to the next-ranked cells of any type. White cells come first.
  Healthy cells come only from the lfm-ohmu scope, which the healthy donors share with the
  patients (the manifest's scope column).
- **Panels b and c, one ICU-control and one Sepsis frame.** The frame is classified
  correctly with 0.6 < P(true) < 0.99 and its cells cover at most 55% of it. Frames are
  ranked by the total |stage-1 effect| of their cells, with one candidate per experiment.
  The top 30 candidates per class are checked for focus (Laplacian variance at or above the
  75th percentile of 30 seeded stage-1 frames of the same experiment), and the best
  in-focus candidate is shown. The tint is the per-class Grad-CAM of the held-out fold model; each
  pixel takes the colour of the class it pushes toward most.

The generic rules are `select_informative_cells` and `select_example_frames` in
`remo_id/interpret/cell_occlusion.py`; the quota, the scope rule and the class colours live
in `figures_paper.py` and `common.py`. The picks (`gallery_cells.csv`, `frames.csv`), the crops
and the Grad-CAM arrays are cached in `--occlusion_cache` (default `<output>/occlusion_cache`), so a
re-render reads only the cache. Delete the cache to recompute it.

## Outputs

Every output uses the bag-500 predictions.

Figures (top level of `--output`):

| File | Content |
|---|---|
| `roc_confusion_umap.png` | patient-level ROC and confusion matrix over per-image UMAPs coloured by true class, model probability and patient |
| `confusion_matrix_argmax_patient.png` / `.csv` | plain-argmax confusion matrix on replicate-averaged probabilities, with per-patient calls |
| `roc_sepsis_vs_icu_control.png` | sepsis vs ICU control ROC, patient level, bootstrap band |
| `cell_occlusion_interpretability.png` (+ `occlusion_cache/`) | informative cells per class, plus an ICU-control and a sepsis frame (rules above) |
| `threshold_sensitivity.png` | F1, precision and recall vs threshold, one-vs-rest, patient level |
| `bag_size_variability.png` (+ `bag_size_variability_per_experiment.csv`) | within-experiment SD of P(true class) vs bag size |
| `modality_comparison.png`, `umaps/crop_density_*_log_magma_arial.png` | static rasters fitted to print width; **not generated by this code** (`--static_dir`) |

Tables, in `tables/{experiment_level,patient_level_combined}/` (each as `.csv` + `.png`):

| File | Content | Level | Needs |
|---|---|---|---|
| `demographics` | cohort demographics | experiment | clinical columns |
| `performance` | per-class performance, 95% bootstrap CIs | both | |
| `one_vs_rest_auc` | per-class one-vs-rest AUC | both | |
| `cohort_characteristics`, `cohort_characteristics_manuscript` | clinical and laboratory characteristics (the second file is in manuscript format) | experiment | clinical columns |
| `pairwise_operating_points` | sepsis vs ICU control and sepsis vs all operating points | both | |
| `pairwise_operating_points_patient` | the same at patient level, with the 3-class argmax rules | experiment | |
| `patient_argmax_summary` | experiment- vs patient-level argmax rules | experiment | |
| `icu_control_diagnoses` | ICU-control patient diagnoses | experiment | clinical columns |
| `misclassifications` | held-out misclassifications with diagnosis | experiment | clinical columns |
| `misclassifications_all_folds` | per-experiment errors across all folds | experiment | clinical columns |
| `confusion_matrix_variants` | aggregation x decision-rule variants | experiment | |
| `indeterminate_patients`, `indeterminate_experiments` | indeterminate patients | experiment | clinical columns |
| `clinical_correlations` | model probabilities vs clinical variables (Spearman) | experiment | clinical columns |
| `sepsis_vs_icu_control_auc_ranking`, `two_variable_search` | sepsis vs ICU-control discrimination: model vs clinical variables and fusions | experiment | clinical columns |

The study's primary analysis used `experiment_level` for the tables. The patient level is a sensitivity analysis.

## Compute summary

| Stage | Resources |
|---|---|
| training | 47 x (1 x 80 GB GPU, 14 CPUs, 128 GB, up to 15 h) |
| loo-predict / loo-per-image / all-experiments | 1 x 80 GB GPU, 14 CPUs, 128 GB per task, up to 12 h |
| infer_indeterminate | 47 x (1 GPU, 128 GB, 2 h) |
| umap_images | CPU, 32 cores, 192 GB, ~1 h |
| make_paper_results | 1 GPU (bag_size_variability only), 32 GB; ~10 min on CPU with `--bagvar_csv` (measured peak RSS 1.3 GB) |
| occlusion screen | 3 x 47 array tasks (the screen array pinned to one GPU type, about 20 min per experiment on an H100) and a CPU report job |
