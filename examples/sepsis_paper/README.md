# Example: sepsis study workflow

This folder shows how remo-id was configured and run for the sepsis study that accompanies

> **TITLE TBD.** Authors TBD. bioRxiv (link TBD), DOI TBD.

It holds the class and cell configs, the training and leave-one-out (LOO) evaluation
commands, the cell occlusion screen and the figure and table builders that generated the
figures in the publication. Use it as a template for your own data: replace
`configs/sepsis_icu_healthy.yaml` with your class config (or train with `--manifest`), the
cell config with yours, and the SLURM resources with your cluster's.

The study's model is a resnet50 attention multiple-instance-learning (MIL) classifier. It
treats each Remoscope acquisition (experiment) as a bag of full-resolution brightfield frames
and classifies it as Sepsis, ICU control or Healthy. It was evaluated by patient-grouped LOO
cross-validation: 47 folds, one patient per fold, 129 experiments. The builders read held-out
predictions made by one fixed-epoch checkpoint of each fold (epoch 50 below).

## Data availability

The image data and the cohort table are distributed with the data release (link TBD). The clinical metadata (demographics, laboratory values, diagnoses) is
**not public**: it comes as `clinical.csv` with the controlled-access clinical data release.
Outputs that need it are built only with `--clinical --clinical_table clinical.csv` (see below).

These files from the data release are deliberately kept out of this repository because they
carry pseudonymous experiment and patient identifiers:

| File | Contents | Made by |
|---|---|---|
| `cohort.csv` | one row per LOO experiment: `experiment_id, sample_id, study_id, scope` | `build_cohort_table.py` (needs the database) |
| `clinical.csv` | clinical input table, one row per experiment (controlled access; see [Clinical table](#clinical-table)) | `build_cohort_table.py --clinical_output` (needs the database) |

Both names are listed in `.gitignore`. Pass their paths with `--cohort` / `--clinical_table`.

## Layout

| Path | What it does |
|---|---|
| `configs/sepsis_icu_healthy.yaml` | class config used for training and every analysis (3 classes) |
| `configs/sepsis_icu_healthy_ohmu.yaml` | the same classes restricted to the lfm-ohmu scope (Fig 1 UMAP only) |
| `configs/wbc_rbc_plt.yaml` | cell typing (WBC / RBC / platelet regionprops gates) for Fig 2 |
| `common.py` | shared helpers; importing it registers the study's class colours |
| `make_paper_results.py` | one command for every figure and table |
| `figures_paper.py` | the output builders called by `make_paper_results.py` |
| `configs/clinical_tables.yaml` | clinical-table spec: the clinical columns, labels, vocabularies and table rows of the `--clinical` outputs |
| `build_cohort_table.py` | writes `cohort.csv` and `clinical.csv` from the database (the only script that reads it) |
| `umap_images.py` | per-image UMAP behind the Fig 1 bottom row |
| `infer_indeterminate.py` | LOO-ensemble predictions for the patients without an adjudicated label (Supplementary Table 3) |
| `occlusion_fold.py` | runs one `remo-id analysis occlusion` stage on one LOO fold's held-out experiments with that fold's checkpoint (one SLURM array task per fold) |
| `run_all.sh` | the whole pipeline as sbatch commands, in stages |

## Setup

Use Python 3.12, a CUDA build of PyTorch, and the `remoscope-database` package (data access):

```bash
conda create -n remoid python=3.12 && conda activate remoid
pip install -e /path/to/remoscope-database
pip install -e /path/to/remo-id-public
```

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
`last.ckpt` is the epoch-50 model, the only checkpoint the steps below use; no per-epoch
checkpoints are needed. Each task needs one 80 GB GPU (full-resolution resnet50-MIL runs out
of memory on 40 GB cards), 14 CPUs and 128 GB RAM.

### 2. Held-out predictions and embeddings at epoch 50 (GPU)

Every step uses `--epoch 50` and the same seed. The bag size is always passed explicitly,
because checkpoints do not store it.

| Step | Command (per array task) | Array | Output |
|---|---|---|---|
| pooled predictions, bag 50 (the training bag) | `remo-id workflow loo-predict-submit --cv_dir $CV --class_config $CFG --epochs 50 --bag_size 50 --n_bags 5 --seed 403` (submits its own job) | - | `$CV/predictions/epoch_50.npz` |
| pooled predictions, bag 300 / 500 | `remo-id workflow loo-predict --cv_dir $CV --class_config $CFG --epoch 50 --bag_size $BAG --n_bags 5 --seed 403 --output_dir $CV/predictions_bag$BAG` | 2 (BAG=300,500) | `$CV/predictions_bag{300,500}/epoch_50.npz` |
| every experiment through every fold (Supplementary Table 1b) | `remo-id workflow loo-predict --cv_dir $CV --class_config $CFG --epoch 50 --bag_size 500 --n_bags 5 --seed 403 --all_experiments --fold_shard $T --fold_nshards 47 --output_dir $CV/predictions_allexp_bag500` | 47, ~4 h each | `epoch_50_shard*.npz` |
| per-image held-out embeddings (Supplementary Fig 9) | `remo-id workflow loo-per-image --cv_dir $CV --class_config $CFG --epoch 50 --fold_idx $T --seed 403` | 47 | `$CV/per_image_predictions/` |
| fold-0 per-image embeddings of every lfm-ohmu experiment (Fig 1 UMAP) | `remo-id workflow loo-per-image --cv_dir $CV --class_config examples/sepsis_paper/configs/sepsis_icu_healthy_ohmu.yaml --epoch 50 --fold_idx 0 --all_experiments --output_dir $CV/per_image_fold0_all --exp_shard $T --exp_nshards 8 --seed 403` | 8 | `$CV/per_image_fold0_all/` |
| indeterminate-patient ensemble (`--clinical`) | `python examples/sepsis_paper/infer_indeterminate.py --cv_dir $CV --class_config $CFG --clinical_table clinical.csv --epoch 50 --bag_size 500 --repeats 5 --fold_index $T`, then the same with `--aggregate` (CPU) | 47 | `$CV/analysis/indeterminate/ensemble_probs_epoch50.npz` |

Resources: one GPU per task. Give the predict, all-experiments and per-image arrays 80 GB GPUs,
`--mem=128G` and `--time=12:00:00`.

Optional: `remo-id workflow loo-report --cv_dir $CV --class_config $CFG --epoch 50 --bag_size 50
--no_regenerate` writes the package's standard LOO summary from the bag-50 predictions. No
output below depends on it, and none depends on `loo-bag-variance` either.

### 3. Cohort table and UMAP (CPU)

```bash
python examples/sepsis_paper/build_cohort_table.py --cv_dir $CV --output cohort.csv \
  --class_config $CFG --clinical_output clinical.csv                    # needs the database; otherwise use the released copies
python examples/sepsis_paper/umap_images.py --embeddings_dir $CV/per_image_fold0_all --cohort cohort.csv \
  --output_dir $CV/full_umap_ohmu                                       # 32 CPUs, --mem=192G
```

UMAP runs multithreaded without a fixed `random_state`, so a rerun gives an equivalent but not
bit-identical layout.

### 4. Cell occlusion screen (Fig 2 inputs)

`remo-id analysis occlusion` scores experiments with one checkpoint
([root README](../../README.md#occlusion-analysis)). To explain every experiment by the fold
model that held it out, `occlusion_fold.py` runs each stage once per fold: array task K
reads `repeat0_foldK/splits.yaml` (the `val` experiments) and the fold's epoch-50 checkpoint
and calls the stage with `--checkpoint <fold ckpt> --experiment_ids <held-out ids>`. All folds
write to one work dir, `$CV/occlusion`, and every stage resumes per experiment. The cell masks
and regionprops tables come with the image data.

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
  --cv_dir $CV --class_config $CFG --epoch 50 --cohort cohort.csv \
  --output results --static_dir <static_assets> [--clinical --clinical_table clinical.csv]
```

The job uses a GPU for Supplementary Fig 9, which re-aggregates bags from the cached
per-image embeddings, and the first time Fig 2 is built, for its two Grad-CAM frames. If you
already have that step's `supfig9_bagvar_per_experiment.csv`, pass `--bagvar_csv`; with a
filled `--fig2_cache` the job then runs on the CPU wrapper. Each output step is independent:
a failing step is reported in the summary and does not stop the rest.

`--clinical` adds every output that reads the clinical table (`--clinical_table`). These
outputs are demographics, laboratory values, clinical diagnoses, the clinical-variable
comparisons and the error tables with diagnoses. Without `--clinical`, everything else is
built from the prediction files, the occlusion screen and `cohort.csv` alone.

#### Clinical table

`clinical.csv` comes with the controlled-access clinical data release (or from
`build_cohort_table.py --clinical_output` with database access). It has one row per
experiment of every study in the class config, healthy donors included:

| column | meaning |
|---|---|
| `experiment_id`, `sample_id` | experiment and patient (sample) |
| `study_id` | the study as named in the class config |
| `path`, `num_images` | experiment directory and frame count (image and cell counts of Table 1) |
| the class-config label columns | e.g. the sepsis class's `column`, which also defines the patient groups of Table 3 |
| clinical variables | every column named in `configs/clinical_tables.yaml` |

`configs/clinical_tables.yaml` is the spec of the clinical outputs: the cleaning rules
(missing-value codes, boolean columns, derived scores), the Table 1 columns, the Table 3
rows (`num` median (IQR), `bin` / `level:<v>` n/N (%), `cat`, `pathogen`, `diagnosis`)
and their manuscript variant, the race and organism vocabularies, and the variables of
Fig 3 and Fig 4. To run the clinical outputs on another study, write its `clinical.csv`
and adapt the spec; no code changes are needed. Rows of different studies share the
file, and a column left empty for every row of a study counts as not recorded by that
study. Only empty fields are missing values. Columns listed under
`single_precision_columns` are read as float32, the precision they are stored in at the
source.

#### Fig 2

`fig2_interpretability_occlusion.png` is built from the occlusion screen (`--occlusion_dir`,
default `$CV/occlusion`). There is no hand-picked selection: the builder in
`figures_paper.py` chooses everything by rule, with seeded tie-breaking (`--seed`):

- **Panel a, informative cells per class.** Rows Sepsis, ICU control and Healthy. Each row
  holds the 10 cells whose single removal lowers the held-out model's true-class log-odds
  most beyond the sham (remove minus sham). A cell qualifies if its removal also beats the
  equal-area null, it does not touch the frame border and its frame is classified correctly.
  Each row takes at most one cell per experiment: the top 5 white cells and the top 5 red
  cells, with free slots going to the next-ranked cells of any type. White cells come first.
  Healthy cells come only from the lfm-ohmu scope, which the healthy donors share with the
  patients (`cohort.csv` scope column).
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
and the Grad-CAM arrays are cached in `--fig2_cache` (default `<output>/fig2_cache`), so a
re-render reads only the cache. Delete the cache to recompute it.

## Outputs

The file prefixes follow the figure and table numbering of the publication draft.

Figures (top level of `--output`):

| File | Content |
|---|---|
| `fig1_cm_roc_umap_sample_average.png` | Fig 1: patient-level ROC and confusion matrix (bag 300) over per-image UMAPs coloured by true class, model probability and patient |
| `fig1_cm_roc_argmax_patient_mean.png` / `.csv` | Fig 1 companion: plain-argmax confusion matrix on replicate-averaged probabilities (bag 500), with per-patient calls |
| `fig1e_sepsis_vs_icu_roc.png` | sepsis vs ICU control ROC, patient level, bootstrap band (bag 500) |
| `fig2_interpretability_occlusion.png` (+ `fig2_cache/`) | Fig 2: informative cells per class, plus an ICU-control and a sepsis frame (rules above) |
| `supfig7_threshold_sensitivity.png` | Supplementary Fig 7: F1, precision and recall vs threshold, one-vs-rest, patient level (bag 500) |
| `supfig9_bagvar_ribbons.png` (+ `supfig9_bagvar_per_experiment.csv`) | Supplementary Fig 9: within-experiment SD of P(true class) vs bag size |
| `modality_comparison.png`, `umaps/crop_density_*_log_magma_arial.png` | static rasters fitted to print width; **not generated by this code** (`--static_dir`) |

Tables, in `tables/bag{50,300,500}/{experiment_level,patient_level_combined}/` (each as
`.csv` + `.png`):

| File | Content | Level | Needs |
|---|---|---|---|
| `table1_demographics` | Table 1: cohort demographics | experiment | `--clinical` |
| `table2_performance` | Table 2: per-class performance, 95% bootstrap CIs | both | |
| `table2b_ovr_auc` | Table 2b: per-class one-vs-rest AUC | both | |
| `table3_cohort_characteristics`, `table3_cohort_characteristics_with_lymphocytes` | Table 3: clinical and laboratory characteristics (the second file is in manuscript format) | experiment | `--clinical` |
| `table4_pairwise_operating_points` | Table 4: sepsis vs ICU control and sepsis vs all operating points | both | |
| `table4b_pairwise_sample_average` | Table 4b: the same at patient level, with the 3-class argmax rows | experiment | |
| `table_patient_argmax_summary` | experiment- vs patient-level argmax rules | experiment | |
| `table_icu_control_diagnoses` | ICU-control patient diagnoses | experiment | `--clinical` |
| `suptable1_error_analysis` | Supplementary Table 1: held-out misclassifications with diagnosis | experiment | `--clinical` |
| `suptable1b_error_analysis_all_folds` | Supplementary Table 1b: per-experiment errors across all folds (bag 500 only) | experiment | `--clinical` |
| `suptable2_cm_variants` | Supplementary Table 2: aggregation x decision-rule variants | experiment | |
| `suptable3_indeterminate`, `suptable3b_indeterminate_experiments` | Supplementary Table 3 / 3b: indeterminate patients | experiment | `--clinical` |
| `fig3_clinical_correlations` | model probabilities vs clinical variables (Spearman) | experiment | `--clinical` |
| `fig4_sepsis_auc_ranking`, `fig4b_two_variable_search` | sepsis vs ICU-control discrimination: model vs clinical variables and fusions | experiment | `--clinical` |

The study's primary analysis used `bag500/experiment_level` for the tables and `bag300` for
Fig 1. The other bag sizes and the patient level are sensitivity analyses.

## Compute summary

| Stage | Resources |
|---|---|
| training | 47 x (1 x 80 GB GPU, 14 CPUs, 128 GB, up to 15 h) |
| loo-predict / loo-per-image / all-experiments | 1 x 80 GB GPU, 14 CPUs, 128 GB per task, up to 12 h |
| infer_indeterminate | 47 x (1 GPU, 128 GB, 2 h) |
| umap_images | CPU, 32 cores, 192 GB, ~1 h |
| make_paper_results | 1 GPU (Supplementary Fig 9 only), 32 GB; ~10 min on CPU with `--bagvar_csv --clinical` (measured peak RSS 1.3 GB) |
| occlusion screen | 3 x 47 array tasks (the screen array pinned to one GPU type, about 20 min per experiment on an H100) and a CPU report job |
