#! /bin/bash
# The sepsis study pipeline as SLURM submissions, in stages (a template: swap the class
# config, cell config and resources for your own data). Run from the repository root on a
# submit host:
#
#   SAVE_DIR=<dir> examples/sepsis_paper/run_all.sh train            # prepares splits, submits 47 fold jobs
#   CV=<LOO dir> COHORT=<cohort.csv> examples/sepsis_paper/run_all.sh predict   # after training (COHORT: UMAP step)
#   CV=<LOO dir> examples/sepsis_paper/run_all.sh occlusion          # after training (Fig 2 inputs)
#   CV=<LOO dir> COHORT=<cohort.csv> OUT=<dir> [STATIC_DIR=<dir>] \
#       [CLINICAL_TABLE=<clinical.csv>] examples/sepsis_paper/run_all.sh report   # after `predict`, `occlusion`
#
# CLINICAL_TABLE (controlled-access clinical.csv) turns on the clinical outputs; in
# `predict` it also runs the indeterminate-patient ensemble.
#
# Stages chain internally with --dependency; across stages wait for the previous one.
# Resources are cluster-specific: the GPU constraint below pins 80 GB cards (full-
# resolution resnet50-MIL runs out of memory on 40 GB GPUs). Adjust for your cluster.
set -euo pipefail

[ "${1:-}" = predict ] && : "${COHORT:?predict needs COHORT=<cohort.csv> (Fig 1 UMAP step)}"

CFG=examples/sepsis_paper/configs/sepsis_icu_healthy.yaml
OHMU_CFG=examples/sepsis_paper/configs/sepsis_icu_healthy_ohmu.yaml
CELLS=examples/sepsis_paper/configs/wbc_rbc_plt.yaml
EPOCH=50
GPU80=(--partition=gpu --gpus-per-node=1 --constraint="h100_80|h200|a100_80"
       --cpus-per-task=14 --mem=128G --time=12:00:00)
GPU=scripts/submit_cmd.sh
CPU=scripts/submit_cpu_cmd.sh
sub() { sbatch --parsable "$@"; }

case "${1:-}" in
train)
  # Outside SLURM, `train --loo` only reads metadata: it writes
  # $SAVE_DIR/<timestamp>_cv_resnet50-mil_loo/cv_splits.yaml and submits one GPU task per fold.
  python -m remo_id.cli train --class_config $CFG --model resnet50-mil --pooling attention \
    --batch_size 200 --n_experiments 4 --accumulate_grad_batches 3 \
    --lr 1e-4 --lr_schedule cosine --warmup_epochs 5 \
    --max_epochs 50 --early_stopping_patience 0 \
    --num_heads 1 --attn_dim 128 --dropout 0.3 --weight_decay 0.01 \
    --finetune_layers layer3,layer4 --pretrained \
    --no-infer --loo --save_dir "$SAVE_DIR"
  ;;

predict)
  # pooled held-out predictions: bag 50 (training bag) -> $CV/predictions/epoch_50.npz
  python -m remo_id.cli workflow loo-predict-submit --cv_dir "$CV" --class_config $CFG \
    --epochs $EPOCH --bag_size 50 --n_bags 5 --seed 403
  # bag 300 / 500 -> $CV/predictions_bag{300,500}/epoch_50.npz
  sub --array=0-1 "${GPU80[@]}" $GPU bash -c 'BAGS=(300 500); B=${BAGS[$SLURM_ARRAY_TASK_ID]};
    python -m remo_id.cli workflow loo-predict --cv_dir "'"$CV"'" --class_config '$CFG' \
      --epoch '$EPOCH' --bag_size $B --n_bags 5 --seed 403 --output_dir "'"$CV"'/predictions_bag$B"'
  # every experiment through every fold (Supplementary Table 1b)
  sub --array=0-46 "${GPU80[@]}" $GPU bash -c 'python -m remo_id.cli workflow loo-predict \
    --cv_dir "'"$CV"'" --class_config '$CFG' --epoch '$EPOCH' --bag_size 500 --n_bags 5 \
    --seed 403 --all_experiments --fold_shard $SLURM_ARRAY_TASK_ID --fold_nshards 47 \
    --output_dir "'"$CV"'/predictions_allexp_bag500"'
  # held-out per-image embeddings (Supplementary Fig 9, Fig 2)
  sub --array=0-46 "${GPU80[@]}" $GPU bash -c 'python -m remo_id.cli workflow loo-per-image \
    --cv_dir "'"$CV"'" --class_config '$CFG' --epoch '$EPOCH' --fold_idx $SLURM_ARRAY_TASK_ID --seed 403'
  # fold-0 per-image embeddings of every lfm-ohmu experiment -> UMAP (Fig 1)
  J=$(sub --array=0-7 "${GPU80[@]}" $GPU bash -c 'python -m remo_id.cli workflow loo-per-image \
    --cv_dir "'"$CV"'" --class_config '$OHMU_CFG' --epoch '$EPOCH' --fold_idx 0 --all_experiments \
    --output_dir "'"$CV"'/per_image_fold0_all" --exp_shard $SLURM_ARRAY_TASK_ID --exp_nshards 8 --seed 403')
  sub --dependency=afterok:$J --cpus-per-task=32 --mem=192G --time=08:00:00 $CPU \
    python examples/sepsis_paper/umap_images.py --embeddings_dir "$CV/per_image_fold0_all" --cohort "$COHORT" \
    --output_dir "$CV/full_umap_ohmu"
  if [ -n "${CLINICAL_TABLE:-}" ]; then
    # indeterminate patients (Supplementary Table 3; images via the internal database)
    J=$(sub --array=0-46 --mem=128G --time=02:00:00 $GPU bash -c 'python examples/sepsis_paper/infer_indeterminate.py \
      --cv_dir "'"$CV"'" --class_config '$CFG' --clinical_table "'"$CLINICAL_TABLE"'" \
      --epoch '$EPOCH' --bag_size 500 --repeats 5 --seed 403 --fold_index $SLURM_ARRAY_TASK_ID')
    sub --dependency=afterok:$J $CPU python examples/sepsis_paper/infer_indeterminate.py --cv_dir "$CV" \
      --class_config $CFG --epoch $EPOCH --bag_size 500 --repeats 5 --aggregate
  fi
  ;;

occlusion)
  # Cell occlusion screen behind Fig 2 (after training): every experiment is scored by the
  # LOO fold model that held it out. occlusion_fold.py maps array task K to fold K's
  # held-out experiments and its epoch-$EPOCH checkpoint and runs one
  # `remo-id analysis occlusion` stage; all folds share $CV/occlusion.
  OCC="$CV/occlusion"
  FOLD=(python examples/sepsis_paper/occlusion_fold.py --cv_dir "$CV" --epoch $EPOCH)
  J1=$(sub --array=0-46 --partition=cpu --cpus-per-task=4 --mem=32G --time=02:00:00 $CPU \
       "${FOLD[@]}" background --work_dir "$OCC")
  J2=$(sub --array=0-46 --partition=gpu --gpus-per-node=1 --cpus-per-task=8 --mem=32G \
       --time=08:00:00 $GPU "${FOLD[@]}" attribution --work_dir "$OCC" --cell_config $CELLS)
  # one GPU type for every fold keeps the numerics consistent across folds
  J3=$(sub --dependency=afterok:$J1:$J2 --array=0-46 --partition=gpu --gpus-per-node=1 \
       --constraint="h100_80|h200" --cpus-per-task=12 --mem=48G --time=08:00:00 $GPU \
       "${FOLD[@]}" screen --work_dir "$OCC" --cell_config $CELLS --random_cell_types wbc,rbc,platelet)
  sub --dependency=afterok:$J3 --mem=64G $CPU python -m remo_id.cli analysis occlusion report \
    --work_dir "$OCC" --class_names "ICU control,Healthy,Sepsis"
  ;;

report)
  EXTRA=()
  [ -n "${STATIC_DIR:-}" ] && EXTRA+=(--static_dir "$STATIC_DIR")
  [ -n "${CLINICAL_TABLE:-}" ] && EXTRA+=(--clinical --clinical_table "$CLINICAL_TABLE")
  sub --mem=32G --time=02:00:00 $GPU python examples/sepsis_paper/make_paper_results.py --cv_dir "$CV" \
    --class_config $CFG --epoch $EPOCH --cohort "$COHORT" --output "$OUT" "${EXTRA[@]}"
  ;;

*)
  sed -n '2,19p' "$0"; exit 1 ;;
esac
