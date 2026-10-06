#! /bin/bash
# The sepsis study pipeline as SLURM submissions, in stages (a template: swap the class
# config, cell config and resources for your own data). Run from the repository root on a
# submit host:
#
#   SAVE_DIR=<dir> examples/sepsis_paper/run_all.sh train            # prepares splits, submits 47 fold jobs
#   CV=<LOO dir> MANIFEST=<manifest.csv> examples/sepsis_paper/run_all.sh predict   # after training
#   CV=<LOO dir> examples/sepsis_paper/run_all.sh occlusion          # after training
#   CV=<LOO dir> MANIFEST=<manifest.csv> OUT=<dir> [STATIC_DIR=<dir>] \
#       examples/sepsis_paper/run_all.sh report                      # after `predict`, `occlusion`
#
# The remo-id commands select data with the class config (Remoscope database). NO_DB=1
# selects it with the manifest instead (--manifest $MANIFEST --classes ICU-ctrl,healthy,sepsis),
# so no stage needs the database. Clinical columns in the manifest turn on the clinical
# outputs and, in `predict`, the indeterminate-patient ensemble.
#
# Stages chain internally with --dependency; across stages wait for the previous one.
# Resources are cluster-specific: the GPU constraint below pins 80 GB cards (full-
# resolution resnet50-MIL runs out of memory on 40 GB GPUs). Adjust for your cluster.
set -euo pipefail

case "${1:-}" in predict|report) : "${MANIFEST:?$1 needs MANIFEST=<manifest.csv>}" ;; esac

CFG=examples/sepsis_paper/configs/sepsis_icu_healthy.yaml
OHMU_CFG=examples/sepsis_paper/configs/sepsis_icu_healthy_ohmu.yaml
CELLS=examples/sepsis_paper/configs/wbc_rbc_plt.yaml
CLASSES=ICU-ctrl,healthy,sepsis
EPOCH=50
BAG=500
if [ "${NO_DB:-0}" = 1 ]; then
  : "${MANIFEST:?NO_DB=1 needs MANIFEST=<manifest.csv>}"
  SRC="--manifest $MANIFEST --classes $CLASSES"
  OHMU_SRC=$SRC      # every experiment; umap_images.py keeps the lfm-ohmu ones
else
  SRC="--class_config $CFG"
  OHMU_SRC="--class_config $OHMU_CFG"
fi
GPU80=(--partition=gpu --gpus-per-node=1 --constraint="h100_80|h200|a100_80"
       --cpus-per-task=14 --mem=128G --time=12:00:00)
GPU=scripts/submit_cmd.sh
CPU=scripts/submit_cpu_cmd.sh
sub() { sbatch --parsable "$@"; }

case "${1:-}" in
train)
  # Outside SLURM, `train --loo` only reads metadata: it writes
  # $SAVE_DIR/<timestamp>_cv_resnet50-mil_loo/cv_splits.yaml and submits one GPU task per fold.
  python -m remo_id.cli train $SRC --model resnet50-mil --pooling attention \
    --batch_size 200 --n_experiments 4 --accumulate_grad_batches 3 \
    --lr 1e-4 --lr_schedule cosine --warmup_epochs 5 \
    --max_epochs 50 --early_stopping_patience 0 \
    --num_heads 1 --attn_dim 128 --dropout 0.3 --weight_decay 0.01 \
    --finetune_layers layer3,layer4 --pretrained \
    --no-infer --loo --save_dir "$SAVE_DIR"
  ;;

predict)
  # pooled held-out predictions at bag 500 -> $CV/predictions/epoch_50.npz
  sub "${GPU80[@]}" $GPU python -m remo_id.cli workflow loo-predict --cv_dir "$CV" $SRC \
    --epoch $EPOCH --bag_size $BAG --n_bags 5 --seed 403
  # every experiment through every fold (misclassifications_all_folds)
  sub --array=0-46 "${GPU80[@]}" $GPU bash -c 'python -m remo_id.cli workflow loo-predict \
    --cv_dir "'"$CV"'" '"$SRC"' --epoch '$EPOCH' --bag_size '$BAG' --n_bags 5 \
    --seed 403 --all_experiments --fold_shard $SLURM_ARRAY_TASK_ID --fold_nshards 47 \
    --output_dir "'"$CV"'/predictions_all_experiments"'
  # held-out per-image embeddings (bag_size_variability)
  sub --array=0-46 "${GPU80[@]}" $GPU bash -c 'python -m remo_id.cli workflow loo-per-image \
    --cv_dir "'"$CV"'" '"$SRC"' --epoch '$EPOCH' --fold_idx $SLURM_ARRAY_TASK_ID --seed 403'
  # fold-0 per-image embeddings of the lfm-ohmu experiments -> UMAP (roc_confusion_umap)
  J=$(sub --array=0-7 "${GPU80[@]}" $GPU bash -c 'python -m remo_id.cli workflow loo-per-image \
    --cv_dir "'"$CV"'" '"$OHMU_SRC"' --epoch '$EPOCH' --fold_idx 0 --all_experiments \
    --output_dir "'"$CV"'/per_image_fold0_all" --exp_shard $SLURM_ARRAY_TASK_ID --exp_nshards 8 --seed 403')
  sub --dependency=afterok:$J --cpus-per-task=32 --mem=192G --time=08:00:00 $CPU \
    python examples/sepsis_paper/umap_images.py --embeddings_dir "$CV/per_image_fold0_all" \
    --manifest "$MANIFEST" --output_dir "$CV/full_umap_ohmu"
  if head -1 "$MANIFEST" | grep -q patient_notes; then
    # indeterminate patients (empty label, recorded diagnosis)
    J=$(sub --array=0-46 --mem=128G --time=02:00:00 $GPU bash -c 'python examples/sepsis_paper/infer_indeterminate.py \
      --cv_dir "'"$CV"'" --manifest "'"$MANIFEST"'" \
      --epoch '$EPOCH' --bag_size '$BAG' --repeats 5 --seed 403 --fold_index $SLURM_ARRAY_TASK_ID')
    sub --dependency=afterok:$J $CPU python examples/sepsis_paper/infer_indeterminate.py --cv_dir "$CV" \
      --manifest "$MANIFEST" --epoch $EPOCH --bag_size $BAG --repeats 5 --aggregate
  fi
  ;;

occlusion)
  # Cell occlusion screen (after training): every experiment is scored by the LOO fold
  # model that held it out. occlusion_fold.py maps array task K to fold K's held-out
  # experiments and its epoch-$EPOCH checkpoint and runs one
  # `remo-id analysis occlusion` stage; all folds share $CV/occlusion.
  OCC="$CV/occlusion"
  FOLD=(python examples/sepsis_paper/occlusion_fold.py --cv_dir "$CV" --epoch $EPOCH)
  J1=$(sub --array=0-46 --partition=cpu --cpus-per-task=4 --mem=32G --time=02:00:00 $CPU \
       "${FOLD[@]}" background --work_dir "$OCC" $SRC)
  J2=$(sub --array=0-46 --partition=gpu --gpus-per-node=1 --cpus-per-task=8 --mem=32G \
       --time=08:00:00 $GPU "${FOLD[@]}" attribution --work_dir "$OCC" $SRC --cell_config $CELLS)
  # one GPU type for every fold keeps the numerics consistent across folds
  J3=$(sub --dependency=afterok:$J1:$J2 --array=0-46 --partition=gpu --gpus-per-node=1 \
       --constraint="h100_80|h200" --cpus-per-task=12 --mem=48G --time=08:00:00 $GPU \
       "${FOLD[@]}" screen --work_dir "$OCC" $SRC --cell_config $CELLS \
       --random_cell_types wbc,rbc,platelet)
  sub --dependency=afterok:$J3 --mem=64G $CPU python -m remo_id.cli analysis occlusion report \
    --work_dir "$OCC" --class_names "ICU control,Healthy,Sepsis"
  ;;

report)
  EXTRA=()
  [ -n "${STATIC_DIR:-}" ] && EXTRA+=(--static_dir "$STATIC_DIR")
  sub --mem=32G --time=02:00:00 $GPU python examples/sepsis_paper/make_paper_results.py --cv_dir "$CV" \
    --manifest "$MANIFEST" --classes $CLASSES --epoch $EPOCH --output "$OUT" "${EXTRA[@]}"
  ;;

*)
  sed -n '2,20p' "$0"; exit 1 ;;
esac
