#! /bin/bash
# Example SLURM wrapper. Adjust partition, resources and constraints for your cluster.

#SBATCH --job-name=RemoIDTraining
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=preempted
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --output=/dev/null

# Redirect output: array jobs get {master}_{task}.out, single jobs get {jobid}.out
mkdir -p ./slurm-outputs
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    exec > ./slurm-outputs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out 2>&1
else
    exec > ./slurm-outputs/${SLURM_JOB_ID}.out 2>&1
fi

# On SIGTERM (preemption signal), let wandb know and exit cleanly
trap 'echo "PREEMPTED at $(date)"; wandb offline; exit 0' SIGTERM

env | grep "^SLURM" | sort
nvcc --version
nvidia-smi

echo "running: $@"

export PYTHONUNBUFFERED=1
wandb online

# Handle remo-id CLI command (convert to python -m remo_id.cli)
if [ "$1" = "remo-id" ]; then
    shift
    python -m remo_id.cli "$@"
else
    "$@"
fi

wandb offline

# Example usage (W&B settings such as WANDB_ENTITY / WANDB_BASE_URL are read from your environment):
# sbatch ./scripts/submit_preempted.sh remo-id train \
#   --manifest examples/manifest_example.csv \
#   --model resnet18-mil \
#   --batch_size 200 --max_epochs 100
