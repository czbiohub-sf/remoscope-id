#! /bin/bash
# Example SLURM wrapper. Adjust partition, resources and constraints for your cluster.

#SBATCH --job-name=RemoIDCPU
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/dev/null

# Redirect output: array jobs get {master}_{task}.out, single jobs get {jobid}.out
mkdir -p ./slurm-outputs
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    exec > ./slurm-outputs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out 2>&1
else
    exec > ./slurm-outputs/${SLURM_JOB_ID}.out 2>&1
fi

env | grep "^SLURM" | sort

echo "running: $@"

export PYTHONUNBUFFERED=1

# Handle remo-id CLI command (convert to python -m remo_id.cli)
if [ "$1" = "remo-id" ]; then
    shift
    python -m remo_id.cli "$@"
else
    "$@"
fi

# Example usage (CPU-only report from saved predictions):
# sbatch ./scripts/submit_cpu_cmd.sh remo-id report performance --checkpoint checkpoints/<run>/best.ckpt
