#!/bin/bash
#SBATCH --job-name=feat_extract_shard
#SBATCH --output=feat_extract_shard_%A_%a.out
#SBATCH --error=feat_extract_shard_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=8
#SBATCH --account=torch_pr_1026_general

# Usage: submit as an array job with --array=0-(N-1) where N is the number of shards.
# Example to run 8 parallel extract jobs (shards 0..7):
# sbatch --array=0-7 extract_features_sharded.sbatch --num-shards 8

# When using array jobs, SLURM_ARRAY_TASK_ID contains the shard id (0-based)
SHARD_ID=${SLURM_ARRAY_TASK_ID}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}

# If you want to override via sbatch extra args, you can pass --num-shards and --shard-id
# The wrapper below prefers explicit CLI args passed to the script when sbatch is called.

cd /home/km4886/seld_mlistening_project

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== [START] Feature extraction shard job ${SLURM_ARRAY_JOB_ID}:${SLURM_ARRAY_TASK_ID} at $(date) on $(hostname) ==="

# Support two modes: if sbatch --array used, use SLURM_ARRAY_TASK_ID; otherwise user can pass --shard-id
if [ -z "$SHARD_ID" ] || [ "$SHARD_ID" == "" ]; then
    SHARD_ID=0
fi

# By default, detect NUM_SHARDS from the array size if available; allow override
if [ -n "$SLURM_ARRAY_TASK_COUNT" ]; then
    NUM_SHARDS=$SLURM_ARRAY_TASK_COUNT
fi

# Call the extraction script in only-extract mode so many shards don't duplicate normalization
./sing << EOF
python -u extract_wavelet_features.py --use_wavelet --wavelet morl --n_wavelet_scales 128 --split dev --num_shards ${NUM_SHARDS} --shard_id ${SHARD_ID} --only_extract
EOF

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "=== [DONE] Feature extraction shard ${SHARD_ID} finished successfully at $(date) ==="
else
    echo "=== [FAILED] Feature extraction shard ${SHARD_ID} failed with exit code ${exit_code} at $(date) ==="
fi

exit $exit_code
