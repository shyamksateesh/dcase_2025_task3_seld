#!/bin/bash
#SBATCH --job-name=feat_extract
#SBATCH --output=feat_extract_%j.out
#SBATCH --error=feat_extract_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=32
#SBATCH --account=torch_pr_1026_general

cd /home/km4886/seld_mlistening_project

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== [START] Feature extraction job ${SLURM_JOB_ID} at $(date) on $(hostname) ==="

./sing << EOF
python -u extract_wavelet_features.py --use_wavelet --wavelet morl --n_wavelet_scales 128 --split dev
EOF

exit_code=$?
if [ $exit_code -eq 0 ]; then
	echo "=== [DONE] Feature extraction job ${SLURM_JOB_ID} finished successfully at $(date) ==="
else
	echo "=== [FAILED] Feature extraction job ${SLURM_JOB_ID} failed with exit code ${exit_code} at $(date) ==="
fi

exit $exit_code