#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --output=gpu_test_%j.out
#SBATCH --error=gpu_test_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu
#SBATCH --account=torch_pr_1026_general

cd /home/km4886/seld_mlistening_project

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== [START] Feature extraction job ${SLURM_JOB_ID} at $(date) on $(hostname) ==="

./sing << EOF
python -u main.py --use_wavelet --wavelet morl --n_wavelet_scales 128 --exp wavelet_test --nb_epochs 100
EOF

exit_code=$?
if [ $exit_code -eq 0 ]; then
	echo "=== [DONE] Feature extraction job ${SLURM_JOB_ID} finished successfully at $(date) ==="
else
	echo "=== [FAILED] Feature extraction job ${SLURM_JOB_ID} failed with exit code ${exit_code} at $(date) ==="
fi

exit $exit_code