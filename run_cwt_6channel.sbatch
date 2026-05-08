#!/bin/bash
#SBATCH --job-name=cwt_6channel_extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/cwt_6channel_extract_%j.log

# Export threading limits to prevent oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Navigate to project directory
cd /path/to/NTU_SNTL_Task3-main

# Load Singularity module (if using container)
# module load singularity

# Run CWT 6-channel extraction
echo "Starting CWT 6-channel feature extraction..."
echo "Wavelet: morl, Scales: 128"
echo "Split: dev"

python extract_cwt_6channel.py \
    --split dev \
    --use_wavelet \
    --wavelet morl \
    --n_wavelet_scales 128

echo "CWT 6-channel extraction complete!"
