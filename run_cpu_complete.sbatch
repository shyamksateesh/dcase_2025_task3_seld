#!/bin/bash
#SBATCH --job-name=wavelet_100
#SBATCH --output=wavelet_%j.out
#SBATCH --error=wavelet_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_1026_general

cd /home/km4886/seld_mlistening_project
./sing --nv python main.py --use_wavelet --wavelet morl --n_wavelet_scales 128 --exp wavelet_test --nb_epochs 100
