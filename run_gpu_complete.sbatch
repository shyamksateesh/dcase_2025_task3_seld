cat > run_gpu_complete.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=wavelet_gpu
#SBATCH --output=wavelet_gpu_%j.out
#SBATCH --error=wavelet_gpu_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_1026_general

cd /home/km4886/seld_mlistening_project

export PYTHONUNBUFFERED=1

./sing << EOF
python -u main.py --use_wavelet --wavelet morl --n_wavelet_scales 128 --exp wavelet_test --nb_epochs 100
EOF
# EOF

# # Wait for CPU job to complete first, then submit:
# sbatch run_gpu_complete.sbatch