#!/bin/bash
#SBATCH --job-name=wavelet_gpu
#SBATCH --output=wavelet_gpu_%j.out
#SBATCH --error=wavelet_gpu_%j.err
#SBATCH --time=6:00:00
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
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "=== [START] Wavelet GPU training job ${SLURM_JOB_ID} at $(date) on $(hostname) ==="

echo "=== [PRECHECK] Host GPU visibility ==="
nvidia-smi || true

echo "=== [PRECHECK] Verifying PyTorch CUDA in singuconda env ==="
timeout 45s ./sing python -u - 2>&1 << 'PY'
import sys
import torch
try:
    print("[PRECHECK] torch.__version__:", torch.__version__, flush=True)
    print("[PRECHECK] torch.version.cuda:", torch.version.cuda, flush=True)
    print("[PRECHECK] cuda.is_available() start", flush=True)
    avail = torch.cuda.is_available()
    print("[PRECHECK] cuda.is_available():", avail, flush=True)
    if avail:
        print("[PRECHECK] device_count:", torch.cuda.device_count(), flush=True)
        print("[PRECHECK] device_0:", torch.cuda.get_device_name(0), flush=True)
        x = torch.tensor([1.0], device='cuda')
        print("[PRECHECK] cuda tensor ok:", float(x.item()), flush=True)
except Exception as e:
    print("[PRECHECK] ERROR:", str(e), file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
PY

precheck_exit=$?
if [ $precheck_exit -ne 0 ]; then
    if [ $precheck_exit -eq 124 ]; then
        echo "=== [FAILED] CUDA precheck timed out after 45s ==="
    else
        echo "=== [FAILED] CUDA precheck failed with exit code ${precheck_exit} ==="
    fi
    exit $precheck_exit
fi

echo "=== [RUN] Starting training on GPU ==="
srun --ntasks=1 ./sing python -u main.py --use_wavelet --wavelet morl --n_wavelet_scales 128 --exp wavelet_test --nb_epochs 100

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "=== [DONE] Wavelet GPU training job ${SLURM_JOB_ID} finished successfully at $(date) ==="
else
    echo "=== [FAILED] Wavelet GPU training job ${SLURM_JOB_ID} failed with exit code ${exit_code} at $(date) ==="
fi

exit $exit_code