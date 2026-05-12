# DCASE 2025 Task 3 — Stereo Sound Event Localization and Detection

**CS-GY 6933 Machine Listening · NYU · Spring 2026**

Mohammed Shipat Uddin · Kevin Mai · Sanjay Menon · Shyam Krishna Sateesh

This repository extends the [Yeow et al. DCASE 2025 Task 3 submission](https://github.com/itsjunwei/NTU_SNTL_Task3) with two architectural modifications to the stereo SELD pipeline: a Transformer Encoder sequence model (replacing the biGRU), and a Continuous Wavelet Transform feature extractor (replacing the STFT).

---

## Branch Overview

The repository is organized across three branches. Each branch contains a complete, self-contained experiment.

### `main` — Yeow GRU Baseline + Transformer Encoder

The primary branch. Contains the reproduced Yeow et al. baseline (MSIC features, FilterAugment, Audio Channel Swapping) and the Transformer Encoder modification.

Key files:
- `model.py` — Transformer Encoder with RoPE positional encoding (our network modification)
- `model_gru.py` — Original biGRU + MHSA baseline model (for reference/ablation)
- `main.py` — Training entry point with `--use_transformer` flag to switch between architectures
- `parameters.py` — Hyperparameters; set `rnn_size: 256`, `fnn_size: 256` for Transformer
- `data_generator.py` — Modified to support ACS augmented fold (`dev-train-realcs`)
- `left_right_swap.py` — Generates ACS augmented training data
- `eval_checkpoint.py` — Evaluates a saved checkpoint without retraining
- `checkpoints/` — Saved training logs (`training_log.csv`) per experiment run

**To run the GRU baseline:**
```bash
WANDB_MODE=offline python main.py \
  --exp Yeow_GRU --ms --iv --gamma --filtaug --nb_mels 96 --nb_epochs 100
```

**To run the Transformer modification:**
```bash
WANDB_MODE=offline python main.py \
  --exp Transformer_RoPE --ms --iv --gamma --filtaug --nb_mels 96 --nb_epochs 100 \
  --use_transformer --tr_nhead 4 --tr_layers 3 --tr_ff_dim 512 --tr_rope --dropout 0.1
```

---

### `cwt` — Continuous Wavelet Transform Features 

Replaces the STFT-based feature extraction with a Continuous Wavelet Transform (CWT) using Morlet wavelets via `librosa.cqt`. All six input channels (MS log-mel, IV, MSC) are recomputed from wavelet coefficients instead of STFT coefficients. The GRU + MHSA sequence model is unchanged.

Key files beyond the baseline:
- `extract_cwt_6channel.py` — Main CWT 6-channel feature extraction pipeline
- `extract_wavelet_features.py` — Sharded extraction for HPC environments
- `utils.py` — Modified `extract_stereo_features()` to use CWT
- `CWT_6CHANNEL_IMPLEMENTATION.md` — Full implementation details
- `CWT_6CHANNEL_QUICKSTART.md` — Quick reference for running the CWT pipeline
- `CWT_6CHANNEL_ARCHITECTURE.md` — Architecture diagrams and channel descriptions
- `COMPLETION_SUMMARY.md` — Summary of what was implemented and tested
- `training_results.csv` — CWT experiment training results
- `.sbatch` files — HPC job scripts for NYU Greene cluster

---

### `combined_code` — Transformer + CWT Combined

Combines both modifications: CWT feature extraction feeding into the Transformer Encoder sequence model. This branch merges the changes from `main` and `cwt`.

Key files:
- `extract_features.py` — Combined CWT + MSIC feature extraction
- `model.py` — Transformer Encoder (same as `main`)
- `main.py` — Updated to support both modifications simultaneously
- `parameters.py` — Configured for combined run
- `checkpoints/` — Combined experiment training logs

---

## Dataset

Download the DCASE 2025 Task 3 dataset from [Zenodo](https://zenodo.org/records/15559774) (~27.6 GB) and place it at `../DCASE_2025_dataset/` relative to the repo root.

Expected structure:
```
DCASE_2025_dataset/
├── stereo_dev/
│   ├── dev-train-sony/    (fold3_*.wav)
│   ├── dev-train-tau/     (fold3_*.wav)
│   ├── dev-test-sony/     (fold4_*.wav)
│   └── dev-test-tau/      (fold4_*.wav)
└── metadata_dev/
    ├── dev-train-sony/
    ├── dev-train-tau/
    ├── dev-test-sony/
    └── dev-test-tau/
```

Note: `fold5` and `fold6` referenced in the original Yeow repository are synthetic data generated via SpatialScaper and are not included in the DCASE download. None of our experiments use synthetic data.

---

## Environment Setup

```bash
conda create --name dcase2025_task3 python=3.9.16 -y
conda activate dcase2025_task3
pip install torch torchvision torchaudio
pip install joblib==1.4.0 librosa==0.10.1 numpy==1.22.4 pandas==2.0.3 \
    rich==14.0.0 scikit-learn==1.3.2 soundfile==0.12.1 \
    torchinfo==1.8.0 tqdm==4.64.1 wandb==0.20.1 tensorboard scipy
```

Use `python` (not `python3`) inside the conda environment.

---

## Results Summary

All experiments trained on real recordings only (fold3 + ACS augmented, ~32k clips), validated on fold4.

| System | F-score | DOA Error | Rel. Dist. Error | Best Epoch |
|--------|---------|-----------|-----------------|------------|
| DCASE 2025 Baseline (published) | 23.7% | 20.8° | 0.347 | — |
| Yeow et al. (published, full data) | 45.3% | 13.2° | 0.262 | — |
| Yeow GRU (ours, `main`) | 27.47% | 15.33° | 0.301 | 53 |
| CWT only (ours, `cwt`) | 19.44% | 19.92° | 0.315 | 21 |
| **Transformer only (ours, `main`)** | **28.51%** | **14.30°** | **0.273** | **35** |
| Transformer + CWT (ours, `combined_code`) | 24.21% | 17.92° | 0.309 | 35 |

---

## Citation

```bibtex
@techreport{Yeow_NTU_task3a_report,
    Author = "Yeow, Jun-Wei and Tan, Ee-Leng and Peksi, Santi and Gan, Woon-Seng",
    title  = "Improving Stereo 3D Sound Event Localization and Detection: Perceptual
              Features, Stereo-Specific Data Augmentation, and Distance Normalization",
    institution = "DCASE2025 Challenge",
    year = "2025"
}
```