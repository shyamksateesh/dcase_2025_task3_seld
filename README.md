# DCASE 2025 Task 3 — Stereo SELD: Reproduction, Transformer, and Wavelet Features
### CS-GY 6933 Machine Listening · NYU · Spring 2026
**Team:** Mohammed Shipat Uddin, Kevin Mai, Sanjay Menon, Shyam Krishna Sateesh  
**Base Repository:** [itsjunwei/NTU_SNTL_Task3](https://github.com/itsjunwei/NTU_SNTL_Task3) (Yeow et al., DCASE 2025)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Environment Setup](#environment-setup)
4. [Dataset](#dataset)
5. [Key Source File Modifications](#key-source-file-modifications)
6. [Data Preparation](#data-preparation)
7. [Running the Yeow GRU Baseline](#running-the-yeow-gru-baseline)
8. [Running the Transformer Modification](#running-the-transformer-modification)
9. [Running Transformer and Wavelet Features](#running-transformer-and-wavelet-features)
10. [Results](#results)
11. [Architecture Details](#architecture-details)
12. [Known Issues & Fixes](#known-issues--fixes)

---

## Project Overview

This repository implements and extends the Yeow et al. DCASE 2025 Task 3 submission — **Stereo Sound Event Localization and Detection (SELD)**. Given a 5-second stereo audio clip, the model simultaneously detects which of 13 sound event classes are active at each 100ms frame, estimates their azimuth angle, and estimates their distance.

**Our contributions:**
1. Reproduced the Yeow et al. baseline (MSIC features + FilterAugment) on real data only
2. Applied Audio Channel Swapping (ACS) augmentation to double directional training data
3. Replaced the biGRU + MHSA sequence model with a **Transformer Encoder using RoPE positional encoding**
4. Added optional **Continuous Wavelet Transform (CWT) scalogram features** as an alternative to mel spectrogram features

**Hardware:** MacBook M3 Pro (Apple Silicon, MPS backend)

---

## Repository Structure

```
project/
├── NTU_SNTL_Task3/              ← Yeow et al. codebase (cloned)
│   ├── main.py                  ← Training entry point (modified)
│   ├── model.py                 ← Model architecture (replaced with Transformer variant)
│   ├── model_gru_original.py    ← Original GRU model backup
│   ├── parameters.py            ← Hyperparameters (modified)
│   ├── data_generator.py        ← Data loading (modified for ACS folds)
│   ├── extract_features.py      ← Feature extraction
│   ├── utils.py                 ← Audio processing utilities
│   ├── training_utils.py        ← Augmentation methods
│   ├── loss.py                  ← ADPIT loss
│   ├── metrics.py               ← SELD evaluation metrics
│   ├── left_right_swap.py       ← ACS augmentation script (path-fixed)
│   ├── track_results.py         ← Compare experiment checkpoints
│   ├── checkpoints/             ← Saved model checkpoints + training_log.csv per run
│   └── wandb/                   ← Offline W&B logs
│
└── DCASE_2025_dataset/          ← Dataset (not in repo, 27.6 GB)
    ├── stereo_dev/
    │   ├── dev-train-sony/      ← fold3_*.wav (~4016 files)
    │   ├── dev-train-tau/       ← fold3_*.wav (~12198 files)
    │   ├── dev-test-sony/       ← fold4_*.wav
    │   ├── dev-test-tau/        ← fold4_*.wav
    │   └── dev-train-realcs/    ← fold3_*_swap.wav (~16214 files, ACS generated)
    ├── metadata_dev/
    │   ├── dev-train-sony/
    │   ├── dev-train-tau/
    │   ├── dev-test-sony/
    │   ├── dev-test-tau/
    │   └── dev-train-realcs/    ← _swap.csv labels (ACS generated)
    ├── stereo_eval/
    │   └── eval/
    ├── mel96_gamma_iv_ms_dnorm/ ← Cached MSIC mel features (auto-generated)
    │   ├── stereo_dev/          ← Raw .pt feature tensors
    │   ├── stereo_dev_normalized/ ← Normalized .pt tensors
    │   ├── metadata_dev_adpit/  ← Multi-ACCDOA label tensors
    │   └── scaler_dev.pkl       ← Fitted StandardScaler
    └── mel128_wavelet_dnorm/    ← Cached CWT wavelet features when --use_wavelet is enabled
```

---

## Environment Setup

```bash
# Create conda environment (Python 3.9 as specified by Yeow)
conda create --name dcase2025_task3 python=3.9.16 -y
conda activate dcase2025_task3

# Install PyTorch for Apple Silicon (MPS support)
# NOTE: requirements.txt specifies torch==1.13.1+cu116 (CUDA build) — DO NOT install this on M3
# Install modern torch instead:
pip install torch torchvision torchaudio

# Install remaining dependencies (skip torch line from requirements.txt)
pip install joblib==1.4.0 librosa==0.10.1 numpy==1.22.4 pandas==2.0.3 \
    rich==14.0.0 scikit-learn==1.3.2 soundfile==0.12.1 \
    torchinfo==1.8.0 tqdm==4.64.1 wandb==0.20.1 tensorboard scipy \
    PyWavelets==1.4.1

# Verify torch and MPS
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"

# Verify wavelet dependency
python -c "import pywt; print(pywt.__version__)"
```

> **Critical:** Always use `python` not `python3` inside this conda environment. On macOS with Homebrew, `python3` resolves to the system Python (3.13), not the conda environment.

---

## Dataset

Download from [Zenodo](https://zenodo.org/records/15559774) (27.6 GB) and place at `../DCASE_2025_dataset/` relative to the repo root.

**Fold structure:**
| Folder | Fold prefix | Role |
|--------|-------------|------|
| `dev-train-sony/` | `fold3_*.wav` | Training |
| `dev-train-tau/` | `fold3_*.wav` | Training |
| `dev-test-sony/` | `fold4_*.wav` | Validation |
| `dev-test-tau/` | `fold4_*.wav` | Validation |
| `dev-train-realcs/` | `fold3_*_swap.wav` | Training (ACS augmented, generated) |

Note: `fold5` and `fold6` referenced in Yeow's original `parameters.py` are synthetic data generated via SpatialScaper — not included in the DCASE download and not used in our experiments.

---

## Key Source File Modifications

### `parameters.py`
Changes from original:

```python
# Fix dataset path (original hardcoded 'DCASE2025_SELD_dataset')
'root_dir': '../DCASE_2025_dataset',

# Add missing key (referenced by main.py but absent from original)
'val_batch_size': 32,

# Add missing key (referenced by data_generator.py get_folds())
'finetune': False,

# Increase mel bins to match Yeow paper (original defaulted to 64)
'nb_mels': 96,

# For Transformer run — increase model capacity
'rnn_size': 256,   # d_model for Transformer (was 128)
'fnn_size': 256,   # decoder hidden dim (was 128)

# Update training folds to include ACS data
'dev_train_folds': ['fold3', 'dev-train-realcs'],

# Wavelet feature defaults
'use_wavelet': False,
'wavelet': 'morl',
'n_wavelet_scales': 128,
```

### `main.py`
Changes:

**1. Fix model import (got corrupted at some point):**
```python
# Line 14 — ensure this reads:
from model import SELDModel
# NOT: from NTU_SNTL_Task3.model import SELDModel
```

**2. Add MPS device support (Apple Silicon):**
```python
# Replace the device detection block at the bottom:
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

**3. Ensure model is on correct device after setup:**
```python
# After setup_model_and_loss() call (~line 196):
seld_model, seld_loss, seld_metrics = setup_model_and_loss(in_feat_shape=in_feat_shape, device=device)
seld_model = seld_model.to(device)  # ← explicit .to(device) required for MPS
```

**4. Add CSV training log:**
```python
# After best_f_score = float('-inf'):
import csv
csv_path = os.path.join(checkpoints_folder, 'training_log.csv')
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerow(['epoch','train_loss','f1','le','de','rde','seld_err'])

# Inside if should_validate block, after val_seld_error computed:
with open(csv_path, 'a', newline='') as f:
    csv.writer(f).writerow([epoch+1, avg_train_loss, round(val_f*100,3),
                             round(val_ang_error,3), round(val_dist_error,3),
                             round(val_rel_dist_error,4), round(val_seld_error,4)])
```

**5. Add Transformer argparse arguments:**
```python
parser.add_argument('--use_transformer', action='store_true', default=False)
parser.add_argument('--tr_nhead', type=int, default=4)
parser.add_argument('--tr_layers', type=int, default=2)
parser.add_argument('--tr_ff_dim', type=int, default=512)
parser.add_argument('--tr_rope', action='store_true', default=True)
```

**6. Add Wavelet argparse arguments:**
```python
parser.add_argument('--use_wavelet', action='store_true', default=False)
parser.add_argument('--wavelet', type=str, default='morl')
parser.add_argument('--n_wavelet_scales', type=int, default=None)
```

### `extract_features.py`
Adds a wavelet feature path that computes CWT magnitude scalograms independently for left and right channels, frames them with the same hop-length timing as the STFT/mel path, and caches them in a separate feature directory so mel and wavelet experiments do not overwrite each other.

Current wavelet mode supports stereo wavelet scalograms only. Binaural STFT-derived feature flags such as `--ipd`, `--gamma`, `--iv`, `--slite`, and `--ms` are not supported together with `--use_wavelet`.

### `data_generator.py`
Fix `get_feature_files()` to handle the `dev-train-realcs` ACS fold (files are named `fold3_*_swap.pt` inside a differently-named folder):

```python
def get_feature_files(self):
    audio_files, label_files = [], []
    for fold in self.folds:
        if fold == 'dev-train-realcs':
            audio_files += glob.glob(os.path.join(self.feat_dir, 
                f'stereo_dev_normalized/fold3*_swap.pt'))
            label_files += glob.glob(os.path.join(self.feat_dir, 
                'metadata_dev{}/fold3*_swap.pt'.format(
                    '_adpit' if self.params['multiACCDOA'] else '')))
        else:
            audio_files += glob.glob(os.path.join(self.feat_dir, 
                f'stereo_dev_normalized/{fold}*.pt'))
            label_files += glob.glob(os.path.join(self.feat_dir, 
                'metadata_dev{}/{}*.pt'.format(
                    '_adpit' if self.params['multiACCDOA'] else '', fold)))
    audio_files = sorted(audio_files, key=lambda x: x.split('/')[-1])
    label_files = sorted(label_files, key=lambda x: x.split('/')[-1])
    return audio_files, label_files
```

### `left_right_swap.py`
Fix hardcoded dataset path in `main()`:

```python
# FROM:
stereo_dir = "DCASE2025_SELD_dataset/stereo_dev"
metadata_dir = "DCASE2025_SELD_dataset/metadata_dev"
# TO:
stereo_dir = "../DCASE_2025_dataset/stereo_dev"
metadata_dir = "../DCASE_2025_dataset/metadata_dev"
```

---

## Data Preparation

Run from inside `NTU_SNTL_Task3/`:

```bash
cd ~/Desktop/NYU/Spring\ 2026/MachineListening/project/NTU_SNTL_Task3
conda activate dcase2025_task3

# Step 1: Generate ACS (Audio Channel Swap) augmented data
# Creates dev-train-realcs/ with ~16,214 channel-swapped training files
# Takes ~20-30 minutes
python left_right_swap.py
```

This doubles the effective directional training data by creating mirror copies of every training clip with swapped L/R channels and negated azimuth labels.

Feature extraction, label processing, and normalization all run **automatically** on the first training run and are cached to disk. Subsequent runs skip extraction entirely (instantaneous).

---

## Running the Yeow GRU Baseline

```bash
cd ~/Desktop/NYU/Spring\ 2026/MachineListening/project/NTU_SNTL_Task3
conda activate dcase2025_task3

WANDB_MODE=offline python main.py \
  --exp Yeow_MSIC_FAFS_ACS \
  --ms --iv --gamma \
  --filtaug \
  --nb_mels 96 \
  --nb_epochs 100
```

**Flag explanation:**
| Flag | Meaning |
|------|---------|
| `--ms` | Add Mid-Side log-mel spectrograms (channels 3+4) |
| `--iv` | Add Mid-Side Intensity Vector (channel 5) |
| `--gamma` | Add Magnitude-Squared Coherence (channel 6) |
| `--filtaug` | Enable FilterAugment data augmentation |
| `--nb_mels 96` | Use 96 mel bins (Yeow paper spec) |
| `--nb_epochs 100` | Train for 100 epochs |

**First run behavior:**
1. Feature extraction: ~16k original + 16k ACS files → ~2 hours, runs once
2. Label extraction: ~5 minutes, runs once  
3. Normalization: fits StandardScaler, saves `scaler_dev.pkl`, runs once
4. Training: 100 epochs, ~13-14 min/epoch on M3 Pro → ~22 hours total

**Confirm correct startup** — check these lines in the output:
```
Device used: mps
Number of batches: 761        ← confirms ACS data loaded (vs 254 without ACS)
In Shape: torch.Size([64, 6, 401, 96])   ← 6 channels, 96 mels
use_transformer: False        ← confirms GRU model
```

---

## Running the Transformer Modification

```bash
# Switch to Transformer model
cp model_gru_original.py model_gru_backup.py   # keep backup
# (model.py should already be the Transformer variant)

WANDB_MODE=offline python main.py \
  --exp Transformer_RoPE_N3_H4_D256 \
  --ms --iv --gamma \
  --filtaug \
  --nb_mels 96 \
  --nb_epochs 100 \
  --use_transformer --tr_nhead 4 --tr_layers 3 --tr_ff_dim 512 --tr_rope \
  --dropout 0.1
```

**Transformer-specific flags:**
| Flag | Value | Meaning |
|------|-------|---------|
| `--use_transformer` | (bool) | Switch sequence model from GRU to Transformer |
| `--tr_nhead` | 4 | Number of attention heads |
| `--tr_layers` | 3 | Number of Transformer encoder layers |
| `--tr_ff_dim` | 512 | Feedforward dimension per layer |
| `--tr_rope` | (bool) | Use RoPE positional encoding (omit for sinusoidal) |
| `--dropout` | 0.1 | Dropout rate (higher than GRU's 0.05 for better regularization) |

> **Note:** `parameters.py` must have `rnn_size: 256` and `fnn_size: 256` for the Transformer run. The `--dropout` flag must be passed explicitly as argparse defaults override parameters.py.

**Feature extraction is fully cached** from the GRU run — goes straight to training.

**Confirm correct startup:**
```
Device used: mps
Number of batches: 761
In Shape: torch.Size([64, 6, 401, 96])
use_transformer: True
tr_nhead: 4, tr_layers: 3, tr_ff_dim: 512, tr_rope: True
dropout: 0.1
```

Model summary should show `TransformerEncoder` with 3 layers, **not** GRU:
```
├─Linear: 1-2    [64, 50, 256]    49,408      ← input projection (192→256)
├─TransformerEncoder: 1-3
│   └─TransformerEncoderLayer: 3-13    526,080
│   └─TransformerEncoderLayer: 3-14    526,080
│   └─TransformerEncoderLayer: 3-15    526,080
Total params: 1,801,589
```

---

## Running Transformer and Wavelet Features

Wavelet mode replaces the default mel/MSIC feature stack with stereo CWT scalograms. This is useful for testing whether multi-scale time-frequency features capture transients and music-like structure better than the Fourier/mel front end.

```bash
WANDB_MODE=offline python main.py \
  --exp new_wavelet_exp_2 \
  --use_wavelet \
  --wavelet morl \
  --n_wavelet_scales 128 \
  --nb_epochs 50 \
  --use_transformer \
  --tr_nhead 4 \
  --tr_layers 3 \
  --tr_ff_dim 512 \
  --tr_rope \
  --filtaug
```

**Wavelet-specific flags:**
| Flag | Value | Meaning |
|------|-------|---------|
| `--use_wavelet` | (bool) | Use CWT scalogram features instead of mel spectrogram features |
| `--wavelet` | `morl` | Wavelet family passed to PyWavelets; `morl` is Morlet |
| `--n_wavelet_scales` | 128 | Number of CWT scales/frequency bins |

**Important compatibility note:** do not combine `--use_wavelet` with `--ms`, `--iv`, `--gamma`, `--ipd`, or `--slite`. Those flags depend on STFT-derived binaural features and currently raise `NotImplementedError` in the wavelet path.

**Expected startup differences:**
```
use_wavelet: True
wavelet: morl
n_wavelet_scales: 128
use_transformer: True
tr_nhead: 4, tr_layers: 3, tr_ff_dim: 512, tr_rope: True
In Shape: torch.Size([64, 2, 401, 128])
```

The first wavelet run builds a separate cache, for example `mel128_wavelet_dnorm/`, so it does not reuse or overwrite the `mel96_gamma_iv_ms_dnorm/` MSIC cache. Subsequent wavelet runs with the same feature settings should skip extraction.

---

## Results

All experiments trained on: fold3 (real recordings, dev-train-sony + dev-train-tau) + dev-train-realcs (ACS augmented), validated on fold4.

### Summary Table

| System | F1 (%) | LE (°) | RDE | Best Epoch | Params |
|--------|--------|--------|-----|------------|--------|
| Yeow published (full data) | 45.3 | 13.2 | 0.262 | — | ~4M |
| DCASE 2025 Baseline | 23.7 | 20.8 | 0.347 | — | — |
| **Our GRU (Yeow features + ACS)** | **27.53** | **15.68** | **0.312** | **45** | **785K** |
| **Our Transformer (RoPE, N=3)** | **28.92** | **13.72** | **0.311** | **39** | **1.80M** |

### GRU vs Transformer — Head to Head

| Metric | GRU Best | Transformer Best | Δ | Winner |
|--------|----------|-----------------|---|--------|
| F1 Score | 27.53% | **28.92%** | +1.39% | Transformer |
| DOA Error | 15.68° | **13.72°** | −1.96° | Transformer |
| Rel. Dist. Error | 0.312 | **0.311** | −0.001 | Transformer |
| SELD Error | 0.373 | **0.366** | −0.007 | Transformer |

**Key finding:** The Transformer Encoder with RoPE positional encoding outperforms the biGRU+MHSA baseline on all metrics. The most significant improvement is in DOA Error (−1.96°), suggesting that direct frame-to-frame attention via RoPE is better suited to spatial audio localization than sequential GRU processing.

### Gap to Yeow Published Results

The ~17% F1 gap to Yeow's published 45.3% is attributed to:
1. **Missing synthetic data** — Yeow used SpatialScaper-generated fold5/fold6 data which significantly mitigates class imbalance in STARSS23 (rare classes: bell F1=0.4%, knock F1=0.0% in our runs)
2. **Real data only** — our training uses fold3 + ACS only (~32k clips vs Yeow's larger combined set)

The GRU vs Transformer comparison is controlled and valid since both models train under identical conditions.

### Per-Class Results (GRU Baseline — Best Checkpoint)

| Class | F1 | DOA Error | RelDist Error |
|-------|----|-----------|---------------|
| Female speech | 0.577 | 14.92° | 0.308 |
| Male speech | 0.643 | 11.97° | 0.348 |
| Clapping | 0.280 | 13.74° | 0.292 |
| Telephone | 0.202 | 15.82° | 0.336 |
| Laughter | 0.258 | 19.06° | 0.181 |
| Domestic sounds | 0.428 | 18.81° | 0.285 |
| Footsteps | 0.091 | 20.93° | 0.262 |
| Door | 0.047 | 16.47° | 0.070 |
| Music | 0.384 | 21.73° | 0.419 |
| Musical instrument | 0.266 | 12.03° | 0.564 |
| Water tap | 0.143 | 11.19° | 0.195 |
| Bell | 0.004 | 11.00° | 0.519 |
| Knock | 0.000 | 23.00° | 0.203 |

---

## Architecture Details

### Input Feature Stack (Yeow MSIC — 6 channels)

```
Stereo WAV (24kHz, 5s)
  ↓ time-domain M/S split
  m[n] = (L + R) / 2,  s[n] = (L - R) / 2
  ↓ STFT (1024-pt FFT, Hann window, hop=300 samples)

Channel 1: log-mel |X_L|²          (T=400, F=96)
Channel 2: log-mel |X_R|²          (T=400, F=96)
Channel 3: log-mel |M|²            (T=400, F=96)  ← MS log-mel
Channel 4: log-mel |S|²            (T=400, F=96)  ← MS log-mel
Channel 5: IV = Re[M·S*]/(|M|²+|S|²+ε), Mel-projected   (T=400, F=96)
Channel 6: MSC = |Φ_LR|²/(Φ_LL·Φ_RR+ε), Mel-projected  (T=400, F=96)

Input tensor: (6, 400, 96)
```

### Input Feature Stack (Wavelet CWT — 2 channels)

```
Stereo WAV (24kHz, 5s)
  ↓ per-channel Continuous Wavelet Transform via PyWavelets
  ↓ magnitude coefficients over configured scales
  ↓ hop-length frame aggregation to match the training timeline

Channel 1: CWT scalogram |W_L(scale, t)|  (T≈400, scales=128)
Channel 2: CWT scalogram |W_R(scale, t)|  (T≈400, scales=128)

Input tensor with --n_wavelet_scales 128: (2, 400, 128)
```

Wavelet mode currently uses only the left/right scalogram channels. It does not include MSIC channels, IPD, coherence/gamma, intensity vector, SALSA-Lite, or Mid-Side features.

### GRU Model (Baseline)

```
Input (B, 6, 401, 96)
  → ConvBlock×3 [pool: (2,4),(2,4),(2,2)] → (B, 64, 50, 3)
  → reshape → (B, 50, 192)
  → biGRU(128, 2 layers) → (B, 50, 256)
  → multiply halves → (B, 50, 128)
  → MHSA(128, 8 heads)×2 + residual + LayerNorm
  → Linear(128→128) + Dropout
  → Linear(128→117)
Output: (B, 50, 117)  [117 = 3 tracks × 13 classes × 3 values]
Total params: ~785K
```

### Transformer Model (Our Modification)

```
Input (B, 6, 401, 96)
  → ConvBlock×3 [identical to GRU] → (B, 64, 50, 3)
  → reshape → (B, 50, 192)
  → Linear(192→256)  [input projection to d_model]
  → TransformerEncoder(d=256, heads=4, layers=3, ff=512, RoPE)
      Pre-norm: LayerNorm → RoPEAttention → residual
      Pre-norm: LayerNorm → FFN(GELU) → residual
      ×3 layers
  → LayerNorm
  → Linear(256→256) + Dropout
  → Linear(256→117)
Output: (B, 50, 117)  [identical output shape]
Total params: ~1.80M
```

With `--use_wavelet --n_wavelet_scales 128`, the Transformer keeps the same temporal encoder design but receives `(B, 2, 401, 128)` input. After the convolutional blocks, the flattened feature dimension becomes `64 × 4 = 256`, so the Transformer can use the same `d_model=256` configuration without the `192→256` projection used by the 6-channel, 96-mel MSIC setup.

### RoPE Positional Encoding

Rotary Position Embedding (Su et al. 2021) encodes **relative** temporal position by rotating Q and K vectors before the attention dot-product:

```
cos, sin = RoPE(seq_len=50, head_dim=64)
q_rot = q * cos + rotate_half(q) * sin
k_rot = k * cos + rotate_half(k) * sin
attn = softmax(q_rot @ k_rot.T / sqrt(head_dim))
```

RoPE is preferred over sinusoidal encoding for audio because it encodes the temporal *distance* between frames rather than their absolute position — the model learns that frame 10 and frame 11 are adjacent regardless of where in the sequence they appear.

---

## Known Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'torch'` | `python3` resolves to Homebrew Python 3.13, not conda env | Always use `python`, not `python3` |
| `RuntimeError: Input type (MPSFloatType) and weight type (torch.FloatTensor)` | Model not moved to MPS device | Add `seld_model = seld_model.to(device)` after `setup_model_and_loss()` |
| `KeyError: 'finetune'` | Key missing from `parameters.py` | Add `'finetune': False` to params dict |
| `KeyError: 'val_batch_size'` | Key missing from `parameters.py` | Add `'val_batch_size': 32` to params dict |
| `left_right_swap.py` processes 0 files | Hardcoded path to `DCASE2025_SELD_dataset/` | Change paths in `main()` to `../DCASE_2025_dataset/` |
| `dropout: 0.05` despite setting 0.1 in parameters.py | argparse `--dropout` default (0.05) overwrites params | Pass `--dropout 0.1` explicitly in CLI |
| ACS files not loaded during training | `data_generator.py` globs `fold*` prefix only | Add `if fold == 'dev-train-realcs'` branch with `fold3*_swap.pt` glob |
| `pin_memory` warning on MPS | PyTorch MPS doesn't support pinned memory | Warning only, not an error — safe to ignore |
| `ModuleNotFoundError: No module named 'pywt'` | PyWavelets not installed | Install `PyWavelets==1.4.1` or run the dependency install command above |
| `NotImplementedError` with `--use_wavelet --gamma` or similar | Wavelet mode does not support STFT-derived binaural feature flags yet | Remove `--ms`, `--iv`, `--gamma`, `--ipd`, and `--slite` when using `--use_wavelet` |
| Wavelet input shape mismatch | Cached features were created with a different scale count or feature mode | Use a new experiment/cache setting or clear the matching wavelet feature cache before rerunning |

---

## Viewing Results

```bash
# Compare all experiment checkpoints
python track_results.py

# Training curves are saved per-run at:
# checkpoints/<exp_name>_audio_multiACCDOA_<timestamp>/training_log.csv
```

---

## Citation

```bibtex
@techreport{Yeow_NTU_task3a_report,
    Author = "Yeow, Jun-Wei and Tan, Ee-Leng and Peksi, Santi and Gan, Woon-Seng",
    title = "Improving Stereo 3D Sound Event Localization and Detection: Perceptual Features, 
             Stereo-Specific Data Augmentation, and Distance Normalization",
    institution = "DCASE2025 Challenge",
    year = "2025"
}
```
