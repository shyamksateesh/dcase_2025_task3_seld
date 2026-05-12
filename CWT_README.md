# CWT 6-Channel Wavelet Pipeline
### CS-GY 6933 Machine Listening · NYU · Spring 2026
**Project:** Continuous Wavelet Transform (CWT) feature pipeline for stereo SELD
**Base Repository:** [shyamksateesh/dcase_2025_task3_seld](https://github.com/shyamksateesh/dcase_2025_task3_seld)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Details](#implementation-details)
5. [Processing Pipeline](#processing-pipeline)
6. [Results](#results)
7. [How to Run](#how-to-run)
8. [Tests](#tests)
9. [Notes](#notes)

---

## Project Overview

This project replaces the original mel-spectrogram-based SELD features with a **pure CWT-based 6-channel representation**. The goal is to keep the same stereo SELD training pipeline while making all feature channels wavelet-derived.

**What changed:**
1. Left and right stereo channels are converted to CWT scalograms
2. Mid and side signals are also converted to CWT scalograms
3. Wavelet-domain intensity vector and magnitude-squared coherence are computed directly from CWT coefficients
4. The resulting 6-channel tensors are normalized and cached for training

---

## Repository Structure

```
project/
├── main.py                      ← Training entry point
├── extract_wavelet_features.py  ← Feature extraction runner
├── extract_features.py          ← Feature extraction + normalization logic
├── utils.py                     ← Audio loading and feature helpers
├── parameters.py                ← Configuration values
├── test_cwt_6channel_simple.py  ← Synthetic sanity test
├── test_cwt_6channel.py         ← End-to-end cached-feature test
├── CWT_6CHANNEL_ARCHITECTURE.md ← Original architecture notes
├── CWT_6CHANNEL_IMPLEMENTATION.md ← Original implementation notes
└── DCASE2025_SELD_dataset/      ← Dataset and cached feature outputs
```

---

## Architecture Overview

The CWT pipeline starts from a stereo WAV file and produces a 6-channel tensor of shape `(6, T, S)`, where `T` is the number of time frames and `S` is the number of wavelet scales.

### High-level data flow

1. Load stereo audio
2. Compute wavelet coefficients for:
   - Left channel
   - Right channel
   - Mid channel `(L + R) / 2`
   - Side channel `(L - R) / 2`
3. Derive wavelet-domain intensity vector from the mid/side coefficients
4. Derive wavelet-domain magnitude-squared coherence from the left/right coefficients
5. Stack all 6 channels into a single feature tensor
6. Normalize with `StandardScaler`
7. Save normalized tensors for model training

### Channel definitions

- **Channel 1:** Left-channel CWT scalogram
- **Channel 2:** Right-channel CWT scalogram
- **Channel 3:** Mid-channel CWT scalogram
- **Channel 4:** Side-channel CWT scalogram
- **Channel 5:** Wavelet-domain intensity vector
- **Channel 6:** Wavelet-domain magnitude-squared coherence

The result is a unified wavelet feature representation with no STFT/mel mixing.

### Architecture diagram

```text
Stereo WAV (2 ch, 24 kHz)
   │
   ▼
load_audio()
   │
   ▼
extract_cwt_6channel_features()
   │
   ├─ CWT(L) ───────────────► Ch 1
   ├─ CWT(R) ───────────────► Ch 2
   ├─ CWT(M=(L+R)/2) ───────► Ch 3
   ├─ CWT(S=(L-R)/2) ───────► Ch 4
   ├─ Re(CWT(M)·CWT(S)*) ───► Ch 5
   └─ |CWT(L)·CWT(R)*|² ────► Ch 6
   │
   ▼
Stack (6, T, S)
   │
   ▼
StandardScaler normalization
   │
   ▼
Cached normalized tensors
   │
   ▼
SELD model training
```

---

## Implementation Details

### New feature extractor

The core logic lives in `utils.py` and is called from `extract_features.py` when wavelet mode is enabled.

#### Output shape

```python
(6, T, n_wavelet_scales)
```

#### Parameters

- `wavelet`: usually `morl`
- `n_wavelet_scales`: typically `128`
- `hop_length`: matches the SELD frame rate

### Channel computation details

#### Channels 1-2: Left and right scalograms

For each stereo channel, the extractor computes a CWT over the requested wavelet scales and frame-aggregates the magnitude within each hop window.

```python
coef = pywt.cwt(channel, scales, wavelet, sampling_period=1 / sr)
mag = abs(coef)
```

#### Channels 3-4: Mid and side scalograms

The derived mid and side signals are defined as:

```python
M = (L + R) / 2
S = (L - R) / 2
```

The pipeline applies the same CWT-based scalogram extraction to each derived signal.

#### Channel 5: Wavelet-domain intensity vector

The intensity vector is computed from the mid/side cross-spectrum:

```python
cross = coef_M * conj(coef_S)
I_num = real(cross)
I_denom = abs(coef_M) ** 2 + abs(coef_S) ** 2 + 1e-8
I_norm = I_num / I_denom
```

This provides a bounded spatial cue derived entirely from wavelet coefficients.

#### Channel 6: Wavelet-domain magnitude-squared coherence

The coherence channel is computed from the left/right cross-spectrum:

```python
cross = coef_L * conj(coef_R)
MSC = abs(cross) ** 2 / (abs(coef_L) ** 2 * abs(coef_R) ** 2 + 1e-8)
```

This captures inter-channel coherence in the wavelet domain.

### Channel computation summary

#### 1-4: Direct scalograms
For each stereo or derived signal, compute CWT coefficients and aggregate them across the hop-length frame window.

#### 5: Intensity vector
Compute the normalized real cross-spectrum from mid and side wavelet coefficients.

#### 6: MSC / coherence
Compute the normalized magnitude-squared coherence from left and right wavelet coefficients.

### Normalization

Features are normalized with `StandardScaler` per scale and cached to disk. The normalized feature directory is used by the training loader.

### Integration points in the codebase

- `extract_wavelet_features.py`: command-line entry point for extraction and normalization
- `extract_features.py`: extracts audio features, labels, and normalized tensors
- `utils.py`: contains the CWT feature extraction helper
- `main.py`: consumes the normalized tensors during training

---

## Processing Pipeline

```text
Stereo WAV
   │
   ▼
Load audio
   │
   ▼
Compute 6-channel CWT features
   │
   ▼
Save raw feature tensors
   │
   ▼
Fit/Apply StandardScaler
   │
   ▼
Save normalized tensors
   │
   ▼
Train SELD model
```

### Output folders

- Raw features: `stereo_dev/`
- Normalized features: `stereo_dev_normalized/`
- Labels: `metadata_dev/` or `metadata_dev_adpit/`
- Scaler: `scaler_dev.pkl`

### Practical processing notes

- The raw feature extraction stage is the most expensive step.
- Normalization is only fitted once per feature set and then reused.
- If you add or change the training audio set, re-run extraction for the new files and then rerun normalization.

---

## Results

The table below compares the best reported checkpoints for Yeow, our updated GRU baseline, and the CWT model.

### Best checkpoint comparison

| System | F1 | LE (°) | DE (°) | RDE | Best Epoch | Notes |
|--------|----:|-------:|-------:|----:|-----------:|-------|
| Yeow published baseline | 45.3 | 13.2 | — | 0.262 | — | Published full-data baseline |
| Updated GRU (our baseline) | 27.53 | 15.68 | — | 0.312 | 45 | Real-data + ACS |
| CWT 6-channel (our run) | 19.44 | 19.92 | 51.29 | 0.315 | 21 | Best checkpoint from `training_results.csv` |

### What this comparison shows

- **Yeow published baseline** remains the strongest overall result in the comparison.
- **Updated GRU** is our strongest in-repo recurrent baseline.
- **CWT 6-channel** is currently below the updated GRU on F1 and LE, but it is a complete wavelet-domain feature pipeline and already reaches a stable best checkpoint at epoch 21.
- The CWT run still provides a useful research baseline for future tuning, especially if you want to explore better wavelets, more scales, stronger augmentation, or a different sequence model.

### Best CWT checkpoint summary

- **Best validation F1:** `19.442`
- **Best epoch:** `21`
- **Best validation losses:** steadily decreased across training, reaching `0.0024` by epoch 50
- **Observed pattern:** validation F1 improved quickly early on, then plateaued around the high teens to low twenties

### Epoch log

```text
epoch,lr,train_time_s,val_time_s,loss,f1,le,de,rde,best_epoch,best_f1,improved
1,4.26e-05,37,235,0.0278,0.012,34.71,52.69,0.340,1,0.01,yes
2,5.05e-05,25,0,0.0224,0.012,34.71,52.69,0.340,1,0.01,no
3,6.35e-05,24,296,0.0189,8.496,19.92,58.45,0.397,3,8.50,yes
4,8.15e-05,24,0,0.0154,8.496,19.92,58.45,0.397,3,8.50,no
5,1.04e-04,24,304,0.0133,10.928,21.30,57.65,0.394,5,10.93,yes
6,1.32e-04,27,0,0.0119,10.928,21.30,57.65,0.394,5,10.93,no
7,1.63e-04,25,309,0.0107,11.105,20.73,56.76,0.389,7,11.10,yes
8,1.99e-04,25,0,0.0099,11.105,20.73,56.76,0.389,7,11.10,no
9,2.38e-04,23,303,0.0092,12.232,19.30,64.04,0.377,9,12.23,yes
10,2.80e-04,24,0,0.0086,12.232,19.30,64.04,0.377,9,12.23,no
11,3.25e-04,24,310,0.0081,12.694,20.07,58.22,0.356,11,12.69,yes
12,3.72e-04,25,0,0.0077,12.694,20.07,58.22,0.356,11,12.69,no
13,4.20e-04,25,315,0.0073,13.809,20.42,55.15,0.356,13,13.81,yes
14,4.70e-04,25,0,0.0069,13.809,20.42,55.15,0.356,13,13.81,no
15,5.20e-04,25,313,0.0066,15.172,21.02,52.70,0.322,15,15.17,yes
16,5.70e-04,26,0,0.0062,15.172,21.02,52.70,0.322,15,15.17,no
17,6.20e-04,24,314,0.0059,15.812,21.71,56.54,0.358,17,15.81,yes
18,6.68e-04,26,0,0.0056,15.812,21.71,56.54,0.358,17,15.81,no
19,7.15e-04,26,321,0.0054,17.797,19.88,55.40,0.343,19,17.80,yes
20,7.60e-04,25,0,0.0052,17.797,19.88,55.40,0.343,19,17.80,no
21,8.02e-04,25,313,0.0050,19.442,19.92,51.29,0.315,21,19.44,yes
22,8.41e-04,25,0,0.0048,19.442,19.92,51.29,0.315,21,19.44,no
23,8.77e-04,26,328,0.0045,16.456,20.74,62.44,0.361,21,19.44,no
24,9.08e-04,26,0,0.0044,16.456,20.74,62.44,0.361,21,19.44,no
25,9.36e-04,25,319,0.0042,17.719,20.11,53.36,0.336,21,19.44,no
26,9.59e-04,26,0,0.0041,17.719,20.11,53.36,0.336,21,19.44,no
27,9.77e-04,26,317,0.0039,18.589,19.15,55.86,0.341,21,19.44,no
28,9.90e-04,25,0,0.0038,18.589,19.15,55.86,0.341,21,19.44,no
29,9.97e-04,25,313,0.0037,18.815,19.74,54.80,0.335,21,19.44,no
30,1.00e-03,26,0,0.0036,18.815,19.74,54.80,0.335,21,19.44,no
31,9.99e-04,24,314,0.0035,17.446,18.88,62.97,0.358,21,19.44,no
32,9.98e-04,24,0,0.0033,17.446,18.88,62.97,0.358,21,19.44,no
33,9.95e-04,24,323,0.0033,18.865,19.08,57.13,0.354,21,19.44,no
34,9.92e-04,26,0,0.0031,18.865,19.08,57.13,0.354,21,19.44,no
35,9.87e-04,25,316,0.0031,18.879,18.94,54.93,0.337,21,19.44,no
36,9.82e-04,24,0,0.0030,18.879,18.94,54.93,0.337,21,19.44,no
37,9.76e-04,23,314,0.0029,19.424,18.68,55.51,0.329,21,19.44,no
38,9.68e-04,26,0,0.0028,19.424,18.68,55.51,0.329,21,19.44,no
39,9.59e-04,25,316,0.0028,19.431,18.72,55.36,0.330,21,19.44,no
40,9.49e-04,24,0,0.0027,19.431,18.72,55.36,0.330,21,19.44,no
41,9.38e-04,25,318,0.0027,19.386,18.79,55.74,0.331,21,19.44,no
42,9.25e-04,25,0,0.0027,19.386,18.79,55.74,0.331,21,19.44,no
43,9.11e-04,24,317,0.0026,19.417,18.63,55.28,0.329,21,19.44,no
44,8.96e-04,25,0,0.0026,19.417,18.63,55.28,0.329,21,19.44,no
45,8.80e-04,25,319,0.0026,19.402,18.81,55.82,0.331,21,19.44,no
46,8.62e-04,24,0,0.0025,19.402,18.81,55.82,0.331,21,19.44,no
47,8.43e-04,24,316,0.0025,19.428,18.70,55.43,0.330,21,19.44,no
48,8.23e-04,25,0,0.0025,19.428,18.70,55.43,0.330,21,19.44,no
49,8.02e-04,25,318,0.0024,19.391,18.76,55.68,0.331,21,19.44,no
50,7.80e-04,24,0,0.0024,19.391,18.76,55.68,0.331,21,19.44,no
```

### Results interpretation

- The model improved rapidly in the first 20–25 epochs.
- The best validation F1 appeared at epoch 21.
- After that point, the validation F1 oscillated around the high teens while the loss continued to fall.
- The learning rate followed a warmup-like rise and then decayed, which matches the observed training pattern.

---

## How to Run

### 1. Extract wavelet features

```bash
python extract_wavelet_features.py --use_wavelet --wavelet morl --n_wavelet_scales 128 --split dev
```

### 2. Normalize only

If raw features are already extracted:

```bash
python extract_wavelet_features.py --use_wavelet --wavelet morl --n_wavelet_scales 128 --split dev --normalize_only
```

### 3. Train the model

```bash
python main.py --use_wavelet --wavelet morl --n_wavelet_scales 128 --nb_epochs 100
```

### 4. Slurm batch example

```bash
sbatch run_cwt_6channel.sbatch
```

---

## Tests

### Synthetic sanity test

```bash
python test_cwt_6channel_simple.py
```

Use this to verify the feature shapes and ensure the CWT channels are distinct and numerically stable.

### End-to-end cached-feature test

```bash
python test_cwt_6channel.py
```

This test checks loading the cached normalized tensors and running the model forward pass.

---

## Notes

- The CWT pipeline is designed to replace mel-spectrogram features entirely.
- All 6 channels are derived in the wavelet domain for consistency.
- `extract_wavelet_features.py` handles feature extraction, label extraction, and normalization.
- Existing cached features can be reused once created; future runs will skip completed files.
- The architecture and implementation details in this README are intentionally kept together so the document can stand alone without the separate CWT notes.

### Practical tips

- Use `--normalize_only` after the raw features are extracted.
- If you add new training files, extract only the missing feature files and rerun normalization if the training set changes.
- Keep `wavelet='morl'` and `n_wavelet_scales=128` for the current setup unless you are explicitly experimenting.

---

## Summary

The CWT pipeline provides a unified 6-channel stereo SELD representation built entirely from wavelet coefficients. It mirrors the original SELD feature flow, but avoids mel/STFT mixing and keeps the feature extraction path consistent from start to finish.
