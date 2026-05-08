# CWT 6-Channel Architecture Diagram

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      STEREO AUDIO FILE (WAV)                    │
│                    (2 channels, 24 kHz, ~1 sec)                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  load_audio()              │
        │  audio: (2, n_samples)     │
        └────────────┬───────────────┘
                     │
                     ▼
    ┌────────────────────────────────────┐
    │  extract_cwt_6channel_features()   │  ◄─ NEW FUNCTION
    └────────────────────────────────────┘
         Wavelet parameters:
         - wavelet: 'morl'
         - n_scales: 128
         - hop_length: 300 samples
         │
         ├─ CWT(L_channel)  ─────────────┐ Channel 1
         │                               │ Channel 2: R-Scalogram
         ├─ CWT(R_channel)  ◄────────────┼─ Channels 3-4
         │                               │ (M and S from CWT)
         ├─ CWT(M_channel)  ◄────────────┘
         │
         ├─ CWT(S_channel)
         │
         ├─ Compute wavelet Intensity    ◄─ Channel 5
         │  I = Re(M·S*) / (|M|²+|S|²+ε)
         │  (per scale, per frame)
         │
         └─ Compute wavelet Coherence    ◄─ Channel 6
            MSC = |L·R*|² / (|L|²·|R|²+ε)
            (per scale, per frame)
                     │
                     ▼
        ┌─────────────────────────────┐
        │ Stack 6 channels            │
        │ output: (6, T, 128)         │
        │ - 6 channels                │
        │ - T time frames (80 frames) │
        │ - 128 wavelet scales        │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌─────────────────────────────┐
        │ StandardScaler.fit_transform │
        │ - Fit on all training files │
        │ - Per-scale normalization   │
        │ - Mean=0, Std=1             │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌─────────────────────────────┐
        │ Save normalized features    │
        │ stereo_dev_normalized/      │
        │ fold*_room*_mix*.pt         │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌─────────────────────────────┐
        │ Load to train SELD model    │
        │ batch: (B, 6, T, 128)      │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌─────────────────────────────┐
        │      SELDModel Forward      │
        │ predictions: (B, 50, 117)   │
        │ (ADPIT output)              │
        └─────────────────────────────┘
```

## Channel Computation Details

```
CHANNEL 1-2: L-Scalogram & R-Scalogram
═════════════════════════════════════
For each channel in {Left, Right}:
  scales = [1, 2, ..., 128]
  coef = pywt.cwt(channel, scales, 'morl', 1/24000)  ← CWT
  mag = |coef|  # (128, n_samples)
  
  For each time frame t:
    output[t, :] = mean(mag[:, t*300:(t+1)*300])  ← Frame aggregate
  Result: (T, 128)

CHANNEL 3-4: M-Scalogram & S-Scalogram
═════════════════════════════════════
M = (Left + Right) / 2
S = (Left - Right) / 2

  coef_M = pywt.cwt(M, scales, 'morl', 1/24000)
  coef_S = pywt.cwt(S, scales, 'morl', 1/24000)
  
  mag_M = |coef_M|  # (128, n_samples)
  mag_S = |coef_S|  # (128, n_samples)
  
  For each time frame t:
    output_M[t, :] = mean(mag_M[:, t*300:(t+1)*300])
    output_S[t, :] = mean(mag_S[:, t*300:(t+1)*300])
  Result: 2 × (T, 128)

CHANNEL 5: Intensity Vector (Wavelet Domain)
═════════════════════════════════════════════
coef_M = pywt.cwt(M, scales, 'morl', 1/24000)  # (128, n_samples)
coef_S = pywt.cwt(S, scales, 'morl', 1/24000)  # (128, n_samples)

Cross-spectrum:
  cross = coef_M × conj(coef_S)  # (128, n_samples)
  I_num = Re(cross)              # (128, n_samples)

Energy:
  E_M = |coef_M|²                # (128, n_samples)
  E_S = |coef_S|²                # (128, n_samples)
  E = E_M + E_S + 1e-8           # (128, n_samples)

Normalized intensity:
  I_norm = I_num / E             # (128, n_samples)

For each time frame t:
  output[t, :] = mean(I_norm[:, t*300:(t+1)*300])  ← Frame aggregate
  
  Range: [-0.5, +0.5] (per scale)
  Result: (T, 128)

CHANNEL 6: Magnitude-Squared Coherence (Wavelet Domain)
═══════════════════════════════════════════════════════
coef_L = pywt.cwt(L, scales, 'morl', 1/24000)  # (128, n_samples)
coef_R = pywt.cwt(R, scales, 'morl', 1/24000)  # (128, n_samples)

Auto-spectra:
  E_L = |coef_L|²                # (128, n_samples)
  E_R = |coef_R|²                # (128, n_samples)

Cross-spectrum:
  cross = coef_L × conj(coef_R)  # (128, n_samples)
  cross_power = |cross|²         # (128, n_samples)

MSC:
  MSC = cross_power / (E_L × E_R + 1e-8)  # (128, n_samples)

For each time frame t:
  output[t, :] = mean(MSC[:, t*300:(t+1)*300])  ← Frame aggregate
  
  Range: [0, 1] (per scale, bounded coherence)
  Result: (T, 128)

FINAL OUTPUT
════════════
Stack all 6 channels:
  (6, T, 128)
  where:
  - 6 = number of channels
  - T = number of time frames (typically 80 for 1-second audio)
  - 128 = number of wavelet scales
```

## Comparison: Old vs. New

```
BEFORE (Mel-spectrogram based)
═══════════════════════════════

STFT(L) ─┐
         ├─→ Mel-scale ─┬─ Ch1: L-Mel
STFT(R) ─┤             │
         │             ├─ Ch2: R-Mel
         └─→ M/S ──────┤
            (separate) │
                       ├─ Ch3-4: M/S-Mel
                       │
                       ├─ Ch5: IV (STFT-based, separate computation)
                       │
                       └─ Ch6: MSC (STFT-based, separate computation)

Issues:
  × Mixing STFT for mel, separate computation for IV/MSC
  × No wavelet domain for binaural features
  × Inconsistent feature sources

AFTER (Pure wavelet-based)
═══════════════════════════

CWT(L) ──┐
         ├─ Ch1: L-Scalogram
CWT(R) ──┤
         ├─ Ch2: R-Scalogram
         │
         ├─ CWT(M) ────┬─ Ch3: M-Scalogram
         │             │
         ├─ CWT(S) ────┼─ Ch4: S-Scalogram
         │             │
         │             ├─ Ch5: Intensity (wavelet cross-spectrum)
         │             │
         └─ M·S* ──────┤
            L·R* ──────└─ Ch6: MSC (wavelet coherence)

Advantages:
  ✓ All channels from CWT (unified)
  ✓ Binaural features in wavelet domain
  ✓ Consistent parameterization
  ✓ No STFT mixing
```

## Processing Pipeline

```
Audio Files (30,000)
       │
       ├─────────────────────────────────┐
       │                                 │
       ▼                                 ▼
   extract_cwt_6channel.py         run_cwt_6channel.sbatch
   (local single-threaded)         (Slurm 32 CPUs)
       │                                 │
       └─────────────────────────────────┘
                      │
                      ▼
      stereo_dev/ (raw features, 150 GB)
         ├─ fold1_room1_mix001.pt
         ├─ fold1_room1_mix002.pt
         └─ ... (30,000 files)
                      │
                      ▼ (StandardScaler.fit)
              Fit normalization
              scaler_dev.pkl
                      │
                      ▼
    stereo_dev_normalized/ (ready for training)
         ├─ fold1_room1_mix001.pt (normalized)
         ├─ fold1_room1_mix002.pt (normalized)
         └─ ... (30,000 files)
                      │
                      ▼
              main.py (training)
         Load batches of normalized features
         Train SELD model
         Monitor val metrics
```

## Model Integration

```
Input to Model
══════════════
Normalized features:  (6, 80, 128)
Batch:               (B=32, 6, 80, 128)
                         │
                         ▼
                  ┌─────────────────┐
                  │  SELDModel      │
                  │ (Conv blocks)   │
                  └─────────────────┘
                         │
                         ▼
                  ┌─────────────────┐
                  │ (64, 50, 3)     │
                  │ (after pooling) │
                  └─────────────────┘
                         │
                         ▼
                  ┌─────────────────┐
                  │  Reshape        │
                  │ (B, 50, 192)    │
                  └─────────────────┘
                         │
                         ▼
                  ┌─────────────────┐
                  │ GRU (or RNN)    │
                  └─────────────────┘
                         │
                         ▼
                  ┌─────────────────┐
                  │ Dense layer     │
                  │ (B, 50, 117)    │
                  │ (ADPIT output)  │
                  └─────────────────┘
```

---

**Key Takeaway**: All 6 channels computed purely in wavelet domain, enabling unified, consistent feature representation for SELD task.
