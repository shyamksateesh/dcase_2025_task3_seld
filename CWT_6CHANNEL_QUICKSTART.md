# CWT 6-Channel Feature Pipeline - Quick Start Guide

## What Is This?

This is a **pure wavelet-domain alternative** to mel-spectrogram features for the SELD task. All 6 channels are derived directly from Continuous Wavelet Transform (CWT) coefficients, without any STFT/mel dependency.

## 6 Channels

| # | Name | Description | Formula |
|---|------|-------------|---------|
| 1 | L-Scalogram | CWT magnitude of left channel | \|CWT(L)\| |
| 2 | R-Scalogram | CWT magnitude of right channel | \|CWT(R)\| |
| 3 | M-Scalogram | CWT magnitude of mid = (L+R)/2 | \|CWT(M)\| |
| 4 | S-Scalogram | CWT magnitude of side = (L-R)/2 | \|CWT(S)\| |
| 5 | Intensity | Wavelet cross-spectrum intensity | Re(CWT(M)·conj(CWT(S))) / (E_M + E_S + ε) |
| 6 | MSC | Magnitude-squared coherence | \|CWT(L)·conj(CWT(R))\|² / (E_L·E_R + ε) |

**Output shape**: `(6, T, n_scales)` where:
- 6 = number of channels
- T = time frames
- n_scales = number of wavelet scales (typically 128)

## Quick Start

### 1. Test on Synthetic Audio (Fast ✓)

Validates the extraction works without needing real data:

```bash
python test_cwt_6channel_simple.py
```

**Expected output**:
- ✓ Features extracted successfully
- ✓ No NaN/Inf values
- ✓ 6 channels extracted
- ✓ Derived channels (M, S, Intensity, MSC) correctly differentiated

### 2. Extract Full Dataset

#### Option A: Local extraction (slow, GPU optional)
```bash
python extract_cwt_6channel.py --split dev --wavelet morl --n_wavelet_scales 128
```

**Time estimate**: ~30,000 files × 2 seconds = 16+ hours on single CPU

#### Option B: Slurm cluster (recommended)
```bash
sbatch run_cwt_6channel.sbatch
```

- Reserves 32 CPUs, 64 GB RAM
- Sets thread limits to prevent oversubscription
- Time estimate: ~2-4 hours with parallelization

### 3. Test Model with Extracted Features

After extraction completes:

```bash
python test_cwt_6channel.py
```

This validates:
- ✓ Model forward pass works
- ✓ Loss computation works
- ✓ Backward propagation works

### 4. Train with CWT Features

```bash
python main.py \
    --use_wavelet \
    --wavelet morl \
    --n_wavelet_scales 128 \
    --nb_epochs 5 \
    --exp cwt_6channel_baseline
```

## File Organization

After extraction, you'll have:

```
DCASE2025_SELD_dataset/
├── mel128_wavelet_dnorm/                    # Feature directory
│   ├── stereo_dev/                          # Raw extracted features
│   │   └── fold*_room*_mix*.pt              # (6, T, 128) tensors
│   ├── stereo_dev_normalized/               # Normalized for training
│   │   └── fold*_room*_mix*.pt              # (6, T, 128) normalized
│   └── scaler_dev.pkl                       # Fitted StandardScaler
└── metadata_dev_adpit/                      # Labels (already exists)
    └── fold*_room*_*.csv
```

## Parameters

```python
--wavelet morl              # Wavelet type (morl, mexh, gaus1, ...)
--n_wavelet_scales 128      # Number of scales (64, 128, 256, ...)
--split dev                 # Dataset split (dev, eval)
--normalize_only            # Skip extraction, re-normalize only
```

### Recommended Settings

| Use Case | Wavelet | Scales | Notes |
|----------|---------|--------|-------|
| **Quick test** | morl | 64 | Fast extraction, lower resolution |
| **Baseline** | morl | 128 | Standard setting, ~30 mins on 32 CPUs |
| **High resolution** | morl | 256 | Double computation, better frequency detail |
| **Comparison** | mexh | 128 | Mexican Hat wavelet, different shape |

## Implementation Details

### Wavelet Transform (CWT)

```python
# For each channel:
scales = np.arange(1, n_scales + 1)
coef = pywt.cwt(channel, scales, wavelet, sampling_period=1/sr)
mag = |coef|  # (n_scales, n_samples)

# Frame-aggregate by averaging within hop_length windows
framed = np.zeros((n_frames, n_scales))
for t in range(n_frames):
    s, e = t * hop_length, (t+1) * hop_length
    framed[t, :] = mag[:, s:e].mean(axis=1)
```

### Intensity Calculation

```python
# Compute cross-spectrum in wavelet domain
M_coef = pywt.cwt(mid, scales, wavelet, ...)
S_coef = pywt.cwt(side, scales, wavelet, ...)

# Cross-spectrum intensity
I_num = Re(M_coef * conj(S_coef))
I_denom = |M_coef|² + |S_coef|² + ε
I_norm = I_num / I_denom

# Frame-aggregate
```

### MSC (Coherence) Calculation

```python
# Compute auto-spectra
L_coef = pywt.cwt(left, scales, wavelet, ...)
R_coef = pywt.cwt(right, scales, wavelet, ...)

LL_auto = |L_coef|²
RR_auto = |R_coef|²

# Cross-spectrum
LR_cross = L_coef * conj(R_coef)

# MSC = |cross|² / (auto_L * auto_R)
MSC = |LR_cross|² / (LL_auto * RR_auto + ε)

# Frame-aggregate
```

## Normalization

Features are normalized using `sklearn.StandardScaler`:

```python
# 1. Fit on training split (all 30,000 files)
scaler = StandardScaler()
scaler.fit(all_features.reshape(-1, n_scales))

# 2. Save scaler
pickle.dump(scaler, open('scaler_dev.pkl', 'wb'))

# 3. Transform all features
normalized = scaler.transform(features.reshape(-1, n_scales))
normalized = normalized.reshape(6, T, n_scales)
```

**Statistics**:
- Mean per scale: 0.0 (by design)
- Std per scale: 1.0 (by design)

## Performance Tips

### Faster Extraction

1. **Use local GPU if available** (CWT can use GPU via PyWavelets on CUDA)
2. **Parallelize with joblib** (add to extract script):
   ```python
   from joblib import Parallel, delayed
   results = Parallel(n_jobs=32)(
       delayed(extract_file)(f) for f in audio_files
   )
   ```
3. **Use lower scale count first** (64 scales = 2× faster than 128)

### Memory Optimization

- Feature size per file: (6, 80 frames, 128 scales) ≈ 5 MB
- Full dataset cache: 30,000 × 5 MB = 150 GB on disk
- Normalization bottleneck: Fit all at once (requires ~400 MB RAM for feature matrix)

## Troubleshooting

### "No feature files found"
```
ERROR: No feature files found in stereo_dev
```
→ Run extraction step first: `python extract_cwt_6channel.py --split dev`

### "NaN values in features"
- Caused by: Zero energy in some scales + instability
- Solution: Check if audio is mostly silence; add floor to energy denominator (`eps=1e-8`)

### "Scaler not found"
```
ERROR: scaler_dev.pkl not found
```
→ Must run normalization step after extraction

### "Model shape mismatch"
```
RuntimeError: Expected input shape (B, 6, T, F) but got (B, 2, T, F)
```
→ Make sure `use_wavelet=True` in parameters when loading model

## Comparison: CWT vs. Mel Spectrograms

| Aspect | Mel Spectrogram | CWT 6-Channel |
|--------|-----------------|---------------|
| **Computation** | FFT-based (fast) | Wavelet-based (slower) |
| **Channels** | 2 or 4 (L/R ± M/S) | 6 (L/R/M/S + Intensity + MSC) |
| **Binaural features** | Separate extraction | Derived in wavelet domain |
| **Time-frequency trade** | Fixed STFT window | Scale-dependent |
| **Scalability** | Fixed mel bins | Flexible scale count |

## Next Steps

1. ✓ Test extraction on synthetic audio
2. → Run full extraction on dev split (48 hours estimated)
3. → Train baseline model with CWT features
4. → Compare with mel-spectrogram baseline
5. → Explore other wavelets (mexh, gaus1) or scale counts (64, 256)

## References

- **PyWavelets**: https://pywavelets.readthedocs.io/
- **CWT**: https://en.wikipedia.org/wiki/Continuous_wavelet_transform
- **SELD Task**: https://dcase.community/challenge2025/task-sound-event-localization-detection

## Questions?

Refer to:
- `CWT_6CHANNEL_IMPLEMENTATION.md` - Technical details
- `utils.extract_cwt_6channel_features()` - Function documentation
- `test_cwt_6channel_simple.py` - Working example
