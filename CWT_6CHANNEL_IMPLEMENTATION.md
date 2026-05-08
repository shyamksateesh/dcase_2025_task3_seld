# CWT 6-Channel Feature Extraction Implementation

## Overview
Replaced all 6 mel-spectrogram channels with CWT (Continuous Wavelet Transform) based representations. This enables a purely wavelet-domain feature pipeline for SELD task.

## Implementation Details

### New Function: `extract_cwt_6channel_features()`
**Location**: `utils.py` (lines ~250-350)

**Purpose**: Extract all 6 channels using CWT only (no STFT/mel required)

**Channel Layout**:
```
Channel 1: Left channel CWT scalogram         (T, scales)
Channel 2: Right channel CWT scalogram        (T, scales)
Channel 3: Mid = (L+R)/2 CWT scalogram       (T, scales)
Channel 4: Side = (L-R)/2 CWT scalogram      (T, scales)
Channel 5: CWT-based Intensity Vector        (T, scales)
           = Re(M·S*) / (|M|² + |S|² + ε)
Channel 6: CWT-based MSC (Magnitude-Squared Coherence) (T, scales)
           = |Φ_LR|² / (Φ_LL·Φ_RR + ε)
```

**Output Shape**: `(6, T, n_scales)`
- Replaces mel-spectrogram channels entirely
- Same temporal resolution as STFT via hop-length frame aggregation
- Frequency resolution = number of wavelet scales

### Algorithm Details

#### Channels 1-4: Direct Scalograms
```python
coef = pywt.cwt(channel, scales, wavelet, sampling_period=1/sr)
mag = |coef|  # (n_scales, n_samples)
# Frame-aggregate by averaging within hop_length windows
```

#### Channel 5: Intensity Vector (Wavelet Domain)
```python
M_coef = pywt.cwt(M_signal, scales, wavelet, ...)
S_coef = pywt.cwt(S_signal, scales, wavelet, ...)

I_num = Re(M_coef * conj(S_coef))      # Cross-spectrum
I_denom = |M_coef|² + |S_coef|² + ε   # Energy
I_norm = I_num / I_denom  # Per scale, per time
# Frame-aggregate
```

#### Channel 6: MSC (Wavelet Domain)
```python
L_coef = pywt.cwt(L_signal, scales, wavelet, ...)
R_coef = pywt.cwt(R_signal, scales, wavelet, ...)

LR_cross = L_coef * conj(R_coef)  # Cross-spectrum
LL_auto = |L_coef|²               # Left auto-spectrum
RR_auto = |R_coef|²               # Right auto-spectrum

MSC = |LR_cross|² / (LL_auto * RR_auto + ε)
# Frame-aggregate
```

### Advantages Over Mixed STFT/Wavelet

1. **Consistency**: All features in wavelet domain (no STFT/mel dependency)
2. **Scalability**: Easy to adjust number of scales (128, 256, etc.) vs. fixed mel bins
3. **Biological relevance**: Wavelets mimic auditory filterbank better than mel
4. **No channel conflicts**: All 6 outputs derived from wavelet representations
5. **Unified parameterization**: Single wavelet type and scale set for all channels

## Test Scripts

### 1. `test_cwt_6channel_simple.py`
- **Purpose**: Validate 6-channel extraction on synthetic audio
- **Input**: Generated stereo sine waves (440 Hz, 550 Hz)
- **Checks**:
  - Shape validation: (6, T, scales)
  - NaN/Inf detection
  - Per-channel statistics
  - Derived channel distinctness (M ≠ L, S ≠ L)
- **Run**: `python test_cwt_6channel_simple.py`

### 2. `test_cwt_6channel.py`
- **Purpose**: End-to-end test on cached normalized wavelets
- **Input**: Load pre-normalized 6-channel wavelets from cache
- **Checks**:
  - Model forward pass
  - ADPIT loss computation
  - Backward propagation
- **Run**: `python test_cwt_6channel.py`
  - **Prerequisites**: Must run `extract_wavelet_features.py --normalize_only` first
  - **Expected feature dir**: `DCASE2025_SELD_dataset/mel128_wavelet_dnorm/stereo_dev_normalized/`

## Integration with Existing Pipeline

### Feature Extraction
```python
# In extract_features.py, when use_wavelet=True with 6-channel mode:
features = utils.extract_cwt_6channel_features(
    audio=audio,
    sr=sr,
    hop_length=hop_length,
    wavelet=params['wavelet'],
    n_wavelet_scales=params['n_wavelet_scales']
)
# Returns (6, T, scales) tensor → normalize and save
```

### Normalization
```python
# Apply StandardScaler per-scale across all samples
scaler = StandardScaler()
features_flat = features.reshape(-1, n_scales)  # Flatten T and channels
features_norm = scaler.fit_transform(features_flat)
features_norm = features_norm.reshape(6, T, n_scales)
# Save to stereo_dev_normalized/
```

### Model Input
```python
# Model expects (B, C, T, F)
batch = features_norm.unsqueeze(0)  # (1, 6, T, scales)
predictions = model(batch)          # (1, 50, 117) for ADPIT
```

## Key Differences from Previous 2-Channel Wavelet Mode

| Aspect | Old (2-channel) | New (6-channel) |
|--------|-----------------|-----------------|
| **Channels** | L, R scalograms only | L, R, M, S, Intensity, MSC |
| **M/S derivation** | STFT-based | Wavelet-based |
| **Intensity (IV)** | Not supported | Derived from wavelet cross-spectrum |
| **MSC/Coherence** | Not supported | Derived from wavelet auto/cross-spectra |
| **Feature dimension** | (2, T, scales) | (6, T, scales) |
| **Independence** | Partial (needs STFT for derived features) | Complete (all wavelet-derived) |

## Next Steps

1. **Test on real data**:
   ```bash
   python extract_wavelet_features.py --use_wavelet --wavelet morl --n_wavelet_scales 128 --normalize_only
   python test_cwt_6channel.py
   ```

2. **Train full model**:
   ```bash
   python main.py --use_wavelet --wavelet morl --n_wavelet_scales 128 --nb_epochs 5
   ```

3. **Compare with baseline**: Evaluate CWT 6-channel vs. mel-spectrogram baseline

## Parameters

- **Wavelet type**: 'morl' (Morlet) - default; also supports 'mexh', 'gaus1', etc.
- **Scales**: 128 typical (matches 128 mel bins for direct comparison)
- **Hop length**: 300 samples (0.0125 s @ 24 kHz) - matches STFT hop
- **Normalization**: StandardScaler per-scale (fitted on training split)

## Performance Considerations

- **Computation**: CWT is ~2-3× slower than FFT, but parallelizable
- **Memory**: (6, T, 128) ≈ same as (2, T, 96) mel with extra 4 channels
- **GPU**: Recommend 32 GB RAM for batch processing with joblib parallelization
