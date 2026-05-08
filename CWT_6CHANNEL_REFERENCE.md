# CWT 6-Channel Feature Pipeline - Reference Card

## 30-Second Overview

**Replace all 6 mel-spectrogram channels with wavelet-derived features:**

```
Stereo Audio (24 kHz, 2 channels)
    ↓
Compute CWT scalograms for: L, R, M=(L+R)/2, S=(L-R)/2
    ↓
Compute intensity: Re(M·S*) / (|M|² + |S|² + ε)
    ↓
Compute coherence:  |L·R*|² / (|L|²·|R|² + ε)
    ↓
Stack 6 channels → (6, T, 128) tensor
    ↓
Normalize with StandardScaler (per-scale)
    ↓
Train SELD model
```

## Usage Commands

### Test (2 minutes)
```bash
python test_cwt_6channel_simple.py
```

### Extract Dataset (2-4 hours on 32 CPUs)
```bash
sbatch run_cwt_6channel.sbatch
```

### Train Model (1-24 hours depending on dataset size)
```bash
python main.py --use_wavelet --wavelet morl --n_wavelet_scales 128 --nb_epochs 5
```

## Channel Breakdown

| Ch | Name | Input | Output | Range | Purpose |
|----|------|-------|--------|-------|---------|
| 1 | L-Scalogram | Left audio | \|CWT(L)\| | [0, ∞) | Left raw spectrum |
| 2 | R-Scalogram | Right audio | \|CWT(R)\| | [0, ∞) | Right raw spectrum |
| 3 | M-Scalogram | (L+R)/2 | \|CWT(M)\| | [0, ∞) | Mid (sum) spectrum |
| 4 | S-Scalogram | (L-R)/2 | \|CWT(S)\| | [0, ∞) | Side (diff) spectrum |
| 5 | Intensity | M, S | Re(M·S*)/E | [-0.5, 0.5] | M-S cross-correlation |
| 6 | MSC | L, R | \|L·R*\|²/E | [0, 1] | L-R coherence |

**Key**: All computed in **wavelet domain** (no STFT required)

## Feature Tensor Shapes

```
Input Audio:     (2, n_samples)  ← Stereo, 1-30 seconds
                 ↓
CWT Scalogram:   (2, n_scales, n_samples)  ← Per channel
                 ↓
Frame Aggregation: (2, T, n_scales)  ← T = n_samples / hop_length
                 ↓
Stack 6 channels: (6, T, n_scales)  ← Output from extract_cwt_6channel_features()
                 ↓
Normalize:       (6, T, n_scales)  ← Mean=0, Std=1 per scale
                 ↓
To Model:        (B, 6, T, n_scales)  ← Batch of features
```

## Implementation Recipes

### Extract One File
```python
import utils
audio, sr = utils.load_audio('audio.wav', sr=24000)
features = utils.extract_cwt_6channel_features(
    audio=audio, sr=sr, hop_length=300,
    wavelet='morl', n_wavelet_scales=128
)  # → (6, T, 128)
```

### Normalize Features
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_norm = scaler.fit_transform(
    features.reshape(-1, 128)
).reshape(6, T, 128)
```

### Train with CWT
```bash
# parameters.py setup (or command-line override)
params['use_wavelet'] = True
params['wavelet'] = 'morl'
params['n_wavelet_scales'] = 128

# Then run main.py normally
python main.py --use_wavelet ...
```

## Comparison Matrix

```
                  | 2-Channel Mel | 6-Channel CWT (New) | Advantage
------------------+---------------+--------------------+----------
Channels          | 2-4           | 6                  | More info
Binaural features | STFT-based    | Wavelet-based      | Unified
Time resolution   | Fixed (STFT)  | Flexible (scale)   | Adaptive
Extraction speed  | Fast (FFT)    | Slow (Wavelet)     | Trade-off
Setup complexity  | Simple        | Simple (integrated)| Same
Training diff     | None          | Retrain from scratch| Expected
```

## Performance Rules of Thumb

| Task | Time | Memory | Notes |
|------|------|--------|-------|
| Extract 1 file | 2 sec | 5 MB | On CPU |
| Normalize 1 file | <0.1 sec | Same | StandardScaler |
| Fit scaler (30K files) | 5 min | 400 MB | All features in RAM |
| Extract full dataset | 16 hrs (CPU) or 2-4 hrs (32 CPUs) | 150 GB | Disk I/O bound |
| Train 1 epoch (SELD) | 2-4 hrs | 24 GB | GPU recommended |

## Default Parameters

```python
sampling_rate = 24000          # Hz
hop_length_s = 0.0125          # seconds = 300 samples
wavelet = 'morl'               # Morlet (default)
n_wavelet_scales = 128         # Recommended (64-256 typical)
```

## File Map

```
Core:
  utils.py                           ← extract_cwt_6channel_features()
  extract_cwt_6channel.py            ← Full pipeline (extract + normalize)

Jobs:
  run_cwt_6channel.sbatch            ← Slurm submission

Tests:
  test_cwt_6channel_simple.py        ← Synthetic audio (fast)
  test_cwt_6channel.py               ← Real data + model (full)

Docs:
  CWT_6CHANNEL_QUICKSTART.md         ← User guide
  CWT_6CHANNEL_IMPLEMENTATION.md     ← Technical details
  CWT_6CHANNEL_STATUS.md             ← Implementation summary
```

## Debugging Quick Links

| Problem | Cause | Solution |
|---------|-------|----------|
| `AttributeError: module 'utils' has no attribute 'extract_cwt_6channel_features'` | Old utils.py | Pull latest version |
| `No feature files found in stereo_dev` | Extraction not run | Run: `sbatch run_cwt_6channel.sbatch` |
| `NaN values in features` | Numerical instability | Check audio is not all-zeros; ε=1e-8 default |
| `Model shape mismatch` | use_wavelet=False | Set `use_wavelet=True` in parameters |
| `scaler_dev.pkl not found` | Normalization skipped | Run: `python extract_cwt_6channel.py` |

## Key Equations

**Left/Right/Mid/Side Scalograms** (Channels 1-4):
```
S_ch(scale, frame) = mean(|CWT(channel)[scale, frame*hop : (frame+1)*hop]|)
```

**Intensity Vector** (Channel 5):
```
I(scale, frame) = Re(M(scale,frame) × conj(S(scale,frame))) 
                  ─────────────────────────────────────────
                  |M(scale,frame)|² + |S(scale,frame)|² + ε
```

**Magnitude-Squared Coherence** (Channel 6):
```
MSC(scale,frame) = |L(scale,frame) × conj(R(scale,frame))|²
                   ──────────────────────────────────────
                   |L(scale,frame)|² × |R(scale,frame)|² + ε
```

## Common Mistakes ❌ vs. Best Practices ✅

| ❌ Wrong | ✅ Right |
|----------|----------|
| Use STFT for I and MSC | Use wavelet cross-spectra (integrated) |
| Normalize globally | Normalize per-scale with StandardScaler |
| Forget to fit scaler on training data | Always fit on full training split |
| Run extraction without GPU | Use 32 CPUs or GPU for parallelization |
| Retrain from random init | Yes, features are new → retrain from scratch |
| Mix wavelet + mel channels | Pick one: all wavelet or all mel |

## Next Actions

1. ✅ Test: `python test_cwt_6channel_simple.py`
2. → Extract: `sbatch run_cwt_6channel.sbatch` (or local if GPU available)
3. → Train: `python main.py --use_wavelet --wavelet morl ...`
4. → Validate: Compare test/val metrics with mel-spectrogram baseline

---

**Quick Links**:
- 📚 Full Docs: `CWT_6CHANNEL_QUICKSTART.md`
- 🔧 Tech Details: `CWT_6CHANNEL_IMPLEMENTATION.md`
- 📊 Status: `CWT_6CHANNEL_STATUS.md`
- 💻 Code: `utils.extract_cwt_6channel_features()`
