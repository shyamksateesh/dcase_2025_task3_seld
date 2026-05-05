# Wavelet Transform Feature Integration

## Overview

This guide explains how to use **Continuous Wavelet Transform (CWT)** based scalogram features as an alternative to Mel spectrograms in the DCASE 2025 Task 3 SELD (Sound Event Localization and Detection) pipeline.

The wavelet branch provides a time-frequency decomposition using wavelets instead of the Fourier transform, which can capture sharper transients and multi-scale temporal patterns in audio—useful for music and complex sound analysis.

---

## Key Features

- **CWT Scalogram Extraction:** Computes magnitude scalograms across a configurable range of wavelet scales per audio channel.
- **Binaural Stereo Support:** Processes left and right channels independently (similar to baseline Mel spectrograms).
- **Hop-Length Frame Alignment:** Aggregates CWT coefficients into frames matching the STFT hop length for temporal consistency.
- **Configurable Wavelet & Scales:** Supports multiple wavelet families (Morlet, Mexican hat, etc.) and scale counts.
- **Limitation:** Binaural STFT-derived features (IPD, gamma/coherence, IV, SALSA-Lite, Mid-Side) are **not yet supported** with wavelet mode.

---

## Installation

### 1. Install PyWavelets

The required dependency `PyWavelets` (pywt) has been added to `requirements.txt`. Install it with:

```bash
pip install -r requirements.txt
```

Or install it directly:

```bash
pip install PyWavelets==1.4.1
```

### 2. Verify Installation

Test that imports work:

```python
import pywt
print(pywt.cwt.__doc__)
```

---

## Usage

### Basic Command Line

To extract wavelet scalogram features and train:

```bash
python main.py \
  --exp my_wavelet_exp \
  --use_wavelet \
  --wavelet morl \
  --n_wavelet_scales 64 \
  --nb_epochs 100 \
  --batch_size 32
```

### Parameters

Add any of these CLI flags when running `main.py`:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_wavelet` | flag | False | Enable wavelet CWT mode (instead of Mel spectrogram). |
| `--wavelet` | str | `'morl'` | Wavelet type for CWT. Examples: `'morl'` (Morlet), `'mexh'` (Mexican hat), `'gaus1'` (Gaussian). |
| `--n_wavelet_scales` | int | None | Number of wavelet scales (if None, defaults to `nb_mels` value). |

### Example Workflows

#### 1. **Baseline Wavelet (64 scales, Morlet)**
```bash
python main.py \
  --exp baseline_wavelet_64 \
  --use_wavelet \
  --n_wavelet_scales 64
```

#### 2. **High-Resolution Wavelet (128 scales, Morlet)**
```bash
python main.py \
  --exp wavelet_128_morl \
  --use_wavelet \
  --wavelet morl \
  --n_wavelet_scales 128 \
  --nb_epochs 150
```

#### 3. **Mexican Hat Wavelet**
```bash
python main.py \
  --exp wavelet_mexican_hat \
  --use_wavelet \
  --wavelet mexh \
  --n_wavelet_scales 80
```

#### 4. **Compare: Wavelet vs Mel Spectrogram**
```bash
# Run with wavelets
python main.py --exp wavelet_run --use_wavelet --n_wavelet_scales 64

# Run with baseline Mel spectrograms
python main.py --exp mel_baseline --nb_mels 64
```

---

## How It Works

### Extraction Pipeline

1. **Load stereo audio** (left & right channels, 2 × N samples).

2. **For each channel independently:**
   - Compute continuous wavelet transform (CWT) using `pywt.cwt()`.
   - Define scales from 1 to `n_wavelet_scales`.
   - Extract magnitude coefficients: `(n_scales, n_samples)`.

3. **Temporal Framing:**
   - Divide scalogram into overlapping windows of size `hop_length`.
   - Average magnitudes within each window → one frame per hop interval.
   - Result: `(T, n_scales)` where T = number of frames.

4. **Stack channels:** `(2, T, n_scales)` → ready for data generator & model.

5. **Normalization:** Applied per-channel using StandardScaler (same as Mel pipeline).

### Feature Shape

| Mode | Output Shape | Notes |
|------|--------------|-------|
| **Mel (default)** | `(2, T, 64)` | Left & Right Mel specs (nb_mels=64) |
| **Wavelet** | `(2, T, 64)` | Left & Right CWT scalograms (n_scales=64) |

---

## Wavelet Types

Common choices with `pywt.continuous_wavelets`:

| Name | Full Name | Use Case |
|------|-----------|----------|
| `'morl'` | Morlet | General-purpose; good for frequency localization |
| `'mexh'` | Mexican Hat (Ricker) | Transient detection; sharp in time domain |
| `'gaus1'`–`'gaus8'` | Gaussian derivatives | Smoother representation; higher order = more oscillatory |
| `'shan'` | Shannon | Narrow frequency bands |

See [PyWavelets wavelets docs](https://pywavelets.readthedocs.io/en/latest/source/wavelets.html) for all available options.

---

## Configuration Files

### `parameters.py`

Wavelet defaults are now included:

```python
'use_wavelet': False,
'wavelet': 'morl',
'n_wavelet_scales': 64,
```

You can override these in the parameters dict programmatically or via CLI.

### Feature Directory Naming

Features are saved with a folder suffix indicating the extraction method:

- **Mel:** `mel64_dnorm/`
- **Wavelet:** `mel64_wavelet_dnorm/`
- **Combined:** `mel64_wavelet_ipd_dnorm/` (if adding binaural features in future)

This ensures different extraction modes do not overwrite each other's feature caches.

---

## Limitations & Future Work

### Current Limitations

1. **No Binaural Features:** IPD, gamma (coherence), IV, SALSA-Lite, and Mid-Side spectrograms are not yet supported with wavelet mode. Enabling these flags will raise `NotImplementedError`.

2. **Scale-to-Frequency Mapping:** Currently uses a simple linear scale progression (1…n_scales). For better frequency correspondence, future versions could use:
   - Logarithmic scale spacing
   - Center frequency mapping (similar to Mel scale)
   - Custom scale-frequency calibration

3. **No Video Features:** Wavelet extraction only handles audio; video feature extraction is unchanged.

### Planned Enhancements

- [ ] Binaural wavelet features (phase-based analogs of IPD, coherence)
- [ ] Log-frequency wavelet scales for perceptually-motivated decomposition
- [ ] Normalization via `scipy.signal.morlet2` for tunable time–frequency resolution
- [ ] Real comparison benchmarks (accuracy, inference speed, convergence)

---

## Troubleshooting

### ImportError: No module named 'pywt'

```bash
pip install PyWavelets
```

### NotImplementedError: Wavelet feature path does not support binaural features

**Error message:** `Wavelet feature path does not support binaural features (ipd/gamma/iv/slite/ms) yet.`

**Cause:** You set `--use_wavelet` **and** one of `--ipd`, `--gamma`, `--iv`, or `--slite`.

**Fix:** Remove the binaural feature flags:

```bash
# ❌ This will fail:
python main.py --use_wavelet --ipd

# ✅ Use only wavelet:
python main.py --use_wavelet
```

### Feature shape mismatch at model input

**Symptom:** Training fails with "input shape mismatch" or "dimension error."

**Cause:** Model expects features from previous run (Mel) but wavelet features have different dimensions.

**Fix:** Clear the feature cache directory before re-running:

```bash
rm -rf DCASE2025_SELD_dataset/mel64_wavelet_dnorm/
python main.py --use_wavelet --n_wavelet_scales 64
```

### Very slow CWT computation

**Symptom:** Feature extraction takes much longer than baseline Mel spectrograms.

**Cause:** High `n_wavelet_scales` or inefficient Morlet wavelet.

**Suggestions:**
1. Reduce scales: `--n_wavelet_scales 48` instead of 128.
2. Try `'mexh'` (Mexican hat) which is faster than Morlet.
3. Profile with: `python -m cProfile -s cumtime main.py --use_wavelet`

---

## Example Results & Benchmarking

To compare wavelet vs. baseline performance on your project:

### 1. Extract & Train with Mel Spectrograms (Baseline)

```bash
python main.py \
  --exp mel_baseline \
  --nb_mels 64 \
  --nb_epochs 50 \
  --batch_size 32
```

Record metrics from `wandb` (F1-score, angular error, etc.).

### 2. Extract & Train with Wavelets

```bash
python main.py \
  --exp wavelet_64_morl \
  --use_wavelet \
  --wavelet morl \
  --n_wavelet_scales 64 \
  --nb_epochs 50 \
  --batch_size 32
```

### 3. Compare Results

Open your Weights & Biases dashboard and compare:
- Training/validation loss curves
- F1-scores and detection metrics
- Convergence speed
- Inference time (if logged)

---

## Integration Notes for Developers

### Code Changes

Modified files:

| File | Change |
|------|--------|
| `requirements.txt` | Added `PyWavelets==1.4.1` |
| `parameters.py` | Added `use_wavelet`, `wavelet`, `n_wavelet_scales` defaults |
| `utils.py` | Implemented `extract_stereo_features()` wavelet branch with CWT + frame aggregation |
| `extract_features.py` | Pass wavelet params to `utils.extract_stereo_features()` |
| `main.py` | Added CLI flags, feature folder naming, param propagation |

### Adding Custom Wavelet Processing

To add custom binaural wavelet features (e.g., wavelet-based phase coherence):

1. Add a new function in `utils.py`, e.g., `compute_wavelet_ipd()`.
2. Call it inside the `use_wavelet` branch of `extract_stereo_features()`.
3. Concatenate output to scalograms.
4. Update the `NotImplementedError` check to allow the new feature.
5. Test and document.

### Performance Considerations

- **Memory:** CWT stores full `(n_scales, n_samples)` before framing; for long audio, consider batch processing.
- **Speed:** Morlet CWT is slower than FFT-based Mel; consider reducing scales or wavelet if extraction is bottlenecked.
- **Model Input:** Model architecture remains unchanged; only feature dimensions may differ.

---

## References & Resources

- **PyWavelets Documentation:** https://pywavelets.readthedocs.io/
- **CWT Tutorial:** https://en.wikipedia.org/wiki/Continuous_wavelet_transform
- **DCASE 2025 Task 3:** https://dcase.community/challenge2025/task-sound-event-localization-and-detection
- **Wavelet Audio Processing:** Torrence & Compo (1998) "A Practical Guide to Wavelet Analysis"

---

## Questions or Issues?

- Check the [Troubleshooting](#troubleshooting) section.
- Review `main.py` for full CLI argument list: `python main.py --help`
- Modify `parameters.py` to set custom defaults.
- Open a GitHub issue with details (error message, command used, environment).

Enjoy experimenting with wavelet features! 🌊

