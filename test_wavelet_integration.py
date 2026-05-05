#!/usr/bin/env python3
"""
test_wavelet_integration.py

Quick smoke-test to verify that wavelet feature extraction works correctly.
This script:
  1. Generates a synthetic stereo audio signal
  2. Extracts wavelet scalogram features using the same pipeline as main.py
  3. Verifies output shapes and dtype
  4. Compares wavelet vs. Mel spectrogram feature shapes

Author: Integration Test
Date: 2025
"""

import numpy as np
import sys
import os

# Add repo to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
import pywt


def test_wavelet_extraction():
    """Test that wavelet CWT extraction works end-to-end."""
    print("=" * 60)
    print("WAVELET INTEGRATION TEST")
    print("=" * 60)

    # 1. Generate synthetic stereo audio (5 seconds at 24 kHz)
    sr = 24000
    duration = 5.0
    n_samples = int(sr * duration)
    
    # Create a simple stereo signal: left=sine wave, right=chirp
    t = np.arange(n_samples) / sr
    left_channel = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    right_channel = 0.3 * np.sin(2 * np.pi * 880 * t)  # 880 Hz tone
    audio = np.stack([left_channel, right_channel], axis=0)  # (2, n_samples)
    
    print(f"\n✓ Generated synthetic stereo audio:")
    print(f"  Shape: {audio.shape}")
    print(f"  Sampling rate: {sr} Hz")
    print(f"  Duration: {duration} s")

    # 2. Test Mel spectrogram (baseline)
    print(f"\n--- MEL SPECTROGRAM (Baseline) ---")
    try:
        mel_features = utils.extract_stereo_features(
            audio, sr=sr,
            n_fft=512, hop_length=300, win_length=512,
            nb_mels=64, max_freq=2000,
            use_gamma=False, use_ipd=False, use_iv=False, use_slite=False, use_ms=False,
            use_wavelet=False  # Explicitly disable wavelet
        )
        print(f"✓ Mel spectrogram extraction succeeded")
        print(f"  Output shape: {mel_features.shape}")
        print(f"  Expected: (2, T, 64) where T = ceil(n_samples / hop_length)")
        expected_t = int(np.ceil(n_samples / 300))
        print(f"  Actual T: {mel_features.shape[1]}, Expected T: {expected_t}")
        assert mel_features.shape == (2, expected_t, 64), f"Shape mismatch: {mel_features.shape}"
        assert mel_features.dtype == np.float32, f"Dtype should be float32, got {mel_features.dtype}"
        print(f"✓ Shape and dtype validation passed")
    except Exception as e:
        print(f"✗ Mel spectrogram extraction failed: {e}")
        return False

    # 3. Test Wavelet scalogram
    print(f"\n--- WAVELET SCALOGRAM (CWT) ---")
    try:
        wavelet_features = utils.extract_stereo_features(
            audio, sr=sr,
            n_fft=512, hop_length=300, win_length=512,
            nb_mels=64,  # Not used in wavelet mode, but passed for compatibility
            max_freq=2000,
            use_gamma=False, use_ipd=False, use_iv=False, use_slite=False, use_ms=False,
            use_wavelet=True,
            wavelet='morl',
            n_wavelet_scales=64
        )
        print(f"✓ Wavelet scalogram extraction succeeded")
        print(f"  Output shape: {wavelet_features.shape}")
        print(f"  Expected: (2, T, 64) where T = ceil(n_samples / hop_length)")
        expected_t = int(np.ceil(n_samples / 300))
        print(f"  Actual T: {wavelet_features.shape[1]}, Expected T: {expected_t}")
        assert wavelet_features.shape == (2, expected_t, 64), f"Shape mismatch: {wavelet_features.shape}"
        assert wavelet_features.dtype == np.float32, f"Dtype should be float32, got {wavelet_features.dtype}"
        print(f"✓ Shape and dtype validation passed")
    except Exception as e:
        print(f"✗ Wavelet scalogram extraction failed: {e}")
        return False

    # 4. Test different wavelet types
    print(f"\n--- TESTING DIFFERENT WAVELETS ---")
    wavelets_to_test = ['morl', 'mexh', 'gaus1']
    for wavelet_type in wavelets_to_test:
        try:
            features = utils.extract_stereo_features(
                audio, sr=sr,
                n_fft=512, hop_length=300, win_length=512,
                nb_mels=64, max_freq=2000,
                use_gamma=False, use_ipd=False, use_iv=False, use_slite=False, use_ms=False,
                use_wavelet=True,
                wavelet=wavelet_type,
                n_wavelet_scales=64
            )
            print(f"✓ {wavelet_type}: shape={features.shape}, dtype={features.dtype}")
        except Exception as e:
            print(f"✗ {wavelet_type} failed: {e}")
            return False

    # 5. Test different scale counts
    print(f"\n--- TESTING DIFFERENT SCALE COUNTS ---")
    scale_counts = [32, 64, 128]
    for n_scales in scale_counts:
        try:
            features = utils.extract_stereo_features(
                audio, sr=sr,
                n_fft=512, hop_length=300, win_length=512,
                nb_mels=64, max_freq=2000,
                use_gamma=False, use_ipd=False, use_iv=False, use_slite=False, use_ms=False,
                use_wavelet=True,
                wavelet='morl',
                n_wavelet_scales=n_scales
            )
            print(f"✓ {n_scales} scales: shape={features.shape}")
            assert features.shape[2] == n_scales, f"Expected {n_scales} scales, got {features.shape[2]}"
        except Exception as e:
            print(f"✗ {n_scales} scales failed: {e}")
            return False

    # 6. Test error handling: wavelet + binaural features should fail
    print(f"\n--- TESTING ERROR HANDLING ---")
    try:
        features = utils.extract_stereo_features(
            audio, sr=sr,
            n_fft=512, hop_length=300, win_length=512,
            nb_mels=64, max_freq=2000,
            use_gamma=True,  # This should cause NotImplementedError
            use_ipd=False, use_iv=False, use_slite=False, use_ms=False,
            use_wavelet=True,
            wavelet='morl',
            n_wavelet_scales=64
        )
        print(f"✗ Should have raised NotImplementedError for wavelet + gamma")
        return False
    except NotImplementedError as e:
        print(f"✓ Correctly raised NotImplementedError: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

    # 7. Verify PyWavelets is available
    print(f"\n--- VERIFYING DEPENDENCIES ---")
    try:
        import pywt
        print(f"✓ PyWavelets {pywt.__version__} is installed")
    except ImportError:
        print(f"✗ PyWavelets not installed. Run: pip install PyWavelets")
        return False

    print(f"\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Run feature extraction: python main.py --use_wavelet --n_wavelet_scales 64")
    print(f"2. Compare with baseline: python main.py --nb_mels 64")
    print(f"3. Check wandb dashboard for metrics comparison")
    return True


if __name__ == '__main__':
    success = test_wavelet_extraction()
    sys.exit(0 if success else 1)
