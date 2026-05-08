#!/usr/bin/env python3
"""
test_cwt_6channel_simple.py

Simple test of the extract_cwt_6channel_features function on raw audio.
This validates that the 6-channel CWT extraction produces correct shapes
and that values are sensible (not NaN/inf).
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
import parameters


def test_extract_cwt_6channel():
    """Test CWT 6-channel extraction on synthetic audio."""
    print("\n" + "="*70)
    print("CWT 6-CHANNEL EXTRACTION - SYNTHETIC AUDIO TEST")
    print("="*70)
    
    params = parameters.params.copy()
    sr = params['sampling_rate']
    hop_length = int(params['hop_length_s'] * sr)
    wavelet = 'morl'
    n_scales = 128
    
    print(f"\nParameters:")
    print(f"  Sample Rate: {sr} Hz")
    print(f"  Hop Length: {hop_length} samples")
    print(f"  Wavelet: {wavelet}")
    print(f"  Scales: {n_scales}")
    
    # Generate synthetic stereo audio (1 second = 24000 samples)
    duration_sec = 1.0
    n_samples = int(duration_sec * sr)
    t = np.arange(n_samples) / sr
    
    # Create stereo audio: left = sine wave, right = slightly different sine
    freq_L = 440  # A4
    freq_R = 550  # C#5
    audio_L = np.sin(2 * np.pi * freq_L * t).astype(np.float32)
    audio_R = np.sin(2 * np.pi * freq_R * t).astype(np.float32)
    audio = np.stack([audio_L, audio_R], axis=0)
    
    print(f"\nAudio:")
    print(f"  Shape: {audio.shape}")
    print(f"  Left channel: sine {freq_L} Hz")
    print(f"  Right channel: sine {freq_R} Hz")
    
    # Extract CWT 6-channel features
    print(f"\nExtracting CWT 6-channel features...")
    try:
        features = utils.extract_cwt_6channel_features(
            audio=audio,
            sr=sr,
            hop_length=hop_length,
            wavelet=wavelet,
            n_wavelet_scales=n_scales
        )
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"[OK] Features extracted successfully")
    print(f"  Shape: {features.shape}")
    
    # Validate shape
    expected_shape = (6, None, n_scales)  # (channels, time_frames, scales)
    if features.shape[0] != 6:
        print(f"ERROR: Expected 6 channels, got {features.shape[0]}")
        return False
    if features.shape[2] != n_scales:
        print(f"ERROR: Expected {n_scales} scales, got {features.shape[2]}")
        return False
    
    print(f"  [OK] Shape validation passed (6 channels, {features.shape[1]} frames, {n_scales} scales)")
    
    # Validate values
    if np.any(np.isnan(features)):
        print(f"ERROR: Features contain NaN values")
        return False
    if np.any(np.isinf(features)):
        print(f"ERROR: Features contain Inf values")
        return False
    
    print(f"[OK] No NaN/Inf values detected")
    
    # Print statistics per channel
    print(f"\nChannel statistics:")
    channel_names = ['L-Scalogram', 'R-Scalogram', 'M-Scalogram', 'S-Scalogram', 'Intensity', 'MSC']
    for ch_idx, ch_name in enumerate(channel_names):
        ch_data = features[ch_idx]
        print(f"  {ch_idx+1}. {ch_name:18s} | "
              f"Min: {ch_data.min():8.4f} | Max: {ch_data.max():8.4f} | "
              f"Mean: {ch_data.mean():8.4f} | Std: {ch_data.std():8.4f}")
    
    # Verify that Mid/Side are derived correctly
    print(f"\nValidation:")
    L_data = features[0]
    R_data = features[1]
    M_data = features[2]
    S_data = features[3]
    
    # Mid and Side should be different from L and R
    if np.allclose(L_data, M_data):
        print(f"  WARNING: Mid (M) is too similar to Left (L)")
    else:
        print(f"  [OK] Mid (M) differs from Left (L)")
    
    if np.allclose(L_data, S_data):
        print(f"  WARNING: Side (S) is too similar to Left (L)")
    else:
        print(f"  [OK] Side (S) differs from Left (L)")
    
    print("\n" + "="*70)
    print("[OK] CWT 6-CHANNEL EXTRACTION TEST PASSED")
    print("="*70 + "\n")
    
    return True


if __name__ == '__main__':
    success = test_extract_cwt_6channel()
    sys.exit(0 if success else 1)
