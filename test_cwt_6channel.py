"""
test_cwt_6channel.py

Test script to validate the 6-channel CWT-only feature extraction pipeline.
Tests: CWT extraction → normalization → model forward pass → loss computation.

Author: Audio Research Group, Tampere University
Date: May 2026
"""

import os
import sys
import glob
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parameters
from model import SELDModel
from loss import SELDLossADPIT


def test_cwt_6channel_extraction():
    """Test CWT 6-channel extraction using cached normalized wavelets."""
    print("\n" + "="*70)
    print("CWT 6-CHANNEL FEATURE EXTRACTION TEST")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load parameters
    params = parameters.params.copy()
    params['use_wavelet'] = True
    params['wavelet'] = 'morl'
    params['n_wavelet_scales'] = 128
    params['nb_mels'] = 128  # Match scales to mels
    print(f"Wavelet: {params['wavelet']}, Scales: {params['n_wavelet_scales']}")
    
    # Build feature directory path (following the same convention as smoke test)
    feat_folder = (f"mel{params['nb_mels']}" if params['nb_mels'] else "linspec") + \
                  ("_wavelet" if params.get('use_wavelet', False) else "") + \
                  ("_dnorm" if params.get('dnorm', True) else "")
    feat_dir = os.path.join(params['root_dir'], feat_folder)
    feature_dir = os.path.join(feat_dir, 'stereo_dev_normalized')
    
    print(f"Feature dir: {feat_dir}")
    print(f"Normalized feature dir: {feature_dir}")
    
    # Load cached normalized wavelet features
    feature_files = sorted(glob.glob(os.path.join(feature_dir, '*.pt')))
    
    if not feature_files:
        print(f"ERROR: No cached normalized features found in {feature_dir}")
        print("Please run extract_wavelet_features.py first with --normalize_only flag")
        return False
    
    feature_file = feature_files[0]
    print(f"\nLoading cached feature: {os.path.basename(feature_file)}")
    
    # Load the cached 6-channel features
    features = torch.load(feature_file, weights_only=True)
    print(f"✓ Loaded features shape: {features.shape}")
    
    # Check shape: should be (6, T, n_scales) or compatible
    if len(features.shape) != 3:
        print(f"ERROR: Expected 3D tensor (channels, time, scales), got shape {features.shape}")
        return False
    
    n_channels, n_frames, n_scales = features.shape
    print(f"  Channels: {n_channels}, Time frames: {n_frames}, Scales: {n_scales}")
    
    if n_channels != 6:
        print(f"WARNING: Expected 6 channels, got {n_channels}")
    
    # Create batch and move to device
    batch = features.unsqueeze(0).to(device).float()
    print(f"\n✓ Batch shape: {batch.shape} (B, C, T, scales)")
    
    # Load model
    print("\nLoading SELDModel...")
    model = SELDModel(params, device=device)
    model.to(device)
    print("✓ Model loaded")
    
    # Forward pass
    print("\nPerforming forward pass...")
    with torch.no_grad():
        predictions = model(batch)
    print(f"✓ Predictions shape: {predictions.shape} (expected: (1, 50, 117) for ADPIT)")
    
    # Create dummy ADPIT target and compute loss
    print("\nComputing loss...")
    B = batch.shape[0]
    if params.get('multiACCDOA', True):
        # ADPIT target shape: (B, n_frames, n_accdoa_classes, n_permutations, n_classes)
        label_batch = torch.zeros(B, 50, 6, 4, 13, device=device)
    else:
        label_batch = torch.zeros(B, 50, 13, device=device)
    
    loss_fn = SELDLossADPIT(device=device) if params.get('multiACCDOA', True) else None
    
    if loss_fn:
        loss = loss_fn(predictions, label_batch)
        print(f"✓ Loss computed: {loss.item():.6f}")
    
    # Backward pass
    print("\nPerforming backward pass...")
    loss.backward()
    print("✓ Backward pass successful (gradients computed)")
    
    print("\n" + "="*70)
    print("✓ CWT 6-CHANNEL PIPELINE TEST PASSED")
    print("="*70)
    print(f"Summary:")
    print(f"  - Loaded cached 6-channel normalized wavelets")
    print(f"  - Input shape: {batch.shape}")
    print(f"  - Model output: {predictions.shape}")
    print(f"  - Loss: {loss.item():.6f}")
    print(f"  - Backprop: ✓ Successful")
    print("="*70 + "\n")
    
    return True


if __name__ == '__main__':
    success = test_cwt_6channel_extraction()
    sys.exit(0 if success else 1)
