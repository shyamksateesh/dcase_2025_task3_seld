#!/usr/bin/env python3
"""
extract_cwt_6channel_features.py

Full extraction pipeline for CWT 6-channel features.
This script extracts all 6 wavelet-based channels for the entire dev split,
normalizes them, and saves to disk.

Usage:
    # Test on single sample (fast):
    python test_cwt_6channel_simple.py

    # Full extraction on all 30,000 samples (slow, use Slurm):
    python extract_cwt_6channel_features.py --split dev --use_wavelet --wavelet morl --n_wavelet_scales 128

    # Or via Slurm (recommended):
    sbatch run_cwt_6channel.sbatch
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from rich.progress import Progress

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
import parameters


def main():
    parser = argparse.ArgumentParser(description='Extract CWT 6-channel features')
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'eval'],
                        help='Dataset split to extract')
    parser.add_argument('--use_wavelet', action='store_true', default=True,
                        help='Use wavelet features (default: True)')
    parser.add_argument('--wavelet', type=str, default='morl',
                        help='Wavelet type: morl, mexh, gaus1, etc.')
    parser.add_argument('--n_wavelet_scales', type=int, default=128,
                        help='Number of wavelet scales')
    parser.add_argument('--normalize_only', action='store_true', default=False,
                        help='Skip extraction; only normalize already-extracted features')
    args = parser.parse_args()

    params = parameters.params.copy()
    params['use_wavelet'] = args.use_wavelet
    params['wavelet'] = args.wavelet
    params['n_wavelet_scales'] = args.n_wavelet_scales
    params['nb_mels'] = args.n_wavelet_scales  # Match scales to mels for consistency

    # Build feature directory
    feat_folder = f"mel{params['n_wavelet_scales']}_wavelet_dnorm"
    feat_dir = os.path.join(params['root_dir'], feat_folder)
    os.makedirs(feat_dir, exist_ok=True)

    sr = params['sampling_rate']
    hop_length = int(params['hop_length_s'] * sr)

    print("="*70)
    print("CWT 6-CHANNEL FEATURE EXTRACTION")
    print("="*70)
    print(f"Split: {args.split}")
    print(f"Wavelet: {args.wavelet}, Scales: {args.n_wavelet_scales}")
    print(f"Feature dir: {feat_dir}")
    print(f"Normalize-only: {args.normalize_only}")
    print("="*70)

    # ========== STEP 1: Extract features ==========
    if not args.normalize_only:
        print("\nStep 1: Extracting CWT 6-channel features from audio...")

        # Find audio files
        if args.split == 'dev':
            audio_glob = os.path.join(params['root_dir'], 'stereo_dev', 'dev-*', '*.wav')
        elif args.split == 'eval':
            audio_glob = os.path.join(params['root_dir'], 'stereo_eval', 'eval', '*.wav')

        audio_files = sorted(glob.glob(audio_glob))
        print(f"Found {len(audio_files)} audio files")

        if not audio_files:
            print(f"ERROR: No audio files found for {args.split}")
            return False

        # Extract features for each file
        stereo_feat_dir = os.path.join(feat_dir, f'stereo_{args.split}')
        os.makedirs(stereo_feat_dir, exist_ok=True)

        with Progress() as progress:
            task = progress.add_task("[cyan]Extracting...", total=len(audio_files))

            for audio_file in audio_files:
                try:
                    # Load audio
                    audio, _ = utils.load_audio(audio_file, sr)

                    # Extract CWT 6-channel features
                    features = utils.extract_cwt_6channel_features(
                        audio=audio,
                        sr=sr,
                        hop_length=hop_length,
                        wavelet=args.wavelet,
                        n_wavelet_scales=args.n_wavelet_scales
                    )  # Shape: (6, T, scales)

                    # Save features
                    basename = Path(audio_file).stem
                    feat_path = os.path.join(stereo_feat_dir, f'{basename}.pt')
                    torch.save(torch.from_numpy(features).float(), feat_path)

                except Exception as e:
                    print(f"\nERROR processing {audio_file}: {e}")
                    progress.update(task, advance=1)
                    continue

                progress.update(task, advance=1)

        print(f"✓ Features extracted to {stereo_feat_dir}")

    else:
        print("\nStep 1: Skipping feature extraction (normalize-only mode)...")

    # ========== STEP 2: Normalize features ==========
    print("\nStep 2: Normalizing features...")

    stereo_feat_dir = os.path.join(feat_dir, f'stereo_{args.split}')
    feature_files = sorted(glob.glob(os.path.join(stereo_feat_dir, '*.pt')))

    if not feature_files:
        print(f"ERROR: No feature files found in {stereo_feat_dir}")
        return False

    print(f"Normalizing {len(feature_files)} features...")

    # Fit scaler on all features
    scaler = StandardScaler()
    all_features_flat = []

    with Progress() as progress:
        task = progress.add_task("[cyan]Reading features...", total=len(feature_files))

        for feat_file in feature_files:
            feat = torch.load(feat_file, weights_only=True).numpy()  # (6, T, scales)
            # Flatten: (6*T, scales)
            feat_flat = feat.reshape(-1, feat.shape[-1])
            all_features_flat.append(feat_flat)
            progress.update(task, advance=1)

    all_features_flat = np.concatenate(all_features_flat, axis=0)
    print(f"Fitting scaler on {all_features_flat.shape[0]} feature vectors...")
    scaler.fit(all_features_flat)

    # Save scaler
    scaler_path = os.path.join(feat_dir, f'scaler_{args.split}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to {scaler_path}")

    # Normalize and save
    normalized_feat_dir = os.path.join(feat_dir, f'stereo_{args.split}_normalized')
    os.makedirs(normalized_feat_dir, exist_ok=True)

    with Progress() as progress:
        task = progress.add_task("[cyan]Normalizing...", total=len(feature_files))

        for feat_file in feature_files:
            feat = torch.load(feat_file, weights_only=True).numpy()  # (6, T, scales)
            original_shape = feat.shape
            feat_flat = feat.reshape(-1, feat.shape[-1])
            feat_norm = scaler.transform(feat_flat)
            feat_norm = feat_norm.reshape(original_shape)

            basename = Path(feat_file).stem
            norm_path = os.path.join(normalized_feat_dir, f'{basename}.pt')
            torch.save(torch.from_numpy(feat_norm).float(), norm_path)

            progress.update(task, advance=1)

    print(f"✓ Normalized features saved to {normalized_feat_dir}")

    print("\n" + "="*70)
    print("CWT 6-CHANNEL EXTRACTION COMPLETE")
    print("="*70)
    print(f"Feature dir: {feat_dir}")
    print(f"Normalized dir: {normalized_feat_dir}")
    print("="*70)

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
