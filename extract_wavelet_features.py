#!/usr/bin/env python3
"""
Standalone feature extraction script.
Run this to pre-extract and normalize wavelet/mel features before training.

Usage:
    python extract_wavelet_features.py --wavelet morl --n_wavelet_scales 128
    python extract_mel_features.py  # for baseline Mel spectrogram
"""

import os
import argparse
from extract_features import SELDFeatureExtractor
from parameters import params as default_params

def main():
    parser = argparse.ArgumentParser(description='Extract audio features for SELD task')
    parser.add_argument('--use_wavelet', action='store_true', default=False, help='Use wavelet features')
    parser.add_argument('--wavelet', type=str, default='morl', help='Wavelet type (morl, mexh, gaus1, etc.)')
    parser.add_argument('--n_wavelet_scales', type=int, default=128, help='Number of wavelet scales')
    parser.add_argument('--nb_mels', type=int, default=128, help='Number of mel bins (if not using wavelets)')
    parser.add_argument('--split', type=str, default='dev', help='Data split: dev, eval')
    parser.add_argument('--normalize_only', action='store_true', default=False,
                        help='Skip feature extraction and labels; only normalize already extracted features')
    parser.add_argument('--num_shards', type=int, default=1, help='Total number of parallel extraction shards')
    parser.add_argument('--shard_id', type=int, default=0, help='Shard id for this job (0-based)')
    parser.add_argument('--only_extract', action='store_true', default=False,
                        help='Only extract features (useful for running multiple parallel extract jobs). Run normalization separately after all shards complete.')
    
    args = parser.parse_args()
    
    # Get base parameters (make a copy so we don't mutate the module-level dict)
    params = default_params.copy()

    # Ensure optional feature flags exist (main.py argparse normally sets these).
    params.setdefault('gamma', False)
    params.setdefault('ipd', False)
    params.setdefault('iv', False)
    params.setdefault('slite', False)
    params.setdefault('ms', False)
    params.setdefault('cutout', False)
    params.setdefault('freqshift', False)
    params.setdefault('filtaug', False)
    params.setdefault('compfreq', False)
    params.setdefault('alltrans', False)
    params.setdefault('itfm', False)
    
    # Update with CLI args
    params['use_wavelet'] = args.use_wavelet
    if args.use_wavelet:
        params['wavelet'] = args.wavelet
        params['n_wavelet_scales'] = args.n_wavelet_scales
        params['nb_mels'] = args.n_wavelet_scales  # Align feature dimension
    else:
        params['nb_mels'] = args.nb_mels
    
    # Set feature folder
    feat_folder = (f"mel{params['nb_mels']}" if params['nb_mels'] else "linspec") + \
                  ("_wavelet" if params.get('use_wavelet', False) else "") + \
                  ("_dnorm" if params['dnorm'] else "")
    params['feat_dir'] = os.path.join(params['root_dir'], feat_folder)
    
    print(f"\n{'='*60}")
    print(f"Feature Extraction Configuration")
    print(f"{'='*60}")
    print(f"Wavelet mode: {params['use_wavelet']}")
    if params['use_wavelet']:
        print(f"  Wavelet type: {params['wavelet']}")
        print(f"  Scales: {params['n_wavelet_scales']}")
    else:
        print(f"  Mel bins: {params['nb_mels']}")
    print(f"Data split: {args.split}")
    print(f"Feature directory: {params['feat_dir']}")
    print(f"{'='*60}\n")
    
    # Initialize extractor
    feature_extractor = SELDFeatureExtractor(params)
    
    # Extract features, labels, and normalize
    if args.normalize_only:
        print("Step 1: Skipping feature extraction (normalize-only mode)...")
        print("Step 2: Skipping label extraction (normalize-only mode)...")
        print("Step 3: Preprocessing (normalizing) features...")
        feature_extractor.preprocess_features(split=args.split)
    else:
        print("Step 1: Extracting audio features...")
        feature_extractor.extract_features(split=args.split, num_shards=args.num_shards, shard_id=args.shard_id)

        if args.only_extract or args.num_shards > 1:
            print("\nNotice: You ran extraction in shard/only-extract mode.\nTo finish the pipeline, run the labels extraction and normalization once all shards have completed:\n")
            print("  # Extract labels (single job or multiple, skipped duplicates will be ignored):")
            print("  python extract_wavelet_features.py --split {} --normalize_only".format(args.split))
            print("\nOr run normalization only after all feature shards are done:")
            print("  python extract_wavelet_features.py --split {} --normalize_only".format(args.split))
        else:
            print("\nStep 2: Extracting labels...")
            feature_extractor.extract_labels(split=args.split)

            print("\nStep 3: Preprocessing (normalizing) features...")
            feature_extractor.preprocess_features(split=args.split)
    
    print(f"\n{'='*60}")
    print(f"✓ Feature extraction complete!")
    print(f"Features saved to: {params['feat_dir']}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
