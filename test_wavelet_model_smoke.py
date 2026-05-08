#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
test_wavelet_model_smoke.py

Quick smoke test that runs one cached normalized wavelet sample through the
SELD model and performs a single forward/backward/optimizer step.

Usage:
    python test_wavelet_model_smoke.py --use_wavelet --wavelet morl --n_wavelet_scales 128
"""

import argparse
import os
import glob
import sys

import torch

from data_generator import DataGenerator
from loss import SELDLossADPIT, SELDLossSingleACCDOA
from model import SELDModel
from parameters import params as default_params


def build_feature_dir(params):
    feat_folder = (f"mel{params['nb_mels']}" if params['nb_mels'] else "linspec") + \
                  ("_wavelet" if params.get('use_wavelet', False) else "") + \
                  ("_gamma" if params['gamma'] else "") + \
                  ("_ipd" if params['ipd'] else "") + \
                  ("_iv" if params['iv'] else "") + \
                  ("_slite" if params['slite'] else "") + \
                  ("_ms" if params['ms'] else "") + \
                  ("_dnorm" if params['dnorm'] else "")
    return os.path.join(params['root_dir'], feat_folder)


def main():
    parser = argparse.ArgumentParser(description='One-sample SELD model smoke test')
    parser.add_argument('--use_wavelet', action='store_true', default=False, help='Use wavelet features')
    parser.add_argument('--wavelet', type=str, default='morl', help='Wavelet type')
    parser.add_argument('--n_wavelet_scales', type=int, default=128, help='Number of wavelet scales')
    parser.add_argument('--sample_index', type=int, default=0, help='Which dataset sample to test')
    parser.add_argument('--device', type=str, default=None, help='Override device (cuda/cpu)')
    args = parser.parse_args()

    params = default_params.copy()
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

    params['use_wavelet'] = args.use_wavelet
    if args.use_wavelet:
        params['wavelet'] = args.wavelet
        params['n_wavelet_scales'] = args.n_wavelet_scales
        params['nb_mels'] = args.n_wavelet_scales

    params['feat_dir'] = build_feature_dir(params)

    feature_dir = os.path.join(params['feat_dir'], 'stereo_dev_normalized')
    sample_index = max(args.sample_index, 0)

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print('=' * 60)
    print('WAVELET MODEL SMOKE TEST')
    print('=' * 60)
    print(f'Device: {device}')
    print(f'Feature dir: {params["feat_dir"]}')
    print(f'Normalized feature dir: {feature_dir}')
    print(f'Wavelet mode: {params["use_wavelet"]}')
    if params['use_wavelet']:
        print(f'Wavelet: {params["wavelet"]} | scales: {params["n_wavelet_scales"]}')

    audio_features = None
    label_batch = None

    dataset = DataGenerator(params=params, mode='dev_train')
    if len(dataset) > 0:
        sample_index = min(sample_index, len(dataset) - 1)
        audio_features, labels = dataset[sample_index]
        if audio_features.ndim != 3:
            raise ValueError(f'Expected 3D audio features, got shape {tuple(audio_features.shape)}')
        if labels.ndim < 2:
            raise ValueError(f'Unexpected label shape: {tuple(labels.shape)}')
        audio_batch = audio_features.unsqueeze(0).to(device)
        label_batch = labels.unsqueeze(0).to(device)
        print(f'Using paired dataset sample at index: {sample_index}')
        print(f'Input shape: {tuple(audio_batch.shape)}')
        print(f'Label shape: {tuple(label_batch.shape)}')
    else:
        feature_files = sorted(glob.glob(os.path.join(feature_dir, '*.pt')))
        if not feature_files:
            raise RuntimeError(f'No cached normalized features found in {feature_dir}')
        sample_index = min(sample_index, len(feature_files) - 1)
        feature_path = feature_files[sample_index]
        try:
            audio_features = torch.load(feature_path, weights_only=True)
        except TypeError:
            audio_features = torch.load(feature_path)
        if audio_features.ndim != 3:
            raise ValueError(f'Expected 3D audio features, got shape {tuple(audio_features.shape)}')
        audio_batch = audio_features.unsqueeze(0).to(device)
        print(f'Using single cached feature file: {os.path.basename(feature_path)}')
        print(f'Input shape: {tuple(audio_batch.shape)}')

    print(f'Sample index: {sample_index}')
    print(f'Input shape: {tuple(audio_batch.shape)}')
    if label_batch is not None:
        print(f'Label shape: {tuple(label_batch.shape)}')

    model = SELDModel(params=params, in_feat_shape=audio_batch.shape).to(device)
    if params['multiACCDOA']:
        loss_fn = SELDLossADPIT(params=params).to(device)
    else:
        loss_fn = SELDLossSingleACCDOA(params=params).to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    optimizer.zero_grad(set_to_none=True)
    predictions = model(audio_batch)
    print(f'Prediction shape: {tuple(predictions.shape)}')

    if label_batch is None:
        if params['multiACCDOA']:
            label_batch = torch.zeros(
                predictions.shape[0],
                predictions.shape[1],
                6,
                4,
                params['nb_classes'],
                device=predictions.device,
                dtype=predictions.dtype,
            )
        else:
            label_batch = torch.zeros_like(predictions)
        print(f'Using dummy zero target with shape: {tuple(label_batch.shape)}')

    if params['multiACCDOA']:
        print('ADPIT target shape is expected to differ from the model output; loss will verify compatibility.')

    loss = loss_fn(predictions, label_batch)
    print(f'Loss: {loss.item():.6f}')
    loss.backward()
    optimizer.step()

    print('✓ Forward/backward pass succeeded on one cached wavelet sample.')
    return 0


if __name__ == '__main__':
    sys.exit(main())