"""Fit one Muon autoencoder and save its calibrated checkpoint; no plots."""

import argparse
import copy
import json
import os
from pathlib import Path
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from scripts.models.config import DIALS, EXPERIMENTS, OUT, SEED
from scripts.data.dials import load_dataset
from scripts.models.core import (FEATURE_EPOCHS, BATCH_SIZE, EPOCHS, HIDDEN_LAYERS,
                                PATIENCE, QUANTILE, calibrate, load_checkpoint, loss, model_spec)
from scripts.models.preprocessing import _histograms, _rows, prepare, training_context


def fit_feature(features, train_mask, monitor_indices, spec):
    """Mean/std pairs through an 8→4→8 bottleneck; monitoring never selects epochs."""
    features = np.asarray(features, dtype=np.float32)
    if (features.ndim != 2 or features.shape[1] == 0 or features.shape[1] % 2
            or not np.isfinite(features).all()):
        raise ValueError('Expected a finite LS-by-feature matrix of mean/std pairs')
    train, monitor = _rows(train_mask, len(features)), _rows(monitor_indices, len(features))
    if np.intersect1d(train, monitor).size:
        raise ValueError('Training and monitoring LS must be disjoint')
    scaler = StandardScaler().fit(features[train])
    x, checked = scaler.transform(features[train]), scaler.transform(features[monitor])
    model = spec.build_model()
    history = {'train': [], 'validation': []}
    for epoch in range(1, FEATURE_EPOCHS + 1):
        model.partial_fit(x, x)
        history['train'].append(float(np.mean((model.predict(x) - x) ** 2)))
        history['validation'].append(float(np.mean((model.predict(checked) - checked) ** 2)))
        if epoch % 20 == 0:
            print(f'feature epoch {epoch:02d}: train {history["train"][-1]:.7g}, '
                  f'monitor {history["validation"][-1]:.7g}', flush=True)
    return {'model': model, 'scaler': scaler, 'hidden_layer_sizes': HIDDEN_LAYERS,
            'seed': SEED, 'train_losses': history['train'],
            'validation_losses': history['validation']}


def _monitor_loss(model, inputs, histograms, totals, indices, distribution):
    model.eval()
    total, weight = 0., 0
    with torch.no_grad():
        for first in range(0, len(indices), BATCH_SIZE):
            rows = indices[first:first + BATCH_SIZE]
            value = loss(model(inputs[rows]), [h[rows] for h in histograms], distribution)
            n = int((totals[rows] > 0).any(axis=1).sum())
            if value is not None:
                total += float(value) * n
                weight += n
    if not weight:
        raise ValueError('No nonempty monitoring LS')
    return total / weight


def fit_histogram(dataset, inputs, train_mask, monitor_indices, spec):
    """Matched K=1 training; the caller supplies Good training/monitoring LS."""
    distribution = spec.DISTRIBUTION
    histograms, totals = _histograms(dataset)
    train, monitor = _rows(train_mask, len(totals)), _rows(monitor_indices, len(totals))
    if np.intersect1d(train, monitor).size:
        raise ValueError('Training and monitoring LS must be disjoint')
    if not (totals[train] > 0).any():
        raise ValueError('No nonempty training LS')
    inputs = torch.as_tensor(inputs, dtype=torch.float32)
    expected = (len(totals), sum(h.shape[1] for h in histograms) + 2 * len(histograms))
    if tuple(inputs.shape) != expected or not torch.isfinite(inputs).all():
        raise ValueError('Invalid prepared histogram inputs')
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = spec.build_model([h.shape[1] for h in histograms])
    model.initialize_heads([h[train].sum(axis=0, dtype=np.float64) for h in histograms])
    parameters = list(model.parameters())
    groups = parameters if distribution == 'mse' else [
        {'params': [p for p in parameters if p is not model.log_concentration]},
        {'params': [model.log_concentration], 'lr': .01}]
    optimizer = torch.optim.Adam(groups, lr=.001, eps=1e-12)
    history = {'train': [], 'validation': [], 'best_epoch': 0,
               'loss': 'MSE' if distribution == 'mse' else 'DM NLL'}
    best, state, stale = float('inf'), None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        shuffled, total, weight = rng.permutation(train), 0., 0
        for first in range(0, len(shuffled), BATCH_SIZE):
            rows = shuffled[first:first + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            value = loss(model(inputs[rows]), [h[rows] for h in histograms], distribution)
            if value is None:
                continue
            if not torch.isfinite(value):
                raise FloatingPointError(f'Nonfinite {distribution} training loss')
            value.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.)
            optimizer.step()
            n = int((totals[rows] > 0).any(axis=1).sum())
            total += float(value.detach()) * n
            weight += n
        checked = _monitor_loss(model, inputs, histograms, totals, monitor, distribution)
        if not np.isfinite(checked):
            raise FloatingPointError(f'Nonfinite {distribution} monitoring loss')
        history['train'].append(total / weight)
        history['validation'].append(checked)
        print(f'{distribution} epoch {epoch:02d}: train {total / weight:.7g}, monitor {checked:.7g}', flush=True)
        if checked < best * (1 - 1e-4):
            best, state, stale = checked, copy.deepcopy(model.state_dict()), 0
            history['best_epoch'] = epoch
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(state)
    model.eval()
    return model, history


def check_compatible(spec, saved, dataset, context):
    """Never reuse a fit when its training cohort or histogram layout changed."""
    if saved.get('signature') != context['signature']:
        raise ValueError('Training selection changed; choose a separate --output directory')
    if spec.KIND == 'feature':
        if saved.get('features') != context['features'] or saved.get('run_split') != dataset['run_split']:
            raise ValueError('Run split or feature layout changed; choose a separate --output directory')
    if spec.KIND != 'feature':
        _, scaler = prepare(dataset, context['train'])
        same_axes = (len(saved['axes']) == len(dataset['axes']) and all(
            np.array_equal(old[key], current[key])
            for old, current in zip(saved['axes'], dataset['axes'])
            for key in ('low', 'high', 'edges')))
        if (saved['me_names'] != dataset['me_names']
                or tuple(saved['bin_sizes']) != tuple(h.shape[1] for h in dataset['histograms'])
                or not same_axes
                or any(not np.array_equal(getattr(saved['scaler'], key), getattr(scaler, key))
                       for key in ('mean_', 'scale_'))):
            raise ValueError('ME layout, axes or training scaling changed; choose a separate --output directory')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=[f'model_{name}' for name in EXPERIMENTS])
    parser.add_argument('--data', type=Path, default=DIALS, help='Cached DIALS directory')
    parser.add_argument('--output', type=Path, help='Checkpoint directory; default: output/model/<model>')
    parser.add_argument('--source', type=Path, help='Model 03 checkpoint for model 04 only')
    args = parser.parse_args()
    spec = model_spec(args.model)
    if args.source and spec.KIND != 'window':
        parser.error('--source is only used by model 04')
    folder = args.output or OUT/'model'/spec.NAME
    filename = 'source.json' if spec.KIND == 'window' else 'model.pkl' if spec.KIND == 'feature' else 'model.pt'
    path = folder/filename
    other = [folder/name for name in ('model.pkl', 'model.pt', 'source.json')
             if name != filename and (folder/name).exists()]
    if other:
        raise ValueError(f'Another model already exists in {folder}; choose a separate --output directory')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dataset = load_dataset(args.data, split=True)
    context = training_context(dataset)
    if path.exists():
        old_spec, saved, _ = load_checkpoint(path)
        if old_spec.NAME != spec.NAME:
            raise ValueError('The output checkpoint belongs to another model')
        check_compatible(spec, saved, dataset, context)
        print(f'Compatible checkpoint preserved: {path}', flush=True)
        return
    if spec.KIND == 'window':
        source_spec, saved, source_file = load_checkpoint(args.source or OUT/'model'/spec.SOURCE)
        if source_spec.NAME != spec.SOURCE:
            raise ValueError(f'{spec.NAME} requires source model {spec.SOURCE}')
        check_compatible(source_spec, saved, dataset, context)
        saved = dict(saved, experiment=spec.NAME, window=spec.WINDOW, centered=True)
    elif spec.KIND == 'feature':
        saved = fit_feature(dataset['features'], context['train'], context['monitor_indices'], spec)
    else:
        inputs, scaler = prepare(dataset, context['train'])
        model, history = fit_histogram(dataset, inputs, context['train'], context['monitor_indices'], spec)
        saved = dict(state_dict=model.state_dict(), scaler=scaler, history=history, bin_sizes=model.bin_sizes)
    saved.update(experiment=spec.NAME, signature=context['signature'], run_split=dataset['run_split'],
                 features=context['features'], me_names=dataset['me_names'], axes=dataset['axes'])
    # Calibration is part of training; prediction only reuses this fixed cut.
    from scripts.predict import predict_terms
    terms, values = predict_terms(spec, saved, dataset)
    _, threshold = calibrate(context['frame'], terms, context['calibration'], values)
    saved.update(threshold=threshold, quantile=QUANTILE)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    if spec.KIND == 'window':
        pointer = dict(experiment=spec.NAME, source=os.path.relpath(source_file.resolve(), folder.resolve()),
                       source_model=source_spec.NAME, window=spec.WINDOW, centered=True,
                       threshold=threshold, quantile=QUANTILE, signature=context['signature'],
                       run_split=dataset['run_split'])
        temporary.write_text(json.dumps(pointer, indent=2)+'\n')
    elif spec.KIND == 'feature':
        temporary.write_bytes(pickle.dumps(saved))
    else:
        torch.save(saved, temporary)
    temporary.replace(path)
    print(f'Checkpoint saved: {path} (threshold={threshold:.6g})', flush=True)


if __name__ == '__main__':
    main()
