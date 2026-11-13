"""Fit one Muon model and save its Good-calibrated checkpoint; no plots."""

import argparse
import copy
import hashlib
from pathlib import Path
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from scripts.models.config import DIALS, EXPERIMENTS, KDE_AE, KDE_MMD_AE, OUT, SEED
from scripts.data.dials import load_dataset
from scripts.models.core import (FEATURE_EPOCHS, BATCH_SIZE, HIDDEN_LAYERS,
                                QUANTILE, calibrate, load_checkpoint, model_spec, score_values)
from scripts.models.networks import histogram_loss
from scripts.models.preprocessing import (_histograms, _rows, condition_values,
                                           load_pileup, model_features, prepare,
                                           training_context)


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


def training_signature(context, pileup):
    """Hash the aligned PU snapshot as well as the existing frozen DQM cohort."""
    raw = np.asarray(pileup, dtype=np.float64)
    selected = np.asarray(context.get('signature_rows', np.ones(len(raw), dtype=bool)))
    if raw.ndim != 1 or selected.dtype != bool or selected.shape != raw.shape:
        raise ValueError('Expected one boolean signature flag per pile-up observation')
    training = np.zeros(len(raw), dtype=bool)
    training[_rows(context['train'], len(raw))] = True
    if (training & ~selected).any():
        raise ValueError('Training LS cannot be excluded from the model signature')
    canonical = np.full(raw.shape, np.nan, dtype='<f8')
    valid = np.isfinite(raw) & (raw >= 0)
    canonical[valid] = raw[valid]
    return dict(context['signature'],
                pileup_sha256=hashlib.sha256(canonical[selected].tobytes()).hexdigest(),
                fit_selection_sha256=hashlib.sha256(training[selected].tobytes()).hexdigest())


def _feature_loss(predicted, observed, mask, feature_me, n_mes):
    """Equal ME weights even when a periodic GMM representation has extra features."""
    indices = feature_me[None, :].expand(len(observed), -1)
    sums = observed.new_zeros((len(observed), n_mes)).scatter_add_(
        1, indices, torch.where(mask, (predicted - observed).square(), 0))
    counts = observed.new_zeros((len(observed), n_mes)).scatter_add_(1, indices, mask.to(observed.dtype))
    valid = counts > 0
    per_me = sums / counts.clamp_min(1)
    per_ls = torch.where(valid, per_me, 0).sum(1) / valid.sum(1).clamp_min(1)
    return per_ls[valid.any(1)].mean()


def fit_conditioned(dataset, context, spec, settings=None):
    """Fit one pile-up-only conditional model; tuning callers supply overrides.

    Only Good training LS fit either scaler. Monitor runs select epochs; separate
    calibration runs and Test are never used for fitting or early stopping.
    """
    settings = dict(spec.SETTINGS, **(settings or {}))
    frame = dataset['frame']
    if 'pileup' not in dataset:
        dataset['pileup'] = load_pileup(frame)
    raw = np.asarray(dataset['pileup'], dtype=float)
    available = np.isfinite(raw) & (raw >= 0)
    train = _rows(context['train'], len(frame))
    monitor = _rows(context['monitor_indices'], len(frame))
    if np.intersect1d(train, monitor).size:
        raise ValueError('Training and monitoring LS overlap')
    is_histogram = spec.KIND == 'mdn'
    signature = training_signature(context, raw)
    saved = dict(settings=settings, signature=signature,
                 condition_digest=signature['pileup_sha256'],
                 me_names=dataset['me_names'], axes=dataset['axes'],
                 run_split=dataset['run_split'], experiment=spec.NAME)
    if spec.KIND in ('shared_gmm', 'kde'):
        from scripts.models.shared_gaussian import fit_basis, training_variance

        basis_train = np.zeros(len(frame), dtype=bool)
        basis_train[train[available[train]]] = True
        if spec.KIND == 'shared_gmm':
            saved['basis'] = fit_basis(dataset['histograms'], dataset['axes'], basis_train,
                                      components=settings['components'],
                                      iterations=settings['basis_iterations'])
        else:
            from scripts.models.kde import train_reference

            saved['basis'] = train_reference(dataset['histograms'], dataset['axes'], basis_train,
                                            bandwidth_bins=settings['bandwidth_bins'])
        saved['bin_variance'] = training_variance(dataset['histograms'], basis_train)
        saved['probability_floor'] = [
            .5 / (dataset['totals'][basis_train, me].sum(dtype=np.float64) + .5 * h.shape[1])
            for me, h in enumerate(dataset['histograms'])]
    if is_histogram:
        histograms, totals = _histograms(dataset)
        available &= (totals > 0).any(1)
        dimensions = [h.shape[1] for h in histograms]
        saved['bin_sizes'] = tuple(dimensions)
    else:
        features, mask, feature_me, names = model_features(dataset, spec, settings,
                                                         basis=saved.get('basis'))
        features, mask = np.asarray(features, dtype=np.float32), np.asarray(mask, dtype=bool)
        if features.shape != mask.shape or len(features) != len(frame):
            raise ValueError('Expected aligned feature values and validity flags')
        available &= mask.any(1)
        dimensions = features.shape[1]
        saved.update(feature_me=np.asarray(feature_me, dtype=int), features=list(names))
    train, monitor = train[available[train]], monitor[available[monitor]]
    if not len(train) or not len(monitor):
        raise ValueError('No usable Good training/monitoring LS with measured OMS pile-up')
    train_mask = np.zeros(len(frame), dtype=bool)
    train_mask[train] = True
    conditions, condition_scaler, _ = condition_values(
        raw, train_mask=train_mask, transform=settings.get('condition_transform', 'linear'))
    conditions = torch.as_tensor(conditions, dtype=torch.float32)
    saved.update(condition_scaler=condition_scaler,
                 training_ls=int(len(train)), monitoring_ls=int(len(monitor)),
                 training_years=sorted(frame.iloc[train].dataset.str.extract(r'/Run(20\d\d)')[0].astype(int).unique().tolist()),
                 oms_available_ls=int((np.isfinite(raw) & (raw >= 0)).sum()))
    if is_histogram:
        inputs, hist_scaler = prepare(dataset, train_mask=train_mask)
        saved['hist_scaler'] = hist_scaler
    else:
        if (mask[train].sum(0) == 0).any():
            raise ValueError('A model feature has no observed Good training values')
        target_values = features
        if spec.KIND == 'kde':
            from scripts.models.kde import raw_features

            target_values = raw_features(dataset['histograms'])
        scaler = StandardScaler().fit(np.where(mask[train], target_values[train], np.nan))
        scaled = scaler.transform(features)
        inputs = torch.from_numpy(np.where(mask, scaled, 0).astype(np.float32, copy=False))
        del scaled
        if spec.KIND == 'kde':
            target_values = scaler.transform(target_values, copy=False)
            target_values[~mask] = 0
            targets = torch.from_numpy(target_values)
        else:
            targets = inputs
        masks = torch.from_numpy(mask)
        feature_me = torch.as_tensor(saved['feature_me'], dtype=torch.long)
        saved['scaler'] = scaler
    inputs = torch.as_tensor(inputs, dtype=torch.float32)
    saved['input_dim'] = inputs.shape[1]
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = spec.build_model(dimensions, settings)
    if is_histogram:
        model.initialize_heads([h[train].sum(0, dtype=np.float64) for h in histograms])
    device = torch.device('mps' if is_histogram and torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    saved['training_device'] = str(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.get('lr', .001),
                                 weight_decay=settings.get('weight_decay', 1e-5))
    batch_size = int(settings.get('batch_size', 1024))

    def objective(rows):
        predicted = model(inputs[rows].to(device), conditions[rows].to(device))
        if is_histogram:
            return histogram_loss(predicted, [h[rows] for h in histograms],
                                  count_weighted=settings.get('count_weighted', False))
        return _feature_loss(predicted, targets[rows], masks[rows], feature_me, len(dataset['me_names']))

    history = dict(train=[], validation=[], best_epoch=0,
                   loss='Multinomial NLL' if is_histogram and settings.get('count_weighted', False)
                   else 'Cross entropy' if is_histogram else 'ME-balanced MSE')
    best, state, stale = float('inf'), None, 0
    for epoch in range(1, int(settings.get('epochs', 80)) + 1):
        model.train()
        total, weight = 0., 0
        shuffled = rng.permutation(train)
        for first in range(0, len(shuffled), batch_size):
            rows = shuffled[first:first + batch_size]
            optimizer.zero_grad(set_to_none=True)
            value = objective(rows)
            if value is None:
                continue
            if not torch.isfinite(value):
                raise FloatingPointError('Nonfinite training loss')
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(value.detach()) * len(rows)
            weight += len(rows)
        if not weight:
            raise ValueError('No observed training LS')
        model.eval()
        checked, checked_weight = 0., 0
        with torch.no_grad():
            for first in range(0, len(monitor), batch_size):
                rows = monitor[first:first + batch_size]
                value = objective(rows)
                if value is not None:
                    checked += float(value) * len(rows)
                    checked_weight += len(rows)
        if not checked_weight or not np.isfinite(checked):
            raise FloatingPointError('No finite monitoring loss')
        checked /= checked_weight
        history['train'].append(total / weight)
        history['validation'].append(checked)
        print(f'{spec.KIND} epoch {epoch:02d}: train {total / weight:.7g}, monitor {checked:.7g}', flush=True)
        if checked < best * (1 - 1e-4):
            best, state, stale = checked, copy.deepcopy(model.state_dict()), 0
            history['best_epoch'] = epoch
        else:
            stale += 1
            if stale >= int(settings.get('patience', 8)):
                break
    model.load_state_dict(state)
    saved.update(state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
                 history=history)
    return saved


def check_compatible(spec, saved, dataset, context):
    """Refuse a silent reuse after a training cohort, pile-up or settings change."""
    expected = context['signature']
    if spec.KIND != 'feature':
        if 'pileup' not in dataset:
            dataset['pileup'] = load_pileup(dataset['frame'])
        expected = training_signature(context, dataset['pileup'])
    if saved.get('signature') != expected or saved.get('run_split') != dataset['run_split']:
        raise ValueError('Training selection or OMS snapshot changed; choose a separate --output directory')
    if spec.KIND == 'feature':
        if saved.get('features') != context['features']:
            raise ValueError('Feature layout changed; choose a separate --output directory')
    else:
        settings = dict(saved.get('settings', {}))
        if spec.NAME == KDE_MMD_AE:
            settings.setdefault('mmd_top_n', None)  # Earlier mean-score checkpoints.
        if settings != spec.SETTINGS:
            raise ValueError('Model settings changed; choose a separate --output directory')


def fit_mmd(dataset, context, spec):
    """Reuse the fixed KDE weights; fit only Good-monitor score references."""
    from scripts.models.kde_scores import fit_ecdf
    from scripts.predict import predict_terms

    source_spec, saved, path = load_checkpoint(OUT/'model'/KDE_AE)
    if source_spec.NAME != KDE_AE:
        raise ValueError(f'Expected a {KDE_AE} source checkpoint at {path}')
    check_compatible(source_spec, saved, dataset, context)
    saved = copy.deepcopy(saved)
    saved.pop('threshold', None)
    saved.update(experiment=spec.NAME, settings=dict(spec.SETTINGS), source_experiment=KDE_AE,
                 source_checkpoint=str(path))
    terms, _ = predict_terms(spec, saved, dataset)
    saved['score_calibration'] = fit_ecdf(terms, context['monitor_indices'])
    return saved, terms, score_values(terms, saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=[f'model_{name}' for name in EXPERIMENTS])
    parser.add_argument('--data', type=Path, default=DIALS, help='Cached DIALS directory')
    parser.add_argument('--years', type=int, nargs='+', help='Use only these cached years; default: all')
    parser.add_argument('--output', type=Path, help='Checkpoint directory; existing fits are preserved')
    args = parser.parse_args()
    spec = model_spec(args.model)
    folder = args.output or OUT/'model'/spec.NAME
    path = folder/('model.pkl' if spec.KIND == 'feature' else 'model.pt')
    other = [folder/name for name in ('model.pkl', 'model.pt')
             if name != path.name and (folder/name).exists()]
    if other:
        raise ValueError(f'Another model already exists in {folder}; choose a separate --output directory')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dataset = load_dataset(args.data, split=True, years=args.years)
    context = training_context(dataset)
    if path.exists():
        old_spec, saved, _ = load_checkpoint(path)
        if old_spec.NAME != spec.NAME:
            raise ValueError('The output checkpoint belongs to another model')
        check_compatible(spec, saved, dataset, context)
        print(f'Compatible checkpoint preserved: {path}', flush=True)
        return
    if spec.NAME == KDE_MMD_AE:
        saved, terms, values = fit_mmd(dataset, context, spec)
    elif spec.KIND == 'feature':
        saved = fit_feature(dataset['features'], context['train'], context['monitor_indices'], spec)
        saved.update(signature=context['signature'], features=context['features'])
    else:
        saved = fit_conditioned(dataset, context, spec)
    saved.update(experiment=spec.NAME, run_split=dataset['run_split'],
                 me_names=dataset['me_names'], axes=dataset['axes'],
                 training_years=saved.get('training_years', sorted(
                     context['frame'].loc[context['train']].dataset.str.extract(r'/Run(20\d\d)')[0].astype(int).unique().tolist())))
    from scripts.predict import predict_terms
    if spec.NAME != KDE_MMD_AE:
        terms, values = predict_terms(spec, saved, dataset)
    _, threshold = calibrate(context['frame'], terms, context['calibration'], values)
    saved.update(threshold=threshold, quantile=QUANTILE)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    if spec.KIND == 'feature':
        temporary.write_bytes(pickle.dumps(saved))
    else:
        torch.save(saved, temporary)
    temporary.replace(path)
    print(f'Checkpoint saved: {path} (threshold={threshold:.6g})', flush=True)


if __name__ == '__main__':
    main()
