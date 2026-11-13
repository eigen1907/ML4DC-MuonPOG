"""Shared moments, histogram inputs and training-only reference selections."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from scripts.models.config import SEED, SPECIAL_RUNS


MONITOR_LIMIT = 4096


def _moments(counts, low, high, eta=False):
    """Historical 0–1 moments; eta is folded and an empty ME gives two zeros."""
    low, high = float(low), float(high)
    edges = np.linspace(low, high, counts.shape[1] + 1, dtype=np.float32)
    centers = (edges[:-1] + edges[1:]) / 2
    positions = np.abs(centers) / max(abs(low), abs(high)) if eta else (centers - low) / (high - low)
    counts = counts.astype(np.float64)
    totals = counts.sum(axis=1)
    weights = np.divide(counts, totals[:, None], out=np.zeros_like(counts), where=totals[:, None] > 0)
    mean = weights @ positions
    std = np.sqrt((weights * (positions - mean[:, None]) ** 2).sum(axis=1))
    return np.column_stack((mean, std))



def _rows(selection, size):
    selection = np.asarray(selection)
    if selection.dtype == bool:
        if selection.shape != (size,):
            raise ValueError('Expected one training flag per LS')
        selection = np.flatnonzero(selection)
    if (selection.ndim != 1 or not np.issubdtype(selection.dtype, np.integer)
            or not len(selection) or len(np.unique(selection)) != len(selection)
            or (selection < 0).any() or (selection >= size).any()):
        raise ValueError('Expected nonempty unique LS indices')
    return selection



def _histograms(dataset):
    histograms = [np.asarray(h) for h in dataset['histograms']]
    totals = np.asarray(dataset['totals'])
    if (not histograms or totals.ndim != 2 or totals.shape[1] != len(histograms)
            or not np.isfinite(totals).all() or (totals < 0).any()
            or (not np.issubdtype(totals.dtype, np.integer) and (totals != np.floor(totals)).any())):
        raise ValueError('Expected finite nonnegative integer histogram totals')
    for index, hist in enumerate(histograms):
        if (hist.ndim != 2 or len(hist) != len(totals) or hist.shape[1] < 1
                or not np.isfinite(hist).all() or (hist < 0).any()
                or (not np.issubdtype(hist.dtype, np.integer) and (hist != np.floor(hist)).any())
                or not np.array_equal(hist.sum(-1, dtype=np.float64), totals[:, index])):
            raise ValueError('Invalid native-bin histogram or inconsistent totals')
    return histograms, totals



def prepare(dataset, train_mask=None, scaler=None):
    """Native fractions + training-scaled log1p(N) + nonempty masks.

    Inference uses the saved scaler and needs no training selection.
    """
    histograms, totals = _histograms(dataset)
    valid = totals > 0
    bins, targets = sum(h.shape[1] for h in histograms), len(histograms)
    values = np.zeros((len(totals), bins + 2 * targets), dtype=np.float32)
    start = 0
    for target, hist in enumerate(histograms):
        stop = start + hist.shape[1]
        np.divide(hist, totals[:, target, None], out=values[:, start:stop], where=valid[:, target, None])
        start = stop
    logs = np.log1p(totals).astype(np.float32)
    if scaler is None:
        train = _rows(train_mask, len(totals))
        scaler = StandardScaler().fit(logs[train])
    values[:, bins:bins + targets] = scaler.transform(logs)
    values[:, bins + targets:] = valid
    return torch.from_numpy(values), scaler



def training_context(dataset):
    """Keep Good train, monitoring and calibration roles separate and reproducible."""
    from scripts.data.dials import fingerprint, validation_roles

    frame = dataset['frame'].copy()
    roles = validation_roles(frame)
    normal = frame.label.eq('good') & ~frame.run_number.isin(SPECIAL_RUNS)
    train = (normal & frame.split.eq('train') & frame.dcs_pass).to_numpy()
    monitor = (normal & frame.split.eq('validation') & frame.dcs_pass & frame.run_number.isin(roles['monitor_runs'])).to_numpy()
    calibration = (normal & frame.split.eq('validation') & frame.dcs_pass & frame.run_number.isin(roles['calibration_runs'])).to_numpy()
    fit_monitor = np.flatnonzero(monitor)
    fit_monitor = np.random.default_rng(SEED).choice(fit_monitor, min(MONITOR_LIMIT, len(fit_monitor)), replace=False)
    if (train & monitor).any() or (train & calibration).any() or (monitor & calibration).any():
        raise ValueError('Training, monitoring and calibration overlap')
    signature = dict(version='five-ae-dcs-v1', data_fingerprint=fingerprint(dataset), seed=SEED,
                     selection='Good training runs AND DCS-pass; no entry/statistics cut',
                     training_ls=int(train.sum()), monitor_runs=roles['monitor_runs'],
                     calibration_runs=roles['calibration_runs'])
    features = [f'{me}__{kind}' for me in dataset['me_names'] for kind in ('mean', 'std')]
    return dict(frame=frame, train=train, monitor_indices=fit_monitor,
                calibration=calibration, signature=signature, features=features)


def load_pileup(frame):
    """Join annual OMS by physical run/LS; missing observations remain NaN."""
    values = np.full(len(frame), np.nan, dtype=np.float64)
    years = frame.dataset.str.extract(r'/Run(20\d\d)', expand=False)
    for year in sorted(years.dropna().unique()):
        path = Path('data/oms') / f'{year}.json'
        if not path.is_file():
            continue
        rows = json.loads(path.read_text()).get('lumisections', [])
        wanted = set(frame.loc[years.eq(year), 'run_number'])
        triples = [(r['run_number'], r['lumisection_number'], r.get('pileup'))
                   for r in rows if r['run_number'] in wanted]
        del rows
        if not triples:
            continue
        table = pd.DataFrame(triples, columns=['run_number', 'ls_number', 'pileup'])
        if table.duplicated(['run_number', 'ls_number']).any():
            raise ValueError(f'Duplicate OMS run/LS rows: {path}')
        table['pileup'] = pd.to_numeric(table.pileup, errors='coerce')
        selected = years.eq(year).to_numpy()
        joined = frame.loc[selected, ['run_number', 'ls_number']].merge(
            table, on=['run_number', 'ls_number'], how='left', sort=False,
            validate='many_to_one')
        values[selected] = joined.pileup.to_numpy(dtype=float, na_value=np.nan)
    values[~np.isfinite(values) | (values < 0)] = np.nan
    return values


def condition_values(raw, train_mask=None, scaler=None, transform='linear'):
    """Fit pile-up scaling on known training LS; never impute missing PU as data."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError('Expected one pile-up value per LS')
    valid = np.isfinite(raw) & (raw >= 0)
    values = np.zeros((len(raw), 1), dtype=np.float32)
    if transform not in ('linear', 'log1p'):
        raise ValueError('Expected linear or log1p pile-up condition')
    values[valid, 0] = np.log1p(raw[valid]) if transform == 'log1p' else raw[valid]
    if scaler is None:
        rows = _rows(train_mask, len(raw))
        rows = rows[valid[rows]]
        if not len(rows):
            raise ValueError('No training LS with valid OMS pile-up')
        scaler = StandardScaler().fit(values[rows])
    if valid.any():
        values[valid] = scaler.transform(values[valid])
    return torch.from_numpy(values), scaler, valid


def model_features(dataset, spec, settings):
    """Return targets, observation masks and the ME owning each feature."""
    if spec.KIND == 'conditional':
        values = np.asarray(dataset['features'], dtype=np.float32)
        return values, np.ones_like(values, dtype=bool), np.repeat(
            np.arange(len(dataset['me_names'])), 2), [
            f'{me}__{kind}' for me in dataset['me_names'] for kind in ('mean', 'std')]
    if spec.KIND != 'gmm':
        raise ValueError('Feature targets are only defined for moment and GMM AEs')
    from scripts.models.gaussian import fit_mixtures

    key = (settings['components'], settings.get('gmm_iterations', 20),
           settings.get('gmm_scale_floor_bins', 12**-.5), settings.get('gmm_pseudocount', .5))
    cache = dataset.setdefault('_gmm_cache', {})
    if key not in cache:
        result = fit_mixtures(dataset['histograms'], dataset['axes'], components=key[0],
                              iterations=key[1], scale_floor_bins=key[2], pseudocount=key[3])
        cache.clear()
        cache[key] = {name: result[name] for name in ('features', 'valid', 'feature_me', 'feature_names')}
    result = cache[key]
    owner = result['feature_me']
    return result['features'], result['valid'][:, owner], owner, result['feature_names']
