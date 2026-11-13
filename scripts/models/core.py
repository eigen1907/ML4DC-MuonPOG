"""Shared scoring, calibration and checkpoint loading; no fitting."""

import importlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

from scripts.models.config import EXPERIMENTS, FEATURE_AE, OUT

HIDDEN_LAYERS = (8, 4, 8)
FEATURE_EPOCHS = 80
BATCH_SIZE = 1024
QUANTILE = .975


def model_spec(name):
    name = name.removeprefix('model_')
    if name not in EXPERIMENTS:
        raise ValueError(f'Unknown model: {name}')
    return importlib.import_module(f'scripts.models.model_{name}')


def load_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        files = [path/name for name in ('model.pkl', 'model.pt') if (path/name).is_file()]
        if len(files) != 1:
            raise ValueError(f'Expected one model checkpoint in {path}')
        path = files[0]
    if path.suffix == '.pkl':
        saved = pickle.loads(path.read_bytes())
        inferred = FEATURE_AE
    elif path.suffix == '.pt':
        saved = torch.load(path, map_location='cpu', weights_only=False)
        inferred = path.parent.name
    else:
        raise ValueError(f'Expected a .pkl or .pt checkpoint: {path}')
    return model_spec(saved.get('experiment', inferred)), saved, path


def restore_model(spec, saved):
    if spec.KIND == 'feature':
        return saved['model']
    dimensions = saved['bin_sizes'] if spec.KIND == 'mdn' else saved['input_dim']
    model = spec.build_model(dimensions, saved['settings'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model


def threshold_for(path, saved, spec):
    value = saved.get('threshold')
    if value is None:
        canonical = OUT/'model'/spec.NAME/('model.pkl' if spec.KIND == 'feature' else 'model.pt')
        filename = Path(path)/canonical.name if Path(path).is_dir() else Path(path)
        metrics = OUT/'predict'/spec.NAME/'metrics.json'
        if filename.resolve() != canonical.resolve() or not metrics.is_file():
            raise FileNotFoundError(f'No fixed calibration threshold in {path}')
        value = json.loads(metrics.read_text())['threshold']
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Calibration threshold must be positive and finite')
    return value


def mean_score(terms):
    terms = np.asarray(terms, dtype=float)
    if terms.ndim != 2 or np.isinf(terms).any():
        raise ValueError('Expected an LS-by-ME array of finite or missing residuals')
    valid = np.isfinite(terms)
    return np.divide(np.where(valid, terms, 0).sum(1), valid.sum(1),
                     out=np.full(len(terms), np.nan), where=valid.any(1))


def score_values(terms, saved):
    """Optional saved, Good-monitor-only rank calibration; default stays unchanged."""
    reference = saved.get('score_calibration')
    if reference is None:
        return mean_score(terms)
    from scripts.models.kde_scores import ecdf_terms, top_mean_score

    return top_mean_score(ecdf_terms(terms, reference), saved['settings'].get('mmd_top_n'))


def feature_terms(features, saved):
    """Keep the original 16-feature baseline scoring unchanged."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not features.shape[1] or features.shape[1] % 2:
        raise ValueError('Expected an LS-by-feature matrix of mean/std pairs')
    x = saved['scaler'].transform(features)
    errors = (x - saved['model'].predict(x)) ** 2
    return errors.reshape(len(errors), features.shape[1] // 2, 2).mean(2), errors.mean(1)


def calibrate(frame, terms, calibration, values=None):
    terms = np.asarray(terms)
    if terms.ndim != 2 or len(terms) != len(frame):
        raise ValueError('Expected one row of ME residuals per LS')
    values = mean_score(terms) if values is None else np.asarray(values, dtype=float)
    calibration = np.asarray(calibration)
    if calibration.shape != (len(frame),) or calibration.dtype != bool or values.shape != (len(frame),):
        raise ValueError('Expected one calibration flag and score per LS')
    reference = values[calibration & np.isfinite(values)]
    if not len(reference):
        raise ValueError('No observed Good calibration scores')
    cut = float(np.quantile(reference, QUANTILE))
    if not np.isfinite(cut) or cut <= 0:
        raise ValueError('Calibration cut must be positive and finite')
    scores = frame.copy()
    scores['anomaly_score'] = values
    for target in range(terms.shape[1]):
        scores[f'term_{target}'] = terms[:, target]
    scores['flagged'] = pd.array(np.where(frame.dcs_pass & np.isfinite(values), values > cut, None), dtype='boolean')
    return scores, cut
